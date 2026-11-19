"""Rebuild the Ozone HR action business case from aggregate salary data and assumptions."""

import json
from math import ceil, floor, sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, t

from .data import ROOT, load_data

TABLES = ROOT / 'reports/tables/hr_action_business_case'
FIGURES = ROOT / 'reports/figures/hr_action_business_case'
REPORTS = ROOT / 'reports'
SCOPES = ('pilot', 'rollout')
COLORS = {'overtime': '#176b87', 'travel': '#84529b', 'careers': '#597248'}
FILTERS = {'overtime': ('OverTime', 'Yes'), 'travel': ('BusinessTravel', 'Travel_Frequently')}


def values(parameters):
    return {key: entry['value'] for key, entry in parameters.items()}


def load_population(config, records):
    pay = records.MonthlyIncome
    if not (np.isfinite(pay).all() and pay.gt(0).all() and records.JobRole.notna().all()):
        raise ValueError('MonthlyIncome doit être positif et fini et JobRole renseigné.')
    roles = records.groupby('JobRole').MonthlyIncome.agg(reference_n='size', gross_monthly='mean')
    eligible = {}
    breakdown = []
    for action, parameters in config['actions'].items():
        subset = records
        if action in FILTERS:
            field, code = FILTERS[action]
            subset = subset[subset[field].eq(code)]
        wanted = parameters['eligible_roles']['value']
        if set(wanted).difference(roles.index):
            raise ValueError(f'Métiers inconnus pour {action}.')
        if wanted:
            subset = subset[subset.JobRole.isin(wanted)]
        if subset.empty:
            raise ValueError(f'Aucun dossier éligible pour {action}.')
        eligible[action] = subset
        mix = subset.JobRole.value_counts().rename('eligible_n').to_frame().join(roles)
        mix['eligible_role_weight'] = mix.eligible_n / len(subset)
        mix.insert(0, 'action', action)
        breakdown.append(mix.reset_index())
    return eligible, pd.concat(breakdown, ignore_index=True)


def scenario_inputs(scenario, action):
    return {
        **values({key: scenario[key] for key in ('coverage_multiplier', 'budget_multiplier',
                                                 'savings_multiplier', 'replacement_multiplier')}),
        'effect_pp': scenario['effect_pp'][action]['value'],
    }


def evaluate(action, scope, base, scenario, n_eligible, monthly_salary, general):
    months = general['months']
    if not isinstance(months, int) or months <= 0:
        raise ValueError('L’horizon doit être strictement positif.')
    year_share = months / 12
    if not all(np.isfinite(value) and value >= 0 for value in scenario.values()):
        raise ValueError('Les paramètres de scénario doivent être finis et positifs ou nuls.')
    target = base[f'{scope}_covered']
    if not isinstance(target, int) or not 0 < target <= n_eligible:
        raise ValueError(f'Couverture de référence invalide pour {action}/{scope}.')
    covered = floor(target * scenario['coverage_multiplier'] + .5)
    if not 0 < covered <= n_eligible:
        raise ValueError(f'Couverture invalide pour {action}/{scope}: {covered} sur {n_eligible}.')
    base_costs = (base[f'{scope}_launch'], base['annual_recurring_per_covered'],
                  base['annual_savings_per_covered'])
    if not all(np.isfinite(cost) and cost >= 0 for cost in base_costs):
        raise ValueError(f'Coûts/économies unitaires invalides pour {action}/{scope}.')
    launch, recurring_per_head, saved_per_head = base_costs
    launch *= scenario['budget_multiplier']
    recurring = recurring_per_head * covered * year_share * scenario['budget_multiplier']
    savings = saved_per_head * covered * year_share * scenario['savings_multiplier']
    cost = launch + recurring
    value = (monthly_salary * 12 * general['loaded_pay_multiplier'] *
             general['eur_per_salary_unit'] * general['replacement_years'] *
             scenario['replacement_multiplier'])
    effect = scenario['effect_pp']
    if not np.isfinite(value) or value <= 0 or effect > 100:
        raise ValueError('Coût de remplacement ou effet de rétention invalide.')
    avoided = covered * effect / 100
    retention = avoided * value
    gross = savings + retention
    net = gross - cost
    breakeven = max(0, (cost - savings) / value)
    return {
        'action': action, 'scope': scope, 'months': months, 'eligible_proxy_n': n_eligible,
        'covered_n': covered, 'coverage_share_of_proxy': covered / n_eligible,
        'reference_monthly_gross_salary_units': monthly_salary,
        'replacement_eur': value, 'launch_eur': launch, 'recurring_eur': recurring,
        'total_cost_eur': cost, 'direct_savings_eur': savings, 'effect_pp': effect,
        'departures_avoided_expected': avoided, 'retention_value_eur': retention,
        'gross_benefit_eur': gross, 'net_gain_eur': net, 'roi': net / cost if cost > 0 else None,
        'breakeven_departures_exact': breakeven, 'breakeven_departures_ceil': ceil(breakeven),
        'breakeven_effect_pp': 100 * breakeven / covered,
    }


def compute(config, records):
    general = values(config['general'])
    population, mix = load_population(config, records)
    weighted_pay = (mix.gross_monthly * mix.eligible_role_weight).groupby(mix.action).sum()
    rows, sensitivities = [], []
    settings = values(config['sensitivity'])
    for action, parameters in config['actions'].items():
        base = {k: v['value'] for k, v in parameters.items() if isinstance(v, dict) and k != 'eligible_roles'}
        n = len(population[action])
        for scope in SCOPES:
            for key, scenario in config['scenarios'].items():
                selected = scenario_inputs(scenario, action)
                row = evaluate(action, scope, base, selected, n, weighted_pay[action], general)
                rows.append({'scenario': key, 'scenario_label': scenario['name'], **row})
        central = scenario_inputs(config['scenarios']['central'], action)
        for dimension, input_key in [('replacement_multipliers', 'replacement_multiplier'),
                                     ('cost_multipliers', 'budget_multiplier'),
                                     ('coverage_multipliers', 'coverage_multiplier'),
                                     ('effect_pp', 'effect_pp')]:
            for point in settings[dimension]:
                changed = point if input_key == 'effect_pp' else point * central[input_key]
                row = evaluate(action, 'pilot', base, {**central, input_key: changed}, n,
                               weighted_pay[action], general)
                sensitivities.append({'dimension': input_key, 'input': point, **row})
    return pd.DataFrame(rows), pd.DataFrame(sensitivities), population, mix


def pilot_precision(design, covered, effect_pp):
    rate = design['control_rate']
    treated = rate - effect_pp / 100
    team_size, icc = design['team_size'], design['icc']
    if not (isinstance(team_size, int) and team_size > 0 and
            0 < design['alpha'] < 1 and .5 < design['power'] < 1):
        raise ValueError('Taille d’équipe, niveau ou puissance invalides.')
    teams = ceil(covered / team_size)
    if not (0 < treated < rate < 1 and teams >= 2 and 0 <= icc < 1):
        raise ValueError('Paramètres de précision du pilote incompatibles.')
    inflation = 1 + (team_size - 1) * icc
    critical = norm.ppf(1 - design['alpha'] / 2)
    half_width = t.ppf(1 - design['alpha'] / 2, 2 * teams - 2) * sqrt(
        inflation * 2 * rate * (1 - rate) / covered)
    pooled = (rate + treated) / 2
    variance = rate * (1 - rate) + treated * (1 - treated)
    n_independent = ((critical * sqrt(2 * pooled * (1 - pooled)) +
                      norm.ppf(design['power']) * sqrt(variance)) / (rate - treated)) ** 2
    return {'covered_per_arm_for_precision': covered, 'teams_per_arm': teams,
            'design_effect': inflation, 'ci_half_width_pp_at_null': 100 * half_width,
            'teams_per_arm_for_target_power': ceil(n_independent * inflation / team_size)}


def save_figure(fig, stem):
    for fmt in ('png', 'svg'):
        path = FIGURES / f'{stem}.{fmt}'
        fig.savefig(path, dpi=320, facecolor='white')
        if fmt == 'svg':
            lines = path.read_text(encoding='utf-8').splitlines()
            path.write_text('\n'.join(line.rstrip() for line in lines) + '\n', encoding='utf-8')
    plt.close(fig)


def draw_figures(central, config, action_key):
    labels = [config['actions'][key]['label_en'] for key in central.index]
    colors = [COLORS[key] for key in central.index]
    months = central.months.iloc[0]
    note = f'{months}-month pilot • Central scenario • Illustrative euros and assumptions'
    plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#233548', 'text.color': '#233548',
                         'axes.titlesize': 15, 'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.75), sharey=True)
    budget = central.total_cost_eur / 1000
    thresholds = central.breakeven_departures_ceil
    for ax, vals, title, limit in [
        (left, budget, 'Total budget (€k)', max(budget.max() * 1.30, 1)),
        (right, thresholds, 'Departures to avoid', max(thresholds.max() * 1.44, 1)),
    ]:
        bars = ax.barh(labels, vals, color=colors, height=.52)
        text_values = ([f'€{v:,.1f}k' for v in vals] if ax is left else
                       [f'{v:d}  ({p:.2f} pp)' for v, p in zip(vals, central.breakeven_effect_pp)])
        ax.bar_label(bars, labels=text_values, padding=6, fontsize=12)
        ax.set(xlim=(0, limit), title=title)
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.tick_params(axis='both', length=0)
        ax.set_xticks([])
    left.invert_yaxis()
    fig.suptitle('What each pilot costs and what it must recover', fontweight='bold', fontsize=19)
    fig.text(.125, .035, note + ' • pp = break-even reduction before rounding', fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .90))
    save_figure(fig, '01_pilot_cost_break_even')

    row = central.loc[action_key]
    amounts = np.array([-row.total_cost_eur, row.direct_savings_eur, row.retention_value_eur]) / 1000
    cumulative = np.r_[0.0, amounts.cumsum()]
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    bar_colors = ['#b4544d', '#4d8093', '#597248', '#243b53']
    for i, amount in enumerate([*amounts, cumulative[-1]]):
        start = cumulative[i] if i < 3 else 0
        end = start + amount
        ax.bar(i, abs(amount), bottom=min(start, end), color=bar_colors[i], width=.58)
        ax.annotate(f'{amount:+,.1f}', (i, max(start, end)), xytext=(0, 7),
                    textcoords='offset points', ha='center', fontweight='bold')
        if i < 3:
            ax.hlines(end, i + .29, i + .71, color='#6a7c8c', linestyle=':')
    ax.axhline(0, color='#6a7c8c', linewidth=.8)
    ax.set(xticks=range(4), xticklabels=['Pilot cost', 'Direct savings', 'Value of departures\navoided', 'Net gain'],
           ylabel='€k', title=f"{config['actions'][action_key]['label_en']} pilot economics")
    padding = max(np.ptp(cumulative) * .15, 1)
    ax.set_ylim(cumulative.min() - padding, cumulative.max() + padding)
    fig.text(.105, .025, note + f' • Assumes a {row.effect_pp:g} pp reduction', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, 1))
    save_figure(fig, '02_overtime_waterfall')

    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    maximum = max(3, central.breakeven_effect_pp.max() * 1.15, central.effect_pp.max() * 1.1)
    x = np.linspace(0, maximum, 200)
    for (key, row), label, color in zip(central.iterrows(), labels, colors):
        y = (row.direct_savings_eur - row.total_cost_eur + x * row.covered_n *
             row.replacement_eur / 100) / 1000
        ax.plot(x, y, color=color, linewidth=2.5,
                label=f'{label} (break-even {row.breakeven_effect_pp:.2f} pp)')
        threshold_y = ((row.direct_savings_eur - row.total_cost_eur) / 1000
                       if row.breakeven_effect_pp == 0 else 0)
        ax.scatter(row.breakeven_effect_pp, threshold_y, color=color, edgecolor='white', s=75, zorder=5)
    ax.axhline(0, color='#405264', linewidth=1, linestyle='--', label='Break-even line (€0)')
    ax.set(xlim=(0, maximum), xlabel='Assumed reduction in voluntary departures (percentage points)',
           ylabel='Net gain (€k)', title='The retention effect drives pilot economics')
    ax.legend(frameon=False, loc='upper left', fontsize=10)
    fig.text(.105, .025, note, fontsize=10)
    fig.tight_layout(rect=(0, .07, 1, 1))
    save_figure(fig, '03_retention_sensitivity')


def pilot_inputs(results, config):
    return results[results.scope.eq('pilot')].set_index(['scenario', 'action']), {
        key: item['label_fr'] for key, item in config['actions'].items()
    }


def decision_facts(records, population):
    odds = pd.read_csv(REPORTS / 'tables/logistic_or_stability.csv').set_index('contrast')
    overtime = records.OverTime.eq('Yes')
    frequent = records.BusinessTravel.eq('Travel_Frequently')
    rare = records.BusinessTravel.eq('Travel_Rarely')
    return {
        'n': len(records), 'departures': int(records.depart.sum()),
        'women_departures': int(records.loc[records.Gender.eq('Female'), 'depart'].sum()),
        'overtime': int(overtime.sum()), 'overtime_departures': int(records.loc[overtime, 'depart'].sum()),
        'other_overtime': int((~overtime).sum()),
        'other_overtime_departures': int(records.loc[~overtime, 'depart'].sum()),
        'travel': int(frequent.sum()), 'travel_departures': int(records.loc[frequent, 'depart'].sum()),
        'rare_travel': int(rare.sum()), 'rare_travel_departures': int(records.loc[rare, 'depart'].sum()),
        'overtime_or': odds.loc['Heures supplémentaires : Oui / Non', 'odds_ratio'],
        'travel_or': odds.loc['Déplacements : fréquents / rares', 'odds_ratio'],
        'overlap': len(population['overtime'].index.intersection(population['travel'].index)),
    }


def fr(value, decimals=0):
    return f'{value:,.{decimals}f}'.replace(',', ' ').replace('.', ',')


def format_roi(value):
    return f'{fr(100 * value)} %' if pd.notna(value) else '—'


def write_decision(results, sensitivity, precision, facts, config):
    pilots, labels = pilot_inputs(results, config)
    central = pilots.loc['central']
    overtime, travel, careers = (central.loc[key] for key in ('overtime', 'travel', 'careers'))
    break_even_vs_travel = 100 * (
        travel.net_gain_eur + overtime.total_cost_eur - overtime.direct_savings_eur
    ) / (overtime.covered_n * overtime.replacement_eur)
    travel_savings_per_year = travel.total_cost_eur / travel.covered_n * 12 / travel.months
    rollout = results[(results.scope == 'rollout') & (results.scenario == 'central')].set_index('action')
    design = values(config['pilot_design'])
    general = values(config['general'])
    action_rows = '\n'.join(
        f"| {labels[key]} | {fr(row.eligible_proxy_n)} | {row.covered_n} | "
        f"{fr(row.launch_eur / 1000, 1)} | {fr(row.recurring_eur / 1000, 1)} | "
        f"{fr(row.direct_savings_eur / 1000, 1)} | {fr(row.replacement_eur / 1000, 1)} | "
        f"{row.breakeven_departures_ceil} | {fr(row.breakeven_effect_pp, 2)} | "
        f"{fr(row.net_gain_eur / 1000, 1)} | {format_roi(row.roi)} |"
        for key, row in central.iterrows()
    )
    overtime_ranges = sensitivity[sensitivity.action.eq('overtime')].groupby('dimension').net_gain_eur.agg(['min', 'max'])
    ranges = {
        key: f"{fr(bounds['min'] / 1000, 1)} à {fr(bounds['max'] / 1000, 1)}"
        for key, bounds in overtime_ranges.iterrows()
    }
    scenario_lines = '\n'.join(
        f"| {scenario['name']} | " + ' | '.join(
            fr(pilots.loc[(key, action), 'net_gain_eur'] / 1000, 1) for action in labels
        ) + ' |' for key, scenario in config['scenarios'].items()
    )
    report = f"""# HR Action Business Case

**Décision proposée.** Tester d’abord la refonte des plannings et la récupération sur des équipes utiles à l’expansion en Afrique, avec couverture de service et renforts sécurisés. Le pilote central de {overtime.months} mois couvre **{overtime.covered_n} salariés** et mobilise **{fr(overtime.total_cost_eur / 1000)} k€ de scénario**. Après {fr(overtime.direct_savings_eur / 1000)} k€ d’économies directes supposées, **{overtime.breakeven_departures_ceil} départ entier** suffit à couvrir le solde central. Avant arrondi, le point mort est **{fr(overtime.breakeven_effect_pp, 2)} point** de baisse absolue. Autoriser d’abord la préparation et la mesure de référence, puis engager le pilote après validation des devis et des seuils de service.

## Pourquoi ces actions

Le fichier contient {fr(facts['n'])} salariés, {facts['departures']} départs observés ({fr(100 * facts['departures'] / facts['n'], 2)} %) dont {facts['women_departures']} femmes. Le fichier ne documente ni la période, ni le caractère volontaire, ni l’affectation aux futurs sites africains. Le business case utilise une cohorte de départ à constituer. Il prend les associations, les odds ratios et les importances comme critères de priorisation des essais. Il laisse les réductions de départs au statut d’hypothèses.

| Action | Éligibilité et couverture | Mise en œuvre et responsable proposé | Lien au diagnostic |
|---|---|---|---|
| Heures supplémentaires | Équipes avec salariés exposés aux heures supplémentaires et compétences à confirmer. {overtime.eligible_proxy_n} dossiers proxy, {overtime.covered_n} salariés effectivement couverts au central. | Opérations et RH. Revoir la répartition des quarts, convenir des règles de récupération et budgéter les renforts. Mesurer les primes réellement évitées, en gardant les renforts au budget. | {facts['overtime_departures']}/{facts['overtime']} départs avec heures supplémentaires contre {facts['other_overtime_departures']}/{facts['other_overtime']} sans. OR ajusté L2 {fr(facts['overtime_or'], 2)}. |
| Déplacements | Équipes à déplacements fréquents. {travel.eligible_proxy_n} dossiers proxy, {travel.covered_n} salariés couverts au central. | Opérations et Achats. Cartographier les missions, basculer les tâches compatibles à distance, mutualiser les trajets et maintenir les visites essentielles. Mesurer billets, nuits et indemnités évités. | {facts['travel_departures']}/{facts['travel']} départs parmi les fréquents contre {facts['rare_travel_departures']}/{facts['rare_travel']} parmi les occasionnels. OR ajusté L2 {fr(facts['travel_or'], 2)}. |
| Carrières | Métiers et viviers de succession prioritaires à identifier avec RH. {fr(careers.eligible_proxy_n)} dossiers dans le cadre proxy actuel, {careers.covered_n} salariés couverts au central. | RH développement et managers. Entretiens de rétention, parcours de mobilité ou formation, temps de remplacement budgété. Aucune économie directe créditée. | Le parcours est le deuxième groupe par importance conjointe dans les deux modèles historiques. Les variables mêlent métier, niveau et ancienneté. |

Les métiers de référence sont les modalités de `JobRole` du fichier. Il faut les rapprocher du plan de compétences Afrique, du calendrier de montée en charge, des besoins de transfert de savoir-faire et des postes difficiles à pourvoir. Les effectifs proxy servent à borner la simulation. Aucun effectif pilote n’est confirmé. [Résultats statistiques](MODEL_RESULTS.md) · [Associations descriptives](tables/descriptive_associations.csv) · [Comparaison des modèles](../notebooks/attrition_models/SYNTHESE.md)

**Équité salariale et promotions.** Mandater RH rémunération pour rapprocher les grilles, primes, quotités, métiers, responsabilités, sites et ancienneté. Reconstituer les populations éligibles à chaque campagne, candidatures, promotions et augmentations datées. Examiner accès et résultats à périmètre comparable, en signalant les petites cellules. L’écart standardisé historique de −1,02 % sur 1 150 dossiers comparables justifie de documenter ces écarts sans mesurer une discrimination ni un effet sur la rétention. Faire chiffrer séparément par Finance le coût récurrent chargé de toute correction une fois l’audit réalisé. [Analyse salariale](tables/gender_standardization.csv)

## Budget, économies et seuils à {overtime.months} mois

Montants en **k€ illustratifs**. La couverture correspond au nombre de salariés recevant effectivement l’action. Les témoins sont exclus du budget d’action. Le statut, la valeur, l’unité et la source de chaque entrée modifiable figurent dans le registre.

| Action | Éligibles proxy | Couverts | Lancement | Récurrent | Économies directes | Remplacement par départ | Départs entiers à éviter | Seuil (points) | Gain net central | ROI central |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{action_rows}

Le brief Capgemini (slides 6 et 21) valorise un remplacement à **un an de salaire chargé**. La slide 14 identifie `MonthlyIncome` comme le salaire mensuel brut et `MonthlyRate` comme un taux de référence. Le calcul prend la moyenne brute par métier dans la base, la pondère selon la composition des éligibles proxy de chaque action, l’annualise (× 12), applique un coefficient chargé/brut de **{fr(general['loaded_pay_multiplier'], 2)}** et une convention de **{fr(general['eur_per_salary_unit'], 2)} € de scénario par unité source**. Ces deux coefficients sont modifiables. La devise de la base reste à confirmer. Les salaires par métier et leur pondération sont observés sur la base de référence, puis utilisés comme approximation de la composition du pilote et des départs évités. Un coût réel de remplacement par métier ou par site fourni par Finance remplacerait cette approximation.

Le budget additionne lancement et récurrent proratisé sur l’horizon. Les départs évités attendus sont la couverture multipliée par la baisse absolue supposée. La valeur de rétention est ce nombre multiplié par le coût de remplacement. Le bénéfice brut ajoute cette valeur aux économies directes. Le gain net déduit le coût total et le ROI divise ce gain par le coût positif. Pour le seuil, on divise le coût à couvrir après économies par le coût de remplacement, on borne à zéro et on arrondit au supérieur pour le nombre entier. Le seuil en points utilise le résultat **avant arrondi** en proportion des salariés couverts. Les espérances fractionnaires sont des sorties du scénario. La valeur de remplacement représente un coût évitable selon le brief, dont la part encaissable reste à confirmer. [Hypothèses](tables/hr_action_business_case/hypotheses.json) · [Composition par métier](tables/hr_action_business_case/replacement_by_role.csv) · [Calculs complets](tables/hr_action_business_case/action_scenarios.csv)

## Scénarios et critères susceptibles de changer l’ordre des actions

| Scénario · gain net en k€ illustratifs | Heures supplémentaires | Déplacements | Carrières |
|---|---:|---:|---:|
{scenario_lines}

Le prudent combine moins de couverture et d’économies, un budget plus élevé, une valeur de remplacement plus faible et une faible rétention. Le favorable inverse ces hypothèses. Le scénario « Aucun effet » conserve coûts, couverture et économies du central. Le scénario central suppose respectivement {fr(overtime.effect_pp, 2)}, {fr(travel.effect_pp, 2)} et {fr(careers.effect_pp, 2)} points de baisse. Ces ordres de grandeur servent à lire les seuils.

À paramètres centraux pour les déplacements, ceux-ci dépassent les heures supplémentaires en gain net si la baisse due à la refonte des horaires est inférieure à **{fr(break_even_vs_travel, 2)} point**. À effet nul, les déplacements équilibrent leur budget à **{fr(travel_savings_per_year)} € d’économies annuelles par salarié couvert** au lieu des {fr(travel.direct_savings_eur / travel.covered_n * 12 / travel.months)} € supposés. Tester les déplacements en premier si les économies sont confirmées à ce niveau avec maintien du service terrain, ou si les renforts d’horaires ne peuvent être assurés. Le pilote déplacements combine au central un budget plus faible et un meilleur ROI, tandis que les horaires présentent un gain net supérieur sous leurs hypothèses de rétention respectives. En sensibilité unitaire, le gain net des horaires varie de {ranges['replacement_multiplier']} k€ avec le coût de remplacement, de {ranges['budget_multiplier']} k€ avec le budget, de {ranges['coverage_multiplier']} k€ avec la couverture et de {ranges['effect_pp']} k€ avec la baisse de départs. Les autres paramètres sont maintenus au central à chaque variation. [Sensibilités chiffrées](tables/hr_action_business_case/one_way_sensitivity.csv) · [Courbes de gain net](figures/hr_action_business_case/03_retention_sensitivity.svg)

Le déploiement est une simulation distincte à confirmer après pilote. Au central, les trois périmètres de déploiement couvrent respectivement {rollout.loc['overtime', 'covered_n']}, {rollout.loc['travel', 'covered_n']} et {rollout.loc['careers', 'covered_n']} salariés, avec des budgets de {fr(rollout.loc['overtime', 'total_cost_eur']/1000, 1)}, {fr(rollout.loc['travel', 'total_cost_eur']/1000, 1)} et {fr(rollout.loc['careers', 'total_cost_eur']/1000, 1)} k€. Les hypothèses marginales sont reprises pour illustrer l’échelle et restent à réestimer. Le déploiement commence sur un nouvel horizon, avec son propre coût de lancement.

**Programme combiné.** Le fichier compte {facts['overlap']} dossiers éligibles aux deux leviers horaires et déplacements avec les filtres actuels. Le recouvrement des équipes effectivement traitées reste à mesurer. Pour chiffrer un programme, calculer l’union des salariés couverts à partir des identifiants et produire des segments mutuellement exclusifs par combinaison reçue. Valoriser les économies sur des postes de dépense dédupliqués, coûts partagés inclus une seule fois. Appliquer à chaque segment un effet propre mesuré sur le traitement reçu, dont un effet conjoint estimé pour les salariés qui reçoivent les deux actions. Un plan à quatre groupes (témoin, horaires, déplacements, les deux) peut identifier cet effet si le nombre d’équipes le permet. Les gains et départs évités des actions isolées ne sont pas totalisés.

## Pilote et critères de décision

1. **Avant lancement.** RH et Opérations cartographient les postes critiques et la capacité de remplacement. Mesurer la référence sur la période précédente. Apparier les équipes sur site, activité, charge, niveau de service et départs historiques. Répartir aléatoirement par équipe au sein des blocs si possible, avec liste d’équipes établie avant le tirage. À défaut, suivre des équipes comparables et vérifier les tendances préalables. Le protocole doit déclarer les mesures et marges de service avant le début.
2. **Évaluation.** Constituer une cohorte des salariés éligibles présents au démarrage. Suivre les départs volontaires datés sur {overtime.months} mois, rapportés aux salariés de départ de chaque bras. Traiter mobilité interne et départ involontaire comme événements distincts, consigner la perte de suivi et documenter toute contamination. Analyser selon l’équipe assignée avec intervalles tenant compte du regroupement. Comparer mensuellement heures supplémentaires payées par ETP, récupération, coûts de renfort, incidents, respect des SLA et éventuels transferts de charge.
3. **À 3 mois.** Poursuivre le pilote si au moins {fr(100 * design['coverage_gate'])} % de la couverture prévue est atteinte, si les heures supplémentaires payées par ETP baissent d’au moins {fr(design['overtime_gate_pp'])} % contre témoin et si le respect des SLA reste dans la marge provisoire de {fr(design['sla_max_loss_pp'])} point. Ajuster plannings ou renforts si l’adoption ou l’économie déçoit avec service préservé. Suspendre en cas de dégradation durable au-delà de la marge validée ou d’incident critique imputable au pilote.
4. **À {overtime.months} mois.** Poursuivre la validation élargie si le service est préservé et si coûts et économies révisés rendent l’effet nécessaire plausible. Conditionner le déploiement général à une preuve mieux dimensionnée que la baisse est compatible avec le seuil économique, idéalement une borne basse au-dessus du seuil quand la rétention est nécessaire à l’équilibre. Si les économies confirmées couvrent déjà les coûts et que le service est préservé, examiner le déploiement sur ce motif. Arrêter si les coûts et les risques opérationnels rendent le seuil irréaliste.

**Précision attendue.** Hypothèse de dimensionnement sur 12 mois de {fr(100 * design['control_rate'])} % de départs volontaires témoin, équipes de {design['team_size']}, corrélation intra-équipe {fr(design['icc'], 2)}. Avec {precision['covered_per_arm_for_precision']} salariés par bras (environ {precision['teams_per_arm']} équipes chacun), l’intervalle bilatéral à {fr(100 * (1 - design['alpha']))} % pour une différence proche de zéro a une demi-largeur approximative de **{fr(precision['ci_half_width_pp_at_null'], 1)} points**. Détecter {fr(precision['target_effect_pp'], 2)} points avec une puissance de {fr(100 * design['power'])} % demanderait environ **{precision['teams_per_arm_for_target_power']} équipes par bras**, tailles égales, approximation normale avec facteur de plan. Le pilote mesure surtout faisabilité, couverture, économies et service. Pour mesurer la rétention, recalculer la puissance avec les données datées du client puis étendre par vagues randomisées avec la même définition de départ et des fenêtres fixes. [Méthode de dimensionnement par grappes](https://www.bmj.com/content/328/7441/702.full) · [Calcul pilote](tables/hr_action_business_case/pilot_precision.csv)

## Chiffres à obtenir avant engagement

- **Finance et Paie.** Devise, contributions patronales, rémunération fixe et variable par métier/site, coûts de recrutement, vacance, intégration et montée en compétence, part évitable, devis de renfort et coûts incrémentaux. Confirmer l’absence de double comptage entre gains et coûts.
- **RH et Direction expansion.** Métiers et compétences critiques, équipes et salariés éligibles, calendrier de mobilité vers l’Afrique, dates et motifs de départ, effectifs au démarrage, renouvellement des équipes, contraintes de formation, intensité de mobilité, quotité/ETP et dossiers de rémunération et promotion.
- **Opérations et Achats.** Heures supplémentaires effectivement payées et récupérées, primes, marges de couverture, SLA, incidents, historique des trajets, coûts annulables et impact des alternatives distancielles.

Les slides du midterm 5–7 fournissent le diagnostic. Les slides 8–9 sont le point d’insertion proposé pour la [trame finale en quatre slides](HR_ACTION_BUSINESS_CASE_SLIDES.md). Les présentations sources ont été consultées dans le dossier parent et restent inchangées.
"""
    (REPORTS / 'HR_ACTION_BUSINESS_CASE.md').write_text(report, encoding='utf-8')
    write_slides(pilots, central, precision, facts)


def write_slides(pilots, central, precision, facts):
    overtime = central.loc['overtime']
    labels = {'overtime': 'Overtime', 'travel': 'Travel', 'careers': 'Careers'}
    pilot_numbers = '\n'.join(
        f"- **{labels[key]}** · {row.covered_n} covered · €{row.total_cost_eur/1000:,.0f}k budget · "
        f"{row.breakeven_departures_ceil:d} {'departure' if row.breakeven_departures_ceil == 1 else 'departures'} "
        f"to avoid ({row.breakeven_effect_pp:.2f} pp before rounding)."
        for key, row in central.iterrows()
    )
    scenario_numbers = '\n'.join(
        f"- **{caption}** · " + ' / '.join(f'{group.loc[key, "net_gain_eur"]/1000:+,.0f}' for key in labels) + ' k€.'
        for scenario, caption in [('prudent', 'Prudent'), ('central', 'Central'),
                                  ('favorable', 'Favourable'), ('no_effect', 'No retention effect')]
        for group in [pilots.loc[scenario]]
    )
    slides = f"""# HR Action Business Case · four-slide storyline

Use these slides after the working conditions and model evidence in the midterm deck.

## 1. Test overtime redesign first to protect skills needed for expansion

**Visual.** Three-row action matrix by workforce exposure, intervention and owner.

- **Overtime · Operations and HR.** Redesign rosters, implement time-off rules and budget backfill. The file records {facts['overtime_departures']} departures among {facts['overtime']:,} people working overtime versus {facts['other_overtime_departures']} among {facts['other_overtime']:,} without overtime.
- **Travel · Operations and Procurement.** Replace avoidable trips and protect essential site visits. The file records {facts['travel_departures']} departures among {facts['travel']:,} frequent travellers.
- **Careers · HR and line managers.** Test stay interviews, mobility and training for confirmed critical roles. Audit pay and promotion decisions as a separate workstream.

**Assumptions to show.** Observed departure proportions have an unknown reference period. Critical roles and sites for the Africa launch require client confirmation.

## 2. The central overtime pilot needs {overtime.breakeven_departures_ceil} avoided departure to break even

**Visual.** Two-panel budget and threshold chart. [PNG](figures/hr_action_business_case/01_pilot_cost_break_even.png) · [SVG](figures/hr_action_business_case/01_pilot_cost_break_even.svg)

{pilot_numbers}

**Assumptions to show.** {overtime.months}-month pilot. Illustrative euros. One year of fully loaded salary per replacement, using the observed role mix as a proxy. Direct savings reduce the budget to recover.

## 3. Savings alone leave the overtime pilot below break-even in the central case

**Visual.** Overtime waterfall with sensitivity curves beside it. [Waterfall PNG](figures/hr_action_business_case/02_overtime_waterfall.png) · [Waterfall SVG](figures/hr_action_business_case/02_overtime_waterfall.svg) · [Sensitivity PNG](figures/hr_action_business_case/03_retention_sensitivity.png) · [Sensitivity SVG](figures/hr_action_business_case/03_retention_sensitivity.svg)

- **Overtime central case.** €{overtime.total_cost_eur/1000:,.0f}k cost, €{overtime.direct_savings_eur/1000:,.0f}k direct savings and €{overtime.retention_value_eur/1000:,.0f}k replacement cost avoided give a net gain of €{overtime.net_gain_eur/1000:,.0f}k, assuming a {overtime.effect_pp:g} pp reduction.
- **Scenario net gains** in order of overtime, travel, careers.
{scenario_numbers}

**Assumptions to show.** Scenario effects are illustrative absolute changes in voluntary departures. Validate travel savings and staffing costs before choosing between actions. Separate action gains cannot be added for overlapping teams.

## 4. Test service and savings now, then expand the trial to establish retention impact

**Visual.** Roadmap with team allocation, a three-month operating review and a {overtime.months}-month retention readout.

- **Allocate comparable teams.** Randomise by team where feasible, with approximately {precision['teams_per_arm']} teams and {precision['covered_per_arm_for_precision']} employees per arm in the design example.
- **Track operations.** Monitor paid overtime per FTE, recovery time, backfill spend, incidents and service levels against the comparison teams. Pause if a critical incident is attributable to the pilot or service falls outside the agreed margin.
- **Capture voluntary departures.** Follow the starting cohort for {overtime.months} months. The planning interval is roughly ±{precision['ci_half_width_pp_at_null']:.1f} pp near zero, too wide to resolve the economic threshold. Expand by randomised waves if the operational and budget review supports it.

**Assumptions to show.** Precision uses an illustrative baseline, equal team sizes and within-team correlation. Collect dated voluntary exits, payroll and actual costs before a rollout decision.
"""
    (REPORTS / 'HR_ACTION_BUSINESS_CASE_SLIDES.md').write_text(slides, encoding='utf-8')


def main():
    plt.switch_backend('Agg')
    config = json.loads((TABLES / 'hypotheses.json').read_text(encoding='utf-8'))
    for action in config['actions']:
        central = scenario_inputs(config['scenarios']['central'], action)
        zero = scenario_inputs(config['scenarios']['no_effect'], action)
        if zero != {**central, 'effect_pp': 0}:
            raise ValueError('Le scénario sans effet doit reprendre les paramètres centraux, avec un effet nul.')
    records = load_data()
    results, sensitivity, population, mix = compute(config, records)
    pilots, _ = pilot_inputs(results, config)
    overtime = pilots.loc[('central', 'overtime')]
    design = values(config['pilot_design'])
    precision = {
        'target_effect_pp': design['target_effect_pp'],
        **pilot_precision(design, int(overtime.covered_n), design['target_effect_pp']),
    }
    general = values(config['general'])
    mix['observed_source'] = 'data/employee.csv.xlsx · MonthlyIncome / JobRole'
    mix['base_replacement_eur'] = (mix.gross_monthly * 12 * general['loaded_pay_multiplier'] *
                                   general['eur_per_salary_unit'] * general['replacement_years'])
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for name, table in [('action_scenarios', results), ('one_way_sensitivity', sensitivity),
                        ('replacement_by_role', mix), ('pilot_precision', pd.DataFrame([precision]))]:
        table.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')
    draw_figures(pilots.loc['central'], config, 'overtime')
    write_decision(results, sensitivity, precision, decision_facts(records, population), config)
    print(f'{len(results)} scénarios action/périmètre, {len(sensitivity)} sensibilités et 3 figures exportés.')


if __name__ == '__main__':
    main()
