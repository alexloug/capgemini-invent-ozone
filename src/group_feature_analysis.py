"""Reproduce the grouped HR analysis: python -m src.group_feature_analysis.

Original models/data are read only. Outputs are isolated in outputs/group_feature_analysis.
"""
import json
import platform
from hashlib import sha256
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import FunctionTransformer
from scipy.stats import rankdata

from .modeling import (BINARY, DATA_PATH, MODEL_DIR, load_model_data, make_split,
                       verify_reference_split)
from .data import ROOT

OUT = ROOT / 'outputs/group_feature_analysis'
REPEATS = 30
SEED = 42
# Exhaustive, disjoint business allocation; no outcome-derived variables.
FEATURE_GROUPS = {
    'Working Conditions': ['OverTime', 'WorkLifeBalance', 'EnvironmentSatisfaction',
                           'BusinessTravel', 'Department'],
    'Gender Equality': ['Gender'],
    'Compensation': ['DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate',
                     'PercentSalaryHike', 'StockOptionLevel'],
    'Career Development': ['JobLevel', 'JobRole', 'NumCompaniesWorked', 'TotalWorkingYears',
                           'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole',
                           'YearsSinceLastPromotion', 'YearsWithCurrManager', 'PerformanceRating'],
    'Engagement & Satisfaction': ['JobSatisfaction', 'JobInvolvement', 'RelationshipSatisfaction'],
    'Employee Profile': ['Age', 'Education', 'EducationField', 'MaritalStatus', 'DistanceFromHome'],
}
FEATURE_TO_GROUP = {c: g for g, columns in FEATURE_GROUPS.items() for c in columns}
ASSIGNMENT_NOTES = {
    'Gender Equality': 'Gender only: demographic signal, NOT a measure of equality. Pay and career outcomes are audited separately by gender, not duplicated as predictors.',
    'Department': 'Organisational context assigned to working conditions; could also reflect career opportunities.',
    'DistanceFromHome': 'Residential profile; also relevant to working conditions. Assignment sensitivity is reported.',
    'EnvironmentSatisfaction': 'Work environment assigned to working conditions rather than engagement.',
    'JobRole': 'Career position; also proxies working conditions and compensation.',
    'PerformanceRating': 'Career assessment; may also affect pay.',
    'DailyRate / HourlyRate / MonthlyRate': 'Kept under compensation, but definitions and comparability with income are undocumented.',
    'Gender': 'Existing Female=1, Male=0 coding retained; no broader gender categories are available.',
}


def save_csv(frame, name):
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')


def binary_subset(frame):
    return frame.apply(lambda s: s.map(BINARY[s.name]).astype(int))


def subset_pipeline(template, columns):
    """Preserve existing encodings and parameters, learning preprocessing per fit."""
    pipe = clone(template)
    transformers = []
    for name, transformer, original in pipe.named_steps['preparation'].transformers:
        selected = [c for c in original if c in columns]
        if not selected:
            continue
        if name == 'binaires':
            transformer = FunctionTransformer(binary_subset, feature_names_out='one-to-one')
        transformers.append((name, transformer, selected))
    pipe.named_steps['preparation'].transformers = transformers
    return pipe


def metrics(y, p):
    return {'roc_auc': roc_auc_score(y, p), 'pr_auc_ap': average_precision_score(y, p),
            'recall': recall_score(y, p >= .5, zero_division=0),
            'precision': precision_score(y, p >= .5, zero_division=0)}


def stratified_bootstrap(y, repeats=500):
    rng = np.random.default_rng(SEED + 1)
    classes = [np.flatnonzero(np.asarray(y) == k) for k in (0, 1)]
    return [np.concatenate([rng.choice(a, len(a), replace=True) for a in classes])
            for _ in range(repeats)]


def batch_roc_auc(y, probabilities):
    """Mann–Whitney ROC-AUC with average ranks for ties, batched by prediction row."""
    positive = np.asarray(y) == 1
    n1, n0 = positive.sum(), (~positive).sum()
    ranks = rankdata(probabilities, method='average', axis=-1)
    return (ranks[..., positive].sum(axis=-1) - n1*(n1+1)/2) / (n1*n0)


def permute_columns(frame, columns, permutation):
    result = frame.copy(deep=True)
    # A shared positional permutation preserves all within-group relationships.
    result.loc[:, columns] = frame.iloc[permutation][columns].to_numpy()
    return result


def permutation_analysis(model, X, y, groups, bootstrap=False):
    original = X.copy(deep=True)
    baseline_p = model.predict_proba(X)[:, 1]
    baseline = roc_auc_score(y, baseline_p)
    permutations = [np.random.default_rng(SEED + r).permutation(len(X)) for r in range(REPEATS)]
    boots = stratified_bootstrap(y) if bootstrap else []
    rows, raw = [], []
    for group, columns in groups.items():
        predictions = np.array([model.predict_proba(permute_columns(X, columns, p))[:, 1]
                                for p in permutations])
        drops = np.array([baseline - roc_auc_score(y, p) for p in predictions])
        row = {'group': group, 'n_features': len(columns), 'baseline_roc_auc': baseline,
               'importance_mean': drops.mean(), 'permutation_sd': drops.std(ddof=1),
               'permutation_q025': np.quantile(drops, .025),
               'permutation_q975': np.quantile(drops, .975), 'repeats': REPEATS}
        if bootstrap:
            # Average AUC decreases, not AUC of averaged predictions.
            all_predictions = np.vstack([baseline_p, predictions])
            ci = []
            for b in boots:
                aucs = batch_roc_auc(np.asarray(y)[b], all_predictions[:, b])
                ci.append(aucs[0] - aucs[1:].mean())
            row.update(bootstrap_ci_low=np.quantile(ci, .025), bootstrap_ci_high=np.quantile(ci, .975))
        print(f'  {group}: mean AUC decrease {row["importance_mean"]:.4f}', flush=True)
        rows.append(row)
        raw.extend({'group': group, 'repeat': r, 'auc_decrease': d} for r, d in enumerate(drops))
    pd.testing.assert_frame_equal(X, original)
    return pd.DataFrame(rows).sort_values('importance_mean', ascending=False), pd.DataFrame(raw)


def cv_comparisons(models, X, y, folds):
    records = []
    for name, template in models.items():
        for fold, (fit, valid) in enumerate(folds):
            cases = [('global', 'All groups', list(X))]
            for group, columns in FEATURE_GROUPS.items():
                cases += [('removed', group, [c for c in X if c not in columns]), ('only', group, columns)]
            for kind, group, columns in cases:
                pipe = subset_pipeline(template, columns)
                pipe.fit(X.iloc[fit][columns], y.iloc[fit])
                p = pipe.predict_proba(X.iloc[valid][columns])[:, 1]
                records.append({'model': name, 'fold': fold, 'kind': kind, 'group': group,
                                **metrics(y.iloc[valid], p)})
            print(f'CV {name}, fold {fold + 1}/5 complete', flush=True)
    detail = pd.DataFrame(records)
    save_csv(detail, 'cv_fold_results')
    global_rows = detail[detail.kind.eq('global')].set_index(['model', 'fold'])
    ablation = detail[detail.kind.eq('removed')].copy()
    for metric in ['roc_auc', 'pr_auc_ap']:
        ablation['global_' + metric] = [global_rows.loc[(r.model, r.fold), metric] for r in ablation.itertuples()]
        ablation['delta_' + metric] = ablation['global_' + metric] - ablation[metric]
    save_csv(ablation, 'group_ablation_results')
    save_csv(detail[detail.kind.eq('only')], 'group_only_model_results')
    return detail, ablation


def gender_audit(X, y):
    frame = X.assign(depart=y)
    outcomes = ['depart', 'MonthlyIncome', 'PercentSalaryHike', 'YearsSinceLastPromotion', 'JobLevel']
    raw = frame.groupby('Gender')[outcomes].agg(['size', 'mean', 'median'])
    raw.columns = ['_'.join(c) for c in raw.columns]
    save_csv(raw.reset_index(), 'gender_outcome_summary')
    strata = frame.groupby(['JobRole', 'JobLevel', 'Gender'], observed=True).agg(
        n=('depart', 'size'), attrition=('depart', 'mean'), income=('MonthlyIncome', 'mean'),
        salary_hike=('PercentSalaryHike', 'mean'), promotion_delay=('YearsSinceLastPromotion', 'mean')).reset_index()
    save_csv(strata, 'gender_role_level_strata')
    counts = strata.pivot(index=['JobRole', 'JobLevel'], columns='Gender', values='n')
    rows = []
    for min_n in (1, 5, 10):
        common = counts.dropna().loc[lambda z: z.min(axis=1).ge(min_n)]
        weights = common.sum(axis=1) / common.sum().sum()
        for outcome in ['attrition', 'income', 'salary_hike', 'promotion_delay']:
            means = strata.pivot(index=['JobRole', 'JobLevel'], columns='Gender', values=outcome).loc[common.index]
            result = means.mul(weights, axis=0).sum()
            rows.append({'minimum_per_gender_per_stratum': min_n, 'outcome': outcome,
                         'covered_n': int(common.sum().sum()), 'total_n': len(frame),
                         'n_strata': len(common), 'female': result.Female, 'male': result.Male,
                         'female_minus_male': result.Female-result.Male,
                         'female_vs_male_percent': 100*(result.Female/result.Male-1)})
    adjusted = pd.DataFrame(rows)
    save_csv(adjusted, 'gender_standardized_outcomes')
    return raw, adjusted


def correlations(X):
    from .features import numeric_redundancies
    numeric = X.select_dtypes('number').columns.tolist()
    pairs, _ = numeric_redundancies(X, numeric)
    pairs['group_1'] = pairs.variable_1.map(FEATURE_TO_GROUP)
    pairs['group_2'] = pairs.variable_2.map(FEATURE_TO_GROUP)
    save_csv(pairs, 'feature_correlations_training')
    # Cramer's V handles nominal variables; diagonal and finite-sample bias are not corrected.
    cats = X.select_dtypes('object').columns.tolist()
    from scipy.stats import chi2_contingency
    rows = []
    for i, a in enumerate(cats):
        for b in cats[i+1:]:
            table = pd.crosstab(X[a], X[b])
            v = np.sqrt(chi2_contingency(table, correction=False)[0] / (len(X)*(min(table.shape)-1)))
            rows.append({'feature_1': a, 'feature_2': b, 'cramers_v': v,
                         'group_1': FEATURE_TO_GROUP[a], 'group_2': FEATURE_TO_GROUP[b]})
    save_csv(pd.DataFrame(rows), 'categorical_associations_training')
    return pairs


def coefficients(model):
    pre = model.named_steps['preparation']
    names = pre.get_feature_names_out()
    coefs = model.named_steps['modele'].coef_[0]
    frame = pd.DataFrame({'transformed_feature': names, 'coefficient': coefs,
                          'exp_coefficient': np.exp(coefs)})
    frame['direction'] = np.where(frame.coefficient > 0, 'positive', 'negative')
    frame['interpretation'] = 'L2 contribution; exp(beta) is not a category-versus-reference odds ratio'
    save_csv(frame, 'logistic_coefficients')
    # Complete one-hot coding: valid category comparisons use beta_a - beta_b.
    contrasts = []
    encoder = pre.named_transformers_['categories_scores']
    start = pre.output_indices_['categories_scores'].start
    for variable, categories in zip(encoder.feature_names_in_, encoder.categories_):
        betas = coefs[start:start+len(categories)]
        for category, beta in zip(categories[1:], betas[1:]):
            contrasts.append({'feature': variable, 'category': category, 'reference': categories[0],
                              'log_odds_difference': beta-betas[0], 'odds_ratio': np.exp(beta-betas[0])})
        start += len(categories)
    for name, beta in zip(names, coefs):
        if name.startswith(('binaires__', 'quantitatives__')):
            contrasts.append({'feature': name.split('__')[1], 'category': '1 versus 0' if name.startswith('binaires') else '+1 training SD',
                              'reference': '0' if name.startswith('binaires') else 'current value',
                              'log_odds_difference': beta, 'odds_ratio': np.exp(beta)})
    contrasts = pd.DataFrame(contrasts)
    save_csv(contrasts, 'logistic_odds_ratios')
    return contrasts


def figures(group_results, individual, cv):
    sns.set_theme(style='whitegrid', font_scale=1.05)
    colors = {'Logistic Regression': '#0070AD', 'Random Forest': '#12ABDB'}
    def finish(fig, filename):
        fig.tight_layout()
        fig.savefig(OUT / 'figures' / filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    for name, result in group_results.items():
        result = result.sort_values('importance_mean')
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.barh(result.group, result.importance_mean, color=colors[name])
        # Endpoint CIs need not contain the sample estimate, so draw segments directly.
        ax.hlines(np.arange(len(result)), result.bootstrap_ci_low, result.bootstrap_ci_high, color='#263746', linewidth=2)
        ax.axvline(0, color='#555555', linewidth=1)
        ax.set(xlabel='Decrease in test ROC-AUC (500 paired bootstrap samples: 95% interval)',
               title=f'{name} | Grouped permutation importance')
        finish(fig, 'group_importance_' + ('logistic' if name.startswith('Logistic') else 'random_forest') + '.png')
    fig, ax = plt.subplots(figsize=(11, 6))
    order = list(group_results['Logistic Regression'].group)[::-1]
    for offset, (name, result) in zip([-.17, .17], group_results.items()):
        r = result.set_index('group').loc[order]
        pos = np.arange(len(order)) + offset
        ax.barh(pos, r.importance_mean, height=.3, label=name, color=colors[name])
        ax.hlines(pos, r.bootstrap_ci_low, r.bootstrap_ci_high, color='#263746')
    ax.set(yticks=np.arange(len(order)), yticklabels=order, xlabel='Decrease in test ROC-AUC',
           title='HR information used by each global model | 95% paired bootstrap intervals')
    ax.axvline(0, color='gray', linewidth=1)
    ax.legend(loc='lower right')
    finish(fig, 'group_importance_comparison.png')
    for name in colors:
        r = individual[individual.model.eq(name)].nlargest(15, 'importance_mean').sort_values('importance_mean')
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.barh(r.group, r.importance_mean, xerr=r.permutation_sd, color=colors[name])
        ax.axvline(0, color='gray', linewidth=1)
        ax.set(title=f'{name} | Top 15 original features', xlabel='Decrease in test ROC-AUC (± 1 permutation SD; not a confidence interval)')
        finish(fig, 'individual_' + ('logistic' if name.startswith('Logistic') else 'random_forest') + '.png')
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for i, name in enumerate(colors):
        d = cv[cv.model.eq(name)]
        for j, metric in enumerate(['roc_auc', 'pr_auc_ap']):
            matrix = pd.DataFrame(index=FEATURE_GROUPS)
            matrix['Global'] = d[d.kind.eq('global')][metric].mean()
            for kind, label in [('only', 'Group only'), ('removed', 'Group removed')]:
                matrix[label] = d[d.kind.eq(kind)].groupby('group')[metric].mean()
            sns.heatmap(matrix, annot=True, fmt='.3f', cmap='Blues', ax=axes[i, j], vmin=0, vmax=1)
            axes[i, j].set_title(f'{name} | {metric} | mean of 5 training folds')
    finish(fig, 'group_model_performance_heatmap.png')


def markdown_table(frame, digits=3):
    # Avoid a new optional tabulate dependency.
    data = frame.copy()
    for c in data.select_dtypes('number'):
        data[c] = data[c].map(lambda x: f'{x:.{digits}f}')
    return '| ' + ' | '.join(data.columns) + ' |\n|' + '|'.join(['---']*len(data.columns)) + '|\n' + '\n'.join('| ' + ' | '.join(map(str, row)) + ' |' for row in data.to_numpy())


def report(performance, groups, individual, ablation, cv, gender, pairs, odds, shap_status):
    observations = []
    for name, result in groups.items():
        first, second = result.iloc[0], result.iloc[1]
        top_feature = individual[individual.model.eq(name)].iloc[0]
        observations.append(f'**{name}** : {first.group} arrive en tête (baisse moyenne d’AUC {first.importance_mean:.3f}), '
                            f'puis {second.group} ({second.importance_mean:.3f}). '
                            f'Première variable individuelle : {top_feature.group} ({top_feature.importance_mean:.3f}).')
    cv_summary = flatten_columns(cv[cv.kind.eq('global')].groupby('model')[['roc_auc', 'pr_auc_ap', 'recall', 'precision']].agg(['mean', 'std']).reset_index())
    sections = ['# Ozone — Importance des groupes de variables RH',
        '## Constats principaux', *observations,
        'Ce classement décrit l’information utilisée par les modèles sur cet échantillon. '
        'Il motive des diagnostics RH et des hypothèses d’action, sans mesurer l’effet d’une intervention. '
        'Le rappel au seuil 0,5 reste limité malgré une performance de classement supérieure à la baseline.',
        '## Protocole et performance observée',
        'Source : `data/variantes/01_baseline.xlsx`. Cible issue du chargeur existant : Attrition Yes=1 (départ), No=0. '
        '1 518 salariés, 30 variables, 248 départs ; 1 214 salariés/198 départs en entraînement et 304/50 en test. '
        'Split et cinq plis ordonnés vérifiés à l’identique des expériences 04/05. Pipelines sauvegardés réutilisés pour le test ; '
        'prétraitement réappris dans chaque pli pour les comparaisons. Paramètres retenus précédemment sur AP, dont class_weight=None pour les deux modèles ; '
        'la recherche existante avait aussi testé balanced. Seuil de classification fixé à 0,5. PR-AUC est ici l’average precision (AP), pas l’intégrale trapézoïdale.',
        markdown_table(performance),
        'Validation croisée des modèles globaux sur l’entraînement (moyenne et écart-type des cinq plis ; pas des IC) :',
        markdown_table(cv_summary),
        'Les IC de performance proviennent de 500 bootstraps stratifiés du test, conditionnels au modèle entraîné. '
        'Le test a déjà été consulté dans les notebooks : cette analyse est exploratoire, sans nouvelle validation externe. '
        'Les scores CV ne sont pas imbriqués et peuvent être optimistes après la sélection historique des paramètres.',
        '## Classement des groupes — constat prédictif']
    for name, result in groups.items():
        sections += ['### ' + name, markdown_table(result[['group', 'importance_mean', 'permutation_sd', 'bootstrap_ci_low', 'bootstrap_ci_high']])]
    sections += ['Permutation conjointe des colonnes originales avec le même ordre de lignes (30 répétitions), puis passage dans le pipeline figé. '
        'Importance = AUC initiale − AUC permutée. Les IC bootstrap rééchantillonnent les mêmes salariés pour les prédictions initiales et permutées '
        'et moyennent les 30 différences d’AUC. Ils ne couvrent ni le réentraînement ni toute la variabilité de sélection. '
        'Les quantiles/écarts-types des permutations décrivent une variabilité de permutation, pas un IC populationnel. '
        'Une valeur négative est conservée ; les importances ne sont ni additives ni des pourcentages d’explication.',
        '## Retrait de groupes et information autonome',
        'Le retrait réentraîne le modèle sans le groupe : Δ positif = perte de performance par rapport au global sur le même pli. '
        'Le modèle par groupe seul mesure une information autonome, pas une contribution unique. Les hyperparamètres restent fixes ; '
        'ces résultats ne comparent pas des modèles réduits réoptimisés. Les CSV contiennent les cinq résultats appariés, sans test t sur des plis dépendants.',
        markdown_table(ablation.groupby(['model', 'group'])[['delta_roc_auc', 'delta_pr_auc_ap']].agg(['mean', 'std']).reset_index().pipe(flatten_columns)),
        markdown_table(cv[cv.kind.eq('only')].groupby(['model', 'group'])[['roc_auc', 'pr_auc_ap']].mean().reset_index()),
        '## Variables individuelles et directions des associations']
    for name in groups:
        sections += ['### ' + name, markdown_table(individual[individual.model.eq(name)].head(15)[['group', 'importance_mean', 'permutation_sd']])]
    directional = pd.concat([odds.nlargest(7, 'log_odds_difference'), odds.nsmallest(7, 'log_odds_difference')])
    sections += [markdown_table(directional),
        'Les différences de log-cotes positives indiquent une association conditionnelle positive avec le départ ; les négatives une association négative. '
        'Les OR catégoriels sont exp(β_modalité − β_référence), car le codage existant conserve toutes les modalités. '
        'Pour les quantitatives, OR par écart-type appris sur l’entraînement ; Gender compare Female à Male et OverTime Yes à No. '
        'Ces paramètres sont pénalisés L2 et sensibles aux variables corrélées. Aucun IC classique ni p-value ne leur est attribué : '
        'le codage complet est redondant et une inférence MLE classique après sélection serait invalide. Les OR ne sont pas des effets causaux.',
        '## Conditions de travail versus égalité femmes-hommes',
        'Le groupe « Gender Equality » contient uniquement Gender. Son importance n’évalue ni les inégalités ni le rendement d’une politique d’égalité. '
        'Les revenus et carrières sont affectés à leurs propres groupes, sans duplication. Comparer cinq variables de conditions de travail à une seule '
        'variable démographique ne constitue pas un test symétrique de politiques RH.',
        markdown_table(pd.concat([r[r.group.isin(['Working Conditions', 'Gender Equality'])].assign(model=n) for n, r in groups.items()])[['model', 'group', 'importance_mean', 'bootstrap_ci_low', 'bootstrap_ci_high']]),
        'Audit descriptif sur les seuls salariés d’entraînement : standardisation avec les mêmes poids de métier/niveau sur les strates communes. '
        'Seuils de 1, 5 et 10 personnes de chaque genre par strate pour exposer la sensibilité à la couverture. '
        'Les écarts ci-dessous sont observés, sans test de significativité et sans ajustement pour l’expérience, le temps de travail ou le pays.',
        markdown_table(gender),
        'Les CSV fournissent aussi les effectifs, moyennes et médianes brutes ainsi que les strates. Le délai depuis la dernière promotion '
        'n’est pas un taux de promotion et ne permet pas de conclure à un accès égal ou inégal aux promotions.',
        '## Robustesse et limites',
        markdown_table(pairs[pairs.group_1.ne(pairs.group_2)].head(8)[['variable_1', 'variable_2', 'spearman', 'pearson', 'group_1', 'group_2']]),
        'Corrélations numériques calculées sur l’entraînement uniquement ; associations nominales (V de Cramér non corrigé) exportées séparément. '
        'Des groupes corrélés peuvent se substituer : permutation et retrait ne mesurent pas la même chose. Permuter peut créer des combinaisons '
        'intergroupes irréalistes. La taille des groupes influe sur leur importance. Une analyse alternative transfère DistanceFromHome du profil '
        'aux conditions de travail (`assignment_sensitivity.csv`). Les petits écarts et IC recouvrants ne justifient pas un classement ferme. '
        'Les IC marginaux ne sont pas des tests multiples ni des IC de différences entre groupes.',
        'Un seul split et 50 départs en test limitent la précision. Si les IC des performances approchent la baseline ou si les contributions '
        'sont proches de zéro, le classement reste fragile. Même au-dessus de la baseline, une validation temporelle indépendante est nécessaire. '
        'Scores hors dictionnaire et 23 incohérences d’ancienneté sur 20 lignes conservés selon le protocole existant. '
        'La variante 01 n’est pas une correction des valeurs. Chronologie, devise, unités et représentativité géographique inconnues ; '
        'une fuite temporelle ne peut pas être exclue par le seul split. Aucune causalité ni discrimination n’est établie.',
        'SHAP : ' + shap_status,
        '## Trois recommandations à tester',
        '1. **Conditions de travail** : prioriser un diagnostic de charge, heures supplémentaires et déplacements ; '
        'tester un pilote de planning/récupération avec groupes comparables et suivi prospectif. Les associations prédictives motivent '
        'le diagnostic ; la baisse des départs reste une hypothèse à évaluer.',
        '2. **Parcours et rétention** : confronter les signaux de carrière, rémunération et engagement aux entretiens de départ ; '
        'tester un accompagnement des mobilités et des managers. Les contributions mesurées ne démontrent pas l’effet d’une promotion ou d’une hausse de salaire.',
        '3. **Égalité** : poursuivre l’audit des rémunérations à responsabilités et temps de travail comparables, collecter candidatures '
        'et décisions de promotion datées. Un faible signal prédictif du genre ne justifie pas de déprioriser l’égalité ; '
        'les écarts descriptifs doivent être vérifiés avec ces données supplémentaires.',
        '## Reproduction',
        '`python -m src.group_feature_analysis` depuis la racine avec les dépendances existantes. '
        'Résultats dans `outputs/group_feature_analysis/`, figures PNG à 300 DPI. '
        'Sources et modèles 04/05 inchangés ; versions, paramètres et empreintes dans `analysis_metadata.json`.']
    (OUT / 'group_feature_analysis_report.md').write_text('\n\n'.join(sections) + '\n', encoding='utf-8')


def flatten_columns(frame):
    frame.columns = ['_'.join(str(v) for v in c if v) if isinstance(c, tuple) else c for c in frame.columns]
    return frame


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'figures').mkdir(exist_ok=True)
    protected = list((ROOT / 'data').rglob('*.xlsx')) + list(MODEL_DIR.glob('*'))
    fingerprints = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    X, y, source_rows, _, _ = load_model_data()
    assert len(FEATURE_TO_GROUP) == sum(map(len, FEATURE_GROUPS.values()))
    assert set(X) == set(FEATURE_TO_GROUP)
    assert y.eq(pd.read_excel(DATA_PATH).Attrition.eq('Yes').astype(int)).all()
    train, test, yt, ytest, folds, split_rows = make_split(X, y, source_rows)
    verify_reference_split(split_rows)
    pd.testing.assert_frame_equal(pd.read_csv(MODEL_DIR / '04_indices_split_plis.csv'),
                                  pd.read_csv(MODEL_DIR / '05_indices_split_plis.csv'))
    meta5 = json.loads((MODEL_DIR / '05_metadonnees.json').read_text())
    assert meta5['sha256_source'] == sha256(DATA_PATH.read_bytes()).hexdigest()
    models = {'Logistic Regression': joblib.load(MODEL_DIR / '04_Logistique.joblib'),
              'Random Forest': joblib.load(MODEL_DIR / '05_RandomForest.joblib')}
    dummy = joblib.load(MODEL_DIR / '04_Baseline.joblib')
    mapping = {'feature_to_group': FEATURE_TO_GROUP, 'groups': FEATURE_GROUPS, 'assignment_notes': ASSIGNMENT_NOTES}
    (OUT / 'group_feature_mapping.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    results, individual, performance, sensitivity = {}, [], [], []
    for name, model in {**models, 'Dummy': dummy}.items():
        assert list(model.classes_) == [0, 1]
        p = model.predict_proba(test)[:, 1]
        row = {'model': name, **metrics(ytest, p)}
        for metric in ['roc_auc', 'pr_auc_ap']:
            vals = [metrics(np.asarray(ytest)[b], p[b])[metric] for b in stratified_bootstrap(ytest)]
            row[metric + '_ci_low'], row[metric + '_ci_high'] = np.quantile(vals, [.025, .975])
        performance.append(row)
        if name == 'Dummy':
            continue
        print(name + ': grouped permutation', flush=True)
        result, raw = permutation_analysis(model, test, ytest, FEATURE_GROUPS, bootstrap=True)
        results[name] = result
        suffix = 'logistic' if name.startswith('Logistic') else 'random_forest'
        save_csv(result, 'group_importance_' + suffix)
        save_csv(raw, 'group_permutation_repeats_' + suffix)
        print(name + ': individual permutation', flush=True)
        indiv, _ = permutation_analysis(model, test, ytest, {c: [c] for c in X})
        individual.append(indiv.assign(model=name))
        alternative = {'Working Conditions + Distance': FEATURE_GROUPS['Working Conditions'] + ['DistanceFromHome'],
                       'Employee Profile - Distance': [c for c in FEATURE_GROUPS['Employee Profile'] if c != 'DistanceFromHome']}
        alt, _ = permutation_analysis(model, test, ytest, alternative)
        sensitivity.append(alt.assign(model=name))
    performance = pd.DataFrame(performance)
    individual = pd.concat(individual, ignore_index=True)
    save_csv(performance, 'model_test_performance')
    save_csv(individual, 'individual_feature_importance')
    save_csv(pd.concat(sensitivity), 'assignment_sensitivity')
    cv, ablation = cv_comparisons(models, train, yt, folds)
    # Dummy evaluated with the exact same folds, refitted per fold.
    dummy_cv = []
    for fold, (fit, valid) in enumerate(folds):
        fitted = clone(dummy).fit(train.iloc[fit], yt.iloc[fit])
        dummy_cv.append({'fold': fold, **metrics(yt.iloc[valid], fitted.predict_proba(train.iloc[valid])[:, 1])})
    save_csv(pd.DataFrame(dummy_cv), 'dummy_cv_results')
    _, gender = gender_audit(train, yt)
    pairs = correlations(train)
    odds = coefficients(models['Logistic Regression'])
    shap_status = 'Non calculé : bibliothèque SHAP absente ; importances individuelles et groupées par permutation disponibles pour les deux modèles.'
    try:
        import shap
    except ImportError:
        pass
    else:
        try:
            rf = models['Random Forest']
            transformed = rf.named_steps['preparation'].transform(test)
            explainer = shap.TreeExplainer(rf.named_steps['modele'])
            values = explainer.shap_values(transformed)
            values = values[1] if isinstance(values, list) else values[:, :, 1] if values.ndim == 3 else values
            shap.summary_plot(values, transformed, feature_names=rf.named_steps['preparation'].get_feature_names_out(), show=False)
            plt.savefig(OUT / 'figures/shap_summary.png', dpi=300, bbox_inches='tight')
            plt.close('all')
            shap_status = 'Résumé TreeSHAP calculé pour le test Random Forest ; colonnes encodées, associations non causales.'
        except Exception as error:
            shap_status = f'Incompatible avec cet environnement : {type(error).__name__}: {error}'
    figures(results, individual, cv)
    report(performance, results, individual, ablation, cv, gender, pairs, odds, shap_status)
    metadata = {'seed': SEED, 'permutations': REPEATS, 'bootstrap_samples': 500,
                'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'numpy': np.__version__, 'pandas': pd.__version__, 'shap': shap_status,
                'source_and_model_sha256': fingerprints,
                'parameters': {n: m.named_steps['modele'].get_params() for n, m in models.items()},
                'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT / 'analysis_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for name, fingerprint in fingerprints.items():
        assert sha256((ROOT / name).read_bytes()).hexdigest() == fingerprint, name
    print(performance.to_string(index=False), flush=True)
    print(f'Completed: {OUT}', flush=True)


if __name__ == '__main__':
    main()
