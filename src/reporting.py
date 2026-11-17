"""Consolider les preuves RH sans nouvelle sélection : python -m src.reporting."""
from hashlib import sha256
import json
from pathlib import Path
import platform
import warnings

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_score,
                             recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from .data import ROOT, load_data
from .modeling import (DATA_PATH, MODEL_DIR, load_model_data, make_split, verify_reference_split)
from .group_feature_analysis import stratified_bootstrap, markdown_table

REPORTS = ROOT / 'reports'
TABLES = REPORTS / 'tables'
FIGURES = REPORTS / 'figures'
GROUPS = ROOT / 'outputs/group_feature_analysis'
ARCHIVE = ROOT / 'outputs/archives/06_random_forest_improvement'
BOOTSTRAPS = 300
LABELS = {'Working Conditions': 'Conditions de travail', 'Career Development': 'Parcours professionnel',
          'Compensation': 'Rémunération', 'Employee Profile': 'Profil salarié',
          'Engagement & Satisfaction': 'Engagement et satisfaction', 'Gender Equality': 'Genre uniquement*'}
CONTRASTS = [
    ('OverTime', 'Yes', 'No', 'Heures supplémentaires : Oui / Non'),
    ('BusinessTravel', 'Travel_Frequently', 'Travel_Rarely', 'Déplacements : fréquents / rares'),
    ('EnvironmentSatisfaction', 1, 4, 'Satisfaction environnement : code 1 / 4'),
    ('JobSatisfaction', 1, 4, 'Satisfaction travail : code 1 / 4'),
    ('JobLevel', 1, 2, 'Niveau hiérarchique : code 1 / 2'),
    ('Gender', 'Female', 'Male', 'Genre : femmes / hommes'),
]


def export(frame, name):
    frame.to_csv(TABLES / f'{name}.csv', index=False, encoding='utf-8-sig')
    return frame


def wilson(departures, n):
    if n == 0:
        return np.nan, np.nan
    z = 1.959963984540054
    rate = departures/n
    center = (rate+z*z/(2*n))/(1+z*z/n)
    radius = z*np.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-radius, center+radius


def selected_log_odds(model):
    coefficients = pd.Series(model.named_steps['modele'].coef_[0],
                             index=model.named_steps['preparation'].get_feature_names_out())
    values = {}
    for variable, category, reference, label in CONTRASTS:
        if variable in ['OverTime', 'Gender']:
            value = coefficients[f'binaires__{variable}']
        else:
            value = (coefficients[f'categories_scores__{variable}_{category}'] -
                     coefficients[f'categories_scores__{variable}_{reference}'])
        values[label] = float(value)
    return values


def adjusted_score_contrast(model, X):
    yes, no = X.copy(), X.copy()
    yes['OverTime'], no['OverTime'] = 'Yes', 'No'
    return float(np.mean(model.predict_proba(yes)[:, 1]-model.predict_proba(no)[:, 1]))


def bootstrap_associations(model, X, y):
    """Réajuster encodeur/scaler et logistique L2 dans chaque bootstrap d'entraînement."""
    rng = np.random.default_rng(20261117)
    classes = [np.flatnonzero(y.to_numpy() == k) for k in (0, 1)]
    original = selected_log_odds(model)
    draws, score_differences = [], []
    for repeat in range(BOOTSTRAPS):
        indices = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        fitted = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            fitted.fit(X.iloc[indices], y.iloc[indices])
        draws.append(dict(repeat=repeat, **selected_log_odds(fitted)))
        score_differences.append(adjusted_score_contrast(fitted, X))
        if (repeat+1) % 50 == 0:
            print(f'Bootstrap de stabilité L2 : {repeat+1}/{BOOTSTRAPS}', flush=True)
    draws = pd.DataFrame(draws)
    export(draws, 'logistic_bootstrap_log_odds')
    rows = []
    for label, beta in original.items():
        rows.append({'contrast': label, 'odds_ratio': np.exp(beta),
                     'stability_low': np.exp(draws[label].quantile(.025)),
                     'stability_high': np.exp(draws[label].quantile(.975)),
                     'positive_fraction': draws[label].gt(0).mean(), 'bootstrap_n': BOOTSTRAPS})
    result = export(pd.DataFrame(rows), 'logistic_or_stability')
    contrast = export(pd.DataFrame([{'variable': 'OverTime Yes - No',
        'mean_model_score_difference': adjusted_score_contrast(model, X),
        'stability_low': np.quantile(score_differences, .025),
        'stability_high': np.quantile(score_differences, .975),
        'population_n': len(X), 'bootstrap_n': BOOTSTRAPS,
        'interpretation': 'Ecart moyen de scores standardises sur X_train fixe; non causal, probabilites non calibrees'}]), 'adjusted_score_contrast')
    return result, contrast


def subgroup_analysis(X, y, train_indices, test_indices):
    """Hypothèses RH fixes, pas de fouille de règles ni de classement sur le test."""
    frame = X.assign(depart=y, level_1=X.JobLevel.eq(1), overtime=X.OverTime.eq('Yes'))
    records = []
    for part, indices in [('ensemble', X.index), ('entrainement', train_indices), ('test', test_indices)]:
        for level in [True, False]:
            for overtime in [True, False]:
                d = frame.loc[indices]
                cell = d[d.level_1.eq(level) & d.overtime.eq(overtime)]
                n, departures = len(cell), int(cell.depart.sum())
                low, high = wilson(departures, n)
                records.append({'part': part, 'level_1': level, 'overtime': overtime,
                                'label': ('Niveau codé 1' if level else 'Autres niveaux') + (' · HS oui' if overtime else ' · HS non'),
                                'n': n, 'departures': departures, 'rate': departures/n if n else np.nan,
                                'wilson_low': low, 'wilson_high': high})
    table = pd.DataFrame(records)
    train = table[table.part.eq('entrainement')].set_index(['level_1', 'overtime'])
    admissible = train.n.ge(80) & train.departures.ge(20) & (train.n-train.departures).ge(20)
    table['training_support_ok'] = [bool(admissible.loc[(r.level_1, r.overtime)]) for r in table.itertuples()]
    # The test support flag is disclosed, not a post-test rule for selecting the largest risks.
    test = table[table.part.eq('test')].set_index(['level_1', 'overtime'])
    support = test.n.ge(20) & test.departures.ge(5) & (test.n-test.departures).ge(10)
    table['test_support_ok'] = [bool(support.loc[(r.level_1, r.overtime)]) for r in table.itertuples()]
    export(table, 'subgroup_support')
    small = pd.read_csv(ROOT / 'outputs/tables/03_conditions_travail/03_overtime_strates.csv')
    small['main_slide_eligible'] = small.effectif.ge(80) & small.departs.ge(20) & (small.effectif-small.departs).ge(20)
    export(small, 'historical_subgroup_review')
    return table


def performance_checks(models, Xtrain, ytrain, Xtest, ytest):
    probabilities, rows, verification = {}, [], []
    reference = pd.read_csv(GROUPS / 'model_test_performance.csv').set_index('model')
    bootstraps = stratified_bootstrap(ytest)
    for name, model in models.items():
        p = model.predict_proba(Xtest)[:, 1]
        probabilities[name] = p
        row = {'model': name, 'n_test': len(ytest), 'departures_test': int(ytest.sum()),
               'roc_auc': roc_auc_score(ytest, p), 'average_precision': average_precision_score(ytest, p),
               'precision_at_05': precision_score(ytest, p >= .5, zero_division=0),
               'recall_at_05': recall_score(ytest, p >= .5, zero_division=0),
               'brier_score': brier_score_loss(ytest, p)}
        for metric, scorer in [('roc_auc', roc_auc_score), ('average_precision', average_precision_score)]:
            values = [scorer(ytest.iloc[b], p[b]) for b in bootstraps]
            row[metric+'_low'], row[metric+'_high'] = np.quantile(values, [.025, .975])
        assert abs(row['roc_auc']-reference.loc[name, 'roc_auc']) < 1e-12
        assert abs(row['roc_auc_low']-reference.loc[name, 'roc_auc_ci_low']) < 1e-12
        assert abs(row['roc_auc_high']-reference.loc[name, 'roc_auc_ci_high']) < 1e-12
        assert abs(row['average_precision']-reference.loc[name, 'pr_auc_ap']) < 1e-12
        rows.append(row)
        # Reproduction is a provenance check, without changing the persisted pipelines.
        replay = clone(model).fit(Xtrain, ytrain)
        difference = np.max(np.abs(replay.predict_proba(Xtest)[:, 1]-p))
        assert difference < 1e-12
        verification.append({'model': name, 'training_n': len(ytrain),
                             'replayed_prediction_max_difference': difference,
                             'source': 'src/modeling.py; indices 04/05; saved pipeline'})
    export(pd.DataFrame(verification), 'model_replay_checks')
    delta = []
    for b in bootstraps:
        delta.append(roc_auc_score(ytest.iloc[b], probabilities['Logistic Regression'][b])-
                     roc_auc_score(ytest.iloc[b], probabilities['Random Forest'][b]))
    paired = export(pd.DataFrame([{'comparison': 'Logistique - RF 05',
        'auc_difference': rows[0]['roc_auc']-rows[1]['roc_auc'],
        'paired_bootstrap_low': np.quantile(delta, .025),
        'paired_bootstrap_high': np.quantile(delta, .975), 'bootstrap_n': 500}]), 'paired_model_difference')
    return export(pd.DataFrame(rows), 'model_performance'), paired


def group_consolidation():
    frames = []
    for name, suffix in [('Logistique', 'logistic'), ('Random Forest', 'random_forest')]:
        d = pd.read_csv(GROUPS / f'group_importance_{suffix}.csv')
        frames.append(d.assign(model=name))
    result = export(pd.concat(frames, ignore_index=True), 'group_permutation_importance')
    ablation = pd.read_csv(GROUPS / 'group_ablation_results.csv')
    summary = ablation.groupby(['model', 'group']).agg(
        delta_auc_mean=('delta_roc_auc', 'mean'), delta_auc_sd=('delta_roc_auc', 'std'),
        delta_ap_mean=('delta_pr_auc_ap', 'mean'), positive_auc_folds=('delta_roc_auc', lambda s: int(s.gt(0).sum()))).reset_index()
    export(summary, 'group_ablation_summary')
    export(pd.read_csv(GROUPS / 'gender_standardized_outcomes.csv'), 'gender_standardization')
    export(pd.read_csv(GROUPS / 'individual_feature_importance.csv'), 'individual_permutation_importance')
    return result, summary


def draw_figures(performance, odds, groups, subgroups):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titlesize': 20, 'axes.labelsize': 14,
                         'svg.fonttype': 'none'})
    def finish(fig, stem, source, right=.94):
        fig.text(.065, .035, source, fontsize=10, color='#536572', va='bottom')
        fig.subplots_adjust(left=.32, right=right, top=.78, bottom=.19)
        for extension in ['png', 'svg']:
            fig.savefig(FIGURES / (stem+'.'+extension), dpi=300, facecolor='white')
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Les modèles détectent un signal de départ, sans garantir la prédiction individuelle', x=.065, ha='left', y=.94, fontsize=21)
    fig.text(.065, .87, 'Test historique : 304 salariés, dont 50 départs • IC conditionnels à 95 %', fontsize=15)
    names = ['Logistique', 'Random Forest', 'Référence constante']
    for i, row in enumerate(performance.itertuples()):
        ax.errorbar(row.roc_auc, i, xerr=[[row.roc_auc-row.roc_auc_low], [row.roc_auc_high-row.roc_auc]],
                    fmt='o', markersize=12, capsize=6, color=['#0070AD', '#12ABDB', '#888888'][i])
        ax.text(1.03, i, f'AUC {row.roc_auc:.3f}  |  AP {row.average_precision:.3f}',
                transform=ax.get_yaxis_transform(), va='center', fontsize=13)
    ax.set(yticks=range(3), yticklabels=names, xlim=(.45, 1), ylim=(2.7, -.6), xlabel='ROC-AUC — qualité de classement des départs')
    ax.axvline(.5, color='gray', linestyle='--'); ax.grid(axis='x', alpha=.2)
    finish(fig, '01_performance', 'Source : reports/tables/model_performance.csv • 500 bootstraps stratifiés du test ; modèles fixés\nLa ROC-AUC ne mesure ni la calibration des probabilités ni l’effet d’une action RH.', right=.74)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Les heures supplémentaires restent associées aux départs après ajustement', x=.065, ha='left', y=.94, fontsize=22)
    fig.text(.065, .87, 'Logistique L2 • 1 214 salariés d’entraînement, 198 départs • contrastes explicités', fontsize=15)
    for i, row in enumerate(odds.itertuples()):
        ax.hlines(i, row.stability_low, row.stability_high, color='#0070AD', linewidth=3)
        ax.plot(row.odds_ratio, i, 'o', color='#0070AD', markersize=9)
        ax.text(1.02, i, f'{row.odds_ratio:.2f} [{row.stability_low:.2f} ; {row.stability_high:.2f}]',
                transform=ax.get_yaxis_transform(), va='center', fontsize=12)
    ax.set(yticks=range(len(odds)), yticklabels=odds.contrast, xscale='log',
           xlabel='Odds ratio ajusté pénalisé (échelle logarithmique)', ylim=(len(odds)-.4, -.6))
    ax.set_xlim(.5, 5)
    ax.xaxis.set_major_locator(FixedLocator([.5, 1, 2, 3, 4, 5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f'{value:g}'))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axvline(1, color='#888888', linestyle='--'); ax.grid(axis='x', alpha=.2)
    finish(fig, '02_odds_ratios', 'Source : reports/tables/logistic_or_stability.csv • 300 réajustements bootstrap sur entraînement\nBarres = intervalles de stabilité à 95 %, pas des IC MLE classiques ; pénalité fixée ; aucune interprétation causale.', right=.78)
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Les conditions de travail portent le premier signal prédictif dans les deux modèles', x=.065, ha='left', y=.94, fontsize=21)
    fig.text(.065, .87, 'Permutation conjointe des variables originales • baisse de ROC-AUC sur le test', fontsize=15)
    order = ['Working Conditions', 'Career Development', 'Employee Profile', 'Compensation', 'Engagement & Satisfaction', 'Gender Equality']
    for name, offset, color in [('Logistique', -.16, '#0070AD'), ('Random Forest', .16, '#12ABDB')]:
        d = groups[groups.model.eq(name)].set_index('group').loc[order]
        pos = np.arange(len(order))+offset
        ax.barh(pos, d.importance_mean, height=.26, color=color, label=name)
        ax.hlines(pos, d.bootstrap_ci_low, d.bootstrap_ci_high, color='#243746', linewidth=1.7)
    ax.set(yticks=range(6), yticklabels=[LABELS[g] for g in order], ylim=(5.6, -.6), xlabel='AUC initiale − AUC après permutation du groupe')
    ax.axvline(0, color='gray', linewidth=1); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=.2)
    finish(fig, '03_groupes_rh', 'Source : reports/tables/group_permutation_importance.csv • 30 permutations ; 500 bootstraps conditionnels\n* Genre uniquement ≠ mesure de l’égalité. Groupes de tailles différentes et corrélés ; importance prédictive ≠ effet causal.')
    full = subgroups[subgroups.part.eq('ensemble') & subgroups.training_support_ok].copy()
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.suptitle('Le cumul heures supplémentaires–niveau codé 1 mérite un diagnostic ciblé', x=.065, ha='left', y=.94, fontsize=22)
    fig.text(.065, .87, 'Proportions observées dans le fichier • segmentation exploratoire, sans taux annuel', fontsize=15)
    for i, row in enumerate(full.itertuples()):
        ax.errorbar(100*row.rate, i, xerr=[[100*(row.rate-row.wilson_low)], [100*(row.wilson_high-row.rate)]],
                    fmt='o', markersize=11, capsize=6, color='#0070AD' if row.overtime else '#7B8E9B')
        ax.text(1.02, i, f'{row.departures}/{row.n} • {100*row.rate:.1f} %', transform=ax.get_yaxis_transform(), va='center', fontsize=13)
    ax.set(yticks=range(len(full)), yticklabels=full.label, xlabel='Départs observés / salariés du groupe (%)',
           xlim=(0, 65), ylim=(len(full)-.4, -.6))
    ax.axvline(248/1518*100, color='gray', linestyle='--', label='Ensemble : 16,3 %'); ax.legend(loc='lower right'); ax.grid(axis='x', alpha=.2)
    finish(fig, '04_sous_groupes', 'Source : reports/tables/subgroup_support.csv • IC de Wilson 95 % ; effectifs et départs affichés\nNiveau 1 = code observé, pas une définition validée de « junior ». Sélection par support d’entraînement ; limites train/test dans le rapport.', right=.79)


def main():
    TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    protected = list((ROOT/'data').rglob('*.xlsx')) + list(MODEL_DIR.glob('*')) + list(ARCHIVE.glob('*.joblib'))
    before = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in protected if p.is_file()}
    X, y, rows, _, _ = load_model_data()
    raw = load_data()
    pd.testing.assert_frame_equal(raw[X.columns], X)
    assert raw.depart.eq(y).all() and (len(X), int(y.sum())) == (1518, 248)
    Xtrain, Xtest, ytrain, ytest, folds, split = make_split(X, y, rows)
    verify_reference_split(split)
    pd.testing.assert_frame_equal(pd.read_csv(MODEL_DIR/'04_indices_split_plis.csv'), pd.read_csv(MODEL_DIR/'05_indices_split_plis.csv'))
    assert not set(Xtrain.index).intersection(Xtest.index)
    models = {'Logistic Regression': joblib.load(MODEL_DIR/'04_Logistique.joblib'),
              'Random Forest': joblib.load(MODEL_DIR/'05_RandomForest.joblib'),
              'Dummy': joblib.load(MODEL_DIR/'04_Baseline.joblib')}
    with threadpool_limits(limits=1):
        performance, paired = performance_checks(models, Xtrain, ytrain, Xtest, ytest)
        odds, adjusted = bootstrap_associations(models['Logistic Regression'], Xtrain, ytrain)
    subgroups = subgroup_analysis(X, y, Xtrain.index, Xtest.index)
    group, ablation = group_consolidation()
    descriptive = []
    for variable in ['OverTime', 'BusinessTravel', 'Gender']:
        source = ROOT/('outputs/tables/02_egalite_femmes_hommes/02_depart_genre.csv' if variable == 'Gender' else f'outputs/tables/03_conditions_travail/03_{variable}.csv')
        stored = pd.read_csv(source).set_index(variable)
        for category, d in raw.groupby(variable):
            n, departures = len(d), int(d.depart.sum())
            assert (n, departures) == (stored.loc[category, 'effectif'], stored.loc[category, 'departs'])
            low, high = wilson(departures, n)
            descriptive.append({'variable': variable, 'category': category, 'n': n,
                                'departures': departures, 'rate': departures/n, 'wilson_low': low, 'wilson_high': high})
    descriptive = export(pd.DataFrame(descriptive), 'descriptive_associations')
    draw_figures(performance, odds, group, subgroups)
    from .report_text import write_reports
    write_reports(performance, paired, odds, adjusted, group, ablation, subgroups, descriptive)
    for path, fingerprint in before.items():
        assert sha256((ROOT/path).read_bytes()).hexdigest() == fingerprint, path
    metadata = {'date': '2026-11-17', 'python': platform.python_version(), 'bootstrap_or_n': BOOTSTRAPS,
                'bootstrap_performance_n': 500, 'protected_sha256': before,
                'scripts': {name: sha256((ROOT/name).read_bytes()).hexdigest() for name in ['src/reporting.py','src/report_text.py']},
                'new_selection': False, 'bootstrap_penalty': 'fixed existing L2 C=0.1',
                'subgroup_training_rule': 'n>=80; departures>=20; nondepartures>=20',
                'subgroup_test_support_rule': 'n>=20; departures>=5; nondepartures>=10; disclosure only'}
    (REPORTS/'consolidation_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print('Restitution générée ; modèles et données protégés inchangés.', flush=True)


if __name__ == '__main__':
    main()
