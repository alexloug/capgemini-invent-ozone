"""Exploration Random Forest du notebook 06, sans modification des expériences 04/05."""
from hashlib import sha256
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

from .data import ROOT
from .group_feature_analysis import FEATURE_GROUPS, subset_pipeline

OUT = ROOT / 'outputs/archives/06_random_forest_improvement'
SEED = 42
SUBSETS = {
    'Toutes': [],
    'Sans remuneration': FEATURE_GROUPS['Compensation'],
    'Sans profil': FEATURE_GROUPS['Employee Profile'],
    'Sans remuneration ni profil': FEATURE_GROUPS['Compensation'] + FEATURE_GROUPS['Employee Profile'],
}


def export(frame, name):
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / (name + '.csv'), index=False, encoding='utf-8-sig')
    return frame


def candidates(reference):
    """20 configurations préspecifiées : 8 contrôles et 12 tirages reproductibles."""
    base = {k: reference.named_steps['modele'].get_params()[k] for k in
            ['n_estimators', 'max_depth', 'min_samples_leaf', 'max_features', 'class_weight', 'max_samples']}
    configurations = [dict(subset=subset, score_encoding=encoding, **base)
                      for subset in SUBSETS for encoding in ['numeric', 'onehot']]
    space = {'subset': list(SUBSETS), 'score_encoding': ['numeric', 'onehot'],
             'n_estimators': [200, 600], 'max_depth': [None, 8, 16],
             'min_samples_leaf': [1, 2, 4, 8], 'max_features': ['sqrt', .5, 1.],
             'class_weight': [None, 'balanced', 'balanced_subsample'], 'max_samples': [None, .8]}
    configurations += list(ParameterSampler(space, n_iter=12, random_state=SEED))
    return [dict(candidate=i, **config) for i, config in enumerate(configurations)]


def make_model(reference, config, columns):
    selected = [c for c in columns if c not in SUBSETS[config['subset']]]
    pipe = subset_pipeline(reference, selected)
    if config['score_encoding'] == 'onehot':
        pre = pipe.named_steps['preparation']
        pre.transformers = [(name, OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                             if name == 'scores' else trans, cols)
                            for name, trans, cols in pre.transformers]
    params = {f'modele__{k}': v for k, v in config.items()
              if k not in ['candidate', 'subset', 'score_encoding']}
    pipe.set_params(**params, modele__n_jobs=1, modele__random_state=SEED)
    return pipe, selected


def scores(y, p, threshold=.5):
    predicted = np.asarray(p) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {'roc_auc': roc_auc_score(y, p), 'average_precision': average_precision_score(y, p),
            'precision': precision_score(y, predicted, zero_division=0),
            'recall': recall_score(y, predicted, zero_division=0),
            'f2': fbeta_score(y, predicted, beta=2, zero_division=0), 'threshold': threshold,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'flagged_fraction': float(predicted.mean())}


def choose_threshold(y, p):
    """F2 privilégie le rappel ; ex æquo départagés par le seuil le plus élevé."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    f2 = 5*precision[:-1]*recall[:-1] / np.maximum(4*precision[:-1]+recall[:-1], 1e-15)
    best = np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]
    return float(thresholds[best])


def search(reference, configurations, X, y, folds, label):
    """Chaque pipeline est ajusté uniquement sur le sous-ensemble de fit du pli."""
    rows, predictions = [], {}
    for config in configurations:
        oof = np.full(len(y), np.nan)
        fold_ap, fold_auc = [], []
        for fit, valid in folds:
            assert set(X.index[fit]).isdisjoint(X.index[valid])
            pipe, cols = make_model(reference, config, X.columns)
            pipe.fit(X.iloc[fit][cols], y.iloc[fit])
            oof[valid] = pipe.predict_proba(X.iloc[valid][cols])[:, 1]
            fold_ap.append(average_precision_score(y.iloc[valid], oof[valid]))
            fold_auc.append(roc_auc_score(y.iloc[valid], oof[valid]))
        assert np.isfinite(oof).all()
        rows.append(dict(config, mean_ap=np.mean(fold_ap), sd_ap=np.std(fold_ap, ddof=1),
                         mean_auc=np.mean(fold_auc), pooled_oof_ap=average_precision_score(y, oof)))
        predictions[config['candidate']] = oof
        print(f'{label} — configuration {config["candidate"]+1}/{len(configurations)} : AP {np.mean(fold_ap):.3f}', flush=True)
    table = pd.DataFrame(rows).sort_values(['mean_ap', 'candidate'], ascending=[False, True])
    winner = next(c for c in configurations if c['candidate'] == int(table.iloc[0].candidate))
    return winner, table, predictions


def nested_experiment(reference, configurations, X, y, outer_folds):
    rows, selections, searches = [], [], []
    # These arrays remain internal, without exporting employee-level risk scores.
    probabilities = {name: np.full(len(y), np.nan) for name in ['Reference', 'Optimise']}
    decisions = {name: np.zeros(len(y), dtype=bool) for name in ['Reference', 'Optimise']}
    fold_ids = np.full(len(y), -1)
    for outer, (fit, valid) in enumerate(outer_folds):
        inner_X, inner_y = X.iloc[fit], y.iloc[fit]
        inner = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED).split(inner_X, inner_y))
        best, table, oof = search(reference, configurations, inner_X, inner_y, inner, f'Externe {outer+1}/5')
        searches.append(table.assign(outer_fold=outer))
        selections.append(dict(best, outer_fold=outer))
        fold_ids[valid] = outer
        for name, config in [('Reference', configurations[0]), ('Optimise', best)]:
            threshold = choose_threshold(inner_y, oof[config['candidate']])
            model, cols = make_model(reference, config, X.columns)
            model.fit(inner_X[cols], inner_y)
            p = model.predict_proba(X.iloc[valid][cols])[:, 1]
            probabilities[name][valid] = p
            decisions[name][valid] = p >= threshold
            for policy, cut in [('0.5', .5), ('F2_interne', threshold)]:
                rows.append(dict(model=name, policy=policy, outer_fold=outer,
                                 candidate=config['candidate'], **scores(y.iloc[valid], p, cut)))
        print(f'Pli externe {outer+1}/5 terminé ; configuration retenue {best["candidate"]}', flush=True)
    assert (fold_ids >= 0).all()
    detail = export(pd.DataFrame(rows), 'nested_fold_scores')
    export(pd.DataFrame(selections), 'nested_selected_configurations')
    export(pd.concat(searches, ignore_index=True), 'nested_inner_searches')
    aggregate = []
    for name, p in probabilities.items():
        assert np.isfinite(p).all()
        aggregate.append(dict(model=name, policy='0.5', **scores(y, p)))
        chosen = decisions[name]
        row = dict(model=name, policy='F2_interne', **scores(y, p))
        row.update(precision=precision_score(y, chosen, zero_division=0),
                   recall=recall_score(y, chosen), f2=fbeta_score(y, chosen, beta=2),
                   threshold=np.nan, flagged_fraction=chosen.mean())
        row.update(dict(zip(['tn', 'fp', 'fn', 'tp'], map(int, confusion_matrix(y, chosen).ravel()))))
        aggregate.append(row)
    aggregate = export(pd.DataFrame(aggregate), 'nested_pooled_scores')
    # Conditional paired bootstrap: not an independence claim about overlapping training folds.
    rng = np.random.default_rng(SEED)
    classes = [np.flatnonzero(y.to_numpy() == k) for k in [0, 1]]
    bootstrap = []
    for _ in range(500):
        b = np.concatenate([rng.choice(c, len(c), replace=True) for c in classes])
        yb = y.iloc[b]
        bootstrap.append({metric: func(yb, probabilities['Optimise'][b])-func(yb, probabilities['Reference'][b])
                          for metric, func in [('delta_ap', average_precision_score), ('delta_auc', roc_auc_score)]})
    boot = pd.DataFrame(bootstrap)
    intervals = export(pd.DataFrame([{'metric': c, 'bootstrap_low': boot[c].quantile(.025),
                                     'bootstrap_high': boot[c].quantile(.975)} for c in boot]), 'nested_paired_intervals')
    return detail, aggregate, intervals


def final_training(reference, configurations, X, y, folds):
    best, table, oof = search(reference, configurations, X, y, folds, 'Selection finale entrainement')
    export(table, 'training_search')
    model, columns = make_model(reference, best, X.columns)
    threshold = choose_threshold(y, oof[best['candidate']])
    reference_threshold = choose_threshold(y, oof[0])
    model.fit(X[columns], y)
    joblib.dump(model, OUT / '06_RandomForest.joblib', compress=3)
    selected = dict(best, selected_columns=columns, f2_threshold=threshold,
                    reference_f2_threshold=reference_threshold,
                    selection='mean inner-fold average precision; threshold maximizes OOF F2')
    (OUT / 'selected_configuration.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    thresholds = []
    for name, probabilities in [('Reference', oof[0]), ('Optimise', oof[best['candidate']])]:
        for cut in np.linspace(.05, .8, 76):
            thresholds.append(dict(model=name, **scores(y, probabilities, float(cut))))
    export(pd.DataFrame(thresholds), 'training_threshold_curves')
    return model, selected, table


def evaluate_final(model, selected, reference, logistic, dummy, X, y):
    rows, predictions = [], {}
    for name, estimator, cols, cut in [
        ('RF 05', reference, list(X), selected['reference_f2_threshold']),
        ('RF 06', model, selected['selected_columns'], selected['f2_threshold']),
        ('Logistique 04', logistic, list(X), None), ('Dummy', dummy, list(X), None)]:
        p = estimator.predict_proba(X[cols])[:, 1]
        predictions[name] = p
        rows.append(dict(model=name, policy='0.5', **scores(y, p)))
        if cut is not None:
            rows.append(dict(model=name, policy='F2_entrainement', **scores(y, p, cut)))
    return export(pd.DataFrame(rows), 'test_comparison'), predictions


def plot_results(detail, test_scores, predictions, ytest):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import PrecisionRecallDisplay
    sns.set_theme(style='whitegrid', palette=['#0070AD', '#12ABDB'])
    figures = OUT / 'figures'
    figures.mkdir(exist_ok=True)
    def finish(fig, filename):
        # File-backed display also works when a shared analysis module selects Agg.
        from IPython import get_ipython
        from IPython.display import Image, display
        fig.tight_layout()
        path = figures / filename
        fig.savefig(path, dpi=300)
        plt.close(fig)
        if get_ipython() is not None:
            display(Image(filename=str(path)))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    source = detail[detail.policy.eq('0.5')]
    fold_colors = ['#0070AD', '#12ABDB', '#7B61A8', '#2A8C6E', '#CC7A00']
    for ax, metric in zip(axes, ['average_precision', 'roc_auc']):
        pivot = source.pivot(index='outer_fold', columns='model', values=metric)
        for fold, row in pivot.iterrows():
            ax.plot([0, 1], row[['Reference', 'Optimise']], marker='o', alpha=.85,
                    color=fold_colors[int(fold)], label=f'Pli {fold+1}')
        ax.set(xticks=[0, 1], xticklabels=['RF référence', 'RF optimisé'], ylabel=metric,
               title='Validation externe : résultats appariés')
    axes[1].legend(title='Pli externe')
    finish(fig, '06_nested_comparison.png')
    fig, ax = plt.subplots(figsize=(8, 5))
    model_colors = {'RF 05': '#0070AD', 'RF 06': '#12ABDB',
                    'Logistique 04': '#7B61A8', 'Dummy': '#777777'}
    for name, p in predictions.items():
        PrecisionRecallDisplay.from_predictions(ytest, p, name=name, ax=ax, color=model_colors[name])
    ax.axhline(ytest.mean(), color='gray', linestyle='--', label='Prévalence test')
    ax.legend(); ax.set_title('Test existant : classement, après sélection sur entraînement')
    finish(fig, '06_test_precision_recall.png')
    curves = pd.read_csv(OUT / 'training_threshold_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, frame in curves.groupby('model'):
        color = '#12ABDB' if name == 'Optimise' else '#0070AD'
        axes[0].plot(frame.threshold, frame.recall, color=color, label=name+' : rappel')
        axes[0].plot(frame.threshold, frame.precision, color=color, linestyle='--', label=name+' : précision')
        axes[1].plot(frame.threshold, frame.f2, color=color, label=name)
    for ax in axes:
        ax.set(xlabel='Seuil', ylim=(0, 1)); ax.legend()
    axes[0].set_title('Prédictions OOF entraînement : compromis')
    axes[1].set_title('Critère F2 : rappel privilégié')
    finish(fig, '06_training_thresholds.png')


def write_summary(detail, aggregate, intervals, test, selected):
    from .group_feature_analysis import markdown_table
    nested = aggregate[aggregate.policy.eq('0.5')].set_index('model')
    delta = nested.loc['Optimise', 'average_precision']-nested.loc['Reference', 'average_precision']
    paired = detail[detail.policy.eq('0.5')].pivot(index='outer_fold', columns='model', values='average_precision')
    wins = int((paired.Optimise > paired.Reference).sum())
    base = test[(test.model.eq('RF 05')) & test.policy.eq('0.5')].iloc[0]
    new = test[(test.model.eq('RF 06')) & test.policy.eq('0.5')].iloc[0]
    threshold_row = test[(test.model.eq('RF 06')) & test.policy.eq('F2_entrainement')].iloc[0]
    ref_threshold = test[(test.model.eq('RF 05')) & test.policy.eq('F2_entrainement')].iloc[0]
    ap_interval = intervals[intervals.metric.eq('delta_ap')].iloc[0]
    robustness = ('Le gain de classement n’est pas confirmé : l’intervalle conditionnel de différence d’AP contient zéro.'
                  if ap_interval.bootstrap_low <= 0 <= ap_interval.bootstrap_high else
                  'L’intervalle conditionnel de différence d’AP exclut zéro ; sa portée reste limitée par le protocole exploratoire.')
    cols = ['model', 'policy', 'average_precision', 'roc_auc', 'precision', 'recall', 'f2', 'threshold', 'fp', 'tp', 'flagged_fraction']
    def score_table(frame):
        frame = frame[cols].copy()
        for column in ['fp', 'tp']:
            frame[column] = frame[column].map(lambda value: str(int(value)))
        frame['threshold'] = frame.threshold.map(lambda value: 'variable par pli' if pd.isna(value) else f'{value:.3f}')
        return markdown_table(frame)
    text = ['# 06 — Exploration pour améliorer le Random Forest',
            '## Résultats observés',
            f'Validation externe : variation d’AP OOF agrégée de {delta:+.3f} ; amélioration sur {wins}/5 plis. '
            'Les scores agrégés mélangent des modèles ajustés dans des plis différents ; consulter aussi les scores appariés par pli.',
            score_table(aggregate), markdown_table(intervals),
            robustness,
            f'Test historique : AP {base.average_precision:.3f} → {new.average_precision:.3f}, '
            f'ROC-AUC {base.roc_auc:.3f} → {new.roc_auc:.3f}. '
            f'Au seuil F2 choisi sur entraînement ({selected["f2_threshold"]:.3f}), '
            f'rappel {threshold_row.recall:.1%}, précision {threshold_row.precision:.1%}, '
            f'{int(threshold_row.tp)} départs détectés et {int(threshold_row.fp)} faux positifs.',
            score_table(test),
            f'La référence bénéficie elle aussi du réglage du seuil : rappel {ref_threshold.recall:.1%}, '
            f'précision {ref_threshold.precision:.1%}, F2 {ref_threshold.f2:.3f}, contre F2 {threshold_row.f2:.3f} '
            'pour le candidat. Le choix du seuil et le choix du modèle sont donc deux décisions distinctes. '
            'La logistique reste une référence concurrente à comparer sur les mêmes objectifs métier.',
            '## Configuration choisie sur entraînement', '```json\n'+json.dumps(selected, indent=2)+'\n```',
            '## Interprétation et limites',
            'La sélection optimise l’average precision, pas le score test. Le seuil maximise F2 sur des prédictions hors pli '
            'd’entraînement : le rappel reçoit davantage de poids que la précision. Ce choix est exploratoire, à remplacer '
            'par un objectif métier de capacité/coût ; il ne modifie ni l’AUC ni l’AP du modèle. '
            'Le réglage du seuil de la référence est également évalué pour distinguer l’effet du seuil de celui du modèle.',
            'Cinq plis externes identiques aux expériences précédentes ; trois plis internes par pli externe pour les 20 configurations. '
            'La configuration et le seuil sont choisis sans les labels du pli externe. La sélection finale réutilise les cinq plis '
            'sur tout l’entraînement. Les recherches incluent des retraits de groupes et un encodage one-hot des scores '
            'pour éviter de supposer leur ordre. Pas de changement des valeurs sources, pas de SMOTE hors pipeline.',
            'Les hypothèses de retrait et la référence viennent de travaux ayant déjà consulté ces données, parfois le test. '
            'La validation imbriquée contrôle la nouvelle recherche, sans effacer ce contexte exploratoire. '
            'Les IC bootstrap sur les prédictions externes sont conditionnels aux modèles et ne représentent pas toute la '
            'variabilité du réentraînement ; les plis ont des ensembles d’apprentissage qui se recouvrent. '
            'Aucun test de significativité fondé sur cinq plis supposés indépendants. '
            'Une différence faible ou un intervalle contenant zéro ne démontre pas une amélioration robuste.',
            '20 configurations forment un budget limité, pas une recherche exhaustive. Une hausse sur le seul test '
            'ne suffit pas à choisir le modèle. Aucun essai additionnel n’est décidé après son évaluation. '
            'Le test existant sert de comparaison historique, pas de validation externe indépendante. '
            'Les scores hors dictionnaire et incohérences d’ancienneté restent présents. '
            'Les associations prédictives ne mesurent pas l’effet causal d’une politique RH.',
            '## Suite proposée',
            'Confronter le gain aux résultats externes par pli et aux intervalles. Valider un seuil avec la capacité '
            'd’accompagnement RH puis évaluer prospectivement sur une nouvelle cohorte temporelle. '
            'Conserver la logistique comme référence concurrente et ne pas remplacer le RF 05 sur la seule base du test.']
    (OUT / '06_exploration_report.md').write_text('\n\n'.join(text)+'\n', encoding='utf-8')


def save_metadata(selected, fingerprints, configurations):
    for path, fingerprint in fingerprints.items():
        assert sha256((ROOT / path).read_bytes()).hexdigest() == fingerprint, path
    metadata = {'seed': SEED, 'python': platform.python_version(), 'sklearn': sklearn.__version__,
                'pandas': pd.__version__, 'numpy': np.__version__, 'external_folds': 5,
                'internal_folds': 3, 'configurations': configurations, 'selected': selected,
                'protected_file_sha256': fingerprints,
                'selection_metric': 'average_precision', 'threshold_objective': 'F2 on training OOF only'}
    (OUT / '06_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
