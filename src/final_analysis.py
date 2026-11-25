"""Four-fold attrition validation and descriptive source tables for the final deck."""
import json

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, confusion_matrix, fbeta_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict

from .data import ROOT
from .group_feature_analysis import FEATURE_TO_GROUP
from .modeling import build_pipelines, load_model_data, make_split, select_models

TABLES = ROOT / 'reports/tables/final'
NOISE = 'Random noise'
SEEDS = (42, 137, 271, 509, 877)


def metrics(y, p, threshold):
    predicted = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return dict(threshold=float(threshold), precision=precision_score(y, predicted, zero_division=0),
                recall=recall_score(y, predicted), f2=fbeta_score(y, predicted, beta=2),
                ap=average_precision_score(y, p), auc=roc_auc_score(y, p),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def operating_point(estimator, X, y, folds):
    p = cross_val_predict(estimator, X, y, cv=folds, method='predict_proba')[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, p, drop_intermediate=False)
    denominator = 4 * precision[:-1] + recall[:-1]
    f2 = np.divide(5 * precision[:-1] * recall[:-1], denominator,
                   out=np.zeros_like(thresholds), where=denominator > 0)
    threshold = thresholds[np.flatnonzero(np.isclose(f2, f2.max(), rtol=0, atol=1e-12))[-1]]
    return metrics(y, p, threshold), p


def noise_study(name, params, X, y, folds, quantitative):
    rows = []
    for seed in SEEDS:
        trial = X.assign(**{NOISE: np.random.default_rng(seed).normal(size=len(X))})
        pipeline = build_pipelines([*quantitative, NOISE])[name].set_params(**params)
        for fold, (fit, valid) in enumerate(folds):
            model = clone(pipeline).fit(trial.iloc[fit], y.iloc[fit])
            result = permutation_importance(model, trial.iloc[valid], y.iloc[valid],
                                            scoring='roc_auc', n_repeats=20,
                                            random_state=seed + fold)
            rows.extend(dict(seed=seed, fold=fold, variable=column, decrease=value)
                        for column, value in zip(trial.columns, result.importances_mean))
    return pd.DataFrame(rows)


def run():
    TABLES.mkdir(parents=True, exist_ok=True)
    X, y, source, _, quantitative = load_model_data()
    train, test, yt, ytest, folds, split_rows = make_split(X, y, source, n_splits=4)
    historical = pd.read_csv(ROOT / 'outputs/models/04_indices_split_plis.csv')
    columns = ['position_dataset', 'ligne_excel_source', 'partie', 'ordre_dans_partie']
    pd.testing.assert_frame_equal(split_rows[columns], historical[columns])
    split_rows.to_csv(TABLES / 'split.csv', index=False)

    searches, _ = select_models(train, yt, folds, quantitative)
    candidates = []
    validation = []
    for name, search in searches.items():
        scores, probabilities = operating_point(search.best_estimator_, train, yt, folds)
        candidates.append((name, search, scores))
        validation.extend(dict(model=name, variant='full', rule=rule, **row) for rule, row in [
            ('Max OOF F2', scores), ('0.50', metrics(yt, probabilities, .5))])
    name, full, full_scores = max(candidates, key=lambda c: (c[2]['f2'], c[2]['ap']))

    importance = noise_study(name, full.best_params_, train, yt, folds, quantitative)
    importance.to_csv(TABLES / 'noise_by_fold.csv', index=False)
    by_seed = importance.groupby(['variable', 'seed']).decrease.mean().unstack()
    excluded = sorted(by_seed.index[by_seed.lt(by_seed.loc[NOISE], axis=1).all(axis=1)])
    selected, scores, variant = full, full_scores, 'full'
    if excluded:
        alternatives, _ = select_models(train, yt, folds, quantitative, [name], excluded=excluded)
        pruned = alternatives[name]
        reduced_scores, probabilities = operating_point(pruned.best_estimator_, train, yt, folds)
        validation.extend(dict(model=name, variant='pruned', rule=rule, **row) for rule, row in [
            ('Max OOF F2', reduced_scores), ('0.50', metrics(yt, probabilities, .5))])
        if reduced_scores['f2'] > full_scores['f2'] and reduced_scores['ap'] >= full_scores['ap']:
            selected, scores, variant = pruned, reduced_scores, 'pruned'
    pd.DataFrame(validation).to_csv(TABLES / 'validation.csv', index=False)

    probabilities = selected.best_estimator_.predict_proba(test)[:, 1]
    heldout = [dict(rule=rule, **metrics(ytest, probabilities, threshold))
               for rule, threshold in [('Max OOF F2', scores['threshold']), ('0.50', .5)]]
    pd.DataFrame(heldout).to_csv(TABLES / 'test.csv', index=False)

    rows = []
    for fold, (fit, valid) in enumerate(folds):
        model = clone(selected.best_estimator_).fit(train.iloc[fit], yt.iloc[fit])
        included = [col for col in train if variant == 'full' or col not in excluded]
        result = permutation_importance(model, train.iloc[valid], yt.iloc[valid],
                                        scoring='roc_auc', n_repeats=30, random_state=42 + fold)
        rows.extend(dict(variable=col, fold=fold, decrease=value)
                    for col, value in zip(train.columns, result.importances_mean) if col in included)
    ranks = pd.DataFrame(rows).groupby('variable').decrease.mean().sort_values(ascending=False)
    ranks.rename('decrease').reset_index().assign(group=lambda d: d.variable.map(FEATURE_TO_GROUP)
                                                 ).to_csv(TABLES / 'importance.csv', index=False)
    (TABLES / 'model.json').write_text(json.dumps(dict(model=name, variant=variant,
        candidate_exclusions=excluded, parameters=selected.best_params_, noise_seeds=SEEDS),
        indent=2) + '\n')
    return pd.DataFrame(heldout)


if __name__ == '__main__':
    print(run().to_string(index=False))
