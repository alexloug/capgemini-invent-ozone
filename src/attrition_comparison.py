"""Protocole commun aux sept expériences d'attrition rétrospectives."""
from hashlib import sha256
from importlib.metadata import version
import json
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    average_precision_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict

from .modeling import DATA_PATH, MODEL_DIR, load_model_data, make_split, verify_reference_split

SEED = 42


def load_comparison():
    X, y, rows, groups, quantitative = load_model_data()
    train, test, y_train, y_test, folds, split = make_split(X, y, rows)
    verify_reference_split(split)
    metadata = json.loads((MODEL_DIR / '05_metadonnees.json').read_text())
    assert metadata['sha256_source'] == sha256(DATA_PATH.read_bytes()).hexdigest()
    pd.testing.assert_frame_equal(
        split, pd.read_csv(MODEL_DIR / '05_indices_split_plis.csv', dtype={'pli_validation': 'Int64'})
    )
    # Contrôler les doublons entre dossiers, y compris après exclusion des champs.
    raw = pd.read_excel(DATA_PATH)
    assert not raw.duplicated().any()
    assert not X.duplicated().any()
    return train, test, y_train, y_test, folds, groups, quantitative


def versions(extra=None):
    packages = ['pandas', 'numpy', 'scikit-learn', 'matplotlib', 'openpyxl', 'nbclient']
    return {'Python': platform.python_version(), **{name: version(name) for name in
            packages + ([extra] if extra else [])}}


def plot_counts(y_train, y_test, folds):
    partitions = {'Entraînement': y_train, 'Test historique': y_test}
    partitions.update({f'Validation {i + 1}': y_train.iloc[valid]
                       for i, (_, valid) in enumerate(folds)})
    counts = pd.DataFrame({name: labels.value_counts() for name, labels in partitions.items()}).T
    counts = counts[[0, 1]].rename(columns={0: 'Non-départs', 1: 'Départs'})
    ax = counts.plot.barh(stacked=True, figsize=(8, 3.5), color=['#426788', '#dd833b'])
    for bars in ax.containers:
        ax.bar_label(bars, label_type='center', color='white')
    ax.set(xlabel='Nombre de dossiers', title='Ozone — classes par ensemble')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.show()


def fit_search(pipeline, configs, train, y_train, folds):
    # Chaque dictionnaire représente une configuration fixée, dans l'ordre de départage.
    grid = [{f'modele__{key}': [value] for key, value in config.items()} for config in configs]
    return GridSearchCV(pipeline, grid, scoring={'AP': 'average_precision', 'AUC': 'roc_auc'},
                        refit=False, cv=folds, n_jobs=1, error_score='raise').fit(train, y_train)


def validation_table(search):
    cv = search.cv_results_
    table = pd.DataFrame({'AP CV': cv['mean_test_AP'], 'Écart-type AP': cv['std_test_AP'],
                          'ROC-AUC CV': cv['mean_test_AUC'], 'Secondes / pli': cv['mean_fit_time'],
                          'Paramètres': [{key.split('__', 1)[1]: value for key, value in p.items()}
                                         for p in cv['params']]})
    return table.sort_values('AP CV', ascending=False, kind='stable')


def plot_search(search, limit=10):
    table = validation_table(search).head(limit)
    fig, ax = plt.subplots(figsize=(8, max(3, len(table) * .36)))
    ax.errorbar(table['AP CV'], np.arange(len(table)), xerr=table['Écart-type AP'],
                fmt='o', capsize=4, color='#426788')
    ax.set(yticks=np.arange(len(table)), yticklabels=[f'Configuration {i}' for i in table.index],
           xlabel='Average precision moyenne ± 1 écart-type entre plis',
           title='Validation croisée sur l’entraînement Ozone')
    ax.invert_yaxis()
    fig.tight_layout()
    plt.show()


def continuous_scores(model, X, method):
    scores = getattr(model, method)(X)
    return scores[:, 1] if method == 'predict_proba' else scores


def select_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    f1 = 2 * precision * recall / np.maximum(precision + recall, np.finfo(float).eps)
    # En cas d'égalité du F1, retenir le seuil le plus haut.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def rank_scores(scores, keys):
    return np.lexsort((np.asarray(keys), -np.asarray(scores)))


def evaluate_config(model, train, y_train, folds, test, y_test, method):
    oof = cross_val_predict(model, train, y_train, cv=folds, method=method, n_jobs=1)
    threshold = select_threshold(y_train, oof[:, 1] if method == 'predict_proba' else oof)
    model.fit(train, y_train)
    scores = continuous_scores(model, test, method)
    return summarize_scores(y_test, scores, threshold), scores


def summarize_scores(y, scores, threshold):
    binary = scores >= threshold
    _, _, _, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel()
    volume = int(np.ceil(.10 * len(y)))
    order = rank_scores(scores, y.index)
    found = int(y.iloc[order[:volume]].sum())
    return {'AP test': average_precision_score(y, scores), 'ROC-AUC test': roc_auc_score(y, scores),
            'Seuil': threshold, 'Précision': precision_score(y, binary, zero_division=0),
            'Rappel': recall_score(y, binary), 'F1': f1_score(y, binary),
            'Alertes au seuil': int(binary.sum()), 'Départs au seuil': int(tp),
            'Alertes à 10 %': volume, 'Départs à 10 %': found,
            'Précision à 10 %': found / volume, 'Rappel à 10 %': found / int(y.sum())}


def plot_test(y, scores, tuned_threshold):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    ax_roc, ax_pr, ax_cm, ax_gains = axes.flat
    for name, values in scores.items():
        RocCurveDisplay.from_predictions(y, values, name=name, ax=ax_roc)
        PrecisionRecallDisplay.from_predictions(y, values, name=name, ax=ax_pr)
        ranked = y.iloc[rank_scores(values, y.index)].to_numpy()
        ax_gains.plot(np.arange(len(y) + 1), np.r_[0, ranked.cumsum()], label=name)
    ax_roc.plot([0, 1], [0, 1], ':', color='gray')
    ax_roc.set(title='ROC', xlabel=f'Taux de faux positifs (n = {int((1-y).sum())})',
               ylabel=f'Taux de vrais positifs (n = {int(y.sum())})')
    ax_pr.axhline(y.mean(), ls=':', color='gray', label='Prévalence')
    ax_pr.set(title='Précision–rappel', xlabel=f'Rappel (n = {int(y.sum())} départs)',
              ylabel='Départs / alertes')
    ax_pr.legend(loc='upper right')
    ConfusionMatrixDisplay.from_predictions(y, scores['Réglage'] >= tuned_threshold,
                                           display_labels=['Non-départ', 'Départ'],
                                           colorbar=False, cmap='Blues', ax=ax_cm)
    ax_cm.set(title='Réglage — au seuil de validation', xlabel='Classe prédite',
              ylabel='Classe observée')
    ax_gains.plot([0, len(y)], [0, y.sum()], ':', color='gray', label='Classement aléatoire')
    ax_gains.axvline(np.ceil(.10 * len(y)), ls='--', color='gray', label='Volume commun')
    ax_gains.set(title='Gains cumulés', xlabel=f'Dossiers signalés (sur {len(y)})',
                 ylabel=f'Départs retrouvés (sur {int(y.sum())})')
    ax_gains.legend()
    fig.suptitle(f'Test historique Ozone — {len(y)} dossiers')
    fig.tight_layout()
    plt.show()
