"""Comparaison reproductible de pipelines d'attrition, sans correction des données."""
from hashlib import sha256
import json
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score, make_scorer)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .data import ROOT, load_data

DATA_PATH = ROOT / 'data/variantes/02_sans_colonnes_valeurs_inchangees.xlsx'
MODEL_DIR = ROOT / 'outputs/models'
BINARY = {'Gender': {'Male': 0, 'Female': 1}, 'OverTime': {'No': 0, 'Yes': 1}}
NOMINAL = ['BusinessTravel', 'Department', 'EducationField', 'JobRole', 'MaritalStatus']
SCORES = ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel',
          'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction',
          'StockOptionLevel', 'WorkLifeBalance']
EXCLUDED = ['Attrition', 'depart', 'EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
SCORING = {'average_precision': 'average_precision', 'roc_auc': 'roc_auc',
           'precision': make_scorer(precision_score, zero_division=0),
           'recall': make_scorer(recall_score, zero_division=0),
           'f1': make_scorer(f1_score, zero_division=0)}


def load_model_data(path=DATA_PATH):
    """Contrôle la variante sans retirer de lignes ni corriger de valeurs.

    La lecture brute vérifie que le chargeur commun n'a pas modifié les valeurs.
    Les vides, types ou codes inattendus provoquent un diagnostic, pas une imputation.
    """
    raw = pd.read_excel(path, engine='openpyxl')
    missing = raw.isna().sum()
    if missing.any():
        raise ValueError('Valeurs manquantes détectées :\n' + missing[missing.gt(0)].to_string())
    for col in ['Attrition', *BINARY]:
        if col not in raw:
            raise ValueError(f'Colonne requise absente : {col}')
        allowed = ['No', 'Yes'] if col == 'Attrition' else list(BINARY[col])
        unexpected = raw.loc[~raw[col].isin(allowed), col].value_counts(dropna=False)
        if not unexpected.empty:
            raise ValueError(f'Modalités inattendues dans {col} :\n{unexpected}')
    loaded = load_data(path)
    # Échec explicite si le chargeur a retiré des espaces ou modifié une valeur.
    pd.testing.assert_frame_equal(loaded[raw.columns], raw)
    if len(loaded) != len(raw):
        raise AssertionError('Le nombre de lignes a changé.')
    X = loaded.drop(columns=EXCLUDED, errors='ignore').copy()
    required = list(BINARY) + NOMINAL + SCORES
    absent = set(required) - set(X)
    if absent:
        raise ValueError(f'Variables explicatives absentes : {sorted(absent)}')
    quantitative = [c for c in X if c not in required]
    non_numeric = [c for c in SCORES + quantitative if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f'Colonnes numériques non reconnues : {non_numeric} ; vérifier les valeurs.')
    if not np.isfinite(X[SCORES + quantitative].to_numpy(dtype=float)).all():
        raise ValueError('Valeurs numériques non finies : vérifier le fichier.')
    if X.isna().any().any():
        raise ValueError('Valeurs manquantes dans X : aucun remplacement automatique autorisé.')
    groups = pd.DataFrame([{'variable': c, 'groupe': group}
                          for group, cols in [('binaire', list(BINARY)), ('nominale', NOMINAL),
                                              ('score_ordonne', SCORES), ('quantitative', quantitative)]
                          for c in cols])
    assert groups.variable.is_unique and set(groups.variable) == set(X.columns)
    assert not set(EXCLUDED).intersection(X)
    # Le repère est conservé à part et ne doit jamais entrer dans X.
    source_rows = pd.Series(np.arange(len(raw)) + 2, index=X.index, name='ligne_excel_source')
    y = loaded.Attrition.map({'No': 0, 'Yes': 1}).rename('depart')
    return X, y, source_rows, groups, quantitative


def encode_binary(frame):
    """Encodage fixe, dans les pipelines ; échec explicite si un code est inconnu."""
    result = frame.copy()
    for col, mapping in BINARY.items():
        if frame[col].isna().any() or not frame[col].isin(mapping).all():
            raise ValueError(f'Valeurs binaires absentes ou inconnues dans {col}')
        result[col] = frame[col].map(mapping).astype(int)
    return result


def make_split(X, y, source_rows):
    """Une seule séparation, puis cinq plis relatifs à l'entraînement."""
    train_idx, test_idx = train_test_split(np.arange(len(X)), test_size=.2,
                                         stratify=y, random_state=42)
    assert not set(train_idx).intersection(test_idx)
    assert set(train_idx).union(test_idx) == set(range(len(X)))
    X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
    y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_train, y_train))
    fold_ids = np.full(len(X), -1, dtype=int)
    for fold, (fit, valid) in enumerate(folds):
        assert not set(fit).intersection(valid)
        assert set(fit).union(valid) == set(range(len(train_idx)))
        fold_ids[train_idx[valid]] = fold
    assert (fold_ids[train_idx] >= 0).all() and (fold_ids[test_idx] == -1).all()
    split_rows = pd.DataFrame({'position_dataset': np.arange(len(X)),
                              'ligne_excel_source': source_rows.to_numpy(),
                              'partie': np.where(fold_ids >= 0, 'entrainement', 'test'),
                              'pli_validation': pd.array(np.where(fold_ids >= 0, fold_ids, np.nan), dtype='Int64')})
    # L'ordre du split est nécessaire pour reconstituer exactement les matrices.
    split_rows['ordre_dans_partie'] = -1
    split_rows.loc[train_idx, 'ordre_dans_partie'] = np.arange(len(train_idx))
    split_rows.loc[test_idx, 'ordre_dans_partie'] = np.arange(len(test_idx))
    return X_train, X_test, y_train, y_test, folds, split_rows


def build_pipelines(quantitative):
    binary = FunctionTransformer(encode_binary, validate=False, feature_names_out='one-to-one')
    def onehot():
        return OneHotEncoder(handle_unknown='ignore', drop=None, sparse_output=False)
    logistic_preparation = ColumnTransformer([
        ('binaires', binary, list(BINARY)),
        ('categories_scores', onehot(), NOMINAL + SCORES),
        ('quantitatives', StandardScaler(), quantitative)], remainder='drop')
    forest_preparation = ColumnTransformer([
        ('binaires', binary, list(BINARY)),
        ('categories', onehot(), NOMINAL),
        ('scores', 'passthrough', SCORES),
        ('quantitatives', 'passthrough', quantitative)], remainder='drop')
    return {
        'Logistique': Pipeline([('preparation', logistic_preparation),
                               ('modele', LogisticRegression(penalty='l2', solver='lbfgs',
                                                             max_iter=5000, random_state=42))]),
        'RandomForest': Pipeline([('preparation', forest_preparation),
                                 ('modele', RandomForestClassifier(n_estimators=200,
                                                                  random_state=42, n_jobs=1))]),
        'Baseline': Pipeline([('modele', DummyClassifier(strategy='prior'))])}


def select_models(X_train, y_train, folds, quantitative):
    """Recherche exclusivement sur l'entraînement, mêmes plis pour les trois modèles."""
    grids = {
        'Logistique': {'modele__C': [.1, 1, 10], 'modele__class_weight': [None, 'balanced']},
        'RandomForest': {'modele__max_depth': [6, None], 'modele__min_samples_leaf': [2, 8],
                         'modele__class_weight': [None, 'balanced']},
        'Baseline': {}}
    searches, comparison = {}, []
    for name, pipeline in build_pipelines(quantitative).items():
        print(f'Validation croisée : {name}', flush=True)
        search = GridSearchCV(pipeline, grids[name], cv=folds, scoring=SCORING,
                              refit='average_precision', n_jobs=1, error_score='raise',
                              return_train_score=False)
        # Une non-convergence doit être corrigée et non noyée dans les sorties.
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            search.fit(X_train, y_train)
        searches[name] = search
        row = {'modele': name, 'evaluation': 'CV_entrainement', 'seuil': .5,
               'parametres': json.dumps(search.best_params_, ensure_ascii=False)}
        for metric in SCORING:
            row[metric] = search.cv_results_[f'mean_test_{metric}'][search.best_index_]
            row[f'{metric}_ecart_type'] = search.cv_results_[f'std_test_{metric}'][search.best_index_]
        comparison.append(row)
        # Les références sont explicitement vérifiées pour la baseline et les modèles.
        assert search.cv is folds
    return searches, pd.DataFrame(comparison)


def evaluate_test(searches, X_test, y_test):
    """Un seul appel predict_proba par modèle ; seuil fixé à 0,5."""
    rows, predictions = [], {}
    for name, search in searches.items():
        probabilities = search.best_estimator_.predict_proba(X_test)[:, 1]
        predicted = (probabilities >= .5).astype(int)
        matrix = confusion_matrix(y_test, predicted, labels=[0, 1])
        predictions[name] = {'probabilites': probabilities, 'predictions': predicted, 'matrice': matrix}
        tn, fp, fn, tp = matrix.ravel()
        rows.append({'modele': name, 'evaluation': 'test_reserve', 'seuil': .5,
                     'average_precision': average_precision_score(y_test, probabilities),
                     'roc_auc': roc_auc_score(y_test, probabilities),
                     'precision': precision_score(y_test, predicted, zero_division=0),
                     'recall': recall_score(y_test, predicted, zero_division=0),
                     'f1': f1_score(y_test, predicted, zero_division=0),
                     'vrais_negatifs': tn, 'faux_positifs': fp, 'faux_negatifs': fn, 'vrais_positifs': tp})
    return pd.DataFrame(rows), predictions


def transformed_features(searches):
    frames = []
    for name in ['Logistique', 'RandomForest']:
        names = searches[name].best_estimator_.named_steps['preparation'].get_feature_names_out()
        frames.append(pd.DataFrame({'modele': name, 'position': np.arange(len(names)), 'variable_transformee': names}))
    return pd.concat(frames, ignore_index=True)


def save_artifacts(searches, split_rows, path=DATA_PATH):
    """Modèles ajustés sur l'entraînement complet, jamais réajustés au test."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, search in searches.items():
        joblib.dump(search.best_estimator_, MODEL_DIR / f'04_{name}.joblib', compress=3)
    split_rows.to_csv(MODEL_DIR / '04_indices_split_plis.csv', index=False, encoding='utf-8-sig')
    metadata = {'fichier': str(path.relative_to(ROOT)), 'sha256_source': sha256(path.read_bytes()).hexdigest(),
                'random_state': 42, 'test_size': .2, 'plis': 5, 'seuil': .5,
                'selection': 'average_precision', 'python': platform.python_version(),
                'pandas': pd.__version__, 'numpy': np.__version__, 'scikit_learn': sklearn.__version__,
                'joblib': joblib.__version__,
                'parametres': {name: search.best_params_ for name, search in searches.items()}}
    (MODEL_DIR / '04_metadonnees.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata
