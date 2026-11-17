"""Check business-level permutation and safe reuse of the existing pipelines."""
import unittest

import numpy as np
import pandas as pd

from src.group_feature_analysis import (FEATURE_GROUPS, FEATURE_TO_GROUP,
                                        batch_roc_auc, permute_columns, subset_pipeline)
from sklearn.metrics import roc_auc_score
from src.modeling import build_pipelines, load_model_data, make_split


class GroupAnalysisTests(unittest.TestCase):
    def test_batched_bootstrap_auc_matches_sklearn_with_ties(self):
        y = np.array([0, 1, 0, 1, 0, 0, 1])
        p = np.array([[.1, .4, .4, .7, .1, .8, .7], [.5]*7, [.8, .6, .2, .2, .8, .4, .6]])
        for indices in [np.arange(len(y)), np.array([6, 0, 2, 3, 3, 4, 6, 1])]:
            expected = [roc_auc_score(y[indices], a[indices]) for a in p]
            np.testing.assert_allclose(batch_roc_auc(y[indices], p[:, indices]), expected, atol=1e-14)

    def test_joint_permutation_preserves_relationships_and_source(self):
        X = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [10, 20, 30, 40],
                          'c': ['x', 'y', 'z', 'w']}, index=[8, 3, 7, 2])
        original = X.copy(deep=True)
        result = permute_columns(X, ['a', 'b'], [2, 0, 3, 1])
        np.testing.assert_array_equal(result.a * 10, result.b)
        np.testing.assert_array_equal(result.a, [3, 1, 4, 2])
        pd.testing.assert_series_equal(result.c, X.c)
        pd.testing.assert_frame_equal(X, original)

    def test_exhaustive_disjoint_mapping_and_single_binary_subsets(self):
        X, y, rows, _, quantitative = load_model_data()
        self.assertEqual(set(X), set(FEATURE_TO_GROUP))
        self.assertEqual(len(X.columns), sum(len(c) for c in FEATURE_GROUPS.values()))
        train, _, yt, _, folds, _ = make_split(X, y, rows)
        fit, valid = folds[0]
        for name, template in build_pipelines(quantitative).items():
            if name == 'Baseline':
                continue
            for group in ['Gender Equality', 'Working Conditions', 'Employee Profile']:
                columns = FEATURE_GROUPS[group]
                pipe = subset_pipeline(template, columns)
                pipe.fit(train.iloc[fit][columns], yt.iloc[fit])
                self.assertTrue(np.isfinite(pipe.predict_proba(train.iloc[valid][columns])).all())
                self.assertEqual(set(pipe.named_steps['preparation'].feature_names_in_), set(columns))
                self.assertFalse(hasattr(template.named_steps['modele'], 'classes_'))


if __name__ == '__main__':
    unittest.main()
