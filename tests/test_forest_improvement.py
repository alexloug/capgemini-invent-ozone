"""Vérifie le choix du seuil et les sous-ensembles du notebook 06."""
import unittest

import numpy as np
from sklearn.metrics import fbeta_score

from src.forest_improvement import candidates, choose_threshold, make_model, scores, SUBSETS
from src.modeling import build_pipelines, load_model_data


class ForestImprovementTests(unittest.TestCase):
    def test_threshold_maximizes_f2_with_consistent_tie_break(self):
        y = np.array([0, 1, 0, 1, 1, 0, 0, 1])
        p = np.array([.1, .2, .2, .4, .4, .7, .8, .9])
        values = [(cut, fbeta_score(y, p >= cut, beta=2)) for cut in np.unique(p)]
        maximum = max(value for _, value in values)
        expected = max(cut for cut, value in values if np.isclose(value, maximum, rtol=0, atol=1e-12))
        self.assertEqual(choose_threshold(y, p), expected)
        low, high = scores(y, p, .2), scores(y, p, .8)
        self.assertEqual(low['roc_auc'], high['roc_auc'])
        self.assertEqual(low['average_precision'], high['average_precision'])
        self.assertGreater(low['recall'], high['recall'])

    def test_all_configs_preserve_feature_exclusion_and_clone_reference(self):
        X, y, _, _, quantitative = load_model_data()
        reference = build_pipelines(quantitative)['RandomForest']
        configs = candidates(reference)
        self.assertEqual(len(configs), 20)
        self.assertEqual(configs, candidates(reference))
        for config in configs[:8]:
            pipe, columns = make_model(reference, config, X.columns)
            self.assertFalse(set(columns).intersection(SUBSETS[config['subset']]))
            self.assertEqual(set(columns) | set(SUBSETS[config['subset']]), set(X))
            pipe.set_params(modele__n_estimators=5)
            pipe.fit(X.iloc[:150][columns], y.iloc[:150])
            self.assertTrue(np.isfinite(pipe.predict_proba(X.iloc[150:155][columns])).all())
            self.assertEqual(set(pipe.named_steps['preparation'].feature_names_in_), set(columns))
        self.assertEqual(reference.named_steps['modele'].n_estimators, 200)
        self.assertFalse(hasattr(reference.named_steps['modele'], 'classes_'))


if __name__ == '__main__':
    unittest.main()
