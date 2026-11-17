"""Contrôles des intervalles descriptifs et des contrastes de la restitution."""
import unittest
import joblib
import numpy as np
from src.modeling import MODEL_DIR
from src.reporting import selected_log_odds, wilson


class ReportingTests(unittest.TestCase):
    def test_wilson_handles_empty_and_extreme_groups(self):
        self.assertTrue(np.isnan(wilson(0, 0)).all())
        lo, hi = wilson(0, 100)
        self.assertAlmostEqual(lo, 0)
        self.assertGreater(hi, 0)
        lo, hi = wilson(100, 100)
        self.assertLess(lo, 1)
        self.assertAlmostEqual(hi, 1)
        lo, hi = wilson(50, 100)
        self.assertAlmostEqual(lo, 1-hi)

    def test_full_onehot_uses_category_differences(self):
        model = joblib.load(MODEL_DIR/'04_Logistique.joblib')
        pre = model.named_steps['preparation']
        beta = dict(zip(pre.get_feature_names_out(), model.named_steps['modele'].coef_[0]))
        values = selected_log_odds(model)
        self.assertAlmostEqual(values['Déplacements : fréquents / rares'],
            beta['categories_scores__BusinessTravel_Travel_Frequently']-
            beta['categories_scores__BusinessTravel_Travel_Rarely'])
        self.assertAlmostEqual(values['Heures supplémentaires : Oui / Non'], beta['binaires__OverTime'])
        self.assertEqual(len(values), 6)


if __name__ == '__main__':
    unittest.main()
