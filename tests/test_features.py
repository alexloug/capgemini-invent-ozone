"""Contrôles des variables candidates : divisions, incohérences et références."""
import unittest
import numpy as np
import pandas as pd
from src.features import build_candidates, fit_income_reference, relative_income


class FeatureTests(unittest.TestCase):
    def test_inconsistent_tenure_and_zero_denominators_are_not_invented(self):
        x = pd.DataFrame({'YearsAtCompany': [0, 5, 3], 'TotalWorkingYears': [0, 3, 5],
                          'YearsInCurrentRole': [0, 2, 4], 'YearsWithCurrManager': [0, 1, 1],
                          'OverTime': ['No', 'Yes', 'Yes'],
                          'BusinessTravel': ['Non-Travel', 'Travel_Frequently', 'Travel_Rarely'],
                          'DistanceFromHome': [1, 4, 6]})
        before = x.copy(deep=True)
        result = build_candidates(x)
        self.assertTrue(np.isnan(result.loc[0, 'CompanyTenureShare']))
        self.assertTrue(np.isnan(result.loc[0, 'RoleTenureShare']))
        self.assertTrue(np.isnan(result.loc[1, 'PriorExperienceYears']))
        self.assertTrue(np.isnan(result.loc[2, 'YearsBeforeCurrentRole']))
        self.assertEqual(result.OverTimeFrequentTravel.tolist(), [0, 1, 0])
        self.assertEqual(result.DistanceIfOverTime.tolist(), [0, 4, 6])
        self.assertFalse(np.isinf(result.to_numpy()).any())
        pd.testing.assert_frame_equal(x, before)

    def test_income_reference_is_fitted_once_and_unknown_groups_stay_missing(self):
        train = pd.DataFrame({'JobRole': ['A']*5, 'JobLevel': [1]*5,
                              'MonthlyIncome': [100, 200, 300, 400, 500]})
        reference = fit_income_reference(train)
        future = pd.DataFrame({'JobRole': ['A', 'B'], 'JobLevel': [1, 1],
                               'MonthlyIncome': [600, 900]}, index=[10, 11])
        result = relative_income(future, reference)
        self.assertEqual(result.loc[10], 2)
        self.assertTrue(np.isnan(result.loc[11]))
        self.assertEqual(reference.median_income.iloc[0], 300)


if __name__ == '__main__':
    unittest.main()
