"""Contrôles des fuites, de l'encodage et des diagnostics de données."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

from src.modeling import (BINARY, EXCLUDED, SCORES, build_pipelines, encode_binary,
                          load_model_data, make_split)


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.rows, cls.groups, cls.quant = load_model_data()

    def test_split_and_folds_cover_every_row_once(self):
        train, test, yt, _, folds, rows = make_split(self.X, self.y, self.rows)
        self.assertTrue(set(train.index).isdisjoint(test.index))
        self.assertEqual(len(train)+len(test), len(self.X))
        self.assertEqual(len(set(np.concatenate([v for _, v in folds]))), len(train))
        self.assertEqual(sum(len(v) for _, v in folds), len(train))
        for fit, valid in folds:
            self.assertTrue(set(fit).isdisjoint(valid))
            self.assertEqual(set(yt.iloc[valid]), {0, 1})
        self.assertEqual(rows.position_dataset.nunique(), len(self.X))
        self.assertFalse(set(EXCLUDED).intersection(self.X.columns))
        self.assertEqual(set(self.groups.variable), set(self.X))

    def test_preparation_learns_only_training_and_preserves_scores(self):
        train, test, _, _, folds, _ = make_split(self.X, self.y, self.rows)
        fit, valid = folds[0]
        sample = train.iloc[fit]
        original = sample.copy(deep=True)
        pipes = build_pipelines(self.quant)
        lr = pipes['Logistique'].named_steps['preparation']
        lr.fit(sample)
        np.testing.assert_allclose(lr.named_transformers_['quantitatives'].mean_, sample[self.quant].mean())
        encoder = lr.named_transformers_['categories_scores']
        self.assertIsNone(encoder.drop)
        for i, col in enumerate(encoder.feature_names_in_):
            self.assertEqual(set(encoder.categories_[i]), set(sample[col]))
        # Une nouvelle catégorie en validation n'est pas apprise ni rejetée.
        future = test.iloc[:1].copy()
        future['JobRole'] = 'Modalité entièrement nouvelle'
        lr.transform(future)
        self.assertNotIn('Modalité entièrement nouvelle', encoder.categories_[3])
        rf = pipes['RandomForest'].named_steps['preparation']
        transformed = rf.fit_transform(sample)
        indices = rf.output_indices_['scores']
        np.testing.assert_array_equal(transformed[:, indices], sample[SCORES].to_numpy())
        indices = rf.output_indices_['quantitatives']
        np.testing.assert_array_equal(transformed[:, indices], sample[self.quant].to_numpy())
        pd.testing.assert_frame_equal(sample, original)
        expected = sample[list(BINARY)].copy()
        for col, mapping in BINARY.items():
            expected[col] = expected[col].map(mapping)
        pd.testing.assert_frame_equal(encode_binary(sample[list(BINARY)]), expected)

    def test_missing_and_unexpected_binary_raise_diagnostic(self):
        frame = pd.DataFrame({'Attrition': ['Yes'], 'Gender': ['Female'], 'OverTime': ['No']})
        for column, value, diagnostic in [('Gender', 'Inconnu', 'Modalités inattendues'),
                                           ('OverTime', None, 'Valeurs manquantes')]:
            with self.subTest(column=column):
                invalid = frame.copy()
                invalid[column] = value
                with patch('src.modeling.pd.read_excel', return_value=invalid):
                    with self.assertRaisesRegex(ValueError, diagnostic):
                        load_model_data()
        with self.assertRaisesRegex(ValueError, 'Valeurs binaires'):
            encode_binary(pd.DataFrame({'Gender': ['Inconnu'], 'OverTime': ['No']}))


if __name__ == '__main__':
    unittest.main()
