
import unittest

import numpy as np
import pandas as pd

from src.baselines import (
    fit_baselines,
    predict_baselines,
    evaluate_baselines,
)


class TestBaselines(unittest.TestCase):

    def setUp(self):
        # Synthetic data with a known linear relationship:
        # concentration = 0.5 * median_b_star + 1.0

        self.X_train = pd.DataFrame({
            "median_b_star": [0.0, 2.0, 4.0, 6.0, 8.0]
        })

        self.y_train = np.array([
            1.0, 2.0, 3.0, 4.0, 5.0
        ])

        self.X_eval = pd.DataFrame({
            "median_b_star": [1.0, 3.0, 5.0]
        })

        self.y_eval = np.array([
            1.5, 2.5, 3.5
        ])

    def test_b0_predicts_training_mean(self):

        models = fit_baselines(
            self.X_train,
            self.y_train,
        )

        predictions = predict_baselines(
            models,
            self.X_eval,
        )

        expected_mean = self.y_train.mean()

        np.testing.assert_allclose(
            predictions["B0_mean"],
            np.full(len(self.X_eval), expected_mean),
        )

    def test_b1_learns_linear_calibration(self):

        models = fit_baselines(
            self.X_train,
            self.y_train,
        )

        predictions = predict_baselines(
            models,
            self.X_eval,
        )

        expected = np.array([1.5, 2.5, 3.5])

        np.testing.assert_allclose(
            predictions["B1_median_b_star"],
            expected,
            atol=1e-10,
        )

    def test_b1_uses_only_median_b_star(self):

        X_train = self.X_train.copy()
        X_train["unrelated_feature"] = [100, 200, 300, 400, 500]

        X_eval = self.X_eval.copy()
        X_eval["unrelated_feature"] = [-100, -200, -300]

        models = fit_baselines(
            X_train,
            self.y_train,
        )

        predictions = predict_baselines(
            models,
            X_eval,
        )

        np.testing.assert_allclose(
            predictions["B1_median_b_star"],
            [1.5, 2.5, 3.5],
            atol=1e-10,
        )

    def test_evaluation_returns_both_baselines(self):

        result = evaluate_baselines(
            self.X_train,
            self.y_train,
            self.X_eval,
            self.y_eval,
        )

        self.assertIn("B0_mean", result["metrics"])
        self.assertIn("B1_median_b_star", result["metrics"])

        self.assertEqual(
            len(result["predictions"]["B0_mean"]), 3
        )

        self.assertEqual(
            len(result["predictions"]["B1_median_b_star"]), 3
        )

    def test_missing_b1_feature_raises_error(self):

        X = pd.DataFrame({
            "some_other_feature": [1, 2, 3]
        })

        with self.assertRaises(ValueError):
            fit_baselines(X, np.array([1.0, 2.0, 3.0]))

    def test_training_length_mismatch_raises_error(self):

        with self.assertRaises(ValueError):
            fit_baselines(
                self.X_train,
                np.array([1.0, 2.0]),
            )

    def test_b1_handles_missing_feature_values(self):

        X_train = pd.DataFrame({
            "median_b_star": [0.0, 2.0, np.nan, 6.0, 8.0]
        })

        models = fit_baselines(
            X_train,
            self.y_train,
        )

        predictions = predict_baselines(
            models,
            self.X_eval,
        )

        self.assertTrue(
            np.isfinite(predictions["B1_median_b_star"]).all()
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
