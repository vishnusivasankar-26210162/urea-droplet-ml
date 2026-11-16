
import unittest

import numpy as np

from src.metrics import (
    compute_regression_metrics,
    compute_detection_metrics,
)


class TestRegressionMetrics(unittest.TestCase):

    def test_basic_regression_metrics(self):

        y_true = np.array([0.0, 0.5, 1.0, 1.5])
        y_pred = np.array([0.1, 0.4, 1.1, 1.4])

        result = compute_regression_metrics(y_true, y_pred)

        self.assertAlmostEqual(result["mae"], 0.1)
        self.assertAlmostEqual(result["rmse"], 0.1)
        self.assertAlmostEqual(result["bias_pred_minus_true"], 0.0)

        self.assertAlmostEqual(
            result["within_tolerance_fraction"], 1.0
        )

        self.assertEqual(result["n"], 4)

    def test_concentration_band_mae(self):

        y_true = np.array([0.1, 0.5, 1.0, 1.5])
        y_pred = np.array([0.2, 0.6, 0.9, 1.3])

        result = compute_regression_metrics(y_true, y_pred)

        bands = result["mae_by_band"]

        self.assertEqual(bands["low_below_0.5"]["n"], 1)
        self.assertEqual(bands["mid_0.5_to_1.0"]["n"], 2)
        self.assertEqual(bands["high_above_1.0"]["n"], 1)

        self.assertAlmostEqual(
            bands["low_below_0.5"]["mae"], 0.1
        )

        self.assertAlmostEqual(
            bands["mid_0.5_to_1.0"]["mae"], 0.1
        )

        self.assertAlmostEqual(
            bands["high_above_1.0"]["mae"], 0.2
        )

    def test_calibration_regression(self):

        y_true = np.array([0.0, 1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 3.0, 5.0, 7.0])

        result = compute_regression_metrics(y_true, y_pred)
        calibration = result["calibration"]

        # actual = 0.5 * predicted - 0.5
        self.assertAlmostEqual(calibration["slope"], 0.5)
        self.assertAlmostEqual(calibration["intercept"], -0.5)

    def test_constant_actual_target_has_undefined_r2(self):

        y_true = np.array([0.5, 0.5, 0.5])
        y_pred = np.array([0.4, 0.5, 0.6])

        result = compute_regression_metrics(y_true, y_pred)

        self.assertTrue(np.isnan(result["r2"]))

    def test_empty_or_mismatched_inputs_raise_error(self):

        with self.assertRaises(ValueError):
            compute_regression_metrics([], [])

        with self.assertRaises(ValueError):
            compute_regression_metrics([0.1, 0.2], [0.1])

    def test_non_finite_inputs_raise_error(self):

        with self.assertRaises(ValueError):
            compute_regression_metrics([0.1, np.nan], [0.1, 0.2])


class TestDetectionMetrics(unittest.TestCase):

    def test_perfect_detection(self):

        y_true = np.array([0.1, 0.3, 0.6, 1.2])
        y_pred = np.array([0.2, 0.4, 0.5, 1.3])

        result = compute_detection_metrics(
            y_true,
            y_pred,
            thresholds=(0.5,),
        )

        metrics = result[0.5]

        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["sensitivity"], 1.0)
        self.assertAlmostEqual(metrics["specificity"], 1.0)

    def test_detection_confusion_counts(self):

        y_true = np.array([0.1, 0.4, 0.8, 1.2])
        y_pred = np.array([0.7, 0.4, 0.9, 1.1])

        result = compute_detection_metrics(
            y_true,
            y_pred,
            thresholds=(0.5,),
        )

        metrics = result[0.5]

        self.assertEqual(metrics["tp"], 2)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 0)

    def test_auc_undefined_for_one_class(self):

        y_true = np.array([0.1, 0.2, 0.3])
        y_pred = np.array([0.2, 0.3, 0.4])

        result = compute_detection_metrics(
            y_true,
            y_pred,
            thresholds=(0.5,),
        )

        self.assertTrue(np.isnan(result[0.5]["roc_auc"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
