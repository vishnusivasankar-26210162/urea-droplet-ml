
"""Evaluation metrics for urea concentration regression.

All errors are expressed in percentage points of added urea (% w/v).

Usage:
    from src.metrics import (
        compute_regression_metrics,
        compute_detection_metrics,
    )

    regression = compute_regression_metrics(y_true, y_pred)
    detection = compute_detection_metrics(y_true, y_pred)
"""

import numpy as np
from sklearn.metrics import roc_auc_score


def _validate_predictions(y_true, y_pred):
    """Validate and convert actual and predicted concentrations."""

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be one-dimensional")

    if len(y_true) == 0:
        raise ValueError("Prediction arrays cannot be empty")

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths")

    if not np.isfinite(y_true).all():
        raise ValueError("y_true contains NaN or infinite values")

    if not np.isfinite(y_pred).all():
        raise ValueError("y_pred contains NaN or infinite values")

    return y_true, y_pred


def _band_mae(y_true, absolute_errors, mask):
    """Return sample count and MAE for one concentration band."""

    count = int(mask.sum())

    if count == 0:
        return {
            "n": 0,
            "mae": float("nan"),
        }

    return {
        "n": count,
        "mae": float(absolute_errors[mask].mean()),
    }


def _calibration_check(y_true, y_pred):
    """Regress actual concentration on predicted concentration.

    Ideal calibration:
        slope = 1
        intercept = 0
    """

    if len(y_true) < 2 or np.unique(y_pred).size < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "note": "Insufficient variation for calibration regression",
        }

    # Model: actual = slope * predicted + intercept
    design = np.column_stack([
        y_pred,
        np.ones(len(y_pred)),
    ])

    coefficients, _, rank, _ = np.linalg.lstsq(
        design,
        y_true,
        rcond=None,
    )

    if rank < 2:
        return {
            "slope": float("nan"),
            "intercept": float("nan"),
            "note": "Calibration design matrix is rank deficient",
        }

    slope, intercept = coefficients

    return {
        "slope": float(slope),
        "intercept": float(intercept),
    }


def compute_regression_metrics(
    y_true,
    y_pred,
    tolerance=0.1,
):
    """Compute all primary regression metrics.

    Parameters
    ----------
    y_true:
        Actual urea concentrations (% w/v).
    y_pred:
        Predicted urea concentrations (% w/v).
    tolerance:
        Absolute error tolerance for the within-tolerance metric.

    Returns
    -------
    dict:
        Overall metrics, concentration-band MAE, and calibration.
    """

    y_true, y_pred = _validate_predictions(y_true, y_pred)

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")

    # Positive error means overprediction.
    errors = y_pred - y_true
    absolute_errors = np.abs(errors)

    mae = float(absolute_errors.mean())
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # R² is undefined if there is only one observation or if
    # every actual target has the same value.
    if len(y_true) >= 2 and np.var(y_true) > 0:
        ss_res = np.sum(errors ** 2)
        ss_total = np.sum((y_true - y_true.mean()) ** 2)

        r2 = float(1.0 - ss_res / ss_total)
    else:
        r2 = float("nan")

    within_tolerance = (
        absolute_errors <= tolerance + 1e-12
    )

    # Concentration bands are defined using ACTUAL concentrations.
    low_mask = y_true < 0.5
    mid_mask = (y_true >= 0.5) & (y_true <= 1.0)
    high_mask = y_true > 1.0

    return {
        "n": int(len(y_true)),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "bias_pred_minus_true": float(errors.mean()),
        "within_tolerance_fraction": float(
            within_tolerance.mean()
        ),
        "within_tolerance_percentage": float(
            100.0 * within_tolerance.mean()
        ),
        "tolerance": float(tolerance),
        "mae_by_band": {
            "low_below_0.5": _band_mae(
                y_true, absolute_errors, low_mask
            ),
            "mid_0.5_to_1.0": _band_mae(
                y_true, absolute_errors, mid_mask
            ),
            "high_above_1.0": _band_mae(
                y_true, absolute_errors, high_mask
            ),
        },
        "calibration": _calibration_check(
            y_true, y_pred
        ),
    }


def compute_detection_metrics(
    y_true,
    y_pred,
    thresholds=(0.2, 0.5, 1.0),
):
    """Optional detection analysis.

    For each concentration threshold T:

        Actual positive    = y_true >= T
        Predicted positive = y_pred >= T

    ROC AUC uses the continuous predicted concentrations as scores.
    Sensitivity and specificity use T as the classification threshold.
    """

    y_true, y_pred = _validate_predictions(y_true, y_pred)

    results = {}

    for threshold in thresholds:

        if threshold < 0:
            raise ValueError("Detection thresholds cannot be negative")

        actual_positive = y_true >= threshold
        predicted_positive = y_pred >= threshold

        tp = int(np.sum(actual_positive & predicted_positive))
        tn = int(np.sum(~actual_positive & ~predicted_positive))
        fp = int(np.sum(~actual_positive & predicted_positive))
        fn = int(np.sum(actual_positive & ~predicted_positive))

        n_positive = tp + fn
        n_negative = tn + fp

        sensitivity = (
            float(tp / n_positive)
            if n_positive > 0
            else float("nan")
        )

        specificity = (
            float(tn / n_negative)
            if n_negative > 0
            else float("nan")
        )

        # AUC is undefined if the actual labels contain only
        # positive cases or only negative cases.
        if n_positive > 0 and n_negative > 0:
            auc = float(
                roc_auc_score(
                    actual_positive.astype(int),
                    y_pred,
                )
            )
        else:
            auc = float("nan")

        results[float(threshold)] = {
            "n_positive": n_positive,
            "n_negative": n_negative,
            "roc_auc": auc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    return results
