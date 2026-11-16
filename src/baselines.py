
"""Baseline models for urea concentration prediction.

B0: predict the mean concentration in the training data.
B1: linear regression using median b* as the only feature.

All concentrations and errors are in % w/v.
"""

import numpy as np
import pandas as pd

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.metrics import compute_regression_metrics


B1_FEATURE = "median_b_star"


def validate_targets(y, name="y"):
    """Validate a concentration target vector."""
    values = np.asarray(y, dtype=float)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if len(values) == 0:
        raise ValueError(f"{name} cannot be empty")

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")

    return values


def validate_b1_features(X, name="X"):
    """Extract and validate the single B1 feature."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame")

    if B1_FEATURE not in X.columns:
        raise ValueError(
            f"{name} must contain the '{B1_FEATURE}' column"
        )

    values = pd.to_numeric(
        X[B1_FEATURE], errors="coerce"
    ).to_numpy(dtype=float)

    if np.isinf(values).any():
        raise ValueError(
            f"{name} contains infinite {B1_FEATURE} values"
        )

    if np.isnan(values).all():
        raise ValueError(
            f"{name} contains no usable {B1_FEATURE} values"
        )

    return values.reshape(-1, 1)


def make_b0():
    """B0: predict the training-set mean concentration."""
    return DummyRegressor(strategy="mean")


def make_b1():
    """B1: fit a straight-line calibration using median b*.

    The imputer learns its median from the training fold only.
    The linear regression is fitted after imputation.
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "regression",
            LinearRegression(),
        ),
    ])


def make_baselines():
    """Return fresh, unfitted baseline estimators."""
    return {
        "B0_mean": make_b0(),
        "B1_median_b_star": make_b1(),
    }


def fit_baselines(X_train, y_train):
    """Fit B0 and B1 using training data only.

    Parameters
    ----------
    X_train:
        DataFrame containing median_b_star.
    y_train:
        Training concentrations.

    Returns
    -------
    dict:
        Fitted baseline estimators.
    """
    y_train = validate_targets(y_train, "y_train")

    b1_values = validate_b1_features(X_train, "X_train")

    if len(X_train) != len(y_train):
        raise ValueError(
            "X_train and y_train must have equal lengths"
        )

    if len(b1_values) < 2:
        raise ValueError("B1 requires at least two training samples")

    models = make_baselines()

    models["B0_mean"].fit(
        np.zeros((len(y_train), 1)),
        y_train,
    )

    models["B1_median_b_star"].fit(
        b1_values,
        y_train,
    )

    return models


def predict_baselines(models, X):
    """Generate predictions from fitted baselines."""

    if "B0_mean" not in models:
        raise ValueError("B0_mean estimator is missing")

    if "B1_median_b_star" not in models:
        raise ValueError("B1_median_b_star estimator is missing")

    b1_values = validate_b1_features(X)

    if len(b1_values) == 0:
        raise ValueError("Evaluation data cannot be empty")

    b0_input = np.zeros((len(b1_values), 1))

    predictions = {
        "B0_mean": models["B0_mean"].predict(b0_input),
        "B1_median_b_star": models[
            "B1_median_b_star"
        ].predict(b1_values),
    }

    return predictions


def evaluate_baselines(
    X_train,
    y_train,
    X_eval,
    y_eval,
):
    """Fit baselines on training data and evaluate on separate data.

    This function never fits a model on X_eval or y_eval.
    """

    y_train = validate_targets(y_train, "y_train")
    y_eval = validate_targets(y_eval, "y_eval")

    if len(X_train) != len(y_train):
        raise ValueError("Training feature/target length mismatch")

    if len(X_eval) != len(y_eval):
        raise ValueError("Evaluation feature/target length mismatch")

    models = fit_baselines(X_train, y_train)
    predictions = predict_baselines(models, X_eval)

    results = {}

    for name, y_pred in predictions.items():
        results[name] = compute_regression_metrics(
            y_eval,
            y_pred,
        )

    return {
        "models": models,
        "predictions": predictions,
        "metrics": results,
    }
