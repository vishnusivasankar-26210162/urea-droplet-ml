
import numpy as np
import pytest

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.ridge_scratch import (
    ridge_fit,
    ridge_predict,
    ridge_fit_gradient_descent,
    ridge_predict_gradient_descent,
    ScratchRidgeRegressor,
)


def make_synthetic_data():
    rng = np.random.default_rng(42)

    X = rng.normal(size=(120, 9))

    true_w = np.array([
        0.8, -0.4, 1.1,
        0.2, 0.5, -0.7,
        0.3, 0.9, -0.2,
    ])

    y = (
        X @ true_w
        + 2.5
        + rng.normal(scale=0.1, size=len(X))
    )

    return X, y


def test_closed_form_matches_sklearn_ridge():
    X, y = make_synthetic_data()
    alpha = 2.0

    scratch_model = ridge_fit(X, y, alpha)
    scratch_predictions = ridge_predict(
        scratch_model,
        X,
    )

    sklearn_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)

    np.testing.assert_allclose(
        scratch_predictions,
        sklearn_predictions,
        atol=1e-6,
        rtol=1e-6,
    )


def test_gradient_descent_matches_sklearn_ridge():
    X, y = make_synthetic_data()
    alpha = 2.0

    scratch_model = ridge_fit_gradient_descent(
        X,
        y,
        alpha,
        max_iter=100000,
        tol=1e-10,
    )

    assert scratch_model["converged"]

    scratch_predictions = ridge_predict_gradient_descent(
        scratch_model,
        X,
    )

    sklearn_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=alpha),
    )

    sklearn_model.fit(X, y)
    sklearn_predictions = sklearn_model.predict(X)

    np.testing.assert_allclose(
        scratch_predictions,
        sklearn_predictions,
        atol=1e-6,
        rtol=1e-6,
    )


def test_scratch_estimator_works_with_sklearn_pipeline():
    X, y = make_synthetic_data()

    model = ScratchRidgeRegressor(alpha=1.0)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == y.shape
    assert np.isfinite(predictions).all()


def test_scratch_estimator_can_be_cloned():
    from sklearn.base import clone

    model = ScratchRidgeRegressor(alpha=3.0)
    copied = clone(model)

    assert copied.alpha == 3.0
