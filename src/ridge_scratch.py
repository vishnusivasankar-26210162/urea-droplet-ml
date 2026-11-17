
"""
Ridge regression implemented from scratch.

Closed-form:
    Z = (X - mu) / sd
    b0 = mean(y)
    w = solve(Z.T @ Z + alpha * I, Z.T @ (y - b0))

Gradient descent minimizes the same objective.

This implementation is compatible with sklearn Pipeline and
GridSearchCV.
"""

from __future__ import annotations

import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
)


def ridge_fit(X, y, lam):
    """Fit Ridge using the closed-form solution."""

    X, y = check_X_y(
        X,
        y,
        dtype=np.float64,
        y_numeric=True,
    )

    if lam < 0:
        raise ValueError("lambda must be non-negative")

    mu = X.mean(axis=0)
    sd = X.std(axis=0)

    # Avoid division by zero for constant features.
    sd[sd == 0] = 1.0

    Z = (X - mu) / sd

    b0 = float(y.mean())

    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    rhs = Z.T @ (y - b0)

    w = np.linalg.solve(A, rhs)

    return mu, sd, b0, w


def ridge_predict(model, X):
    """Predict using a model returned by ridge_fit."""

    mu, sd, b0, w = model

    X = check_array(
        X,
        dtype=np.float64,
    )

    return ((X - mu) / sd) @ w + b0


def ridge_fit_gradient_descent(
    X,
    y,
    lam,
    max_iter=100000,
    tol=1e-10,
):
    """
    Fit Ridge using gradient descent.

    The objective is:
        0.5 * ||Z @ w - (y - b0)||^2
        + 0.5 * lam * ||w||^2

    Step size is selected from the Lipschitz constant
    for stable convergence.
    """

    X, y = check_X_y(
        X,
        y,
        dtype=np.float64,
        y_numeric=True,
    )

    if lam < 0:
        raise ValueError("lambda must be non-negative")

    if max_iter < 1:
        raise ValueError("max_iter must be positive")

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0

    Z = (X - mu) / sd
    b0 = float(y.mean())
    yc = y - b0

    # Hessian: Z.T @ Z + lambda * I.
    A = Z.T @ Z + lam * np.eye(Z.shape[1])

    # Largest eigenvalue gives a conservative stable step size.
    largest_eigenvalue = float(
        np.linalg.eigvalsh(A).max()
    )

    learning_rate = 1.0 / max(largest_eigenvalue, 1e-12)

    w = np.zeros(Z.shape[1], dtype=np.float64)

    converged = False

    for iteration in range(1, max_iter + 1):
        gradient = Z.T @ (Z @ w - yc) + lam * w

        new_w = w - learning_rate * gradient

        if np.linalg.norm(new_w - w, ord=np.inf) <= tol:
            w = new_w
            converged = True
            break

        w = new_w

    return {
        "model": (mu, sd, b0, w),
        "iterations": iteration,
        "converged": converged,
        "learning_rate": learning_rate,
    }


def ridge_predict_gradient_descent(model, X):
    """Predict using the gradient-descent result."""

    return ridge_predict(model["model"], X)


class ScratchRidgeRegressor(RegressorMixin, BaseEstimator):
    """
    Scikit-learn compatible Ridge estimator.

    The input is standardized inside fit(), so it is safe to use
    after a fold-local imputer in a Pipeline.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

        self.model_ = ridge_fit(
            X,
            y,
            self.alpha,
        )

        self.n_features_in_ = np.asarray(X).shape[1]

        return self

    def predict(self, X):
        check_is_fitted(self, "model_")

        return ridge_predict(
            self.model_,
            X,
        )
