
"""
Leakage-safe scikit-learn pipelines.

Every learned transformation is fitted inside the training fold
when the complete pipeline is passed to cross-validation.
"""

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor


def make_numeric_pipeline(
    estimator,
    *,
    standardize=True,
    select_k=None,
    pca_components=None,
):
    """
    Build a numeric regression Pipeline.

    Steps:
        1. Median imputation
        2. Optional standardization
        3. Optional feature selection
        4. Optional PCA
        5. Estimator

    Every learned step is fitted only on the training fold.

    Parameters
    ----------
    estimator:
        sklearn-compatible estimator, e.g. Ridge(), SVR(),
        RandomForestRegressor().

    standardize:
        Whether to fit StandardScaler inside the pipeline.

    select_k:
        Optional number of features to retain using f_regression.
        Must be <= number of input features.

    pca_components:
        Optional number of PCA components.
        Must be compatible with the training fold dimensions.

    Returns
    -------
    sklearn.pipeline.Pipeline
    """

    steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        )
    ]

    if standardize:
        steps.append(
            ("scaler", StandardScaler())
        )

    if select_k is not None:
        if not isinstance(select_k, int) or select_k < 1:
            raise ValueError("select_k must be a positive integer")

        steps.append(
            (
                "selector",
                SelectKBest(
                    score_func=f_regression,
                    k=select_k,
                ),
            )
        )

    if pca_components is not None:
        if (
            not isinstance(pca_components, int)
            or pca_components < 1
        ):
            raise ValueError(
                "pca_components must be a positive integer"
            )

        if not standardize:
            raise ValueError(
                "Use standardize=True when applying PCA"
            )

        steps.append(
            (
                "pca",
                PCA(n_components=pca_components),
            )
        )

    steps.append(("model", estimator))

    return Pipeline(steps)


def make_b0_pipeline():
    """
    B0: predict the mean training-fold concentration.

    DummyRegressor learns the mean separately in each fold.
    """

    return Pipeline([
        (
            "model",
            DummyRegressor(strategy="mean"),
        )
    ])


def make_b1_pipeline():
    """
    B1: linear regression using median b* only.

    Median imputation is fitted inside the training fold.
    Scaling is unnecessary for this one-feature linear model.
    """

    return make_numeric_pipeline(
        LinearRegression(),
        standardize=False,
    )


def make_ridge_pipeline(alpha=1.0):
    """Ridge regression with fold-local imputation and scaling."""

    return make_numeric_pipeline(
        Ridge(alpha=alpha),
        standardize=True,
    )
