import numpy as np

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.linear_model import Ridge, LinearRegression

from src.model_pipelines import (
    make_numeric_pipeline,
    make_b0_pipeline,
    make_b1_pipeline,
    make_ridge_pipeline,
)


def test_preprocessing_is_fitted_inside_each_fold():
    X = np.array([
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0],
        [7.0, 70.0],
        [8.0, 80.0],
        [9.0, 90.0],
        [10.0, 100.0],
        [11.0, 110.0],
        [12.0, 120.0],
    ])

    X[0, 0] = np.nan
    X[5, 1] = np.nan

    y = np.arange(12, dtype=float)

    # Each pair belongs to one physical group.
    groups = np.repeat(np.arange(6), 2)

    cv = GroupKFold(n_splits=3)
    splits = list(cv.split(X, y, groups))

    pipeline = make_ridge_pipeline(alpha=1.0)

    results = cross_validate(
        pipeline,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="neg_mean_absolute_error",
        return_estimator=True,
    )

    assert len(results["estimator"]) == 3

    for (train_indices, _), fitted_pipeline in zip(
        splits,
        results["estimator"],
    ):
        imputer = fitted_pipeline.named_steps["imputer"]
        scaler = fitted_pipeline.named_steps["scaler"]

        # Median imputation must use this fold's training rows only.
        expected_medians = np.nanmedian(
            X[train_indices],
            axis=0,
        )

        np.testing.assert_allclose(
            imputer.statistics_,
            expected_medians,
        )

        imputed_train = np.where(
            np.isnan(X[train_indices]),
            expected_medians,
            X[train_indices],
        )

        # Scaler statistics must also use only this fold's training rows.
        expected_means = imputed_train.mean(axis=0)

        np.testing.assert_allclose(
            scaler.mean_,
            expected_means,
        )


def test_optional_feature_selection_and_pca_are_pipeline_steps():
    pipeline = make_numeric_pipeline(
        Ridge(alpha=1.0),
        standardize=True,
        select_k=3,
        pca_components=2,
    )

    names = list(pipeline.named_steps.keys())

    assert names == [
        "imputer",
        "scaler",
        "selector",
        "pca",
        "model",
    ]


def test_b0_pipeline():
    pipeline = make_b0_pipeline()

    assert list(pipeline.named_steps.keys()) == ["model"]
    assert pipeline.named_steps["model"].strategy == "mean"


def test_b1_pipeline():
    pipeline = make_b1_pipeline()

    assert list(pipeline.named_steps.keys()) == [
        "imputer",
        "model",
    ]

    assert isinstance(
        pipeline.named_steps["model"],
        LinearRegression,
    )


def test_ridge_pipeline_has_fold_local_preprocessing():
    pipeline = make_ridge_pipeline(alpha=2.0)

    assert list(pipeline.named_steps.keys()) == [
        "imputer",
        "scaler",
        "model",
    ]

    assert pipeline.named_steps["model"].alpha == 2.0
