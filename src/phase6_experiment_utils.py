
"""Shared utilities for Phase 6.5 experiments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.feature_selection import f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
)
from sklearn.svm import SVR

try:
    from src.ridge_scratch import ScratchRidgeRegressor
except ModuleNotFoundError:
    from ridge_scratch import ScratchRidgeRegressor


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data/processed/dataset.csv"
REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"
WALKING_OOF_PATH = ROOT / "results/walking_skeleton_oof.csv"

S2_COLUMNS = [
    "L_mean", "a_mean", "b_mean",
    "L_med", "a_med", "b_med",
    "L_std", "a_std", "b_std",
]

CONDITIONS = [
    "dino_off",
    "dino_on",
    "phone_off",
    "phone_on",
]

OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_STATE = 42

ALPHA_GRID = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]


def load_training_data():
    """Load only training rows. No image files are opened."""

    df = pd.read_csv(DATASET_PATH)

    required = {
        "image_id", "tube_id", "role", "conc",
        "device", "lamp", "b_med", *S2_COLUMNS,
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")

    df = df[df["role"].astype(str) == "train"].copy()

    df["image_id"] = df["image_id"].astype(str)
    df["tube_id"] = df["tube_id"].astype(str)
    df["conc"] = pd.to_numeric(df["conc"], errors="raise")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in training data")

    if df["tube_id"].isna().any():
        raise ValueError("Missing tube_id in training data")

    df["condition"] = [
        normalize_condition(d, l)
        for d, l in zip(df["device"], df["lamp"])
    ]

    return df.reset_index(drop=True)


def normalize_condition(device, lamp):
    d = str(device).strip().lower()
    l = str(lamp).strip().lower()

    if "dino" in d:
        d = "dino"
    elif "phone" in d or "samsung" in d or "a31" in d:
        d = "phone"
    else:
        raise ValueError(f"Unknown device: {device!r}")

    if l in {"off", "false", "0"}:
        l = "off"
    elif l in {"on", "true", "1"}:
        l = "on"
    else:
        raise ValueError(f"Unknown lamp: {lamp!r}")

    return f"{d}_{l}"


def load_registry():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_saved_outer_fold_map():
    """
    Read exact image-to-fold assignments from Phase 6.1 B1 OOF output.
    """

    old = pd.read_csv(WALKING_OOF_PATH)

    old = old[old["model"].astype(str) == "B1"].copy()

    if old.empty:
        raise ValueError("No B1 rows in Phase 6.1 OOF file")

    if old["image_id"].duplicated().any():
        raise ValueError("Duplicate image IDs in Phase 6.1 B1 OOF")

    mapping = dict(
        zip(
            old["image_id"].astype(str),
            old["fold"].astype(int),
        )
    )

    tube_fold = old[["tube_id", "fold"]].drop_duplicates()

    if tube_fold.groupby("tube_id")["fold"].nunique().max() != 1:
        raise ValueError("A tube has multiple Phase 6.1 fold assignments")

    return mapping


def exact_saved_outer_splits(df, fold_map):
    """Return indices matching the exact Phase 6.1 outer folds."""

    folds = df["image_id"].map(fold_map)

    if folds.isna().any():
        missing = df.loc[folds.isna(), "image_id"].tolist()
        raise ValueError(
            f"Some images have no Phase 6.1 fold: {missing[:10]}"
        )

    folds = folds.to_numpy(dtype=int)
    groups = df["tube_id"].astype(str).to_numpy()

    splits = []

    for fold in sorted(np.unique(folds)):
        valid_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)

        if not set(groups[train_idx]).isdisjoint(groups[valid_idx]):
            raise RuntimeError(f"Tube leakage in saved fold {fold}")

        splits.append((train_idx, valid_idx))

    if len(splits) != OUTER_SPLITS:
        raise ValueError("Saved outer fold count is not five")

    return splits, folds


def grouped_outer_splits(df, n_splits=5):
    """Create fresh group folds, used for LOCO/transfer subsets."""

    groups = df["tube_id"].astype(str).to_numpy()

    if len(np.unique(groups)) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} tubes, found {len(np.unique(groups))}"
        )

    cv = GroupKFold(n_splits=n_splits)

    return list(
        cv.split(
            np.zeros((len(df), 1)),
            df["conc"].to_numpy(dtype=float),
            groups,
        )
    )


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    low = y_true < 0.5
    medium = (y_true >= 0.5) & (y_true <= 1.0)
    high = y_true > 1.0

    def band_mae(mask):
        return (
            float(absolute_error[mask].mean())
            if mask.any()
            else np.nan
        )

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_pred_minus_actual": float(error.mean()),
        "within_plus_minus_0_1": float(
            np.mean(absolute_error <= 0.1)
        ),
        "mae_low_actual_lt_0_5": band_mae(low),
        "mae_medium_0_5_to_1_0": band_mae(medium),
        "mae_high_actual_gt_1_0": band_mae(high),
    }


def build_model(model_name):
    """Return pipeline and compact grid for a model."""

    imputer = SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    if model_name == "B0":
        return (
            Pipeline([
                ("model", DummyRegressor(strategy="mean")),
            ]),
            {},
        )

    if model_name == "B1":
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                ("model", LinearRegression()),
            ]),
            {},
        )

    if model_name in {"Ridge", "Ridge_S2"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                ("model", ScratchRidgeRegressor(alpha=1.0)),
            ]),
            {"model__alpha": ALPHA_GRID},
        )

    if model_name in {"Polynomial_Ridge", "Polynomial_Ridge_b_med"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                (
                    "poly",
                    PolynomialFeatures(
                        degree=2,
                        include_bias=False,
                    ),
                ),
                ("scaler", StandardScaler()),
                ("model", Ridge()),
            ]),
            {"model__alpha": ALPHA_GRID},
        )

    if model_name in {"SVR", "SVR_RBF_S2"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf")),
            ]),
            {
                "model__C": [1, 10, 30],
                "model__gamma": [0.03, 0.1, 0.3],
                "model__epsilon": [0.02, 0.1],
            },
        )

    if model_name in {"Random_Forest", "Random_Forest_S2"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]),
            {"model__min_samples_leaf": [2, 5, 10]},
        )

    if model_name in {"Gradient_Boosting", "Gradient_Boosting_S2"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                (
                    "model",
                    GradientBoostingRegressor(
                        subsample=0.7,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            {
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [100, 300],
            },
        )

    if model_name in {"MLP", "MLP_S2"}:
        return (
            Pipeline([
                ("imputer", clone(imputer)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        solver="lbfgs",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]),
            {
                "model__hidden_layer_sizes": [(8,), (16,), (32,)],
                "model__alpha": [0.001, 0.01, 0.1],
            },
        )

    raise ValueError(f"Unknown model: {model_name}")


def fit_with_inner_cv(
    model_name,
    X_train,
    y_train,
    groups_train,
):
    """
    Tune using 4-fold GroupKFold inside the current outer-training set.
    Returns a fitted estimator and selected parameters.
    """

    estimator, grid = build_model(model_name)

    if not grid:
        estimator.fit(X_train, y_train)
        return estimator, {}, np.nan

    n_groups = len(np.unique(groups_train))
    n_inner = min(INNER_SPLITS, n_groups)

    if n_inner < 2:
        raise ValueError("Not enough groups for inner cross-validation")

    inner = GroupKFold(n_splits=n_inner)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=grid,
        scoring="neg_mean_absolute_error",
        cv=inner,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    search.fit(
        X_train,
        y_train,
        groups=groups_train,
    )

    return (
        search.best_estimator_,
        search.best_params_,
        float(search.best_score_),
    )


def get_X(df, columns, model_name):
    """Return numeric model input; B0 receives a dummy column."""

    if model_name == "B0":
        return pd.DataFrame(
            {"dummy": np.zeros(len(df), dtype=float)},
            index=df.index,
        )

    missing = set(columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")

    return df[list(columns)].apply(pd.to_numeric, errors="coerce")


def append_metrics(rows, path=None):
    if path is None:
        path = ROOT / "results/metrics.csv"

    path.parent.mkdir(parents=True, exist_ok=True)

    new = pd.DataFrame(rows)

    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat(
            [old, new],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = new

    combined.to_csv(path, index=False)


def nested_oof(
    df,
    model_name,
    columns,
    outer_splits,
    experiment,
    label,
):
    """Run nested group CV for one configuration."""

    df = df.reset_index(drop=True).copy()

    y = df["conc"].to_numpy(dtype=float)
    groups = df["tube_id"].astype(str).to_numpy()
    X = get_X(df, columns, model_name)

    predictions = np.full(len(df), np.nan)
    fold_numbers = np.full(len(df), -1)

    param_rows = []

    for fold_number, (train_idx, valid_idx) in enumerate(
        outer_splits,
        start=1,
    ):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])

        if not train_groups.isdisjoint(valid_groups):
            raise RuntimeError(
                f"Tube leakage in {experiment}/{label}, fold {fold_number}"
            )

        fitted, params, inner_score = fit_with_inner_cv(
            model_name,
            X.iloc[train_idx],
            y[train_idx],
            groups[train_idx],
        )

        predictions[valid_idx] = fitted.predict(
            X.iloc[valid_idx]
        )

        fold_numbers[valid_idx] = fold_number

        param_rows.append({
            "experiment": experiment,
            "configuration": label,
            "model": model_name,
            "outer_fold": fold_number,
            "n_train_tubes": len(train_groups),
            "n_heldout_tubes": len(valid_groups),
            "inner_best_neg_mae": inner_score,
            "best_params": json.dumps(params, sort_keys=True),
        })

        print(
            f"    {label}/{model_name}: fold "
            f"{fold_number}/{len(outer_splits)}"
        )

    if np.isnan(predictions).any() or np.any(fold_numbers < 0):
        raise RuntimeError(
            f"Missing OOF predictions: {experiment}/{label}/{model_name}"
        )

    out = df[
        ["image_id", "tube_id", "conc", "role"]
    ].copy()

    if "condition" in df.columns:
        out["condition"] = df["condition"].values

    out["experiment"] = experiment
    out["configuration"] = label
    out["model"] = model_name
    out["fold"] = fold_numbers
    out["actual"] = y
    out["predicted"] = predictions
    out["absolute_error"] = np.abs(predictions - y)

    metrics = calculate_metrics(y, predictions)

    return out, metrics, param_rows
