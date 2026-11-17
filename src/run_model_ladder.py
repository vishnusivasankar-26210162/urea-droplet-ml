
"""
Phase 6.2: Model ladder with nested tube-grouped CV.

Outer folds:
    Exact image-to-fold assignments saved by Phase 6.1.

Inner folds:
    4-fold GroupKFold, grouped by tube_id.

Training data only:
    666 images, 31 physical tubes.

Models:
    B0
    B1
    Scratch Ridge on S2
    Degree-2 Polynomial Ridge on b_med
    RBF-SVR on S2
    Random Forest on S2
    Gradient Boosting on S2
    MLP on S2

Validation and test partitions are excluded.

Outputs:
    results/metrics.csv                     (append)
    results/model_ladder_oof.csv
    results/model_ladder_best_params.csv
    results/figures/phase6_2_model_ladder_predicted_vs_actual.png
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
)
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
PREVIOUS_OOF_PATH = ROOT / "results/walking_skeleton_oof.csv"

METRICS_PATH = ROOT / "results/metrics.csv"
OOF_PATH = ROOT / "results/model_ladder_oof.csv"
PARAMS_PATH = ROOT / "results/model_ladder_best_params.csv"

FIGURE_PATH = (
    ROOT
    / "results/figures/phase6_2_model_ladder_predicted_vs_actual.png"
)

OUTER_SPLITS = 5
INNER_SPLITS = 4
RANDOM_STATE = 42

S2_COLUMNS = [
    "L_mean", "a_mean", "b_mean",
    "L_med", "a_med", "b_med",
    "L_std", "a_std", "b_std",
]

B1_COLUMNS = ["b_med"]

ALPHA_GRID = [
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    abs_error = np.abs(error)

    low = y_true < 0.5
    medium = (y_true >= 0.5) & (y_true <= 1.0)
    high = y_true > 1.0

    def band_mae(mask):
        if not mask.any():
            return np.nan
        return float(abs_error[mask].mean())

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_pred_minus_actual": float(error.mean()),
        "within_plus_minus_0_1": float(
            np.mean(abs_error <= 0.1)
        ),
        "mae_low_actual_lt_0_5": band_mae(low),
        "mae_medium_0_5_to_1_0": band_mae(medium),
        "mae_high_actual_gt_1_0": band_mae(high),
    }


# ============================================================
# Reuse the exact Phase 6.1 fold assignments
# ============================================================

def load_outer_folds(train):
    """
    Read B1 fold assignments from the Phase 6.1 OOF file.

    This guarantees the model ladder uses the same outer folds
    as the walking skeleton instead of merely recreating them.
    """

    previous = pd.read_csv(PREVIOUS_OOF_PATH)

    required = {"image_id", "fold", "model"}

    if not required.issubset(previous.columns):
        raise ValueError(
            "Phase 6.1 OOF file lacks image_id, fold or model"
        )

    previous_b1 = previous[
        previous["model"].astype(str) == "B1"
    ].copy()

    if previous_b1.empty:
        raise ValueError("No B1 rows found in Phase 6.1 OOF file")

    if previous_b1["image_id"].duplicated().any():
        raise ValueError("Duplicate B1 image IDs in Phase 6.1 OOF")

    previous_b1["image_id"] = previous_b1["image_id"].astype(str)

    train_ids = set(train["image_id"].astype(str))
    old_ids = set(previous_b1["image_id"])

    if train_ids != old_ids:
        missing = train_ids - old_ids
        extra = old_ids - train_ids

        raise ValueError(
            "Training image IDs differ from Phase 6.1 folds. "
            f"Missing old folds: {len(missing)}, extra old IDs: {len(extra)}"
        )

    fold_map = previous_b1.set_index("image_id")["fold"].to_dict()

    folds = train["image_id"].astype(str).map(fold_map).to_numpy()

    if pd.isna(folds).any():
        raise ValueError("Some training rows have no saved fold")

    folds = folds.astype(int)

    unique_folds = sorted(np.unique(folds).tolist())

    if len(unique_folds) != OUTER_SPLITS:
        raise ValueError(
            f"Expected {OUTER_SPLITS} saved folds; found {unique_folds}"
        )

    # Convert the stored fold IDs into train/validation indices.
    splits = []

    y = train["conc"].to_numpy(dtype=float)
    groups = train["tube_id"].astype(str).to_numpy()

    for fold in unique_folds:
        valid_idx = np.where(folds == fold)[0]
        train_idx = np.where(folds != fold)[0]

        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])

        if not train_groups.isdisjoint(valid_groups):
            raise RuntimeError(
                f"Tube leakage in saved fold {fold}"
            )

        splits.append((train_idx, valid_idx))

    print("Reused Phase 6.1 saved fold assignments.")
    print("Outer folds:", len(splits))

    return splits, folds


# ============================================================
# Model definitions
# ============================================================

def build_models():
    """Return model specifications with compact nested grids."""

    median_imputer = lambda: SimpleImputer(
        strategy="median",
        keep_empty_features=True,
    )

    models = []

    # B0: mean of the training fold.
    models.append({
        "name": "B0",
        "feature_set": "training_mean",
        "columns": [],
        "kind": "fixed",
        "estimator": Pipeline([
            ("model", DummyRegressor(strategy="mean")),
        ]),
    })

    # B1: linear regression on median b* only.
    models.append({
        "name": "B1",
        "feature_set": "b_med",
        "columns": B1_COLUMNS,
        "kind": "fixed",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            ("model", LinearRegression()),
        ]),
    })

    # Ridge on S2: use our own implementation.
    models.append({
        "name": "Ridge_S2_scratch",
        "feature_set": "S2_colour_only",
        "columns": S2_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            ("model", ScratchRidgeRegressor(alpha=1.0)),
        ]),
        "param_grid": {
            "model__alpha": ALPHA_GRID,
        },
    })

    # Degree-2 polynomial Ridge on b_med.
    models.append({
        "name": "Polynomial_Ridge_b_med",
        "feature_set": "b_med_degree_2",
        "columns": B1_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
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
        "param_grid": {
            "model__alpha": ALPHA_GRID,
        },
    })

    # RBF-SVR on S2.
    models.append({
        "name": "SVR_RBF_S2",
        "feature_set": "S2_colour_only",
        "columns": S2_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            ("scaler", StandardScaler()),
            ("model", SVR(kernel="rbf")),
        ]),
        "param_grid": {
            "model__C": [1, 10, 30],
            "model__gamma": [0.03, 0.1, 0.3],
            "model__epsilon": [0.02, 0.1],
        },
    })

    # Random forest on S2.
    models.append({
        "name": "Random_Forest_S2",
        "feature_set": "S2_colour_only",
        "columns": S2_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),
        "param_grid": {
            "model__min_samples_leaf": [2, 5, 10],
        },
    })

    # Gradient boosting on S2.
    models.append({
        "name": "Gradient_Boosting_S2",
        "feature_set": "S2_colour_only",
        "columns": S2_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            (
                "model",
                GradientBoostingRegressor(
                    subsample=0.7,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "param_grid": {
            "model__max_depth": [2, 3],
            "model__learning_rate": [0.03, 0.1],
            "model__n_estimators": [100, 300],
        },
    })

    # MLP with one hidden layer on S2.
    models.append({
        "name": "MLP_S2",
        "feature_set": "S2_colour_only",
        "columns": S2_COLUMNS,
        "kind": "grid",
        "estimator": Pipeline([
            ("imputer", median_imputer()),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    solver="lbfgs",
                    max_iter=3000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
        "param_grid": {
            "model__hidden_layer_sizes": [(8,), (16,), (32,)],
            "model__alpha": [0.001, 0.01, 0.1],
        },
    })

    return models


# ============================================================
# One model across the five outer folds
# ============================================================

def run_one_model(spec, train, y, groups, splits, fold_ids):
    model_name = spec["name"]
    columns = spec["columns"]

    if model_name == "B0":
        X = np.zeros((len(train), 1), dtype=float)
    else:
        X = train[columns].copy()

    predictions = np.full(len(train), np.nan, dtype=float)

    best_param_rows = []

    for fold_number, (outer_train_idx, outer_valid_idx) in enumerate(
        splits,
        start=1,
    ):
        X_train = (
            X.iloc[outer_train_idx]
            if isinstance(X, pd.DataFrame)
            else X[outer_train_idx]
        )

        X_valid = (
            X.iloc[outer_valid_idx]
            if isinstance(X, pd.DataFrame)
            else X[outer_valid_idx]
        )

        y_train = y[outer_train_idx]
        groups_train = groups[outer_train_idx]

        if spec["kind"] == "fixed":
            fitted_model = clone(spec["estimator"])
            fitted_model.fit(X_train, y_train)
            best_params = {}
            inner_score = np.nan

        else:
            inner_cv = GroupKFold(n_splits=INNER_SPLITS)

            search = GridSearchCV(
                estimator=clone(spec["estimator"]),
                param_grid=spec["param_grid"],
                scoring="neg_mean_absolute_error",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )

            search.fit(
                X_train,
                y_train,
                groups=groups_train,
            )

            fitted_model = search.best_estimator_
            best_params = search.best_params_
            inner_score = float(search.best_score_)

        predictions[outer_valid_idx] = fitted_model.predict(X_valid)

        valid_tubes = sorted(
            set(groups[outer_valid_idx].tolist())
        )
        training_tubes = sorted(
            set(groups[outer_train_idx].tolist())
        )

        if not set(valid_tubes).isdisjoint(training_tubes):
            raise RuntimeError(
                f"Tube leakage for {model_name}, fold {fold_number}"
            )

        best_param_rows.append({
            "model": model_name,
            "outer_fold": fold_number,
            "n_train_tubes": len(training_tubes),
            "n_heldout_tubes": len(valid_tubes),
            "inner_best_neg_mae": inner_score,
            "best_params": json.dumps(best_params, sort_keys=True),
        })

        print(
            f"  {model_name}: outer fold {fold_number}/{OUTER_SPLITS} "
            f"complete; best params={best_params}"
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"{model_name}: missing outer OOF predictions"
        )

    metrics = calculate_metrics(y, predictions)

    oof = train[
        [
            "image_id",
            "tube_id",
            "instance_id",
            "role",
            "conc",
        ]
    ].copy()

    oof["fold"] = fold_ids
    oof["model"] = model_name
    oof["feature_set"] = spec["feature_set"]
    oof["feature_columns"] = ";".join(columns)
    oof["actual"] = y
    oof["predicted"] = predictions
    oof["absolute_error"] = np.abs(predictions - y)

    return oof, metrics, best_param_rows


# ============================================================
# Output utilities
# ============================================================

def append_metrics(rows):
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_metrics = pd.DataFrame(rows)

    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)
        combined = pd.concat(
            [old, new_metrics],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = new_metrics

    combined.to_csv(
        METRICS_PATH,
        index=False,
    )


def make_plot(oof):
    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    all_values = np.concatenate([
        oof["actual"].to_numpy(dtype=float),
        oof["predicted"].to_numpy(dtype=float),
    ])

    low = min(0.0, float(all_values.min()))
    high = max(2.0, float(all_values.max()))

    ax.plot(
        [low, high],
        [low, high],
        linestyle="--",
        label="Ideal",
    )

    for name, group in oof.groupby("model"):
        ax.scatter(
            group["actual"],
            group["predicted"],
            alpha=0.42,
            s=22,
            label=name,
        )

    ax.set_xlabel("Actual added urea (% w/v)")
    ax.set_ylabel("Predicted added urea (% w/v)")
    ax.set_title("Phase 6.2: Tube-grouped nested CV model ladder")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(
        FIGURE_PATH,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    print("PHASE 6.2: MODEL LADDER")
    print("=" * 72)

    dataset = pd.read_csv(DATASET_PATH)

    required = {
        "image_id",
        "tube_id",
        "role",
        "conc",
        *S2_COLUMNS,
        "b_med",
    }

    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Dataset missing columns: {sorted(missing)}"
        )

    # Use the training partition only.
    train = dataset[
        dataset["role"].astype(str) == "train"
    ].copy()

    train["conc"] = pd.to_numeric(
        train["conc"],
        errors="raise",
    )

    if train["image_id"].duplicated().any():
        raise ValueError("Duplicate training image_id")

    if train["tube_id"].isna().any():
        raise ValueError("Missing tube_id in training rows")

    y = train["conc"].to_numpy(dtype=float)
    groups = train["tube_id"].astype(str).to_numpy()

    if len(np.unique(groups)) != 31:
        raise ValueError(
            f"Expected 31 training tubes, found {len(np.unique(groups))}"
        )

    # Reuse exact Phase 6.1 folds.
    splits, fold_ids = load_outer_folds(train)

    print("\nDATA")
    print("-" * 72)
    print("Training images:", len(train))
    print("Training tubes:", len(np.unique(groups)))
    print("Validation used: no")
    print("Test used: no")
    print("Outer CV: 5 folds, exact Phase 6.1 assignments")
    print("Inner CV: 4-fold GroupKFold")
    print("S2 features:", S2_COLUMNS)

    specs = build_models()

    all_oof = []
    metrics_rows = []
    parameter_rows = []

    run_id = datetime.now().strftime(
        "phase6_2_%Y%m%d_%H%M%S"
    )

    for spec in specs:
        print(f"\nRUNNING {spec['name']}")
        print("-" * 72)

        oof, metrics, params = run_one_model(
            spec=spec,
            train=train,
            y=y,
            groups=groups,
            splits=splits,
            fold_ids=fold_ids,
        )

        all_oof.append(oof)
        parameter_rows.extend(params)

        metric_record = {
            "run_id": run_id,
            "date": "2026-11-17",
            "experiment": "phase6_2_model_ladder",
            "model": spec["name"],
            "feature_set": spec["feature_set"],
            "feature_columns": ";".join(spec["columns"]),
            "cv_method": "Nested GroupKFold",
            "outer_splits": OUTER_SPLITS,
            "inner_splits": (
                INNER_SPLITS if spec["kind"] == "grid" else 0
            ),
            "group_column": "tube_id",
            "evaluation_role": "train",
            "n_images": len(train),
            "n_tubes": len(np.unique(groups)),
        }

        metric_record.update(metrics)
        metrics_rows.append(metric_record)

        print("\nPooled outer OOF metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    oof_all = pd.concat(
        all_oof,
        ignore_index=True,
    )

    OOF_PATH.parent.mkdir(parents=True, exist_ok=True)

    oof_all.to_csv(
        OOF_PATH,
        index=False,
    )

    pd.DataFrame(parameter_rows).to_csv(
        PARAMS_PATH,
        index=False,
    )

    append_metrics(metrics_rows)
    make_plot(oof_all)

    summary = pd.DataFrame(metrics_rows)[
        [
            "model",
            "mae",
            "rmse",
            "r2",
            "bias_pred_minus_actual",
            "within_plus_minus_0_1",
        ]
    ].sort_values("mae")

    print("\nMODEL LADDER SUMMARY — SORTED BY MAE")
    print("=" * 72)
    print(summary.to_string(index=False))

    print("\nOUTPUTS")
    print("=" * 72)
    print("Metrics appended:", METRICS_PATH.relative_to(ROOT))
    print("OOF predictions:", OOF_PATH.relative_to(ROOT))
    print("Best parameters:", PARAMS_PATH.relative_to(ROOT))
    print("Plot:", FIGURE_PATH.relative_to(ROOT))

    print("\nAll models used the same saved Phase 6.1 outer folds.")
    print("Hyperparameter tuning used inner tube-grouped CV.")
    print("All learned preprocessing was fitted inside each fold.")
    print("Validation and test rows were not used.")


if __name__ == "__main__":
    main()
