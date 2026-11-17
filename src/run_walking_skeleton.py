
"""
Phase 6.1: Walking-skeleton ML experiment.

Models:
    B0: training-fold mean predictor
    B1: linear regression using b_med
    Ridge-S1: Ridge(alpha=1.0) using L_mean, a_mean, b_mean

Evaluation:
    Five-fold GroupKFold grouped by tube_id.
    Training partition only.

Outputs:
    results/metrics.csv                  (appended)
    results/walking_skeleton_oof.csv
    results/condition_tube_consistency.csv
    results/figures/phase6_1_walking_skeleton_predicted_vs_actual.png

Validation and test rows are not used for fitting or scoring.
No image files are opened by this script.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold

try:
    from src.model_pipelines import (
        make_b0_pipeline,
        make_b1_pipeline,
        make_ridge_pipeline,
    )
except ModuleNotFoundError:
    from model_pipelines import (
        make_b0_pipeline,
        make_b1_pipeline,
        make_ridge_pipeline,
    )


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
DATASET_PATH = ROOT / "data/processed/dataset.csv"

METRICS_PATH = ROOT / "results/metrics.csv"
OOF_PATH = ROOT / "results/walking_skeleton_oof.csv"
CONDITION_REPORT_PATH = (
    ROOT / "results/condition_tube_consistency.csv"
)
FIGURE_PATH = (
    ROOT
    / "results/figures/phase6_1_walking_skeleton_predicted_vs_actual.png"
)

N_SPLITS = 5
RIDGE_ALPHA = 1.0

S1_COLUMNS = ["L_mean", "a_mean", "b_mean"]


# ============================================================
# Tube and condition consistency
# ============================================================

def normalize_condition(device, lamp):
    device_text = str(device).strip().lower()
    lamp_text = str(lamp).strip().lower()

    if "dino" in device_text:
        device_name = "dino"
    elif (
        "phone" in device_text
        or "samsung" in device_text
        or "a31" in device_text
    ):
        device_name = "phone"
    else:
        raise ValueError(f"Unknown device value: {device!r}")

    if lamp_text in {"off", "false", "0"}:
        lamp_name = "off"
    elif lamp_text in {"on", "true", "1"}:
        lamp_name = "on"
    else:
        raise ValueError(f"Unknown lamp value: {lamp!r}")

    return f"{device_name}_{lamp_name}"


def normalized_instance(value):
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return str(value)


def validate_tube_identity(metadata):
    """
    Verify deterministic tube identity and split isolation.

    The same (instance_id, concentration) must map to exactly
    one tube_id. A tube_id must map to one instance, concentration
    and role.
    """

    required = {
        "image_id",
        "instance_id",
        "tube_id",
        "conc",
        "role",
        "device",
        "lamp",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata missing required columns: {sorted(missing)}"
        )

    if metadata["tube_id"].isna().any():
        raise ValueError("Some metadata rows have missing tube_id")

    if metadata["conc"].isna().any():
        raise ValueError("Some metadata rows have missing concentration")

    metadata = metadata.copy()
    metadata["conc"] = pd.to_numeric(
        metadata["conc"],
        errors="raise",
    )
    metadata["instance_id"] = (
        metadata["instance_id"].map(normalized_instance)
    )

    # One physical tube for each instance/concentration pair.
    mapping_counts = (
        metadata.groupby(
            ["instance_id", "conc"],
            dropna=False,
        )["tube_id"]
        .nunique()
    )

    inconsistent = mapping_counts[mapping_counts != 1]

    if not inconsistent.empty:
        raise ValueError(
            "An instance/concentration maps to multiple tube IDs:\n"
            f"{inconsistent}"
        )

    # A tube cannot belong to multiple instances, concentrations or roles.
    for column in ("instance_id", "conc", "role"):
        counts = metadata.groupby("tube_id")[column].nunique()
        bad = counts[counts != 1]

        if not bad.empty:
            raise ValueError(
                f"Some tube IDs map to multiple {column} values:\n"
                f"{bad}"
            )

    metadata["condition"] = [
        normalize_condition(device, lamp)
        for device, lamp in zip(
            metadata["device"],
            metadata["lamp"],
        )
    ]

    expected_by_instance = {
        "1": {"dino_off", "dino_on", "phone_off", "phone_on"},
        "2": {"dino_off", "dino_on", "phone_off", "phone_on"},
        "3": {"dino_off", "dino_on"},
    }

    rows = []

    for tube_id, group in metadata.groupby("tube_id"):
        instance = str(group["instance_id"].iloc[0])
        observed = set(group["condition"].unique())

        expected = expected_by_instance.get(
            instance,
            {"dino_off", "dino_on", "phone_off", "phone_on"},
        )

        missing_conditions = sorted(expected - observed)

        rows.append({
            "tube_id": str(tube_id),
            "instance_id": instance,
            "conc": float(group["conc"].iloc[0]),
            "role": str(group["role"].iloc[0]),
            "observed_conditions": ";".join(sorted(observed)),
            "expected_conditions": ";".join(sorted(expected)),
            "missing_conditions": ";".join(missing_conditions),
            "n_conditions_observed": len(observed),
            "condition_complete": len(missing_conditions) == 0,
        })

    coverage = pd.DataFrame(rows).sort_values(
        ["instance_id", "conc"]
    )

    print("\nTUBE IDENTITY CHECK")
    print("=" * 64)
    print("Physical tubes checked:", metadata["tube_id"].nunique())
    print("Instance/concentration mappings: consistent")
    print("Tube-to-instance mappings: consistent")
    print("Tube-to-concentration mappings: consistent")
    print("Tube-to-role mappings: consistent")

    print("\nCONDITION COVERAGE")
    print("-" * 64)

    summary = (
        coverage.groupby("instance_id")
        .agg(
            tubes=("tube_id", "count"),
            complete=("condition_complete", "sum"),
        )
    )

    summary["incomplete"] = summary["tubes"] - summary["complete"]

    print(summary.to_string())

    incomplete = coverage[
        ~coverage["condition_complete"]
    ]

    if not incomplete.empty:
        print("\nWARNING: Some tubes lack one or more expected conditions.")
        print(
            incomplete[
                [
                    "tube_id",
                    "instance_id",
                    "observed_conditions",
                    "missing_conditions",
                ]
            ].to_string(index=False)
        )
    else:
        print("\nAll tubes have their expected acquisition conditions.")

    CONDITION_REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    coverage.to_csv(
        CONDITION_REPORT_PATH,
        index=False,
    )

    print(
        "\nCondition report:",
        CONDITION_REPORT_PATH.relative_to(ROOT),
    )

    return metadata


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    low = y_true < 0.5
    medium = (y_true >= 0.5) & (y_true <= 1.0)
    high = y_true > 1.0

    def band_mae(mask):
        if not np.any(mask):
            return np.nan
        return float(np.mean(absolute_error[mask]))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_pred_minus_actual": float(np.mean(error)),
        "within_plus_minus_0_1": float(
            np.mean(absolute_error <= 0.1)
        ),
        "mae_low_actual_lt_0_5": band_mae(low),
        "mae_medium_0_5_to_1_0": band_mae(medium),
        "mae_high_actual_gt_1_0": band_mae(high),
    }


# ============================================================
# Grouped out-of-fold predictions
# ============================================================

def make_oof_predictions(
    estimator,
    X,
    y,
    groups,
    base_rows,
    model_name,
    feature_set,
    feature_columns,
    cv_splits,
):
    """
    Fit a fresh cloned estimator in every fold.

    Returns one out-of-fold prediction per image.
    """

    predictions = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold_number, (train_idx, valid_idx) in enumerate(
        cv_splits,
        start=1,
    ):
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])

        if not train_groups.isdisjoint(valid_groups):
            raise RuntimeError(
                f"Tube leakage detected in fold {fold_number}"
            )

        model = clone(estimator)

        if isinstance(X, pd.DataFrame):
            X_train = X.iloc[train_idx]
            X_valid = X.iloc[valid_idx]
        else:
            X_train = X[train_idx]
            X_valid = X[valid_idx]

        model.fit(X_train, y[train_idx])

        fold_predictions = model.predict(X_valid)

        predictions[valid_idx] = fold_predictions
        fold_ids[valid_idx] = fold_number

        print(
            f"  {model_name}: fold {fold_number}/{N_SPLITS} "
            f"({len(train_groups)} train tubes, "
            f"{len(valid_groups)} held-out tubes)"
        )

    if np.isnan(predictions).any():
        raise RuntimeError(
            f"{model_name}: some images lack out-of-fold predictions"
        )

    if np.any(fold_ids < 1):
        raise RuntimeError(
            f"{model_name}: some images lack fold assignments"
        )

    output = base_rows.copy()
    output["fold"] = fold_ids
    output["model"] = model_name
    output["feature_set"] = feature_set
    output["feature_columns"] = ";".join(feature_columns)
    output["actual"] = y
    output["predicted"] = predictions
    output["absolute_error"] = np.abs(predictions - y)

    metrics = calculate_metrics(y, predictions)

    return output, metrics


# ============================================================
# Save metrics and plot
# ============================================================

def append_metrics(new_results):
    METRICS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_frame = pd.DataFrame(new_results)

    if METRICS_PATH.exists():
        old_frame = pd.read_csv(METRICS_PATH)

        combined = pd.concat(
            [old_frame, new_frame],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = new_frame

    combined.to_csv(
        METRICS_PATH,
        index=False,
    )


def make_predicted_vs_actual_plot(oof):
    FIGURE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    all_values = np.concatenate([
        oof["actual"].to_numpy(dtype=float),
        oof["predicted"].to_numpy(dtype=float),
    ])

    lower = min(0.0, float(np.min(all_values)))
    upper = max(2.0, float(np.max(all_values)))

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        label="Ideal prediction",
    )

    for model_name, group in oof.groupby("model"):
        ax.scatter(
            group["actual"],
            group["predicted"],
            alpha=0.55,
            label=model_name,
        )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)

    ax.set_xlabel("Actual added urea (% w/v)")
    ax.set_ylabel("Predicted added urea (% w/v)")
    ax.set_title(
        "Walking skeleton: tube-grouped out-of-fold predictions"
    )

    ax.grid(True, alpha=0.25)
    ax.legend()
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
    print("PHASE 6.1: WALKING SKELETON")
    print("=" * 64)

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(METADATA_PATH)

    if not DATASET_PATH.is_file():
        raise FileNotFoundError(DATASET_PATH)

    # Read metadata for tube consistency checks.
    # No image files are opened here.
    metadata = pd.read_csv(METADATA_PATH)

    validate_tube_identity(metadata)

    # Load the development feature table.
    dataset = pd.read_csv(DATASET_PATH)

    required = {
        "image_id",
        "tube_id",
        "role",
        "conc",
        "instance_id",
        "device",
        "lamp",
        "b_med",
        "L_mean",
        "a_mean",
        "b_mean",
    }

    missing = required - set(dataset.columns)

    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    # Use training rows only. Validation and test stay untouched.
    train = dataset[
        dataset["role"].astype(str) == "train"
    ].copy()

    if train.empty:
        raise ValueError("No training rows found")

    if train["tube_id"].isna().any():
        raise ValueError("Training rows contain missing tube_id")

    if train["conc"].isna().any():
        raise ValueError("Training rows contain missing concentration")

    if train["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in training data")

    train["conc"] = pd.to_numeric(
        train["conc"],
        errors="raise",
    )

    # Every tube must have one concentration.
    target_counts = train.groupby("tube_id")["conc"].nunique()

    if (target_counts != 1).any():
        raise ValueError(
            "At least one training tube has multiple concentration labels"
        )

    y = train["conc"].to_numpy(dtype=float)
    groups = train["tube_id"].astype(str).to_numpy()

    n_tubes = len(np.unique(groups))

    if n_tubes < N_SPLITS:
        raise ValueError(
            f"Need at least {N_SPLITS} training tubes; found {n_tubes}"
        )

    print("\nCROSS-VALIDATION SETUP")
    print("-" * 64)
    print("Training images:", len(train))
    print("Training tubes:", n_tubes)
    print("Validation rows used: 0")
    print("Test rows used: 0")
    print("CV:", f"{N_SPLITS}-fold GroupKFold")
    print("Grouping variable: tube_id")

    # Same outer folds for all three models.
    splitter = GroupKFold(n_splits=N_SPLITS)

    cv_splits = list(
        splitter.split(
            np.zeros((len(train), 1)),
            y,
            groups,
        )
    )

    base_rows = train[
        [
            "image_id",
            "tube_id",
            "instance_id",
            "role",
            "conc",
        ]
    ].copy()

    # B0 ignores the dummy input and predicts the training-fold mean.
    X_b0 = np.zeros((len(train), 1), dtype=float)

    # B1 uses median b* only.
    X_b1 = train[["b_med"]].copy()

    # Ridge-S1 uses the three masked CIELAB means.
    X_ridge = train[S1_COLUMNS].copy()

    model_specs = [
        {
            "name": "B0",
            "feature_set": "B0_training_mean",
            "feature_columns": [],
            "estimator": make_b0_pipeline(),
            "X": X_b0,
        },
        {
            "name": "B1",
            "feature_set": "S1_b_med_only",
            "feature_columns": ["b_med"],
            "estimator": make_b1_pipeline(),
            "X": X_b1,
        },
        {
            "name": "Ridge_S1",
            "feature_set": "S1_colour_only",
            "feature_columns": S1_COLUMNS,
            "estimator": make_ridge_pipeline(
                alpha=RIDGE_ALPHA
            ),
            "X": X_ridge,
        },
    ]

    all_oof = []
    metric_records = []

    run_id = datetime.now().strftime(
        "phase6_1_%Y%m%d_%H%M%S"
    )

    for spec in model_specs:
        print(f"\nRUNNING {spec['name']}")
        print("-" * 64)

        oof, metrics = make_oof_predictions(
            estimator=spec["estimator"],
            X=spec["X"],
            y=y,
            groups=groups,
            base_rows=base_rows,
            model_name=spec["name"],
            feature_set=spec["feature_set"],
            feature_columns=spec["feature_columns"],
            cv_splits=cv_splits,
        )

        all_oof.append(oof)

        metric_record = {
            "run_id": run_id,
            "date": "2026-11-17",
            "experiment": "phase6_1_walking_skeleton",
            "model": spec["name"],
            "feature_set": spec["feature_set"],
            "feature_columns": ";".join(
                spec["feature_columns"]
            ),
            "cv_method": "GroupKFold",
            "n_splits": N_SPLITS,
            "group_column": "tube_id",
            "evaluation_role": "train",
            "n_images": len(train),
            "n_tubes": n_tubes,
            "ridge_alpha": (
                RIDGE_ALPHA
                if spec["name"] == "Ridge_S1"
                else np.nan
            ),
        }

        metric_record.update(metrics)
        metric_records.append(metric_record)

        print(
            f"\n{spec['name']} OOF metrics:"
        )

        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")

    oof_all = pd.concat(
        all_oof,
        ignore_index=True,
    )

    oof_all.to_csv(
        OOF_PATH,
        index=False,
    )

    append_metrics(metric_records)
    make_predicted_vs_actual_plot(oof_all)

    print("\nPHASE 6.1 OUTPUTS")
    print("=" * 64)
    print("Metrics appended:", METRICS_PATH.relative_to(ROOT))
    print("OOF predictions:", OOF_PATH.relative_to(ROOT))
    print("Condition report:", CONDITION_REPORT_PATH.relative_to(ROOT))
    print("Plot:", FIGURE_PATH.relative_to(ROOT))

    print("\nMetrics summary:")
    print(
        pd.DataFrame(metric_records)[
            [
                "model",
                "mae",
                "rmse",
                "r2",
                "bias_pred_minus_actual",
                "within_plus_minus_0_1",
            ]
        ].to_string(index=False)
    )

    print("\nAll three models used identical tube-grouped folds.")
    print("All preprocessing was fitted inside each training fold.")
    print("Validation and test data were not used for model fitting or scoring.")


if __name__ == "__main__":
    main()
