"""
Phase 6.6 — Independent validation of shortlisted models.

Training:
    666 images, 31 physical tubes.
    Used for GroupKFold hyperparameter selection and model fitting.

Validation:
    198 images, 10 physical tubes.
    Used once for candidate comparison and selection.

Test:
    Not loaded or used.

Run from project root:
    python src/run_phase6_6_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data/processed/dataset.csv"
REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"

RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

METRICS_PATH = RESULTS_DIR / "metrics.csv"
VALIDATION_METRICS_PATH = (
    RESULTS_DIR / "phase6_6_validation_metrics.csv"
)
PREDICTIONS_PATH = (
    RESULTS_DIR / "phase6_6_validation_predictions.csv"
)
TUBE_METRICS_PATH = (
    RESULTS_DIR / "phase6_6_validation_tube_metrics.csv"
)
PARAMS_PATH = (
    RESULTS_DIR / "phase6_6_validation_best_params.csv"
)
FIGURE_PATH = (
    RESULTS_DIR / "figures/phase6_6_validation_comparison.png"
)
REPORT_PATH = REPORTS_DIR / "phase6_6_validation.md"

RANDOM_STATE = 42
INNER_SPLITS = 4
EXPERIMENT_NAME = "Phase 6.6 validation"

S2_COLUMNS = [
    "L_mean", "a_mean", "b_mean",
    "L_med", "a_med", "b_med",
    "L_std", "a_std", "b_std",
]


# ------------------------------------------------------------
# Data loading and integrity checks
# ------------------------------------------------------------

def load_data():
    """Load only the development dataset and separate train/validation."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    df = pd.read_csv(DATASET_PATH)

    required = {
        "image_id",
        "tube_id",
        "role",
        "conc",
        "feature_status",
        "device_dino",
        "device_phone",
        "lamp_off",
        "lamp_on",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset missing required columns: {sorted(missing)}"
        )

    df["role"] = df["role"].astype(str).str.strip().str.lower()

    # The Phase 5 dataset should contain only development rows.
    # Stop if test rows are present rather than accidentally using them.
    roles = set(df["role"].dropna().unique())
    if "test" in roles:
        raise ValueError(
            "Test-role rows detected in dataset.csv. "
            "Stop: the test partition must remain locked."
        )

    unexpected = roles - {"train", "validation"}
    if unexpected:
        raise ValueError(f"Unexpected roles in dataset: {unexpected}")

    df["image_id"] = df["image_id"].astype(str)
    df["tube_id"] = df["tube_id"].astype(str)
    df["conc"] = pd.to_numeric(df["conc"], errors="raise")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values in dataset.")

    if df["tube_id"].isna().any():
        raise ValueError("Missing tube_id values in dataset.")

    train = df[df["role"] == "train"].copy().reset_index(drop=True)
    validation = (
        df[df["role"] == "validation"].copy().reset_index(drop=True)
    )

    if train.empty or validation.empty:
        raise ValueError("Training or validation partition is empty.")

    train_tubes = set(train["tube_id"])
    validation_tubes = set(validation["tube_id"])

    if not train_tubes.isdisjoint(validation_tubes):
        raise ValueError("Tube leakage between training and validation.")

    # Expected split sizes. These checks protect the pre-existing split.
    expected = {
        "train_images": 666,
        "validation_images": 198,
        "train_tubes": 31,
        "validation_tubes": 10,
    }

    observed = {
        "train_images": len(train),
        "validation_images": len(validation),
        "train_tubes": train["tube_id"].nunique(),
        "validation_tubes": validation["tube_id"].nunique(),
    }

    if observed != expected:
        raise ValueError(
            "Unexpected split structure.\\n"
            f"Expected: {expected}\\n"
            f"Observed: {observed}"
        )

    return train, validation


def load_feature_columns():
    """Read the canonical Phase 5 feature registry."""

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    svr_s1 = registry["S1"]["with_device_and_lamp"]
    rf_s2 = registry["S2"]["colour_only"]

    expected_svr_s1 = [
        "L_mean",
        "a_mean",
        "b_mean",
        "device_dino",
        "device_phone",
        "lamp_off",
        "lamp_on",
    ]

    if svr_s1 != expected_svr_s1:
        raise ValueError(
            "Unexpected S1 + device/lamp feature definition.\\n"
            f"Expected: {expected_svr_s1}\\n"
            f"Found: {svr_s1}"
        )

    if rf_s2 != S2_COLUMNS:
        raise ValueError(
            "Unexpected S2 colour-only feature definition.\\n"
            f"Expected: {S2_COLUMNS}\\n"
            f"Found: {rf_s2}"
        )

    return svr_s1, rf_s2


def get_feature_matrix(df, columns):
    """Select numeric model inputs without fitting any transformation."""

    missing = set(columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing feature columns: {sorted(missing)}")

    X = df[columns].apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


# ------------------------------------------------------------
# Candidate pipelines and training-only parameter grids
# ------------------------------------------------------------

def make_svr_pipeline():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
        ("model", SVR(kernel="rbf")),
    ])


def make_rf_pipeline():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])


def make_gb_pipeline():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        (
            "model",
            GradientBoostingRegressor(
                subsample=0.7,
                random_state=RANDOM_STATE,
            ),
        ),
    ])


def build_candidates(svr_s1_columns, s2_columns):
    """
    Candidate models and compact hyperparameter grids.

    All preprocessing is contained inside each pipeline.
    Grid search sees training data only.
    """

    svr_grid = {
        "model__C": [1, 10, 30],
        "model__gamma": [0.03, 0.1, 0.3],
        "model__epsilon": [0.02, 0.1],
    }

    return [
        {
            "name": "SVR_S1_device_lamp",
            "feature_set": "S1_with_device_and_lamp",
            "columns": svr_s1_columns,
            "estimator": make_svr_pipeline(),
            "param_grid": svr_grid,
        },
        {
            "name": "Random_Forest_S2",
            "feature_set": "S2_colour_only",
            "columns": s2_columns,
            "estimator": make_rf_pipeline(),
            "param_grid": {
                "model__min_samples_leaf": [2, 5, 10],
            },
        },
        {
            "name": "Gradient_Boosting_S2",
            "feature_set": "S2_colour_only",
            "columns": s2_columns,
            "estimator": make_gb_pipeline(),
            "param_grid": {
                "model__max_depth": [2, 3],
                "model__learning_rate": [0.03, 0.1],
                "model__n_estimators": [100, 300],
            },
        },
        {
            "name": "SVR_S2",
            "feature_set": "S2_colour_only",
            "columns": s2_columns,
            "estimator": make_svr_pipeline(),
            "param_grid": svr_grid,
        },
    ]


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    abs_error = np.abs(error)

    low = y_true < 0.5
    medium = (y_true >= 0.5) & (y_true <= 1.0)
    high = y_true > 1.0

    def band_mae(mask):
        if not np.any(mask):
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


# ------------------------------------------------------------
# Fit and evaluate
# ------------------------------------------------------------

def fit_and_validate(candidate, train, validation):
    name = candidate["name"]
    columns = candidate["columns"]

    X_train = get_feature_matrix(train, columns)
    X_validation = get_feature_matrix(validation, columns)

    y_train = train["conc"].to_numpy(dtype=float)
    y_validation = validation["conc"].to_numpy(dtype=float)

    groups = train["tube_id"].astype(str).to_numpy()

    if len(np.unique(groups)) < INNER_SPLITS:
        raise ValueError(
            f"Not enough training tubes for {INNER_SPLITS}-fold CV."
        )

    cv = GroupKFold(n_splits=INNER_SPLITS)

    search = GridSearchCV(
        estimator=candidate["estimator"],
        param_grid=candidate["param_grid"],
        scoring="neg_mean_absolute_error",
        cv=cv,
        n_jobs=1,
        refit=True,
        return_train_score=False,
        error_score="raise",
    )

    # Hyperparameter selection uses training images and tube groups only.
    search.fit(X_train, y_train, groups=groups)

    # GridSearchCV(refit=True) has already refitted the best estimator
    # on all 666 training images.
    best_model = search.best_estimator_

    # Independent validation prediction.
    y_pred = best_model.predict(X_validation)

    metrics = calculate_metrics(y_validation, y_pred)

    metrics_row = {
        "experiment": EXPERIMENT_NAME,
        "model": name,
        "feature_set": candidate["feature_set"],
        "split": "validation",
        "n_train_images": len(train),
        "n_train_tubes": train["tube_id"].nunique(),
        "n_validation_images": len(validation),
        "n_validation_tubes": validation["tube_id"].nunique(),
        "inner_cv_folds": INNER_SPLITS,
        "inner_best_cv_mae": float(-search.best_score_),
        **metrics,
    }

    prediction_df = validation[
        ["image_id", "tube_id", "conc", "device", "lamp"]
    ].copy()

    prediction_df = prediction_df.rename(
        columns={"conc": "actual"}
    )

    prediction_df["model"] = name
    prediction_df["feature_set"] = candidate["feature_set"]
    prediction_df["predicted"] = y_pred
    prediction_df["residual_pred_minus_actual"] = (
        y_pred - y_validation
    )
    prediction_df["absolute_error"] = np.abs(
        y_pred - y_validation
    )

    parameter_row = {
        "model": name,
        "feature_set": candidate["feature_set"],
        "best_inner_cv_mae": float(-search.best_score_),
        "best_params": json.dumps(
            search.best_params_,
            sort_keys=True,
        ),
    }

    print(f"\\n{name}")
    print(f"  Best inner-CV MAE: {-search.best_score_:.6f}")
    print(f"  Best parameters: {search.best_params_}")
    print(f"  Validation MAE: {metrics['mae']:.6f}")
    print(f"  Validation RMSE: {metrics['rmse']:.6f}")
    print(f"  Validation R2: {metrics['r2']:.6f}")

    return metrics_row, prediction_df, parameter_row


# ------------------------------------------------------------
# Tube-level supporting summary
# ------------------------------------------------------------

def make_tube_metrics(predictions):
    rows = []

    for model_name, part in predictions.groupby("model"):
        for tube_id, tube in part.groupby("tube_id"):
            errors = (
                tube["predicted"].to_numpy(dtype=float)
                - tube["actual"].to_numpy(dtype=float)
            )

            rows.append({
                "model": model_name,
                "tube_id": tube_id,
                "concentration": float(tube["actual"].iloc[0]),
                "n_images": len(tube),
                "mae": float(np.mean(np.abs(errors))),
                "rmse": float(np.sqrt(np.mean(errors ** 2))),
                "bias_pred_minus_actual": float(np.mean(errors)),
            })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Results, plot and report
# ------------------------------------------------------------

def make_comparison_plot(metrics_df, predictions):
    RESULTS_DIR.joinpath("figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    model_order = metrics_df.sort_values("mae")["model"].tolist()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 10),
        constrained_layout=True,
    )

    axes = axes.ravel()

    for ax, model_name in zip(axes, model_order):
        part = predictions[predictions["model"] == model_name]

        ax.scatter(
            part["actual"],
            part["predicted"],
            alpha=0.75,
        )

        ax.plot([0, 2], [0, 2], linestyle="--")

        ax.set_xlim(0, 2.05)
        ax.set_ylim(0, 2.05)
        ax.set_xlabel("Actual added urea (% w/v)")
        ax.set_ylabel("Predicted added urea (% w/v)")
        ax.set_title(
            f"{model_name}\\n"
            f"Validation MAE = "
            f"{metrics_df.loc[metrics_df['model'] == model_name, 'mae'].iloc[0]:.3f}"
        )
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Phase 6.6 — Independent validation",
        fontsize=14,
    )

    fig.savefig(FIGURE_PATH, dpi=180)
    plt.close(fig)


def write_report(metrics_df, params_df, train, validation, selected):
    lines = [
        "# Phase 6.6 — Independent validation",
        "",
        "## Evaluation protocol",
        "",
        f"- Training: {len(train)} images, "
        f"{train['tube_id'].nunique()} tubes.",
        f"- Validation: {len(validation)} images, "
        f"{validation['tube_id'].nunique()} tubes.",
        "- Test partition: not accessed.",
        "- Hyperparameter selection: four-fold GroupKFold on training tubes.",
        "- Primary selection metric: validation MAE.",
        "",
        "## Validation results",
        "",
        "| Model | MAE | RMSE | R2 | Bias | Within ±0.1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for _, row in metrics_df.sort_values("mae").iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['mae']:.6f} "
            f"| {row['rmse']:.6f} "
            f"| {row['r2']:.6f} "
            f"| {row['bias_pred_minus_actual']:.6f} "
            f"| {row['within_plus_minus_0_1']:.4f} |"
        )

    selected_row = metrics_df[
        metrics_df["model"] == selected
    ].iloc[0]

    lines.extend([
        "",
        "## Selected candidate",
        "",
        f"**{selected}** was selected because it achieved the lowest "
        "validation MAE among the four predeclared candidates.",
        "",
        f"- Validation MAE: {selected_row['mae']:.6f}",
        f"- Validation RMSE: {selected_row['rmse']:.6f}",
        f"- Validation R2: {selected_row['r2']:.6f}",
        f"- Validation bias: "
        f"{selected_row['bias_pred_minus_actual']:.6f}",
        "",
        "The validation set was used for candidate selection, not for "
        "hyperparameter tuning.",
        "",
        "## Training-only hyperparameter selection",
        "",
        "| Model | Best inner-CV MAE | Selected parameters |",
        "|---|---:|---|",
    ])

    for _, row in params_df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['best_inner_cv_mae']:.6f} "
            f"| `{row['best_params']}` |"
        )

    lines.extend([
        "",
        "## Next step",
        "",
        "Refit the selected pipeline on all 864 development images "
        "(training plus validation), with the selected hyperparameters "
        "fixed. Keep the 200-image test partition locked until Phase 7.",
        "",
    ])

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\\n".join(lines), encoding="utf-8")


def update_main_metrics(metrics_df, force=False):
    """
    Append Phase 6.6 rows to results/metrics.csv while preserving
    existing experiment records and any existing columns.
    """

    new_rows = metrics_df.copy()

    if METRICS_PATH.exists():
        old = pd.read_csv(METRICS_PATH)

        if "experiment" in old.columns:
            old = old[
                old["experiment"].astype(str) != EXPERIMENT_NAME
            ].copy()

        if not force and (
            VALIDATION_METRICS_PATH.exists()
        ):
            raise FileExistsError(
                f"{VALIDATION_METRICS_PATH} already exists. "
                "Refusing to rerun Phase 6.6 accidentally. "
                "Use --force only if you intentionally want to rerun."
            )

        combined = pd.concat(
            [old, new_rows],
            ignore_index=True,
            sort=False,
        )
    else:
        combined = new_rows

    combined.to_csv(METRICS_PATH, index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Phase 6.6 outputs intentionally.",
    )
    args = parser.parse_args()

    output_paths = [
        VALIDATION_METRICS_PATH,
        PREDICTIONS_PATH,
        TUBE_METRICS_PATH,
        PARAMS_PATH,
        REPORT_PATH,
        FIGURE_PATH,
    ]

    existing = [p for p in output_paths if p.exists()]
    if existing and not args.force:
        raise FileExistsError(
            "Phase 6.6 output already exists. "
            "Refusing to rerun without --force:\\n"
            + "\\n".join(str(p) for p in existing)
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train, validation = load_data()
    svr_s1_columns, s2_columns = load_feature_columns()

    candidates = build_candidates(
        svr_s1_columns,
        s2_columns,
    )

    print("Phase 6.6 — Independent validation")
    print("----------------------------------")
    print(
        f"Training: {len(train)} images, "
        f"{train['tube_id'].nunique()} tubes"
    )
    print(
        f"Validation: {len(validation)} images, "
        f"{validation['tube_id'].nunique()} tubes"
    )
    print("Test partition: not loaded or used.")

    metric_rows = []
    prediction_frames = []
    parameter_rows = []

    for candidate in candidates:
        metrics_row, prediction_df, parameter_row = (
            fit_and_validate(
                candidate,
                train,
                validation,
            )
        )

        metric_rows.append(metrics_row)
        prediction_frames.append(prediction_df)
        parameter_rows.append(parameter_row)

    metrics_df = pd.DataFrame(metric_rows)
    predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )
    params_df = pd.DataFrame(parameter_rows)

    # Primary selection rule: lowest validation MAE.
    metrics_df = metrics_df.sort_values(
        "mae",
        ascending=True,
    ).reset_index(drop=True)

    selected = str(metrics_df.iloc[0]["model"])

    tube_metrics = make_tube_metrics(predictions)

    metrics_df.to_csv(
        VALIDATION_METRICS_PATH,
        index=False,
    )
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )
    tube_metrics.to_csv(
        TUBE_METRICS_PATH,
        index=False,
    )
    params_df.to_csv(
        PARAMS_PATH,
        index=False,
    )

    make_comparison_plot(metrics_df, predictions)

    write_report(
        metrics_df,
        params_df,
        train,
        validation,
        selected,
    )

    update_main_metrics(metrics_df, force=args.force)

    print("\\n==============================")
    print("Validation ranking")
    print("==============================")
    print(
        metrics_df[
            ["model", "mae", "rmse", "r2",
             "bias_pred_minus_actual", "within_plus_minus_0_1"]
        ].to_string(index=False)
    )

    print("\\nSelected candidate:", selected)
    print("Validation metrics:", VALIDATION_METRICS_PATH)
    print("Predictions:", PREDICTIONS_PATH)
    print("Tube metrics:", TUBE_METRICS_PATH)
    print("Best parameters:", PARAMS_PATH)
    print("Figure:", FIGURE_PATH)
    print("Report:", REPORT_PATH)
    print("Updated main metrics:", METRICS_PATH)


if __name__ == "__main__":
    main()
