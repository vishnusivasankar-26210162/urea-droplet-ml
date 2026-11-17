"""
Phase 6.7 — Final development refit.

Fits the frozen Phase 6.6 candidate on all 864 development images:
    666 training images
    198 validation images

The test partition is never loaded or used.

Run from project root:
    python src/run_phase6_7_final_refit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "data/processed/dataset.csv"
REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"
VALIDATION_METRICS_PATH = (
    ROOT / "results/phase6_6_validation_metrics.csv"
)

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "final_svr_s1_device_lamp.joblib"
CONFIG_PATH = REPORTS_DIR / "final_model_config.json"
AUDIT_PATH = RESULTS_DIR / "phase6_7_refit_data_audit.json"
MANIFEST_PATH = RESULTS_DIR / "phase6_7_refit_manifest.csv"
REPORT_PATH = REPORTS_DIR / "phase6_7_final_refit.md"

MODEL_NAME = "SVR_S1_device_lamp"

EXPECTED_ROLE_COUNTS = {
    "train": 666,
    "validation": 198,
}

EXPECTED_TUBE_COUNTS = {
    "train": 31,
    "validation": 10,
}

EXPECTED_FEATURES = [
    "L_mean",
    "a_mean",
    "b_mean",
    "device_dino",
    "device_phone",
    "lamp_off",
    "lamp_on",
]

FIXED_PARAMS = {
    "C": 10,
    "epsilon": 0.02,
    "gamma": 0.1,
    "kernel": "rbf",
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\\n",
        encoding="utf-8",
    )


# ------------------------------------------------------------
# Load and audit development data
# ------------------------------------------------------------

def load_development_data():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(DATASET_PATH)

    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(REGISTRY_PATH)

    if not VALIDATION_METRICS_PATH.exists():
        raise FileNotFoundError(
            "Phase 6.6 validation metrics are missing. "
            "Complete Phase 6.6 before final refitting."
        )

    df = pd.read_csv(DATASET_PATH)

    required = {
        "image_id",
        "tube_id",
        "role",
        "conc",
        "device_dino",
        "device_phone",
        "lamp_off",
        "lamp_on",
        *EXPECTED_FEATURES[:3],
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["role"] = (
        df["role"].astype(str).str.strip().str.lower()
    )

    # Never allow the test partition into this operation.
    roles = set(df["role"].dropna().unique())

    if "test" in roles:
        raise ValueError(
            "Test-role data found in dataset.csv. "
            "Aborting to preserve the locked test partition."
        )

    if roles != {"train", "validation"}:
        raise ValueError(
            f"Unexpected development roles: {sorted(roles)}"
        )

    df["image_id"] = df["image_id"].astype(str)
    df["tube_id"] = df["tube_id"].astype(str)
    df["conc"] = pd.to_numeric(df["conc"], errors="raise")

    if df["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id values found.")

    if df["image_id"].isna().any():
        raise ValueError("Missing image_id values found.")

    if df["tube_id"].isna().any():
        raise ValueError("Missing tube_id values found.")

    if df["conc"].isna().any():
        raise ValueError("Missing concentration targets found.")

    role_counts = (
        df["role"].value_counts().to_dict()
    )

    if role_counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(
            f"Unexpected role counts: {role_counts}"
        )

    tube_counts = (
        df.groupby("role")["tube_id"].nunique().to_dict()
    )

    if tube_counts != EXPECTED_TUBE_COUNTS:
        raise ValueError(
            f"Unexpected tube counts: {tube_counts}"
        )

    # Every tube must belong to exactly one role and one concentration.
    tube_audit = (
        df.groupby("tube_id")
        .agg(
            n_roles=("role", "nunique"),
            n_concentrations=("conc", "nunique"),
        )
    )

    if (tube_audit["n_roles"] != 1).any():
        raise ValueError("A tube crosses development partitions.")

    if (tube_audit["n_concentrations"] != 1).any():
        raise ValueError(
            "A tube maps to multiple concentration labels."
        )

    # Check the canonical Phase 5 feature registry.
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        registry = json.load(f)

    registered_features = (
        registry["S1"]["with_device_and_lamp"]
    )

    if registered_features != EXPECTED_FEATURES:
        raise ValueError(
            "The S1 + device/lamp feature definition changed.\\n"
            f"Expected: {EXPECTED_FEATURES}\\n"
            f"Found: {registered_features}"
        )

    X = (
        df[EXPECTED_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    y = df["conc"].to_numpy(dtype=float)

    # Stop if a feature has no observed values in the complete
    # development data.
    all_missing = X.columns[X.isna().all()].tolist()

    if all_missing:
        raise ValueError(
            f"Features entirely missing in development data: "
            f"{all_missing}"
        )

    missing_counts = (
        X.isna().sum().astype(int).to_dict()
    )

    audit = {
        "n_images": int(len(df)),
        "n_tubes": int(df["tube_id"].nunique()),
        "role_counts": {
            str(k): int(v)
            for k, v in role_counts.items()
        },
        "tube_counts": {
            str(k): int(v)
            for k, v in tube_counts.items()
        },
        "concentration_levels": sorted(
            float(x) for x in df["conc"].unique()
        ),
        "feature_columns": EXPECTED_FEATURES,
        "missing_values_per_feature": missing_counts,
        "n_rows_with_any_missing_selected_feature": int(
            X.isna().any(axis=1).sum()
        ),
        "test_partition_loaded": False,
    }

    manifest = df[
        ["image_id", "tube_id", "role", "conc"]
    ].copy()

    return df, X, y, audit, manifest


# ------------------------------------------------------------
# Verify Phase 6.6 model selection
# ------------------------------------------------------------

def load_validation_evidence():
    metrics = pd.read_csv(VALIDATION_METRICS_PATH)

    selected = metrics[
        metrics["model"].astype(str) == MODEL_NAME
    ].copy()

    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one Phase 6.6 row for {MODEL_NAME}; "
            f"found {len(selected)}."
        )

    row = selected.iloc[0]

    expected_mae = 0.198015

    if not np.isclose(
        float(row["mae"]),
        expected_mae,
        atol=1e-6,
    ):
        raise ValueError(
            "Validation MAE differs from the recorded Phase 6.6 "
            f"selection result: {row['mae']}"
        )

    if str(metrics.sort_values("mae").iloc[0]["model"]) != MODEL_NAME:
        raise ValueError(
            "The selected candidate is not the lowest-MAE candidate "
            "in the saved Phase 6.6 results."
        )

    return {
        "selected_model": MODEL_NAME,
        "validation_mae": float(row["mae"]),
        "validation_rmse": float(row["rmse"]),
        "validation_r2": float(row["r2"]),
        "validation_bias": float(
            row["bias_pred_minus_actual"]
        ),
        "validation_within_plus_minus_0_1": float(
            row["within_plus_minus_0_1"]
        ),
    }


# ------------------------------------------------------------
# Build and fit frozen final pipeline
# ------------------------------------------------------------

def build_final_pipeline():
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
        (
            "model",
            SVR(
                kernel=FIXED_PARAMS["kernel"],
                C=FIXED_PARAMS["C"],
                epsilon=FIXED_PARAMS["epsilon"],
                gamma=FIXED_PARAMS["gamma"],
            ),
        ),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing Phase 6.7 outputs intentionally.",
    )
    args = parser.parse_args()

    output_paths = [
        MODEL_PATH,
        CONFIG_PATH,
        AUDIT_PATH,
        MANIFEST_PATH,
        REPORT_PATH,
    ]

    existing = [p for p in output_paths if p.exists()]

    if existing and not args.force:
        raise FileExistsError(
            "Phase 6.7 outputs already exist. Refusing to overwrite:\\n"
            + "\\n".join(str(p) for p in existing)
            + "\\nUse --force only for an intentional refit."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Phase 6.7 — Final development refit")
    print("------------------------------------")

    df, X, y, audit, manifest = load_development_data()
    validation_evidence = load_validation_evidence()

    print(f"Development images: {len(df)}")
    print(f"Development tubes: {df['tube_id'].nunique()}")
    print("Role counts:", audit["role_counts"])
    print("Test partition: not loaded or used.")
    print("Selected model:", MODEL_NAME)
    print("Fixed parameters:", FIXED_PARAMS)

    # No hyperparameter tuning here. The candidate and parameters
    # were selected before this refit.
    pipeline = build_final_pipeline()
    pipeline.fit(X, y)

    # Save fitted estimator.
    joblib.dump(pipeline, MODEL_PATH)

    model_hash = sha256_file(MODEL_PATH)
    dataset_hash = sha256_file(DATASET_PATH)

    audit["dataset_sha256"] = dataset_hash
    audit["model_sha256"] = model_hash
    audit["fit_timestamp_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    write_json(AUDIT_PATH, audit)

    manifest.to_csv(MANIFEST_PATH, index=False)

    config = {
        "project": "urea-droplet-ml",
        "phase": "6.7",
        "model_name": MODEL_NAME,
        "task": "Regression: estimate added urea concentration",
        "target": {
            "column": "conc",
            "unit": "% w/v",
            "description": "Added urea concentration",
        },
        "features": EXPECTED_FEATURES,
        "feature_set": "S1_with_device_and_lamp",
        "pipeline": [
            {
                "step": "imputer",
                "method": "SimpleImputer",
                "strategy": "median",
                "keep_empty_features": True,
            },
            {
                "step": "scaler",
                "method": "StandardScaler",
            },
            {
                "step": "model",
                "method": "SVR",
                "kernel": "rbf",
                "C": FIXED_PARAMS["C"],
                "epsilon": FIXED_PARAMS["epsilon"],
                "gamma": FIXED_PARAMS["gamma"],
            },
        ],
        "fit_data": {
            "images": int(len(df)),
            "tubes": int(df["tube_id"].nunique()),
            "training_images": int(
                (df["role"] == "train").sum()
            ),
            "validation_images": int(
                (df["role"] == "validation").sum()
            ),
            "test_partition_accessed": False,
        },
        "selection_evidence": validation_evidence,
        "provenance": {
            "dataset_path": "data/processed/dataset.csv",
            "dataset_sha256": dataset_hash,
            "model_path": "models/final_svr_s1_device_lamp.joblib",
            "model_sha256": model_hash,
            "fit_timestamp_utc": audit["fit_timestamp_utc"],
        },
        "software_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "reproducibility": {
            "hyperparameters_frozen": True,
            "validation_used_for_hyperparameter_tuning": False,
            "test_partition_loaded": False,
            "fit_metrics_reported": False,
            "test_predictions_generated": False,
        },
    }

    write_json(CONFIG_PATH, config)

    report = f"""# Phase 6.7 — Final development refit

## Selected model

**{MODEL_NAME}**

RBF-SVR with S1 colour features and device/lamp indicators.

## Fixed hyperparameters

- C: {FIXED_PARAMS["C"]}
- epsilon: {FIXED_PARAMS["epsilon"]}
- gamma: {FIXED_PARAMS["gamma"]}
- kernel: {FIXED_PARAMS["kernel"]}

## Input features

{", ".join(EXPECTED_FEATURES)}

## Preprocessing

1. Median imputation.
2. StandardScaler.
3. RBF-SVR.

All preprocessing parameters were fitted on the combined development
partition by the saved pipeline.

## Refit data

- Training: 666 images from 31 tubes.
- Validation: 198 images from 10 tubes.
- Combined development data: {len(df)} images from
  {df["tube_id"].nunique()} tubes.

The selected model was refitted using all development images.
The validation-selected hyperparameters were kept fixed.

## Independent validation evidence

- MAE: {validation_evidence["validation_mae"]:.6f}
- RMSE: {validation_evidence["validation_rmse"]:.6f}
- R²: {validation_evidence["validation_r2"]:.6f}

These are independent validation metrics, not in-sample refit metrics.

## Artifacts

- Fitted model: `models/final_svr_s1_device_lamp.joblib`
- Configuration: `reports/final_model_config.json`
- Data audit: `results/phase6_7_refit_data_audit.json`
- Fit manifest: `results/phase6_7_refit_manifest.csv`

## Test isolation

The 200-image test partition was not loaded, transformed, predicted
or otherwise accessed. No final test metrics are reported here.

Phase 7 will perform the one-time final test evaluation.
"""

    REPORT_PATH.write_text(report, encoding="utf-8")

    print("\\nFinal development refit completed.")
    print("Model:", MODEL_PATH)
    print("Configuration:", CONFIG_PATH)
    print("Audit:", AUDIT_PATH)
    print("Manifest:", MANIFEST_PATH)
    print("Report:", REPORT_PATH)
    print("Model SHA256:", model_hash)
    print("\\nNo test predictions were generated.")


if __name__ == "__main__":
    main()
