
"""Phase 7: one-shot final test evaluation.

Run from the project root:
    python src/run_phase7_final_test.py

The script intentionally refuses to run if its one-shot marker or
final test outputs already exist. Do not remove the marker to rerun.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image

from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
FEATURES_PATH = ROOT / "data/processed/features.csv"
DATASET_PATH = ROOT / "data/processed/dataset.csv"

CONFIG_PATH = ROOT / "reports/final_model_config.json"
MODEL_PATH = ROOT / "models/final_svr_s1_device_lamp.joblib"

RESULTS_DIR = ROOT / "results"
TEST_MASK_DIR = ROOT / "data/interim/test_roi_masks"
TEST_ERODED_DIR = ROOT / "data/interim/test_eroded_masks"
TEST_CLEANED_DIR = ROOT / "data/interim/test_cleaned_roi_rgba"

ACCESS_MARKER = RESULTS_DIR / "phase7_test_accessed.json"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
PROCESSING_PATH = RESULTS_DIR / "test_processing_manifest.csv"
METRICS_PATH = RESULTS_DIR / "test_results.csv"
BOOTSTRAP_PATH = RESULTS_DIR / "test_tube_bootstrap_samples.csv"
REPORT_PATH = ROOT / "reports/phase7_final_test.md"

EXPECTED_MODEL_SHA256 = (
    "5afb9b97a4bd73eb7fa569ffc2483ddc37646d86fb162277bc669ec4cf22adaa"
)
EXPECTED_DATASET_SHA256 = (
    "d88d8b2208e7d914b655b2f3a5cda4611d329590b21a1018bc6a706d634aaad8"
)

EXPECTED_DEV_IMAGES = 864
EXPECTED_TEST_IMAGES = 200
EXPECTED_DEV_TUBES = 41
EXPECTED_TEST_TUBES = 10

EROSION_FRACTION = 0.035
GLARE_VALUE_THRESHOLD = 0.92
GLARE_SATURATION_THRESHOLD = 0.25

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0


# These modules contain the already-frozen project implementations.
# Importing them does not execute their main() functions.
import make_roi_masks as SEG
import trim_roi_masks as TRIM
import extract_colour_features as FEAT
import build_feature_sets as FSET


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def require_columns(frame, columns, name):
    missing = set(columns) - set(frame.columns)

    if missing:
        raise ValueError(
            f"{name} is missing columns: {sorted(missing)}"
        )


def prepare_output_locations():
    """Reject an existing or partial prior Phase 7 run."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_files = [
        ACCESS_MARKER,
        PREDICTIONS_PATH,
        PROCESSING_PATH,
        METRICS_PATH,
        BOOTSTRAP_PATH,
        REPORT_PATH,
    ]

    existing = [str(path.relative_to(ROOT)) for path in output_files if path.exists()]

    if existing:
        raise FileExistsError(
            "A Phase 7 run or partial run already exists. "
            "The test partition must not be reopened. Existing outputs: "
            + ", ".join(existing)
        )

    test_dirs = [
        TEST_MASK_DIR,
        TEST_ERODED_DIR,
        TEST_CLEANED_DIR,
    ]

    for directory in test_dirs:
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(
                f"Test-specific directory is not empty: "
                f"{directory.relative_to(ROOT)}"
            )


def load_and_validate_inputs():
    """Validate split and frozen artifacts before opening test images."""

    for path in [
        METADATA_PATH,
        FEATURES_PATH,
        DATASET_PATH,
        CONFIG_PATH,
        MODEL_PATH,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    model_hash = sha256_file(MODEL_PATH)
    dataset_hash = sha256_file(DATASET_PATH)

    if model_hash != EXPECTED_MODEL_SHA256:
        raise ValueError(
            "Final model SHA256 differs from the frozen Phase 6.7 model."
        )

    if dataset_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "Development dataset SHA256 differs from the Phase 6.7 dataset."
        )

    with CONFIG_PATH.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if config.get("model_name") != "SVR_S1_device_lamp":
        raise ValueError("Unexpected model in final_model_config.json")

    if config.get("reproducibility", {}).get("test_partition_loaded") is not False:
        raise ValueError("Model config does not show a locked test partition.")

    if config.get("fit_data", {}).get("images") != EXPECTED_DEV_IMAGES:
        raise ValueError("Unexpected Phase 6.7 development image count.")

    if config.get("provenance", {}).get("model_sha256") != model_hash:
        raise ValueError("Model hash does not match the configuration.")

    if config.get("provenance", {}).get("dataset_sha256") != dataset_hash:
        raise ValueError("Dataset hash does not match the configuration.")

    # Check that the currently installed project preprocessing is the
    # frozen version recorded for Phase 7.
    if not np.isclose(TRIM.EROSION_FRACTION, EROSION_FRACTION):
        raise ValueError("Trimming module has a different erosion fraction.")

    if not np.isclose(
        TRIM.GLARE_VALUE_THRESHOLD,
        GLARE_VALUE_THRESHOLD,
    ):
        raise ValueError("Trimming module has a different glare V threshold.")

    if not np.isclose(
        TRIM.GLARE_SATURATION_THRESHOLD,
        GLARE_SATURATION_THRESHOLD,
    ):
        raise ValueError("Trimming module has a different glare S threshold.")

    metadata = pd.read_csv(METADATA_PATH)
    features = pd.read_csv(FEATURES_PATH)

    require_columns(
        metadata,
        [
            "image_id",
            "image_path",
            "role",
            "conc",
            "tube_id",
            "instance_id",
            "device",
            "lamp",
        ],
        "metadata.csv",
    )
    require_columns(
        features,
        ["image_id", "b_med"],
        "features.csv",
    )

    metadata["image_id"] = metadata["image_id"].astype(str)
    features["image_id"] = features["image_id"].astype(str)

    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in metadata.csv.")

    if features["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in features.csv.")

    dev = metadata[
        metadata["role"].astype(str).isin(["train", "validation"])
    ].copy()

    test = metadata[
        metadata["role"].astype(str).eq("test")
    ].copy()

    if len(dev) != EXPECTED_DEV_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_DEV_IMAGES} development rows; found {len(dev)}."
        )

    if len(test) != EXPECTED_TEST_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_TEST_IMAGES} test rows; found {len(test)}."
        )

    if dev["tube_id"].nunique() != EXPECTED_DEV_TUBES:
        raise ValueError("Unexpected development tube count.")

    if test["tube_id"].nunique() != EXPECTED_TEST_TUBES:
        raise ValueError("Unexpected test tube count.")

    dev_tubes = set(dev["tube_id"].astype(str))
    test_tubes = set(test["tube_id"].astype(str))

    if not dev_tubes.isdisjoint(test_tubes):
        raise ValueError("Tube leakage between development and test.")

    if set(dev["image_id"]) & set(test["image_id"]):
        raise ValueError("Image overlap between development and test.")

    # The development features are used only to fit B0 and B1.
    dev_baseline = dev[
        ["image_id", "conc", "tube_id"]
    ].merge(
        features[["image_id", "b_med"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if len(dev_baseline) != EXPECTED_DEV_IMAGES:
        raise ValueError("Development baseline join changed the row count.")

    if dev_baseline["conc"].isna().any():
        raise ValueError("Missing development concentration values.")

    # Verify that every test image path exists without opening image pixels.
    missing_paths = []

    for value in test["image_path"]:
        path = (ROOT / str(value)).resolve()

        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            missing_paths.append(str(value))

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} test image paths are missing or invalid. "
            f"Examples: {missing_paths[:5]}"
        )

    feature_columns = config.get("features")

    if feature_columns != [
        "L_mean",
        "a_mean",
        "b_mean",
        "device_dino",
        "device_phone",
        "lamp_off",
        "lamp_on",
    ]:
        raise ValueError("Unexpected frozen model feature list.")

    model = joblib.load(MODEL_PATH)

    return config, model, feature_columns, metadata, dev_baseline, test


def fit_development_baselines(dev_baseline):
    """Fit both baselines using all development rows only."""

    y_dev = dev_baseline["conc"].to_numpy(dtype=float)

    b0 = DummyRegressor(strategy="mean")
    b0.fit(np.zeros((len(y_dev), 1)), y_dev)

    b1 = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True,
            ),
        ),
        ("model", LinearRegression()),
    ])

    b1.fit(
        dev_baseline[["b_med"]],
        y_dev,
    )

    return b0, b1


def create_test_directories():
    for directory in [
        TEST_MASK_DIR,
        TEST_ERODED_DIR,
        TEST_CLEANED_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def process_test_images(test, feature_columns):
    """Create one feature row for every test image, including failures."""

    create_test_directories()

    feature_records = []
    processing_records = []

    for index, row in enumerate(
        test.itertuples(index=False),
        start=1,
    ):
        image_id = str(row.image_id)
        safe_id = SEG.safe_filename(image_id)

        feature_record = {
            "image_id": image_id,
            **{column: np.nan for column in feature_columns},
        }

        # Device/lamp indicators are known from metadata even when
        # image segmentation or colour extraction fails.
        feature_record.update(
            FSET.encode_device_lamp(row.device, row.lamp)
        )

        processing_record = {
            "image_id": image_id,
            "role": "test",
            "tube_id": str(row.tube_id),
            "instance_id": int(row.instance_id),
            "conc": float(row.conc),
            "device": str(row.device),
            "lamp": str(row.lamp),
            "mask_status": "failed",
            "trim_status": "failed",
            "feature_status": "failed",
            "failure_reason": "",
        }

        try:
            original = SEG.load_rgb(
                SEG.image_path(row.image_path)
            )

            small = SEG.resize_for_segmentation(original)

            mask_small, mask_diagnostics = (
                SEG.find_central_droplet_mask(small)
            )

            height, width = original.shape[:2]

            base_mask = SEG.upscale_mask(
                mask_small,
                (width, height),
            )

            if not base_mask.any():
                raise ValueError("Upscaled segmentation mask is empty.")

            mask_status, mask_reasons = SEG.assess_mask(
                mask_diagnostics
            )

            mask_rgba = np.dstack([
                original,
                base_mask.astype(np.uint8) * 255,
            ])

            Image.fromarray(mask_rgba).save(
                TEST_MASK_DIR / f"{safe_id}.png"
            )

            eroded, glare, cleaned, trim_diagnostics = TRIM.trim_mask(
                original,
                base_mask,
                EROSION_FRACTION,
            )

            Image.fromarray(
                eroded.astype(np.uint8) * 255
            ).save(
                TEST_ERODED_DIR / f"{safe_id}.png"
            )

            cleaned_rgba = np.dstack([
                original,
                cleaned.astype(np.uint8) * 255,
            ])

            cleaned_path = TEST_CLEANED_DIR / f"{safe_id}.png"

            Image.fromarray(cleaned_rgba).save(cleaned_path)

            processing_record.update(mask_diagnostics)
            processing_record["mask_status"] = mask_status
            processing_record["mask_flag_reasons"] = mask_reasons
            processing_record.update(trim_diagnostics)

            if mask_status == "flagged":
                processing_record["trim_status"] = "source_mask_flagged"
            elif trim_diagnostics["trim_warning"]:
                processing_record["trim_status"] = "needs_review"
            else:
                processing_record["trim_status"] = "candidate"

            if not cleaned.any():
                raise ValueError("Cleaned ROI is empty after trimming.")

            # Use the exact Phase 5.3–5.4 feature extractor.
            colour_features = FEAT.compute_colour_features(
                cleaned_rgba
            )

            feature_record.update(colour_features)
            processing_record["feature_status"] = "ok"

        except Exception as exc:
            processing_record["failure_reason"] = (
                f"{type(exc).__name__}: {exc}"
            )

        feature_records.append(feature_record)
        processing_records.append(processing_record)

        if index % 25 == 0 or index == len(test):
            print(f"Processed test images: {index}/{len(test)}")

    feature_frame = pd.DataFrame(feature_records)
    processing_frame = pd.DataFrame(processing_records)

    if len(feature_frame) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Test feature row count changed.")

    if feature_frame["image_id"].duplicated().any():
        raise RuntimeError("Duplicate test feature image_id.")

    return feature_frame, processing_frame


def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    error = y_pred - y_true
    absolute_error = np.abs(error)

    low = y_true < 0.5
    medium = (y_true >= 0.5) & (y_true <= 1.0)
    high = y_true > 1.0

    def band_mae(mask):
        return float(absolute_error[mask].mean()) if mask.any() else np.nan

    # Calibration convention: actual = slope * predicted + intercept.
    if np.ptp(y_pred) <= np.finfo(float).eps:
        calibration_slope = np.nan
        calibration_intercept = np.nan
    else:
        calibration_slope, calibration_intercept = np.polyfit(
            y_pred,
            y_true,
            deg=1,
        )

    return {
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": float(
            1.0 - np.sum(error ** 2) / np.sum((y_true - y_true.mean()) ** 2)
        ),
        "bias_pred_minus_actual": float(error.mean()),
        "within_plus_minus_0_1": float(np.mean(absolute_error <= 0.1)),
        "mae_low_actual_lt_0_5": band_mae(low),
        "mae_medium_0_5_to_1_0": band_mae(medium),
        "mae_high_actual_gt_1_0": band_mae(high),
        "calibration_slope_actual_on_predicted": float(calibration_slope),
        "calibration_intercept": float(calibration_intercept),
    }


def tube_bootstrap(predictions):
    """Paired bootstrap: resample physical tubes, not individual images."""

    tube_values = predictions["tube_id"].astype(str).to_numpy()
    unique_tubes = np.unique(tube_values)

    tube_indices = {
        tube: np.flatnonzero(tube_values == tube)
        for tube in unique_tubes
    }

    model_columns = {
        "SVR_S1_device_lamp": "predicted_svr",
        "B0_mean": "predicted_b0",
        "B1_b_med_linear": "predicted_b1",
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_records = []

    # Use the same resampled tubes for all models in each replicate.
    for replicate in range(BOOTSTRAP_REPS):
        sampled_tubes = rng.choice(
            unique_tubes,
            size=len(unique_tubes),
            replace=True,
        )

        sample_indices = np.concatenate([
            tube_indices[tube]
            for tube in sampled_tubes
        ])

        actual = predictions["actual"].to_numpy(dtype=float)[sample_indices]

        for model_name, prediction_column in model_columns.items():
            predicted = predictions[prediction_column].to_numpy(
                dtype=float
            )[sample_indices]

            bootstrap_records.append({
                "model": model_name,
                "replicate": replicate,
                "mae": float(np.mean(np.abs(predicted - actual))),
            })

    samples = pd.DataFrame(bootstrap_records)

    intervals = (
        samples.groupby("model")["mae"]
        .quantile([0.025, 0.975])
        .unstack()
        .rename(columns={
            0.025: "tube_bootstrap_mae_ci_low",
            0.975: "tube_bootstrap_mae_ci_high",
        })
    )

    return samples, intervals


def write_report(predictions, metrics, processing, config):
    lines = [
        "# Phase 7 — Final test evaluation",
        "",
        f"Evaluation date: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Frozen model",
        "",
        "- Model: SVR_S1_device_lamp",
        "- Development fit: 864 images from 41 physical tubes",
        "- Features: L_mean, a_mean, b_mean, device/lamp indicators",
        "- Hyperparameters: RBF kernel, C=10, epsilon=0.02, gamma=0.1",
        f"- Model SHA256: `{EXPECTED_MODEL_SHA256}`",
        "",
        "## Test partition",
        "",
        f"- Images: {len(predictions)}",
        f"- Physical tubes: {predictions['tube_id'].nunique()}",
        "- Test images were processed once with the frozen pipeline.",
        "- No model or preprocessing decisions were changed using test results.",
        "",
        "## Segmentation and feature processing",
        "",
    ]

    status_counts = processing["feature_status"].value_counts(dropna=False)

    for status, count in status_counts.items():
        lines.append(f"- Feature status `{status}`: {int(count)}")

    lines += [
        "",
        "Questionable masks remain identified in the processing manifest. "
        "Rows with failed segmentation or feature extraction were retained; "
        "the model's development-fitted imputer handled missing colour features.",
        "",
        "## Test metrics",
        "",
        "| Model | MAE | RMSE | R² | Bias | Within ±0.1 | Tube-bootstrap 95% CI for MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} "
            f"| {row.mae:.4f} "
            f"| {row.rmse:.4f} "
            f"| {row.r2:.4f} "
            f"| {row.bias_pred_minus_actual:.4f} "
            f"| {row.within_plus_minus_0_1:.3f} "
            f"| [{row.tube_bootstrap_mae_ci_low:.4f}, "
            f"{row.tube_bootstrap_mae_ci_high:.4f}] |"
        )

    lines += [
        "",
        "## Concentration-band MAE",
        "",
        "| Model | Low (<0.5%) | Medium (0.5–1.0%) | High (>1.0%) |",
        "|---|---:|---:|---:|",
    ]

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} "
            f"| {row.mae_low_actual_lt_0_5:.4f} "
            f"| {row.mae_medium_0_5_to_1_0:.4f} "
            f"| {row.mae_high_actual_gt_1_0:.4f} |"
        )

    lines += [
        "",
        "## Calibration",
        "",
        "Calibration is reported as actual concentration regressed on predicted concentration.",
        "",
        "| Model | Slope | Intercept |",
        "|---|---:|---:|",
    ]

    for row in metrics.itertuples(index=False):
        lines.append(
            f"| {row.model} "
            f"| {row.calibration_slope_actual_on_predicted:.4f} "
            f"| {row.calibration_intercept:.4f} |"
        )

    lines += [
        "",
        "## Uncertainty method",
        "",
        f"- {BOOTSTRAP_REPS:,} paired bootstrap resamples.",
        f"- Resampling unit: physical tube ({EXPECTED_TEST_TUBES} tubes).",
        f"- Random seed: {BOOTSTRAP_SEED}.",
        "- Interval: 2.5th to 97.5th percentile of bootstrap MAE.",
        "- The same sampled tubes were used for all three models in each replicate.",
        "",
        "## Interpretation boundary",
        "",
        "These are final test results for the frozen pipeline. They must not be used "
        "to tune the model, features, segmentation, erosion, glare thresholds, or "
        "other preprocessing choices. Any future model changes require a new "
        "independent test set.",
        "",
        "## Output files",
        "",
        "- `results/test_predictions.csv`",
        "- `results/test_processing_manifest.csv`",
        "- `results/test_results.csv`",
        "- `results/test_tube_bootstrap_samples.csv`",
        "- `results/phase7_test_accessed.json`",
        "- `reports/phase7_final_test.md`",
        "",
    ]

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main():
    prepare_output_locations()

    (
        config,
        final_model,
        feature_columns,
        metadata,
        dev_baseline,
        test,
    ) = load_and_validate_inputs()

    # Fit B0 and B1 on development data only, before test-image processing.
    b0, b1 = fit_development_baselines(dev_baseline)

    # Write the one-shot marker immediately before opening test images.
    marker = {
        "phase": "7",
        "test_partition_accessed": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_images": int(len(test)),
        "test_tubes": int(test["tube_id"].nunique()),
        "development_images_for_baselines": int(len(dev_baseline)),
        "selected_model": config["model_name"],
        "model_sha256": EXPECTED_MODEL_SHA256,
        "dataset_sha256": EXPECTED_DATASET_SHA256,
        "erosion_fraction": EROSION_FRACTION,
        "glare_value_threshold": GLARE_VALUE_THRESHOLD,
        "glare_saturation_threshold": GLARE_SATURATION_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }

    ACCESS_MARKER.write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )

    print("PHASE 7 — ONE-SHOT FINAL TEST")
    print("=" * 60)
    print("Development images used to fit baselines:", len(dev_baseline))
    print("Test images:", len(test))
    print("Test tubes:", test["tube_id"].nunique())
    print("Frozen model:", config["model_name"])
    print("Test partition is now being processed once.")

    test_features, processing = process_test_images(
        test,
        feature_columns,
    )

    # Preserve metadata order and exactly one prediction row per test image.
    test_rows = test[
        [
            "image_id",
            "tube_id",
            "instance_id",
            "conc",
            "device",
            "lamp",
        ]
    ].copy()

    test_rows["image_id"] = test_rows["image_id"].astype(str)

    test_rows = test_rows.merge(
        test_features,
        on="image_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )

    if len(test_rows) != EXPECTED_TEST_IMAGES:
        raise RuntimeError("Test prediction join changed the row count.")

    X_final = test_rows[feature_columns]

    test_rows["predicted_svr"] = final_model.predict(X_final)

    test_rows["predicted_b0"] = b0.predict(
        np.zeros((len(test_rows), 1))
    )

    test_rows["predicted_b1"] = b1.predict(
        test_rows[["b_med"]]
    )

    predictions = test_rows.rename(
        columns={"conc": "actual"}
    )

    predictions.to_csv(PREDICTIONS_PATH, index=False)
    processing.to_csv(PROCESSING_PATH, index=False)

    model_predictions = {
        "SVR_S1_device_lamp": "predicted_svr",
        "B0_mean": "predicted_b0",
        "B1_b_med_linear": "predicted_b1",
    }

    metric_records = []

    for model_name, prediction_column in model_predictions.items():
        metric_records.append({
            "model": model_name,
            **calculate_metrics(
                predictions["actual"],
                predictions[prediction_column],
            ),
        })

    metrics = pd.DataFrame(metric_records)

    bootstrap_samples, intervals = tube_bootstrap(predictions)

    metrics = metrics.merge(
        intervals,
        left_on="model",
        right_index=True,
        how="left",
        validate="one_to_one",
    )

    metrics.to_csv(METRICS_PATH, index=False)
    bootstrap_samples.to_csv(BOOTSTRAP_PATH, index=False)

    write_report(
        predictions,
        metrics,
        processing,
        config,
    )

    # Update the marker only after all outputs are successfully written.
    marker["completed"] = True
    marker["feature_status_counts"] = (
        processing["feature_status"]
        .value_counts(dropna=False)
        .to_dict()
    )
    marker["output_files"] = [
        str(PREDICTIONS_PATH.relative_to(ROOT)),
        str(PROCESSING_PATH.relative_to(ROOT)),
        str(METRICS_PATH.relative_to(ROOT)),
        str(BOOTSTRAP_PATH.relative_to(ROOT)),
        str(REPORT_PATH.relative_to(ROOT)),
    ]

    ACCESS_MARKER.write_text(
        json.dumps(marker, indent=2),
        encoding="utf-8",
    )

    print("\nTEST PROCESSING SUMMARY")
    print("=" * 60)
    print(processing["feature_status"].value_counts(dropna=False).to_string())

    print("\nFINAL TEST METRICS")
    print("=" * 60)
    print(metrics.to_string(index=False))

    print("\nSaved:")
    print(PREDICTIONS_PATH.relative_to(ROOT))
    print(PROCESSING_PATH.relative_to(ROOT))
    print(METRICS_PATH.relative_to(ROOT))
    print(BOOTSTRAP_PATH.relative_to(ROOT))
    print(REPORT_PATH.relative_to(ROOT))
    print(ACCESS_MARKER.relative_to(ROOT))

    print("\nPhase 7 completed. Do not rerun this script.")


if __name__ == "__main__":
    main()
