
"""
Phase 5.5: Build feature sets S0-S4.

Inputs:
    data/processed/metadata.csv
    data/processed/features.csv
    data/interim/roi_mask_manifest.csv
    data/interim/phase5_trim_manifest.csv

Outputs:
    data/processed/dataset.csv
    data/processed/feature_set_columns.json

Only train and validation images are opened.
No test images are accessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2hsv


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
FEATURES_PATH = ROOT / "data/processed/features.csv"
MASK_MANIFEST_PATH = ROOT / "data/interim/roi_mask_manifest.csv"
TRIM_MANIFEST_PATH = ROOT / "data/interim/phase5_trim_manifest.csv"

DATASET_PATH = ROOT / "data/processed/dataset.csv"
REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"

DEVELOPMENT_ROLES = {"train", "validation"}
ALPHA_THRESHOLD = 127

DEVICE_LAMP_COLUMNS = [
    "device_dino",
    "device_phone",
    "lamp_off",
    "lamp_on",
]


# ============================================================
# Utilities
# ============================================================

def safe_project_path(value: str | Path) -> Path:
    """Resolve a path inside the project root."""

    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes project root: {value}")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_rgba(path: str | Path) -> np.ndarray:
    with Image.open(safe_project_path(path)) as image:
        image.load()
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def summarize(values: np.ndarray, prefix: str) -> dict:
    """Mean, median, population SD, p10 and p90."""

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        raise ValueError(f"No pixel values for {prefix}")

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_med": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


# ============================================================
# S0: Old-style unmasked RGB means
# ============================================================

def compute_s0_unmasked_rgb_means(rgba: np.ndarray) -> dict:
    """
    Calculate RGB means over the entire image.

    Pixels outside the original droplet mask are made black,
    reproducing the transparent-black background convention
    of the old cut-out images.

    All pixels, including those representing transparent
    background, contribute to the mean.
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA array; received {rgba.shape}")

    rgb = rgba[:, :, :3].astype(np.float64).copy()
    alpha = rgba[:, :, 3]

    droplet_mask = alpha > ALPHA_THRESHOLD

    if not droplet_mask.any():
        raise ValueError("Original droplet mask is empty")

    # Explicitly zero RGB outside the droplet.
    # Then average every pixel, not only visible pixels.
    rgb[~droplet_mask] = 0.0

    return {
        "S0_R_mean": float(rgb[:, :, 0].mean()),
        "S0_G_mean": float(rgb[:, :, 1].mean()),
        "S0_B_mean": float(rgb[:, :, 2].mean()),
    }


# ============================================================
# S4: RGB and HSV statistics inside cleaned ROI
# ============================================================

def compute_s4_rgb_hsv_statistics(rgba: np.ndarray) -> dict:
    """
    Compute RGB and HSV statistics only inside the cleaned ROI.

    RGB: 0-255
    HSV: normalized to [0, 1]
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f"Expected RGBA array; received {rgba.shape}")

    mask = rgba[:, :, 3] > ALPHA_THRESHOLD

    if not mask.any():
        raise ValueError("Cleaned ROI mask is empty")

    rgb_pixels = rgba[:, :, :3][mask]

    rgb_float = rgb_pixels.astype(np.float32) / 255.0

    hsv_pixels = rgb2hsv(
        rgb_float.reshape(1, -1, 3)
    ).reshape(-1, 3)

    result = {}

    # RGB statistics.
    for index, channel in enumerate(("R", "G", "B")):
        result.update(
            summarize(
                rgb_pixels[:, index],
                f"S4_rgb_{channel}",
            )
        )

    # HSV statistics.
    for index, channel in enumerate(("H", "S", "V")):
        result.update(
            summarize(
                hsv_pixels[:, index],
                f"S4_hsv_{channel}",
            )
        )

    # Circular mean for hue, because hue wraps at 0 and 1.
    hue_radians = hsv_pixels[:, 0] * 2.0 * np.pi

    circular_mean = np.arctan2(
        np.mean(np.sin(hue_radians)),
        np.mean(np.cos(hue_radians)),
    )

    result["S4_hsv_H_circular_mean"] = float(
        (circular_mean / (2.0 * np.pi)) % 1.0
    )

    return result


# ============================================================
# Device and lamp one-hot encoding
# ============================================================

def encode_device_lamp(device, lamp) -> dict:
    device_text = str(device).strip().lower()
    lamp_text = str(lamp).strip().lower()

    if "dino" in device_text:
        device_value = "dino"
    elif (
        "phone" in device_text
        or "samsung" in device_text
        or "a31" in device_text
    ):
        device_value = "phone"
    else:
        raise ValueError(f"Unrecognized device: {device!r}")

    if lamp_text in {"off", "false", "0"}:
        lamp_value = "off"
    elif lamp_text in {"on", "true", "1"}:
        lamp_value = "on"
    else:
        raise ValueError(f"Unrecognized lamp: {lamp!r}")

    return {
        "device_dino": int(device_value == "dino"),
        "device_phone": int(device_value == "phone"),
        "lamp_off": int(lamp_value == "off"),
        "lamp_on": int(lamp_value == "on"),
    }


# ============================================================
# Load and join development data
# ============================================================

def load_development_data() -> pd.DataFrame:
    metadata = pd.read_csv(METADATA_PATH)
    features = pd.read_csv(FEATURES_PATH)
    masks = pd.read_csv(MASK_MANIFEST_PATH)
    trim = pd.read_csv(TRIM_MANIFEST_PATH)

    frames = {
        "metadata": metadata,
        "features": features,
        "mask manifest": masks,
        "trim manifest": trim,
    }

    for name, frame in frames.items():
        if "image_id" not in frame.columns:
            raise ValueError(f"{name} has no image_id column")

        if frame["image_id"].duplicated().any():
            raise ValueError(f"Duplicate image_id in {name}")

    required_metadata = {"role", "device", "lamp"}
    required_masks = {"mask_path"}
    required_trim = {"cleaned_rgba_path", "trim_status"}

    for required, frame, name in (
        (required_metadata, metadata, "metadata"),
        (required_masks, masks, "mask manifest"),
        (required_trim, trim, "trim manifest"),
    ):
        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"{name} missing columns: {sorted(missing)}"
            )

    # Critical: test rows are removed before any paths are opened.
    dev = metadata[
        metadata["role"].astype(str).isin(DEVELOPMENT_ROLES)
    ].copy()

    dev = dev.merge(
        features,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    dev = dev.merge(
        masks[["image_id", "mask_path"]],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    trim_columns = [
        "image_id",
        "cleaned_rgba_path",
        "trim_status",
    ]

    dev = dev.merge(
        trim[trim_columns],
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if dev["mask_path"].isna().any():
        raise ValueError("Some development images lack source masks")

    return dev


# ============================================================
# Build feature-set registry
# ============================================================

def build_feature_registry(available_columns) -> dict:
    available_columns = set(available_columns)

    def require(columns):
        missing = [
            column for column in columns
            if column not in available_columns
        ]

        if missing:
            raise ValueError(
                f"Missing required feature columns: {missing}"
            )

        return columns

    # S0: unmasked RGB means, transparent background included.
    s0 = require([
        "S0_R_mean",
        "S0_G_mean",
        "S0_B_mean",
    ])

    # S1: masked Lab means.
    s1 = require([
        "L_mean",
        "a_mean",
        "b_mean",
    ])

    # S2: S1 + medians and standard deviations.
    s2 = s1 + require([
        "L_med", "a_med", "b_med",
        "L_std", "a_std", "b_std",
    ])

    # S3: S2 + percentiles and pale fraction.
    # Chroma and hue from Phase 5.4 are included here as well.
    s3 = s2 + require([
        "L_p10", "a_p10", "b_p10",
        "L_p90", "a_p90", "b_p90",
        "pale_fraction",
        "chroma_mean",
        "chroma_med",
        "chroma_std",
        "chroma_p10",
        "chroma_p90",
        "hue_angle_deg",
    ])

    # S4: S3 + RGB and HSV statistics.
    s4_additions = []

    for colour_space, channels in (
        ("rgb", ("R", "G", "B")),
        ("hsv", ("H", "S", "V")),
    ):
        for channel in channels:
            prefix = f"S4_{colour_space}_{channel}"

            for statistic in ("mean", "med", "std", "p10", "p90"):
                s4_additions.append(
                    f"{prefix}_{statistic}"
                )

    s4_additions.append("S4_hsv_H_circular_mean")

    s4 = s3 + require(s4_additions)

    base_sets = {
        "S0": s0,
        "S1": s1,
        "S2": s2,
        "S3": s3,
        "S4": s4,
    }

    registry = {}

    for name, columns in base_sets.items():
        registry[name] = {
            "colour_only": columns,
            "with_device_and_lamp": (
                columns + DEVICE_LAMP_COLUMNS
            ),
        }

    # Safety check: no QC or metadata columns in model inputs.
    forbidden = {
        "image_id", "tube_id", "conc", "role",
        "device", "lamp", "image_path", "image_id",
    }

    for name, variants in registry.items():
        for variant, columns in variants.items():
            bad = [
                column for column in columns
                if column.startswith("qc_")
                or column in forbidden
            ]

            if bad:
                raise RuntimeError(
                    f"Forbidden columns in {name}/{variant}: {bad}"
                )

    return registry


# ============================================================
# Main
# ============================================================

def main():
    dev = load_development_data()

    print("PHASE 5.5: BUILD FEATURE SETS S0-S4")
    print("=" * 64)
    print("Development images:", len(dev))
    print("Training images:", int((dev["role"] == "train").sum()))
    print("Validation images:", int((dev["role"] == "validation").sum()))
    print("Test images: excluded and unopened.")

    records = []

    for index, row in enumerate(
        dev.itertuples(index=False),
        start=1,
    ):
        record = {
            "image_id": str(row.image_id),
        }

        # S0 uses the original segmentation image.
        try:
            source_rgba = load_rgba(row.mask_path)

            record.update(
                compute_s0_unmasked_rgb_means(source_rgba)
            )

            record["S0_status"] = "ok"

        except Exception as exc:
            record.update({
                "S0_R_mean": np.nan,
                "S0_G_mean": np.nan,
                "S0_B_mean": np.nan,
                "S0_status": "failed",
                "S0_error": f"{type(exc).__name__}: {exc!r}",
            })

        # S4 RGB/HSV features use the cleaned ROI.
        try:
            if (
                pd.isna(row.cleaned_rgba_path)
                or not str(row.cleaned_rgba_path).strip()
            ):
                raise ValueError("No cleaned ROI path")

            cleaned_rgba = load_rgba(row.cleaned_rgba_path)

            record.update(
                compute_s4_rgb_hsv_statistics(cleaned_rgba)
            )

            record["S4_status"] = "ok"

        except Exception as exc:
            record["S4_status"] = "failed"
            record["S4_error"] = (
                f"{type(exc).__name__}: {exc!r}"
            )

        # Device and lamp one-hot features.
        record.update(
            encode_device_lamp(row.device, row.lamp)
        )

        records.append(record)

        if index % 50 == 0 or index == len(dev):
            print(f"Processed {index}/{len(dev)}")

    additions = pd.DataFrame(records)

    if additions["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image_id in derived features")

    dataset = dev.merge(
        additions,
        on="image_id",
        how="left",
        validate="one_to_one",
    )

    if len(dataset) != len(dev):
        raise RuntimeError("Dataset row count changed during merge")

    registry = build_feature_registry(dataset.columns)

    # Write dataset and feature-column registry.
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    dataset.to_csv(DATASET_PATH, index=False)

    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2),
        encoding="utf-8",
    )

    print("\nFEATURE SET SIZES")
    print("=" * 64)

    for name, variants in registry.items():
        print(
            f"{name}: "
            f"{len(variants['colour_only'])} colour columns; "
            f"{len(variants['with_device_and_lamp'])} "
            "columns with device/lamp"
        )

    print("\nS0 extraction:")
    print(dataset["S0_status"].value_counts(dropna=False).to_string())

    print("\nS4 extraction:")
    print(dataset["S4_status"].value_counts(dropna=False).to_string())

    print("\nOne-hot totals:")
    for column in DEVICE_LAMP_COLUMNS:
        print(f"{column}: {int(dataset[column].sum())}")

    print("\nDataset:", DATASET_PATH.relative_to(ROOT))
    print("Registry:", REGISTRY_PATH.relative_to(ROOT))
    print("Rows:", len(dataset))

    print("\nQC fields are retained for diagnostics only.")
    print("QC fields are not in any feature-set column list.")
    print("No test images were opened.")
    print("No feature set has been selected as best.")


if __name__ == "__main__":
    main()
