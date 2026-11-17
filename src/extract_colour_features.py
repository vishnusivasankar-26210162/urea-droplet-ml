
"""
Phase 5.3–5.4: per-image CIELAB colour feature extraction.

Inputs:
    data/processed/metadata.csv
    data/interim/phase5_trim_manifest.csv
    data/interim/cleaned_roi_rgba/

Output:
    data/processed/features.csv

Only train and validation images are processed.
Test images are never opened.

Colour space:
    scikit-image CIELAB, D65 illuminant, 2-degree observer.
    L*: approximately 0 to 100.
    a*, b*: signed values.

All qc_* columns are diagnostics, not model inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
TRIM_MANIFEST_PATH = ROOT / "data/interim/phase5_trim_manifest.csv"
OUTPUT_PATH = ROOT / "data/processed/features.csv"

DEVELOPMENT_ROLES = {"train", "validation"}

ALPHA_THRESHOLD = 127
PALE_B_OFFSET = 8.0

A019_EXPECTED_B_MED = 24.3
A019_EXPECTED_L_MEAN = 82.1
A019_TOLERANCE = 0.5


def safe_project_path(value: str | Path) -> Path:
    """Resolve a project-relative path and prevent escaping project root."""

    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Path escapes project root: {value}")

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    """Calculate the five requested descriptive statistics."""

    values = np.asarray(values, dtype=np.float64)

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_med": float(np.median(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
    }


def compute_colour_features(rgba: np.ndarray) -> dict[str, float]:
    """
    Extract features from an RGBA image.

    Only pixels with alpha > 127 are included.
    Transparent RGB values are ignored.
    """

    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(
            f"Expected an HxWx4 RGBA array; received {rgba.shape}"
        )

    if rgba.dtype != np.uint8:
        rgba = np.asarray(rgba, dtype=np.uint8)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = alpha > ALPHA_THRESHOLD

    if not mask.any():
        raise ValueError("Cleaned ROI alpha mask is empty")

    height, width = mask.shape

    # Coordinates and RGB values have the same row-major ordering.
    y_coords, x_coords = np.where(mask)
    rgb_pixels = rgb[mask]

    # Convert only included pixels to CIELAB.
    # This avoids allocating a full-resolution floating-point image.
    rgb_float = rgb_pixels.astype(np.float32) / 255.0

    lab = rgb2lab(
        rgb_float.reshape(1, -1, 3)
    ).reshape(-1, 3)

    lab = lab.astype(np.float64, copy=False)

    L = lab[:, 0]
    a = lab[:, 1]
    b = lab[:, 2]

    features = {}

    # L*, a*, b*: mean, median, standard deviation, p10 and p90.
    features.update(summarize(L, "L"))
    features.update(summarize(a, "a"))
    features.update(summarize(b, "b"))

    # CIELAB chroma: C*ab = sqrt(a*^2 + b*^2).
    chroma = np.hypot(a, b)
    features.update(summarize(chroma, "chroma"))

    # CIELAB hue angle: h_ab = atan2(b*, a*).
    # Circular mean handles angles near 0 and 360 degrees correctly.
    hue = np.arctan2(b, a)

    hue_mean = np.arctan2(
        np.mean(np.sin(hue)),
        np.mean(np.cos(hue)),
    )

    features["hue_angle_deg"] = float(
        np.degrees(hue_mean) % 360.0
    )

    # Pale-pixel fraction:
    # Define the top third using the ROI bounding box.
    # A pixel is pale if its b* is at least 8 units below the
    # median b* value in the top third.
    y_min = int(y_coords.min())
    y_max = int(y_coords.max())

    roi_height = y_max - y_min + 1
    top_cutoff = y_min + roi_height / 3.0

    top_third = y_coords < top_cutoff

    if not np.any(top_third):
        raise ValueError("Could not determine top-third ROI reference")

    top_b_median = float(np.median(b[top_third]))
    pale_threshold = top_b_median - PALE_B_OFFSET

    features["pale_fraction"] = float(
        np.mean(b <= pale_threshold)
    )

    # Additional diagnostics from the cleaned ROI.
    roi_pixels = int(mask.sum())
    total_pixels = int(height * width)

    features["qc_cleaned_roi_area_fraction"] = (
        roi_pixels / total_pixels
    )
    features["qc_cleaned_roi_pixel_count"] = roi_pixels
    features["qc_image_width"] = width
    features["qc_image_height"] = height
    features["qc_pale_reference_b_median"] = top_b_median
    features["qc_pale_b_threshold"] = float(pale_threshold)

    return features


def load_development_table() -> pd.DataFrame:
    """
    Join development metadata to the trimming manifest.

    Test rows are filtered out before any image paths are opened.
    """

    metadata = pd.read_csv(METADATA_PATH)
    trim = pd.read_csv(TRIM_MANIFEST_PATH)

    required_metadata = {"image_id", "role"}
    required_trim = {
        "image_id",
        "trim_status",
        "cleaned_rgba_path",
    }

    if not required_metadata.issubset(metadata.columns):
        missing = sorted(required_metadata - set(metadata.columns))
        raise ValueError(f"Metadata is missing columns: {missing}")

    if not required_trim.issubset(trim.columns):
        missing = sorted(required_trim - set(trim.columns))
        raise ValueError(f"Trim manifest is missing columns: {missing}")

    if metadata["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in metadata.csv")

    if trim["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in phase5_trim_manifest.csv")

    # Filter to development before touching image paths.
    dev = metadata[
        metadata["role"].astype(str).isin(DEVELOPMENT_ROLES)
    ].copy()

    trim_columns = [
        "image_id",
        "trim_status",
        "trim_warning",
        "cleaned_rgba_path",
        "base_mask_pixels",
        "eroded_mask_pixels",
        "glare_pixels_removed",
        "glare_removed_fraction",
        "erosion_retained_fraction",
        "cleaned_area_fraction",
        "source_mask_status",
    ]

    available_columns = [
        column
        for column in trim_columns
        if column in trim.columns
    ]

    dev = dev.merge(
        trim[available_columns],
        on="image_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    dev["trim_record_found"] = dev["_merge"].eq("both")
    dev = dev.drop(columns="_merge")

    return dev


def extract_one_row(row: pd.Series) -> dict:
    """Extract features for one image while retaining failed rows."""

    record = {
        "image_id": str(row["image_id"]),
        "colour_space": "CIELAB_skimage_D65_2deg",
        "feature_status": "failed",
        "feature_error": "",
        "qc_trim_status": row.get("trim_status", ""),
        "qc_trim_warning": row.get("trim_warning", ""),
    }

    if not bool(row.get("trim_record_found", False)):
        record["feature_error"] = "Missing trimming-manifest row"
        return record

    cleaned_path_value = row.get("cleaned_rgba_path")

    if (
        cleaned_path_value is None
        or pd.isna(cleaned_path_value)
        or not str(cleaned_path_value).strip()
    ):
        record["feature_error"] = "No cleaned ROI output path"
        return record

    try:
        cleaned_path = safe_project_path(
            str(cleaned_path_value)
        )

        with Image.open(cleaned_path) as image:
            image.load()
            rgba = np.asarray(
                image.convert("RGBA"),
                dtype=np.uint8,
            )

        features = compute_colour_features(rgba)
        record.update(features)

        # Carry preprocessing diagnostics into features.csv.
        diagnostic_columns = [
            ("erosion_retained_fraction",
             "qc_erosion_retained_fraction"),
            ("base_mask_pixels",
             "qc_original_mask_pixels"),
            ("eroded_mask_pixels",
             "qc_eroded_mask_pixels"),
            ("glare_pixels_removed",
             "qc_glare_pixels_removed"),
            ("cleaned_area_fraction",
             "qc_manifest_cleaned_area_fraction"),
            ("source_mask_status",
             "qc_source_mask_status"),
        ]

        for source, destination in diagnostic_columns:
            value = row.get(source)

            if value is not None and not pd.isna(value):
                record[destination] = value

        # Original droplet area and transparency are based on the
        # pre-erosion mask, not the cleaned ROI.
        height, width = rgba.shape[:2]
        total_pixels = float(height * width)

        base_pixels = row.get("base_mask_pixels")
        glare_pixels = row.get("glare_pixels_removed")

        if pd.notna(base_pixels):
            base_pixels = float(base_pixels)

            record["qc_droplet_area_fraction"] = (
                base_pixels / total_pixels
            )
            record["qc_transparent_fraction"] = (
                1.0 - base_pixels / total_pixels
            )
            record["qc_original_droplet_pixels"] = int(base_pixels)
        else:
            record["qc_droplet_area_fraction"] = np.nan
            record["qc_transparent_fraction"] = np.nan

        # Numerator: glare removed inside the eroded ROI.
        # Denominator: original pre-erosion droplet area.
        if (
            pd.notna(glare_pixels)
            and pd.notna(base_pixels)
            and base_pixels > 0
        ):
            record["qc_glare_fraction"] = (
                float(glare_pixels) / base_pixels
            )
        else:
            record["qc_glare_fraction"] = np.nan

        # Also retain the manifest's glare fraction, whose denominator
        # is the eroded ROI area.
        manifest_glare_fraction = row.get(
            "glare_removed_fraction"
        )

        record["qc_glare_fraction_of_eroded_roi"] = (
            float(manifest_glare_fraction)
            if pd.notna(manifest_glare_fraction)
            else np.nan
        )

        record["feature_status"] = "ok"

    except Exception as exc:
        record["feature_status"] = "failed"
        record["feature_error"] = (
            f"{type(exc).__name__}: {exc!r}"
        )

    return record


def main() -> None:
    dev = load_development_table()

    print("PHASE 5.3–5.4: CIELAB COLOUR FEATURE EXTRACTION")
    print("=" * 64)
    print("Development images:", len(dev))
    print("Training images:", int((dev["role"] == "train").sum()))
    print("Validation images:", int((dev["role"] == "validation").sum()))
    print("Test images: excluded; no test image paths will be opened.")
    print("Colour space: scikit-image CIELAB, D65, 2-degree observer")
    print("Pale rule: b* <= top-third median b* - 8.0")

    records = []

    for index, (_, row) in enumerate(
        dev.iterrows(),
        start=1,
    ):
        records.append(extract_one_row(row))

        if index % 50 == 0 or index == len(dev):
            print(f"Processed {index}/{len(dev)}")

    features = pd.DataFrame(records)

    if len(features) != len(dev):
        raise RuntimeError(
            f"Feature row count {len(features)} != "
            f"development metadata rows {len(dev)}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(".tmp")

    features.to_csv(
        temporary_path,
        index=False,
    )

    temporary_path.replace(OUTPUT_PATH)

    print("\nFEATURE EXTRACTION SUMMARY")
    print("=" * 64)

    print(
        features["feature_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    if "qc_trim_status" in features.columns:
        print("\nRows by trim status:")

        print(
            pd.crosstab(
                features["qc_trim_status"],
                features["feature_status"],
                dropna=False,
            ).to_string()
        )

    print("\nOutput:", OUTPUT_PATH.relative_to(ROOT))
    print("Rows:", len(features))
    print("Only train and validation images were opened.")
    print("Columns prefixed qc_ are diagnostics, not model inputs.")


if __name__ == "__main__":
    main()
