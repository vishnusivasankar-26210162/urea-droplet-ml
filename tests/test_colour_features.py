
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skimage.color import rgb2lab

from src.extract_colour_features import (
    compute_colour_features,
)


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
FEATURES_PATH = ROOT / "data/processed/features.csv"

A019_TOKEN = "A019"

EXPECTED_B_MED = 25.689733505249023
EXPECTED_L_MEAN = 81.69629719449716
A019_TOLERANCE = 0.05


def test_transparent_pixels_are_ignored():
    """RGB values beneath alpha=0 must not affect colour features."""

    rgba = np.zeros((20, 20, 4), dtype=np.uint8)

    # Transparent pixels contain a different colour.
    rgba[:, :] = [255, 0, 0, 0]

    # Only this central region is visible.
    rgba[5:15, 5:15] = [110, 90, 80, 255]

    actual = compute_colour_features(rgba)

    rgb = np.array([110, 90, 80], dtype=np.float32) / 255.0

    expected_lab = rgb2lab(
        rgb.reshape(1, 1, 3)
    )[0, 0]

    assert actual["L_mean"] == pytest.approx(
        expected_lab[0], abs=1e-5
    )
    assert actual["a_mean"] == pytest.approx(
        expected_lab[1], abs=1e-5
    )
    assert actual["b_mean"] == pytest.approx(
        expected_lab[2], abs=1e-5
    )


def test_uniform_colour_returns_known_lab():
    """A uniform opaque image should return its known CIELAB value."""

    rgb = np.array([120, 80, 60], dtype=np.uint8)

    rgba = np.zeros((30, 30, 4), dtype=np.uint8)
    rgba[:, :, :3] = rgb
    rgba[:, :, 3] = 255

    features = compute_colour_features(rgba)

    expected_lab = rgb2lab(
        (rgb.astype(np.float32) / 255.0).reshape(1, 1, 3)
    )[0, 0]

    assert features["L_mean"] == pytest.approx(
        expected_lab[0], abs=1e-5
    )
    assert features["a_mean"] == pytest.approx(
        expected_lab[1], abs=1e-5
    )
    assert features["b_mean"] == pytest.approx(
        expected_lab[2], abs=1e-5
    )

    assert features["L_std"] == pytest.approx(0.0, abs=1e-6)
    assert features["a_std"] == pytest.approx(0.0, abs=1e-6)
    assert features["b_std"] == pytest.approx(0.0, abs=1e-6)


def find_a019_image_id() -> str:
    """Find the A019 image using the metadata's path or image ID."""

    metadata = pd.read_csv(
        METADATA_PATH,
        dtype=str,
    )

    search_columns = [
        column
        for column in ("image_id", "image_path", "orig_path")
        if column in metadata.columns
    ]

    if not search_columns:
        raise AssertionError(
            "metadata.csv has no image_id, image_path or orig_path column."
        )

    matches = pd.Series(
        False,
        index=metadata.index,
    )

    for column in search_columns:
        matches |= metadata[column].fillna("").str.contains(
            A019_TOKEN,
            case=False,
            regex=False,
        )

    image_ids = (
        metadata.loc[matches, "image_id"]
        .dropna()
        .unique()
        .tolist()
    )

    if len(image_ids) != 1:
        raise AssertionError(
            f"Expected exactly one A019 image, found {len(image_ids)}: "
            f"{image_ids}. Update A019_TOKEN to the exact identifier."
        )

    return str(image_ids[0])


def test_a019_current_frozen_pipeline_values():
    """
    Regression test for A019 using the current frozen ROI pipeline.

    These values differ from the historical roadmap reference.
    The discrepancy is documented in reports/decision_log.md.
    """

    if not FEATURES_PATH.is_file():
        raise FileNotFoundError(
            "Run python src/extract_colour_features.py first."
        )

    image_id = find_a019_image_id()

    features = pd.read_csv(FEATURES_PATH)

    rows = features[
        features["image_id"].astype(str) == image_id
    ]

    assert len(rows) == 1

    row = rows.iloc[0]

    assert row["feature_status"] == "ok", (
        f"A019 extraction failed: {row.get('feature_error', '')}"
    )

    assert row["b_med"] == pytest.approx(
        EXPECTED_B_MED,
        abs=A019_TOLERANCE,
    )

    assert row["L_mean"] == pytest.approx(
        EXPECTED_L_MEAN,
        abs=A019_TOLERANCE,
    )

def test_feature_row_count_matches_development_metadata():
    """One feature row must exist for every train/validation image."""

    metadata = pd.read_csv(METADATA_PATH)
    features = pd.read_csv(FEATURES_PATH)

    expected_count = metadata["role"].isin(
        {"train", "validation"}
    ).sum()

    assert len(features) == expected_count

    assert features["image_id"].is_unique
