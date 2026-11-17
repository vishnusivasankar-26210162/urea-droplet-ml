
"""
Phase 5.7: Finalize features.csv and dataset.csv.

Inputs:
    data/processed/metadata.csv
    data/processed/dataset.csv          (current Phase 5.5 output)
    data/processed/feature_set_columns.json

Outputs:
    data/processed/features.csv
    data/processed/dataset.csv

Only development rows (train + validation) are retained.
Test images are never opened.
"""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
CURRENT_DATASET_PATH = ROOT / "data/processed/dataset.csv"
FEATURE_REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"

FEATURES_PATH = ROOT / "data/processed/features.csv"
FINAL_DATASET_PATH = ROOT / "data/processed/dataset.csv"

DEVELOPMENT_ROLES = {"train", "validation"}

# These are paths/status fields used during preprocessing, not features.
DROP_FROM_FEATURES = {
    "mask_path",
    "cleaned_rgba_path",
    "trim_status",
}


def main():
    metadata = pd.read_csv(METADATA_PATH)
    current = pd.read_csv(CURRENT_DATASET_PATH)

    with open(FEATURE_REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    for name, frame in (
        ("metadata", metadata),
        ("current dataset", current),
    ):
        if "image_id" not in frame.columns:
            raise ValueError(f"{name} does not contain image_id")

        if frame["image_id"].duplicated().any():
            raise ValueError(f"Duplicate image_id in {name}")

    # Keep only development metadata. Do not access test image files.
    dev_metadata = metadata[
        metadata["role"].astype(str).isin(DEVELOPMENT_ROLES)
    ].copy()

    dev_ids = set(dev_metadata["image_id"].astype(str))

    current_dev = current[
        current["image_id"].astype(str).isin(dev_ids)
    ].copy()

    if set(current_dev["image_id"].astype(str)) != dev_ids:
        missing = dev_ids - set(current_dev["image_id"].astype(str))
        extra = set(current_dev["image_id"].astype(str)) - dev_ids

        raise ValueError(
            f"Image ID mismatch. Missing={len(missing)}, extra={len(extra)}"
        )

    # Feature table consists of columns not already present in metadata,
    # excluding temporary paths used during preprocessing.
    feature_columns = [
        "image_id"
    ] + [
        column
        for column in current_dev.columns
        if (
            column != "image_id"
            and column not in metadata.columns
            and column not in DROP_FROM_FEATURES
        )
    ]

    features = current_dev[feature_columns].copy()

    if features["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in features table")

    # Validate all feature-set registry columns.
    for set_name, variants in registry.items():
        for variant_name, columns in variants.items():
            missing = [
                column for column in columns
                if column not in features.columns
            ]

            if missing:
                raise ValueError(
                    f"{set_name}/{variant_name} columns missing "
                    f"from features.csv: {missing}"
                )

            forbidden = [
                column for column in columns
                if (
                    column.startswith("qc_")
                    or column in {
                        "image_id",
                        "tube_id",
                        "conc",
                        "role",
                        "device",
                        "lamp",
                    }
                )
            ]

            if forbidden:
                raise ValueError(
                    f"Forbidden model columns in "
                    f"{set_name}/{variant_name}: {forbidden}"
                )

    # Save the standalone feature table.
    features.to_csv(FEATURES_PATH, index=False)

    # Reconstruct the final long dataset with metadata.
    dataset = dev_metadata.merge(
        features,
        on="image_id",
        how="inner",
        validate="one_to_one",
    )

    if len(dataset) != len(dev_metadata):
        raise RuntimeError(
            "Final dataset row count does not match development metadata"
        )

    if dataset["image_id"].duplicated().any():
        raise RuntimeError("Duplicate image_id in final dataset")

    dataset.to_csv(FINAL_DATASET_PATH, index=False)

    print("PHASE 5.7: FINAL DATASET VALIDATION")
    print("=" * 64)

    print("Development metadata rows:", len(dev_metadata))
    print("Feature rows:", len(features))
    print("Final dataset rows:", len(dataset))

    print("\nRoles:")
    print(dataset["role"].value_counts().to_string())

    print("\nFeature columns:", len(features.columns) - 1)
    print("Feature-set definitions:", len(registry))

    for name, variants in registry.items():
        print(
            f"{name}: "
            f"{len(variants['colour_only'])} colour features, "
            f"{len(variants['with_device_and_lamp'])} "
            "including device/lamp"
        )

    print("\nFeature table:", FEATURES_PATH.relative_to(ROOT))
    print("Dataset:", FINAL_DATASET_PATH.relative_to(ROOT))

    print("\nAll feature-set columns validated.")
    print("One row per development image_id.")
    print("QC columns excluded from feature registries.")
    print("Test images were not opened.")


if __name__ == "__main__":
    main()
