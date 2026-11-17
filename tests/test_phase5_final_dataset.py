
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_features_and_dataset_match_development_metadata():
    metadata = pd.read_csv(
        ROOT / "data/processed/metadata.csv"
    )

    features = pd.read_csv(
        ROOT / "data/processed/features.csv"
    )

    dataset = pd.read_csv(
        ROOT / "data/processed/dataset.csv"
    )

    dev = metadata[
        metadata["role"].isin(["train", "validation"])
    ]

    expected_ids = set(dev["image_id"].astype(str))

    assert len(features) == len(dev)
    assert len(dataset) == len(dev)

    assert features["image_id"].is_unique
    assert dataset["image_id"].is_unique

    assert set(features["image_id"].astype(str)) == expected_ids
    assert set(dataset["image_id"].astype(str)) == expected_ids


def test_all_feature_sets_exist_in_features_csv():
    features = pd.read_csv(
        ROOT / "data/processed/features.csv",
        nrows=1,
    )

    with open(
        ROOT / "data/processed/feature_set_columns.json",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    for name, variants in registry.items():
        assert name in {"S0", "S1", "S2", "S3", "S4"}

        for variant, columns in variants.items():
            assert variant in {
                "colour_only",
                "with_device_and_lamp",
            }

            for column in columns:
                assert column in features.columns, (
                    f"Missing {column} in {name}/{variant}"
                )


def test_qc_fields_are_not_model_inputs():
    with open(
        ROOT / "data/processed/feature_set_columns.json",
        encoding="utf-8",
    ) as file:
        registry = json.load(file)

    for variants in registry.values():
        for columns in variants.values():
            assert not any(
                column.startswith("qc_")
                for column in columns
            )

            assert "conc" not in columns
            assert "tube_id" not in columns
            assert "image_id" not in columns
            assert "role" not in columns
