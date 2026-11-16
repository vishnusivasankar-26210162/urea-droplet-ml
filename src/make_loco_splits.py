
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "processed" / "metadata.csv"
OUTPUT_PATH = ROOT / "splits" / "loco_folds.csv"

DEVELOPMENT_ROLES = {"train", "validation"}


def main():
    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    required = {"tube_id", "conc", "instance_id", "role"}
    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Verify that every physical tube has one concentration,
    # one instance and one primary partition role.
    tube_check = metadata.groupby("tube_id").agg(
        n_concentrations=("conc", "nunique"),
        n_instances=("instance_id", "nunique"),
        n_roles=("role", "nunique"),
    )

    if (tube_check != 1).any().any():
        raise ValueError(
            "A tube has inconsistent concentration, instance or role"
        )

    # Build one row per physical tube.
    tubes = (
        metadata[
            ["tube_id", "conc", "instance_id", "role"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # Exclude the locked test partition completely.
    dev = tubes[
        tubes["role"].isin(DEVELOPMENT_ROLES)
    ].copy()

    test = tubes[tubes["role"] == "test"]

    if len(dev) != 41:
        raise ValueError(
            f"Expected 41 development tubes, found {len(dev)}"
        )

    if len(test) != 10:
        raise ValueError(
            f"Expected 10 locked test tubes, found {len(test)}"
        )

    concentrations = sorted(dev["conc"].unique())

    if len(concentrations) != 27:
        raise ValueError(
            f"Expected 27 development concentrations, "
            f"found {len(concentrations)}"
        )

    test_tubes = set(test["tube_id"])

    fold_rows = []

    # Generate one fold for every development concentration.
    for concentration in concentrations:

        validation = dev[
            dev["conc"] == concentration
        ]

        training = dev[
            dev["conc"] != concentration
        ]

        train_tubes = set(training["tube_id"])
        validation_tubes = set(validation["tube_id"])

        # Leakage checks.
        if train_tubes.intersection(validation_tubes):
            raise AssertionError(
                f"Tube leakage at concentration {concentration}"
            )

        if train_tubes.intersection(test_tubes):
            raise AssertionError("Test tube entered LOCO training")

        if validation_tubes.intersection(test_tubes):
            raise AssertionError("Test tube entered LOCO validation")

        if (training["conc"] == concentration).any():
            raise AssertionError(
                "Held-out concentration found in training"
            )

        if not (validation["conc"] == concentration).all():
            raise AssertionError(
                "LOCO validation contains another concentration"
            )

        fold_id = f"C{concentration:.2f}"

        for _, row in training.iterrows():
            fold_rows.append({
                "fold_id": fold_id,
                "held_out_conc": float(concentration),
                "tube_id": row["tube_id"],
                "conc": float(row["conc"]),
                "instance_id": int(row["instance_id"]),
                "fold_role": "train",
            })

        for _, row in validation.iterrows():
            fold_rows.append({
                "fold_id": fold_id,
                "held_out_conc": float(concentration),
                "tube_id": row["tube_id"],
                "conc": float(row["conc"]),
                "instance_id": int(row["instance_id"]),
                "fold_role": "validation",
            })

    result = pd.DataFrame(fold_rows)

    # Each fold must contain exactly all 41 development tubes.
    for fold_id, fold in result.groupby("fold_id"):

        if fold["tube_id"].nunique() != 41:
            raise AssertionError(
                f"{fold_id}: expected 41 development tubes"
            )

        if set(fold["tube_id"]) != set(dev["tube_id"]):
            raise AssertionError(
                f"{fold_id}: development tube set mismatch"
            )

    # Final uniqueness check: one tube appears once in each fold.
    if result.duplicated(["fold_id", "tube_id"]).any():
        raise AssertionError("Duplicate tube within a LOCO fold")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    summary = (
        result.groupby(["fold_id", "fold_role"])
        .agg(
            tube_count=("tube_id", "nunique"),
            concentration=("held_out_conc", "first"),
        )
        .reset_index()
    )

    print("\nLEAVE-ONE-CONCENTRATION-OUT SPLIT")
    print("=" * 55)
    print("Development tubes:", len(dev))
    print("Excluded test tubes:", len(test))
    print("LOCO folds:", len(concentrations))

    print("\nFold summary:")
    for concentration in concentrations:
        fold_id = f"C{concentration:.2f}"
        fold = result[result["fold_id"] == fold_id]

        train_count = (
            fold.loc[fold["fold_role"] == "train", "tube_id"].nunique()
        )
        val_count = (
            fold.loc[
                fold["fold_role"] == "validation", "tube_id"
            ].nunique()
        )

        print(
            f"{fold_id}: train={train_count}, "
            f"validation={val_count}"
        )

    print("\nSaved:", OUTPUT_PATH)
    print("All LOCO leakage checks passed.")


if __name__ == "__main__":
    main()
