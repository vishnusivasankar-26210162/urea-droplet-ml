from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data" / "processed" / "metadata.csv"
ROLES_PATH = ROOT / "splits" / "tube_roles.csv"

SEED = 42

EXPECTED_ROLE_COUNTS = {
    "train": 31,
    "validation": 10,
    "test": 10,
}

EXPECTED_CONCENTRATION_REPLICATES = {
    1: 7,   # Seven concentrations with one tube
    2: 16,  # Sixteen concentrations with two tubes
    3: 4,   # Four concentrations with three tubes
}


def build_tube_table(metadata):
    required = {"tube_id", "conc", "instance_id"}
    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if metadata[list(required)].isna().any().any():
        raise ValueError("Missing tube identity information")

    # Every tube must represent exactly one concentration and instance.
    checks = metadata.groupby("tube_id").agg(
        n_concentrations=("conc", "nunique"),
        n_instances=("instance_id", "nunique"),
    )

    if (checks["n_concentrations"] != 1).any():
        raise ValueError("A tube has multiple concentrations")

    if (checks["n_instances"] != 1).any():
        raise ValueError("A tube belongs to multiple instances")

    tubes = (
        metadata[
            ["tube_id", "conc", "instance_id"]
        ]
        .drop_duplicates()
        .sort_values(["conc", "instance_id", "tube_id"])
        .reset_index(drop=True)
    )

    if len(tubes) != 51:
        raise ValueError(
            f"Expected 51 physical tubes, found {len(tubes)}"
        )

    return tubes


def assign_roles(tubes):
    """Create a deterministic, concentration-aware 60/20/20 split."""

    counts = tubes.groupby("conc").size()

    singles = sorted(
        counts[counts == 1].index.tolist()
    )
    pairs = sorted(
        counts[counts == 2].index.tolist()
    )
    triples = sorted(
        counts[counts == 3].index.tolist()
    )

    if (
        len(singles) != 7
        or len(pairs) != 16
        or len(triples) != 4
    ):
        raise ValueError(
            "The concentration replication structure has changed. "
            "Review the split design before proceeding."
        )

    triple_levels = {0.50, 1.00, 1.50, 2.00}

    if not np.allclose(sorted(triples), sorted(triple_levels)):
        raise ValueError(
            f"Unexpected three-tube concentration levels: {triples}"
        )

    rng = np.random.default_rng(SEED)
    assignment = {}

    # 1. Single-tube concentrations must remain in training.
    for concentration in singles:
        group = tubes[tubes["conc"] == concentration]

        assignment[group.iloc[0]["tube_id"]] = "train"

    # 2. Three-tube concentrations: one tube per partition.
    # Rotate instance order to distribute preparation instances.
    for index, concentration in enumerate(triples):

        group = (
            tubes[tubes["conc"] == concentration]
            .sort_values("instance_id")
        )

        members = group["tube_id"].tolist()

        if set(group["instance_id"]) != {1, 2, 3}:
            raise ValueError(
                f"{concentration}% does not contain one tube "
                "from each instance"
            )

        shift = index % 3
        members = members[shift:] + members[:shift]

        for tube_id, role in zip(
            members, ["train", "validation", "test"]
        ):
            assignment[tube_id] = role

    # 3. Choose four two-tube concentration levels to keep
    # entirely in training. They are spread over the range.
    positions = np.round(
        np.linspace(0, len(pairs) - 1, 4)
    ).astype(int)

    train_only_levels = {
        pairs[int(position)] for position in positions
    }

    remaining_pairs = [
        c for c in pairs if c not in train_only_levels
    ]

    validation_levels = remaining_pairs[::2]
    test_levels = remaining_pairs[1::2]

    if len(validation_levels) != 6 or len(test_levels) != 6:
        raise ValueError("Unexpected validation/test concentration allocation")

    # The 0.25% pair is from instances 1 and 3.
    if 0.25 not in validation_levels:
        raise ValueError("The expected 0.25% allocation changed")

    # Pick held-out instance identities with seed 42.
    # This gives balanced instance counts in validation and test.
    validation_other = [
        c for c in validation_levels if c != 0.25
    ]

    validation_instance1 = set(
        rng.choice(
            validation_other, size=3, replace=False
        ).tolist()
    )

    test_instance1 = set(
        rng.choice(
            test_levels, size=3, replace=False
        ).tolist()
    )

    # 4. Assign the two-tube concentration groups.
    for concentration in pairs:

        group = (
            tubes[tubes["conc"] == concentration]
            .sort_values("instance_id")
        )

        members = group.to_dict("records")

        if concentration in train_only_levels:
            # Both physical tubes stay in training.
            for member in members:
                assignment[member["tube_id"]] = "train"

            continue

        if np.isclose(concentration, 0.25):
            held_instance = 3
            held_role = "validation"

        elif concentration in validation_levels:
            held_instance = (
                1 if concentration in validation_instance1 else 2
            )
            held_role = "validation"

        elif concentration in test_levels:
            held_instance = (
                1 if concentration in test_instance1 else 2
            )
            held_role = "test"

        else:
            raise ValueError(
                f"Unexpected paired concentration: {concentration}"
            )

        held = [
            member for member in members
            if member["instance_id"] == held_instance
        ]

        if len(held) != 1:
            raise ValueError(
                f"Could not identify held-out tube at {concentration}%"
            )

        held_id = held[0]["tube_id"]

        other = [
            member["tube_id"]
            for member in members
            if member["tube_id"] != held_id
        ]

        if len(other) != 1:
            raise ValueError("Unexpected two-tube group")

        assignment[held_id] = held_role
        assignment[other[0]] = "train"

    if set(assignment) != set(tubes["tube_id"]):
        raise ValueError("Some tubes have not been assigned")

    roles = tubes.copy()
    roles["role"] = roles["tube_id"].map(assignment)

    return roles


def validate_roles(roles, tubes):

    if roles["tube_id"].duplicated().any():
        raise ValueError("Duplicate tube IDs in split file")

    if set(roles["tube_id"]) != set(tubes["tube_id"]):
        raise ValueError("Split file does not contain exactly the current tubes")

    expected_roles = set(EXPECTED_ROLE_COUNTS)

    if set(roles["role"].unique()) != expected_roles:
        raise ValueError("Unexpected role labels")

    counts = roles["role"].value_counts().to_dict()

    if counts != EXPECTED_ROLE_COUNTS:
        raise ValueError(
            f"Expected role counts {EXPECTED_ROLE_COUNTS}, got {counts}"
        )

    # Every concentration level must occur in training.
    all_levels = set(tubes["conc"])
    train_levels = set(
        roles.loc[roles["role"] == "train", "conc"]
    )

    if train_levels != all_levels:
        raise ValueError(
            "Training partition does not cover every concentration"
        )

    # Each validation and test partition has only one tube per level.
    for role in ("validation", "test"):
        part = roles[roles["role"] == role]

        if (part.groupby("conc").size() != 1).any():
            raise ValueError(
                f"{role} contains multiple tubes at one concentration"
            )

    # The four three-tube concentration levels are shared by validation
    # and test. The paired concentration levels are disjoint.
    val_levels = set(
        roles.loc[roles["role"] == "validation", "conc"]
    )
    test_levels = set(
        roles.loc[roles["role"] == "test", "conc"]
    )

    if val_levels.intersection(test_levels) != {
        0.50, 1.00, 1.50, 2.00
    }:
        raise ValueError(
            "Unexpected overlap between validation and test concentrations"
        )


def load_or_create_roles(tubes):

    proposed = assign_roles(tubes)

    if ROLES_PATH.exists():
        existing = pd.read_csv(ROLES_PATH)

        if not {"tube_id", "conc", "role"}.issubset(existing.columns):
            raise ValueError(
                "Existing tube_roles.csv has an unexpected structure"
            )

        if set(existing["role"].dropna()) != set(EXPECTED_ROLE_COUNTS):
            raise RuntimeError(
                "An older split file exists (possibly using dev/test labels). "
                "The script will not overwrite it automatically. "
                "Back it up and remove it only if the old test set was never "
                "inspected or used."
            )

        comparison = proposed[["tube_id", "conc", "role"]].merge(
            existing[["tube_id", "conc", "role"]],
            on="tube_id",
            suffixes=("_new", "_saved"),
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        same = (
            comparison["_merge"].eq("both").all()
            and np.allclose(
                comparison["conc_new"],
                comparison["conc_saved"],
            )
            and comparison["role_new"].equals(
                comparison["role_saved"]
            )
        )

        if not same:
            raise RuntimeError(
                "An existing locked split differs from the proposed split. "
                "It will not be overwritten automatically."
            )

        print("Existing 60/20/20 split found. Reusing it.")
        roles = existing.merge(
            tubes[["tube_id", "instance_id"]],
            on="tube_id",
            how="left",
            validate="one_to_one",
        )

    else:
        roles = proposed
        ROLES_PATH.parent.mkdir(parents=True, exist_ok=True)
        roles.to_csv(ROLES_PATH, index=False)
        print("New split saved:", ROLES_PATH)

    validate_roles(roles, tubes)

    return roles


def main():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(METADATA_PATH)

    metadata = pd.read_csv(METADATA_PATH)

    tubes = build_tube_table(metadata)
    roles = load_or_create_roles(tubes)

    role_map = roles.set_index("tube_id")["role"]

    metadata["role"] = metadata["tube_id"].map(role_map)

    if metadata["role"].isna().any():
        raise ValueError("Some image rows have no assigned role")

    # All photographs of one tube must have one role.
    if (metadata.groupby("tube_id")["role"].nunique() != 1).any():
        raise ValueError("A physical tube appears in multiple partitions")

    metadata.to_csv(METADATA_PATH, index=False)

    print("\nTUBE-LEVEL SPLIT SUMMARY")
    print("=" * 45)

    print("\nPhysical tubes by role:")
    print(roles["role"].value_counts().to_string())

    print("\nPhysical tubes by instance and role:")
    print(
        roles.groupby(["role", "instance_id"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nPhotographs by role:")
    print(metadata["role"].value_counts().to_string())

    print("\nConcentration levels:")
    print("Training:", metadata.loc[
        metadata["role"] == "train", "conc"
    ].nunique())
    print("Validation:", metadata.loc[
        metadata["role"] == "validation", "conc"
    ].nunique())
    print("Test:", metadata.loc[
        metadata["role"] == "test", "conc"
    ].nunique())

    print("\nUpdated:", METADATA_PATH)
    print("Locked split:", ROLES_PATH)
    print("\nNo photographs were opened by this script.")


if __name__ == "__main__":
    main()