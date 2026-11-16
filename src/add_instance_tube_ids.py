
from pathlib import Path, PurePosixPath

import pandas as pd


# Project root: urea-droplet-ml/
ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "data" / "processed" / "metadata.csv"
OUTPUT = ROOT / "data" / "processed" / "metadata_with_tube_ids.csv"


def get_instance_id(image_path):
    """
    Determine the preparation instance from the image's path.

    Instance 1:
        data/raw/Dinolite/ (excluding validation data)
        data/raw/SamsungA31/

    Instance 2:
        data/raw/new_data/

    Instance 3:
        data/raw/Dinolite/4thnov_validationdata/
    """

    if pd.isna(image_path):
        raise ValueError("Found an image with a missing image_path")

    # Normalize Windows and Linux paths
    normalized = str(image_path).replace("\\", "/")

    parts = {
        part.casefold()
        for part in PurePosixPath(normalized).parts
    }

    # Check instance 2
    if "new_data" in parts:
        return 2

    # Check instance 3 before the general Dinolite rule
    if "dinolite" in parts and "4thnov_validationdata" in parts:
        return 3

    # Check instance 1
    if "dinolite" in parts or "samsunga31" in parts:
        return 1

    raise ValueError(
        f"Could not determine instance for image path: {image_path}"
    )


def main():

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {INPUT}"
        )

    df = pd.read_csv(INPUT)

    # Verify required columns
    required_columns = {"image_path", "conc"}

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Metadata is missing these columns: {missing}"
        )

    # Assign instance IDs from the image paths
    df["instance_id"] = df["image_path"].apply(get_instance_id)

    # Convert concentration to numeric
    concentrations = pd.to_numeric(
        df["conc"],
        errors="coerce"
    )

    if concentrations.isna().any():
        raise ValueError(
            "Some concentration values are missing or invalid"
        )

    # Generate tube IDs using instance and concentration
    df["tube_id"] = [
        f"I{instance}_C{concentration:.2f}"
        for instance, concentration in zip(
            df["instance_id"],
            concentrations
        )
    ]

    # Validate that every tube ID represents one instance
    # and one concentration
    validation = df.assign(
        numeric_conc=concentrations
    ).groupby("tube_id").agg(
        num_instances=("instance_id", "nunique"),
        num_concentrations=("numeric_conc", "nunique")
    )

    if (validation["num_instances"] != 1).any():
        raise ValueError(
            "A tube ID is associated with multiple instances"
        )

    if (validation["num_concentrations"] != 1).any():
        raise ValueError(
            "A tube ID is associated with multiple concentrations"
        )

    # Save without modifying the original metadata
    df.to_csv(OUTPUT, index=False)

    # Print summary
    print("\nMetadata processing completed.")

    print(f"Total images: {len(df)}")
    print(f"Unique tubes: {df['tube_id'].nunique()}")

    print("\nImages by instance:")
    print(df["instance_id"].value_counts().sort_index())

    print("\nUnique tubes by instance:")
    print(
        df.groupby("instance_id")["tube_id"]
        .nunique()
        .to_string()
    )

    print("\nTube IDs:")
    tubes = (
        df[["instance_id", "conc", "tube_id"]]
        .drop_duplicates()
        .sort_values(["instance_id", "conc"])
    )

    print(tubes.to_string(index=False))

    print(f"\nOutput saved to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
