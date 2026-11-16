"""Final Phase 2 integrity audit for the urea-droplet-ml metadata.

Run from any working directory:
    python src/check_phase2_integrity.py

The script expects data/processed/metadata.csv and the original data/raw tree.
It writes reports/phase2_integrity_report.txt. Known count anomalies are
reported for review; they do not automatically invalidate the whole dataset.
"""

from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / "data" / "processed" / "metadata.csv"
REPORT = ROOT / "reports" / "phase2_integrity_report.txt"

INSTANCE_BY_COHORT = {
    "original": 1,
    "new_tubes": 2,
    "nov4_validation": 3,
}
EXPECTED_ROWS_BY_COHORT = {
    "original": 576,
    "new_tubes": 456,
    "nov4_validation": 32,
}
EXPECTED_TUBES_BY_INSTANCE = {1: 24, 2: 19, 3: 8}
EXPECTED_CONCENTRATIONS = {
    "original": sorted(
        {0.0, 0.15}
        | {round(0.20 + 0.05 * i, 2) for i in range(7)}
        | {round(0.60 + 0.10 * i, 2) for i in range(15)}
    ),
    "new_tubes": [round(0.20 + 0.10 * i, 2) for i in range(19)],
    "nov4_validation": [0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00],
}
EXPECTED_CONDITION_COUNTS = {
    ("original", "dino", "off"): 140,
    ("original", "dino", "on"): 143,
    ("original", "phone", "off"): 146,
    ("original", "phone", "on"): 147,
    ("new_tubes", "dino", "off"): 114,
    ("new_tubes", "dino", "on"): 114,
    ("new_tubes", "phone", "off"): 114,
    ("new_tubes", "phone", "on"): 114,
    ("nov4_validation", "dino", "off"): 16,
    ("nov4_validation", "dino", "on"): 16,
}


def assert_or_record(condition, message, errors):
    if not condition:
        errors.append(message)


def concentration_list(values):
    return sorted(round(float(x), 6) for x in values.dropna().unique())


def main():
    lines = []
    errors = []
    warnings = []

    def log(message=""):
        lines.append(str(message))

    if not METADATA.is_file():
        raise FileNotFoundError(f"Metadata not found: {METADATA}")

    df = pd.read_csv(METADATA, dtype={"timestamp": "string"})
    required = {
        "image_id", "image_path", "conc", "device", "lamp", "timestamp",
        "seq", "tube_id", "cohort", "instance_id", "source_excel",
        "source_excel_row",
    }
    missing_columns = required - set(df.columns)
    assert_or_record(
        not missing_columns,
        f"Missing columns: {sorted(missing_columns)}",
        errors
    )
    if missing_columns:
        raise ValueError("Cannot continue without required metadata columns")

    log("PHASE 2 INTEGRITY REPORT")
    log("=" * 72)
    log(f"Metadata file: {METADATA.relative_to(ROOT)}")
    log(f"Rows: {len(df)}")
    log(f"Unique image IDs: {df['image_id'].nunique(dropna=True)}")
    log(f"Unique image paths: {df['image_path'].nunique(dropna=True)}")
    log("")

    # Row count and key uniqueness.
    assert_or_record(
        len(df) == 1064,
        f"Expected 1064 rows; found {len(df)}",
        errors
    )

    for col in ("image_id", "image_path"):
        assert_or_record(
            df[col].notna().all(),
            f"{col} contains missing values",
            errors
        )
        assert_or_record(
            not df[col].duplicated().any(),
            f"{col} contains duplicates",
            errors
        )

    # Cohort and instance map.
    observed_cohorts = set(df["cohort"].dropna().unique())
    assert_or_record(
        observed_cohorts == set(INSTANCE_BY_COHORT),
        f"Unexpected cohort set: {sorted(observed_cohorts)}",
        errors
    )

    for cohort, expected_rows in EXPECTED_ROWS_BY_COHORT.items():
        part = df[df["cohort"] == cohort]

        assert_or_record(
            len(part) == expected_rows,
            f"{cohort}: expected {expected_rows} rows; found {len(part)}",
            errors
        )

        if not part.empty:
            expected_instance = INSTANCE_BY_COHORT[cohort]

            assert_or_record(
                part["instance_id"].nunique() == 1
                and int(part["instance_id"].iloc[0]) == expected_instance,
                f"{cohort}: incorrect instance_id assignment",
                errors
            )

            actual_levels = concentration_list(part["conc"])
            expected_levels = EXPECTED_CONCENTRATIONS[cohort]

            assert_or_record(
                actual_levels == expected_levels,
                f"{cohort}: concentration grid mismatch; "
                f"expected {expected_levels}, found {actual_levels}",
                errors
            )

    log("Rows by cohort:")
    for cohort, count in df.groupby("cohort").size().items():
        log(f"  {cohort}: {count}")
    log("")

    # Tube IDs must encode one instance and concentration only.
    if (
        df["instance_id"].isna().any()
        or df["conc"].isna().any()
        or df["tube_id"].isna().any()
    ):
        errors.append("instance_id, conc or tube_id contains missing values")
    else:
        expected_tube_ids = [
            f"I{int(instance)}_C{float(conc):.2f}"
            for instance, conc in zip(df["instance_id"], df["conc"])
        ]

        assert_or_record(
            expected_tube_ids == df["tube_id"].astype(str).tolist(),
            "Some tube_id values do not match instance_id + concentration",
            errors
        )

        tube_check = df.groupby("tube_id").agg(
            instance_count=("instance_id", "nunique"),
            concentration_count=("conc", "nunique"),
        )

        assert_or_record(
            (tube_check["instance_count"] == 1).all(),
            "At least one tube_id maps to multiple instances",
            errors
        )

        assert_or_record(
            (tube_check["concentration_count"] == 1).all(),
            "At least one tube_id maps to multiple concentrations",
            errors
        )

        observed_tubes = (
            df.groupby("instance_id")["tube_id"].nunique().to_dict()
        )

        assert_or_record(
            observed_tubes == EXPECTED_TUBES_BY_INSTANCE,
            f"Tube counts by instance mismatch: {observed_tubes}",
            errors
        )

        log(f"Unique tubes: {df['tube_id'].nunique()}")
        log(f"Unique tubes by instance: {observed_tubes}")
        log("")

    # Workbook row provenance: every workbook row should appear once.
    source_rows = df.groupby("source_excel").size()
    duplicate_source_rows = df.duplicated(
        ["source_excel", "source_excel_row"]
    ).sum()

    assert_or_record(
        duplicate_source_rows == 0,
        f"Found {duplicate_source_rows} duplicate source Excel row references",
        errors
    )

    log("Rows by source workbook:")

    for source, count in source_rows.items():
        workbook = ROOT / source
        log(
            f"  {source}: {count} metadata rows; "
            f"workbook exists={workbook.is_file()}"
        )

        assert_or_record(
            workbook.is_file(),
            f"Missing source workbook: {source}",
            errors
        )

        if workbook.is_file():
            try:
                workbook_rows = len(
                    pd.read_excel(workbook, header=None)
                )

                assert_or_record(
                    workbook_rows == count,
                    f"{source}: workbook has {workbook_rows} rows, "
                    f"metadata has {count}",
                    errors
                )
            except Exception as exc:
                errors.append(
                    f"Could not read workbook {source}: {exc}"
                )

    log("")

    # Device/lamp totals.
    observed_conditions = (
        df.groupby(["cohort", "device", "lamp"]).size().to_dict()
    )

    log("Rows by cohort/device/lamp:")

    for key, expected_count in EXPECTED_CONDITION_COUNTS.items():
        actual_count = int(observed_conditions.get(key, 0))

        log(
            f"  {key}: {actual_count} (expected {expected_count})"
        )

        assert_or_record(
            actual_count == expected_count,
            f"Condition count mismatch for {key}: "
            f"{actual_count} vs {expected_count}",
            errors
        )

    assert_or_record(
        set(observed_conditions) == set(EXPECTED_CONDITION_COUNTS),
        "Observed cohort/device/lamp combinations differ "
        "from the expected set",
        errors
    )

    log("")

    # Per-concentration image counts. Irregular original-cohort counts are
    # preserved and reported rather than padded or deleted. Also check missing
    # combinations explicitly, so a zero-count condition cannot go unnoticed.
    counts = df.groupby(
        ["cohort", "conc", "device", "lamp"]
    ).size()

    condition_combos = {
        cohort: [
            (device, lamp)
            for c, device, lamp in EXPECTED_CONDITION_COUNTS
            if c == cohort
        ]
        for cohort in EXPECTED_ROWS_BY_COHORT
    }

    irregular = []

    for cohort, levels in EXPECTED_CONCENTRATIONS.items():
        typical = 6 if cohort in ("original", "new_tubes") else 2

        for conc in levels:
            for device, lamp in condition_combos[cohort]:
                count = int(
                    counts.get((cohort, conc, device, lamp), 0)
                )

                if count != typical:
                    irregular.append(
                        (cohort, float(conc), device, lamp, count, typical)
                    )

    log("Per-concentration counts differing from the usual condition count:")

    if irregular:
        for cohort, conc, device, lamp, count, typical in irregular:
            msg = (
                f"  {cohort}, {conc:.2f}%, {device}, {lamp}: "
                f"{count} images (usual count {typical})"
            )
            log(msg)
            warnings.append(msg)
    else:
        log("  None")

    log("")

    # Timestamps must retain seconds in ISO format when available.
    timestamp_present = (
        df["timestamp"].notna()
        & df["timestamp"].astype(str).ne("")
    )

    parsed = pd.to_datetime(
        df.loc[timestamp_present, "timestamp"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce",
    )

    bad_timestamp_count = int(parsed.isna().sum())

    assert_or_record(
        bad_timestamp_count == 0,
        f"{bad_timestamp_count} nonblank timestamps "
        "are not ISO seconds precision",
        errors
    )

    missing_timestamp_count = int((~timestamp_present).sum())

    log(f"Images with timestamps: {int(timestamp_present.sum())}")
    log(f"Images without timestamps: {missing_timestamp_count}")

    if missing_timestamp_count:
        log("Missing timestamps by cohort/device/lamp:")

        for key, count in (
            df.loc[~timestamp_present]
            .groupby(["cohort", "device", "lamp"])
            .size()
            .items()
        ):
            log(f"  {key}: {count}")

    log("")

    # Duplicate sequence numbers are a diagnostic, not a duplicate-image error.
    dino = df[
        (df["device"] == "dino") & df["seq"].notna()
    ]

    repeated = dino[
        dino.duplicated(
            ["cohort", "lamp", "seq"],
            keep=False
        )
    ]

    log("Repeated Dino-Lite sequence numbers (inspect; IDs/paths remain unique):")

    if repeated.empty:
        log("  None")
    else:
        for _, row in repeated.sort_values(
            ["cohort", "lamp", "seq", "timestamp"]
        ).iterrows():
            log(
                f"  {row['image_id']} | {row['conc']}% | {row['lamp']} | "
                f"A{int(row['seq']):03d} | {row['timestamp']} | "
                f"{row['image_path']}"
            )

    log("")

    # Original images must exist and decode. This replaces the PNG alpha-channel
    # check from the original roadmap because this project uses original photos.
    unreadable = []
    missing_images = []

    for rel_path in df["image_path"].astype(str):
        path = (ROOT / rel_path).resolve()

        try:
            path.relative_to(ROOT.resolve())
        except ValueError:
            missing_images.append(
                (rel_path, "path escapes project root")
            )
            continue

        if not path.is_file():
            missing_images.append((rel_path, "file not found"))
            continue

        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                image.load()

        except Exception as exc:
            unreadable.append((rel_path, str(exc)))

    assert_or_record(
        not missing_images,
        f"{len(missing_images)} image paths are missing "
        "or outside project root",
        errors
    )

    assert_or_record(
        not unreadable,
        f"{len(unreadable)} original images could not be decoded",
        errors
    )

    log(f"Original images missing/outside project root: {len(missing_images)}")
    log(f"Original images unreadable: {len(unreadable)}")

    for item in missing_images[:20]:
        log(f"  MISSING: {item[0]} ({item[1]})")

    for item in unreadable[:20]:
        log(f"  UNREADABLE: {item[0]} ({item[1]})")

    log("")

    # Droplet and view identities are intentionally not required by the agreed
    # tube-level split protocol.
    log(
        f"Blank droplet_id rows: "
        f"{int(df['droplet_id'].isna().sum()) if 'droplet_id' in df else 'column absent'}"
    )
    log(
        f"Blank view_id rows: "
        f"{int(df['view_id'].isna().sum()) if 'view_id' in df else 'column absent'}"
    )
    log("These identifiers are intentionally unpopulated; evaluation groups by tube_id.")
    log("")

    log("WARNINGS / ITEMS TO DOCUMENT")
    log("- Irregular per-concentration image counts are not silently corrected.")
    log("- Repeated A056 sequence numbers refer to distinct paths/timestamps; verify/document from lab records if possible.")
    log("- Missing timestamps remain missing; do not infer them from neighboring photos.")
    log("")

    log("ERRORS")

    if errors:
        for error in errors:
            log(f"- {error}")
    else:
        log("None")

    log("")

    log(
        "STATUS: "
        + (
            "FAIL — fix errors before declaring Phase 2 complete"
            if errors
            else "PASS — structural integrity checks passed"
        )
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("\n".join(lines))
    print(f"\nReport saved to: {REPORT.relative_to(ROOT)}")

    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()