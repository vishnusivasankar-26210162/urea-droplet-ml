import re
import hashlib
import ntpath
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ExifTags

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "metadata.csv"
MAPPING_REPORT = ROOT / "reports" / "excel_image_mapping.csv"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Every workbook maps to the ORIGINAL photograph directory.
# orig_path preserves exactly what the source Excel file recorded.
#
# lamp_mode:
#   fixed     = lamp condition is specified by the workbook
#   block12   = six lamp-on, then six lamp-off photos per concentration
#   block4    = two lamp-on, then two lamp-off photos per concentration

SOURCES = [
    {
        "excel": "data/raw/Dinolite/dinolite_withlight.xlsx",
        "image_dir": "data/raw/Dinolite/withlight",
        "cohort": "original",
        "device": "dino",
        "fixed_lamp": "on",
        "lamp_mode": "fixed",
    },
    {
        "excel": "data/raw/Dinolite/dinolite_withoutlight.xlsx",
        "image_dir": "data/raw/Dinolite/without_light",
        "cohort": "original",
        "device": "dino",
        "fixed_lamp": "off",
        "lamp_mode": "fixed",
    },
    {
        "excel": "data/raw/SamsungA31/A31_withlight.xlsx",
        "image_dir": "data/raw/SamsungA31/withlight",
        "cohort": "original",
        "device": "phone",
        "fixed_lamp": "on",
        "lamp_mode": "fixed",
    },
    {
        "excel": "data/raw/SamsungA31/A31_withoutlight.xlsx",
        "image_dir": "data/raw/SamsungA31/withoutlight",
        "cohort": "original",
        "device": "phone",
        "fixed_lamp": "off",
        "lamp_mode": "fixed",
    },
    {
        "excel": "data/raw/new_data/dino_proc_img.xlsx",
        "image_dir": "data/raw/new_data/dinolite/img",
        "cohort": "new_tubes",
        "device": "dino",
        "fixed_lamp": None,
        "lamp_mode": "block12",
    },
    {
        "excel": "data/raw/new_data/A31_proc_img.xlsx",
        "image_dir": "data/raw/new_data/A31/img",
        "cohort": "new_tubes",
        "device": "phone",
        "fixed_lamp": None,
        "lamp_mode": "block12",
    },
    {
        "excel": "data/raw/Dinolite/4thnov_validation.xlsx",
        "image_dir": "data/raw/Dinolite/4thnov_validationdata",
        "cohort": "nov4_validation",
        "device": "dino",
        "fixed_lamp": None,
        "lamp_mode": "block4",
    },
]

# User-confirmed preparation batches. Each (instance, concentration) is one
# physical tube; all photos from that tube share the same tube_id.
INSTANCE_BY_COHORT = {
    "original": 1,
    "new_tubes": 2,
    "nov4_validation": 3,
}

DINO_PATTERN = re.compile(
    r"A(?P<seq>\d+)[ _-]+(?P<date>\d{8})_(?P<time>\d{6})",
    re.IGNORECASE,
)
TIME_PATTERN = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})")


def rel(path):
    return path.relative_to(ROOT).as_posix()


def basename_windows(value):
    return ntpath.basename(str(value).replace("/", "\\").strip())


def concentration_key(value):
    return round(float(value), 6)


def get_exif_timestamp(path):
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            value = exif.get(36867) or exif.get(306)

            if not value:
                try:
                    nested = exif.get_ifd(ExifTags.IFD.Exif)
                    value = nested.get(36867)
                except Exception:
                    pass

        if value:
            return pd.to_datetime(
                value, format="%Y:%m:%d %H:%M:%S", errors="coerce"
            )
    except Exception:
        pass

    return pd.NaT


def parse_image_name(path):
    """Read sequence and timestamp where available; never invent either."""
    match = DINO_PATTERN.search(path.stem)
    seq = int(match.group("seq")) if match else None
    match = match or TIME_PATTERN.search(path.stem)

    if match:
        timestamp = pd.to_datetime(
            match.group("date") + match.group("time"),
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        if pd.notna(timestamp):
            return seq, timestamp, "filename"

    timestamp = get_exif_timestamp(path)
    if pd.notna(timestamp):
        return seq, timestamp, "EXIF"

    return seq, pd.NaT, "missing"


def safe_stem(path):
    return re.sub(
        r"[^A-Za-z0-9]+", "_", path.stem
    ).strip("_").lower()


def make_image_id(device, lamp, path, seq, timestamp):
    # Follow the roadmap's sequence-plus-timestamp convention for Dino-Lite.
    if device == "dino" and seq is not None and pd.notna(timestamp):
        return (
            f"dino_{lamp}_{seq:03d}_"
            f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
        )

    # Timestamp-based smartphone IDs; use the filename as a stable fallback.
    if pd.notna(timestamp):
        return f"{device}_{lamp}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    stem = safe_stem(path)
    path_key = rel(path)
    suffix = hashlib.sha1(
        path_key.encode("utf-8")
    ).hexdigest()[:8]
    return f"{device}_{lamp}_{stem}_{suffix}"


def assign_lamps(df, source):
    """Apply user-confirmed ordering for the two added cohorts."""
    mode = source["lamp_mode"]

    if mode == "fixed":
        return {
            idx: source["fixed_lamp"] for idx in df.index
        }, {
            idx: "workbook_condition" for idx in df.index
        }

    group_size = 12 if mode == "block12" else 4
    on_count = 6 if mode == "block12" else 2

    lamp_by_index = {}
    method_by_index = {}

    # Preserve the spreadsheet row order within each concentration.
    for concentration, group in df.groupby("conc", sort=False):
        indices = list(group.index)

        if len(indices) != group_size:
            raise ValueError(
                f"{source['excel']}: concentration {concentration:g}% "
                f"has {len(indices)} rows; expected {group_size}. "
                "Lamp assignment was not made. Check spreadsheet ordering."
            )

        for position, idx in enumerate(indices):
            lamp_by_index[idx] = "on" if position < on_count else "off"
            method_by_index[idx] = (
                "user_confirmed_order_within_concentration"
            )

    return lamp_by_index, method_by_index


def map_excel_to_originals(source):
    excel_path = ROOT / source["excel"]
    image_dir = ROOT / source["image_dir"]

    if not excel_path.is_file():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    df = pd.read_excel(excel_path, header=None)

    if df.shape[1] != 2:
        raise ValueError(
            f"{source['excel']}: expected two columns, found {df.shape[1]}"
        )

    df.columns = ["orig_path", "conc"]
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")

    if df["orig_path"].isna().any() or df["conc"].isna().any():
        bad_rows = df.index[
            df["orig_path"].isna() | df["conc"].isna()
        ].tolist()
        raise ValueError(
            f"{source['excel']}: missing path/concentration at "
            f"Excel rows {[i + 1 for i in bad_rows]}"
        )

    df["conc"] = df["conc"].map(concentration_key)

    lamp_by_idx, lamp_method_by_idx = assign_lamps(df, source)

    # Index all candidate original images recursively.
    image_files = sorted(
        p for p in image_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    by_name = defaultdict(list)
    by_stem = defaultdict(list)

    for p in image_files:
        by_name[p.name.casefold()].append(p)
        by_stem[p.stem.casefold()].append(p)

    rows = []
    mapping_rows = []
    used_paths = []
    instance_id = INSTANCE_BY_COHORT[source["cohort"]]

    for idx, record in df.iterrows():
        original_path = str(record["orig_path"]).strip()
        excel_name = basename_windows(original_path)

        exact = by_name.get(excel_name.casefold(), [])
        stem = by_stem.get(Path(excel_name).stem.casefold(), [])

        if len(exact) == 1:
            image_path = exact[0]
            match_method = "exact_filename"
        elif len(exact) > 1:
            raise ValueError(
                f"Ambiguous exact filename match: {source['excel']} "
                f"row {idx + 1}: {excel_name}"
            )
        elif len(stem) == 1:
            image_path = stem[0]
            match_method = "same_stem_different_extension"
        elif len(stem) > 1:
            raise ValueError(
                f"Ambiguous filename-stem match: {source['excel']} "
                f"row {idx + 1}: {excel_name}"
            )
        else:
            raise FileNotFoundError(
                f"No original image matched {excel_name} from "
                f"{source['excel']} row {idx + 1}"
            )

        seq, timestamp, timestamp_source = parse_image_name(image_path)
        lamp = lamp_by_idx[idx]

        image_id = make_image_id(
            source["device"], lamp, image_path, seq, timestamp
        )

        row = {
            "image_id": image_id,
            # Active, portable path to the original photograph.
            "image_path": rel(image_path),
            # Exact path retained from the source spreadsheet.
            # For added cohorts this points to the processed PNG listed there.
            "orig_path": original_path,
            "conc": float(record["conc"]),
            "device": source["device"],
            "lamp": lamp,
            "timestamp": (
                timestamp.strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(timestamp) else None
            ),
            "session": (
                timestamp.strftime("%Y-%m-%d")
                if pd.notna(timestamp) else None
            ),
            "seq": seq,
            "instance_id": instance_id,
            "tube_id": f"I{instance_id}_C{float(record['conc']):.2f}",
            "droplet_id": None,
            "view_id": None,
            "cohort": source["cohort"],
            "timestamp_source": timestamp_source,
            "lamp_assignment_source": lamp_method_by_idx[idx],
            "match_method": match_method,
            "source_excel": source["excel"],
            "source_excel_row": int(idx) + 1,
        }
        rows.append(row)
        used_paths.append(image_path.resolve())

        mapping_rows.append({
            "cohort": source["cohort"],
            "source_excel": source["excel"],
            "excel_row": int(idx) + 1,
            "excel_path_value": original_path,
            "excel_filename": excel_name,
            "image_path": rel(image_path),
            "match_method": match_method,
            "concentration": float(record["conc"]),
            "device": source["device"],
            "lamp": lamp,
            "lamp_assignment_source": lamp_method_by_idx[idx],
        })

    # Confirm that each source image was mapped exactly once.
    used_counts = Counter(used_paths)
    reused = [path for path, count in used_counts.items() if count != 1]
    unused = [
        p for p in image_files
        if p.resolve() not in used_counts
    ]

    if reused:
        raise ValueError(
            f"{source['excel']}: at least one original image was mapped "
            "multiple times."
        )

    if unused:
        sample = [rel(p) for p in unused[:10]]
        raise ValueError(
            f"{source['excel']}: {len(unused)} image(s) in its folder "
            f"have no spreadsheet row. Examples: {sample}"
        )

    print(
        f"{source['cohort']:16} {source['device']:5} "
        f"{len(rows):4} rows | exact="
        f"{sum(r['match_method'] == 'exact_filename' for r in rows):3} "
        f"| stem="
        f"{sum(r['match_method'] != 'exact_filename' for r in rows):3}"
    )

    return rows, mapping_rows


def main():
    all_rows = []
    all_mappings = []

    print("Building metadata for all specified cohorts\n")

    for source in SOURCES:
        rows, mappings = map_excel_to_originals(source)
        all_rows.extend(rows)
        all_mappings.extend(mappings)

    meta = pd.DataFrame(all_rows)

    # Verify the expected total from the inventory.
    if len(meta) != 1064:
        raise ValueError(
            f"Expected 1064 rows from the current inventory, got {len(meta)}. "
            "Investigate before accepting the metadata."
        )

    # A single original image must correspond to one metadata row.
    for column in ["image_path", "image_id"]:
        duplicates = meta.loc[
            meta[column].duplicated(keep=False), column
        ].tolist()
        if duplicates:
            raise ValueError(
                f"Duplicate {column} values detected: {duplicates[:10]}"
            )

    # Verify that the instance/tube hierarchy agrees with the experimental
    # design: one physical tube per concentration in each preparation instance.
    expected_instance = meta["cohort"].map(INSTANCE_BY_COHORT)
    if expected_instance.isna().any() or not (
        expected_instance.astype(int).to_numpy() == meta["instance_id"].to_numpy()
    ).all():
        raise ValueError("Cohort-to-instance assignment is inconsistent")

    tube_check = meta.groupby("tube_id").agg(
        n_instances=("instance_id", "nunique"),
        n_concentrations=("conc", "nunique"),
    )
    if (tube_check["n_instances"] != 1).any():
        raise ValueError("A tube_id is associated with multiple instances")
    if (tube_check["n_concentrations"] != 1).any():
        raise ValueError("A tube_id is associated with multiple concentrations")

    expected_tubes_by_instance = {1: 24, 2: 19, 3: 8}
    observed_tubes_by_instance = (
        meta.groupby("instance_id")["tube_id"].nunique().to_dict()
    )
    if observed_tubes_by_instance != expected_tubes_by_instance:
        raise ValueError(
            "Unexpected tube counts by instance: "
            f"{observed_tubes_by_instance}; expected "
            f"{expected_tubes_by_instance}"
        )

    print("\nRows by cohort:")
    print(meta.groupby("cohort").size().to_string())

    print("\nRows by cohort, device and lamp:")
    print(
        meta.groupby(["cohort", "device", "lamp"])
        .size()
        .to_string()
    )

    print("\nConcentration counts by cohort:")
    for cohort, group in meta.groupby("cohort", sort=False):
        print(
            f"\n{cohort}: {group['conc'].nunique()} concentration levels"
        )
        print(
            group.groupby("conc").size().to_string()
        )

    # Roadmap concentration grid for auditing the original cohort.
    expected = sorted(
        {0.0, 0.15}
        | {round(0.20 + 0.05 * i, 2) for i in range(7)}
        | {round(0.60 + 0.10 * i, 2) for i in range(15)}
    )
    original_levels = sorted(
        meta.loc[meta["cohort"] == "original", "conc"].unique()
    )
    missing_levels = [
        c for c in expected
        if not any(abs(c - v) < 1e-6 for v in original_levels)
    ]
    unexpected_levels = [
        c for c in original_levels
        if not any(abs(c - v) < 1e-6 for v in expected)
    ]

    print("\nOriginal-cohort concentration audit:")
    print(f"  Expected levels in roadmap: {len(expected)}")
    print(f"  Observed levels: {len(original_levels)}")
    print(f"  Expected levels absent: {missing_levels}")
    print(f"  Unexpected levels: {unexpected_levels}")

    # Report repeated Dino-Lite sequence numbers for inspection.
    dino = meta[meta["device"] == "dino"].dropna(subset=["seq"])
    duplicate_seq = dino[
        dino.duplicated(
            ["cohort", "lamp", "seq"], keep=False
        )
    ].sort_values(["cohort", "lamp", "seq", "timestamp"])

    print("\nRepeated Dino-Lite sequence numbers within cohort/lamp:")
    if duplicate_seq.empty:
        print("  None found")
    else:
        print(
            duplicate_seq[
                ["image_id", "cohort", "lamp", "seq",
                 "timestamp", "conc", "image_path"]
            ].to_string(index=False)
        )

    missing_time = meta[meta["timestamp"].isna()]
    print(f"\nImages with no filename/EXIF timestamp: {len(missing_time)}")
    if not missing_time.empty:
        print(
            missing_time[
                ["image_id", "cohort", "image_path"]
            ].to_string(index=False)
        )

    # Sorting is for convenient inspection, not an identity or split rule.
    meta = meta.sort_values(
        ["cohort", "device", "lamp", "timestamp", "image_id"],
        na_position="last",
    ).reset_index(drop=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    MAPPING_REPORT.parent.mkdir(parents=True, exist_ok=True)

    meta.to_csv(OUT, index=False)
    pd.DataFrame(all_mappings).to_csv(MAPPING_REPORT, index=False)

    print("\nMetadata saved:", rel(OUT))
    print("Excel mapping report saved:", rel(MAPPING_REPORT))
    print("Total metadata rows:", len(meta))
    print("Unique image IDs:", meta["image_id"].nunique())
    print("\nInstance and tube IDs were assigned from the confirmed hierarchy.")
    print("Droplet and view IDs are intentionally blank by project decision.")
    print("Do not assign dev/test roles until Phase 3.")


if __name__ == "__main__":
    main()