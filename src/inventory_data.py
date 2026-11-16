from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
import ntpath

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

IMAGE_EXTENSIONS = {
    ".bmp", ".jpg", ".jpeg", ".png",
    ".tif", ".tiff", ".webp", ".gif", ".jfif",
}
EXCEL_EXTENSIONS = {".xlsx", ".xls"}

REPORTS.mkdir(parents=True, exist_ok=True)

if not RAW.is_dir():
    raise FileNotFoundError(f"Raw data directory not found: {RAW}")


def relative(path):
    return path.relative_to(ROOT).as_posix()


def windows_basename(value):
    """Extract a filename from either Windows or Unix-style paths."""
    return ntpath.basename(str(value).replace("/", "\\"))


# ------------------------------------------------------------
# 1. Recursively find EVERY image under data/raw.
# ------------------------------------------------------------
images = sorted(
    p for p in RAW.rglob("*")
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)

by_name = defaultdict(list)
by_stem = defaultdict(list)

for path in images:
    by_name[path.name.casefold()].append(path)
    by_stem[path.stem.casefold()].append(path)

image_records = []
readable_count = 0
unreadable_count = 0

for path in images:
    record = {
        "image_path": relative(path),
        "folder": relative(path.parent),
        "source_group": path.relative_to(RAW).parts[0],
        "filename": path.name,
        "stem": path.stem,
        "extension": path.suffix.lower(),
        "size_bytes": path.stat().st_size,
        "readable": False,
        "width": None,
        "height": None,
        "mode": None,
        "has_alpha": None,
        "error": "",
    }

    try:
        with Image.open(path) as image:
            image.load()
            record["width"], record["height"] = image.size
            record["mode"] = image.mode
            record["has_alpha"] = (
                "A" in image.getbands()
                or "transparency" in image.info
            )
            record["readable"] = True
            readable_count += 1
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        unreadable_count += 1

    image_records.append(record)

image_df = pd.DataFrame(image_records)
image_df.to_csv(REPORTS / "raw_image_inventory.csv", index=False)


# ------------------------------------------------------------
# 2. Summarize images by exact parent folder.
# Each image is counted once in this folder summary.
# ------------------------------------------------------------
folder_counts = Counter(relative(p.parent) for p in images)
extension_counts = Counter(p.suffix.lower() for p in images)
source_counts = Counter(
    p.relative_to(RAW).parts[0] for p in images
)

print("=" * 72)
print("COMPLETE RAW IMAGE INVENTORY")
print("=" * 72)
print(f"Raw directory: {RAW}")
print(f"Total images: {len(images)}")
print(f"Readable images: {readable_count}")
print(f"Unreadable images: {unreadable_count}")

print("\nImages by top-level source:")
for source, count in sorted(source_counts.items()):
    print(f"  {source}: {count}")

print("\nImages by extension:")
for ext, count in sorted(extension_counts.items()):
    print(f"  {ext}: {count}")

print("\nImages by exact parent folder:")
for folder, count in sorted(folder_counts.items()):
    print(f"  {folder}: {count}")

# Check for duplicate filenames and duplicate stems across folders.
duplicate_names = {
    name: paths for name, paths in by_name.items() if len(paths) > 1
}
duplicate_stems = {
    stem: paths for stem, paths in by_stem.items() if len(paths) > 1
}

print(f"\nRepeated filenames across folders: {len(duplicate_names)}")
print(f"Repeated filename stems across folders: {len(duplicate_stems)}")

for name, paths in sorted(duplicate_names.items()):
    print(f"  DUPLICATE NAME: {name}")
    for path in paths:
        print(f"    {relative(path)}")

# Report PNGs without alpha. This is a diagnostic only;
# the Phase 2 requirement applies to the cut-out PNGs.
raw_pngs = [r for r in image_records if r["extension"] == ".png"]
png_without_alpha = [
    r for r in raw_pngs
    if r["readable"] and not r["has_alpha"]
]
print(f"\nPNG files under data/raw/: {len(raw_pngs)}")
print(f"Readable PNGs without alpha: {len(png_without_alpha)}")


# ------------------------------------------------------------
# 3. Inventory every Excel workbook and match its image paths
# against every image found anywhere under data/raw.
# Exact filename matching is preferred; stem matching is a
# diagnostic for differences such as .png vs .jpg.
# ------------------------------------------------------------
excel_files = sorted(
    p for p in RAW.rglob("*")
    if p.is_file()
    and p.suffix.lower() in EXCEL_EXTENSIONS
    and not p.name.startswith("~$")
)

workbook_records = []
mapping_records = []

for workbook in excel_files:
    try:
        df = pd.read_excel(workbook, header=None)

        workbook_records.append({
            "excel_path": relative(workbook),
            "rows": len(df),
            "columns": len(df.columns),
            "expected_two_columns": len(df.columns) == 2,
            "error": "",
        })

        print(f"\n{'-' * 72}")
        print(f"EXCEL: {relative(workbook)}")
        print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        print("First five rows:")
        print(df.head(5).to_string(index=False, header=False))

        if len(df.columns) < 2:
            print("WARNING: fewer than two columns; cannot map path and concentration.")
            continue

        exact_count = 0
        stem_count = 0
        missing_count = 0
        ambiguous_count = 0

        for row_idx, row in df.iterrows():
            raw_path = row.iloc[0]
            concentration = row.iloc[1]

            if pd.isna(raw_path):
                excel_path_value = ""
                basename = ""
                exact_matches = []
                stem_matches = []
                status = "missing_excel_path"
            else:
                excel_path_value = str(raw_path).strip()
                basename = windows_basename(excel_path_value)

                exact_matches = by_name.get(basename.casefold(), [])
                stem_matches = by_stem.get(
                    Path(basename).stem.casefold(), []
                )

                if len(exact_matches) == 1:
                    status = "exact_match"
                    exact_count += 1
                elif len(exact_matches) > 1:
                    status = "ambiguous_exact_match"
                    ambiguous_count += 1
                elif len(stem_matches) == 1:
                    status = "stem_match_extension_differs"
                    stem_count += 1
                elif len(stem_matches) > 1:
                    status = "ambiguous_stem_match"
                    ambiguous_count += 1
                else:
                    status = "no_match_in_raw"
                    missing_count += 1

            matches = exact_matches if exact_matches else stem_matches

            mapping_records.append({
                "excel_path": relative(workbook),
                "excel_row": int(row_idx) + 1,
                "original_path": excel_path_value,
                "excel_filename": basename,
                "concentration": concentration,
                "match_status": status,
                "matching_image_count": len(matches),
                "matching_image_paths": " ; ".join(
                    relative(p) for p in matches
                ),
            })

        print(
            "Mapping summary: "
            f"exact={exact_count}, "
            f"stem_only={stem_count}, "
            f"missing={missing_count}, "
            f"ambiguous={ambiguous_count}"
        )

    except Exception as exc:
        workbook_records.append({
            "excel_path": relative(workbook),
            "rows": None,
            "columns": None,
            "expected_two_columns": False,
            "error": f"{type(exc).__name__}: {exc}",
        })
        print(f"ERROR reading {relative(workbook)}: {exc}")


pd.DataFrame(workbook_records).to_csv(
    REPORTS / "excel_inventory.csv", index=False
)
pd.DataFrame(mapping_records).to_csv(
    REPORTS / "excel_image_mapping.csv", index=False
)

print("\n" + "=" * 72)
print("REPORTS CREATED")
print("=" * 72)
print(f"  {relative(REPORTS / 'raw_image_inventory.csv')}")
print(f"  {relative(REPORTS / 'excel_inventory.csv')}")
print(f"  {relative(REPORTS / 'excel_image_mapping.csv')}")
print("\nNo source images or Excel files were modified.")
