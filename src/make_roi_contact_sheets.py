from pathlib import Path
import ntpath

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "roi_inspection"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [0.0, 0.15, 0.50, 1.0, 2.0]

# The four primary conditions specified in the roadmap.
CONDITIONS = {
    "dino_on": {
        "label": "Dino-Lite | Lamp ON",
        "excel": "data/raw/Dinolite/dinolite_withlight.xlsx",
        "images": "data/raw/Dinolite/withlight",
    },
    "dino_off": {
        "label": "Dino-Lite | Lamp OFF",
        "excel": "data/raw/Dinolite/dinolite_withoutlight.xlsx",
        "images": "data/raw/Dinolite/without_light",
    },
    "phone_on": {
        "label": "Samsung A31 | Lamp ON",
        "excel": "data/raw/SamsungA31/A31_withlight.xlsx",
        "images": "data/raw/SamsungA31/withlight",
    },
    "phone_off": {
        "label": "Samsung A31 | Lamp OFF",
        "excel": "data/raw/SamsungA31/A31_withoutlight.xlsx",
        "images": "data/raw/SamsungA31/withoutlight",
    },
}

CELL_W = 260
CELL_H = 270
IMAGE_W = 240
IMAGE_H = 205
HEADER_H = 45
FONT = ImageFont.load_default()


def basename_from_excel(value):
    # Excel contains Windows paths even though this script runs in WSL.
    return ntpath.basename(str(value).replace("/", "\\").strip())


def load_condition(config):
    excel_path = ROOT / config["excel"]
    image_dir = ROOT / config["images"]

    if not excel_path.is_file():
        raise FileNotFoundError(excel_path)
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    df = pd.read_excel(excel_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f"Expected path and concentration columns: {excel_path}")

    df = df.iloc[:, :2].copy()
    df.columns = ["orig_path", "conc"]
    df["conc"] = pd.to_numeric(df["conc"], errors="coerce")

    # Index only this condition's original images.
    files = [
        p for p in image_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".bmp", ".jpg", ".jpeg"}
    ]
    by_name = {p.name.casefold(): p for p in files}
    by_stem = {}
    for p in files:
        by_stem.setdefault(p.stem.casefold(), []).append(p)

    matched_paths = []
    statuses = []

    for _, row in df.iterrows():
        value = row["orig_path"]

        if pd.isna(value):
            matched_paths.append(None)
            statuses.append("missing_excel_path")
            continue

        name = basename_from_excel(value)
        exact = by_name.get(name.casefold())

        if exact is not None:
            matched_paths.append(exact)
            statuses.append("exact")
        else:
            candidates = by_stem.get(Path(name).stem.casefold(), [])
            if len(candidates) == 1:
                matched_paths.append(candidates[0])
                statuses.append("stem_only")
            else:
                matched_paths.append(None)
                statuses.append(
                    "ambiguous" if candidates else "missing_image"
                )

    df["image_path"] = matched_paths
    df["match_status"] = statuses

    print(f"\n{config['label']}")
    print(f"  Excel rows: {len(df)}")
    print(f"  Images in folder: {len(files)}")
    print(f"  Exact matches: {(df['match_status'] == 'exact').sum()}")
    print(f"  Other matches: {(df['match_status'] == 'stem_only').sum()}")
    print(f"  Unmatched rows: {df['image_path'].isna().sum()}")

    return df


def choose_examples(df, target):
    # Prefer an exact concentration; otherwise use the nearest available.
    available = sorted(df.loc[df["image_path"].notna(), "conc"].dropna().unique())
    if not available:
        return [], None

    exact_levels = [c for c in available if abs(float(c) - target) < 1e-8]
    actual = float(exact_levels[0]) if exact_levels else min(
        map(float, available), key=lambda c: abs(c - target)
    )

    group = df[
        df["image_path"].notna()
        & df["conc"].notna()
        & ((df["conc"] - actual).abs() < 1e-8)
    ]

    # Use early and late records to expose within-level/session variation.
    if len(group) <= 2:
        chosen = group
    else:
        chosen = group.iloc[[0, -1]]

    return list(chosen.iterrows()), actual


def draw_tile(canvas, x, y, record, target, actual, sample_no):
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (x + 8, y + 3),
        f"Target {target:g}% | actual {actual:g}%",
        fill="black",
        font=FONT,
    )

    image_path = record["image_path"]
    if image_path is None:
        draw.text((x + 8, y + 45), "IMAGE NOT MATCHED", fill="red", font=FONT)
        return

    try:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image.thumbnail((IMAGE_W, IMAGE_H), Image.Resampling.LANCZOS)

        image_x = x + (CELL_W - image.width) // 2
        image_y = y + 25 + (IMAGE_H - image.height) // 2
        canvas.paste(image, (image_x, image_y))

        draw.text(
            (x + 8, y + 235),
            f"Sample {sample_no}: {image_path.name[:34]}",
            fill="black",
            font=FONT,
        )
        draw.text(
            (x + 8, y + 250),
            f"Concentration: {float(record['conc']):g}%",
            fill="black",
            font=FONT,
        )
    except Exception as exc:
        draw.text((x + 8, y + 45), f"ERROR: {exc}", fill="red", font=FONT)


for key, config in CONDITIONS.items():
    df = load_condition(config)

    canvas = Image.new(
        "RGB",
        (5 * CELL_W, HEADER_H + 2 * CELL_H),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 12), config["label"], fill="black", font=FONT)

    for col, target in enumerate(TARGETS):
        examples, actual = choose_examples(df, target)

        if actual is None:
            continue

        if abs(actual - target) > 1e-8:
            print(
                f"  NOTE: {target:g}% unavailable; "
                f"using nearest concentration {actual:g}%"
            )

        for row in range(2):
            x = col * CELL_W
            y = HEADER_H + row * CELL_H

            draw.rectangle(
                (x, y, x + CELL_W - 1, y + CELL_H - 1),
                outline="#cccccc",
            )

            if row < len(examples):
                _, record = examples[row]
                draw_tile(
                    canvas, x, y, record,
                    target, actual, row + 1,
                )
            else:
                draw.text(
                    (x + 8, y + 45),
                    "No second image available",
                    fill="gray",
                    font=FONT,
                )

    output = OUT / f"{key}_contact_sheet.jpg"
    canvas.save(output, quality=92)
    print(f"  Saved: {output.relative_to(ROOT)}")

print("\nContact sheets generated. No raw files were changed.")
