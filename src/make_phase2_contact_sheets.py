from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
META_PATH = ROOT / "data/processed/metadata.csv"
OUT = ROOT / "reports/roi_inspection/all_concentrations"
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(META_PATH)
font = ImageFont.load_default()

# Fixed order makes the sheets easy to compare.
CONDITION_ORDER = [
    ("dino", "on"),
    ("dino", "off"),
    ("phone", "on"),
    ("phone", "off"),
]

LABELS = {
    ("dino", "on"): "Dino-Lite | Lamp ON",
    ("dino", "off"): "Dino-Lite | Lamp OFF",
    ("phone", "on"): "Samsung A31 | Lamp ON",
    ("phone", "off"): "Samsung A31 | Lamp OFF",
}

CELL_W = 250
CELL_H = 210
THUMB_W = 230
THUMB_H = 160
HEADER_H = 55

index_rows = []


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


for (cohort, conc), group in meta.groupby(
    ["cohort", "conc"], sort=True
):
    # The November validation cohort contains Dino-Lite only.
    conditions = (
        [("dino", "on"), ("dino", "off")]
        if cohort == "nov4_validation"
        else CONDITION_ORDER
    )

    condition_groups = {}

    for device, lamp in conditions:
        selection = group[
            (group["device"] == device)
            & (group["lamp"] == lamp)
        ].copy()

        selection = selection.sort_values(
            ["timestamp", "image_id"], na_position="last"
        )

        condition_groups[(device, lamp)] = selection

    max_rows = max(
        1,
        max(len(g) for g in condition_groups.values())
    )

    width = len(conditions) * CELL_W
    height = HEADER_H + max_rows * CELL_H

    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)

    title = f"{cohort} | concentration = {conc:g}% | {len(group)} images"
    draw.text((8, 8), title, fill="black", font=font)

    for column, condition in enumerate(conditions):
        selection = condition_groups[condition]
        x = column * CELL_W

        draw.rectangle(
            (x, HEADER_H - 1, x + CELL_W - 1, height - 1),
            outline="#bbbbbb",
        )
        draw.text(
            (x + 6, HEADER_H + 5),
            f"{LABELS[condition]} (n={len(selection)})",
            fill="black",
            font=font,
        )

        for row_number, (_, record) in enumerate(selection.iterrows()):
            y = HEADER_H + row_number * CELL_H + 28
            tile_x = x + 10

            image_path = ROOT / str(record["image_path"])

            try:
                with Image.open(image_path) as im:
                    im = im.convert("RGB")
                    im.thumbnail(
                        (THUMB_W, THUMB_H),
                        Image.Resampling.LANCZOS,
                    )
                    px = tile_x + (THUMB_W - im.width) // 2
                    sheet.paste(im, (px, y))

                label_y = y + THUMB_H + 3
                filename = image_path.name

                # Keep the name identifiable without crowding the tile.
                if len(filename) > 30:
                    filename = filename[:27] + "..."

                draw.text(
                    (tile_x, label_y),
                    filename,
                    fill="black",
                    font=font,
                )
                draw.text(
                    (tile_x, label_y + 14),
                    f"seq={record['seq']}, "
                    f"time={record['timestamp'] or 'unknown'}",
                    fill="black",
                    font=font,
                )

            except Exception as exc:
                draw.text(
                    (tile_x, y + 20),
                    f"ERROR opening image: {exc}",
                    fill="red",
                    font=font,
                )

    output_name = safe_name(
        f"{cohort}_conc_{conc:.2f}.jpg"
    )
    output_path = OUT / output_name
    sheet.save(output_path, quality=88)

    index_rows.append({
        "cohort": cohort,
        "conc": conc,
        "image_count": len(group),
        "contact_sheet": output_path.relative_to(ROOT).as_posix(),
    })

index_path = OUT / "index.csv"
pd.DataFrame(index_rows).to_csv(index_path, index=False)

print(f"Contact sheets created: {len(index_rows)}")
print(f"Images represented: {len(meta)}")
print(f"Output directory: {OUT}")
print(f"Index: {index_path}")
print("\nSheets per cohort:")
print(pd.DataFrame(index_rows).groupby("cohort").size().to_string())
