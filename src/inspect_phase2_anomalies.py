from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
META = pd.read_csv(ROOT / "data/processed/metadata.csv")

OUT = ROOT / "reports/roi_inspection"
OUT.mkdir(parents=True, exist_ok=True)

cases = [
    ("Dino-Lite OFF, 0.20%", "original", "dino", "off", 0.20),
    ("Dino-Lite OFF, 0.40%", "original", "dino", "off", 0.40),
    ("Dino-Lite ON, 0.45%", "original", "dino", "on", 0.45),
    ("Phone OFF, 0.20%", "original", "phone", "off", 0.20),
    ("Phone ON, 0.20%", "original", "phone", "on", 0.20),
]

font = ImageFont.load_default()
thumb_w, thumb_h = 240, 200
cell_w, cell_h = 260, 255
tiles = []
report_rows = []

for title, cohort, device, lamp, conc in cases:
    group = META[
        (META["cohort"] == cohort)
        & (META["device"] == device)
        & (META["lamp"] == lamp)
        & ((META["conc"] - conc).abs() < 1e-6)
    ].sort_values(["timestamp", "image_id"], na_position="last")

    print(f"\n{title}: {len(group)} rows")

    for _, row in group.iterrows():
        print(
            f"  Excel row {row['source_excel_row']}: "
            f"{row['image_path']}"
        )

        report_rows.append({
            "case": title,
            "image_id": row["image_id"],
            "conc": row["conc"],
            "device": row["device"],
            "lamp": row["lamp"],
            "timestamp": row["timestamp"],
            "seq": row["seq"],
            "image_path": row["image_path"],
            "orig_path": row["orig_path"],
            "source_excel": row["source_excel"],
            "source_excel_row": row["source_excel_row"],
        })

        image_path = ROOT / row["image_path"]
        tile = Image.new("RGB", (cell_w, cell_h), "white")
        draw = ImageDraw.Draw(tile)

        draw.text((5, 5), title, fill="black", font=font)

        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                x = (cell_w - im.width) // 2
                tile.paste(im, (x, 25))

            draw.text(
                (5, 230),
                image_path.name[:43],
                fill="black",
                font=font,
            )
        except Exception as exc:
            draw.text((5, 50), f"ERROR: {exc}", fill="red", font=font)

        tiles.append(tile)

report = pd.DataFrame(report_rows)
report.to_csv(OUT / "phase2_anomalies.csv", index=False)

columns = 3
rows = (len(tiles) + columns - 1) // columns
sheet = Image.new(
    "RGB",
    (columns * cell_w, rows * cell_h),
    "#dddddd",
)

for i, tile in enumerate(tiles):
    x = (i % columns) * cell_w
    y = (i // columns) * cell_h
    sheet.paste(tile, (x, y))

output = OUT / "phase2_anomalies.jpg"
sheet.save(output, quality=92)

print(f"\nSaved report: {OUT / 'phase2_anomalies.csv'}")
print(f"Saved contact sheet: {output}")
