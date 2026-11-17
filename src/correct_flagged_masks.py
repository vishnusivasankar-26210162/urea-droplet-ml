
"""Interactively review and correct flagged development ROI masks.

Keyboard:
    A       Accept existing automatic mask
    Enter   Finish manually drawn polygon and save correction
    Escape  Clear current polygon
    S       Skip this image
    Q       Quit review

Only flagged train/validation images are reviewed.
Test images are never opened.

Outputs overwrite only the corresponding interim masks/crops/overlays.
Original photographs remain untouched.
"""
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import PolygonSelector
from PIL import Image, ImageDraw
from skimage.draw import polygon
from skimage.measure import find_contours


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT / "data/interim/roi_mask_manifest.csv"
)

MASK_DIR = ROOT / "data/interim/masked_rgba"
CROP_DIR = ROOT / "data/interim/roi_crops"
OVERLAY_DIR = ROOT / "results/figures/phase5_mask_overlays"

POLYGON_DIR = ROOT / "data/interim/manual_roi_polygons"

BACKUP_PATH = (
    ROOT / "data/interim/roi_mask_manifest_before_manual_review.csv"
)

CROP_MARGIN_FRACTION = 0.06


def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def resolve_path(value):
    path = (ROOT / str(value)).resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"Path escapes project root: {value}"
        )

    return path


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def load_existing_mask(row, shape):
    """Load current full-resolution automatic mask."""

    path = resolve_path(row["mask_path"])

    with Image.open(path) as im:
        rgba = np.asarray(
            im.convert("RGBA"),
            dtype=np.uint8,
        )

    mask = rgba[:, :, 3] > 127

    if mask.shape != shape:
        raise ValueError(
            f"Mask shape {mask.shape} differs from image shape {shape}"
        )

    return mask


def get_output_paths(row):
    """Use existing manifest output paths."""

    return {
        "mask": resolve_path(row["mask_path"]),
        "crop": resolve_path(row["crop_path"]),
        "overlay": resolve_path(row["overlay_path"]),
    }


def make_overlay(rgb, mask):
    """Create visual overlay with green ROI and red contour."""

    result = rgb.astype(np.float32).copy()

    green = np.array(
        [0, 255, 0],
        dtype=np.float32,
    )

    result[mask] = (
        0.65 * result[mask]
        + 0.35 * green
    )

    result = np.clip(
        result, 0, 255
    ).astype(np.uint8)

    image = Image.fromarray(result)
    draw = ImageDraw.Draw(image)

    contours = find_contours(
        mask.astype(float),
        level=0.5,
    )

    for contour in contours:

        points = [
            (float(x), float(y))
            for y, x in contour
        ]

        if len(points) >= 2:
            draw.line(
                points,
                fill=(255, 0, 0),
                width=max(2, round(max(rgb.shape[:2]) / 700)),
            )

    return image


def save_mask_outputs(rgb, mask, row):
    """Save manually corrected mask, crop and overlay."""

    h, w = rgb.shape[:2]

    rgba = np.dstack([
        rgb,
        mask.astype(np.uint8) * 255,
    ])

    rgba_image = Image.fromarray(rgba)

    paths = get_output_paths(row)

    paths["mask"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["crop"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths["overlay"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Full-resolution masked image.
    rgba_image.save(paths["mask"])

    # Bounding box around the polygon with a small margin.
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Empty manual polygon")

    margin = int(
        CROP_MARGIN_FRACTION * max(w, h)
    )

    left = max(0, int(xs.min()) - margin)
    right = min(w, int(xs.max()) + 1 + margin)

    top = max(0, int(ys.min()) - margin)
    bottom = min(h, int(ys.max()) + 1 + margin)

    rgba_image.crop(
        (left, top, right, bottom)
    ).save(paths["crop"])

    # Overlay.
    overlay = make_overlay(rgb, mask)

    overlay.save(
        paths["overlay"],
        quality=90,
    )

    return {
        "area_fraction": float(mask.mean()),
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
    }


def save_polygon_vertices(image_id, vertices):
    """Save polygon coordinates for reproducibility."""

    POLYGON_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        POLYGON_DIR
        / f"{safe_filename(image_id)}.json"
    )

    data = {
        "image_id": str(image_id),
        "method": "manual_polygon",
        "vertices_xy": [
            [float(x), float(y)]
            for x, y in vertices
        ],
    }

    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    return path


def correct_one(row, manifest, index):
    """Open one flagged image and let the user decide."""

    image_id = str(row["image_id"])

    original_path = resolve_path(
        row["image_path"]
    )

    rgb = load_rgb(original_path)

    h, w = rgb.shape[:2]

    auto_mask = load_existing_mask(
        row,
        (h, w),
    )

    state = {
        "action": None,
    }

    fig, ax = plt.subplots(figsize=(11, 9))

    ax.imshow(rgb)

    # Show existing mask as a red boundary.
    contours = find_contours(
        auto_mask.astype(float),
        level=0.5,
    )

    for contour in contours:
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            color="red",
            linewidth=1.8,
        )

    old_reason = row.get(
        "failure_reason", ""
    )

    ax.set_title(
        f"{index} | {image_id}\n"
        f"Instance {row['instance_id']} | "
        f"{row['device']} | lamp {row['lamp']}\n"
        f"Flag: {old_reason}\n\n"
        "A: accept existing | "
        "Click perimeter + Enter: manual correction | "
        "S: skip | Q: quit",
        fontsize=10,
    )

    ax.axis("off")

    selector_holder = {}

    def on_polygon_complete(vertices):
        """Save the manually drawn polygon."""

        vertices = np.asarray(
            vertices,
            dtype=float,
        )

        if len(vertices) < 3:
            print("Polygon needs at least three points.")
            return

        xs = vertices[:, 0]
        ys = vertices[:, 1]

        rr, cc = polygon(
            ys,
            xs,
            shape=(h, w),
        )

        manual_mask = np.zeros(
            (h, w),
            dtype=bool,
        )

        manual_mask[rr, cc] = True

        if not manual_mask.any():
            print("Empty polygon. Try again.")
            return

        # Save corrected files.
        diagnostics = save_mask_outputs(
            rgb,
            manual_mask,
            row,
        )

        polygon_path = save_polygon_vertices(
            image_id,
            vertices,
        )

        manifest.at[index - 1, "mask_status"] = (
            "manual_corrected"
        )

        manifest.at[index - 1, "failure_reason"] = ""

        manifest.at[index - 1, "review_action"] = (
            "manual_polygon"
        )

        manifest.at[index - 1, "manual_polygon_path"] = (
            str(polygon_path.relative_to(ROOT))
        )

        for key, value in diagnostics.items():
            manifest.at[index - 1, key] = value

        manifest.to_csv(
            MANIFEST_PATH,
            index=False,
        )

        state["action"] = "corrected"

        print(
            f"Manually corrected: {image_id} "
            f"(area={diagnostics['area_fraction']:.3f})"
        )

        plt.close(fig)

    selector = PolygonSelector(
        ax,
        on_polygon_complete,
        useblit=False,
        props={
            "color": "yellow",
            "linewidth": 2,
            "alpha": 0.9,
        },
    )

    selector_holder["selector"] = selector

    def on_key(event):

        if event.key == "a":

            # Accept the existing automatic mask.
            manifest.at[index - 1, "mask_status"] = (
                "manual_accepted"
            )

            manifest.at[index - 1, "failure_reason"] = ""

            manifest.at[index - 1, "review_action"] = (
                "accepted_auto"
            )

            manifest.to_csv(
                MANIFEST_PATH,
                index=False,
            )

            state["action"] = "accepted"

            print(f"Accepted automatic mask: {image_id}")

            plt.close(fig)

        elif event.key == "s":

            state["action"] = "skipped"

            print(f"Skipped: {image_id}")

            plt.close(fig)

        elif event.key == "q":

            state["action"] = "quit"

            print("Stopping manual review.")

            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    plt.tight_layout()

    # Force the Matplotlib window to the foreground.
    manager = fig.canvas.manager
    window = manager.window

    try:
        window.deiconify()
        window.lift()
        window.attributes("-topmost", True)
        window.focus_force()

        # Remove always-on-top after one second.
        window.after(
            1000,
            lambda: window.attributes("-topmost", False)
        )

        # Maximize the window.
        window.state("zoomed")

    except Exception as exc:
        print("Could not maximize window:", exc)

    plt.show(block=True)

    return state["action"]


def main():

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    manifest = pd.read_csv(MANIFEST_PATH)

    required = {
        "image_id",
        "image_path",
        "mask_path",
        "crop_path",
        "overlay_path",
        "mask_status",
        "failure_reason",
        "role",
    }

    missing = required - set(manifest.columns)

    if missing:
        raise ValueError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    # Make one backup before changing any records.
    if not BACKUP_PATH.exists():
        shutil.copy2(
            MANIFEST_PATH,
            BACKUP_PATH,
        )

    # New audit columns.
    for col in [
        "review_action",
        "manual_polygon_path",
    ]:
        if col not in manifest.columns:
            manifest[col] = ""

    # Only flagged development images.
    pending = manifest[
        (manifest["mask_status"] == "flagged")
        & (manifest["role"].isin(["train", "validation"]))
    ].copy()

    if pending.empty:
        print("No flagged development images remain.")
        return

    # Use actual manifest row indices.
    pending_indices = pending.index.tolist()

    print("\nMANUAL ROI CORRECTION")
    print("=" * 55)
    print("Flagged images:", len(pending_indices))
    print("\nInstructions:")
    print("  A       Accept existing mask")
    print("  Enter   Complete manually drawn polygon")
    print("  Escape  Clear current polygon")
    print("  S       Skip this image")
    print("  Q       Quit review")

    for count, manifest_index in enumerate(
        pending_indices,
        start=1,
    ):

        row = manifest.loc[manifest_index]

        # Function uses positional manifest index in its updates.
        # Convert to one-based row position.
        position = manifest.index.get_loc(manifest_index) + 1

        action = correct_one(
            row,
            manifest,
            position,
        )

        if action == "quit":
            break

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("\nUPDATED MASK STATUS")
    print("=" * 55)

    print(
        manifest["mask_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nManual review actions:")
    print(
        manifest.get(
            "review_action",
            pd.Series(dtype=str),
        )
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nManifest updated:", MANIFEST_PATH)
    print("Backup:", BACKUP_PATH)


if __name__ == "__main__":
    main()
