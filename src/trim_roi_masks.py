
"""Phase 5.1-5.2: erode droplet masks and remove specular glare.

Inputs:
    data/processed/metadata.csv
    data/interim/roi_mask_manifest.csv
    data/interim/masked_rgba/

Outputs:
    data/interim/eroded_masks/
    data/interim/cleaned_roi_rgba/
    data/interim/cleaned_roi_crops/
    data/interim/phase5_trim_manifest.csv
    results/figures/phase5_trim_overlays/
    results/figures/phase5_trim_preview/

Only train and validation images are processed.
Test images are not opened.

The starting erosion parameter is 3.5% of image width.
Preview sheets compare 3%, 3.5% and 4% on development images.
"""

from pathlib import Path
import math
import re
import gc

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw, ImageFont

from scipy import ndimage as ndi



ROOT = Path(__file__).resolve().parents[1]

METADATA_PATH = ROOT / "data/processed/metadata.csv"
MASK_MANIFEST_PATH = ROOT / "data/interim/roi_mask_manifest.csv"

ERODED_DIR = ROOT / "data/interim/eroded_masks"
CLEANED_DIR = ROOT / "data/interim/cleaned_roi_rgba"
CROP_DIR = ROOT / "data/interim/cleaned_roi_crops"

OVERLAY_DIR = ROOT / "results/figures/phase5_trim_overlays"
PREVIEW_DIR = ROOT / "results/figures/phase5_trim_preview"

OUTPUT_MANIFEST = ROOT / "data/interim/phase5_trim_manifest.csv"

# Starting candidate. It is not frozen until previews are checked.
# fixed it.
EROSION_FRACTION = 0.035

# Compare these values visually on representative development images.
PREVIEW_EROSION_FRACTIONS = [0.03, 0.035, 0.04]

# Roadmap glare starting thresholds, HSV values scaled from 0 to 1.
GLARE_VALUE_THRESHOLD = 0.92
GLARE_SATURATION_THRESHOLD = 0.25

CROP_MARGIN_FRACTION = 0.06

DEVELOPMENT_ROLES = {"train", "validation"}


# =========================================================
# General utilities
# =========================================================

def safe_filename(value):
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def safe_project_path(value):
    path = (ROOT / str(value)).resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"Path escapes project root: {value}"
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def relative_path(path):
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def load_rgb(path):
    with Image.open(path) as im:
        im.load()
        return np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )


def load_alpha_mask(path, expected_shape):
    """Read RGBA and use alpha > 127 as the droplet mask."""

    with Image.open(path) as im:
        im.load()
        rgba = np.asarray(
            im.convert("RGBA"),
            dtype=np.uint8,
        )

    if rgba.shape[:2] != expected_shape:
        raise ValueError(
            f"Mask dimensions {rgba.shape[:2]} do not match "
            f"original image dimensions {expected_shape}"
        )

    mask = rgba[:, :, 3] > 127

    if not mask.any():
        raise ValueError("Source alpha mask is empty")

    return mask


# =========================================================
# 5.1 Recover original droplet mask
# =========================================================

def recover_droplet_mask(original_rgb, rgba_mask_path):
    """Recover binary droplet mask from RGBA PNG."""

    return load_alpha_mask(
        rgba_mask_path,
        original_rgb.shape[:2],
    )


# =========================================================
# 5.2 Erode and remove glare
# =========================================================

def erode_mask(mask, image_width, erosion_fraction):
    """Erode with a circular footprint using a distance transform.

    The distance-transform formulation avoids constructing and applying
    a large disk footprint for high-resolution images.
    """
    radius = max(1, int(round(image_width * erosion_fraction)))
    height, width = mask.shape

    # Explicit zero-padding preserves border_value=0 behavior.
    padded = np.pad(
        mask,
        ((radius, radius), (radius, radius)),
        mode="constant",
        constant_values=False,
    )

    distances = ndi.distance_transform_edt(padded)
    eroded = (
        distances[radius:radius + height, radius:radius + width] > radius
    )

    return eroded, radius


def find_glare_mask(rgb, roi_mask):
    """Find bright, low-saturation pixels inside the ROI.

    This applies the same HSV thresholds without allocating a full HSV
    image. For RGB, V=max(R,G,B)/255 and S=(max-min)/max.
    """
    maximum = rgb.max(axis=2).astype(np.uint16)
    minimum = rgb.min(axis=2).astype(np.uint16)
    spread = maximum - minimum

    value_threshold = int(np.floor(GLARE_VALUE_THRESHOLD * 255)) + 1
    bright = maximum >= value_threshold
    low_saturation = (4 * spread) < maximum * (
        4 * GLARE_SATURATION_THRESHOLD
    )

    return bright & low_saturation & roi_mask


def trim_mask(rgb, base_mask, erosion_fraction):
    """Erode original mask and remove glare pixels."""

    height, width = base_mask.shape

    eroded, radius = erode_mask(
        base_mask,
        width,
        erosion_fraction,
    )

    glare = find_glare_mask(
        rgb,
        eroded,
    )

    cleaned = eroded & ~glare

    base_area = int(base_mask.sum())
    eroded_area = int(eroded.sum())
    glare_area = int(glare.sum())
    cleaned_area = int(cleaned.sum())

    erosion_retained_fraction = (
        eroded_area / base_area
        if base_area else 0.0
    )

    glare_removed_fraction = (
        glare_area / eroded_area
        if eroded_area else 0.0
    )

    cleaned_fraction_of_image = (
        cleaned_area / (height * width)
    )

    diagnostics = {
        "erosion_fraction": float(erosion_fraction),
        "erosion_radius_px": int(radius),
        "base_mask_pixels": base_area,
        "eroded_mask_pixels": eroded_area,
        "glare_pixels_removed": glare_area,
        "cleaned_mask_pixels": cleaned_area,
        "erosion_retained_fraction": float(
            erosion_retained_fraction
        ),
        "glare_removed_fraction": float(
            glare_removed_fraction
        ),
        "cleaned_area_fraction": float(
            cleaned_fraction_of_image
        ),
    }

    warnings = []

    if eroded_area == 0:
        warnings.append("erosion_removed_entire_mask")
    elif erosion_retained_fraction < 0.40:
        warnings.append("erosion_removed_over_60_percent")

    if cleaned_area == 0:
        warnings.append("glare_removal_removed_entire_mask")
    elif eroded_area and glare_removed_fraction > 0.30:
        warnings.append("glare_removal_over_30_percent")

    diagnostics["trim_warning"] = ";".join(warnings)

    return eroded, glare, cleaned, diagnostics


# =========================================================
# Save masks and images
# =========================================================

def crop_bounds(mask, width, height):
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Cannot crop an empty mask")

    margin = int(
        CROP_MARGIN_FRACTION * max(width, height)
    )

    left = max(0, int(xs.min()) - margin)
    right = min(width, int(xs.max()) + 1 + margin)

    top = max(0, int(ys.min()) - margin)
    bottom = min(height, int(ys.max()) + 1 + margin)

    return left, top, right, bottom


def make_overlay(rgb, mask):
    """Create a memory-efficient green ROI overlay with a red boundary."""
    overlay = rgb.copy()

    # Tint the ROI green in-place, one channel at a time.
    for channel_index in (0, 2):
        channel = overlay[:, :, channel_index]
        values = channel[mask].astype(np.uint16)
        channel[mask] = ((values * 65) // 100).astype(np.uint8)

    green = overlay[:, :, 1]
    values = green[mask].astype(np.uint16)
    green[mask] = ((values * 65 + 255 * 35) // 100).astype(np.uint8)

    # Draw a one-pixel red boundary without converting the full mask to float.
    inner = ndi.binary_erosion(
        mask,
        structure=np.ones((3, 3), dtype=bool),
        border_value=0,
    )
    boundary = mask & ~inner
    overlay[boundary] = (255, 0, 0)

    return Image.fromarray(overlay)


def save_trimmed_outputs(rgb, eroded_mask, cleaned_mask, image_id):
    height, width = rgb.shape[:2]
    safe_id = safe_filename(image_id)

    eroded_path = ERODED_DIR / f"{safe_id}.png"
    cleaned_path = CLEANED_DIR / f"{safe_id}.png"
    crop_path = CROP_DIR / f"{safe_id}.png"
    overlay_path = OVERLAY_DIR / f"{safe_id}.jpg"

    for path in [eroded_path, cleaned_path, crop_path, overlay_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    # Save eroded mask before glare removal.
    Image.fromarray(
        eroded_mask.astype(np.uint8) * 255
    ).save(eroded_path)

    # Save glare-cleaned ROI using original RGB pixel values.
    rgba = np.dstack([
        rgb,
        cleaned_mask.astype(np.uint8) * 255,
    ])

    rgba_image = Image.fromarray(rgba, mode="RGBA")
    rgba_image.save(cleaned_path)

    left, top, right, bottom = crop_bounds(
        cleaned_mask, width, height
    )

    rgba_image.crop(
        (left, top, right, bottom)
    ).save(crop_path)

    make_overlay(
        rgb, cleaned_mask
    ).save(overlay_path, quality=90)

    return {
        "eroded_mask_path": relative_path(eroded_path),
        "cleaned_rgba_path": relative_path(cleaned_path),
        "cleaned_crop_path": relative_path(crop_path),
        "trimmed_overlay_path": relative_path(overlay_path),
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
    }


# =========================================================
# Preview 3%, 3.5%, 4% on representative images
# =========================================================

def make_preview_tile(rgb, mask, width=280, height=230):
    overlay = make_overlay(rgb, mask)

    overlay.thumbnail(
        (width - 12, height - 12),
        Image.Resampling.LANCZOS,
    )

    tile = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    tile.paste(
        overlay,
        (
            (width - overlay.width) // 2,
            (height - overlay.height) // 2,
        ),
    )

    return tile


def choose_preview_rows(dev):
    """Select min, middle and max concentration per instance/condition."""

    selected = []

    for (instance, device, lamp), group in dev.groupby(
        ["instance_id", "device", "lamp"]
    ):

        group = group.sort_values(
            ["conc", "image_id"]
        )

        if len(group) <= 3:
            chosen = group
        else:
            positions = [
                0,
                len(group) // 2,
                len(group) - 1,
            ]

            chosen = group.iloc[positions]

        selected.append(chosen)

    if not selected:
        return dev.iloc[0:0].copy()

    return pd.concat(
        selected,
        ignore_index=True,
    ).drop_duplicates("image_id")


def create_erosion_previews(dev, mask_rows):
    """Create one comparison sheet per instance/device/lamp group."""

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    samples = choose_preview_rows(dev)

    for (instance, device, lamp), group in samples.groupby(
        ["instance_id", "device", "lamp"]
    ):

        tiles = []
        labels = []

        for _, row in group.iterrows():

            rgb = load_rgb(
                safe_project_path(row["image_path"])
            )

            manifest_row = mask_rows.loc[
                mask_rows["image_id"].astype(str)
                == str(row["image_id"])
            ].iloc[0]

            base_mask = recover_droplet_mask(
                rgb,
                safe_project_path(manifest_row["mask_path"]),
            )

            for fraction in PREVIEW_EROSION_FRACTIONS:

                _, _, cleaned, diagnostics = trim_mask(
                    rgb,
                    base_mask,
                    fraction,
                )

                tile = make_preview_tile(rgb, cleaned)
                tiles.append(tile)

                labels.append(
                    f"{row['image_id']} | {row['conc']:.2f}% | "
                    f"{fraction * 100:.1f}% erosion | "
                    f"retained {diagnostics['erosion_retained_fraction']:.0%}"
                )
                del tile, diagnostics, cleaned

            del rgb, base_mask

        # Three columns: 3%, 3.5%, 4%. Each source image gets a row.
        ncols = 3
        tile_w, tile_h = 280, 230
        label_h = 34
        rows = len(group)

        sheet = Image.new(
            "RGB",
            (
                ncols * tile_w,
                45 + rows * (tile_h + label_h),
            ),
            "white",
        )

        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()

        draw.text(
            (10, 10),
            f"Instance {instance} | {device} | lamp {lamp}",
            fill="black",
            font=font,
        )

        for i, tile in enumerate(tiles):

            r = i // ncols
            c = i % ncols

            x = c * tile_w
            y = 45 + r * (tile_h + label_h)

            sheet.paste(tile, (x, y))

            draw.text(
                (x + 5, y + tile_h),
                labels[i][:46],
                fill="black",
                font=font,
            )

        filename = (
            f"instance_{instance}_{device}_{lamp}.jpg"
        )

        sheet.save(
            PREVIEW_DIR / filename,
            quality=90,
        )


# =========================================================
# Main processing
# =========================================================

def main():

    for directory in [
        ERODED_DIR,
        CLEANED_DIR,
        CROP_DIR,
        OVERLAY_DIR,
        PREVIEW_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    metadata = pd.read_csv(METADATA_PATH)
    masks = pd.read_csv(MASK_MANIFEST_PATH)

    required_metadata = {
        "image_id",
        "image_path",
        "role",
    }

    required_masks = {
        "image_id",
        "mask_path",
        "mask_status",
    }

    if not required_metadata.issubset(metadata.columns):
        raise ValueError("Metadata columns are missing")

    if not required_masks.issubset(masks.columns):
        raise ValueError("Mask manifest columns are missing")

    # Exclude the test images before opening any files.
    dev = metadata[
        metadata["role"].isin(DEVELOPMENT_ROLES)
    ].copy()

    if dev["image_id"].duplicated().any():
        raise ValueError("Duplicate image_id in development metadata")

    # Preserve current segmentation statuses in the output manifest.
    dev = dev.merge(
        masks,
        on="image_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_mask"),
    )

    if dev["mask_path"].isna().any():
        raise ValueError(
            "Some development images do not have a mask"
        )

    print("PHASE 5.1–5.2: MASK EROSION AND GLARE REMOVAL")
    print("=" * 60)
    print("Development images:", len(dev))
    print(
        "Test images excluded:",
        int((metadata["role"] == "test").sum()),
    )
    print("Candidate erosion:", EROSION_FRACTION)
    print("Glare: V >", GLARE_VALUE_THRESHOLD)
    print("Glare: S <", GLARE_SATURATION_THRESHOLD)

    # Resume from the previous manifest when possible. Only failed rows from
    # Instances 2 and 3 with blank exception messages are retried; all other
    # existing records and their output files are preserved.
    previous_lookup = {}
    retry_ids = set(dev["image_id"].astype(str))

    if OUTPUT_MANIFEST.exists():
        previous = pd.read_csv(OUTPUT_MANIFEST)
        previous_lookup = {
            str(record["image_id"]): record
            for record in previous.to_dict("records")
        }

        if {"trim_status", "trim_warning", "instance_id"}.issubset(previous.columns):
            warning_text = previous["trim_warning"].fillna("").astype(str).str.strip()
            instance_numbers = pd.to_numeric(
                previous["instance_id"], errors="coerce"
            )
            retry_rows = previous[
                (previous["trim_status"] == "failed")
                & (warning_text == "")
                & instance_numbers.isin([2, 3])
            ]
            retry_ids = set(retry_rows["image_id"].astype(str))
        else:
            raise ValueError(
                "Existing trim manifest is missing columns required for resume. "
                "Move it aside only if you intentionally want to start over."
            )

        print("\nRESUME MODE")
        print("Existing manifest rows:", len(previous))
        print("Images to retry:", len(retry_ids))
        print("Existing successful records will be preserved.")
    else:
        print("\nNo previous trim manifest found; processing all development images.")

    records = []

    for index, row in enumerate(
        dev.itertuples(index=False),
        start=1,
    ):
        image_id = str(row.image_id)

        if image_id in previous_lookup and image_id not in retry_ids:
            records.append(previous_lookup[image_id])
            if index % 50 == 0 or index == len(dev):
                print(f"Processed {index}/{len(dev)}")
            continue

        record = {
            "image_id": image_id,
            "role": str(row.role),
            "instance_id": int(row.instance_id),
            "tube_id": str(row.tube_id),
            "conc": float(row.conc),
            "device": str(row.device),
            "lamp": str(row.lamp),
            "source_mask_status": str(row.mask_status),
            "trim_status": "failed",
            "trim_warning": "",
        }

        try:
            original = load_rgb(safe_project_path(row.image_path))
            base_mask = recover_droplet_mask(
                original,
                safe_project_path(row.mask_path),
            )
            eroded, glare, cleaned, diagnostics = trim_mask(
                original,
                base_mask,
                EROSION_FRACTION,
            )
            output_paths = save_trimmed_outputs(
                original,
                eroded,
                cleaned,
                image_id,
            )

            record.update(diagnostics)
            record.update(output_paths)

            if row.mask_status == "flagged":
                record["trim_status"] = "source_mask_flagged"
            elif diagnostics["trim_warning"]:
                record["trim_status"] = "needs_review"
            else:
                record["trim_status"] = "candidate"

        except Exception as exc:
            record["trim_status"] = "failed"
            record["trim_warning"] = f"{type(exc).__name__}: {exc!r}"

        finally:
            # Release full-resolution arrays before loading the next image.
            try:
                del original
            except UnboundLocalError:
                pass
            try:
                del base_mask
            except UnboundLocalError:
                pass
            try:
                del eroded, glare, cleaned
            except UnboundLocalError:
                pass

        records.append(record)

        if index % 10 == 0:
            gc.collect()
        if index % 50 == 0 or index == len(dev):
            print(f"Processed {index}/{len(dev)}")

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_MANIFEST,
        index=False,
    )

    print("\nTRIMMING SUMMARY")
    print("=" * 60)

    print(
        result["trim_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nMean retained fraction after erosion:")
    print(
        result.groupby(
            ["device", "lamp"]
        )["erosion_retained_fraction"]
        .mean()
        .to_string()
    )

    print("\nMean glare removed fraction:")
    print(
        result.groupby(
            ["device", "lamp"]
        )["glare_removed_fraction"]
        .mean()
        .to_string()
    )

    print("\nRows with warnings or failures:")
    print(
        result[
            result["trim_status"].isin(
                ["needs_review", "source_mask_flagged", "failed"]
            )
        ][
            [
                "image_id",
                "trim_status",
                "trim_warning",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nGenerating erosion comparison previews...")

    create_erosion_previews(
        dev,
        masks,
    )

    print("\nOUTPUTS")
    print("=" * 60)
    print("Trim manifest:", OUTPUT_MANIFEST.relative_to(ROOT))
    print("Eroded masks:", ERODED_DIR.relative_to(ROOT))
    print("Cleaned RGBA:", CLEANED_DIR.relative_to(ROOT))
    print("Cropped ROIs:", CROP_DIR.relative_to(ROOT))
    print("Overlays:", OVERLAY_DIR.relative_to(ROOT))
    print("3%, 3.5%, 4% previews:", PREVIEW_DIR.relative_to(ROOT))

    print("\nOriginal masks and photographs were not modified.")
    print("Test images were not opened.")
    print(
        "The 3.5% erosion parameter is frozen now for this project."
    )


if __name__ == "__main__":
    main()
