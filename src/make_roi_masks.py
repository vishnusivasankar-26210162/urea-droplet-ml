
"""Detect droplet contours and generate ROI masks/crops.

Input:
    data/processed/metadata.csv

Outputs:
    data/interim/masked_rgba/
        Full-resolution images with transparent background outside ROI.

    data/interim/roi_crops/
        Cropped transparent droplet images.

    data/interim/roi_mask_manifest.csv
        One record per development image with mask diagnostics.

    results/figures/phase5_mask_overlays/
        Image overlays showing detected contours.

Only train and validation images are processed. Locked test
images are not opened.

The segmentation method is heuristic and must be checked on
development images before its thresholds are frozen.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from PIL import Image, ImageDraw

from scipy import ndimage as ndi

from skimage.color import rgb2lab
from skimage.filters import threshold_otsu, gaussian
from skimage.measure import label, regionprops
from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_objects,
)


ROOT = Path(__file__).resolve().parents[1]

META_PATH = ROOT / "data" / "processed" / "metadata.csv"

MASK_DIR = ROOT / "data" / "interim" / "masked_rgba"
CROP_DIR = ROOT / "data" / "interim" / "roi_crops"
OVERLAY_DIR = ROOT / "results" / "figures" / "phase5_mask_overlays"

MANIFEST_PATH = ROOT / "data" / "interim" / "roi_mask_manifest.csv"


# ---------------------------------------------------------
# Fixed segmentation parameters
# ---------------------------------------------------------

# Downsampling is used only to find the contour.
# The resulting mask is resized back to original resolution.
MAX_SEGMENTATION_SIDE = 900

# Width of the image-border band used to estimate background colour.
BACKGROUND_BORDER_FRACTION = 0.06

# Ignore a narrow outer boundary during candidate detection.
SEARCH_MARGIN_FRACTION = 0.025

# Smooth the colour-distance image.
GAUSSIAN_SIGMA_FRACTION = 0.003

# Minimum Lab distance threshold.
# Otsu threshold is used, but never below this value.
MIN_LAB_DISTANCE = 4.0

# Morphological closing radius as a fraction of image width.
CLOSING_RADIUS_FRACTION = 0.012

# Minimum component area.
MIN_AREA_FRACTION = 0.003

# Crop margin around the detected contour.
CROP_MARGIN_FRACTION = 0.06

# Confidence/quality flagging thresholds.
MIN_DROPLET_AREA = 0.005
MAX_DROPLET_AREA = 0.85
MAX_CENTRE_DISTANCE = 0.25
MIN_SOLIDITY = 0.65

# Expected acquisition instances.
EXPECTED_INSTANCES = {1, 2, 3}


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def safe_filename(value):
    """Convert image_id into a filesystem-safe filename."""
    return re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        str(value),
    )


def rel(path):
    return path.relative_to(ROOT).as_posix()

def image_path(value):
    """Resolve an image path safely inside the project."""
    path = (ROOT / str(value)).resolve()

    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError(
            f"Image path escapes project root: {value}"
        )

    if not path.is_file():
        raise FileNotFoundError(path)

    return path


def load_rgb(path):
    """Load an image as RGB, respecting EXIF orientation."""

    if not path.is_file():
        raise FileNotFoundError(path)

    with Image.open(path) as im:
        im.load()
        rgb = np.asarray(
            im.convert("RGB"),
            dtype=np.uint8,
        )

    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Unexpected image shape: {rgb.shape}")

    return rgb


def resize_for_segmentation(rgb):
    """Downsample large images while preserving aspect ratio."""

    h, w = rgb.shape[:2]
    largest = max(h, w)

    if largest <= MAX_SEGMENTATION_SIDE:
        return rgb.copy()

    scale = MAX_SEGMENTATION_SIDE / largest

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    im = Image.fromarray(rgb)

    return np.asarray(
        im.resize(
            (new_w, new_h),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )


# ---------------------------------------------------------
# Background estimation and droplet segmentation
# ---------------------------------------------------------

def estimate_background_lab(lab):
    """Estimate the background Lab colour from the outer border."""

    h, w = lab.shape[:2]

    border_width = max(
        2,
        int(round(
            BACKGROUND_BORDER_FRACTION * min(h, w)
        )),
    )

    border_mask = np.zeros((h, w), dtype=bool)

    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    border_pixels = lab[border_mask]

    if len(border_pixels) == 0:
        raise ValueError("Could not estimate image background")

    return np.median(
        border_pixels,
        axis=0,
    )


def find_central_droplet_mask(rgb_small):
    """Return a central droplet mask and segmentation diagnostics.

    Segmentation is based on Lab colour distance from the estimated
    outer-border background.

    The connected component closest to the image centre is selected.
    """

    h, w = rgb_small.shape[:2]
    min_dim = min(h, w)

    if min_dim < 80:
        raise ValueError("Image too small for reliable segmentation")

    rgb_float = rgb_small.astype(np.float32) / 255.0

    lab = rgb2lab(rgb_float)

    background_lab = estimate_background_lab(lab)

    # Per-pixel Euclidean Lab distance from estimated background.
    distance = np.linalg.norm(
        lab - background_lab[None, None, :],
        axis=2,
    )

    sigma = max(
        0.8,
        GAUSSIAN_SIGMA_FRACTION * min_dim,
    )

    distance_smooth = gaussian(
        distance,
        sigma=sigma,
        preserve_range=True,
    )

    # Otsu threshold is estimated only in the central search area.
    margin_y = int(SEARCH_MARGIN_FRACTION * h)
    margin_x = int(SEARCH_MARGIN_FRACTION * w)

    search = np.zeros((h, w), dtype=bool)

    search[
        margin_y:h - margin_y,
        margin_x:w - margin_x,
    ] = True

    values = distance_smooth[search]

    if np.allclose(values.min(), values.max()):
        raise ValueError("No useful colour contrast found")

    otsu_value = float(threshold_otsu(values))

    threshold = max(
        otsu_value,
        MIN_LAB_DISTANCE,
    )

    candidate = (
        (distance_smooth > threshold)
        & search
    )

    # Join small gaps in the detected boundary.
    radius = max(
        2,
        int(round(
            CLOSING_RADIUS_FRACTION * min_dim
        )),
    )

    candidate = binary_closing(
        candidate,
        footprint=disk(radius),
    )

    # Fill internal gaps and small holes.
    candidate = ndi.binary_fill_holes(candidate)

    minimum_pixels = max(
        20,
        int(MIN_AREA_FRACTION * h * w),
    )

    candidate = remove_small_objects(
        candidate,
        min_size=minimum_pixels,
    )

    labels = label(candidate, connectivity=2)

    regions = regionprops(labels)

    if not regions:
        raise ValueError("No candidate droplet contour detected")

    centre_y = (h - 1) / 2
    centre_x = (w - 1) / 2

    candidates = []

    for region in regions:

        area_fraction = region.area / (h * w)

        if not (
            MIN_AREA_FRACTION
            <= area_fraction
            <= MAX_DROPLET_AREA
        ):
            continue

        cy, cx = region.centroid

        distance_from_center = np.sqrt(
            ((cx - centre_x) / w) ** 2
            + ((cy - centre_y) / h) ** 2
        )

        if distance_from_center > 0.40:
            continue

        minr, minc, maxr, maxc = region.bbox

        bbox_touches_edge = (
            minr <= 1
            or minc <= 1
            or maxr >= h - 1
            or maxc >= w - 1
        )

        # A contour connected to the frame edge is likely background.
        if bbox_touches_edge:
            continue

        # Lower score is better:
        # prefer a central, compact component.
        score = (
            distance_from_center
            - 0.08 * float(region.solidity)
            - 0.03 * float(region.extent)
        )

        candidates.append(
            (score, region)
        )

    if not candidates:
        raise ValueError(
            "No central, non-border droplet candidate found"
        )

    _, best = min(
        candidates,
        key=lambda item: item[0],
    )

    mask = labels == best.label

    mask = binary_closing(
        mask,
        footprint=disk(radius),
    )

    mask = ndi.binary_fill_holes(mask)

    # Keep only the largest component after closing/filling.
    final_labels = label(mask, connectivity=2)
    final_regions = regionprops(final_labels)

    if not final_regions:
        raise ValueError("Selected contour disappeared")

    largest = max(
        final_regions,
        key=lambda region: region.area,
    )

    mask = final_labels == largest.label

    # Recompute diagnostics on the final mask.
    area_fraction = float(mask.mean())

    cy, cx = ndi.center_of_mass(mask)

    centre_distance = float(
        np.sqrt(
            ((cx - centre_x) / w) ** 2
            + ((cy - centre_y) / h) ** 2
        )
    )

    minr, minc, maxr, maxc = largest.bbox

    aspect_ratio = float(
        (maxc - minc) / max(1, maxr - minr)
    )

    solidity = float(largest.solidity)

    bbox_touches_edge = (
        minr <= 1
        or minc <= 1
        or maxr >= h - 1
        or maxc >= w - 1
    )

    diagnostics = {
        "threshold_lab_distance": threshold,
        "area_fraction": area_fraction,
        "centre_distance_normalised": centre_distance,
        "solidity": solidity,
        "bbox_aspect_ratio": aspect_ratio,
        "touches_image_edge": bool(bbox_touches_edge),
    }

    return mask.astype(bool), diagnostics


# ---------------------------------------------------------
# Confidence flags
# ---------------------------------------------------------

def assess_mask(diagnostics):
    """Flag questionable masks.

    This score/flag is a diagnostic heuristic, not a calibrated
    probability that the contour is correct.
    """

    reasons = []

    area = diagnostics["area_fraction"]
    centre_distance = diagnostics["centre_distance_normalised"]
    solidity = diagnostics["solidity"]
    aspect = diagnostics["bbox_aspect_ratio"]

    if area < MIN_DROPLET_AREA:
        reasons.append("very_small_mask")

    if area > MAX_DROPLET_AREA:
        reasons.append("mask_too_large")

    if centre_distance > MAX_CENTRE_DISTANCE:
        reasons.append("droplet_far_from_image_centre")

    if solidity < MIN_SOLIDITY:
        reasons.append("irregular_or_fragmented_contour")

    if not (0.40 <= aspect <= 2.50):
        reasons.append("unusual_bbox_aspect_ratio")

    if diagnostics["touches_image_edge"]:
        reasons.append("mask_touches_image_edge")

    return (
        "flagged" if reasons else "auto_ok",
        ";".join(reasons),
    )


# ---------------------------------------------------------
# Resize mask to original image dimensions
# ---------------------------------------------------------

def upscale_mask(mask_small, original_size):
    """Nearest-neighbour upscaling preserves a binary mask."""

    width, height = original_size

    mask_image = Image.fromarray(
        (mask_small.astype(np.uint8) * 255)
    )

    mask_image = mask_image.resize(
        (width, height),
        Image.Resampling.NEAREST,
    )

    return np.asarray(mask_image, dtype=np.uint8) > 127


# ---------------------------------------------------------
# Save full-size mask, cropped ROI and overlay
# ---------------------------------------------------------

def crop_bbox(mask, image_width, image_height):
    """Calculate bounding box with a small proportional margin."""

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Empty mask")

    left = int(xs.min())
    right = int(xs.max()) + 1
    top = int(ys.min())
    bottom = int(ys.max()) + 1

    margin = int(
        CROP_MARGIN_FRACTION
        * max(image_width, image_height)
    )

    left = max(0, left - margin)
    top = max(0, top - margin)

    right = min(image_width, right + margin)
    bottom = min(image_height, bottom + margin)

    return left, top, right, bottom


def save_outputs(
    rgb,
    mask,
    image_id,
):
    """Save full-size RGBA mask, cropped RGBA image and overlay."""

    h, w = rgb.shape[:2]

    alpha = (
        mask.astype(np.uint8) * 255
    )

    rgba_array = np.dstack([
        rgb,
        alpha,
    ])

    rgba_image = Image.fromarray(
        rgba_array,
        mode="RGBA",
    )

    filename = safe_filename(image_id) + ".png"

    mask_path = MASK_DIR / filename
    crop_path = CROP_DIR / filename
    overlay_path = OVERLAY_DIR / (
        safe_filename(image_id) + ".jpg"
    )

    rgba_image.save(mask_path)

    # Crop is for inspection/convenience.
    # Original pixels are preserved; no resizing is performed.
    left, top, right, bottom = crop_bbox(mask, w, h)

    rgba_image.crop(
        (left, top, right, bottom)
    ).save(crop_path)

    # Generate a visual overlay.
    overlay = rgb.copy().astype(np.float32)

    # Green tint indicates selected ROI.
    green = np.array([0, 255, 0], dtype=np.float32)

    overlay[mask] = (
        0.65 * overlay[mask]
        + 0.35 * green
    )

    overlay = np.clip(
        overlay, 0, 255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(overlay)

    draw = ImageDraw.Draw(overlay_image)

    # Draw the actual contour in red.
    # The contour is derived from the binary mask.
    from skimage.measure import find_contours

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
                width=max(2, int(round(max(w, h) / 700))),
            )

    overlay_image.save(
        overlay_path,
        quality=88,
    )

    return {
        "mask_path": rel(mask_path),
        "crop_path": rel(crop_path),
        "overlay_path": rel(overlay_path),
        "crop_left": left,
        "crop_top": top,
        "crop_right": right,
        "crop_bottom": bottom,
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    for directory in [
        MASK_DIR,
        CROP_DIR,
        OVERLAY_DIR,
        MANIFEST_PATH.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    metadata = pd.read_csv(META_PATH)

    required = {
        "image_id",
        "image_path",
        "conc",
        "device",
        "lamp",
        "cohort",
        "instance_id",
        "tube_id",
        "role",
    }

    missing = required - set(metadata.columns)

    if missing:
        raise ValueError(
            f"Metadata missing columns: {sorted(missing)}"
        )

    # Strictly exclude test rows before opening any images.
    dev = metadata[
        metadata["role"].isin(["train", "validation"])
    ].copy()

    if dev.empty:
        raise ValueError("No development images found")

    if not set(dev["instance_id"].unique()).issubset(
        EXPECTED_INSTANCES
    ):
        raise ValueError("Unexpected instance ID")

    print("PHASE 5 — DROPLET ROI MASK GENERATION")
    print("=" * 60)
    print("Development images:", len(dev))
    print(
        "Locked test images excluded:",
        int((metadata["role"] == "test").sum()),
    )

    records = []

    for index, (_, row) in enumerate(
        dev.iterrows(),
        start=1,
    ):

        image_id = str(row["image_id"])

        record = {
            "image_id": image_id,
            "image_path": row["image_path"],
            "cohort": row["cohort"],
            "instance_id": int(row["instance_id"]),
            "tube_id": row["tube_id"],
            "conc": float(row["conc"]),
            "device": row["device"],
            "lamp": row["lamp"],
            "role": row["role"],
            "mask_status": "failed",
            "failure_reason": "",
            "mask_path": "",
            "crop_path": "",
            "overlay_path": "",
        }

        try:

            original = load_rgb(
                image_path(row["image_path"])
            )

            small = resize_for_segmentation(original)

            mask_small, diagnostics = (
                find_central_droplet_mask(small)
            )

            original_h, original_w = original.shape[:2]

            mask = upscale_mask(
                mask_small,
                (original_w, original_h),
            )

            if not mask.any():
                raise ValueError("Upscaled mask is empty")

            status, reasons = assess_mask(
                diagnostics
            )

            output_paths = save_outputs(
                original,
                mask,
                image_id,
            )

            record.update(diagnostics)
            record.update(output_paths)
            record["mask_status"] = status
            record["failure_reason"] = reasons

        except Exception as exc:

            record["mask_status"] = "failed"
            record["failure_reason"] = str(exc)

        records.append(record)

        if index % 50 == 0 or index == len(dev):
            print(
                f"Processed {index}/{len(dev)} images"
            )

    manifest = pd.DataFrame(records)

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
    )

    print("\nMASK GENERATION SUMMARY")
    print("=" * 60)

    print(
        manifest["mask_status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print("\nStatus by instance:")
    print(
        manifest.groupby(
            ["instance_id", "mask_status"]
        )
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    print("\nMean mask area by condition:")
    print(
        manifest.groupby(
            ["device", "lamp"]
        )["area_fraction"]
        .mean()
        .to_string()
    )

    print("\nFailure/flag summary:")

    flagged = manifest[
        manifest["mask_status"] != "auto_ok"
    ]

    if flagged.empty:
        print("No flagged detections.")
    else:
        print(
            flagged[
                [
                    "image_id",
                    "instance_id",
                    "mask_status",
                    "failure_reason",
                ]
            ].head(30).to_string(index=False)
        )

    print("\nSaved manifest:", rel(MANIFEST_PATH))
    print("Full-resolution masked images:", rel(MASK_DIR))
    print("Cropped ROI images:", rel(CROP_DIR))
    print("Mask overlays:", rel(OVERLAY_DIR))

    print(
        "\nIMPORTANT: flagged masks are not automatically accepted "
        "as reliable. Test images were not opened."
    )


if __name__ == "__main__":
    main()
