
# Phase 5.3–5.4: Colour feature extraction

## Canonical colour space

The pipeline uses `skimage.color.rgb2lab` with scikit-image's D65 / 2-degree defaults.

Model features retain true CIELAB values:
- L*: approximately 0–100.
- a* and b*: signed coordinates.

For comparison with old OpenCV-style plots only:
- L_old = 2.55 × L*
- a_old = a* + 128
- b_old = b* + 128

These transformed values are not saved as the canonical model features.

## Input and mask

Input is the cleaned RGBA ROI produced in Phase 5.1–5.2.

A pixel is included when alpha > 127. Transparent pixels are ignored.

Conversion to CIELAB is performed only on included pixels to reduce memory use.

## Features per image

For each of L*, a* and b*:
- Mean
- Median
- Population standard deviation
- 10th percentile
- 90th percentile

Additional descriptors:
- Chroma: C*ab = sqrt(a*² + b*²), with the same five summary statistics.
- Hue angle: circular mean of pixel-level atan2(b*, a*) angles, in degrees.
- Pale fraction: fraction of included pixels whose b* is at least 8 units below the median b* in the top third of the droplet ROI bounding box.

The pale fraction is a heuristic derived from the A019 observation, not a ground-truth label.

## QC fields

All QC columns begin with `qc_` and must be excluded from model inputs.

They include:
- Original transparent fraction.
- Original droplet area fraction.
- Glare fraction and its denominator convention.
- Cleaned ROI area and pixel count.
- Erosion retention.
- Source mask status.
- Image dimensions.
- Per-image pale-pixel reference and threshold.

## Processing scope

- Only train and validation images are processed.
- Test images are not opened.
- One feature row is retained per development image_id.
- Failed rows remain in the output with feature_status=failed.
- Rows with trimming warnings are retained when a cleaned ROI exists.
- Original photographs, masks and frozen trimming outputs are not modified.

## A019 reference test

The A019 image is expected to give approximately:
- b_med = 24.3
- L_mean = 82.1

The test uses an absolute tolerance of 0.5 for each value.

If the test fails, verify the A019 image match and the preprocessing pipeline before changing any frozen thresholds.

## Implementation

- `src/extract_colour_features.py`
- `tests/test_colour_features.py`

Output:
- `data/processed/features.csv`

Phase 5.5 feature-set ablations (S0–S4) have not yet been implemented.
