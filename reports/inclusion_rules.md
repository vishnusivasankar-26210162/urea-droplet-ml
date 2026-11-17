# Phase 4 image inclusion rules

Review development images only (train + validation);
do not open test images.

## Include an image when

- The original file opens and decodes.
- The intended droplet is visible.
- Its boundary can be identified.
- Focus is adequate.
- It is not severely cropped.
- Glare or obstruction does not hide most of the droplet.
- Image-to-metadata mapping is unambiguous.

## Exclude an image only when

- The file is corrupt or unreadable.
- No intended droplet can be identified.
- Severe crop or blur prevents meaningful ROI identification.
- Severe obstruction/glare hides most of the droplet.
- Image-to-label mapping is known to be incorrect or ambiguous.

## Do not exclude based on

- Concentration or unexpected colour.
- Whether the image improves a future model score.
- Device/lamp/session membership alone.
- A future model prediction error.

Record every excluded image by image_id and give a reason
in image_quality_review.csv.

Blank reviews are pending; they are not automatically
included or excluded.

The source images are original opaque BMP/JPG photographs,
not transparent cut-outs. Alpha transparency describes the
source file only, not droplet segmentation quality.

Droplet ROI occupancy and mask quality belong to the
later masking/feature stage.
