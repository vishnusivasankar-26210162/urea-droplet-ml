# Decision Log

Record each important decision, its reason, and its observed impact.

| Date | Decision | Reason | Evidence or impact |
|---|---|---|---|

2026-10-23 — Expanded metadata to include the original four-condition
experiment, the 456-image new-tube cohort, and the 32-image November 4
validation cohort. Reason: the additional images represent distinct tube
preparations and are useful for studying generalization. Keep cohort identity
explicit and do not mix validation data into development/model selection.

2026-10-23 — Use original photographs and derive an ROI mask during feature
extraction instead of requiring pre-cut transparent PNGs. Reason: preserve
the complete scene so ROI-only features can be compared against ROI plus
background/illumination features. The active image_path points to the original
photo; orig_path preserves the exact path recorded by the source spreadsheet.

2026-10-23 — Documented the current understanding of the three
experimental cohorts and their unresolved tube/droplet identities.
Reason: the original, new-tube and November 4 collections have
different acquisition histories and must not be combined blindly.
Exact tube and droplet IDs will only be assigned where supported
by experimental records or sufficiently strong pairing evidence.
Unrecoverable identities will be flagged and handled using the
roadmap's conservative fallback.

2026-10-23 — Retained all 1,064 original photographs as rows in
the master metadata, with a cohort identifier. The November 4
validation cohort will remain separate from development and model
selection.

## Phase 2 decision-log addendum — 16 Nov 2026

1. **Use original photographs.** The active `image_path` points to original BMP/JPG files. Pre-cut PNGs are not used as model inputs. ROI masks and colour features will be derived from original photographs; ROI-only and ROI-plus-background feature sets will be compared.
2. **Preparation instances.** `original` maps to `instance_id=1`; `new_tubes` maps to `instance_id=2`; `nov4_validation` maps to `instance_id=3`.
3. **Physical tube identity.** There is exactly one physical tube for each concentration in each preparation instance. Therefore `tube_id` is deterministic from `(instance_id, concentration)`, using IDs such as `I1_C0.20`. All images of the same tube share this ID.
4. **Droplet/view identities.** `droplet_id` and `view_id` remain blank because reconstructing each droplet/photo pairing is difficult and is not required for the planned tube-level evaluation. The primary split unit remains `tube_id`.
5. **Lamp assignment for added cohorts.** Instance 2 uses the experimenter-confirmed ordering of six lamp-on then six lamp-off rows per concentration. Instance 3 uses two lamp-on then two lamp-off rows per concentration.
6. **Timestamp handling.** Missing timestamps are not inferred. Machine-generated timestamps must retain seconds precision; rebuild metadata from the source workbooks and image filenames rather than editing/re-saving the CSV in spreadsheet software.
7. **Metadata source of truth.** `src/build_metadata.py` is updated to generate `instance_id` and `tube_id` directly in `data/processed/metadata.csv`. The separate enriched CSV is an intermediate check, not the canonical rebuild target.
8. **Known integrity issues retained for documentation.** The original cohort has uneven counts at specified concentration/condition combinations, and Dino-Lite sequence A056 occurs twice with different file paths/timestamps. These are not silently corrected or deleted.

## Phase 3 — Revised 60/20/20 split — 16 Nov 2026

The initial one-test-tube-per-eligible-concentration split was revised before feature extraction. The dataset contains 51 physical tubes, so a 60/20/20 allocation was adopted:

- 31 training tubes
- 10 validation tubes
- 10 test tubes

The split is performed at the physical tube level. Seven single-tube concentrations remain in training. Four three-tube concentrations contribute one tube to each partition. For the sixteen two-tube concentrations, at least one tube remains in training; four pairs remain fully in training, six contribute a validation tube and six contribute a test tube.

Seed 42 is used for deterministic assignment, with instance balancing across the three-tube concentration groups. All 27 concentration levels remain represented in training.

Nested GroupKFold uses the 31 training tubes. The ten validation tubes are reserved for assessment of the selected candidate. Test tubes remain locked for Phase 7.

The split protocol is described in reports/evaluation_plan.md and implemented by src/make_splits.py.

## Phase 3.6 — Metric definitions — 16 Nov 2026

The primary metric is MAE in percentage points of added urea (% w/v).
Additional metrics are RMSE, R², bias (predicted minus actual), and
the fraction of predictions within ±0.1 percentage points.

Band-wise MAE uses actual concentrations:
low <0.5%, medium 0.5–1.0% inclusive, and high >1.0%.

Percentage error is excluded because it is undefined at zero and
unstable near zero.

Calibration is checked by regressing actual concentration on predicted
concentration. Ideal slope is 1 and intercept is 0.

The optional detection analysis evaluates thresholds 0.2%, 0.5% and
1.0%, using ROC AUC, sensitivity and specificity.

Metrics are implemented in src/metrics.py and tested in
tests/test_metrics.py. No model results are produced in this phase.

## Phase 3.7 — Baselines — 16 Nov 2026

Two baselines are implemented:

- B0: predict the training-set mean concentration.
- B1: linear regression using median b* as its only feature.

B1 uses a training-fitted median imputer if feature values are missing.
All fitting and preprocessing must occur within the training fold.

B2, the old unmasked-channel-mean pipeline with random image split,
is omitted because the previous pipeline implementation is unavailable.

Implementation: src/baselines.py
Tests: tests/test_baselines.py

## Phase 3.8–3.9 — Uncertainty and plan control — 16 Nov 2026

### Uncertainty

Uncertainty estimation will use a cluster bootstrap by physical tube.
The procedure will use 2,000 resamples with replacement and report
the 2.5th–97.5th percentile interval.

Model comparisons will use paired bootstrap differences, resampling
the same physical tubes for both models.

Bootstrap implementation is deferred to Phase 6, as specified
in the roadmap.

### Evaluation-plan change control

The evaluation plan is maintained in reports/evaluation_plan.md.
It must be committed before Phase 5.

Every future change to the plan must be documented in this decision
log with its reason and consequences.

The locked test partition must remain unchanged and untouched
until Phase 7.

## Phase 4 — Instance-based exploratory analysis — 17 Nov 2026

1. **Current split policy:** Retain the 60/20/20 tube-level split across all 51 tubes and all three preparation instances. This supersedes the earlier decision to keep the November 4 cohort entirely separate from development/model selection. The role column in metadata.csv now determines whether an image belongs to development or test.

2. **Development data:** Phase 4 analyses use only train and validation images. All test images remain excluded and unopened.

3. **Instance-based grouping:** Use instance_id as the preparation-batch grouping variable for exploratory plots and comparisons. This replaces session/timestamp grouping as the primary analysis.

4. **Timestamp handling:** Timestamps remain in metadata for provenance but are not used for Phase 4 grouping or temporal-trend plots. Temporal drift within an instance is not assessed by this analysis.

5. **Lab plots:** Plot tube-level mean L*, a* and b* against concentration, with points coloured by instance and separate panels for device/lamp conditions. Averaging photographs within a tube prevents tubes with more photographs from receiving greater weight.

6. **Instance comparisons:** Compare b* at matched concentrations across available instances and acquisition conditions. These are comparisons between separate physical tubes, not repeated measurements of the same tube.

7. **Variance analysis:** Use a descriptive nested sum-of-squares decomposition for concentration, tube within concentration, and photograph within tube. Analyse device and lamp differences separately using matched measurements from the same physical tube.

8. **Interpretation limitation:** There is one physical tube per concentration within each instance. Consequently, between-tube variation at a fixed concentration is also between-instance variation; these effects cannot be independently separated.

9. **Image quality:** Review development images using the predefined inclusion rules. Record every exclusion by image_id and reason. Do not exclude images based on concentration, colour, instance, or future model performance.

### Phase 5.1–5.2: ROI erosion and glare removal — frozen decision

**Decision:** Freeze the erosion fraction at 3.5% of the original image width.

**Implementation:**
- Recover the original droplet segmentation masks.
- Erode masks using a circular structuring element equivalent to the distance-transform implementation.
- Remove specular glare using the existing HSV-equivalent thresholds: V > 0.92 and S < 0.25.
- Preserve original photographs and original segmentation masks.
- Process only training and validation images. Test images remain excluded.

**Processing results:**
- 864 development images processed.
- 717 candidate.
- 135 needs review.
- 11 source-mask flagged.
- 1 failed because the cleaned mask was empty.

**Decision rationale:** The 3.5% setting is retained as a practical fixed preprocessing parameter to allow the project to proceed within the available time. A complete visual comparison of the 3%, 3.5%, and 4% previews was not performed.

**Limitations:** The erosion parameter is proportional to image width, not droplet size. Phone images retain less mask area after erosion than Dino-Lite images. Glare removal also differs substantially between devices. These differences may introduce device-dependent effects into extracted colour features.

**Reproducibility:** The frozen parameter and glare thresholds must remain unchanged across training, validation, and test processing. The same preprocessing procedure must be applied to all three splits.


## Phase 5.3–5.4 — CIELAB colour feature extraction — 17 Nov 2026

### Colour-space decision

**Decision:** Use `skimage.color.rgb2lab` and retain its CIELAB values directly.

- `L*` uses the approximately 0–100 scale.
- `a*` and `b*` retain their signed values.
- Use scikit-image's D65 illuminant and 2-degree observer defaults.
- Do not rescale the saved model features to OpenCV's 8-bit LAB representation.
- For comparison with old OpenCV-style plots only, transform values using:
  - `L_old = 2.55 * L*`
  - `a_old = a* + 128`
  - `b_old = b* + 128`
- These transformed values are for comparison only. The canonical stored features remain in scikit-image CIELAB units.

### Feature definitions

Features are calculated from the cleaned ROI RGBA images produced by Phase 5.1–5.2. A pixel is included only when its alpha value is greater than 127.

For each of L*, a* and b*, extract:
- Mean
- Median
- Population standard deviation (`ddof=0`)
- 10th percentile
- 90th percentile

Additional features:
- Chroma: calculate `sqrt(a*² + b*²)` for each included pixel and save its mean, median, standard deviation, 10th percentile and 90th percentile.
- Hue angle: calculate pixel-level `atan2(b*, a*)` and save the circular mean angle in degrees, in the range [0, 360).
- Pale fraction: fraction of cleaned ROI pixels whose b* is at least 8 units below the median b* in the top third of that image's droplet ROI bounding box.

The pale-pixel threshold is an operational heuristic motivated by the A019 observation. It is not a manually labelled ground-truth classification.

### QC fields

QC fields are prefixed with `qc_` and must not be used as model inputs.

Record:
- `qc_transparent_fraction`: fraction outside the original pre-erosion droplet mask.
- `qc_droplet_area_fraction`: original droplet-mask area divided by image area.
- `qc_glare_fraction`: detected glare pixels removed during trimming divided by original droplet-mask pixels.
- `qc_glare_fraction_of_eroded_roi`: glare fraction using the eroded mask as denominator.
- Additional diagnostics: cleaned ROI area, pixel counts, erosion retention, source mask status, image dimensions, and pale-pixel reference threshold.

The glare fraction uses the glare pixels detected inside the eroded ROI. Its numerator and denominator must be interpreted according to these definitions.

All QC fields are diagnostics, not model inputs.

### Processing scope

- Process only images assigned `train` or `validation` roles.
- Do not open or use test images.
- Preserve one row per development image_id in `data/processed/features.csv`.
- Keep feature data separate from metadata. Join using `image_id`.
- Retain rows with `needs_review` and `source_mask_flagged` trimming statuses when cleaned ROI outputs exist.
- Retain failed rows with `feature_status=failed` and missing feature values rather than silently deleting them.
- Do not modify original photographs, segmentation masks, or frozen Phase 5.1–5.2 outputs.

### A019 reference unit test

Use the A019 photograph as the first data-specific unit test.

Expected approximate values:
- `b_med = 24.3`
- `L_mean = 82.1`

Use an absolute tolerance of 0.5 for each value. If the test fails, first verify the A019 image match and the preprocessing pipeline. Do not change the frozen erosion or glare thresholds merely to force the reference values to match.

### Reproducibility

The feature extraction implementation is in:
- `src/extract_colour_features.py`
- `tests/test_colour_features.py`

Output:
- `data/processed/features.csv`

The feature extraction script must use the same frozen 3.5% erosion and glare-removal outputs for all development images.

Phase 5.5 feature-set ablations (S0–S4) have not yet been implemented or selected.


## Phase 5.3–5.4 — A019 reference reconciliation — 17 Nov 2026

### A019 reference discrepancy

**Image verified:** `dino_off_019_20240911_170044`

- Instance: 1
- Tube: I1_C0.25
- Concentration: 0.25%
- Role: train
- Device: Dino-Lite
- Lamp: off
- Original image dimensions: 787 × 819

The roadmap's Phase 5.4 example states approximate reference values:
- b_med ≈ 24.3
- L_mean ≈ 82.1

The current implementation produces:
- b_med = 25.6897
- L_mean = 81.6963

L_mean is within the original ±0.5 tolerance, but b_med is not.

### Investigation

The following alternatives were evaluated on the exact A019 original photograph and source segmentation mask:

| Processing | L_mean | b_med |
|---|---:|---:|
| Original mask, no glare removal | 81.756 | 24.598 |
| Original mask, glare removed | 81.202 | 24.828 |
| Legacy repeated binary erosion + glare removal | 81.667 | 25.555 |
| Frozen circular erosion + glare removal | 81.696 | 25.690 |

The frozen circular erosion uses a radius of 29 pixels, corresponding to 3.5% of the 819-pixel image width.

The legacy erosion uses 28 iterations of the default SciPy binary erosion structure. Its mask differs from the frozen circular mask by 10,697 pixels.

The saved cleaned ROI was compared with an independently reconstructed frozen circular erosion followed by glare removal. The number of differing pixels was zero.

### Decision

1. Retain the frozen 3.5% circular erosion and existing glare thresholds.
2. Do not change preprocessing parameters to force agreement with the roadmap's approximate A019 reference.
3. Preserve the roadmap's reference values as historical reference values, not as verified expected values for the current ROI pipeline.
4. Replace the strict A019 reference assertion with a regression test for the actual current pipeline output.
5. Record the source-mask/pre-cut-image difference as an unresolved provenance limitation. The exact cause cannot be established from the current data alone.

Current A019 regression values:
- b_med = 25.6897
- L_mean = 81.6963
- Regression tolerance = ±0.05

This test verifies reproducibility of the current implementation. It does not establish that the old reference calculation and the current ROI pipeline are equivalent.

The preprocessing pipeline remains frozen.


## Phase 5.5 — Feature-set ablation preparation — 17 Nov 2026

**Decision:** Prepare five nested feature sets (S0–S4), each with a colour-only variant and a variant that additionally includes device and lamp one-hot variables.

The feature sets will be compared using development data in Phase 6. No feature set is selected as best in Phase 5.5.

### Feature definitions

**S0 — Unmasked RGB means**
- Mean R, G and B over the entire segmentation image.
- Pixels outside the original droplet mask are explicitly represented as black (RGB = 0).
- Alpha is ignored during averaging, so transparent-background pixels contribute to the mean.
- This approximates the old cut-out PNG baseline using the current original-photo-derived mask.

**S1 — Masked means**
- Mean L*, a* and b* inside the cleaned ROI.

**S2 — Masked distribution statistics**
- S1 plus median and population standard deviation of L*, a* and b*.

**S3 — Extended colour distribution**
- S2 plus 10th and 90th percentiles for L*, a* and b*.
- Add pale-pixel fraction.
- Retain the Phase 5.4 chroma statistics and hue angle.

**S4 — RGB and HSV extension**
- S3 plus mean, median, population standard deviation, 10th percentile and 90th percentile for R, G, B, H, S and V inside the cleaned ROI.
- RGB uses the 0–255 scale; HSV uses normalized [0, 1] values.
- Circular mean hue is also retained.

### Device and lamp variants

Every feature set has two variants:
1. Colour-only.
2. Colour plus four one-hot columns: `device_dino`, `device_phone`, `lamp_off`, `lamp_on`.

### Leakage and QC controls

- All `qc_` columns are excluded from model feature lists.
- `image_id`, `tube_id`, `conc`, `role`, raw device/lamp categories, paths and other metadata are excluded from feature lists.
- Concentration is the target, and tube_id is the grouping variable for cross-validation.
- No imputation, scaling, feature selection or other learned transformation is performed in Phase 5.5. These must be fitted inside training folds in Phase 6.

### Scope and output

- Development images only: train and validation.
- Test images remain unopened.
- Preserve one row per development image, including rows with failed feature extraction.
- Do not silently drop flagged or warning images when usable cleaned ROI files exist.
- Save the combined table as `data/processed/dataset.csv`.
- Save the exact feature lists as `data/processed/feature_set_columns.json`.

The actual ablation comparison and choice of feature set are deferred to Phase 6 development cross-validation. No test data may be used to select features.


## Phase 5.6–5.7 — Leakage-safe preprocessing and final dataset — 17 Nov 2026

### Phase 5.6 — Learned preprocessing

**Decision:** All learned preprocessing must be contained inside a scikit-learn Pipeline and passed as a complete estimator to cross-validation.

The following operations must be fitted using only the training portion of each fold:
- Median imputation
- Feature scaling and standardization
- Feature selection
- PCA

No global scaling, imputation, feature selection or PCA is permitted before cross-validation.

A reusable pipeline constructor is implemented in `src/model_pipelines.py`.

B1 uses a median imputer and linear regression inside a pipeline. Ridge and other scale-sensitive models use median imputation followed by standardization. Optional feature selection and PCA are also available as pipeline steps.

### Phase 5.7 — Feature and dataset files

**Decision:** Keep the feature table separate from metadata and join using `image_id`.

Outputs:
- `data/processed/features.csv`: one row per development image, containing S0–S4 features, extraction status and QC diagnostics.
- `data/processed/dataset.csv`: development metadata joined with the feature table using a validated one-to-one image_id merge.
- `data/processed/feature_set_columns.json`: exact model input columns for all five feature sets, with and without device/lamp.

The current dataset contains 864 development images:
- 666 training images
- 198 validation images

The locked test partition is intentionally excluded. This differs from a literal all-image Phase 5 dataset so that the project’s pre-existing test lock remains intact. The 200 test images are not opened or used for preprocessing, feature selection or model development.

### Unit tests and validation

The following checks are required:
1. Transparent pixels with alpha = 0 do not affect masked colour features.
2. A uniformly coloured synthetic image returns its known CIELAB values.
3. Feature and dataset row counts match development metadata, with unique image_id values.
4. Feature registries contain no targets, IDs, metadata or QC columns.
5. Learned preprocessing steps are fitted independently inside each training fold.

The existing A019 regression test checks reproducibility of the current frozen ROI pipeline. Its difference from the original roadmap reference is documented above.

**Completion criteria:** The feature table and development dataset contain all available S0–S4 features, tests pass, and all learned transformations are fold-local. The test partition remains untouched until Phase 7.
