
# Decision Log

Record important experimental, data-processing and modelling decisions, together with their rationale, evidence and consequences.

## 1. Dataset and experimental design

### 23 October 2026 — Initial dataset organization

The dataset contains three preparation cohorts:

- Original cohort: Dino-Lite and Samsung A31 photographs.
- New-tube cohort: 456 photographs.
- November 4 cohort: 32 photographs.

All 1,064 original photographs are retained in the master metadata.

Original photographs are the source of truth. The active `image_path` points to the original BMP/JPG image, while `orig_path` preserves the path recorded in the source spreadsheet. Pre-cut transparent PNGs are not used as original model inputs.

The original plan was to keep the November 4 cohort separate from development. This was subsequently superseded by the revised tube-level split documented in Phase 3 and Phase 4.

### 16 November 2026 — Preparation instances and tube identities

The three preparation instances are defined as:

| Cohort | instance_id |
|---|---:|
| Original | 1 |
| New tubes | 2 |
| November 4 | 3 |

There is exactly one physical tube per concentration within each preparation instance. Therefore:

`tube_id = (instance_id, concentration)`

Example: `I1_C0.20`.

All photographs of the same physical tube, including different devices and lamp conditions, share its `tube_id`.

`droplet_id` and `view_id` remain blank because reliable reconstruction of individual droplet/photo pairings is not available or required for tube-level evaluation.

Lamp assignment for the added cohorts follows experimenter-confirmed acquisition ordering:

- Instance 2: six lamp-on followed by six lamp-off photographs per concentration.
- Instance 3: two lamp-on followed by two lamp-off photographs per concentration.

Missing timestamps are not inferred. Timestamps are retained for provenance, with seconds precision where available.

Known data irregularities are retained rather than silently corrected:
- Uneven photograph counts at certain concentration/device/lamp combinations.
- Duplicate Dino-Lite sequence A056 with different paths and timestamps.
- Forty-seven original phone/off photographs without timestamps.

`src/build_metadata.py` generates the canonical metadata, including `instance_id` and `tube_id`.

## 2. Train, validation and test split

### 16–17 November 2026 — Revised tube-level split

The original one-test-tube-per-eligible-concentration split was revised to a 60/20/20 allocation across the 51 physical tubes:

| Partition | Physical tubes |
|---|---:|
| Training | 31 |
| Validation | 10 |
| Test | 10 |

The assignment uses seed 42 and preserves representation of all 27 concentration levels in training.

Seven single-tube concentrations remain in training. Four three-tube concentrations contribute one tube to each partition. The two-tube concentration groups are allocated so that at least one tube remains in training.

The split is performed by physical `tube_id`, never by individual photograph.

Nested/grouped cross-validation uses training tubes. Validation tubes are reserved for later candidate assessment, and test tubes remain locked until Phase 7.

Implementation: `src/make_splits.py`

Evaluation protocol: `reports/evaluation_plan.md`

## 3. Metrics and baselines

### 16 November 2026 — Evaluation metrics

The primary metric is MAE in percentage points of added urea (% w/v).

Additional metrics:
- RMSE
- R²
- Bias (prediction minus actual)
- Fraction of predictions within ±0.1 percentage points
- Band-wise MAE

Concentration bands:
- Low: below 0.5%.
- Medium: 0.5–1.0%, inclusive.
- High: above 1.0%.

Percentage error is excluded because it is undefined at zero and unstable near zero.

Calibration is assessed using the slope and intercept from regressing actual concentration on predicted concentration. Ideal slope is 1 and intercept is 0.

Optional detection analysis uses thresholds 0.2%, 0.5% and 1.0%, with ROC AUC, sensitivity and specificity.

Implementation: `src/metrics.py`

Tests: `tests/test_metrics.py`

### 16 November 2026 — Baselines

Two baselines are implemented:

- **B0:** Predict the training-fold mean concentration.
- **B1:** Linear regression using median b* (`b_med`) as its only feature.

B1 uses training-fitted median imputation where required.

All learned preprocessing must be fitted inside the training fold.

B2, the previous unmasked-channel-mean model with random image splitting, is omitted because the previous implementation is unavailable.

Implementation: `src/baselines.py`

Tests: `tests/test_baselines.py`

## 4. Phase 4 — Exploratory data analysis

### 17 November 2026 — Instance-based analysis

The 60/20/20 split across all 51 tubes and three preparation instances is retained. This supersedes the initial plan to keep the November 4 cohort entirely separate.

Phase 4 development analysis uses training and validation images. Test images remain excluded.

`instance_id` is the primary preparation-batch grouping variable. Timestamps remain provenance information and are not used for the primary grouping or temporal-trend analysis.

Exploratory analysis includes:
- Tube-level mean L*, a* and b* against concentration.
- Comparisons across instances at matched concentrations and acquisition conditions.
- Separate device/lamp comparisons using matched measurements from the same physical tube.
- Descriptive variance decomposition across concentration, tube within concentration, and photograph within tube.

There is one physical tube per concentration within each instance. Consequently, between-tube and between-instance variation cannot be independently separated.

### Image-quality limitation

Contact sheets were inspected, but complete manual image-by-image quality classification was skipped because it was impractical within the project timeline.

All structurally valid images were retained. No photograph was excluded based on its concentration, colour or potential model performance.

Device and lamp quality differences observed during inspection are treated as qualitative assumptions, not measured image-quality labels.

## 5. Phase 5 — Image preprocessing and features

### 5.1–5.2 ROI erosion and glare removal — 17 November 2026

**Decision:** Freeze the erosion fraction at 3.5% of original image width.

The pipeline:
- Starts from the original segmentation masks.
- Uses circular erosion, implemented through a distance-transform approach.
- Removes bright, low-saturation glare using the equivalent thresholds V > 0.92 and S < 0.25.
- Preserves the original photographs and segmentation masks.

The preprocessing parameter is retained as a practical fixed choice due to the project deadline. A complete visual comparison of the 3%, 3.5% and 4% previews was not performed.

Processing results across 864 development images:

| Status | Images |
|---|---:|
| Candidate | 717 |
| Needs review | 135 |
| Source-mask flagged | 11 |
| Failed | 1 |
| Total | 864 |

The 3.5% erosion is proportional to image width rather than droplet size. Phone images retain less mask area after erosion than Dino-Lite images, and glare removal differs between devices. These differences may affect extracted colour features.

The same frozen preprocessing procedure must be used across all partitions. Test images are not opened during development.

Implementation: `src/trim_roi_masks.py`

Outputs:
- `data/interim/eroded_masks/`
- `data/interim/cleaned_roi_rgba/`
- `data/interim/cleaned_roi_crops/`
- `data/interim/phase5_trim_manifest.csv`

### 5.3–5.4 CIELAB colour features — 17 November 2026

**Decision:** Use `skimage.color.rgb2lab` with its D65 illuminant and 2-degree observer defaults. Store genuine CIELAB values without OpenCV-style rescaling.

- L*: approximately 0–100.
- a* and b*: signed values.

For comparison with historical OpenCV-style plots only:

- `L_old = 2.55 * L*`
- `a_old = a* + 128`
- `b_old = b* + 128`

For each of L*, a* and b*, extract:
- Mean
- Median
- Population standard deviation (`ddof=0`)
- 10th percentile
- 90th percentile

Additional features:
- Chroma: `sqrt(a*² + b*²)`, with the same five summary statistics.
- Hue angle: circular mean of pixel-level `atan2(b*, a*)`, in degrees.
- Pale fraction: fraction of cleaned ROI pixels whose b* is at least 8 units below the median b* of the top third of the droplet ROI bounding box.

The pale-pixel threshold is an operational heuristic motivated by the observed A019 droplet. It is not a manually labelled ground-truth classification.

QC fields are retained for diagnostics, not model inputs. They include:
- Transparent fraction.
- Original droplet area fraction.
- Glare fraction.
- Glare fraction relative to the eroded ROI.
- Cleaned ROI area and pixel count.
- Erosion retention.
- Source mask status.
- Image dimensions and pale-pixel reference threshold.

Only training and validation images are processed. Rows with usable cleaned ROIs are retained even if their trimming status is `needs_review` or `source_mask_flagged`. Failed rows remain in the feature table with missing feature values and a failure status.

Implementation: `src/extract_colour_features.py`

Output: `data/processed/features.csv`

### A019 reference discrepancy

The A019 image was verified as:

- `dino_off_019_20240911_170044`
- Instance 1, tube `I1_C0.25`
- Concentration 0.25%
- Training partition
- Dino-Lite, lamp off
- Original dimensions: 787 × 819

The roadmap's historical reference values were approximately:

- b_med = 24.3
- L_mean = 82.1

The current frozen ROI pipeline produces:

- b_med = 25.6897
- L_mean = 81.6963

The original reference values are not reproduced by the current pipeline.

Investigation on the exact original photograph and source mask:

| Processing | L* mean | b* median |
|---|---:|---:|
| Original mask, no glare removal | 81.756 | 24.598 |
| Original mask, glare removed | 81.202 | 24.828 |
| Legacy repeated binary erosion + glare removal | 81.667 | 25.555 |
| Frozen circular erosion + glare removal | 81.696 | 25.690 |

The legacy erosion uses 28 iterations of SciPy's default binary erosion structure. Its mask differs from the frozen circular erosion mask by 10,697 pixels.

The saved cleaned ROI matches an independently reconstructed frozen circular erosion followed by glare removal exactly (zero differing pixels).

**Decision:** Retain the frozen 3.5% circular erosion and glare thresholds. Do not change preprocessing to force agreement with the historical reference values. The exact provenance difference between the old reference and current ROI cannot be established from the available data.

The A019 test is therefore a regression test for the current pipeline:

- b_med = 25.6897
- L_mean = 81.6963
- Absolute tolerance = ±0.05

This verifies reproducibility of the current implementation, not equivalence with the historical reference calculation.

### 5.5 Feature sets S0–S4 — 17 November 2026

Five feature sets were constructed for later development ablation:

| Set | Features |
|---|---|
| S0 | Unmasked RGB means over the full segmentation image, with transparent background represented as black |
| S1 | Masked L*, a*, b* means |
| S2 | S1 plus medians and standard deviations |
| S3 | S2 plus 10th/90th percentiles, pale fraction, chroma statistics and hue angle |
| S4 | S3 plus RGB and HSV statistics inside the cleaned ROI |

Each set has:
- A colour-only variant.
- A variant with `device_dino`, `device_phone`, `lamp_off` and `lamp_on` one-hot inputs.

QC columns, identifiers, target concentration, split role, paths and raw metadata are excluded from model feature lists.

Feature-set sizes:

| Set | Colour-only | With device/lamp |
|---|---:|---:|
| S0 | 3 | 7 |
| S1 | 3 | 7 |
| S2 | 9 | 13 |
| S3 | 22 | 26 |
| S4 | 53 | 57 |

No feature set was selected as best during Phase 5. The comparison is deferred to Phase 6.

Implementation: `src/build_feature_sets.py`

Outputs:
- `data/processed/dataset.csv`
- `data/processed/feature_set_columns.json`

### 5.6–5.7 Learned preprocessing and dataset validation

All learned operations must be fitted inside the training fold through a scikit-learn Pipeline. This includes imputation, scaling, feature selection and PCA.

No global fitting of these transformations is permitted before cross-validation.

B1 uses median imputation. Ridge and other scale-sensitive models use median imputation and standardization inside their pipelines.

A reusable pipeline constructor is available in `src/model_pipelines.py`.

The feature table is maintained separately from metadata and joined using `image_id`.

Final development dataset:

| Partition | Images |
|---|---:|
| Training | 666 |
| Validation | 198 |
| Total development | 864 |

The 200 locked test images are excluded from `features.csv` and `dataset.csv` and remain unopened until Phase 7.

Files:
- `data/processed/features.csv`
- `data/processed/dataset.csv`
- `data/processed/feature_set_columns.json`

Unit tests verify alpha masking, synthetic CIELAB values, A019 pipeline reproducibility, row counts, unique image IDs, feature registry exclusions and fold-local preprocessing.

All tests passed after the A019 regression test was updated to the verified current-pipeline values.

## 6. Phase 6 — Model experiments

### 6.1 Walking skeleton — 17 November 2026

The tube-identity and condition audit passed:

- 51 physical tubes checked.
- Instance 1: 24 tubes, all with four expected conditions.
- Instance 2: 19 tubes, all with four expected conditions.
- Instance 3: 8 tubes, all with two expected Dino-Lite conditions.
- Each instance/concentration maps to one tube ID.
- Each tube maps to one instance, concentration and split role.
- No tube crosses train, validation or test partitions.

The first experiment used:
- 666 training images.
- 31 training tubes.
- Five-fold GroupKFold grouped by `tube_id`.
- The same folds for B0, B1 and Ridge-S1.
- Validation and test partitions excluded.

Models:
- B0: training-fold mean predictor.
- B1: linear regression on `b_med`.
- Ridge-S1: Ridge using `L_mean`, `a_mean`, `b_mean`, fixed alpha = 1.0.

Pooled outer out-of-fold results:

| Model | MAE | RMSE | R² | Bias | Within ±0.1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.546092 | 0.622511 | -0.040597 | 0.000703 | 0.072072 |
| B1 | 0.355177 | 0.434717 | 0.492541 | -0.003290 | 0.166667 |
| Ridge-S1 | 0.306479 | 0.386162 | 0.599570 | -0.005465 | 0.189189 |

Ridge-S1 improved on both baselines. This established that the image-feature-to-prediction pipeline worked end to end.

Outputs:
- `results/metrics.csv`
- `results/walking_skeleton_oof.csv`
- `results/condition_tube_consistency.csv`
- `results/figures/phase6_1_walking_skeleton_predicted_vs_actual.png`

### 6.2–6.3 Model ladder and nested tuning — 17 November 2026

The model ladder reused the exact five outer folds from Phase 6.1.

Protocol:
- Outer loop: five-fold GroupKFold.
- Inner loop: four-fold GroupKFold.
- Grouping variable: `tube_id`.
- Inner `GridSearchCV.fit()` receives the outer-training groups.
- Scoring: negative MAE.
- All imputing, scaling and transformations are fitted inside their respective training folds.
- Validation and test partitions remain excluded.

Models and features:
- B0: training-fold mean predictor.
- B1: linear regression on `b_med`.
- Ridge-S2: from-scratch Ridge on nine S2 Lab features.
- Polynomial Ridge: degree-2 polynomial expansion of `b_med`.
- RBF-SVR: S2.
- Random Forest: S2, 300 trees.
- Gradient Boosting: S2, subsample 0.7.
- MLP: S2, one hidden layer.

Search grids contain no more than 20 combinations:
- Ridge: 7.
- Polynomial Ridge: 7.
- RBF-SVR: 18.
- Random Forest: 3.
- Gradient Boosting: 8.
- MLP: 9.

The Ridge closed-form and gradient-descent implementations were verified against scikit-learn Ridge to a prediction tolerance of 1e-6 on synthetic data. The from-scratch closed-form implementation was used in the model ladder.

Pooled outer out-of-fold results:

| Model | MAE | RMSE | R² | Bias | Within ±0.1 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.546092 | 0.622511 | -0.040597 | 0.000703 | 0.072072 |
| B1 | 0.355177 | 0.434717 | 0.492541 | -0.003290 | 0.166667 |
| Ridge-S2 | 0.302931 | 0.384902 | 0.602177 | -0.009518 | 0.211712 |
| Polynomial Ridge | 0.348651 | 0.429725 | 0.504129 | -0.002985 | 0.169670 |
| RBF-SVR S2 | 0.221360 | 0.295599 | 0.765365 | 0.026735 | 0.319820 |
| Random Forest S2 | 0.210330 | 0.286353 | 0.779813 | 0.014111 | 0.336336 |
| Gradient Boosting S2 | 0.215895 | 0.295152 | 0.766074 | 0.004475 | 0.337838 |
| MLP S2 | 0.234756 | 0.323485 | 0.719006 | -0.030098 | 0.300300 |

Random Forest achieved the lowest raw MAE and highest R². Gradient Boosting had the highest fraction within ±0.1. RBF-SVR was also competitive.

The Random Forest and Gradient Boosting MAE difference was only 0.005565 percentage points. No significance claim was made from these point estimates alone.

MLP convergence warnings occurred because L-BFGS reached its 3,000-iteration limit in several fits. Its result is retained but is not considered a consistently converged comparison.

Outputs:
- `results/metrics.csv`
- `results/model_ladder_oof.csv`
- `results/model_ladder_best_params.csv`
- `results/figures/phase6_2_model_ladder_predicted_vs_actual.png`

### 6.4 One-standard-error model selection — 17 November 2026

**Decision:** Apply the one-standard-error rule using a cluster bootstrap by physical tube.

Bootstrap protocol:
- 2,000 resamples.
- Random seed 0.
- Resample the 31 training tubes with replacement.
- Recompute image-pooled MAE from per-tube error sums and image counts.
- Use the sample standard deviation of bootstrap MAEs as the standard error.
- Report the 2.5th and 97.5th percentiles.
- Use identical tube resampling draws across models.

The bootstrap operates on existing outer OOF predictions; it does not refit models.

Simplicity order, fixed before inspecting eligibility:

1. B0 — constant mean predictor.
2. B1 — one-feature linear regression.
3. Polynomial Ridge on b*.
4. Ridge-S2.
5. RBF-SVR-S2.
6. Random Forest-S2.
7. Gradient Boosting-S2.
8. MLP-S2.

Bootstrap results:

| Model | OOF MAE | Bootstrap SE | 95% interval | Within one SE |
|---|---:|---:|---:|---|
| B0 | 0.546092 | 0.054950 | [0.440801, 0.657280] | No |
| B1 | 0.355177 | 0.027685 | [0.300664, 0.409960] | No |
| Polynomial Ridge | 0.348651 | 0.026859 | [0.297385, 0.402581] | No |
| Ridge-S2 | 0.302931 | 0.027025 | [0.252143, 0.359578] | No |
| RBF-SVR S2 | 0.221360 | 0.020333 | [0.184076, 0.262500] | Yes |
| Random Forest S2 | 0.210330 | 0.019909 | [0.172472, 0.249744] | Yes |
| Gradient Boosting S2 | 0.215895 | 0.020629 | [0.178324, 0.256061] | Yes |
| MLP S2 | 0.234756 | 0.019608 | [0.199843, 0.276382] | No |

One-standard-error calculation:

- Best raw MAE: Random Forest S2 = 0.210330.
- Bootstrap SE of Random Forest = 0.019909.
- Cutoff = 0.210330 + 0.019909 = 0.230240.
- Eligible models: Random Forest S2, Gradient Boosting S2 and RBF-SVR S2.
- Simplest eligible model under the predeclared ordering: **RBF-SVR S2**.

**Decision:** Select `SVR_RBF_S2` as the current development candidate under the one-standard-error rule.

Random Forest has the best raw MAE, but SVR is selected because it is within one standard error of the best and is simpler under the declared ordering.

This is a provisional development candidate, not a final validated model. The 10 validation tubes remain reserved for candidate assessment, and the 10 test tubes remain locked.

No statistical significance claim is made from overlap of individual bootstrap intervals. Paired bootstrap differences have not yet been evaluated.

Outputs:
- `results/tube_bootstrap_summary.csv`
- `results/tube_bootstrap_samples.csv`
- `results/model_selection_one_se.csv`
- `reports/model_selection_6_4.md`
- `reports/final_candidate.json`

## 7. Current project status

Completed:
- Phases 1–5: data organization, splitting, exploratory analysis, preprocessing and feature construction.
- Phase 6.1: walking skeleton.
- Phase 6.2: model ladder.
- Phase 6.3: nested tube-grouped tuning.
- Phase 6.4: tube bootstrap and one-standard-error selection.

Current development candidate: **RBF-SVR using S2 colour features**.

The validation partition has not yet been used for candidate assessment. The locked test partition has not been opened or used. No final test performance or deployment claim is made at this stage.


## Phase 6.5 — Development experiments — 17 November 2026

### General protocol

The experiments E1–E5 use the training partition only:
- 666 training images.
- 31 physical training tubes.
- Validation: 10 tubes, reserved.
- Test: 10 tubes, locked.

All grouped evaluation uses `tube_id` as the grouping variable. No image from a held-out tube may appear in the corresponding training fold.

The saved Phase 6.1 outer-fold assignments are reused wherever the experiment uses the ordinary five-fold tube-grouped evaluation. Nested experiments use four-fold GroupKFold inside the outer training data for hyperparameter selection.

All learned preprocessing remains inside scikit-learn pipelines.

No validation or test result may be used to tune or select a model.

### E1 — Leakage demonstration (core)

**Question:** How optimistic is an image-level random split compared with a physical-tube split?

Use the same B1 model and same `b_med` feature in both evaluations.

- Random evaluation: five-fold shuffled image-level KFold, seed 42.
- Grouped evaluation: the exact five outer GroupKFold assignments from Phase 6.1.
- Both evaluations use the 666 training images only.
- Record how many physical tubes occur on both sides of each random split.
- Compare pooled MAE, RMSE and R².

The random split deliberately allows photographs of the same physical tube to occur in training and validation folds. The grouped split prevents this. The difference demonstrates the effect of leakage.

### E2 — Feature ablation (core)

**Question:** How much does masking and richer colour information improve prediction?

Compare S0, S1, S2, S3 and S4 using:
- Ridge regression.
- RBF-SVR.

Evaluate both colour-only and colour-plus-device/lamp variants.

Use the same five tube-grouped outer folds for all configurations and four-fold grouped inner CV for alpha or SVR hyperparameter selection.

All imputation and scaling are fitted inside the pipeline. Report pooled outer OOF metrics for each configuration.

No feature set is selected using validation or test data.

### E3 — Condition-specific and pooled model evaluation (core)

Evaluate the model ladder separately for:
- Dino-Lite, lamp off.
- Dino-Lite, lamp on.
- Phone, lamp off.
- Phone, lamp on.

Also evaluate the pooled dataset with device and lamp one-hot inputs.

For condition-specific experiments, use S2 colour-only features because device and lamp are constant within a condition. For the pooled experiment, use S2 plus device/lamp one-hot inputs.

Use the same model families as the Phase 6.2 model ladder, five-fold tube-grouped outer evaluation and four-fold grouped inner tuning. The same physical tube must remain in a single outer fold across all images and conditions.

Report MAE, RMSE, R², bias and the fraction within ±0.1 for each condition and the pooled experiment.

### E4 — Leave-one-concentration-out generalization (core)

**Question:** Can a model predict a concentration level that was absent from its training data?

Evaluate the three strongest models from Phase 6.2 by raw outer-CV MAE:
- Random Forest-S2.
- Gradient Boosting-S2.
- RBF-SVR-S2.

Use S2 colour-only features. In each outer iteration, hold out every training image belonging to one actual concentration value, including all physical tubes at that concentration. Train only on the remaining concentration levels.

Tune model hyperparameters using four-fold tube-grouped inner CV on the outer-training concentrations. No image from the held-out concentration may participate in fitting or tuning.

Report per-concentration MAE and pooled image-level MAE for each model. Also report the number of held-out images and tubes for every concentration.

This is a concentration-extrapolation/generalization diagnostic, not the final test evaluation.

### E5 — Cross-condition transfer (should)

**Question:** Does a model trained on one device/lamp condition generalize to another condition?

Use the selected development candidate, RBF-SVR with S2 colour-only features.

Evaluate all 16 source-condition/target-condition combinations:
- Dino-Lite off.
- Dino-Lite on.
- Phone off.
- Phone on.

For each matrix cell, use only tubes that have both the source and target condition. Split those common physical tubes into five outer folds. For each fold:
- Train only on source-condition images from training tubes outside the held-out fold.
- Predict target-condition images from the held-out tubes.
- Keep the held-out tube IDs disjoint from all training tube IDs.

Tune the SVR only within the source-condition training tubes using four-fold GroupKFold. Do not use target-condition images from training tubes as training inputs.

For every cell, report MAE, RMSE, number of common tubes and number of target images. The diagonal represents same-condition grouped evaluation; off-diagonal cells represent cross-condition transfer.

The phone conditions do not exist in Instance 3. Therefore, matched-tube counts can differ between cells and must be reported rather than treated as identical.

### Result logging and outputs

Each experiment must append its metrics to `results/metrics.csv` without overwriting previous results.

Additional outputs:
- E1: `results/e1_leakage_comparison.csv`
- E1: `results/e1_random_split_tube_overlap.csv`
- E2: `results/e2_feature_ablation.csv`
- E3: `results/e3_condition_model_ladder.csv`
- E4: `results/e4_loco_predictions.csv`
- E4: `results/e4_loco_summary.csv`
- E5: `results/e5_cross_condition_mae_matrix.csv`
- E5: `results/e5_cross_condition_details.csv`

All predictions must be generated without using validation or test data. Actual findings, differences, warnings and decisions will be added to this log after each experiment completes.



## Phase 6.5 — Development experiments E1–E5 — 17 November 2026

### Common evaluation scope

All five experiments used training data only:
- 666 images.
- 31 physical tubes.
- Validation: excluded.
- Locked test partition: excluded.

Tube-level evaluation uses `tube_id`. Learned preprocessing and hyperparameter tuning remain inside training folds.

### E1 — Leakage demonstration

**Question:** Does random image-level splitting produce an optimistic estimate compared with tube-grouped splitting?

The same B1 linear regression model and `b_med` feature were used.

| Split | MAE | RMSE | R² | Within ±0.1 |
|---|---:|---:|---:|---:|
| Random image-level KFold | 0.344528 | 0.424124 | 0.516971 | 0.190691 |
| Tube-grouped GroupKFold | 0.355177 | 0.434717 | 0.492541 | 0.166667 |

Random image-level splitting had an MAE approximately 0.010648 lower, corresponding to about 3.0% apparent improvement relative to the grouped result.

However, an average of 29.4 out of 31 physical tubes appeared on both sides of each random fold. Individual folds had 28–31 overlapping tubes.

**Interpretation:** Random image-level splitting violates the physical-tube independence requirement. It allows photographs of the same physical reaction to appear in both training and evaluation data. The measured optimism for B1 is modest, not dramatic, but the random split is still invalid for estimating performance on unseen tubes.

**Decision:** Retain tube-grouped CV as the required evaluation protocol. Do not use random image-level splitting for model selection or reported generalization performance.

Outputs:
- `results/e1_leakage_comparison.csv`
- `results/e1_random_split_tube_overlap.csv`
- `results/e1_random_oof.csv`

### E2 — Feature ablation S0–S4

**Question:** Does masking the background matter, and do richer colour features or acquisition metadata improve prediction?

Five feature sets were compared using Ridge and RBF-SVR, each with colour-only and colour-plus-device/lamp variants. All configurations reused the same five outer tube-grouped folds and four-fold inner grouped tuning.

#### RBF-SVR results

| Feature set | Colour-only MAE | With device/lamp MAE |
|---|---:|---:|
| S0 | 0.277957 | 0.256488 |
| S1 | 0.227257 | **0.191504** |
| S2 | 0.221360 | 0.209579 |
| S3 | 0.225678 | 0.210156 |
| S4 | 0.204667 | 0.197579 |

#### Ridge results

| Feature set | Colour-only MAE | With device/lamp MAE |
|---|---:|---:|
| S0 | 0.431783 | 0.399584 |
| S1 | 0.305726 | 0.316318 |
| S2 | **0.302931** | 0.302698 |
| S3 | 0.307514 | 0.301003 |
| S4 | 0.307080 | **0.284686** |

The best observed configuration was **RBF-SVR with S1 and device/lamp one-hot inputs**, with MAE 0.191504, RMSE 0.262491 and R² 0.814980.

Key findings:

1. S0 consistently performed worse than the masked feature sets. For SVR, moving from S0 to S1 reduced MAE from 0.277957 to 0.227257 without device/lamp inputs, and from 0.256488 to 0.191504 with them. This supports the importance of removing background pixels.
2. S1, containing only mean L*, a* and b*, was the best SVR feature set. Adding more colour statistics did not consistently improve performance.
3. Device/lamp one-hot inputs improved SVR results for all five feature sets.
4. The largest feature set, S4, did not outperform the compact S1 representation.
5. Ridge benefited less consistently from device/lamp inputs. Its best result was S4 with device/lamp, MAE 0.284686.

**Decision:** Masking the background is important for this dataset. Compact masked Lab means are a strong feature representation, and device/lamp metadata appears useful for pooled SVR. However, the E2 winner is not automatically declared the final candidate: its tube-bootstrap uncertainty and one-standard-error eligibility have not yet been evaluated against all E2 configurations.

Output:
- `results/e2_feature_ablation.csv`
- `results/e2_feature_ablation_oof.csv`
- `results/e2_feature_ablation_best_params.csv`

### E3 — Condition-specific and pooled model ladder

**Question:** How does model performance vary across acquisition conditions, and does device/lamp information help in pooled modelling?

The condition-specific experiments used fresh five-fold GroupKFold splits within each condition. This was necessary because the phone conditions contain 27 tubes, whereas the Dino-Lite conditions contain 31.

The pooled experiment used the exact saved Phase 6.1 outer folds and included device/lamp one-hot inputs.

Best observed model in each condition:

| Condition | Best model | MAE |
|---|---|---:|
| Dino-Lite, lamp off | Polynomial Ridge on b_med | 0.159982 |
| Dino-Lite, lamp on | Polynomial Ridge on b_med | 0.194515 |
| Phone, lamp off | RBF-SVR S2 | 0.244794 |
| Phone, lamp on | MLP S2 | 0.195460 |
| Pooled with device/lamp | MLP S2 | 0.207508 |

For Dino-Lite lamp-on, Polynomial Ridge (0.194515) and Random Forest (0.194567) were nearly tied.

Pooled results were closely grouped:

| Model | MAE |
|---|---:|
| MLP S2 + device/lamp | 0.207508 |
| Random Forest S2 + device/lamp | 0.208694 |
| Gradient Boosting S2 + device/lamp | 0.209307 |
| RBF-SVR S2 + device/lamp | 0.209579 |

Key findings:

- The best model differs by acquisition condition.
- Phone lamp-off was the most difficult condition among the four, with best MAE 0.244794.
- Dino-Lite conditions had strong results from a simple polynomial model on median b*.
- Pooled models with device/lamp inputs achieved MAEs around 0.208–0.210.
- The pooled results do not show a decisive advantage for one model family.

**Decision:** Acquisition condition materially affects the modelling problem. Report condition-specific performance in addition to pooled performance. Do not assume that a model's pooled performance describes every device/lamp condition equally well.

The condition-specific and pooled CV fold-generation methods differ as documented above; their results should not be treated as perfectly paired comparisons.

Output:
- `results/e3_condition_model_ladder.csv`
- `results/e3_condition_model_ladder_oof.csv`
- `results/e3_condition_model_ladder_best_params.csv`

### E4 — Leave-one-concentration-out evaluation

**Question:** Can the models predict concentration levels that are absent from their training data?

The three strongest models by raw MAE in the original Phase 6.2 model ladder were evaluated:
- Random Forest S2.
- Gradient Boosting S2.
- RBF-SVR S2.

All training images at one concentration were held out together. Hyperparameter tuning used four-fold tube-grouped CV on the remaining concentrations.

Pooled LOCO results:

| Model | MAE | RMSE | R² | Within ±0.1 |
|---|---:|---:|---:|---:|
| RBF-SVR S2 | **0.213590** | 0.291451 | 0.771903 | 0.355856 |
| Random Forest S2 | 0.222554 | 0.306796 | 0.747253 | 0.343844 |
| Gradient Boosting S2 | 0.227850 | 0.309712 | 0.742425 | 0.328829 |

SVR produced the best pooled LOCO result.

All 666 training images received predictions in the LOCO experiment. Each concentration was held out as a complete group. Most held-out concentration levels had one physical tube; some had two. A few levels had only four photographs, so their individual concentration-level MAEs are based on very small samples.

**Metric limitation:** Per-concentration R² is not meaningful because all actual targets within a held-out concentration group are constant. The extremely large negative values in the per-concentration summary are numerical artefacts from calculating R² with a near-zero target variance. Use per-concentration MAE, RMSE, bias, fraction within ±0.1, and counts. Pooled R² across all LOCO predictions remains meaningful.

A non-fatal scikit-learn parallel-configuration warning appeared during execution; the experiment completed and produced predictions.

**Decision:** SVR is the strongest of the three models in pooled unseen-concentration evaluation. This is encouraging, but the limited number of independent tubes per concentration means that generalization to unseen preparation batches remains uncertain.

Outputs:
- `results/e4_loco_predictions.csv`
- `results/e4_loco_summary.csv`
- `results/e4_loco_pooled_summary.csv`
- `results/e4_loco_best_params.csv`

### E5 — Cross-condition transfer

**Question:** Does a model trained on one device/lamp condition transfer to another?

The previously selected `SVR_RBF_S2` colour-only model was used. Each source/target condition pair used only physical tubes with both conditions available.

For each fold, the model was trained using source-condition images from training tubes and evaluated on target-condition images from held-out tubes. The same physical tube was not allowed on both sides.

The 4 × 4 MAE matrix:

| Source / Target | Dino off | Dino on | Phone off | Phone on |
|---|---:|---:|---:|---:|
| Dino off | 0.193308 | 0.637700 | 0.575471 | 0.576072 |
| Dino on | 0.778660 | 0.212077 | 0.519764 | 0.450544 |
| Phone off | 0.559533 | 0.566181 | 0.248953 | 0.544825 |
| Phone on | 0.571707 | 0.630908 | 0.444185 | 0.218741 |

Matched physical tubes:
- Dino-to-Dino cells: 31 tubes.
- Cells involving phone conditions: 27 tubes.

The diagonal same-condition MAEs range from 0.193308 to 0.248953. Off-diagonal cross-condition MAEs range from 0.444185 to 0.778660.

Key findings:

1. Same-condition evaluation is substantially better than cross-condition transfer.
2. Changing lamp condition causes large performance degradation even when the device remains the same.
3. Changing device causes large performance degradation even when the lamp state remains the same.
4. Transfer is directional: training Dino-Lite lamp-on and testing Dino-Lite lamp-off (MAE 0.778660) is much worse than the reverse direction (MAE 0.637700).
5. Phone and Dino-Lite conditions cannot be treated as interchangeable measurement sources without additional calibration or representative multi-condition training.

**Decision:** Acquisition-condition shift is a major limitation of the current colour-feature model. A pooled model's cross-validation performance must not be interpreted as evidence of reliable transfer to a new device/lamp condition.

The E5 matrix uses the earlier S2 colour-only SVR candidate. E2 subsequently identified a stronger S1 plus device/lamp configuration; that configuration has not yet been evaluated in E5.

Outputs:
- `results/e5_cross_condition_predictions.csv`
- `results/e5_cross_condition_details.csv`
- `results/e5_cross_condition_mae_matrix.csv`
- `results/e5_cross_condition_mae_pivot.csv`

### Phase 6.5 overall interpretation and candidate status

The five experiments establish the following:

- Random image-level splitting causes substantial tube overlap, although the measured MAE optimism for B1 was modest.
- Masking the background improves performance relative to the unmasked S0 baseline.
- More features do not necessarily improve prediction. S1 means were particularly effective for SVR.
- Device/lamp metadata improves pooled SVR results.
- Condition-specific performance differs, especially for phone lamp-off.
- SVR performs best among the tested models in pooled leave-one-concentration-out evaluation.
- Cross-condition transfer is poor, demonstrating that acquisition changes are a major source of domain shift.

The Phase 6.4 one-standard-error rule selected `SVR_RBF_S2` as the development candidate before E2 was performed.

E2 has since identified `SVR_RBF_S1` with device/lamp one-hot inputs as the best observed feature configuration (MAE 0.191504). This is a new candidate, but the one-standard-error rule has not yet been reapplied across all E2 configurations. Therefore, the Phase 6.4 selection is retained as the historical selection, while the S1-plus-device/lamp SVR is recorded as the leading new candidate pending uncertainty analysis.

No validation or test data were used in E1–E5. No final model or cross-condition deployment claim is made.


## Phase 6.5 follow-up — E2 model selection and E4 metric correction — 17 November 2026

### E2 — One-standard-error selection across feature configurations

The initial E2 ablation compared 20 configurations:
- Five feature sets: S0–S4.
- Two variants: colour-only and colour plus device/lamp.
- Two estimators: Ridge and RBF-SVR.

The best observed configuration was RBF-SVR with S1 plus device/lamp:
- MAE = 0.191504
- RMSE = 0.262491
- R² = 0.814980
- Fraction within ±0.1 = 0.396396

This result is better than the previously selected S2 colour-only SVR candidate from Phase 6.4. However, it has not yet been assessed by the one-standard-error rule across all 20 E2 configurations.

**Follow-up decision:** Apply a common tube bootstrap to all 20 configurations using:
- 2,000 bootstrap replicates.
- Random seed 0.
- Physical tube (`tube_id`) as the resampling unit.
- The same sampled tube indices for every configuration.
- Per-tube absolute-error sums and image counts to recompute image-pooled MAE.

The cutoff is the best E2 configuration's pooled OOF MAE plus its tube-bootstrap standard error.

Among eligible configurations, use this predeclared simplicity order:
1. Ridge before RBF-SVR.
2. Fewer input features before more input features.
3. Colour-only before device/lamp inputs when feature counts are otherwise equal.
4. Lower OOF MAE as the final tie-breaker.

The registry in `data/processed/feature_set_columns.json` supplies the feature counts. No configuration is selected from the bootstrap confidence-interval overlap.

Outputs:
- `results/e2_tube_bootstrap_summary.csv`
- `results/e2_tube_bootstrap_samples.csv`
- `results/e2_one_se_selection.csv`
- `reports/e2_one_se_selection.md`
- `reports/e2_candidate.json`

The outcome of this follow-up will determine whether the E2 S1-plus-device/lamp SVR replaces the Phase 6.4 S2 colour-only SVR as the current development candidate. Neither is a final validated model.

### E4 — Leave-one-concentration-out R² correction

Each E4 outer iteration holds out one concentration, so all actual target values within that held-out concentration group are identical.

Consequently, R² for an individual held-out concentration is undefined. The extremely large negative values previously written in `results/e4_loco_summary.csv` are numerical artefacts and must not be interpreted.

**Correction:** Rebuild the per-concentration summary with R² explicitly set to missing/undefined. Retain the meaningful per-concentration metrics:
- MAE
- RMSE
- Bias
- Fraction within ±0.1
- Number of images and physical tubes

Retain the pooled R² in `results/e4_loco_pooled_summary.csv`. The pooled evaluation combines different concentration levels, so its R² remains meaningful.

This is a reporting correction only. E4 models are not retrained, and no predictions are changed.

Outputs:
- Corrected `results/e4_loco_summary.csv`
- Preserved `results/e4_loco_pooled_summary.csv`

No validation or locked test data is used in either follow-up.


## Phase 6.5 follow-up — E2 one-standard-error selection and E4 correction — 17 November 2026

### E2 — One-standard-error selection across all 20 configurations

The original E2 experiment compared 20 configurations:
- Five feature sets: S0–S4.
- Colour-only and colour-plus-device/lamp variants.
- Ridge and RBF-SVR.

The follow-up used 2,000 tube-bootstrap resamples, seed 0, resampling physical tubes with replacement. Identical bootstrap draws were used for all configurations.

Data:
- 666 training images.
- 31 training tubes.
- Validation and test data excluded.

The lowest raw MAE was achieved by SVR using S1 plus device/lamp:
- OOF MAE: 0.191504.
- Bootstrap SE: 0.019929.
- 95% bootstrap interval: [0.154639, 0.231872].
- One-standard-error cutoff: 0.211432.

Five configurations were within the cutoff:

| Configuration | Features | OOF MAE | Bootstrap SE |
|---|---:|---:|---:|
| SVR S1 + device/lamp | 7 | 0.191504 | 0.019929 |
| SVR S2 + device/lamp | 13 | 0.209579 | 0.019877 |
| SVR S3 + device/lamp | 26 | 0.210156 | 0.022572 |
| SVR S4 colour-only | 53 | 0.204667 | 0.024475 |
| SVR S4 + device/lamp | 57 | 0.197579 | 0.023340 |

No Ridge configuration was eligible.

Applying the predeclared simplicity ordering selected:

**SVR S1 + device/lamp (`SVR__S1_with_device_and_lamp`).**

Its inputs are:
- Mean L*.
- Mean a*.
- Mean b*.
- `device_dino`.
- `device_phone`.
- `lamp_off`.
- `lamp_on`.

This configuration has only seven inputs and achieved the best observed OOF MAE among the 20 configurations.

### Candidate change

The Phase 6.4 one-standard-error selection of `SVR_RBF_S2` colour-only is retained as a historical result.

It is superseded as the current development candidate by `SVR__S1_with_device_and_lamp`, following the E2 comparison and its one-standard-error analysis.

The new candidate is still provisional. Its results are from training-tube cross-validation, not validation or test evaluation.

### E4 — Per-concentration R² correction

Within each leave-one-concentration-out evaluation group, all actual target values are identical. Consequently, per-concentration R² is undefined.

The previous extreme negative per-concentration R² values were numerical artefacts and are not interpreted.

The corrected `results/e4_loco_summary.csv` reports R² as missing/undefined for each held-out concentration while retaining MAE, RMSE, bias, fraction within ±0.1 and image/tube counts.

The pooled E4 summary remains unchanged:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| SVR S2 | 0.213590 | 0.291451 | 0.771903 |
| Random Forest S2 | 0.222554 | 0.306796 | 0.747253 |
| Gradient Boosting S2 | 0.227850 | 0.309712 | 0.742425 |

Pooled R² remains meaningful because pooled predictions cover multiple concentration levels.

No models were retrained for this reporting correction.

### Outputs

E2 selection:
- `results/e2_tube_bootstrap_summary.csv`
- `results/e2_tube_bootstrap_samples.csv`
- `results/e2_one_se_selection.csv`
- `reports/e2_one_se_selection.md`
- `reports/e2_candidate.json`

E4:
- `results/e4_loco_summary.csv` — corrected.
- `results/e4_loco_pooled_summary.csv` — preserved.

### Current development candidate

**RBF-SVR using S1 colour means plus device/lamp one-hot inputs.**

This candidate supersedes the Phase 6.4 S2 colour-only SVR selection. The validation partition remains reserved, and the test partition remains locked.


## Phase 6.6 — Independent validation and final model selection — 17 November 2026

### Evaluation strategy

The previous Phase 6.1–6.5 experiments used grouped cross-validation within the 666-image training partition. These results are development estimates, not independent validation performance.

The next stage follows a separate train–validation–test procedure.

### Step 1 — Candidate assessment on validation

Training partition:
- 666 images
- 31 physical tubes

Validation partition:
- 198 images
- 10 physical tubes

Evaluate a predeclared shortlist:
1. SVR S1 with device/lamp inputs.
2. Random Forest S2, colour-only.
3. Gradient Boosting S2, colour-only.
4. SVR S2, colour-only.

For each candidate:
- Select hyperparameters using tube-grouped cross-validation within the training partition only.
- Fit the complete preprocessing and model pipeline on all 666 training images.
- Predict the 198 validation images.
- Do not fit imputers, scalers or model parameters using validation data.

Compare validation performance using MAE as the primary metric, with RMSE, R², bias and fraction within ±0.1 as additional metrics.

Report image-pooled errors and per-tube errors so that differences in photograph counts do not conceal poor performance on individual physical tubes.

The validation set may be used to select the final candidate, but it must not be repeatedly used for unrestricted hyperparameter tuning.

### Step 2 — Final candidate selection

Choose the model using the validation results and the predeclared primary metric, MAE.

Record:
- Selected model family.
- Selected feature set.
- Device/lamp input choice.
- Hyperparameters.
- Validation metrics.
- Validation errors by physical tube.
- Known limitations.

Once selected, freeze the model configuration and feature definitions.

### Step 3 — Final refitting

Combine training and validation:
- 864 development images.
- 41 physical tubes.

Refit the complete selected pipeline using both partitions.

Any final hyperparameter selection must use tube-grouped CV within these 41 development tubes. Do not use the test partition for tuning.

The fitted preprocessing transformations and model are then frozen.

### Step 4 — Locked test evaluation

Test partition:
- 200 images.
- 10 physical tubes.

Apply the frozen preprocessing and final fitted model to the test images.

Report final test MAE, RMSE, R², bias, fraction within ±0.1, concentration-band errors, and per-tube errors.

Test predictions and metrics are reported once. Do not change features, preprocessing, hyperparameters or model selection in response to test performance.

### Separation of evaluation roles

- Training: cross-validation, model comparison and hyperparameter selection.
- Validation: independent candidate assessment and final candidate selection.
- Training + validation: final model refitting.
- Test: final untouched evaluation.

The Phase 6.4 and Phase 6.5 cross-validation results remain development results. They are not presented as validation or test performance.


### Feature scaling policy — Phase 6

Feature scaling is model-dependent:

- B0: no preprocessing.
- B1: median imputation; no scaling.
- Ridge: median imputation followed by training-fold standardization. The scratch Ridge implementation performs standardization internally.
- Polynomial Ridge: median imputation, degree-2 polynomial expansion, then standardization.
- RBF-SVR: median imputation followed by standardization of all input columns, including device/lamp one-hot variables.
- Random Forest and Gradient Boosting: median imputation; no standardization.
- MLP: median imputation followed by standardization.

All imputation and scaling parameters are fitted exclusively on the corresponding training fold or training partition. No global preprocessing is performed before splitting.

The same policy must be followed during independent validation and final model fitting.

## E6–E10 — Additional diagnostics and feasibility decisions — 17 Nov 2026

### E6 — Session and concentration-range challenge

Run an exploratory diagnostic using Phase 4 whole-frame mean b* values.

- Training: 11 Sep 2024, 0.20–1.00% nominal range.
- Challenge: 13 Sep 2024, 1.10–2.00% nominal range.
- Use only rows marked `role=train`; exclude validation and official test rows.
- Aggregate photographs by physical tube and acquisition condition.
- Fit a separate one-feature linear calibration for each acquisition condition.

The available training-session concentrations actually range from 0.20% to
0.90%. After excluding validation tubes, the challenge set contains 1.10%,
1.30%, 1.80%, 1.90% and 2.00%.

Observed challenge MAE:
- Dino-Lite off: 0.7095
- Dino-Lite on: 0.8087
- Phone off: 0.9279
- Phone on: 0.8799

All four conditions under-predicted the challenge concentrations. None of
the challenge predictions were within ±0.1 percentage points.

Interpretation: this is a combined concentration-extrapolation and
session-shift challenge. Session and concentration range are confounded,
so their individual effects cannot be identified. These are exploratory
Phase 4 feature results, not final Phase 6 model metrics.

### E7 — Variance and residual diagnostics

The Phase 4 descriptive variance decomposition is retained:

- Between concentrations: approximately 34–35%.
- Between tubes within concentration: approximately 64–65%.
- Between photographs within a tube: approximately 0.6–1.3%.

The tube component is confounded with preparation instance because there
is one tube per concentration per instance.

The additional residual-clustering analysis must use training-only
out-of-fold predictions from the selected Phase 6 candidate,
SVR__S1_with_device_and_lamp. Group residuals by tube_id and compare
between-tube residual variation with within-tube residual variation.
Do not use validation or test predictions for this analysis.

### E8 — LOD/LOQ

Do not report an LOD or LOQ estimate from the current dataset. There is
only one independent 0.00% added-urea tube and one 0.15% tube. Repeated
photographs of the same tube are not independent blank replicates and
cannot provide a reliable estimate of blank/preparation variability.

### E9 — Multi-view prediction

True four-view droplet prediction is not supported because droplet_id
and view_id are unavailable. Do not infer droplet/view pairings from
file order or timestamps.

A possible substitute is tube-level multi-image aggregation: average
out-of-fold predictions within each tube and acquisition condition,
then compare tube-level error with single-image error. This must be
reported as tube-level aggregation, not as four views of one droplet.

### E10 — Prediction intervals

Defer split-conformal prediction intervals. The ten validation tubes are
reserved for candidate selection and cannot also serve as an independent
calibration set. The locked test tubes must not be used for calibration.
A separate set of calibration tubes would be required for a defensible
split-conformal analysis.

## Phase 6 — Paired bootstrap model comparison skipped — 17 Nov 2026

**Decision:** Skip the additional pairwise paired-bootstrap difference analysis described in the roadmap.

**Reason:** The project has a limited number of independent physical tubes, and the additional comparison is not necessary to continue the evaluation. The existing tube-level bootstrap and one-standard-error model-selection results are retained.

**Interpretation restriction:**
- Do not claim that two models are statistically equivalent.
- Do not claim that one model is significantly better based on overlapping individual confidence intervals.
- Report the existing model metrics and selection rule as previously calculated.
- Do not rerun the completed Phase 6 experiments for this comparison.

**Next step:** Proceed to independent validation using the reserved 10 validation tubes (198 images). The 10 test tubes remain locked and must not be used until final Phase 7 evaluation.

## Phase 6.6 — Independent validation — 17 Nov 2026

**Objective:** Compare the shortlisted development candidates on the
reserved validation partition and select the final model.

**Data separation:**
- Training: 666 images from 31 tubes.
- Validation: 198 images from 10 tubes.
- Test: 200 images from 10 tubes; remain locked and unopened.

**Candidate models:**
1. SVR with S1 colour features and device/lamp indicators.
2. Random Forest with S2 colour features.
3. Gradient Boosting with S2 colour features.
4. SVR with S2 colour features.

**Protocol:**
- Use only training tubes for model fitting and hyperparameter selection.
- Keep all learned preprocessing inside model pipelines.
- Fit each shortlisted candidate using the complete training partition.
- Evaluate each candidate once on the independent validation partition.
- Compare candidates using MAE as the primary metric, with RMSE,
  R², bias and fraction within ±0.1 as supporting metrics.
- Select the final candidate based on validation performance.
- Do not use validation results to modify the locked test partition.

**After selection:**
Refit the selected pipeline on all 864 development images
(training plus validation), keeping the selected hyperparameters fixed.
The test partition remains untouched until Phase 7.

The additional paired-bootstrap comparison proposed in the roadmap
is skipped. No claims of statistical equivalence or significance
will be made from individual confidence-interval overlap.

## Phase 6.6 — Independent validation protocol — 17 Nov 2026

**Decision:** Proceed directly to independent validation. Skip the
additional pairwise paired-bootstrap comparison.

**Candidates:**
1. SVR — S1 with device and lamp indicators.
2. Random Forest — S2 colour-only.
3. Gradient Boosting — S2 colour-only.
4. SVR — S2 colour-only.

**Protocol:**
- Use the 31 training tubes (666 images) for all fitting and
  hyperparameter selection.
- Use four-fold GroupKFold, grouped by tube_id, for training-only
  hyperparameter selection.
- Fit each selected candidate on all training images.
- Evaluate once on the 10 validation tubes (198 images).
- Select using validation MAE; report RMSE, R², bias and fraction
  within ±0.1 as supporting metrics.
- Do not use validation results to tune hyperparameters.
- Do not open or use the 10 locked test tubes.

After selection, refit the chosen pipeline on all 864 development
images using its selected hyperparameters. Preserve the test lock
until Phase 7.

## Phase 6.6 — Independent validation — 17 Nov 2026

**Decision:** Proceed to independent validation using the four shortlisted candidates. The additional pairwise paired-bootstrap comparison is skipped.

**Candidates:**
1. SVR — S1 with device and lamp indicators.
2. Random Forest — S2 colour-only.
3. Gradient Boosting — S2 colour-only.
4. SVR — S2 colour-only.

**Protocol:**
- Training: 666 images from 31 tubes.
- Validation: 198 images from 10 tubes.
- Test: 200 images from 10 tubes; remain locked.
- Select hyperparameters using four-fold GroupKFold on training tubes only.
- Fit each candidate on all training images.
- Evaluate each candidate once on the independent validation images.
- Use validation MAE as the primary selection metric. Also report RMSE, R², bias and fraction within ±0.1.
- Do not use validation results to tune hyperparameters.

**After selection:**
Refit the selected pipeline on all 864 development images (training plus validation), keeping the selected hyperparameters fixed. Preserve the test lock until Phase 7.

No claims of statistical significance or equivalence will be made from this validation comparison.

## Phase 6.6 — Independent validation results — 17 Nov 2026

**Validation completed successfully.**

The four shortlisted candidates were evaluated on the independent
validation partition: 198 images from 10 physical tubes.

Hyperparameters were selected using four-fold GroupKFold on the
666 training images from 31 training tubes. Each selected estimator
was then fitted on all training images and evaluated on validation.

### Observed results

| Candidate | Validation MAE | RMSE | R² |
|---|---:|---:|---:|
| SVR S1 + device/lamp | 0.198015 | 0.240571 | 0.801961 |
| SVR S2 | 0.214287 | 0.264545 | 0.760523 |
| Random Forest S2 | 0.232805 | 0.283647 | 0.724689 |
| Gradient Boosting S2 | 0.233652 | 0.288114 | 0.715950 |

### Final candidate selection

Select **SVR with S1 colour features and device/lamp indicators**.

Training-only cross-validation selected:
- C = 10
- epsilon = 0.02
- gamma = 0.1

The selected candidate achieved validation MAE = 0.198015,
RMSE = 0.240571 and R² = 0.801961.

The validation set was used for candidate selection, not
hyperparameter tuning.

### Evaluation integrity

- Training: 666 images, 31 tubes.
- Validation: 198 images, 10 tubes.
- Test: 200 images, 10 tubes; not accessed.
- The additional paired-bootstrap comparison remains skipped.

The selected candidate is now frozen for final development refitting.
Its hyperparameters and feature definition must not be changed
using test results.

### Next step

Refit the selected SVR pipeline on all 864 development images
(training plus validation), using the fixed hyperparameters.
Save the fitted model and configuration. Do not evaluate on the
test partition until Phase 7.

## Phase 6.7 — Final development refit — 17 Nov 2026

**Decision:** Freeze the Phase 6.6 selected candidate and refit it on
the complete development dataset.

**Selected model:** RBF-SVR with S1 colour features and device/lamp
indicators.

**Fixed hyperparameters:**
- C = 10
- epsilon = 0.02
- gamma = 0.1

**Input features:**
- L_mean
- a_mean
- b_mean
- device_dino
- device_phone
- lamp_off
- lamp_on

**Pipeline:**
1. Median imputation.
2. StandardScaler.
3. RBF-SVR.

**Fit data:**
- Training: 666 images, 31 tubes.
- Validation: 198 images, 10 tubes.
- Combined development data: 864 images, 41 tubes.

The hyperparameters are fixed from training-only grouped CV.
The validation set selected the candidate; it is now included in
the final development refit.

Do not report in-sample predictions or fit metrics as generalization
performance. The independent validation metrics remain the model
selection results.

Save the fitted pipeline and a reproducibility configuration.
Do not load, preprocess, predict on, or otherwise access the
200-image test partition. Final test evaluation remains Phase 7.

### Observed completion — 17 Nov 2026

The final development refit completed successfully.

- Development images used: 864.
- Physical tubes used: 41.
- Training images: 666.
- Validation images: 198.
- Test images accessed: 0.

The final fitted pipeline was saved to:

`models/final_svr_s1_device_lamp.joblib`

The configuration was saved to:

`reports/final_model_config.json`

The data audit and fit manifest were saved to:

- `results/phase6_7_refit_data_audit.json`
- `results/phase6_7_refit_manifest.csv`

Model SHA-256:

`5afb9b97a4bd73eb7fa569ffc2483ddc37646d86fb162277bc669ec4cf22adaa`

No in-sample performance metrics or test predictions were generated.
Phase 6.7 is complete.

## Phase 7 — One-shot final test — 17 Nov 2026

### Frozen model and configuration

The final model was selected using training-only grouped cross-validation
and independent validation.

- Model: RBF-SVR.
- Features: S1 colour features plus device/lamp indicators.
- Inputs: L_mean, a_mean, b_mean, device_dino, device_phone,
  lamp_off, lamp_on.
- Imputation: median.
- Scaling: StandardScaler.
- Hyperparameters: C=10, epsilon=0.02, gamma=0.1.
- Fitted artifact: models/final_svr_s1_device_lamp.joblib.
- Configuration: reports/final_model_config.json.

The fitted pipeline was trained on all 864 development images
from 41 tubes. Its SHA-256 is:
5afb9b97a4bd73eb7fa569ffc2483ddc37646d86fb162277bc669ec4cf22adaa

### Test evaluation protocol

The locked test partition contains 200 images from 10 physical tubes.
It will be accessed for final evaluation only.

- Apply the frozen Phase 5 ROI erosion and glare-removal pipeline.
- Extract the exact frozen S1 features and device/lamp indicators.
- Do not tune preprocessing, features, model or hyperparameters using
  test images or test results.
- Generate predictions from the frozen final model once.
- Fit and evaluate B0 and B1 as the predeclared baselines, using
  development data only for fitting.
- B2 remains omitted because its previous implementation is unavailable.

Report for each evaluated model:
- MAE
- RMSE
- R²
- Bias (prediction minus actual)
- Fraction within ±0.1 percentage points
- MAE by low, medium and high concentration bands
- 95% tube-bootstrap interval for MAE

Use 2,000 tube-level bootstrap resamples with seed 0. Resample
physical tubes, not individual photographs.

After test predictions are generated, do not modify the model or
repeat the evaluation. Any subsequently discovered bug must be
documented; corrected results must be identified as post-test
development results, not a fresh independent test.

The test partition must not be used for model selection or tuning.

## Phase 7 — one-shot final test evaluation

The final model is frozen as SVR_S1_device_lamp, fitted on all 864 development images from 41 tubes. Its hyperparameters are C=10, epsilon=0.02, gamma=0.1, RBF kernel. The selected features are L_mean, a_mean, b_mean, device_dino, device_phone, lamp_off and lamp_on.

The locked test partition contains 200 images from 10 physical tubes. It will be opened once for final evaluation. No model, feature, segmentation, erosion, glare-removal or hyperparameter decisions will be changed based on test results.

Test preprocessing will use the existing central-droplet Lab-distance segmentation, the frozen 3.5% circular erosion, the frozen V>0.92 and S<0.25 glare thresholds, the existing CIELAB feature extraction and the existing device/lamp encoding.

The final SVR and the B0/B1 baselines will be evaluated on the same test images. B0 and B1 will be fitted using development data only. Report MAE, RMSE, R², bias, fraction within ±0.1 percentage points, concentration-band MAE, and calibration slope/intercept. Estimate uncertainty using 2,000 physical-tube bootstrap resamples with seed 0.

Test images will not be used to revise the pipeline or select a model. Test-specific outputs will be stored separately from development artifacts.


## Phase 7 — Final test evaluation results — 17 Nov 2026

### Execution

The one-shot final test evaluation completed successfully.

- Test images: 200
- Physical test tubes: 10
- Feature extraction succeeded for all 200 images.
- No test image failed segmentation or feature extraction.
- The frozen Phase 6.7 model was used without modification.
- B0 and B1 were fitted using development data only.
- The test partition was accessed once.

### Final model

Selected model: SVR_S1_device_lamp

- Kernel: RBF
- C = 10
- epsilon = 0.02
- gamma = 0.1
- Features: L_mean, a_mean, b_mean, device_dino,
  device_phone, lamp_off, lamp_on

### Final test results

| Model | MAE | RMSE | R² | Bias | Within ±0.1 |
|---|---:|---:|---:|---:|---:|
| SVR_S1_device_lamp | 0.146228 | 0.188272 | 0.855366 | -0.028406 | 0.485 |
| B0_mean | 0.431625 | 0.514860 | -0.081626 | -0.141437 | 0.240 |
| B1_b_med_linear | 0.316045 | 0.390000 | 0.379375 | -0.042953 | 0.230 |

All errors are in percentage points of added urea concentration (% w/v).

### Concentration-band MAE

| Model | Low (<0.5%) | Medium (0.5–1.0%) | High (>1.0%) |
|---|---:|---:|---:|
| SVR_S1_device_lamp | 0.064513 | 0.184126 | 0.137038 |
| B0_mean | 0.676562 | 0.175576 | 0.567438 |
| B1_b_med_linear | 0.313178 | 0.263171 | 0.356918 |

### Tube-bootstrap uncertainty

2,000 paired bootstrap resamples were performed by physical tube, using seed 0.

| Model | 95% MAE interval |
|---|---:|
| SVR_S1_device_lamp | 0.106800–0.184947 |
| B0_mean | 0.252356–0.612118 |
| B1_b_med_linear | 0.266166–0.374598 |

The bootstrap resampling unit was the physical tube, not the individual photograph. The test set contains only ten tubes, so the uncertainty interval should be interpreted cautiously.

### Calibration

Calibration uses actual concentration regressed on predicted concentration.

- SVR slope: 0.901654
- SVR intercept: 0.135563

B0 calibration slope and intercept are undefined because B0 generates a constant prediction.

### Decision and conclusion

The frozen SVR_S1_device_lamp model outperformed both baselines on the independent test partition, achieving MAE 0.146228 % w/v and R² 0.855366.

All test images were processed successfully. The final test evaluation is complete. No further model selection, feature engineering, preprocessing adjustment or hyperparameter tuning will be performed using this test set.

### Artifacts

- results/test_predictions.csv
- results/test_processing_manifest.csv
- results/test_results.csv
- results/test_tube_bootstrap_samples.csv
- results/phase7_test_accessed.json
- reports/phase7_final_test.md
