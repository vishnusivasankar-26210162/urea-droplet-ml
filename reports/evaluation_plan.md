# Evaluation Plan — Urea Droplet ML

**Date:** 16 November 2026  
**Status:** Protocol to be locked with `splits/tube_roles.csv`

## 1. Evaluation objective

Estimate how accurately droplet-image features predict added urea concentration (% w/v) for a previously unseen physical tube of milk.

The independent unit is the physical tube, not the photograph. All images belonging to one tube must remain in the same split, including images from different devices, lamp settings and droplets.

## 2. Dataset and grouping

The metadata contains 1,064 photographs from 51 physical tubes and three preparation instances.

- Instance 1: 24 tubes
- Instance 2: 19 tubes
- Instance 3: 8 tubes

One physical tube exists for each concentration within each instance. `tube_id` is therefore determined by `(instance_id, concentration)`.

Individual `droplet_id` and `view_id` values are unavailable and will remain blank. This does not change the tube-level split.

## 3. Locked test set

Select one tube per concentration from concentration levels represented by at least two physical tubes.

Use random seed 42 and sorted candidate tube IDs for deterministic selection.

There are 20 eligible concentration levels. The resulting split contains:

- Test: 20 tubes
- Development: 31 tubes

Seven concentration levels have only one physical tube: 0.00, 0.15, 0.35, 0.45, 0.75, 1.25 and 1.75 % w/v. These remain in development because no independent tube is available for a held-out test sample at those levels.

The assignments are stored in `splits/tube_roles.csv`. The `role` column is also attached to `data/processed/metadata.csv`.

The test set must not be used for exploratory plots, feature selection, preprocessing decisions, hyperparameter tuning or model selection. Test images are not to be inspected again until Phase 7.

## 4. Model development and validation

Use development data only.

**Primary validation:** five-fold `GroupKFold`, grouping by `tube_id`.

**Hyperparameter tuning:** four-fold `GroupKFold` inside each outer training fold. All learned preprocessing, scaling and model fitting must be performed using training data only.

**Unseen-concentration experiment:** leave one concentration level out at a time from development data. Hold out every development tube at that concentration and train on all other development concentrations.

The tube-grouped experiment measures performance on a new tube at a concentration represented in training. Leave-one-concentration-out measures performance at a concentration absent from training.

The concentration-level fallback split in the original roadmap is not used because the physical tube identities are supported by the experiment records.

## 5. Metrics

Primary metric:

- Mean absolute error (MAE), in percentage points of added urea (% w/v).

Also report:

- RMSE
- R², interpreted alongside the baseline
- Mean signed error (bias)
- Fraction of predictions within ±0.1 % w/v
- MAE for low (<0.5%), medium (0.5–1.0%) and high (>1.0%) concentration bands
- Calibration slope and intercept from regressing true concentration on predicted concentration

Avoid percentage error because it is undefined at zero and unstable near zero.

A detection analysis at thresholds 0.2%, 0.5% and 1.0% is a secondary analysis, reporting ROC AUC, sensitivity and specificity.

## 6. Baselines

- **B0:** Predict the training-fold mean concentration.
- **B1:** Fit a straight-line calibration using median droplet b* colour as the single input feature.
- **B2:** Reproduce the old unmasked-channel-mean pipeline with a random image-level split once, as a leakage demonstration only. Do not present this as a valid estimate of unseen-tube performance.

Every primary model must be compared with B0 and B1 under the same tube-grouped folds.

## 7. Uncertainty

Use a cluster bootstrap by physical tube, with 2,000 resamples, for confidence intervals of MAE and other reported metrics. Resample tubes, not individual photographs.

Use paired tube-level bootstrap differences when comparing models. Report the 2.5th and 97.5th percentiles.
## 3.4 Unseen-concentration experiment

A separate leave-one-concentration-out (LOCO) experiment is performed
using development data only.

Development data consists of the 31 training tubes and 10 validation
tubes from the primary 60/20/20 partition. The 10 locked test tubes
are excluded from every LOCO fold.

There are 27 concentration levels in development, so 27 LOCO folds
are generated.

For each fold:

- Every development tube at one concentration is held out for validation.
- All other development tubes are used for training.
- No test tube is included.
- All photographs belonging to a tube follow that tube's fold assignment.

The fold definitions are saved in splits/loco_folds.csv.

Tube-grouped evaluation asks whether the model generalises to a new
physical tube at a concentration represented during training.

LOCO asks whether the model generalises to a concentration that was
absent from training.

LOCO is a separate generalisation experiment. It does not replace
the primary grouped evaluation or the locked test evaluation.

## 8. Constraints and limitations

There are only 51 physical tubes and three preparation instances. The random tube-level split evaluates unseen physical tubes, but it does not guarantee an entirely unseen preparation instance. This limitation must be stated.

Some concentration levels have only one tube and cannot contribute to the locked test set. Individual droplet identities are not reconstructed, so repeated photographs of the same droplet cannot be separately grouped or averaged.

All conclusions are limited to the represented milk, reagent, slide, devices and laboratory acquisition conditions. This is not a certified food-safety test.

## 9. Reproducibility and change control

The split is generated by `src/make_splits.py` with seed 42 and saved to `splits/tube_roles.csv`. Once created, it must not be redrawn. Any change requires a dated explanation in `reports/decision_log.md`.

Before feature extraction, the split unit, metrics and baselines must be frozen and committed to Git.

## 3.6 Metrics

The primary regression metric is MAE, reported in percentage points
of added urea (% w/v).

Additional regression metrics:

- RMSE, to penalize large errors.
- R², interpreted alongside the training-mean baseline.
- Bias, defined as mean(predicted - actual); positive values indicate
  overprediction.
- Fraction of predictions with absolute error <= 0.1 percentage points.
- MAE for three actual-concentration bands:
  - Low: below 0.5%.
  - Medium: 0.5% through 1.0%, inclusive.
  - High: above 1.0%.

Percentage error will not be used because it is undefined at 0%
and unstable near zero.

Calibration will be checked by fitting:

    actual concentration = slope * predicted concentration + intercept

Ideal calibration has slope 1 and intercept 0.

Metrics will be computed on the combined out-of-fold predictions
for development evaluation and on the complete locked test predictions
in Phase 7. Metrics will not be averaged naively across individual
LOCO folds. In particular, R² and calibration are undefined or
uninformative within a LOCO validation fold containing only one
actual concentration.

Secondary detection analysis (should):

For thresholds T = 0.2%, 0.5% and 1.0%, define:

    Actual positive = actual concentration >= T
    Predicted positive = predicted concentration >= T

Report ROC AUC using continuous predictions as scores, and report
sensitivity and specificity using T as the prediction threshold.

If a subset contains only one actual class, ROC AUC is undefined
and will be reported as unavailable rather than forced to a value.

Implementation is in src/metrics.py, with unit tests in
tests/test_metrics.py.

## 3.7 Baselines

Two baselines will be implemented. The old pipeline replication (B2)
is omitted because the previous implementation is unavailable.

### B0 — Training mean

Predict the mean concentration of the training data for every
evaluation sample.

This is the minimum reference model. Every proposed model should
be compared against B0 under the same data split and evaluation
procedure.

### B1 — Median b* linear calibration

Fit a straight-line linear regression using only median b* as the
input feature:

    predicted concentration = intercept + slope * median_b_star

This represents a simple, interpretable chemical calibration model.

If median b* is missing, imputation is fitted using training data
only, within the model pipeline.

### Evaluation protocol

Both baselines are fitted independently inside each training fold.

- During grouped cross-validation, fit on each outer training fold
  and predict its corresponding outer validation fold.
- During the separate validation assessment, fit using the training
  partition and evaluate on the validation partition.
- The locked test partition remains untouched until Phase 7.

Compare both baselines using the metrics defined in Section 3.6.
B0 is the minimum reference that proposed models should improve upon.

B2 is intentionally omitted; no old pipeline is available for
reproduction.

Implementation: `src/baselines.py`
Tests: `tests/test_baselines.py`

## 3.8 Uncertainty estimation

Uncertainty intervals will be estimated using a cluster bootstrap
by physical tube.

For each bootstrap replicate:

1. Resample physical tubes with replacement.
2. Include all evaluation observations belonging to each selected tube.
3. Recompute the evaluation metric on the resampled observations.

Use 2,000 bootstrap replicates for the final analysis.

Report the 2.5th and 97.5th percentiles as the 95% bootstrap
interval.

For model comparisons, use paired bootstrap differences. Each
replicate must use the same resampled tubes for both models.

Report the 95% percentile interval of the paired metric differences.
For MAE, define the difference as:

    MAE(model A) - MAE(model B)

A negative difference means model A has lower error.

Bootstrap implementation is deferred to Phase 6. It must resample
physical tubes rather than individual photographs.

## 3.9 Evaluation-plan change control

This evaluation plan is the reference protocol for model development
and evaluation.

The plan must be committed to Git before Phase 5 begins.

Every subsequent change to the evaluation plan must be documented
in reports/decision_log.md, including:

- Date of change.
- The original decision.
- The revised decision.
- The reason for the change.
- Any consequences for evaluation or interpretation.

The locked tube assignments in splits/tube_roles.csv must not be
changed without a documented reason.

The test partition must remain untouched until Phase 7.