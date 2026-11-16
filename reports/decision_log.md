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


