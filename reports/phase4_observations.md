# Phase 4 observations — complete after reviewing figures

Development images analysed: 864 (train + validation).
Test images excluded.
Sessions found: 2024-09-11, 2024-09-13, 2024-11-04, 2024-11-14, unknown.

Figures: `results/figures/phase4/`.
Tables: `results/phase4/`.

## Questions and one-line answers

1. **Is L* bimodal?**
   Review `06_Lstar_distribution_by_condition.png`.
   Answer: [one line; check alignment with device/lamp/session].

2. **Is there a session offset?**
   Review `phase4_session_contrasts.csv` for 1.00% (11 Sep) vs
   1.10% (13 Sep), and 0.20% (11 Sep) vs 0.15%/0.00% (13 Sep).
   Answer: [one line; note session/concentration confounding].

3. **Is there drift?**
   Review `04_bstar_vs_timestamp_by_session.png` and
   `phase4_drift_summary.csv`.
   Answer: [one line; slopes are descriptive because concentration/order changes with time].

4. **Does the response saturate?**
   Review `07_bstar_response_and_saturation.png` and
   `phase4_bstar_response_by_concentration.csv`.
   Answer: [one line].

5. **How large is within-concentration spread relative to the full response?**
   Review `phase4_within_vs_across_spread.csv` and
   `08_bstar_within_vs_across_spread.png`.
   Answer: [one line with ratios].

6. **Are any images unusable?**
   Complete `image_quality_review.csv` using `inclusion_rules.md`.
   Answer: [count, image IDs, reasons; or none after reviewing all dev rows].

7. **(Should) Variance decomposition:**
   Review `phase4_variance_components_descriptive.csv` and
   `phase4_device_lamp_paired_differences.csv`.
   Answer: [one line; these are descriptive, not a balanced mixed-effects model].

## Alpha-channel result

No source image has an alpha channel; alpha fraction is zero for opaque originals. This does not measure droplet segmentation.

## Masking limitation

All L*, a* and b* values are means over the entire original frame,
including background. They are pre-mask diagnostics, not droplet-only
features. Whether L* modes collapse after masking can only be answered
after Phase 5 masks are created and validated.
