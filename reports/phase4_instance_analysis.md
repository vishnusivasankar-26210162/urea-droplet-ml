Phase 4 analysis — instance-based exploration
Analysis source: uploaded phase4_image_eda.csv, phase4_bstar_response_by_concentration.csv, phase4_instance_contrasts_by_concentration.csv, phase4_instance_contrast_summary.csv, phase4_device_lamp_paired_differences.csv, phase4_device_lamp_paired_summary.csv, phase4_within_vs_across_spread.csv, phase4_variance_components_descriptive.csv, and image_quality_review.csv.

Data scope: 864 development images (train + validation), from 41 development tubes. Locked test images are not included in these uploaded EDA data. No test images were inspected here.
Executive findings
1. Whole-frame Lab measurements differ substantially by preparation instance, particularly b*. Instance 1 b* values are far higher than Instance 2 at matched concentrations in all four acquisition conditions. This indicates a strong batch/acquisition-scene effect in the unmasked images.
2. Device and lamp effects are not stable across instances. Lamp-on shifts b* upward much more in Instance 1 than in Instance 2. The phone-versus-Dino-Lite difference even changes sign between Instances 1 and 2.
3. The apparent concentration response differs by instance. Instance 1 shows a rise at lower concentrations followed by a much flatter response above roughly 0.5%; Instance 2 is nearly flat over most of the range. Instance 3 shows a moderate increase but has only six development tubes per Dino-Lite condition.
4. The average within-concentration tube SD is approximately 50–60% of the full b* response span. Under this experimental design, these between-tube differences at a fixed concentration are also between-instance differences; pure tube variation cannot be isolated from instance effects.
5. The descriptive variance decomposition attributes about 34–35% of whole-frame b* sum of squares to concentration, 64–65% to tubes within concentration, and 0.6–1.3% to photo-to-photo variation within tube. Since one tube exists per concentration per instance, the large tube-within-concentration component is confounded with preparation instance.
6. All source images are opaque: no alpha channel is present and the transparent-pixel fraction is zero. This metric does not measure droplet occupancy.
7. The image-quality review is not complete: all 864 development rows have blank manual review fields. No image can yet be declared visually usable or unusable based on this CSV.
1. Dataset scope and counts
Instance	Development images	Development tubes
1	480	20
2	360	15
3	24	6
Total	864	41


Per instance and acquisition condition:
Instance	Device	Lamp	Images	Tubes
1	Dino-Lite	Off	116	20
1	Dino-Lite	On	119	20
1	Samsung A31	Off	122	20
1	Samsung A31	On	123	20
2	Dino-Lite	Off	90	15
2	Dino-Lite	On	90	15
2	Samsung A31	Off	90	15
2	Samsung A31	On	90	15
3	Dino-Lite	Off	12	6
3	Dino-Lite	On	12	6


Instance 3 has no phone photographs. Its phone condition comparisons are therefore unavailable.
2. L* distribution / bimodality
Per-image whole-frame mean L* averages by instance and condition:
Device / lamp	Instance 1	Instance 2	Instance 3
Dino-Lite off	82.11 ± 3.15	76.41 ± 1.81	75.68 ± 3.42
Dino-Lite on	81.57 ± 2.10	75.53 ± 1.97	76.26 ± 2.40
Phone off	52.56 ± 3.45	50.19 ± 2.34	Not available
Phone on	55.59 ± 3.35	55.28 ± 2.08	Not available


Interpretation: the Dino-Lite whole-frame L* distributions show a clear shift between Instance 1 and Instances 2/3; pooling instances can therefore create a multi-group or apparently bimodal distribution. Phone-on distributions overlap substantially; phone-off has some higher-L* observations but does not show the same clean instance separation. This is not evidence of two droplet populations. It is more consistent with differences associated with acquisition/preparation instance in the full frame.

The uploaded data are already CIELAB L* values on the standard 0–100 scale. The earlier values near 140 and 210 refer to a different intensity representation, not standard CIELAB L*.

Important limitation: this analysis uses whole-frame means including background. Whether apparent L* modes collapse after droplet masking cannot be answered until Phase 5 ROI masks are generated.
3. Preparation-instance effect in b*
Mean paired-concentration differences in tube-level mean b*, defined as Instance A minus Instance B:
Condition	Comparison	Matched concentration levels	Mean difference in b*
Dino-Lite off	I1 − I2	11	+19.25
Dino-Lite on	I1 − I2	11	+23.29
Phone off	I1 − I2	11	+26.32
Phone on	I1 − I2	11	+32.23
Dino-Lite off	I1 − I3	2	+12.50
Dino-Lite on	I1 − I3	2	+17.53
Dino-Lite off	I2 − I3	1	−6.68
Dino-Lite on	I2 − I3	1	−7.47


The I1–I2 differences are large and consistently positive. The I1–I3 and I2–I3 comparisons have very few matched concentrations, so they are weak evidence and should not be generalized.

Interpretation: preparation instance is a major covariate in the unmasked whole-frame colour measurements. It should be retained in all diagnostic plots and discussed as a limitation. Phase 5 should determine whether this difference remains after ROI masking.
4. Device and lamp effects
Paired differences use the same physical tube; values are mean b* differences.
Comparison	Instance	Paired tubes	Mean difference	SD
Lamp on − off, Dino-Lite	1	20	+4.75	1.18
Lamp on − off, Dino-Lite	2	15	+0.40	0.23
Lamp on − off, Dino-Lite	3	6	+0.85	0.31
Lamp on − off, phone	1	20	+6.47	1.61
Lamp on − off, phone	2	15	+0.57	0.39
Phone − Dino-Lite, lamp off	1	20	+2.05	2.29
Phone − Dino-Lite, lamp off	2	15	−4.84	0.46
Phone − Dino-Lite, lamp on	1	20	+3.78	1.72
Phone − Dino-Lite, lamp on	2	15	−4.67	0.61


Interpretation: device and lamp shifts are instance-dependent. The lamp effect is much larger in Instance 1 than Instance 2. The phone-minus-Dino-Lite difference changes sign between Instances 1 and 2. Therefore, a single pooled device or lamp correction is unlikely to be adequate for these unmasked measurements. Instance, device and lamp effects need to be considered when interpreting later models.
5. Concentration response and possible saturation
Descriptive least-squares slopes of tube-level mean b* versus concentration, in b* units per % w/v. Early slope uses 0.20–0.50%; late slope uses 0.50–2.00% where enough levels are available.
Condition	Instance	Early slope	Late slope	Interpretation
Dino-Lite off	1	+22.07	+4.11	Strong flattening after low range
Dino-Lite on	1	+27.89	+4.01	Strong flattening after low range
Phone off	1	+38.86	+2.91	Strong flattening after low range
Phone on	1	+36.46	+2.70	Strong flattening after low range
Dino-Lite off	2	+5.25	+0.94	Weak/near-flat response
Dino-Lite on	2	+6.92	+0.86	Weak/near-flat response
Phone off	2	+1.23	+0.65	Nearly flat
Phone on	2	−1.90	+0.90	Nearly flat and non-monotonic at low range
Dino-Lite off	3	Not enough early levels	+3.38	Moderate upward trend; sparse data
Dino-Lite on	3	Not enough early levels	+3.54	Moderate upward trend; sparse data


Interpretation: instance 1 shows a steep low-concentration response but a much flatter high-concentration response. Instance 2 has a weak whole-frame response across the range. Instance 3 increases moderately but is sparse. Do not claim one universal saturation curve across all instances. These are pre-mask whole-frame values, so Phase 5 may change the result substantially.
6. Within-concentration spread versus full response
Condition	Mean within-concentration tube SD	Full b* response span	Ratio
Dino-Lite off	12.29	24.44	0.503
Dino-Lite on	15.09	30.18	0.500
Phone off	18.61	31.98	0.582
Phone on	22.79	38.14	0.597


Interpretation: the average within-concentration tube SD is roughly 50–60% of the full b* response span. That is substantial. In this design, however, each concentration has one tube per instance; this statistic is therefore strongly influenced by between-instance differences and is not a clean estimate of independent tube repeatability within a batch.
7. Descriptive variance decomposition
Fractions of whole-frame b* sum of squares:
Condition	Between concentrations	Tubes within concentration	Photos within tube
Dino-Lite off	34.20%	64.96%	0.83%
Dino-Lite on	34.52%	64.86%	0.62%
Phone off	34.99%	64.13%	0.88%
Phone on	34.41%	64.33%	1.26%


Interpretation: in the current descriptive decomposition, most variation is between tubes within concentration; photo-to-photo variation within a tube is small. Because one tube exists per concentration per instance, the tube-within-concentration term is confounded with instance. The small within-tube component should not be taken to mean individual photographs are statistically independent.

The paired device/lamp analysis above separately describes those acquisition-condition differences. The table is descriptive ANOVA-style sum-of-squares decomposition, not a balanced mixed-effects variance-components model.
8. Transparent-pixel fraction
No source image has an alpha channel. The transparent-pixel fraction is zero for all 864 images in every condition. This only confirms that the original BMP/JPG images are opaque. It does not measure the fraction of the image occupied by the droplet or provide a droplet mask.
9. Image usability and inclusion
image_quality_review.csv contains 864 development-image rows, but the manual fields focus_ok, droplet_visible, boundary_distinguishable, not_severely_cropped, no_severe_obstruction_or_glare, usable, exclusion_reason, and review_notes are blank for all 864 rows.

Therefore, no visual image-quality decision has yet been made. Phase 2 confirmed that all original images can be opened and decoded, but that does not establish that every droplet is visually suitable for ROI extraction.

Do not exclude any image based on these CSVs alone. Review the development contact sheets and fill the QC sheet using the prewritten inclusion rules. Record each exclusion by image_id and reason. Leave test images unopened.
10. Timestamp analysis
The earlier files phase4_session_contrasts.csv, phase4_drift_summary.csv, and the uploaded phase4_observations.md are from the timestamp-based version of the analysis. They are not used for the current instance-based Phase 4 conclusions. Timestamp remains in metadata as provenance, but within-instance temporal drift is not assessed in this analysis.
Ready-to-use one-line answers for the decision log
- L bimodality:* Whole-frame L* distributions differ by instance, most clearly for Dino-Lite; pooling instances can create apparent modes. This is not evidence of two droplet populations, and whether the modes collapse after masking remains to be checked in Phase 5.
- Instance offset: Instance 1 has substantially higher whole-frame b* than Instance 2 at 11 matched concentrations in every condition (mean differences +19.25 to +32.23 b*); comparisons with Instance 3 are based on only one or two shared concentrations.
- Temporal drift: Not assessed; timestamps remain provenance only, and the current Phase 4 analysis groups by preparation instance.
- Saturation: Instance 1's whole-frame b* response flattens markedly above about 0.5%, while Instance 2 is nearly flat across most of the range and Instance 3 has a moderate but sparse upward trend; there is no single common response curve across instances.
- Within-level spread: Mean within-concentration tube SD is 50–60% of the full response span, but this is confounded with instance because there is one tube per concentration per instance.
- Unusable images: Not determined yet; all 864 manual QC records are blank. Contact-sheet review is required before exclusions can be made.
- Variance decomposition: About 34–35% of whole-frame b* sum of squares is between concentrations, 64–65% between tubes within concentration, and 0.6–1.3% between photos within a tube; the tube component is confounded with instance.