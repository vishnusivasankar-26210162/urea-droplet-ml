# Phase 7 — Final test evaluation

Evaluation date: 2026-11-17T15:57:45.194269+00:00

## Frozen model

- Model: SVR_S1_device_lamp
- Development fit: 864 images from 41 physical tubes
- Features: L_mean, a_mean, b_mean, device/lamp indicators
- Hyperparameters: RBF kernel, C=10, epsilon=0.02, gamma=0.1
- Model SHA256: `5afb9b97a4bd73eb7fa569ffc2483ddc37646d86fb162277bc669ec4cf22adaa`

## Test partition

- Images: 200
- Physical tubes: 10
- Test images were processed once with the frozen pipeline.
- No model or preprocessing decisions were changed using test results.

## Segmentation and feature processing

- Feature status `ok`: 200

Questionable masks remain identified in the processing manifest. Rows with failed segmentation or feature extraction were retained; the model's development-fitted imputer handled missing colour features.

## Test metrics

| Model | MAE | RMSE | R² | Bias | Within ±0.1 | Tube-bootstrap 95% CI for MAE |
|---|---:|---:|---:|---:|---:|---:|
| SVR_S1_device_lamp | 0.1462 | 0.1883 | 0.8554 | -0.0284 | 0.485 | [0.1068, 0.1849] |
| B0_mean | 0.4316 | 0.5149 | -0.0816 | -0.1414 | 0.240 | [0.2524, 0.6121] |
| B1_b_med_linear | 0.3160 | 0.3900 | 0.3794 | -0.0430 | 0.230 | [0.2662, 0.3746] |

## Concentration-band MAE

| Model | Low (<0.5%) | Medium (0.5–1.0%) | High (>1.0%) |
|---|---:|---:|---:|
| SVR_S1_device_lamp | 0.0645 | 0.1841 | 0.1370 |
| B0_mean | 0.6766 | 0.1756 | 0.5674 |
| B1_b_med_linear | 0.3132 | 0.2632 | 0.3569 |

## Calibration

Calibration is reported as actual concentration regressed on predicted concentration.

| Model | Slope | Intercept |
|---|---:|---:|
| SVR_S1_device_lamp | 0.9017 | 0.1356 |
| B0_mean | nan | nan |
| B1_b_med_linear | 0.8824 | 0.1694 |

## Uncertainty method

- 2,000 paired bootstrap resamples.
- Resampling unit: physical tube (10 tubes).
- Random seed: 0.
- Interval: 2.5th to 97.5th percentile of bootstrap MAE.
- The same sampled tubes were used for all three models in each replicate.

## Interpretation boundary

These are final test results for the frozen pipeline. They must not be used to tune the model, features, segmentation, erosion, glare thresholds, or other preprocessing choices. Any future model changes require a new independent test set.

## Output files

- `results/test_predictions.csv`
- `results/test_processing_manifest.csv`
- `results/test_results.csv`
- `results/test_tube_bootstrap_samples.csv`
- `results/phase7_test_accessed.json`
- `reports/phase7_final_test.md`
