# Phase 6.7 — Final development refit

## Selected model

**SVR_S1_device_lamp**

RBF-SVR with S1 colour features and device/lamp indicators.

## Fixed hyperparameters

- C: 10
- epsilon: 0.02
- gamma: 0.1
- kernel: rbf

## Input features

L_mean, a_mean, b_mean, device_dino, device_phone, lamp_off, lamp_on

## Preprocessing

1. Median imputation.
2. StandardScaler.
3. RBF-SVR.

All preprocessing parameters were fitted on the combined development
partition by the saved pipeline.

## Refit data

- Training: 666 images from 31 tubes.
- Validation: 198 images from 10 tubes.
- Combined development data: 864 images from
  41 tubes.

The selected model was refitted using all development images.
The validation-selected hyperparameters were kept fixed.

## Independent validation evidence

- MAE: 0.198015
- RMSE: 0.240571
- R²: 0.801961

These are independent validation metrics, not in-sample refit metrics.

## Artifacts

- Fitted model: `models/final_svr_s1_device_lamp.joblib`
- Configuration: `reports/final_model_config.json`
- Data audit: `results/phase6_7_refit_data_audit.json`
- Fit manifest: `results/phase6_7_refit_manifest.csv`

## Test isolation

The 200-image test partition was not loaded, transformed, predicted
or otherwise accessed. No final test metrics are reported here.

Phase 7 will perform the one-time final test evaluation.
