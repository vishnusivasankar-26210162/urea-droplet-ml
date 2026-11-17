# Phase 6.4 — One-standard-error model selection

- Training images: 666
- Training tubes: 31
- Bootstrap repetitions: 2000
- Bootstrap seed: 0
- Resampling unit: physical tube (`tube_id`)
- Validation used: no
- Test used: no

## Bootstrap results

| Model | OOF MAE | Bootstrap SE | 95% CI | Within one SE | Selected |
|---|---:|---:|---:|---|---|
| B0 | 0.546092 | 0.054950 | [0.4408, 0.6573] | No | No |
| B1 | 0.355177 | 0.027685 | [0.3007, 0.4100] | No | No |
| Polynomial_Ridge_b_med | 0.348651 | 0.026859 | [0.2974, 0.4026] | No | No |
| Ridge_S2_scratch | 0.302931 | 0.027025 | [0.2521, 0.3596] | No | No |
| SVR_RBF_S2 | 0.221360 | 0.020333 | [0.1841, 0.2625] | Yes | **Yes** |
| Random_Forest_S2 | 0.210330 | 0.019909 | [0.1725, 0.2497] | Yes | No |
| Gradient_Boosting_S2 | 0.215895 | 0.020629 | [0.1783, 0.2561] | Yes | No |
| MLP_S2 | 0.234756 | 0.019608 | [0.1998, 0.2764] | No | No |

## One-standard-error calculation

- Best model by OOF MAE: **Random_Forest_S2**
- Best OOF MAE: 0.210330
- Bootstrap standard error of best model: 0.019909
- Cutoff: 0.210330 + 0.019909 = **0.230240**
- Selected simplest eligible model: **SVR_RBF_S2**

Selection uses the predeclared simplicity order:

1. B0
2. B1
3. Polynomial_Ridge_b_med
4. Ridge_S2_scratch
5. SVR_RBF_S2
6. Random_Forest_S2
7. Gradient_Boosting_S2
8. MLP_S2

## Interpretation

This is a development-only model candidate selected by the one-standard-error rule. It is not final test performance. The 10 validation tubes remain reserved for candidate assessment, and the 10 test tubes remain locked.

Bootstrap uncertainty is estimated by resampling physical tubes with replacement and recomputing image-pooled MAE from per-tube error sums and image counts.
