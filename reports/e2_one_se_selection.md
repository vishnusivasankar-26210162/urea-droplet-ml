# E2 One-Standard-Error Selection

- Configurations compared: 20
- Training images: 666
- Training tubes: 31
- Bootstrap repetitions: 2000
- Seed: 0
- Resampling unit: physical tube
- Same bootstrap draws used for all configurations
- Validation used: no
- Test used: no

## Results

| Configuration | Features | OOF MAE | Bootstrap SE | 95% interval | Within one SE | Selected |
|---|---:|---:|---:|---:|---|---|
| SVR__S1_with_device_and_lamp | 7 | 0.191504 | 0.019929 | [0.154639, 0.231872] | Yes | **Yes** |
| SVR__S2_with_device_and_lamp | 13 | 0.209579 | 0.019877 | [0.174311, 0.250026] | Yes | No |
| SVR__S3_with_device_and_lamp | 26 | 0.210156 | 0.022572 | [0.167205, 0.255436] | Yes | No |
| SVR__S4_colour_only | 53 | 0.204667 | 0.024475 | [0.161242, 0.255897] | Yes | No |
| SVR__S4_with_device_and_lamp | 57 | 0.197579 | 0.023340 | [0.154891, 0.245339] | Yes | No |
| Ridge__S1_colour_only | 3 | 0.305726 | 0.027973 | [0.252812, 0.363620] | No | No |
| Ridge__S0_colour_only | 3 | 0.431783 | 0.035158 | [0.363688, 0.501306] | No | No |
| Ridge__S1_with_device_and_lamp | 7 | 0.316318 | 0.038193 | [0.245588, 0.395240] | No | No |
| Ridge__S0_with_device_and_lamp | 7 | 0.399584 | 0.034479 | [0.335069, 0.466743] | No | No |
| Ridge__S2_colour_only | 9 | 0.302931 | 0.027025 | [0.252143, 0.359578] | No | No |
| Ridge__S2_with_device_and_lamp | 13 | 0.302698 | 0.037515 | [0.232869, 0.378603] | No | No |
| Ridge__S3_colour_only | 22 | 0.307514 | 0.026998 | [0.257459, 0.361437] | No | No |
| Ridge__S3_with_device_and_lamp | 26 | 0.301003 | 0.037334 | [0.230416, 0.374606] | No | No |
| Ridge__S4_colour_only | 53 | 0.307080 | 0.030671 | [0.249402, 0.367903] | No | No |
| Ridge__S4_with_device_and_lamp | 57 | 0.284686 | 0.032150 | [0.223607, 0.349406] | No | No |
| SVR__S1_colour_only | 3 | 0.227257 | 0.022355 | [0.183229, 0.272605] | No | No |
| SVR__S0_colour_only | 3 | 0.277957 | 0.025263 | [0.230714, 0.328547] | No | No |
| SVR__S0_with_device_and_lamp | 7 | 0.256488 | 0.028612 | [0.201892, 0.314922] | No | No |
| SVR__S2_colour_only | 9 | 0.221360 | 0.020333 | [0.184076, 0.262500] | No | No |
| SVR__S3_colour_only | 22 | 0.225678 | 0.024713 | [0.180097, 0.277192] | No | No |

## Selection

- Best raw configuration: **SVR__S1_with_device_and_lamp**
- Best raw OOF MAE: 0.191504
- Bootstrap SE of best configuration: 0.019929
- One-SE cutoff: **0.211432**
- Selected simplest eligible configuration: **SVR__S1_with_device_and_lamp**

Simplicity order: Ridge before RBF-SVR; fewer features before more features; colour-only before device/lamp when counts tie; lower MAE as final tie-breaker.

The selected configuration is a development candidate only. Validation and test data were not used.
