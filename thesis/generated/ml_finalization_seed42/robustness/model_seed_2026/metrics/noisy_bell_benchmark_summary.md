# Noisy Bell-State Benchmark

This benchmark is separate from the full held-out test-set evaluation. It does not measure performance on the full generated density-matrix distribution.

Instead, it evaluates the already-trained classifiers on the one-parameter noisy Bell-state family from Eq. (10): `rho(p)=p|psi><psi|+(1-p)I/4`, where `|psi>=(|00>+|11>)/sqrt(2)`.

The theoretical transition is `p=1/3`: states with `p<1/3` are separable and states with `p>1/3` are entangled. The PPT margin is recorded for analysis only and is not used as an ANN input feature.

| Nh | Accuracy | Type-I count | Type-I rate | Type-II count | Type-II rate | p_star grid | p_star linear | BLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.913087 | 0 | 0.000000 | 87 | 0.130435 | 0.421000 | 0.420005 | 0.086671 |
| 10 | 0.997003 | 0 | 0.000000 | 3 | 0.004498 | 0.337000 | 0.336151 | 0.002818 |
| 50 | 1.000000 | 0 | 0.000000 | 0 | 0.000000 | 0.334000 | 0.333702 | 0.000368 |
