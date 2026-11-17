# Noisy Bell-State Benchmark

This benchmark is separate from the full held-out test-set evaluation. It does not measure performance on the full generated density-matrix distribution.

Instead, it evaluates the already-trained classifiers on the one-parameter noisy Bell-state family from Eq. (10): `rho(p)=p|psi><psi|+(1-p)I/4`, where `|psi>=(|00>+|11>)/sqrt(2)`.

The theoretical transition is `p=1/3`: states with `p<1/3` are separable and states with `p>1/3` are entangled. The PPT margin is recorded for analysis only and is not used as an ANN input feature.

| Nh | Accuracy | Type-I count | Type-I rate | Type-II count | Type-II rate | p_star grid | p_star linear | BLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.910090 | 0 | 0.000000 | 90 | 0.134933 | 0.424000 | 0.423048 | 0.089715 |
| 10 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335185 | 0.001851 |
| 50 | 0.999001 | 1 | 0.002994 | 0 | 0.000000 | 0.333000 | 0.332143 | 0.001190 |
