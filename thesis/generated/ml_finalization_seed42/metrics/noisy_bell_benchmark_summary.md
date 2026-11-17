# Noisy Bell-State Benchmark

This benchmark is separate from the full held-out test-set evaluation. It does not measure performance on the full generated density-matrix distribution.

Instead, it evaluates the already-trained classifiers on the one-parameter noisy Bell-state family from Eq. (10): `rho(p)=p|psi><psi|+(1-p)I/4`, where `|psi>=(|00>+|11>)/sqrt(2)`.

The theoretical transition is `p=1/3`: states with `p<1/3` are separable and states with `p>1/3` are entangled. The PPT margin is recorded for analysis only and is not used as an ANN input feature.

| Nh | Accuracy | Type-I count | Type-I rate | Type-II count | Type-II rate | p_star grid | p_star linear | BLE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.911089 | 0 | 0.000000 | 89 | 0.133433 | 0.423000 | 0.422413 | 0.089080 |
| 5 | 0.973027 | 0 | 0.000000 | 27 | 0.040480 | 0.361000 | 0.360390 | 0.027057 |
| 10 | 0.995005 | 0 | 0.000000 | 5 | 0.007496 | 0.339000 | 0.338903 | 0.005570 |
| 15 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335198 | 0.001864 |
| 20 | 0.997003 | 0 | 0.000000 | 3 | 0.004498 | 0.337000 | 0.336445 | 0.003112 |
| 25 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335917 | 0.002584 |
| 30 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335463 | 0.002130 |
| 35 | 0.997003 | 0 | 0.000000 | 3 | 0.004498 | 0.337000 | 0.336109 | 0.002775 |
| 40 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335548 | 0.002214 |
| 45 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335240 | 0.001906 |
| 50 | 0.998002 | 0 | 0.000000 | 2 | 0.002999 | 0.336000 | 0.335647 | 0.002314 |
