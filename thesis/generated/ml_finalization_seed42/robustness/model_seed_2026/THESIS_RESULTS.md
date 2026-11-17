# Thesis Results Summary

## Dataset
- Total samples: `350000`
- Entangled samples: `194243`
- Separable samples: `155757`
- Basis phase `phi`: `0.0`

## Best Model
- Hidden neurons: `50`
- Test accuracy: `0.9748`
- Model file: `thesis/generated/ml_finalization_seed42/robustness/model_seed_2026/models/model_nh_50.keras`

## Model Comparison

| Nh | Accuracy | Type I Error | Type II Error | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7542 | 11603 | 14209 | 35124 | 11603 | 14209 | 44064 |
| 10 | 0.9711 | 946 | 2088 | 45781 | 946 | 2088 | 56185 |
| 50 | 0.9748 | 951 | 1692 | 45776 | 951 | 1692 | 56581 |

## Generated Figures
- Accuracy trend: `thesis/generated/ml_finalization_seed42/robustness/model_seed_2026/plots/accuracy_by_architecture.png`
- Type I / Type II trend: `thesis/generated/ml_finalization_seed42/robustness/model_seed_2026/plots/errors_by_architecture.png`
- Per-model confusion and error figures are stored in the same `plots/` directory in both PNG and PDF formats.
