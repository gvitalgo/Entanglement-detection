# Thesis Results Summary

## Dataset
- Total samples: `350000`
- Entangled samples: `194243`
- Separable samples: `155757`
- Basis phase `phi`: `0.0`

## Best Model
- Hidden neurons: `50`
- Test accuracy: `0.9760`
- Model file: `thesis/generated/ml_finalization_seed42/robustness/model_seed_7/models/model_nh_50.keras`

## Model Comparison

| Nh | Accuracy | Type I Error | Type II Error | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7531 | 11549 | 14372 | 35178 | 11549 | 14372 | 43901 |
| 10 | 0.9693 | 1217 | 2009 | 45510 | 1217 | 2009 | 56264 |
| 50 | 0.9760 | 872 | 1648 | 45855 | 872 | 1648 | 56625 |

## Generated Figures
- Accuracy trend: `thesis/generated/ml_finalization_seed42/robustness/model_seed_7/plots/accuracy_by_architecture.png`
- Type I / Type II trend: `thesis/generated/ml_finalization_seed42/robustness/model_seed_7/plots/errors_by_architecture.png`
- Per-model confusion and error figures are stored in the same `plots/` directory in both PNG and PDF formats.
