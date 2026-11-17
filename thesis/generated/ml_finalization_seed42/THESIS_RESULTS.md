# Thesis Results Summary

## Dataset
- Total samples: `350000`
- Entangled samples: `194243`
- Separable samples: `155757`
- Basis phase `phi`: `0.0`

## Best Model
- Hidden neurons: `45`
- Test accuracy: `0.9759`
- Model file: `thesis/generated/ml_finalization_seed42/models/model_nh_45.keras`

## Model Comparison

| Nh | Accuracy | Type I Error | Type II Error | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.7535 | 11642 | 14240 | 35085 | 11642 | 14240 | 44033 |
| 5 | 0.9060 | 2920 | 6955 | 43807 | 2920 | 6955 | 51318 |
| 10 | 0.9716 | 740 | 2240 | 45987 | 740 | 2240 | 56033 |
| 15 | 0.9716 | 1167 | 1819 | 45560 | 1167 | 1819 | 56454 |
| 20 | 0.9758 | 681 | 1864 | 46046 | 681 | 1864 | 56409 |
| 25 | 0.9750 | 792 | 1830 | 45935 | 792 | 1830 | 56443 |
| 30 | 0.9757 | 751 | 1796 | 45976 | 751 | 1796 | 56477 |
| 35 | 0.9753 | 832 | 1761 | 45895 | 832 | 1761 | 56512 |
| 40 | 0.9753 | 852 | 1741 | 45875 | 852 | 1741 | 56532 |
| 45 | 0.9759 | 826 | 1708 | 45901 | 826 | 1708 | 56565 |
| 50 | 0.9736 | 1017 | 1755 | 45710 | 1017 | 1755 | 56518 |

## Generated Figures
- Accuracy trend: `thesis/generated/ml_finalization_seed42/plots/accuracy_by_architecture.png`
- Type I / Type II trend: `thesis/generated/ml_finalization_seed42/plots/errors_by_architecture.png`
- Per-model confusion and error figures are stored in the same `plots/` directory in both PNG and PDF formats.
