from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .config import PathConfig


class ThesisReportWriter:
    def __init__(self, path_config: PathConfig) -> None:
        self.path_config = path_config

    def write(self, summary: dict[str, Any]) -> None:
        self._write_csv(summary)
        self._write_latex(summary)
        self._write_markdown(summary)

    def _write_csv(self, summary: dict[str, Any]) -> None:
        with self.path_config.training_table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(
                [
                    "hidden_neurons",
                    "test_accuracy",
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "true_negative",
                    "false_positive",
                    "false_negative",
                    "true_positive",
                    "type_i_error",
                    "type_ii_error",
                    "type_i_error_rate",
                    "type_ii_error_rate",
                    "trainable_parameter_count",
                    "training_loss_min",
                    "validation_loss_min",
                    "runtime_seconds",
                    "stopping_epoch",
                    "model_seed",
                    "split_seed",
                    "model_path",
                    "prediction_artifact",
                ]
            )
            for model in summary["models"]:
                confusion = model["confusion_matrix"]
                writer.writerow(
                    [
                        model["hidden_neurons"],
                        f"{model['test_accuracy']:.6f}",
                        f"{model['balanced_accuracy']:.6f}",
                        f"{model['precision']:.6f}",
                        f"{model['recall']:.6f}",
                        f"{model['f1']:.6f}",
                        f"{model['roc_auc']:.6f}",
                        confusion["tn"],
                        confusion["fp"],
                        confusion["fn"],
                        confusion["tp"],
                        confusion["fp"],
                        confusion["fn"],
                        f"{model['type_i_error_rate']:.8f}",
                        f"{model['type_ii_error_rate']:.8f}",
                        model["trainable_parameter_count"],
                        f"{model.get('training_loss_min', float('nan')):.8f}",
                        f"{model.get('validation_loss_min', float('nan')):.8f}",
                        f"{model.get('runtime_seconds', float('nan')):.3f}",
                        model.get("stopping_epoch", ""),
                        model.get("model_seed", ""),
                        model.get("split_seed", ""),
                        model["model_path"],
                        model["prediction_artifact"],
                    ]
                )

    def _write_latex(self, summary: dict[str, Any]) -> None:
        path = self.path_config.metrics_dir / "hidden_width_sweep_table.tex"
        lines = [
            "% Automatically generated; do not edit by hand.",
            "\\begin{table}[htbp]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{rrrrrrrr}",
            "\\toprule",
            "$N_h$ & Params & Accuracy & Bal. acc. & Type I & Type II & Epoch & Time (s) \\\\",
            "\\midrule",
        ]
        for model in summary["models"]:
            lines.append(
                f"{model['hidden_neurons']} & {model['trainable_parameter_count']} & "
                f"{100.0 * model['test_accuracy']:.2f}\\% & "
                f"{100.0 * model['balanced_accuracy']:.2f}\\% & "
                f"{100.0 * model['type_i_error_rate']:.2f}\\% & "
                f"{100.0 * model['type_ii_error_rate']:.2f}\\% & "
                f"{model.get('stopping_epoch', '--')} & {model.get('runtime_seconds', float('nan')):.1f} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "\\caption{Complete hidden-width sweep on the fixed canonical train/test split.}",
                "\\label{tab:hidden-width-sweep}",
                "\\end{table}",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_markdown(self, summary: dict[str, Any]) -> None:
        manifest = summary["dataset_manifest"]
        best_model = summary["best_model"]

        lines = [
            "# Thesis Results Summary",
            "",
            "## Dataset",
            f"- Total samples: `{manifest['total_samples']}`",
            f"- Entangled samples: `{manifest['class_counts']['entangled']}`",
            f"- Separable samples: `{manifest['class_counts']['separable']}`",
            f"- Basis phase `phi`: `{manifest['basis_config']['phi']}`",
            "",
            "## Best Model",
            f"- Hidden neurons: `{best_model['hidden_neurons']}`",
            f"- Test accuracy: `{best_model['test_accuracy']:.4f}`",
            f"- Model file: `{best_model['model_path']}`",
            "",
            "## Model Comparison",
            "",
            "| Nh | Accuracy | Type I Error | Type II Error | TN | FP | FN | TP |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]

        for model in summary["models"]:
            confusion = model["confusion_matrix"]
            lines.append(
                f"| {model['hidden_neurons']} | {model['test_accuracy']:.4f} | "
                f"{confusion['fp']} | {confusion['fn']} | {confusion['tn']} | "
                f"{confusion['fp']} | {confusion['fn']} | {confusion['tp']} |"
            )

        lines.extend(
            [
                "",
                "## Generated Figures",
                f"- Accuracy trend: `{self.path_config.plots_dir / 'accuracy_by_architecture.png'}`",
                f"- Type I / Type II trend: `{self.path_config.plots_dir / 'errors_by_architecture.png'}`",
                "- Per-model confusion and error figures are stored in the same `plots/` directory in both PNG and PDF formats.",
            ]
        )

        self.path_config.thesis_report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
