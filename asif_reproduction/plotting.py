from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from ._dependencies import require_matplotlib

LOGGER = logging.getLogger(__name__)


class ExperimentPlotter:
    def __init__(self, output_dir: Path, logger: logging.Logger | None = None) -> None:
        self.output_dir = output_dir
        self.logger = logger or LOGGER

    def plot_model_metrics(self, metrics: dict[str, Any]) -> None:
        try:
            pyplot = self._pyplot()
        except ModuleNotFoundError:
            self.logger.warning("matplotlib is not installed; skipping per-model plots.")
            return

        confusion = metrics["confusion_matrix"]
        hidden_neurons = metrics["hidden_neurons"]
        self._plot_confusion_bars(pyplot, hidden_neurons, confusion)
        self._plot_error_bars(pyplot, hidden_neurons, confusion)

    def plot_summary(self, summary: dict[str, Any]) -> None:
        try:
            pyplot = self._pyplot()
        except ModuleNotFoundError:
            self.logger.warning("matplotlib is not installed; skipping summary plots.")
            return

        self._plot_accuracy_curve(pyplot, summary["models"])
        self._plot_error_comparison(pyplot, summary["models"])

    def plot_noisy_bell_benchmark(
        self,
        p_values,
        probabilities_by_architecture: dict[int, Any],
        theoretical_threshold: float,
    ) -> None:
        try:
            pyplot = self._pyplot()
        except ModuleNotFoundError:
            self.logger.warning("matplotlib is not installed; skipping noisy Bell benchmark plot.")
            return

        figure, axis = pyplot.subplots(figsize=(7.2, 4.5))
        axis.axvspan(0.0, theoretical_threshold, color="#2a9d8f", alpha=0.10, label="Separable region")
        axis.axvspan(theoretical_threshold, 1.0, color="#e76f51", alpha=0.08, label="Entangled region")
        axis.axvline(theoretical_threshold, color="#111111", linestyle="--", linewidth=1.4, label="p = 1/3")

        colors = {0: "#1d3557", 10: "#f77f00", 50: "#6a4c93"}
        for hidden_neurons, probabilities in probabilities_by_architecture.items():
            axis.plot(
                p_values,
                probabilities,
                linewidth=2,
                color=colors.get(hidden_neurons),
                label=f"Nh={hidden_neurons}",
            )

        axis.set_title("Noisy Bell-State Benchmark")
        axis.set_xlabel("Noise factor p")
        axis.set_ylabel("Predicted probability of entanglement")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(loc="best")
        figure.tight_layout()
        self._save_figure(figure, "noisy_bell_benchmark")
        pyplot.close(figure)

    def plot_ble_by_architecture(self, model_metrics: dict[int, dict[str, Any]]) -> None:
        try:
            pyplot = self._pyplot()
        except ModuleNotFoundError:
            self.logger.warning("matplotlib is not installed; skipping BLE plot.")
            return

        widths = sorted(model_metrics)
        ble_values = [model_metrics[width]["ble"] for width in widths]
        figure, axis = pyplot.subplots(figsize=(7.0, 4.2))
        axis.plot(widths, ble_values, marker="o", linewidth=1.7, color="#1d3557")
        axis.set_xlabel("Hidden-layer width $N_h$")
        axis.set_ylabel("Pathwise BLE")
        axis.set_title("Noisy Bell-State Boundary-Location Error")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        self._save_figure(figure, "ble_by_architecture")
        pyplot.close(figure)

    def _plot_confusion_bars(self, pyplot, hidden_neurons: int, confusion: dict[str, int]) -> None:
        labels = ["Truly Separable", "Falsely Entangled", "Falsely Separable", "Truly Entangled"]
        values = [confusion["tn"], confusion["fp"], confusion["fn"], confusion["tp"]]
        colors = ["#2a9d8f", "#e76f51", "#f4a261", "#264653"]

        figure = pyplot.figure(figsize=(8, 4.5))
        pyplot.bar(labels, values, color=colors)
        pyplot.title(f"Confusion Categories for Nh={hidden_neurons}")
        pyplot.ylabel("Number of States")
        pyplot.xticks(rotation=12, ha="right")
        pyplot.tight_layout()
        self._save_figure(figure, f"confusion_nh_{hidden_neurons}")
        pyplot.close(figure)

    def _plot_error_bars(self, pyplot, hidden_neurons: int, confusion: dict[str, int]) -> None:
        labels = ["Type I Error", "Type II Error"]
        values = [confusion["fp"], confusion["fn"]]
        colors = ["#d62828", "#f77f00"]

        figure = pyplot.figure(figsize=(6, 4))
        pyplot.bar(labels, values, color=colors)
        pyplot.title(f"Type I / Type II Errors for Nh={hidden_neurons}")
        pyplot.ylabel("Number of States")
        pyplot.tight_layout()
        self._save_figure(figure, f"errors_nh_{hidden_neurons}")
        pyplot.close(figure)

    def _plot_accuracy_curve(self, pyplot, model_metrics: list[dict[str, Any]]) -> None:
        hidden_neurons = [item["hidden_neurons"] for item in model_metrics]
        accuracies = [item["test_accuracy"] for item in model_metrics]

        figure = pyplot.figure(figsize=(7, 4.2))
        pyplot.plot(hidden_neurons, accuracies, marker="o", color="#1d3557", linewidth=2)
        pyplot.title("Accuracy by Hidden-Layer Width")
        pyplot.xlabel("Hidden Neurons")
        pyplot.ylabel("Test Accuracy")
        pyplot.ylim(0.0, 1.0)
        pyplot.grid(alpha=0.25)
        pyplot.tight_layout()
        self._save_figure(figure, "accuracy_by_architecture")
        pyplot.close(figure)

    def _plot_error_comparison(self, pyplot, model_metrics: list[dict[str, Any]]) -> None:
        hidden_neurons = [item["hidden_neurons"] for item in model_metrics]
        type_i = [item["type_i_error_rate"] for item in model_metrics]
        type_ii = [item["type_ii_error_rate"] for item in model_metrics]

        figure = pyplot.figure(figsize=(7, 4.2))
        pyplot.plot(hidden_neurons, type_i, marker="o", color="#d62828", label="Type I Error")
        pyplot.plot(hidden_neurons, type_ii, marker="o", color="#f77f00", label="Type II Error")
        pyplot.title("Type I and Type II Errors by Hidden-Layer Width")
        pyplot.xlabel("Hidden Neurons")
        pyplot.ylabel("Conditional error rate")
        pyplot.ylim(bottom=0.0)
        pyplot.grid(alpha=0.25)
        pyplot.legend()
        pyplot.tight_layout()
        self._save_figure(figure, "errors_by_architecture")
        pyplot.close(figure)

    def _pyplot(self):
        matplotlib = require_matplotlib()
        cache_dir = self.output_dir / ".matplotlib-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        return pyplot

    def _save_figure(self, figure, stem: str) -> None:
        png_path = self.output_dir / f"{stem}.png"
        pdf_path = self.output_dir / f"{stem}.pdf"
        figure.savefig(png_path, dpi=180)
        figure.savefig(pdf_path)
