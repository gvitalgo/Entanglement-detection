from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._dependencies import require_qutip, require_tensorflow
from .basis_factory import CoherenceBasisFactory
from .ble import estimate_pathwise_ble
from .coherence_extractor import RelativeEntropyCoherenceExtractor
from .config import BasisConfig, PathConfig
from .ppt_labeler import PPTLabeler
from .plotting import ExperimentPlotter

THEORETICAL_THRESHOLD = 1.0 / 3.0
FEATURE_NAMES = ("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)")


@dataclass(slots=True)
class PreparedBenchmark:
    p_values: np.ndarray
    analytic_labels: np.ndarray
    ppt_labels: np.ndarray
    ppt_margins: np.ndarray
    features: np.ndarray


def noisy_bell_state(p: float) -> np.ndarray:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"Noise factor p must be in [0, 1], got {p}.")
    psi = np.asarray([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    pure_state = np.outer(psi, psi.conj())
    return p * pure_state + (1.0 - p) * np.eye(4, dtype=np.complex128) / 4.0


def generate_noisy_bell_grid(p_values: np.ndarray) -> list[np.ndarray]:
    return [noisy_bell_state(float(p_value)) for p_value in np.asarray(p_values, dtype=np.float64)]


class NoisyBellBenchmarkRunner:
    def __init__(
        self,
        path_config: PathConfig,
        basis_config: BasisConfig | None = None,
        architectures: tuple[int, ...] = (0, 10, 50),
        num_points: int = 1001,
    ) -> None:
        if num_points < 2:
            raise ValueError("num_points must be at least 2.")
        self.path_config = path_config
        self.basis_config = basis_config or BasisConfig()
        self.architectures = architectures
        self.num_points = num_points
        self.plotter = ExperimentPlotter(self.path_config.plots_dir)

    def run(self) -> dict[str, Any]:
        self.path_config.ensure_directories()
        benchmark = self._prepare_benchmark()
        probabilities_by_architecture = self._predict_probabilities(benchmark.features)
        labels_by_architecture = {
            hidden_neurons: (probabilities >= 0.5).astype(np.int64)
            for hidden_neurons, probabilities in probabilities_by_architecture.items()
        }

        model_metrics = {
            hidden_neurons: self._classification_metrics(
                benchmark.p_values,
                benchmark.analytic_labels,
                predictions,
                probabilities_by_architecture[hidden_neurons],
            )
            for hidden_neurons, predictions in labels_by_architecture.items()
        }
        sensitivity = self._ble_sensitivity(benchmark, probabilities_by_architecture)

        rows = self._build_rows(benchmark, probabilities_by_architecture, labels_by_architecture)
        csv_path = self.path_config.metrics_dir / "noisy_bell_benchmark.csv"
        json_path = self.path_config.metrics_dir / "noisy_bell_benchmark.json"
        self._write_csv(csv_path, rows)
        self._write_metrics_reports(model_metrics)

        summary = {
            "benchmark": "noisy_bell_state_eq_10",
            "artifact_dir": str(self.path_config.output_dir),
            "num_points": self.num_points,
            "theoretical_threshold": THEORETICAL_THRESHOLD,
            "state_family": "rho(p)=p|Phi+><Phi+|+(1-p)I_4/4, |Phi+>=(|00>+|11>)/sqrt(2)",
            "ground_truth_rule": "separable for p <= 1/3; entangled for p > 1/3",
            "grid": {
                "minimum": 0.0,
                "maximum": 1.0,
                "num_points": self.num_points,
                "spacing": float(1.0 / (self.num_points - 1)),
            },
            "prediction_threshold": 0.5,
            "pathwise_only": True,
            "feature_names": list(FEATURE_NAMES),
            "architectures": list(self.architectures),
            "models": {str(hidden_neurons): metrics for hidden_neurons, metrics in model_metrics.items()},
            "grid_resolution_sensitivity": sensitivity,
            "outputs": {
                "csv": str(csv_path),
                "json": str(json_path),
                "plot_png": str(self.path_config.plots_dir / "noisy_bell_benchmark.png"),
                "plot_pdf": str(self.path_config.plots_dir / "noisy_bell_benchmark.pdf"),
            },
        }
        self._write_json(json_path, summary)
        self.plotter.plot_noisy_bell_benchmark(benchmark.p_values, probabilities_by_architecture, THEORETICAL_THRESHOLD)
        self.plotter.plot_ble_by_architecture(model_metrics)
        return summary

    def _prepare_benchmark(self) -> PreparedBenchmark:
        qutip = require_qutip()
        labeler = PPTLabeler()
        bases = CoherenceBasisFactory(self.basis_config).build()
        extractor = RelativeEntropyCoherenceExtractor(bases)
        p_values = np.linspace(0.0, 1.0, self.num_points, dtype=np.float64)

        feature_rows: list[np.ndarray] = []
        ppt_labels: list[int] = []
        ppt_margins: list[float] = []

        for rho_matrix in generate_noisy_bell_grid(p_values):
            rho = qutip.Qobj(rho_matrix, dims=[[2, 2], [2, 2]])
            feature_rows.append(extractor.extract_features(rho))
            ppt_labels.append(labeler.label(rho))
            ppt_margins.append(labeler.minimum_partial_transpose_eigenvalue(rho))

        if extractor.feature_names != FEATURE_NAMES:
            raise ValueError(f"Expected feature order {FEATURE_NAMES}, got {extractor.feature_names}.")

        return PreparedBenchmark(
            p_values=p_values,
            analytic_labels=(p_values > THEORETICAL_THRESHOLD).astype(np.int64),
            ppt_labels=np.asarray(ppt_labels, dtype=np.int64),
            ppt_margins=np.asarray(ppt_margins, dtype=np.float64),
            features=np.asarray(feature_rows, dtype=np.float64),
        )

    def _predict_probabilities(self, features: np.ndarray) -> dict[int, np.ndarray]:
        tf = require_tensorflow()
        scaler = self._fit_saved_training_scaler()
        scaled_features = scaler.transform(features)
        probabilities_by_architecture: dict[int, np.ndarray] = {}

        for hidden_neurons in self.architectures:
            model_path = self.path_config.model_dir / f"model_nh_{hidden_neurons}.keras"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            model = tf.keras.models.load_model(model_path)
            probabilities_by_architecture[hidden_neurons] = model.predict(scaled_features, verbose=0).reshape(-1)

        return probabilities_by_architecture

    def _fit_saved_training_scaler(self):
        if self.path_config.scaler_path.exists():
            scaler_data = np.load(self.path_config.scaler_path)
            _, StandardScaler = _require_sklearn_scaler()
            scaler = StandardScaler()
            scaler.mean_ = scaler_data["mean"]
            scaler.scale_ = scaler_data["scale"]
            scaler.var_ = scaler_data["var"]
            scaler.n_features_in_ = int(scaler.mean_.shape[0])
            scaler.n_samples_seen_ = scaler_data["n_samples_seen"]
            return scaler

        train_test_split, StandardScaler = _require_sklearn_scaler()
        dataset_path = self.path_config.dataset_npz_path
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset archive not found: {dataset_path}")

        data = np.load(dataset_path)
        feature_names = tuple(str(name) for name in data["feature_names"].tolist())
        if feature_names != FEATURE_NAMES:
            raise ValueError(f"Expected dataset feature order {FEATURE_NAMES}, got {feature_names}.")

        training_config = self._load_training_config()
        X_train, _, _, _ = train_test_split(
            data["X"],
            data["y"],
            test_size=float(training_config.get("test_size", 0.30)),
            random_state=int(training_config.get("random_state", 42)),
            stratify=data["y"],
        )
        return StandardScaler().fit(X_train)

    def _load_training_config(self) -> dict[str, Any]:
        summary_path = self.path_config.training_summary_path
        if not summary_path.exists():
            return {"test_size": 0.30, "random_state": 42}
        with summary_path.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
        return dict(summary.get("training_config", {"test_size": 0.30, "random_state": 42}))

    def _build_rows(
        self,
        benchmark: PreparedBenchmark,
        probabilities_by_architecture: dict[int, np.ndarray],
        labels_by_architecture: dict[int, np.ndarray],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for index, p_value in enumerate(benchmark.p_values):
            row: dict[str, Any] = {
                "p": float(p_value),
                "analytic_label": int(benchmark.analytic_labels[index]),
                "ppt_label": int(benchmark.ppt_labels[index]),
                "ppt_margin": float(benchmark.ppt_margins[index]),
                "Cr_QS": float(benchmark.features[index, 0]),
                "Cr_RS": float(benchmark.features[index, 1]),
                "Cr_RT": float(benchmark.features[index, 2]),
                "Cr_QT": float(benchmark.features[index, 3]),
            }
            for hidden_neurons in self.architectures:
                row[f"prob_entangled_Nh{hidden_neurons}"] = float(probabilities_by_architecture[hidden_neurons][index])
            for hidden_neurons in self.architectures:
                row[f"pred_label_Nh{hidden_neurons}"] = int(labels_by_architecture[hidden_neurons][index])
            rows.append(row)
        return rows

    def _classification_metrics(
        self,
        p_values: np.ndarray,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> dict[str, Any]:
        true_negative = int(np.sum((labels == 0) & (predictions == 0)))
        false_positive = int(np.sum((labels == 0) & (predictions == 1)))
        false_negative = int(np.sum((labels == 1) & (predictions == 0)))
        true_positive = int(np.sum((labels == 1) & (predictions == 1)))
        samples = int(labels.shape[0])
        ble = estimate_pathwise_ble(
            p_values,
            probabilities,
            ground_truth_transition=THEORETICAL_THRESHOLD,
            prediction_threshold=0.5,
        )
        p_star = ble.grid_crossing
        return {
            "accuracy": _safe_divide(true_negative + true_positive, samples),
            "tn": true_negative,
            "fp": false_positive,
            "fn": false_negative,
            "tp": true_positive,
            "type_i_count": false_positive,
            "type_ii_count": false_negative,
            "type_i_error": false_positive,
            "type_i_error_rate": _safe_divide(false_positive, false_positive + true_negative),
            "type_i_rate": _safe_divide(false_positive, false_positive + true_negative),
            "type_ii_error": false_negative,
            "type_ii_error_rate": _safe_divide(false_negative, false_negative + true_positive),
            "type_ii_rate": _safe_divide(false_negative, false_negative + true_positive),
            "confusion_matrix": {
                "tn": true_negative,
                "fp": false_positive,
                "fn": false_negative,
                "tp": true_positive,
            },
            "samples": samples,
            "p_star": p_star,
            "p_star_interpolated": ble.interpolated_crossing,
            "p_star_error": ble.ble_grid,
            "ble": ble.ble_interpolated,
            "ble_definition": "absolute pathwise transition-location error after linear interpolation",
            "ble_details": ble.to_dict(),
            "transition_estimate": p_star,
            "transition_error": ble.ble_grid,
        }

    def _ble_sensitivity(
        self,
        benchmark: PreparedBenchmark,
        probabilities_by_architecture: dict[int, np.ndarray],
    ) -> dict[str, Any]:
        candidate_points = (101, 1001, self.num_points)
        sensitivity: dict[str, Any] = {}
        for point_count in dict.fromkeys(candidate_points):
            if point_count == self.num_points:
                grid = benchmark.p_values
                probabilities = probabilities_by_architecture
            else:
                grid = np.linspace(0.0, 1.0, point_count, dtype=np.float64)
                prepared = self._prepare_benchmark_for_grid(grid)
                probabilities = self._predict_probabilities(prepared.features)
            sensitivity[str(point_count)] = {
                "grid_spacing": float(1.0 / (point_count - 1)),
                "models": {
                    str(hidden_neurons): estimate_pathwise_ble(
                        grid,
                        model_probabilities,
                        ground_truth_transition=THEORETICAL_THRESHOLD,
                        prediction_threshold=0.5,
                    ).to_dict()
                    for hidden_neurons, model_probabilities in probabilities.items()
                },
            }
        return sensitivity

    def _prepare_benchmark_for_grid(self, p_values: np.ndarray) -> PreparedBenchmark:
        original_num_points = self.num_points
        try:
            self.num_points = int(p_values.size)
            return self._prepare_benchmark()
        finally:
            self.num_points = original_num_points

    def _write_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "p",
            "analytic_label",
            "ppt_label",
            "ppt_margin",
            "Cr_QS",
            "Cr_RS",
            "Cr_RT",
            "Cr_QT",
        ]
        fieldnames.extend(f"prob_entangled_Nh{hidden_neurons}" for hidden_neurons in self.architectures)
        fieldnames.extend(f"pred_label_Nh{hidden_neurons}" for hidden_neurons in self.architectures)

        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _write_metrics_reports(self, model_metrics: dict[int, dict[str, Any]]) -> None:
        metrics_rows = [
            {
                "hidden_neurons": hidden_neurons,
                "samples": metrics["samples"],
                "tn": metrics["tn"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "tp": metrics["tp"],
                "type_i_count": metrics["type_i_count"],
                "type_ii_count": metrics["type_ii_count"],
                "type_i_rate": metrics["type_i_rate"],
                "type_ii_rate": metrics["type_ii_rate"],
                "accuracy": metrics["accuracy"],
                "p_star": metrics["p_star"],
                "p_star_interpolated": metrics["p_star_interpolated"],
                "p_star_error": metrics["p_star_error"],
                "ble": metrics["ble"],
            }
            for hidden_neurons, metrics in model_metrics.items()
        ]
        payload = {
            "benchmark": "noisy_bell_state_eq_10",
            "description": (
                "Aggregate classifier metrics for the one-parameter noisy Bell-state family "
                "rho(p)=p|psi><psi|+(1-p)I/4 with |psi>=(|00>+|11>)/sqrt(2)."
            ),
            "artifact_dir": str(self.path_config.output_dir),
            "num_points": self.num_points,
            "theoretical_threshold": THEORETICAL_THRESHOLD,
            "metrics": {str(row["hidden_neurons"]): row for row in metrics_rows},
        }

        self._write_json(self.path_config.metrics_dir / "noisy_bell_benchmark_metrics.json", payload)
        self._write_metrics_csv(self.path_config.metrics_dir / "noisy_bell_benchmark_metrics.csv", metrics_rows)
        self._write_latex_table(self.path_config.metrics_dir / "noisy_bell_benchmark_table.tex", metrics_rows)
        self._write_markdown_summary(self.path_config.metrics_dir / "noisy_bell_benchmark_summary.md", metrics_rows)

    def _write_metrics_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        fieldnames = [
            "hidden_neurons",
            "samples",
            "tn",
            "fp",
            "fn",
            "tp",
            "type_i_count",
            "type_ii_count",
            "type_i_rate",
            "type_ii_rate",
            "accuracy",
            "p_star",
            "p_star_interpolated",
            "p_star_error",
            "ble",
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def _write_latex_table(self, path: Path, rows: list[dict[str, Any]]) -> None:
        lines = [
            "\\begin{table}[H]",
            "\\centering",
            "\\small",
            "\\resizebox{\\textwidth}{!}{%",
            "\\begin{tabular}{ccccccccc}",
            "\\toprule",
            "$N_h$ & Accuracy & Type-I count & Type-I rate & Type-II count & Type-II rate & $p_\\ast^{grid}$ & $p_\\ast^{lin}$ & BLE \\\\",
            "\\midrule",
        ]
        for row in rows:
            lines.append(
                f"{row['hidden_neurons']} & "
                f"{100.0 * row['accuracy']:.2f}\\% & "
                f"{row['type_i_count']} & "
                f"{100.0 * row['type_i_rate']:.2f}\\% & "
                f"{row['type_ii_count']} & "
                f"{100.0 * row['type_ii_rate']:.2f}\\% & "
                f"{_format_optional_float(row['p_star'], precision=3)} & "
                f"{_format_optional_float(row['p_star_interpolated'], precision=5)} & "
                f"{_format_optional_float(row['ble'], precision=5)} \\\\"
            )
        lines.extend(
            [
                "\\bottomrule",
                "\\end{tabular}",
                "}%",
                (
                    "\\caption{Noisy Bell-state benchmark for the trained classifiers. "
                    "Type-I count is FP and Type-II count is FN on the one-parameter family "
                    "$\\rho(p)=p\\ket{\\psi}\\bra{\\psi}+(1-p)I/4$, with theoretical "
                    "transition at $p=1/3$.}"
                ),
                "\\label{tab:noisy_bell_benchmark}",
                "\\end{table}",
            ]
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _write_markdown_summary(self, path: Path, rows: list[dict[str, Any]]) -> None:
        lines = [
            "# Noisy Bell-State Benchmark",
            "",
            "This benchmark is separate from the full held-out test-set evaluation. "
            "It does not measure performance on the full generated density-matrix distribution.",
            "",
            "Instead, it evaluates the already-trained classifiers on the one-parameter noisy "
            "Bell-state family from Eq. (10): `rho(p)=p|psi><psi|+(1-p)I/4`, where "
            "`|psi>=(|00>+|11>)/sqrt(2)`.",
            "",
            "The theoretical transition is `p=1/3`: states with `p<1/3` are separable and "
            "states with `p>1/3` are entangled. The PPT margin is recorded for analysis only "
            "and is not used as an ANN input feature.",
            "",
            "| Nh | Accuracy | Type-I count | Type-I rate | Type-II count | Type-II rate | p_star grid | p_star linear | BLE |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in rows:
            lines.append(
                f"| {row['hidden_neurons']} | "
                f"{row['accuracy']:.6f} | "
                f"{row['type_i_count']} | "
                f"{row['type_i_rate']:.6f} | "
                f"{row['type_ii_count']} | "
                f"{row['type_ii_rate']:.6f} | "
                f"{_format_optional_float(row['p_star'], precision=6)} | "
                f"{_format_optional_float(row['p_star_interpolated'], precision=6)} | "
                f"{_format_optional_float(row['ble'], precision=6)} |"
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _format_optional_float(value: float | None, precision: int) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _require_sklearn_scaler():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return train_test_split, StandardScaler
