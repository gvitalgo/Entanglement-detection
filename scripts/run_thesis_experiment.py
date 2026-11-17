#!/usr/bin/env python3
"""Run the fixed thesis experiment and write self-contained review artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import sys
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Override experiment.output_dir without changing scientific parameters.")
    parser.add_argument("--skip-robustness", action="store_true", help="Run only the canonical sweep; robustness can be run separately.")
    return parser.parse_args()


def resolve_from_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def classification_metrics(y_true: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray | None = None) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    payload: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "type_i_error_rate": float(fp / (tn + fp)) if tn + fp else 0.0,
        "type_ii_error_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if probabilities is not None:
        payload["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    return payload


def copy_verified_dataset(config: dict[str, Any], output_dir: Path):
    from asif_reproduction.config import PathConfig
    from asif_reproduction.dataset_repository import DatasetRepository
    from asif_reproduction.provenance import dataset_logical_fingerprint, sha256_file

    experiment = config["experiment"]
    if experiment["dataset_mode"] != "verified_artifact":
        raise ValueError("The finalization configuration supports only dataset_mode='verified_artifact'.")
    source_dir = resolve_from_root(experiment["dataset_source_dir"])
    source_npz = source_dir / "dataset.npz"
    source_manifest = source_dir / "manifest.json"
    actual_sha256 = sha256_file(source_npz)
    expected_sha256 = experiment["expected_dataset_sha256"]
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"Canonical dataset SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}.")

    path_config = PathConfig(output_dir=output_dir)
    path_config.ensure_directories()
    shutil.copy2(source_npz, path_config.dataset_npz_path)
    shutil.copy2(source_manifest, path_config.dataset_manifest_path)
    with path_config.dataset_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["finalization_dataset_source"] = {
        "mode": "verified_artifact",
        "source_path": str(source_npz),
        "source_sha256": actual_sha256,
        "copied_artifact_sha256": sha256_file(path_config.dataset_npz_path),
        "logical_fingerprint_sha256": dataset_logical_fingerprint(path_config.dataset_npz_path),
    }
    write_json(path_config.dataset_manifest_path, manifest)
    dataset = DatasetRepository(path_config).load()
    expected_order = tuple(config["basis"]["feature_order"])
    if dataset.feature_names != expected_order:
        raise RuntimeError(f"Feature order mismatch: expected {expected_order}, got {dataset.feature_names}.")
    return path_config, dataset


def build_training_config(config: dict[str, Any], widths: tuple[int, ...], model_seed: int):
    from asif_reproduction.config import TrainingConfig

    training = config["training"]
    return TrainingConfig(
        hidden_neurons=widths,
        epochs=int(training["epochs"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        test_size=float(training["test_size"]),
        validation_split=float(training["validation_split"]),
        early_stopping_patience=int(training["early_stopping_patience"]),
        random_state=int(training["split_seed"]),
        model_seed=model_seed,
    )


def evaluate_witness(dataset, path_config, config: dict[str, Any]) -> dict[str, Any]:
    from asif_reproduction.witness import (
        SOURCE_CITATION,
        WITNESS_THRESHOLD_BITS,
        witness_predict,
        witness_score_from_features,
    )

    split = np.load(path_config.split_path)
    test_indices = split["test_indices"]
    features = dataset.features[test_indices]
    labels = dataset.labels[test_indices]
    feature_names = tuple(dataset.feature_names)
    scores = witness_score_from_features(features, feature_names=feature_names, input_log_base="e")
    predictions = witness_predict(features, feature_names=feature_names, input_log_base="e")
    metrics = classification_metrics(labels, predictions, probabilities=None)
    metrics.update(
        {
            "status": "REPRODUCED WITH EXPLICIT NATURAL-LOG-TO-BIT CONVERSION",
            "formula": "Cr(QS)+Cr(RS)+Cr(RT)-Cr(QT) > 4",
            "threshold": WITNESS_THRESHOLD_BITS,
            "threshold_convention": "strictly greater than 4 labels entangled",
            "input_feature_log_base": "e",
            "witness_log_base": "2",
            "conversion": "divide each natural-log coherence feature by ln(2)",
            "feature_order": list(feature_names),
            "citation": SOURCE_CITATION,
            "scope_caveat": "Proposition 5.1 proves the bound for all separable two-qubit states. A score at or below four bits is inconclusive.",
        }
    )
    artifact = path_config.predictions_dir / "witness_predictions.npz"
    np.savez_compressed(
        artifact,
        test_indices=test_indices,
        y_true=labels,
        witness_score_bits=scores,
        y_pred=predictions,
        threshold=np.asarray(WITNESS_THRESHOLD_BITS),
    )
    metrics["prediction_artifact"] = str(artifact)
    write_json(path_config.metrics_dir / "witness_baseline.json", metrics)
    return metrics


def compare_with_thesis(summary: dict[str, Any], benchmark: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    expected_path = REPOSITORY_ROOT / "configs" / "thesis_reported_metrics.json"
    with expected_path.open("r", encoding="utf-8") as handle:
        expected = json.load(handle)
    rows: list[dict[str, Any]] = []

    def add(metric: str, thesis_value: float, reproduced_value: float, explanation: str = "same fixed artifact and protocol") -> None:
        difference = abs(float(thesis_value) - float(reproduced_value))
        tolerance = 5e-10 if isinstance(thesis_value, int) else 5e-7
        rows.append(
            {
                "metric": metric,
                "thesis_value": thesis_value,
                "reproduced_value": reproduced_value,
                "absolute_difference": difference,
                "explanation": explanation if difference <= tolerance else "retrained model differs from recorded thesis artifact",
                "status": "MATCH" if difference <= tolerance else "DISCREPANCY",
            }
        )

    manifest = summary["dataset_manifest"]
    add("dataset.total_samples", expected["dataset"]["total_samples"], manifest["total_samples"])
    add("dataset.separable", expected["dataset"]["separable"], manifest["class_counts"]["separable"])
    add("dataset.entangled", expected["dataset"]["entangled"], manifest["class_counts"]["entangled"])
    add("split.train_samples", expected["dataset"]["train_samples"], 245000)
    add("split.test_samples", expected["dataset"]["test_samples"], summary["models"][0]["test_samples"])
    model_map = {str(model["hidden_neurons"]): model for model in summary["models"]}
    for width, thesis_model in expected["models"].items():
        if width not in model_map:
            continue
        reproduced = model_map[width]
        add(f"Nh={width}.accuracy", thesis_model["accuracy"], reproduced["test_accuracy"])
        for cell in ("tn", "fp", "fn", "tp"):
            add(f"Nh={width}.{cell}", thesis_model[cell], reproduced["confusion_matrix"][cell])
    for width, thesis_ble in expected["ble_grid"].items():
        if width not in benchmark["models"]:
            continue
        add(f"Nh={width}.BLE_grid", thesis_ble, benchmark["models"][width]["ble_details"]["ble_grid"])

    metrics_dir = output_dir / "metrics"
    write_json(metrics_dir / "thesis_comparison.json", {"rows": rows, "source": str(expected_path)})
    with (metrics_dir / "thesis_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    latex = [
        "% Automatically generated; do not edit by hand.",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Metric & Thesis & Reproduced & Abs. diff. & Status \\\\",
        "\\midrule",
    ]
    for row in rows:
        escaped_metric = row["metric"].replace("_", "\\_")
        latex.append(
            f"{escaped_metric} & {row['thesis_value']} & {row['reproduced_value']} & "
            f"{row['absolute_difference']:.3g} & {row['status']} \\\\"
        )
    latex.extend(["\\bottomrule", "\\end{tabular}"])
    (metrics_dir / "thesis_comparison_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return rows


def run_robustness(config: dict[str, Any], output_dir: Path, dataset, canonical_summary: dict[str, Any], canonical_benchmark: dict[str, Any]) -> dict[str, Any]:
    from asif_reproduction.ann_experiment import ANNExperiment
    from asif_reproduction.config import BasisConfig, PathConfig
    from asif_reproduction.noisy_bell_benchmark import NoisyBellBenchmarkRunner

    robustness = config["robustness"]
    widths = tuple(int(value) for value in robustness["architectures"])
    seeds = tuple(int(value) for value in robustness["model_seeds"])
    rows: list[dict[str, Any]] = []
    canonical_models = {model["hidden_neurons"]: model for model in canonical_summary["models"]}

    for seed in seeds:
        if seed == int(config["training"]["model_seed"]):
            seed_models = {width: canonical_models[width] for width in widths}
            seed_ble = canonical_benchmark["models"]
        else:
            seed_dir = output_dir / "robustness" / f"model_seed_{seed}"
            seed_path_config = PathConfig(output_dir=seed_dir)
            seed_path_config.ensure_directories()
            shutil.copy2(output_dir / "datasets" / "dataset.npz", seed_path_config.dataset_npz_path)
            shutil.copy2(output_dir / "datasets" / "manifest.json", seed_path_config.dataset_manifest_path)
            training_config = build_training_config(config, widths, seed)
            seed_summary = ANNExperiment(training_config, seed_path_config).train(dataset)
            seed_models = {model["hidden_neurons"]: model for model in seed_summary["models"]}
            seed_benchmark = NoisyBellBenchmarkRunner(
                path_config=seed_path_config,
                basis_config=BasisConfig(phi=float(config["basis"]["phi"])),
                architectures=widths,
                num_points=int(config["ble"]["num_points"]),
            ).run()
            seed_ble = seed_benchmark["models"]
        for width in widths:
            model = seed_models[width]
            rows.append(
                {
                    "model_seed": seed,
                    "dataset_seed": int(robustness["dataset_seed"]),
                    "split_seed": int(robustness["split_seed"]),
                    "hidden_neurons": width,
                    "accuracy": model["test_accuracy"],
                    "balanced_accuracy": model["balanced_accuracy"],
                    "type_i_error_rate": model["type_i_error_rate"],
                    "type_ii_error_rate": model["type_ii_error_rate"],
                    "ble": seed_ble[str(width)]["ble"],
                    "runtime_seconds": model.get("runtime_seconds"),
                    "stopping_epoch": model.get("stopping_epoch"),
                }
            )

    aggregate: dict[str, Any] = {}
    metric_names = ("accuracy", "balanced_accuracy", "type_i_error_rate", "type_ii_error_rate", "ble")
    for width in widths:
        width_rows = [row for row in rows if row["hidden_neurons"] == width]
        aggregate[str(width)] = {}
        for metric in metric_names:
            values = [float(row[metric]) for row in width_rows]
            aggregate[str(width)][metric] = {
                "mean": statistics.fmean(values),
                "sample_standard_deviation": statistics.stdev(values),
                "values": values,
            }

    metrics_dir = output_dir / "metrics"
    with (metrics_dir / "seed_robustness.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "policy": {
            "dataset_seed": robustness["dataset_seed"],
            "split_seed": robustness["split_seed"],
            "model_seeds": list(seeds),
            "varied_randomness": robustness["varied_randomness"],
            "not_varied": ["generated dataset", "train/test split", "preprocessing fit", "hyperparameters"],
        },
        "rows": rows,
        "aggregate": aggregate,
    }
    write_json(metrics_dir / "seed_robustness.json", payload)
    latex = [
        "% Automatically generated; do not edit by hand.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "$N_h$ & Accuracy & Bal. acc. & Type I & Type II & Pathwise BLE \\\\",
        "\\midrule",
    ]
    for width in widths:
        values = aggregate[str(width)]
        latex.append(
            f"{width} & "
            f"{100.0 * values['accuracy']['mean']:.2f} $\\pm$ {100.0 * values['accuracy']['sample_standard_deviation']:.2f} & "
            f"{100.0 * values['balanced_accuracy']['mean']:.2f} $\\pm$ {100.0 * values['balanced_accuracy']['sample_standard_deviation']:.2f} & "
            f"{100.0 * values['type_i_error_rate']['mean']:.2f} $\\pm$ {100.0 * values['type_i_error_rate']['sample_standard_deviation']:.2f} & "
            f"{100.0 * values['type_ii_error_rate']['mean']:.2f} $\\pm$ {100.0 * values['type_ii_error_rate']['sample_standard_deviation']:.2f} & "
            f"{values['ble']['mean']:.5f} $\\pm$ {values['ble']['sample_standard_deviation']:.5f} \\\\"
        )
    latex.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "}%",
            "\\caption{Fixed-dataset, fixed-split robustness across three model-training seeds. Values are mean $\\pm$ sample standard deviation.}",
            "\\label{tab:seed-robustness}",
            "\\end{table}",
        ]
    )
    (metrics_dir / "seed_robustness_table.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    args = parse_args()
    config_path = resolve_from_root(args.config)
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    output_dir = resolve_from_root(args.output_dir or config["experiment"]["output_dir"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib-cache"))

    from asif_reproduction.ann_experiment import ANNExperiment
    from asif_reproduction.config import BasisConfig
    from asif_reproduction.noisy_bell_benchmark import NoisyBellBenchmarkRunner
    from asif_reproduction.provenance import (
        collect_environment,
        collect_git_state,
        dataset_logical_fingerprint,
        sha256_file,
    )

    overall_started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    shutil.copy2(config_path, output_dir / "experiment_config.toml")
    path_config, dataset = copy_verified_dataset(config, output_dir)
    widths = tuple(int(value) for value in config["training"]["hidden_widths"])
    training_config = build_training_config(config, widths, int(config["training"]["model_seed"]))

    training_started = time.perf_counter()
    summary = ANNExperiment(training_config, path_config).train(dataset)
    training_seconds = time.perf_counter() - training_started

    benchmark_started = time.perf_counter()
    benchmark = NoisyBellBenchmarkRunner(
        path_config=path_config,
        basis_config=BasisConfig(phi=float(config["basis"]["phi"])),
        architectures=tuple(int(value) for value in config["ble"]["architectures"]),
        num_points=int(config["ble"]["num_points"]),
    ).run()
    benchmark_seconds = time.perf_counter() - benchmark_started

    witness = evaluate_witness(dataset, path_config, config) if config["witness"]["enabled"] else {"status": "DISABLED"}
    comparison = compare_with_thesis(summary, benchmark, output_dir)
    robustness = None
    if config["robustness"]["enabled"] and not args.skip_robustness:
        robustness = run_robustness(config, output_dir, dataset, summary, benchmark)

    elapsed_seconds = time.perf_counter() - overall_started
    manifest = {
        "schema_version": 1,
        "experiment_name": config["experiment"]["name"],
        "started_at_utc": started_at,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "stage_runtime_seconds": {"training": training_seconds, "ble_benchmark": benchmark_seconds},
        "configuration": config,
        "configuration_file": str(config_path),
        "environment": collect_environment(),
        "git": collect_git_state(REPOSITORY_ROOT),
        "dataset": {
            "path": str(path_config.dataset_npz_path),
            "sha256": sha256_file(path_config.dataset_npz_path),
            "logical_fingerprint_sha256": dataset_logical_fingerprint(path_config.dataset_npz_path),
            "feature_order": list(dataset.feature_names),
            "shape": list(dataset.features.shape),
            "class_counts": dataset.manifest["class_counts"],
        },
        "training_summary": str(path_config.training_summary_path),
        "ble_summary": str(path_config.metrics_dir / "noisy_bell_benchmark.json"),
        "witness_summary": str(path_config.metrics_dir / "witness_baseline.json"),
        "thesis_comparison": str(path_config.metrics_dir / "thesis_comparison.json"),
        "seed_robustness": None if robustness is None else str(path_config.metrics_dir / "seed_robustness.json"),
        "comparison_status_counts": {
            status: sum(row["status"] == status for row in comparison) for status in ("MATCH", "DISCREPANCY")
        },
        "witness_status": witness["status"],
    }
    write_json(path_config.run_manifest_path, manifest)
    print(json.dumps({"output_dir": str(output_dir), "elapsed_seconds": elapsed_seconds}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
