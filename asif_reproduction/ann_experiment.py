from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ._dependencies import require_tensorflow
from .config import PathConfig, TrainingConfig
from .dataset_builder import DatasetArtifact
from .plotting import ExperimentPlotter
from .reporting import ThesisReportWriter

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PreparedSplit:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    train_indices: np.ndarray
    test_indices: np.ndarray


class ANNExperiment:
    def __init__(
        self,
        training_config: TrainingConfig,
        path_config: PathConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.training_config = training_config
        self.path_config = path_config
        self.logger = logger or LOGGER
        self.plotter = ExperimentPlotter(self.path_config.plots_dir, logger=self.logger)
        self.report_writer = ThesisReportWriter(self.path_config)

    def train(self, dataset: DatasetArtifact) -> dict[str, Any]:
        self.path_config.ensure_directories()
        prepared = self.prepare_data(dataset)
        per_model_metrics: list[dict[str, Any]] = []

        for hidden_neurons in self.training_config.hidden_neurons:
            metrics = self._train_single_model(prepared, hidden_neurons)
            per_model_metrics.append(metrics)
            metrics_path = self.path_config.metrics_dir / f"per_model_nh_{hidden_neurons}.json"
            self._write_json(metrics_path, metrics)
            self.plotter.plot_model_metrics(metrics)

        summary = self._build_summary(dataset, per_model_metrics)
        self._write_json(self.path_config.training_summary_path, summary)
        self.plotter.plot_summary(summary)
        self.report_writer.write(summary)
        return summary

    def evaluate(self, dataset: DatasetArtifact) -> dict[str, Any]:
        prepared = self.prepare_data(dataset)
        evaluation = []
        for hidden_neurons in self.training_config.hidden_neurons:
            model_path = self._model_path(hidden_neurons)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            metrics = self._evaluate_saved_model(prepared, hidden_neurons)
            evaluation.append(metrics)
            metrics_path = self.path_config.metrics_dir / f"per_model_nh_{hidden_neurons}.json"
            self._write_json(metrics_path, metrics)
            self.plotter.plot_model_metrics(metrics)

        summary = self._build_summary(dataset, evaluation)
        self._write_json(self.path_config.training_summary_path, summary)
        self.plotter.plot_summary(summary)
        self.report_writer.write(summary)
        return summary

    def prepare_data(self, dataset: DatasetArtifact) -> PreparedSplit:
        train_test_split, StandardScaler = _require_sklearn_training()
        all_indices = np.arange(dataset.labels.shape[0], dtype=np.int64)
        X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
            dataset.features,
            dataset.labels,
            all_indices,
            test_size=self.training_config.test_size,
            random_state=self.training_config.split_seed,
            stratify=dataset.labels,
        )

        scaler = StandardScaler().fit(X_train)
        prepared = PreparedSplit(
            X_train=scaler.transform(X_train),
            X_test=scaler.transform(X_test),
            y_train=y_train.astype(np.float32),
            y_test=y_test.astype(np.int64),
            scaler=scaler,
            train_indices=train_indices,
            test_indices=test_indices,
        )
        self._persist_preprocessing(prepared)
        return prepared

    def build_model(self, feature_count: int, hidden_neurons: int):
        tf = require_tensorflow()
        tf.keras.utils.set_random_seed(self.training_config.effective_model_seed)

        layers = [tf.keras.layers.Input(shape=(feature_count,))]
        if hidden_neurons > 0:
            layers.append(tf.keras.layers.Dense(hidden_neurons, activation="relu"))
        layers.append(tf.keras.layers.Dense(1, activation="sigmoid"))

        model = tf.keras.Sequential(layers)
        model.compile(
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=self.training_config.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def _train_single_model(self, prepared: PreparedSplit, hidden_neurons: int) -> dict[str, Any]:
        tf = require_tensorflow()
        model_path = self._model_path(hidden_neurons)
        model = self.build_model(prepared.X_train.shape[1], hidden_neurons)

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=0,
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.training_config.early_stopping_patience,
            restore_best_weights=True,
            verbose=0,
        )

        self.logger.info("Training model for Nh=%s.", hidden_neurons)
        started_at = time.perf_counter()
        history = model.fit(
            prepared.X_train,
            prepared.y_train,
            epochs=self.training_config.epochs,
            batch_size=self.training_config.batch_size,
            validation_split=self.training_config.validation_split,
            callbacks=[checkpoint, early_stopping],
            verbose=0,
        )
        elapsed_seconds = time.perf_counter() - started_at

        best_model = tf.keras.models.load_model(model_path)
        metrics = self._evaluate_model(best_model, prepared, hidden_neurons)
        metrics["history"] = {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        }
        metrics["epochs_ran"] = len(history.history.get("loss", []))
        metrics["stopping_epoch"] = metrics["epochs_ran"]
        metrics["runtime_seconds"] = float(elapsed_seconds)
        metrics["training_loss_final"] = float(history.history["loss"][-1])
        metrics["training_loss_min"] = float(min(history.history["loss"]))
        metrics["validation_loss_final"] = float(history.history["val_loss"][-1])
        metrics["validation_loss_min"] = float(min(history.history["val_loss"]))
        metrics["model_seed"] = self.training_config.effective_model_seed
        metrics["split_seed"] = self.training_config.split_seed
        return metrics

    def _evaluate_saved_model(self, prepared: PreparedSplit, hidden_neurons: int) -> dict[str, Any]:
        tf = require_tensorflow()
        model = tf.keras.models.load_model(self._model_path(hidden_neurons))
        return self._evaluate_model(model, prepared, hidden_neurons)

    def _evaluate_model(self, model, prepared: PreparedSplit, hidden_neurons: int) -> dict[str, Any]:
        metrics_api = _require_sklearn_metrics()
        probabilities = model.predict(prepared.X_test, verbose=0).reshape(-1)
        predictions = (probabilities >= 0.5).astype(np.int64)
        accuracy = float(metrics_api["accuracy_score"](prepared.y_test, predictions))
        tn, fp, fn, tp = metrics_api["confusion_matrix"](prepared.y_test, predictions, labels=[0, 1]).ravel()
        prediction_path = self.path_config.predictions_dir / f"predictions_nh_{hidden_neurons}.npz"
        np.savez_compressed(
            prediction_path,
            test_indices=prepared.test_indices,
            y_true=prepared.y_test,
            probability_entangled=probabilities.astype(np.float64),
            y_pred=predictions,
            decision_threshold=np.asarray(0.5, dtype=np.float64),
        )

        return {
            "hidden_neurons": hidden_neurons,
            "test_accuracy": accuracy,
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "test_samples": int(prepared.y_test.shape[0]),
            "balanced_accuracy": float(metrics_api["balanced_accuracy_score"](prepared.y_test, predictions)),
            "precision": float(metrics_api["precision_score"](prepared.y_test, predictions, zero_division=0)),
            "recall": float(metrics_api["recall_score"](prepared.y_test, predictions, zero_division=0)),
            "f1": float(metrics_api["f1_score"](prepared.y_test, predictions, zero_division=0)),
            "roc_auc": float(metrics_api["roc_auc_score"](prepared.y_test, probabilities)),
            "type_i_error_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,
            "type_ii_error_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "decision_threshold": 0.5,
            "trainable_parameter_count": int(model.count_params()),
            "model_path": str(self._model_path(hidden_neurons)),
            "prediction_artifact": str(prediction_path),
        }

    def _build_summary(self, dataset: DatasetArtifact, per_model_metrics: list[dict[str, Any]]) -> dict[str, Any]:
        best_model = max(per_model_metrics, key=lambda item: item["test_accuracy"])
        return {
            "dataset_manifest": dataset.manifest,
            "training_config": self.training_config.to_dict(),
            "models": per_model_metrics,
            "best_model": {
                "hidden_neurons": best_model["hidden_neurons"],
                "test_accuracy": best_model["test_accuracy"],
                "model_path": best_model["model_path"],
            },
        }

    def _model_path(self, hidden_neurons: int) -> Path:
        return self.path_config.model_dir / f"model_nh_{hidden_neurons}.keras"

    def _persist_preprocessing(self, prepared: PreparedSplit) -> None:
        self.path_config.ensure_directories()
        np.savez_compressed(
            self.path_config.scaler_path,
            mean=prepared.scaler.mean_,
            scale=prepared.scaler.scale_,
            var=prepared.scaler.var_,
            n_samples_seen=np.asarray(prepared.scaler.n_samples_seen_),
            feature_names=np.asarray(("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)"), dtype=str),
        )
        np.savez_compressed(
            self.path_config.split_path,
            train_indices=prepared.train_indices,
            test_indices=prepared.test_indices,
            split_seed=np.asarray(self.training_config.split_seed, dtype=np.int64),
            test_size=np.asarray(self.training_config.test_size, dtype=np.float64),
        )

    @staticmethod
    def _write_json(path: Path, payload: dict[str, Any]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)


def _require_sklearn_training():
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return train_test_split, StandardScaler


def _require_sklearn_metrics():
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "confusion_matrix": confusion_matrix,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "roc_auc_score": roc_auc_score,
    }
