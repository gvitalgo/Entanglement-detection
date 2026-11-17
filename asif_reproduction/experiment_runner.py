from __future__ import annotations

import logging
from typing import Any

from .ann_experiment import ANNExperiment
from .config import BasisConfig, DatasetConfig, PathConfig, TrainingConfig
from .dataset_builder import DatasetArtifact, DatasetBuilder
from .dataset_repository import DatasetRepository
from .sanity_checks import SanityChecker

LOGGER = logging.getLogger(__name__)


class ExperimentRunner:
    def __init__(
        self,
        basis_config: BasisConfig,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        path_config: PathConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self.basis_config = basis_config
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.path_config = path_config
        self.logger = logger or LOGGER
        self.repository = DatasetRepository(path_config)
        self.sanity_checker = SanityChecker(basis_config=basis_config, dataset_seed=dataset_config.seed)

    def generate(self) -> DatasetArtifact:
        self.sanity_checker.run_quantum_checks()
        self.logger.info("Generating dataset with %s total samples.", self.dataset_config.total_samples)
        builder = DatasetBuilder(
            dataset_config=self.dataset_config,
            basis_config=self.basis_config,
            path_config=self.path_config,
            logger=self.logger,
        )
        dataset = builder.build()
        self.sanity_checker.validate_dataset_manifest(dataset.manifest)
        self.repository.save(dataset)
        self.logger.info("Saved dataset to %s.", self.path_config.dataset_dir)
        return dataset

    def train(self, dataset: DatasetArtifact | None = None) -> dict[str, Any]:
        dataset = dataset or self.repository.load()
        summary = ANNExperiment(self.training_config, self.path_config, logger=self.logger).train(dataset)
        for hidden_neurons in self.training_config.hidden_neurons:
            self.sanity_checker.validate_saved_model_architecture(
                self.path_config.model_dir / f"model_nh_{hidden_neurons}.keras",
                hidden_neurons=hidden_neurons,
            )
        self._log_summary(summary)
        return summary

    def evaluate(self, dataset: DatasetArtifact | None = None) -> dict[str, Any]:
        dataset = dataset or self.repository.load()
        summary = ANNExperiment(self.training_config, self.path_config, logger=self.logger).evaluate(dataset)
        for hidden_neurons in self.training_config.hidden_neurons:
            self.sanity_checker.validate_saved_model_architecture(
                self.path_config.model_dir / f"model_nh_{hidden_neurons}.keras",
                hidden_neurons=hidden_neurons,
            )
        self._log_summary(summary)
        return summary

    def run(self) -> dict[str, Any]:
        dataset = self.generate()
        return self.train(dataset)

    def smoke(self) -> dict[str, Any]:
        return self.run()

    def _log_summary(self, summary: dict[str, Any]) -> None:
        best_model = summary["best_model"]
        dataset_manifest = summary["dataset_manifest"]
        self.logger.info(
            "Run summary: total_samples=%s, entangled=%s, separable=%s, best_nh=%s, best_accuracy=%.4f",
            dataset_manifest["total_samples"],
            dataset_manifest["class_counts"]["entangled"],
            dataset_manifest["class_counts"]["separable"],
            best_model["hidden_neurons"],
            best_model["test_accuracy"],
        )
