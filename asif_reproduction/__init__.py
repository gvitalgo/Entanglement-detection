from __future__ import annotations

from importlib import import_module

__all__ = [
    "ANNExperiment",
    "BasisConfig",
    "BipartiteStateFactory",
    "CoherenceBasisFactory",
    "DatasetBuilder",
    "DatasetConfig",
    "DatasetRepository",
    "ExperimentRunner",
    "ExperimentPlotter",
    "PPTLabeler",
    "PathConfig",
    "RelativeEntropyCoherenceExtractor",
    "SanityChecker",
    "TrainingConfig",
    "NoisyBellBenchmarkRunner",
    "generate_noisy_bell_grid",
    "noisy_bell_state",
    "BLEResult",
    "estimate_pathwise_ble",
    "witness_predict",
    "witness_score_from_features",
]

_EXPORT_MAP = {
    "ANNExperiment": "asif_reproduction.ann_experiment",
    "BasisConfig": "asif_reproduction.config",
    "BipartiteStateFactory": "asif_reproduction.state_factory",
    "CoherenceBasisFactory": "asif_reproduction.basis_factory",
    "DatasetBuilder": "asif_reproduction.dataset_builder",
    "DatasetConfig": "asif_reproduction.config",
    "DatasetRepository": "asif_reproduction.dataset_repository",
    "ExperimentRunner": "asif_reproduction.experiment_runner",
    "ExperimentPlotter": "asif_reproduction.plotting",
    "PPTLabeler": "asif_reproduction.ppt_labeler",
    "PathConfig": "asif_reproduction.config",
    "RelativeEntropyCoherenceExtractor": "asif_reproduction.coherence_extractor",
    "SanityChecker": "asif_reproduction.sanity_checks",
    "TrainingConfig": "asif_reproduction.config",
    "NoisyBellBenchmarkRunner": "asif_reproduction.noisy_bell_benchmark",
    "generate_noisy_bell_grid": "asif_reproduction.noisy_bell_benchmark",
    "noisy_bell_state": "asif_reproduction.noisy_bell_benchmark",
    "BLEResult": "asif_reproduction.ble",
    "estimate_pathwise_ble": "asif_reproduction.ble",
    "witness_predict": "asif_reproduction.witness",
    "witness_score_from_features": "asif_reproduction.witness",
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MAP[name])
    return getattr(module, name)
