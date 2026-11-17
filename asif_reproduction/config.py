from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

FAMILY_NOISY_01_10 = "noisy_family_01_10"
FAMILY_NOISY_00_11 = "noisy_family_00_11"
FAMILY_RANDOM_ENTANGLED = "random_entangled"
FAMILY_RANDOM_SEPARABLE = "random_separable"
FAMILY_RANDOM_PURE_MIXED = "random_pure_mixed"
FAMILY_NOISY_PSI_PLUS = "noisy_psi_plus"
FAMILY_NOISY_PHI_PLUS = "noisy_phi_plus"

FAMILY_NAMES: tuple[str, ...] = (
    FAMILY_NOISY_01_10,
    FAMILY_NOISY_00_11,
    FAMILY_RANDOM_ENTANGLED,
    FAMILY_RANDOM_SEPARABLE,
    FAMILY_RANDOM_PURE_MIXED,
    FAMILY_NOISY_PSI_PLUS,
    FAMILY_NOISY_PHI_PLUS,
)


@dataclass(slots=True, frozen=True)
class BasisConfig:
    phi: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class DatasetConfig:
    samples_family_01_10: int
    samples_family_00_11: int
    random_entangled: int
    random_separable: int
    random_pure_mixed: int
    bell_psi_plus: int
    bell_phi_plus: int
    noise_min: float = 0.0
    noise_max: float = 1.0
    seed: int = 42

    @classmethod
    def uniform(cls, samples_per_family: int, seed: int = 42) -> "DatasetConfig":
        return cls(
            samples_family_01_10=samples_per_family,
            samples_family_00_11=samples_per_family,
            random_entangled=samples_per_family,
            random_separable=samples_per_family,
            random_pure_mixed=samples_per_family,
            bell_psi_plus=samples_per_family,
            bell_phi_plus=samples_per_family,
            seed=seed,
        )

    @classmethod
    def smoke(cls, seed: int = 42) -> "DatasetConfig":
        return cls.uniform(samples_per_family=100, seed=seed)

    @classmethod
    def full(cls, seed: int = 42) -> "DatasetConfig":
        return cls.uniform(samples_per_family=50_000, seed=seed)

    @property
    def total_samples(self) -> int:
        return sum(count for _, count in self.family_counts())

    def family_counts(self) -> tuple[tuple[str, int], ...]:
        return (
            (FAMILY_NOISY_01_10, self.samples_family_01_10),
            (FAMILY_NOISY_00_11, self.samples_family_00_11),
            (FAMILY_RANDOM_ENTANGLED, self.random_entangled),
            (FAMILY_RANDOM_SEPARABLE, self.random_separable),
            (FAMILY_RANDOM_PURE_MIXED, self.random_pure_mixed),
            (FAMILY_NOISY_PSI_PLUS, self.bell_psi_plus),
            (FAMILY_NOISY_PHI_PLUS, self.bell_phi_plus),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["total_samples"] = self.total_samples
        return payload


@dataclass(slots=True, frozen=True)
class TrainingConfig:
    hidden_neurons: tuple[int, ...] = (0, 5, 10, 50)
    epochs: int = 100
    batch_size: int = 256
    learning_rate: float = 1e-3
    test_size: float = 0.30
    validation_split: float = 0.20
    early_stopping_patience: int = 10
    random_state: int = 42
    model_seed: int | None = None

    @property
    def split_seed(self) -> int:
        """Seed used only for the stratified train/test split."""
        return self.random_state

    @property
    def effective_model_seed(self) -> int:
        """Seed used for model initialization and TensorFlow training randomness."""
        return self.random_state if self.model_seed is None else self.model_seed

    @classmethod
    def smoke(cls, random_state: int = 42) -> "TrainingConfig":
        return cls(hidden_neurons=(0, 5, 10, 50), random_state=random_state)

    @classmethod
    def full(cls, random_state: int = 42) -> "TrainingConfig":
        return cls(hidden_neurons=tuple(range(0, 55, 5)), random_state=random_state)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["split_seed"] = self.split_seed
        payload["effective_model_seed"] = self.effective_model_seed
        return payload


@dataclass(slots=True)
class PathConfig:
    output_dir: Path = field(default_factory=lambda: Path("artifacts") / "asif_entanglement")
    save_raw_states: bool = False
    raw_state_limit_per_family: int = 0

    @property
    def dataset_dir(self) -> Path:
        return self.output_dir / "datasets"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"

    @property
    def predictions_dir(self) -> Path:
        return self.output_dir / "predictions"

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    @property
    def raw_state_dir(self) -> Path:
        return self.output_dir / "raw_states"

    @property
    def dataset_npz_path(self) -> Path:
        return self.dataset_dir / "dataset.npz"

    @property
    def dataset_manifest_path(self) -> Path:
        return self.dataset_dir / "manifest.json"

    @property
    def training_summary_path(self) -> Path:
        return self.metrics_dir / "summary.json"

    @property
    def training_table_path(self) -> Path:
        return self.metrics_dir / "model_comparison.csv"

    @property
    def scaler_path(self) -> Path:
        return self.output_dir / "preprocessing" / "standard_scaler.npz"

    @property
    def split_path(self) -> Path:
        return self.output_dir / "preprocessing" / "split_indices.npz"

    @property
    def run_manifest_path(self) -> Path:
        return self.output_dir / "run_manifest.json"

    @property
    def thesis_report_path(self) -> Path:
        return self.output_dir / "THESIS_RESULTS.md"

    def ensure_directories(self) -> None:
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.scaler_path.parent.mkdir(parents=True, exist_ok=True)
        if self.save_raw_states:
            self.raw_state_dir.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "save_raw_states": self.save_raw_states,
            "raw_state_limit_per_family": self.raw_state_limit_per_family,
        }
