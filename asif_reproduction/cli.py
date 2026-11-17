from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import BasisConfig, DatasetConfig, PathConfig, TrainingConfig
from .experiment_runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m asif_reproduction",
        description="Paper-faithful Python/QuTiP reproduction package for bipartite entanglement detection.",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--output-dir", type=Path, default=Path("artifacts") / "asif_entanglement")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--phi", type=float, default=0.0)
    common.add_argument("--full-dataset", action="store_true")
    common.add_argument("--samples-per-family", type=int)
    common.add_argument("--architectures", type=str, help="Comma-separated list, for example: 0,5,10,50")
    common.add_argument("--epochs", type=int, default=100)
    common.add_argument("--batch-size", type=int, default=256)
    common.add_argument("--save-raw-states", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in ("generate", "train", "evaluate", "run", "smoke"):
        subparsers.add_parser(command_name, parents=[common])

    benchmark = subparsers.add_parser(
        "noisy-bell-benchmark",
        help="Evaluate saved ANN models on the Eq. (10) noisy Bell-state benchmark.",
    )
    benchmark.add_argument("--artifact-dir", type=Path, required=True)
    benchmark.add_argument("--num-points", type=int, default=1001)
    benchmark.add_argument("--architectures", type=str, default="0,10,50")
    benchmark.add_argument("--phi", type=float, default=0.0)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    if args.command == "noisy-bell-benchmark":
        from .noisy_bell_benchmark import NoisyBellBenchmarkRunner

        summary = NoisyBellBenchmarkRunner(
            path_config=PathConfig(output_dir=args.artifact_dir),
            basis_config=BasisConfig(phi=args.phi),
            architectures=_parse_architectures(args.architectures),
            num_points=args.num_points,
        ).run()
        for hidden_neurons, metrics in summary["models"].items():
            logging.getLogger(__name__).info(
                "Noisy Bell benchmark Nh=%s: accuracy=%.4f, FP=%s, FN=%s, p*=%.6s",
                hidden_neurons,
                metrics["accuracy"],
                metrics["confusion_matrix"]["fp"],
                metrics["confusion_matrix"]["fn"],
                metrics["transition_estimate"],
            )
        return 0

    runner = ExperimentRunner(
        basis_config=BasisConfig(phi=args.phi),
        dataset_config=_dataset_config_from_args(args),
        training_config=_training_config_from_args(args),
        path_config=PathConfig(output_dir=args.output_dir, save_raw_states=args.save_raw_states),
    )

    if args.command == "generate":
        runner.generate()
    elif args.command == "train":
        runner.train()
    elif args.command == "evaluate":
        runner.evaluate()
    elif args.command == "run":
        runner.run()
    elif args.command == "smoke":
        runner.smoke()
    else:  # pragma: no cover - argparse enforces valid values
        parser.error(f"Unsupported command: {args.command}")
    return 0


def _dataset_config_from_args(args: argparse.Namespace) -> DatasetConfig:
    if args.command == "smoke":
        base_config = DatasetConfig.smoke(seed=args.seed)
    elif args.full_dataset:
        base_config = DatasetConfig.full(seed=args.seed)
    elif args.samples_per_family is not None:
        base_config = DatasetConfig.uniform(samples_per_family=args.samples_per_family, seed=args.seed)
    else:
        base_config = DatasetConfig.smoke(seed=args.seed)

    return DatasetConfig(
        samples_family_01_10=base_config.samples_family_01_10,
        samples_family_00_11=base_config.samples_family_00_11,
        random_entangled=base_config.random_entangled,
        random_separable=base_config.random_separable,
        random_pure_mixed=base_config.random_pure_mixed,
        bell_psi_plus=base_config.bell_psi_plus,
        bell_phi_plus=base_config.bell_phi_plus,
        noise_min=base_config.noise_min,
        noise_max=base_config.noise_max,
        seed=args.seed,
    )


def _training_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    if args.command == "smoke":
        base_config = TrainingConfig.smoke(random_state=args.seed)
    elif args.full_dataset:
        base_config = TrainingConfig.full(random_state=args.seed)
    else:
        base_config = TrainingConfig.smoke(random_state=args.seed)

    hidden_neurons = _parse_architectures(args.architectures) if args.architectures else base_config.hidden_neurons
    return TrainingConfig(
        hidden_neurons=hidden_neurons,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=base_config.learning_rate,
        test_size=base_config.test_size,
        validation_split=base_config.validation_split,
        early_stopping_patience=base_config.early_stopping_patience,
        random_state=args.seed,
    )


def _parse_architectures(raw: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in raw.split(",") if value.strip())
    if not values:
        raise ValueError("At least one architecture must be provided.")
    if any(value < 0 for value in values):
        raise ValueError("Hidden neuron counts must be non-negative.")
    return values
