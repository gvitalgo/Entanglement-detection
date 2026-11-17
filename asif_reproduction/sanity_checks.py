from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

from .basis_factory import CoherenceBasisFactory
from .coherence_extractor import RelativeEntropyCoherenceExtractor
from .config import BasisConfig
from .ppt_labeler import PPTLabeler
from .state_factory import BipartiteStateFactory


class SanityChecker:
    def __init__(self, basis_config: BasisConfig, dataset_seed: int = 42) -> None:
        self.basis_config = basis_config
        self.dataset_seed = dataset_seed

    def run_quantum_checks(self) -> None:
        bases = CoherenceBasisFactory(self.basis_config).build()
        self._assert_normalized_bases(bases)
        self._assert_bell_state_definitions()
        self._assert_ppt_labels()
        self._assert_feature_shape_and_reality(bases)

    def validate_dataset_manifest(self, manifest: dict) -> None:
        expected_total = sum(int(count) for count in manifest["family_counts"].values())
        if expected_total != int(manifest["total_samples"]):
            raise AssertionError(
                f"Dataset manifest total mismatch: family sum {expected_total}, total_samples {manifest['total_samples']}."
            )

    def validate_saved_model_architecture(self, model_path: Path, hidden_neurons: int) -> None:
        with zipfile.ZipFile(model_path) as archive:
            config = json.loads(archive.read("config.json"))

        layers = config["config"]["layers"]
        layer_summary = [(layer["class_name"], layer["config"].get("units"), layer["config"].get("activation")) for layer in layers]

        if hidden_neurons == 0:
            expected = [("InputLayer", None, None), ("Dense", 1, "sigmoid")]
        else:
            expected = [("InputLayer", None, None), ("Dense", hidden_neurons, "relu"), ("Dense", 1, "sigmoid")]

        if layer_summary != expected:
            raise AssertionError(
                f"Unexpected architecture in {model_path}: expected {expected}, got {layer_summary}."
            )

    @staticmethod
    def _assert_normalized_bases(bases) -> None:
        for _, basis_states in bases.ordered_feature_bases:
            if len(basis_states) != 4:
                raise AssertionError("Each product basis must contain exactly four states.")
            for basis_state in basis_states:
                norm = float(np.real_if_close(basis_state.norm()))
                if not np.isclose(norm, 1.0, atol=1e-10):
                    raise AssertionError(f"Basis state is not normalized: norm={norm}")

    def _assert_bell_state_definitions(self) -> None:
        from .config import DatasetConfig

        factory = BipartiteStateFactory(DatasetConfig.smoke(seed=self.dataset_seed))
        psi_expected = (factory.ket_01 + factory.ket_10).unit()
        phi_expected = (factory.ket_00 + factory.ket_11).unit()

        if (factory.psi_plus - psi_expected).norm() > 1e-10:
            raise AssertionError("|psi+> does not match (|01> + |10>) / sqrt(2).")
        if (factory.phi_plus - phi_expected).norm() > 1e-10:
            raise AssertionError("|phi+> does not match (|00> + |11>) / sqrt(2).")

    def _assert_ppt_labels(self) -> None:
        from .config import DatasetConfig

        labeler = PPTLabeler()
        factory = BipartiteStateFactory(DatasetConfig.smoke(seed=self.dataset_seed), labeler=labeler)
        bell_rho = factory.psi_plus * factory.psi_plus.dag()
        separable_rho = factory.ket_00 * factory.ket_00.dag()

        if labeler.label(bell_rho) != 1:
            raise AssertionError("Bell state should be labeled as entangled.")
        if labeler.label(separable_rho) != 0:
            raise AssertionError("|00><00| should be labeled as separable.")

    @staticmethod
    def _assert_feature_shape_and_reality(bases) -> None:
        from .config import DatasetConfig

        factory = BipartiteStateFactory(DatasetConfig.smoke())
        extractor = RelativeEntropyCoherenceExtractor(bases)
        rho = factory.phi_plus * factory.phi_plus.dag()
        features = extractor.extract_features(rho)

        if features.shape != (4,):
            raise AssertionError(f"Feature vector must have shape (4,), got {features.shape}.")
        if not np.isrealobj(features):
            raise AssertionError("Feature vector must be real-valued.")
