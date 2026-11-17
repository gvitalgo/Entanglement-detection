from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import qutip

from asif_reproduction.ann_experiment import ANNExperiment
from asif_reproduction.basis_factory import CoherenceBasisFactory
from asif_reproduction.coherence_extractor import RelativeEntropyCoherenceExtractor
from asif_reproduction.config import BasisConfig, DatasetConfig, PathConfig, TrainingConfig
from asif_reproduction.dataset_builder import DatasetArtifact, DatasetBuilder
from asif_reproduction.ppt_labeler import PPTLabeler
from asif_reproduction.state_factory import BipartiteStateFactory


class QuantumCorrectnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labeler = PPTLabeler()
        self.bases = CoherenceBasisFactory(BasisConfig(phi=0.0)).build()
        self.extractor = RelativeEntropyCoherenceExtractor(self.bases)

    def test_generated_density_matrices_are_valid(self) -> None:
        factory = BipartiteStateFactory(DatasetConfig.uniform(samples_per_family=1, seed=19))
        for family_name, rho in factory.iter_all_states():
            matrix = rho.full()
            self.assertTrue(np.allclose(matrix, matrix.conj().T, atol=1e-12), family_name)
            self.assertGreaterEqual(np.linalg.eigvalsh(matrix).min(), -1e-12, family_name)
            self.assertAlmostEqual(float(np.trace(matrix).real), 1.0, places=12, msg=family_name)

    def test_partial_transpose_and_known_labels(self) -> None:
        zero = qutip.basis(2, 0)
        one = qutip.basis(2, 1)
        product = qutip.tensor(zero, one).proj()
        bell = ((qutip.tensor(zero, zero) + qutip.tensor(one, one)).unit()).proj()
        self.assertEqual(self.labeler.label(product), 0)
        self.assertEqual(self.labeler.label(bell), 1)
        self.assertAlmostEqual(self.labeler.minimum_partial_transpose_eigenvalue(product), 0.0, places=12)
        self.assertAlmostEqual(self.labeler.minimum_partial_transpose_eigenvalue(bell), -0.5, places=12)
        explicit = qutip.partial_transpose(bell, [0, 1]).eigenenergies().min()
        self.assertAlmostEqual(self.labeler.minimum_partial_transpose_eigenvalue(bell), float(explicit), places=12)

    def test_entropy_and_basis_diagonal_coherence(self) -> None:
        zero_zero = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0)).proj()
        maximally_mixed = qutip.qeye([2, 2]) / 4.0
        self.assertAlmostEqual(qutip.entropy_vn(zero_zero, base=np.e), 0.0, places=12)
        self.assertAlmostEqual(qutip.entropy_vn(maximally_mixed, base=np.e), np.log(4.0), places=12)
        self.assertAlmostEqual(self.extractor.relative_entropy_coherence(zero_zero, self.bases.qt), 0.0, places=12)

    def test_feature_shape_order_and_fixed_seed_determinism(self) -> None:
        self.assertEqual(self.extractor.feature_names, ("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)"))
        state = qutip.tensor(qutip.basis(2, 0), qutip.basis(2, 0)).proj()
        self.assertEqual(self.extractor.extract_features(state).shape, (4,))
        with tempfile.TemporaryDirectory() as directory:
            config = DatasetConfig.uniform(samples_per_family=2, seed=23)
            path = PathConfig(output_dir=Path(directory))
            first = DatasetBuilder(config, BasisConfig(phi=0.0), path).build()
            second = DatasetBuilder(config, BasisConfig(phi=0.0), path).build()
        np.testing.assert_array_equal(first.labels, second.labels)
        np.testing.assert_array_equal(first.family_ids, second.family_ids)
        np.testing.assert_allclose(first.features, second.features, rtol=0.0, atol=0.0)


class PreprocessingTests(unittest.TestCase):
    def test_scaler_is_fit_only_on_training_rows(self) -> None:
        rng = np.random.default_rng(91)
        features = rng.normal(size=(40, 4))
        features[-1] = 1_000_000.0
        labels = np.asarray([0, 1] * 20, dtype=np.int64)
        dataset = DatasetArtifact(
            features=features,
            labels=labels,
            family_ids=np.zeros(40, dtype=np.int64),
            family_names=("synthetic",),
            feature_names=("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)"),
            manifest={"total_samples": 40, "class_counts": {"separable": 20, "entangled": 20}},
        )
        with tempfile.TemporaryDirectory() as directory:
            experiment = ANNExperiment(
                TrainingConfig(hidden_neurons=(0,), random_state=42),
                PathConfig(output_dir=Path(directory)),
            )
            prepared = experiment.prepare_data(dataset)
        expected_training_mean = features[prepared.train_indices].mean(axis=0)
        np.testing.assert_allclose(prepared.scaler.mean_, expected_training_mean)
        if prepared.test_indices.tolist().count(39):
            self.assertFalse(np.allclose(prepared.scaler.mean_, features.mean(axis=0)))
        np.testing.assert_allclose(prepared.X_train.mean(axis=0), np.zeros(4), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
