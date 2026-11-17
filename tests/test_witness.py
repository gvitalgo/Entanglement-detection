from __future__ import annotations

import unittest

import numpy as np

from asif_reproduction.witness import FEATURE_NAMES, witness_predict, witness_score_from_features


class WitnessTests(unittest.TestCase):
    def test_feature_order_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            witness_score_from_features(np.zeros((1, 4)), feature_names=tuple(reversed(FEATURE_NAMES)))

    def test_natural_log_features_are_converted_to_bits(self) -> None:
        feature_bits = np.asarray([[2.0, 2.0, 2.0, 1.0]])
        feature_nats = feature_bits * np.log(2.0)
        self.assertAlmostEqual(float(witness_score_from_features(feature_nats)[0]), 5.0, places=12)
        self.assertEqual(int(witness_predict(feature_nats)[0]), 1)

    def test_threshold_is_strict(self) -> None:
        feature_bits = np.asarray([[2.0, 1.0, 1.0, 0.0]])
        self.assertEqual(int(witness_predict(feature_bits, input_log_base="2")[0]), 0)


if __name__ == "__main__":
    unittest.main()
