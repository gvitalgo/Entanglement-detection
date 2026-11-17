import unittest
import numpy as np

from asif_reproduction.counterexample import (
    coherence_features, density_matrix, entropy, exact_endpoint_certificate,
    find_collision, partial_transpose, qutip_check, rotate_first_qubit, search_collisions,
)


class CounterexampleTests(unittest.TestCase):
    def test_canonical_opposite_labels_and_equal_features(self):
        pair = find_collision(0.60, 0.01, upper=0.1)
        self.assertIsNotNone(pair)
        self.assertTrue(all(exact_endpoint_certificate().values()))
        self.assertAlmostEqual(pair.t, 0.079178202144160, places=13)
        matrices = [density_matrix(w) for w in (pair.entangled_weights, pair.separable_weights)]
        for rho in matrices:
            self.assertGreater(np.linalg.eigvalsh(rho).min(), 0)
            np.testing.assert_allclose(partial_transpose(partial_transpose(rho)), rho, atol=1e-14)
        self.assertAlmostEqual(np.linalg.eigvalsh(partial_transpose(matrices[0])).min(), -.01, places=13)
        self.assertAlmostEqual(np.linalg.eigvalsh(partial_transpose(matrices[1])).min(), .01, places=13)
        np.testing.assert_allclose(coherence_features(matrices[0]), coherence_features(matrices[1]), atol=1e-13, rtol=0)

    def test_matrix_calculation_matches_reduced_formula(self):
        for weights in np.random.default_rng(7).dirichlet(np.ones(4), size=25):
            s = weights[0]+weights[1]
            h = entropy(weights)
            expected = [np.log(4)-h]*3 + [entropy(np.array([s/2, (1-s)/2, (1-s)/2, s/2]))-h]
            np.testing.assert_allclose(coherence_features(density_matrix(weights)), expected, atol=1e-13, rtol=0)

    def test_every_fixed_phase_has_rotated_collision(self):
        pair = find_collision(upper=.1)
        for phase in [0, .37, np.pi/2, np.pi, -1.21]:
            features = []
            for weights in [pair.entangled_weights, pair.separable_weights]:
                original = density_matrix(weights)
                rotated = rotate_first_qubit(original, phase)
                np.testing.assert_allclose(np.linalg.eigvalsh(partial_transpose(rotated)),
                                           np.linalg.eigvalsh(partial_transpose(original)), atol=1e-13)
                features.append(coherence_features(rotated, phase))
            np.testing.assert_allclose(*features, atol=1e-13, rtol=0)

    def test_search_finds_distinct_pairs_and_handles_missing_bracket(self):
        pairs, attempts = search_collisions()
        self.assertEqual(attempts, 28)
        self.assertGreater(len(pairs), 1)
        self.assertGreater(len({round(pair.pair_sum, 8) for pair in pairs}), 1)
        self.assertIsNone(find_collision(upper=1e-4))
        for pair in pairs:
            self.assertLess(pair.entangled_pt_min, -1e-10)
            self.assertGreater(pair.separable_pt_min, 1e-10)
            self.assertLess(pair.feature_max_abs_difference, 1e-12)
        with self.assertRaises(ValueError):
            find_collision(pair_sum=.4)

    def test_independent_qutip_extractor(self):
        try:
            import qutip
        except ImportError:
            self.skipTest('Install thesis requirements for the independent QuTiP check.')
        self.assertEqual(qutip_check(find_collision(upper=.1), .37)['labels_entangled_then_separable'], [1, 0])


if __name__ == '__main__':
    unittest.main()
