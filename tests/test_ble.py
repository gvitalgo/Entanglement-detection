from __future__ import annotations

import unittest

import numpy as np

from asif_reproduction.ble import estimate_pathwise_ble


class PathwiseBLETests(unittest.TestCase):
    def test_linear_interpolation_recovers_known_crossing(self) -> None:
        grid = np.linspace(0.0, 1.0, 11)
        probabilities = grid + 0.13  # probability 0.5 at p=0.37
        result = estimate_pathwise_ble(
            grid,
            probabilities,
            ground_truth_transition=0.40,
            prediction_threshold=0.5,
        )
        self.assertAlmostEqual(result.interpolated_crossing, 0.37, places=14)
        self.assertAlmostEqual(result.ble_interpolated, 0.03, places=14)
        self.assertIsNotNone(result.selected_bracket)
        self.assertAlmostEqual(result.selected_bracket[0], 0.3, places=14)
        self.assertAlmostEqual(result.selected_bracket[1], 0.4, places=14)

    def test_multiple_crossings_are_counted_and_first_upward_is_selected(self) -> None:
        grid = np.arange(6, dtype=float)
        probabilities = np.asarray([0.1, 0.6, 0.4, 0.7, 0.3, 0.8])
        result = estimate_pathwise_ble(grid, probabilities, ground_truth_transition=1.0)
        self.assertEqual(result.crossing_count, 5)
        self.assertEqual(result.upward_crossing_count, 3)
        self.assertEqual(result.grid_crossing, 1.0)
        self.assertEqual(result.status, "MULTIPLE_CROSSINGS_FIRST_UPWARD_SELECTED")

    def test_no_crossing_is_explicit(self) -> None:
        grid = np.linspace(0.0, 1.0, 5)
        result = estimate_pathwise_ble(grid, np.full(5, 0.2), ground_truth_transition=1.0 / 3.0)
        self.assertEqual(result.status, "NO_UPWARD_CROSSING")
        self.assertIsNone(result.interpolated_crossing)
        self.assertIsNone(result.ble_interpolated)


if __name__ == "__main__":
    unittest.main()
