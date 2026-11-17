"""Pathwise boundary-location error for a sampled one-parameter state family."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class BLEResult:
    ground_truth_transition: float
    prediction_threshold: float
    grid_spacing_min: float
    grid_spacing_max: float
    crossing_selection_rule: str
    interpolation_rule: str
    crossing_count: int
    upward_crossing_count: int
    selected_bracket: tuple[float, float] | None
    grid_crossing: float | None
    interpolated_crossing: float | None
    ble_grid: float | None
    ble_interpolated: float | None
    status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def estimate_pathwise_ble(
    p_values: np.ndarray,
    probabilities: np.ndarray,
    *,
    ground_truth_transition: float,
    prediction_threshold: float = 0.5,
) -> BLEResult:
    """Estimate the first below-to-above crossing and its absolute location error.

    The samples must be ordered on a strictly increasing one-dimensional path.  All
    binary threshold transitions are counted.  If there are several upward
    crossings, the first is selected and the multiplicity is retained in the
    result.  Linear interpolation is performed inside the selected bracket.  If
    the first sample is already above threshold, that first grid location is used.
    If there is no upward crossing, all crossing and BLE values are ``None``.
    """

    grid = np.asarray(p_values, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    if grid.ndim != 1 or values.ndim != 1 or grid.shape != values.shape:
        raise ValueError("p_values and probabilities must be one-dimensional arrays of equal shape.")
    if grid.size < 2:
        raise ValueError("At least two path samples are required.")
    if not np.all(np.isfinite(grid)) or not np.all(np.isfinite(values)):
        raise ValueError("Path samples and probabilities must be finite.")
    spacing = np.diff(grid)
    if np.any(spacing <= 0.0):
        raise ValueError("p_values must be strictly increasing.")

    above = values >= prediction_threshold
    transitions = np.flatnonzero(above[1:] != above[:-1]) + 1
    upward = np.flatnonzero((~above[:-1]) & above[1:]) + 1

    if above[0]:
        selected_bracket = (float(grid[0]), float(grid[0]))
        grid_crossing = float(grid[0])
        interpolated_crossing = grid_crossing
        status = "INITIAL_SAMPLE_ABOVE_THRESHOLD"
    elif upward.size:
        right_index = int(upward[0])
        left_index = right_index - 1
        left_p = float(grid[left_index])
        right_p = float(grid[right_index])
        left_probability = float(values[left_index])
        right_probability = float(values[right_index])
        selected_bracket = (left_p, right_p)
        grid_crossing = right_p
        if right_probability == left_probability:
            interpolated_crossing = right_p
            status = "FLAT_BRACKET_USED_RIGHT_ENDPOINT"
        else:
            fraction = (prediction_threshold - left_probability) / (right_probability - left_probability)
            interpolated_crossing = float(left_p + fraction * (right_p - left_p))
            status = "MULTIPLE_CROSSINGS_FIRST_UPWARD_SELECTED" if upward.size > 1 else "OK"
    else:
        selected_bracket = None
        grid_crossing = None
        interpolated_crossing = None
        status = "NO_UPWARD_CROSSING"

    return BLEResult(
        ground_truth_transition=float(ground_truth_transition),
        prediction_threshold=float(prediction_threshold),
        grid_spacing_min=float(spacing.min()),
        grid_spacing_max=float(spacing.max()),
        crossing_selection_rule="first below-to-above threshold crossing along the ordered path",
        interpolation_rule="linear interpolation of probability within selected grid bracket",
        crossing_count=int(transitions.size),
        upward_crossing_count=int(upward.size) + int(above[0]),
        selected_bracket=selected_bracket,
        grid_crossing=grid_crossing,
        interpolated_crossing=interpolated_crossing,
        ble_grid=None if grid_crossing is None else abs(grid_crossing - ground_truth_transition),
        ble_interpolated=(
            None if interpolated_crossing is None else abs(interpolated_crossing - ground_truth_transition)
        ),
        status=status,
    )
