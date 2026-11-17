"""Construct cross-label collisions for the thesis's fixed coherence features.

The search is restricted analytically to Bell-diagonal states. It is a
reproducible numerical exploration, not a claim of a historical blind search.
The exact existence proof is in thesis/sections/coherence_limitation.tex.
Only NumPy and SciPy are required; QuTiP provides an optional independent check.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import brentq

FEATURE_ORDER = ("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)")
_E = np.eye(4)
BELL_VECTORS = np.column_stack((
    _E[0] + _E[3], _E[0] - _E[3], _E[1] + _E[2], _E[1] - _E[2]
)) / np.sqrt(2)


def entropy(probabilities: np.ndarray) -> float:
    """Natural-log entropy; tolerate only roundoff-sized negative values."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Expected a finite probability vector.")
    if values.min() < -1e-12 or not np.isclose(values.sum(), 1, atol=1e-12, rtol=0):
        raise ValueError("Probabilities must be nonnegative and sum to one.")
    positive = values[values > 0]
    return float(-np.dot(positive, np.log(positive)))


def density_matrix(weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (4,):
        raise ValueError("Four ordered Bell weights are required.")
    entropy(weights)
    return (BELL_VECTORS * weights) @ BELL_VECTORS.conj().T


def partial_transpose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4):
        raise ValueError("A two-qubit 4x4 matrix is required.")
    # Matrix entries rho[a,b,c,d] become rho[a,d,c,b].
    return matrix.reshape(2, 2, 2, 2).transpose(0, 3, 2, 1).reshape(4, 4)


def product_bases(phase: float = 0.0) -> tuple[np.ndarray, ...]:
    if not np.isfinite(phase):
        raise ValueError("The fixed basis phase must be finite.")
    q = np.eye(2, dtype=complex)
    r = np.array([[1, 1], [1j*np.exp(1j*phase), -1j*np.exp(1j*phase)]]) / np.sqrt(2)
    s = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    return np.kron(q, s), np.kron(r, s), np.kron(r, q), np.kron(q, q)


def rotate_first_qubit(matrix: np.ndarray, phase: float) -> np.ndarray:
    unitary = np.kron(np.diag([1, np.exp(1j*phase)]), np.eye(2))
    return unitary @ matrix @ unitary.conj().T


def coherence_features(matrix: np.ndarray, phase: float = 0.0) -> np.ndarray:
    """Compute measurement entropies minus state entropy from full matrices."""
    matrix = np.asarray(matrix)
    if matrix.shape != (4, 4) or not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=0):
        raise ValueError("Expected a Hermitian 4x4 density matrix.")
    state_entropy = entropy(np.linalg.eigvalsh(matrix))
    return np.array([
        entropy(np.diag(basis.conj().T @ matrix @ basis).real) - state_entropy
        for basis in product_bases(phase)
    ])


@dataclass(frozen=True)
class Collision:
    pair_sum: float
    margin: float
    t: float
    bracket: tuple[float, float]
    endpoint_entropy_differences: tuple[float, float]
    entangled_weights: tuple[float, ...]
    separable_weights: tuple[float, ...]
    feature_max_abs_difference: float
    entangled_pt_min: float
    separable_pt_min: float

    def to_dict(self):
        return asdict(self)


def find_collision(pair_sum: float = 0.60, margin: float = 0.01,
                   upper: float | None = None) -> Collision | None:
    """Solve an entropy matching problem, returning None if no root is bracketed.

    qE=(1/2+d, s-1/2-d, (1-s)/2, (1-s)/2)
    qS=(1/2-d, s-1/2+d, (1-s)/2+t, (1-s)/2-t).
    Search parameters maintain full rank and opposite PPT labels throughout.
    """
    if not (np.isfinite(pair_sum) and np.isfinite(margin)
            and 0.5 < pair_sum < 1 and 0 < margin < min(pair_sum-0.5, 1-pair_sum)):
        raise ValueError("Require 1/2<s<1 and 0<d<min(s-1/2,1-s).")
    tail = (1-pair_sum)/2
    upper = 0.95*tail if upper is None else upper
    if not (np.isfinite(upper) and 0 < upper < tail):
        raise ValueError("The upper endpoint must lie strictly between zero and the tail mass.")
    entangled = np.array([0.5+margin, pair_sum-0.5-margin, tail, tail])

    def separable(t):
        return np.array([0.5-margin, pair_sum-0.5+margin, tail+t, tail-t])

    def difference(t):
        return entropy(separable(t))-entropy(entangled)

    endpoints = difference(0), difference(upper)
    if not endpoints[0] > 0 > endpoints[1]:
        return None
    root = float(brentq(difference, 0, upper, xtol=1e-15))
    sep = separable(root)
    rho_e, rho_s = density_matrix(entangled), density_matrix(sep)
    gap = float(np.max(np.abs(coherence_features(rho_e)-coherence_features(rho_s))))
    pt_e = float(np.linalg.eigvalsh(partial_transpose(rho_e)).min())
    pt_s = float(np.linalg.eigvalsh(partial_transpose(rho_s)).min())
    if not gap < 1e-12 or not pt_e < -1e-10 or not pt_s > 1e-10:
        raise ArithmeticError("The candidate failed matrix-level verification.")
    return Collision(pair_sum, margin, root, (0.0, upper), endpoints,
                     tuple(entangled), tuple(sep), gap, pt_e, pt_s)


def search_collisions() -> tuple[list[Collision], int]:
    """Deterministic grid, avoiding dependence on pseudorandom generators."""
    found = []
    attempts = 0
    for sum_percent in range(56, 81, 4):
        for margin_percent in range(1, 5):
            attempts += 1
            candidate = find_collision(sum_percent/100, margin_percent/100)
            if candidate is not None:
                found.append(candidate)
    return found, attempts


def exact_endpoint_certificate() -> dict[str, bool]:
    """Exact integer signs for the canonical entropy bracket [0,1/10]."""
    return {
        "f_zero_positive": 51**51 * 9**9 > 49**49 * 11**11,
        "f_one_tenth_negative": 51**51 * 9**9 * 20**40 < 49**49 * 11**11 * 30**30 * 10**10,
    }


def qutip_check(collision: Collision, phase: float = 0) -> dict:
    """Independently check through the thesis's original QuTiP implementation."""
    import qutip
    from .basis_factory import CoherenceBasisFactory
    from .coherence_extractor import RelativeEntropyCoherenceExtractor
    from .config import BasisConfig
    from .ppt_labeler import PPTLabeler

    extractor = RelativeEntropyCoherenceExtractor(CoherenceBasisFactory(BasisConfig(phi=phase)).build())
    labeler = PPTLabeler()
    features, labels = [], []
    for weights in (collision.entangled_weights, collision.separable_weights):
        matrix = rotate_first_qubit(density_matrix(weights), phase)
        state = qutip.Qobj(matrix, dims=[[2, 2], [2, 2]])
        values = extractor.extract_features(state)
        if not np.allclose(values, coherence_features(matrix, phase), atol=1e-12, rtol=0):
            raise ArithmeticError("Independent feature extractors disagree.")
        features.append(values)
        labels.append(labeler.label(state))
    gap = float(np.max(np.abs(features[0]-features[1])))
    if labels != [1, 0] or gap > 1e-12:
        raise ArithmeticError("Independent QuTiP check failed.")
    return {"phase": phase, "labels_entangled_then_separable": labels,
            "feature_max_abs_difference": gap}
