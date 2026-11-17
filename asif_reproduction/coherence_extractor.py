from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._dependencies import require_qutip
from .basis_factory import CoherenceBases
from .ppt_labeler import PPTLabeler

if TYPE_CHECKING:  # pragma: no cover
    from qutip import Qobj


class RelativeEntropyCoherenceExtractor:
    def __init__(self, bases: CoherenceBases) -> None:
        self.bases = bases

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.bases.feature_names

    def extract_features(self, rho: "Qobj") -> np.ndarray:
        density_matrix = PPTLabeler.ensure_two_qubit_density_matrix(rho)
        features = [
            self.relative_entropy_coherence(density_matrix, basis_states)
            for _, basis_states in self.bases.ordered_feature_bases
        ]
        return np.asarray(features, dtype=np.float64)

    def relative_entropy_coherence(self, rho: "Qobj", basis_states: tuple["Qobj", ...]) -> float:
        qutip = require_qutip()
        rho_diagonal = self.diagonal_state_in_basis(rho, basis_states)
        diagonal_entropy = qutip.entropy_vn(rho_diagonal, base=np.e)
        von_neumann_entropy = qutip.entropy_vn(rho, base=np.e)
        return float(np.real_if_close(diagonal_entropy - von_neumann_entropy))

    def diagonal_state_in_basis(self, rho: "Qobj", basis_states: tuple["Qobj", ...]) -> "Qobj":
        diagonal_state = None
        for basis_state in basis_states:
            overlap = basis_state.dag() * rho * basis_state
            amplitude = complex(overlap)
            probability = float(np.real_if_close(amplitude))
            projector = basis_state * basis_state.dag()
            component = probability * projector
            diagonal_state = component if diagonal_state is None else diagonal_state + component
        if diagonal_state is None:
            raise ValueError("A non-empty basis is required to diagonalize the state.")
        return diagonal_state
