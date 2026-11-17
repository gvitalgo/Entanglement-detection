from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._dependencies import require_qutip

if TYPE_CHECKING:  # pragma: no cover
    from qutip import Qobj


class PPTLabeler:
    def __init__(self, tolerance: float = 1e-10) -> None:
        self.tolerance = tolerance

    def label(self, rho: "Qobj") -> int:
        return int(self.is_entangled(rho))

    def is_entangled(self, rho: "Qobj") -> bool:
        return self.minimum_partial_transpose_eigenvalue(rho) < -self.tolerance

    def minimum_partial_transpose_eigenvalue(self, rho: "Qobj") -> float:
        qutip = require_qutip()
        density_matrix = self.ensure_two_qubit_density_matrix(rho)
        rho_tb = qutip.partial_transpose(density_matrix, [0, 1])
        eigenvalues = rho_tb.eigenenergies()
        return float(np.min(np.real_if_close(eigenvalues)))

    @staticmethod
    def ensure_two_qubit_density_matrix(rho: "Qobj") -> "Qobj":
        qutip = require_qutip()
        if rho.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 density matrix, got shape {rho.shape}.")
        if rho.dims != [[2, 2], [2, 2]]:
            return qutip.Qobj(rho.full(), dims=[[2, 2], [2, 2]])
        return rho
