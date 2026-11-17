from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np

from ._dependencies import require_qutip
from .config import (
    FAMILY_NAMES,
    FAMILY_NOISY_00_11,
    FAMILY_NOISY_01_10,
    FAMILY_NOISY_PHI_PLUS,
    FAMILY_NOISY_PSI_PLUS,
    FAMILY_RANDOM_ENTANGLED,
    FAMILY_RANDOM_PURE_MIXED,
    FAMILY_RANDOM_SEPARABLE,
    DatasetConfig,
)
from .ppt_labeler import PPTLabeler

if TYPE_CHECKING:  # pragma: no cover
    from qutip import Qobj


class BipartiteStateFactory:
    def __init__(self, config: DatasetConfig, labeler: PPTLabeler | None = None) -> None:
        self.config = config
        self.labeler = labeler or PPTLabeler()
        self.rng = np.random.default_rng(config.seed)
        self._initialize_basis_states()

    def iter_all_states(self) -> Iterator[tuple[str, "Qobj"]]:
        for family_name, sample_count in self.config.family_counts():
            yield from self._iter_family_states(family_name, sample_count)

    def _initialize_basis_states(self) -> None:
        qutip = require_qutip()
        ket_0 = qutip.basis(2, 0)
        ket_1 = qutip.basis(2, 1)

        self.ket_00 = qutip.tensor(ket_0, ket_0)
        self.ket_01 = qutip.tensor(ket_0, ket_1)
        self.ket_10 = qutip.tensor(ket_1, ket_0)
        self.ket_11 = qutip.tensor(ket_1, ket_1)
        self.psi_plus = (self.ket_01 + self.ket_10).unit()
        self.phi_plus = (self.ket_00 + self.ket_11).unit()
        self.identity = qutip.qeye([2, 2]) / 4.0

    def _iter_family_states(self, family_name: str, sample_count: int) -> Iterator[tuple[str, "Qobj"]]:
        if family_name not in FAMILY_NAMES:
            raise ValueError(f"Unsupported family name: {family_name}")

        for _ in range(sample_count):
            if family_name == FAMILY_NOISY_01_10:
                yield family_name, self._noisy_family_01_10_state()
            elif family_name == FAMILY_NOISY_00_11:
                yield family_name, self._noisy_family_00_11_state()
            elif family_name == FAMILY_RANDOM_ENTANGLED:
                yield family_name, self._random_entangled_state()
            elif family_name == FAMILY_RANDOM_SEPARABLE:
                yield family_name, self._random_separable_state()
            elif family_name == FAMILY_RANDOM_PURE_MIXED:
                yield family_name, self._random_pure_mixed_state()
            elif family_name == FAMILY_NOISY_PSI_PLUS:
                yield family_name, self._noisy_bell_state(self.psi_plus)
            elif family_name == FAMILY_NOISY_PHI_PLUS:
                yield family_name, self._noisy_bell_state(self.phi_plus)

    def _noisy_family_01_10_state(self) -> "Qobj":
        theta, phase = self._sample_angles()
        ket = np.cos(theta / 2.0) * self.ket_01 + np.exp(1j * phase) * np.sin(theta / 2.0) * self.ket_10
        return self._mix_with_white_noise(ket.unit())

    def _noisy_family_00_11_state(self) -> "Qobj":
        theta, phase = self._sample_angles()
        ket = np.cos(theta / 2.0) * self.ket_00 + np.exp(1j * phase) * np.sin(theta / 2.0) * self.ket_11
        return self._mix_with_white_noise(ket.unit())

    def _random_entangled_state(self) -> "Qobj":
        while True:
            rho = self._random_density_matrix((2, 2))
            if self.labeler.is_entangled(rho):
                return rho

    def _random_separable_state(self) -> "Qobj":
        qutip = require_qutip()
        rho_a = self._random_density_matrix((2,))
        rho_b = self._random_density_matrix((2,))
        return qutip.tensor(rho_a, rho_b)

    def _random_pure_mixed_state(self) -> "Qobj":
        ket = self._random_pure_ket((2, 2))
        return self._mix_with_white_noise(ket)

    def _noisy_bell_state(self, bell_state: "Qobj") -> "Qobj":
        return self._mix_with_white_noise(bell_state)

    def _mix_with_white_noise(self, pure_state: "Qobj") -> "Qobj":
        p_value = self.rng.uniform(self.config.noise_min, self.config.noise_max)
        rho_pure = pure_state * pure_state.dag()
        return p_value * rho_pure + (1.0 - p_value) * self.identity

    def _sample_angles(self) -> tuple[float, float]:
        theta = float(self.rng.uniform(0.0, np.pi))
        phase = float(self.rng.uniform(0.0, 2.0 * np.pi))
        return theta, phase

    def _random_density_matrix(self, dims: tuple[int, ...]) -> "Qobj":
        qutip = require_qutip()
        dimension = int(np.prod(dims))
        ginibre = self.rng.normal(size=(dimension, dimension)) + 1j * self.rng.normal(size=(dimension, dimension))
        density = ginibre @ ginibre.conj().T
        density /= np.trace(density)
        return qutip.Qobj(density, dims=[list(dims), list(dims)])

    def _random_pure_ket(self, dims: tuple[int, ...]) -> "Qobj":
        qutip = require_qutip()
        dimension = int(np.prod(dims))
        vector = self.rng.normal(size=dimension) + 1j * self.rng.normal(size=dimension)
        vector /= np.linalg.norm(vector)
        return qutip.Qobj(vector.reshape((dimension, 1)), dims=[list(dims), [1] * len(dims)])
