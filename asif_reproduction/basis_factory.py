from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._dependencies import require_qutip
from .config import BasisConfig

if TYPE_CHECKING:  # pragma: no cover
    from qutip import Qobj


@dataclass(slots=True, frozen=True)
class CoherenceBases:
    q: tuple["Qobj", "Qobj"]
    r: tuple["Qobj", "Qobj"]
    s: tuple["Qobj", "Qobj"]
    t: tuple["Qobj", "Qobj"]
    qs: tuple["Qobj", "Qobj", "Qobj", "Qobj"]
    rs: tuple["Qobj", "Qobj", "Qobj", "Qobj"]
    rt: tuple["Qobj", "Qobj", "Qobj", "Qobj"]
    qt: tuple["Qobj", "Qobj", "Qobj", "Qobj"]

    @property
    def ordered_feature_bases(self) -> tuple[tuple[str, tuple["Qobj", ...]], ...]:
        return (
            ("Cr(QS)", self.qs),
            ("Cr(RS)", self.rs),
            ("Cr(RT)", self.rt),
            ("Cr(QT)", self.qt),
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.ordered_feature_bases)


class CoherenceBasisFactory:
    def __init__(self, config: BasisConfig) -> None:
        self.config = config

    def build(self) -> CoherenceBases:
        qutip = require_qutip()
        ket_0 = qutip.basis(2, 0)
        ket_1 = qutip.basis(2, 1)

        phase = np.exp(1j * self.config.phi)
        r_plus = (ket_0 + 1j * phase * ket_1).unit()
        r_minus = (ket_0 - 1j * phase * ket_1).unit()
        s_plus = (ket_0 + ket_1).unit()
        s_minus = (ket_0 - ket_1).unit()

        q_basis = (ket_0, ket_1)
        r_basis = (r_plus, r_minus)
        s_basis = (s_plus, s_minus)
        t_basis = (ket_0, ket_1)

        return CoherenceBases(
            q=q_basis,
            r=r_basis,
            s=s_basis,
            t=t_basis,
            qs=self._tensor_basis(q_basis, s_basis),
            rs=self._tensor_basis(r_basis, s_basis),
            rt=self._tensor_basis(r_basis, t_basis),
            qt=self._tensor_basis(q_basis, t_basis),
        )

    @staticmethod
    def _tensor_basis(left_basis: tuple["Qobj", "Qobj"], right_basis: tuple["Qobj", "Qobj"]) -> tuple["Qobj", "Qobj", "Qobj", "Qobj"]:
        qutip = require_qutip()
        return tuple(
            qutip.tensor(left_state, right_state)
            for left_state in left_basis
            for right_state in right_basis
        )
