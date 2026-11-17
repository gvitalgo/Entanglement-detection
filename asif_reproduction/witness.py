"""Fixed coherence-witness baseline documented in the thesis and cited source."""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("Cr(QS)", "Cr(RS)", "Cr(RT)", "Cr(QT)")
WITNESS_THRESHOLD_BITS = 4.0
SOURCE_CITATION = "Bu et al., arXiv:1603.06322; Theorem 2 (product states); thesis Proposition 5.1 (separable states)"


def witness_score_from_features(
    features: np.ndarray,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    input_log_base: str = "e",
) -> np.ndarray:
    """Return ``Cr(QS)+Cr(RS)+Cr(RT)-Cr(QT)`` in base-2 entropy units.

    The maintained feature extractor uses natural logarithms.  The cited witness
    threshold is expressed in bits, so natural-log features are divided by ln(2)
    before the documented linear combination is evaluated.
    """

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected a two-dimensional feature array with four ordered columns.")
    if tuple(feature_names) != FEATURE_NAMES:
        raise ValueError(f"Expected feature order {FEATURE_NAMES}, got {tuple(feature_names)}.")
    if input_log_base == "e":
        values = values / np.log(2.0)
    elif input_log_base != "2":
        raise ValueError("input_log_base must be 'e' or '2'.")
    return values[:, 0] + values[:, 1] + values[:, 2] - values[:, 3]


def witness_predict(
    features: np.ndarray,
    *,
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    input_log_base: str = "e",
    threshold: float = WITNESS_THRESHOLD_BITS,
) -> np.ndarray:
    """Label a state entangled only when the documented score strictly exceeds 4."""

    scores = witness_score_from_features(
        features,
        feature_names=feature_names,
        input_log_base=input_log_base,
    )
    return (scores > threshold).astype(np.int64)
