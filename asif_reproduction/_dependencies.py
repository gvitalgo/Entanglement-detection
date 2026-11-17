from __future__ import annotations

from types import ModuleType

_QUTIP_IMPORT_ERROR: ModuleNotFoundError | None = None
_TENSORFLOW_IMPORT_ERROR: ModuleNotFoundError | None = None
_MATPLOTLIB_IMPORT_ERROR: ModuleNotFoundError | None = None
_matplotlib = None

try:
    import qutip as _qutip
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    _qutip = None
    _QUTIP_IMPORT_ERROR = exc

try:
    import tensorflow as _tensorflow
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    _tensorflow = None
    _TENSORFLOW_IMPORT_ERROR = exc

def require_qutip() -> ModuleType:
    if _qutip is None:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "qutip is required for the quantum-state pipeline. "
            "Install it in the active interpreter before running this command."
        ) from _QUTIP_IMPORT_ERROR
    return _qutip


def require_tensorflow() -> ModuleType:
    if _tensorflow is None:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "tensorflow is required for ANN training and evaluation. "
            "Install it in the active interpreter before running this command."
        ) from _TENSORFLOW_IMPORT_ERROR
    return _tensorflow


def require_matplotlib() -> ModuleType:
    global _matplotlib, _MATPLOTLIB_IMPORT_ERROR
    if _matplotlib is None:
        try:
            import matplotlib as _loaded_matplotlib
        except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
            _MATPLOTLIB_IMPORT_ERROR = exc
            _loaded_matplotlib = None
        _matplotlib = _loaded_matplotlib
    if _matplotlib is None:  # pragma: no cover - environment dependent
        raise ModuleNotFoundError(
            "matplotlib is required for plot generation. "
            "Install it in the active interpreter before running this command."
        ) from _MATPLOTLIB_IMPORT_ERROR
    return _matplotlib
