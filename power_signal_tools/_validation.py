"""Numerical input validation shared by public analytical functions."""
import numpy as np


def finite_signal(signal, *, min_size=1):
    """Return a finite, nonempty 1-D float64 signal, or raise ValueError."""
    try:
        x = np.asarray(signal, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("signal must be a numeric one-dimensional array") from exc
    if x.ndim != 1:
        raise ValueError("signal must be 1-D")
    if x.size < min_size:
        raise ValueError(f"signal must contain at least {min_size} samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("signal contains non-finite values (NaN or infinity)")
    return x


def positive_finite(value, name):
    """Validate a positive finite scalar and return float."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite positive number") from exc
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return v
