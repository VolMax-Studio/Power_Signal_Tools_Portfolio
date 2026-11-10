"""Harmonic THD estimation by least-squares sinusoidal decomposition.

This is a stationary-signal, known-fundamental analysis, NOT an IEC/IEEE
compliance measurement. Unlike nearest-bin FFT extraction, harmonic fitting
works for non-integer numbers of fundamental periods. The supplied fundamental
frequency must match the signal; no PLL or frequency tracking is performed.
"""
from typing import Optional, Tuple
import numpy as np
from .._validation import finite_signal, positive_finite


def compute_thd(
    signal: np.ndarray,
    fs: float,
    fundamental_hz: float = 50.0,
    n_harmonics: int = 10,
    window: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """Return (THD-F, THD-R, harmonic_rms) for a known fundamental frequency.

    A joint least-squares fit estimates DC, sine and cosine coefficients at
    harmonics 1..n_harmonics. The harmonic RMS amplitude is hypot(a,b)/sqrt(2).

      THD-F = sqrt(sum(V_h**2 for h>=2)) / V_1
      THD-R = sqrt(sum(V_h**2 for h>=2)) / sqrt(sum(V_h**2 for h>=1))

    THD-R here uses ONLY the modelled harmonics (no DC, interharmonics or
    residual noise). It is not the ratio to the raw total-signal RMS.
    If the fundamental amplitude is effectively zero THD-F is infinite;
    for an all-zero signal THD-R is zero.

    The optional window argument provides nonnegative sample weights in the
    least-squares fit (e.g., Hann weights), not FFT apodization. By default
    every sample has equal weight. Frequencies are fixed by fundamental_hz.
    Too-short, aliased, invalid or rank-deficient configurations raise ValueError.
    """
    x = finite_signal(signal, min_size=2)
    fs = positive_finite(fs, "fs")
    f0 = positive_finite(fundamental_hz, "fundamental_hz")
    if isinstance(n_harmonics, (bool, np.bool_)) or not isinstance(n_harmonics, (int, np.integer)) or n_harmonics < 2:
        raise ValueError("n_harmonics must be an integer >= 2")
    if n_harmonics * f0 >= fs / 2:
        raise ValueError("Nyquist limit exceeded by highest harmonic")
    if x.size < int(np.ceil(2.0 * fs / f0)):
        raise ValueError("signal too short: at least two fundamental cycles required")
    if x.size > 250_000:
        raise ValueError("signal too long: analyze in segments of <= 250000 samples")

    if window is None:
        sqrt_weights = None
    else:
        weights = np.asarray(window, dtype=np.float64)
        if weights.shape != x.shape or not np.all(np.isfinite(weights)):
            raise ValueError("window weights must be finite and match signal length")
        if np.any(weights < 0) or np.count_nonzero(weights) < 2 * n_harmonics + 1:
            raise ValueError("window requires sufficient nonnegative weights")
        sqrt_weights = np.sqrt(weights)

    phases = (2.0 * np.pi * f0 / fs) * np.arange(x.size, dtype=np.float64)
    columns = [np.ones(x.size, dtype=np.float64)]
    for h in range(1, n_harmonics + 1):
        columns.extend((np.cos(h * phases), np.sin(h * phases)))
    design = np.column_stack(columns)
    if sqrt_weights is not None:
        design = design * sqrt_weights[:, None]
        target = x * sqrt_weights
    else:
        target = x
    coef, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("harmonic fit is rank-deficient at this sampling configuration")

    components = coef[1:].reshape(n_harmonics, 2)
    rms = np.hypot(components[:, 0], components[:, 1]) / np.sqrt(2.0)
    distortion = float(np.linalg.norm(rms[1:]))
    fundamental = float(rms[0])
    total_harmonic_rms = float(np.linalg.norm(rms))
    thd_f = distortion / fundamental if fundamental > 1e-12 else np.inf
    thd_r = distortion / total_harmonic_rms if total_harmonic_rms > 1e-12 else 0.0
    return float(thd_f), float(thd_r), rms
