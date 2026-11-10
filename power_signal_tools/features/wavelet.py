"""
wavelet.py — Discrete Wavelet Transform for Transient Localization
===================================================================
Provides DWT-based multi-level signal decomposition and transient detection.
Uses PyWavelets (pywt) as the primary backend; falls back to a manual
single-level Haar implementation if pywt is unavailable.

Physical context — why wavelets for power signals
--------------------------------------------------
FFT gives frequency content but no time information. A voltage sag lasting
2 ms at t=50 ms and the same sag at t=80 ms produce identical FFT spectra.
DWT resolves this: high-frequency detail coefficients localize transients
(sags, swells, notches, spikes) to specific time intervals.

Level-frequency mapping (approximate, for Daubechies-4, fs=10 kHz):
  Level 1 detail: 2500–5000 Hz  (sub-harmonic switching noise)
  Level 2 detail: 1250–2500 Hz
  Level 3 detail:  625–1250 Hz  (inter-harmonic region)
  Level 4 detail:  312–625 Hz   (5th–13th harmonics for 50 Hz grid)
  Level 5 detail:  156–312 Hz   (3rd harmonic and above)
  Level 5 approx: 0–156 Hz      (fundamental + low harmonics)

Rule of thumb: use db4 for power quality (good frequency localization),
sym8 for smooth signals, haar for step-change detection.

Dependencies
------------
Primary:  pip install PyWavelets
Fallback: numpy only (single-level Haar)
"""

import numpy as np
from typing import List, Optional, Tuple

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False


def dwt_decompose(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: Optional[int] = None,
    mode: str = "periodization",
) -> List[np.ndarray]:
    """
    Multi-level DWT decomposition of a 1-D signal.

    Returns coefficients from coarsest (approximation) to finest (detail)
    level, matching the convention used in energy and PQ literature.

    Parameters
    ----------
    signal : np.ndarray
        1-D signal array. Length should ideally be a power of 2 for clean
        level decomposition, but arbitrary lengths are handled via padding.
    wavelet : str
        PyWavelets wavelet name. Common choices:
        - 'db4'  : Daubechies-4, good general-purpose PQ wavelet
        - 'haar' : Simplest, good for step-change detection
        - 'sym8' : Symlet-8, near-symmetric, low phase distortion
        Default: 'db4'.
    level : int, optional
        Number of decomposition levels. If None, uses pywt.dwt_max_level()
        to determine the maximum for the given signal and wavelet.
        Capped at log2(len(signal)) to avoid over-decomposition.
    mode : str
        Signal extension mode for border handling.
        'periodization' minimizes output length (standard for PQ).
        'reflect' reduces boundary artifacts for non-periodic signals.
        Default: 'periodization'.

    Returns
    -------
    list of np.ndarray
        [cA_n, cD_n, cD_(n-1), ..., cD_1]
        where cA_n = approximation at level n (low-frequency residual)
        and cD_k = detail at level k (transient content at scale k).
        Index 0 is coarsest; index -1 is finest (highest frequency).

    Raises
    ------
    ImportError
        If pywt is not installed and wavelet != 'haar'.
    ValueError
        If signal is not 1-D.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 10_000
    >>> t = np.arange(0, 0.1, 1/fs)
    >>> v = np.sin(2*np.pi*50*t)
    >>> # Inject a spike at sample 500
    >>> v[500] += 5.0
    >>> coeffs = dwt_decompose(v, wavelet='db4', level=5)
    >>> len(coeffs)  # [cA5, cD5, cD4, cD3, cD2, cD1]
    6
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")

    if not _PYWT_AVAILABLE:
        if wavelet.lower() != "haar":
            raise ImportError(
                "PyWavelets (pywt) is required for non-Haar wavelets. "
                "Install with: pip install PyWavelets"
            )
        return _haar_decompose(signal, level)

    if level is None:
        max_level = pywt.dwt_max_level(len(signal), wavelet)
        level = min(max_level, int(np.floor(np.log2(len(signal)))))

    coeffs = pywt.wavedec(signal, wavelet, mode=mode, level=level)
    return coeffs  # [cA_n, cD_n, ..., cD_1]


def detect_transients(
    signal: np.ndarray,
    fs: float,
    wavelet: str = "db4",
    detail_level: int = 1,
    threshold_sigma: float = 3.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect transient events (spikes, notches, fast sags) using DWT detail
    coefficients at the specified level.

    The detail coefficients at level 1 respond most strongly to high-frequency
    discontinuities (sub-cycle events: notches, current spikes, switching
    transients). Thresholding by σ-multiples of the detail coefficient
    distribution is equivalent to a matched filter for impulsive events.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal.
    fs : float
        Sampling frequency [Hz]. Used to map detected sample indices to time.
    wavelet : str
        Wavelet family. Default: 'db4'.
    detail_level : int
        Which detail level to analyze. 1 = finest (highest frequency).
        Use level 2–3 for mid-frequency transients (harmonic region).
        Default: 1.
    threshold_sigma : float
        Detection threshold in standard deviations of the detail coefficient
        distribution. 3.5σ gives ~0.02% false alarm rate for Gaussian noise.
        Default: 3.5.

    Returns
    -------
    event_times_s : np.ndarray
        Approximate timestamps [seconds] of detected transient events.
        Resolution = 2^detail_level / fs (DWT decimation factor).
    event_magnitudes : np.ndarray
        Absolute detail coefficient values at event locations.
        Larger values indicate stronger transients.

    Notes
    -----
    DWT coefficient positions map back to original signal with a factor of
    2^level decimation. The returned timestamps are center estimates;
    true transient onset may be ±(filter_length/2)/fs seconds earlier.
    """
    signal = np.asarray(signal, dtype=np.float64)

    coeffs = dwt_decompose(signal, wavelet=wavelet)
    # coeffs[0] = cA_n (approximation), coeffs[-detail_level] = target detail
    # In [cA_n, cD_n, ..., cD_1], cD_1 is last, cD_n is second
    n_levels = len(coeffs) - 1  # number of detail levels

    if detail_level < 1 or detail_level > n_levels:
        raise ValueError(
            f"detail_level={detail_level} out of range [1, {n_levels}]"
        )

    # cD_1 is at index -1, cD_2 at index -2, etc.
    detail = coeffs[-detail_level]

    # Threshold: median absolute deviation for robustness to outliers
    # (standard deviation inflated by presence of transients themselves)
    mad = np.median(np.abs(detail - np.median(detail)))
    sigma_robust = mad / 0.6745  # MAD to σ conversion for Gaussian
    threshold = threshold_sigma * max(sigma_robust, 1e-12)

    event_indices = np.where(np.abs(detail) > threshold)[0]
    event_magnitudes = np.abs(detail[event_indices])

    # Map coefficient indices back to original signal time
    decimation = 2 ** detail_level
    event_times_s = (event_indices * decimation + decimation // 2) / fs

    return event_times_s, event_magnitudes


def _haar_decompose(signal: np.ndarray, level: Optional[int]) -> List[np.ndarray]:
    """Fallback single-level Haar decomposition using numpy only."""
    if level is None:
        level = min(5, int(np.floor(np.log2(len(signal)))))

    result = []
    current = signal.copy()
    for _ in range(level):
        n = (len(current) // 2) * 2  # ensure even length
        c = current[:n]
        approx = (c[0::2] + c[1::2]) / np.sqrt(2)
        detail = (c[0::2] - c[1::2]) / np.sqrt(2)
        result.append(detail)
        current = approx

    result.append(current)  # final approximation
    result.reverse()  # [cA, cD_n, ..., cD_1]
    return result
