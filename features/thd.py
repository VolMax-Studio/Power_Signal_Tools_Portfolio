"""
thd.py — Total Harmonic Distortion
====================================
Computes THD-F (relative to fundamental) and THD-R (relative to total RMS)
from a time-domain signal using FFT harmonic extraction.

Definitions
-----------
    THD_F = sqrt(sum(V_h^2, h=2..H)) / V_1
    THD_R = sqrt(sum(V_h^2, h=2..H)) / V_rms

where V_h is the RMS amplitude of the h-th harmonic, V_1 is the fundamental
RMS amplitude, and V_rms is the total signal RMS.

Physical context
----------------
THD_F is the standard metric in IEC 61000-3-2 and IEEE 519. A pure sine wave
yields THD_F = 0. Typical limits: THD_F < 5% for grid voltage (EN 50160),
< 8% for current at point of common coupling (IEEE 519, utility side).

THD_R is bounded in [0, 1) and is preferred in some audio and power converter
contexts because it does not blow up when V_1 → 0. THD_F has no upper bound.
"""

import numpy as np
from typing import Tuple, Optional


def compute_thd(
    signal: np.ndarray,
    fs: float,
    fundamental_hz: float = 50.0,
    n_harmonics: int = 10,
    window: Optional[np.ndarray] = None,
) -> Tuple[float, float, np.ndarray]:
    """
    Compute THD-F and THD-R for a 1-D time-domain signal.

    The function isolates harmonic content by extracting FFT bins within a
    narrow band (±0.5 * bin_resolution) around each expected harmonic
    frequency. This is robust to small fundamental frequency drift (<0.1 Hz).

    Parameters
    ----------
    signal : np.ndarray
        1-D array of voltage or current samples. Should contain at least
        2 full cycles of the fundamental for reliable harmonic estimation.
        Recommended: 10+ cycles (200 ms at 50 Hz, 10 kHz sampling).
    fs : float
        Sampling frequency [Hz].
    fundamental_hz : float
        Expected fundamental frequency [Hz]. Default: 50.0.
    n_harmonics : int
        Number of harmonics to include (h=2 to h=n_harmonics).
        Default: 10 (up to 500 Hz for 50 Hz system — within IEC 61000 scope).
    window : np.ndarray, optional
        Apodization window of length len(signal). If None, a Hann window is
        applied to reduce spectral leakage. Pass np.ones(N) to disable.

    Returns
    -------
    thd_f : float
        THD relative to fundamental. Dimensionless (not percent).
        Multiply by 100 for percent.
    thd_r : float
        THD relative to total RMS. Dimensionless.
    harmonic_rms : np.ndarray
        RMS amplitudes of harmonics [h=1, h=2, ..., h=n_harmonics].
        Index 0 = fundamental, index 1 = 2nd harmonic, etc.
        Units match input signal units (V, A, or W).

    Raises
    ------
    ValueError
        If n_harmonics * fundamental_hz >= fs/2 (Nyquist limit exceeded),
        or if signal has fewer than 2 fundamental cycles.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 10_000.0
    >>> t = np.arange(0, 0.2, 1/fs)              # 200 ms = 10 cycles at 50 Hz
    >>> pure_sine = np.sin(2*np.pi*50*t)
    >>> thd_f, thd_r, hrms = compute_thd(pure_sine, fs=fs)
    >>> abs(thd_f) < 1e-3   # THD of pure sine ≈ 0
    True

    >>> # 10% 3rd harmonic injection
    >>> distorted = np.sin(2*np.pi*50*t) + 0.1*np.sin(2*np.pi*150*t)
    >>> thd_f, _, _ = compute_thd(distorted, fs=fs)
    >>> abs(thd_f - 0.1) < 0.005   # THD-F ≈ 0.10
    True
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)

    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")

    # Validate Nyquist constraint
    max_harmonic_freq = n_harmonics * fundamental_hz
    if max_harmonic_freq >= fs / 2:
        raise ValueError(
            f"n_harmonics={n_harmonics} at {fundamental_hz} Hz gives "
            f"{max_harmonic_freq} Hz, which exceeds Nyquist limit {fs/2} Hz. "
            f"Reduce n_harmonics to <= {int((fs/2 - 1) // fundamental_hz)}."
        )

    # Validate minimum signal length: 2 full cycles
    min_samples = int(2 * fs / fundamental_hz)
    if N < min_samples:
        raise ValueError(
            f"Signal too short: {N} samples < {min_samples} (2 cycles at "
            f"{fundamental_hz} Hz, fs={fs} Hz). THD estimate would be unreliable."
        )

    # Apply window to reduce spectral leakage
    if window is None:
        w = np.hanning(N)
    else:
        w = np.asarray(window, dtype=np.float64)
        if len(w) != N:
            raise ValueError(f"window length {len(w)} != signal length {N}")

    # Coherent gain correction: scale so window does not attenuate RMS estimate
    coherent_gain = np.mean(w)
    windowed = signal * w

    # One-sided FFT, magnitude in original signal units
    fft_coeffs = np.fft.rfft(windowed) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Amplitude spectrum: factor of 2 for one-sided (except DC and Nyquist)
    amplitude = np.abs(fft_coeffs) * 2 / coherent_gain
    amplitude[0] /= 2  # DC bin: no doubling
    if N % 2 == 0:
        amplitude[-1] /= 2  # Nyquist bin: no doubling

    # RMS spectrum (amplitude -> RMS: divide by sqrt(2) for sinusoids)
    rms_spectrum = amplitude / np.sqrt(2)
    rms_spectrum[0] = np.abs(fft_coeffs[0]) / coherent_gain  # DC stays as-is

    # Extract RMS at each harmonic using nearest-bin lookup
    bin_resolution = fs / N  # Hz per FFT bin
    harmonic_rms = np.zeros(n_harmonics)

    for h in range(1, n_harmonics + 1):
        target_freq = h * fundamental_hz
        # Find bin closest to target harmonic
        nearest_bin = int(round(target_freq / bin_resolution))
        nearest_bin = min(nearest_bin, len(rms_spectrum) - 1)
        harmonic_rms[h - 1] = rms_spectrum[nearest_bin]

    # Compute THD metrics
    V1 = harmonic_rms[0]  # fundamental RMS
    distortion_rms = np.sqrt(np.sum(harmonic_rms[1:] ** 2))

    if V1 < 1e-12:
        thd_f = np.inf  # degenerate: no fundamental
    else:
        thd_f = distortion_rms / V1

    # Total RMS from signal directly (not from FFT — more accurate)
    v_rms = np.sqrt(np.mean(signal ** 2))
    if v_rms < 1e-12:
        thd_r = 0.0
    else:
        thd_r = distortion_rms / v_rms

    return float(thd_f), float(thd_r), harmonic_rms
