"""
spectral.py — FFT with Physical Frequency Axis
================================================
Wraps numpy FFT to return interpretable physical units: frequencies in Hz,
magnitudes in the same units as the input (V, A, W/m², etc.), and phase in
radians. Eliminates the bin-index arithmetic that causes errors in field use.

Engineering note
----------------
For a signal of length N sampled at fs Hz:
  - Frequency resolution: Δf = fs / N  [Hz/bin]
  - Max resolvable frequency: fs/2  [Nyquist]
  - Frequency of bin k: f_k = k * fs / N  [Hz]

Spectral leakage: when the signal contains frequencies not aligned to the
FFT's frequency grid, energy "leaks" into adjacent bins. A Hann window reduces
peak leakage from -13 dB (rectangular) to -31 dB at the cost of 1.5x wider
main lobe. For power quality analysis (harmonics at exact multiples of 50/60 Hz
on a grid-locked signal), this trade-off is almost always acceptable.
"""

import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class SpectralResult:
    """
    Container for FFT output with physical labeling.

    Attributes
    ----------
    freqs_hz : np.ndarray
        Frequency axis [Hz]. Length = N//2 + 1 (one-sided).
    magnitude : np.ndarray
        RMS amplitude spectrum in input signal units.
        For a pure sine of amplitude A: peak value at f = A/sqrt(2).
    phase_rad : np.ndarray
        Phase spectrum [radians]. Range: [-π, π].
    power_db : np.ndarray
        Power spectrum in dBV (or dBW if input is power). Reference = 1.0.
        Useful for visualizing dynamic range over many decades.
    fs : float
        Sampling frequency used [Hz].
    freq_resolution_hz : float
        Frequency bin width [Hz]. Equal to fs / N_original.
    """
    freqs_hz: np.ndarray
    magnitude: np.ndarray
    phase_rad: np.ndarray
    power_db: np.ndarray
    fs: float
    freq_resolution_hz: float


def compute_fft(
    signal: np.ndarray,
    fs: float,
    window: Optional[np.ndarray] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> SpectralResult:
    """
    Compute one-sided FFT with physical frequency axis and RMS magnitude.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal (voltage, current, or power samples).
    fs : float
        Sampling frequency [Hz]. Must be > 0.
    window : np.ndarray, optional
        Apodization window of length len(signal). None applies Hann window.
        Pass np.ones(N) to use rectangular window (no apodization).
    f_min : float, optional
        Lower frequency bound for output [Hz]. Trims the returned spectrum.
        Does not affect computation — purely a display/analysis filter.
    f_max : float, optional
        Upper frequency bound for output [Hz]. Default: fs/2 (Nyquist).

    Returns
    -------
    SpectralResult
        Dataclass with freqs_hz, magnitude, phase_rad, power_db, fs,
        freq_resolution_hz.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 10_000.0
    >>> t = np.arange(0, 0.1, 1/fs)          # 100 ms, 1000 samples
    >>> v = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)  # 230 Vrms at 50 Hz
    >>> result = compute_fft(v, fs=fs)
    >>> # Fundamental should be at 50 Hz
    >>> peak_idx = np.argmax(result.magnitude)
    >>> abs(result.freqs_hz[peak_idx] - 50.0) < result.freq_resolution_hz
    True
    >>> # Magnitude at 50 Hz should be ≈ 230 V (RMS)
    >>> abs(result.magnitude[peak_idx] - 230.0) < 2.0
    True
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    N = len(signal)

    # Window application
    if window is None:
        w = np.hanning(N)
    else:
        w = np.asarray(window, dtype=np.float64)
        if len(w) != N:
            raise ValueError(f"window length {len(w)} != signal length {N}")

    coherent_gain = np.mean(w)
    windowed = signal * w

    # FFT
    fft_coeffs = np.fft.rfft(windowed) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # One-sided amplitude: double all bins except DC and Nyquist
    amplitude = np.abs(fft_coeffs) * 2 / coherent_gain
    amplitude[0] /= 2  # DC
    if N % 2 == 0:
        amplitude[-1] /= 2  # Nyquist

    # Convert amplitude to RMS (for sinusoidal components)
    # DC and Nyquist are already in their correct form
    rms_magnitude = amplitude / np.sqrt(2)
    rms_magnitude[0] = amplitude[0]  # DC: already RMS (it's a constant)

    # Phase spectrum (from un-scaled coefficients)
    phase = np.angle(fft_coeffs)

    # Power in dB (reference = 1.0 V or 1.0 A)
    # Guard against log(0)
    with np.errstate(divide='ignore'):
        power_db = 20 * np.log10(np.maximum(rms_magnitude, 1e-15))

    freq_resolution = fs / N

    # Frequency band filter
    mask = np.ones(len(freqs), dtype=bool)
    if f_min is not None:
        mask &= freqs >= f_min
    if f_max is not None:
        mask &= freqs <= f_max

    return SpectralResult(
        freqs_hz=freqs[mask],
        magnitude=rms_magnitude[mask],
        phase_rad=phase[mask],
        power_db=power_db[mask],
        fs=fs,
        freq_resolution_hz=freq_resolution,
    )


def dominant_frequencies(
    result: SpectralResult,
    n_peaks: int = 5,
    min_magnitude: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the N largest spectral peaks from a SpectralResult.

    Parameters
    ----------
    result : SpectralResult
        Output from compute_fft().
    n_peaks : int
        Number of peaks to return. Default: 5.
    min_magnitude : float
        Minimum RMS magnitude to consider. Filters noise floor.
        Default: 0.0 (return all peaks regardless of amplitude).

    Returns
    -------
    freqs : np.ndarray
        Frequencies of dominant peaks [Hz], sorted descending by magnitude.
    magnitudes : np.ndarray
        RMS magnitudes at those frequencies.
    """
    mag = result.magnitude.copy()
    mag[mag < min_magnitude] = 0.0

    peak_indices = np.argsort(mag)[::-1][:n_peaks]
    return result.freqs_hz[peak_indices], mag[peak_indices]
