"""
envelope.py — Hilbert Envelope for Amplitude Anomaly Detection
==============================================================
Extracts the instantaneous amplitude envelope of a bandpass signal using the
analytic signal (Hilbert transform). Detects sub-cycle voltage sags, swells,
flicker, and power oscillation damping in real time.

Physical context
----------------
For a narrowband signal x(t) ≈ A(t)·cos(2π·f₀·t + φ(t)), the analytic signal
is:
    z(t) = x(t) + j·H{x(t)} = A(t)·exp(j·(2π·f₀·t + φ(t)))

where H{·} is the Hilbert transform. The instantaneous envelope is:
    A(t) = |z(t)| = sqrt(x(t)² + H{x(t)}²)

For a pure 230 Vrms sinusoid: A(t) = 230·√2 ≈ 325.3 V (peak amplitude).
The envelope should be constant for undistorted signals. Deviations indicate
amplitude modulation, sags, swells, or oscillatory instability.

Limitation: Hilbert-based envelope is valid only for narrowband signals.
For wideband signals (THD > 20%), bandpass-filter first to isolate the
fundamental before applying this function.
"""

import numpy as np
from scipy.signal import hilbert, butter, sosfiltfilt
from typing import Optional, Tuple


def hilbert_envelope(
    signal: np.ndarray,
    fs: float,
    prefilter: bool = True,
    fundamental_hz: float = 50.0,
    bw_hz: float = 20.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract instantaneous amplitude envelope via Hilbert transform.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain signal (voltage or current samples).
    fs : float
        Sampling frequency [Hz].
    prefilter : bool
        If True, apply a Butterworth bandpass filter centered on
        fundamental_hz ± bw_hz/2 before computing envelope. Strongly
        recommended for distorted signals (THD > 5%).
        Default: True.
    fundamental_hz : float
        Center frequency for prefilter [Hz]. Default: 50.0.
    bw_hz : float
        Bandwidth of prefilter [Hz]. Symmetric around fundamental.
        Default: 20.0 → passband = [40, 60] Hz for 50 Hz grid.

    Returns
    -------
    envelope : np.ndarray
        Instantaneous amplitude envelope. Same length as signal.
        Units: same as input (V, A).
        For undistorted 230 Vrms: envelope ≈ 325.3 V (constant).
    instantaneous_phase : np.ndarray
        Instantaneous phase [radians]. Useful for phase jump detection
        (e.g., voltage phase angle shifts during grid fault recovery).
        Range: [-π, π] (wrapped). Use np.unwrap() for continuous phase.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 10_000.0
    >>> t = np.arange(0, 0.1, 1/fs)
    >>> # 230 Vrms sine, no distortion
    >>> v = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)
    >>> env, phase = hilbert_envelope(v, fs=fs)
    >>> # Envelope should be ≈ 325.3 V (constant after filter settling)
    >>> steady = env[200:-200]  # exclude filter transients at edges
    >>> abs(np.mean(steady) - 230*np.sqrt(2)) < 5.0
    True
    >>> np.std(steady) < 2.0  # low ripple for undistorted signal
    True
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")

    N = len(signal)
    nyquist = fs / 2.0

    if prefilter:
        low = (fundamental_hz - bw_hz / 2) / nyquist
        high = (fundamental_hz + bw_hz / 2) / nyquist

        if low <= 0 or high >= 1.0:
            filtered = signal
        else:
            # SOS form mandatory for narrow bandpass (fractional BW < 5%).
            # ba-form Butterworth is numerically unstable when
            # (high - low) << 1, causing gain errors of 3-10x.
            # Order 2 is sufficient; higher orders amplify quantization error.
            sos = butter(2, [low, high], btype='bandpass', output='sos')
            filtered = sosfiltfilt(sos, signal)
    else:
        filtered = signal

    # Analytic signal via Hilbert transform
    analytic = hilbert(filtered)
    envelope = np.abs(analytic)
    instantaneous_phase = np.angle(analytic)

    return envelope, instantaneous_phase


def detect_sags_swells(
    envelope: np.ndarray,
    fs: float,
    nominal_rms: float,
    sag_threshold: float = 0.9,
    swell_threshold: float = 1.1,
    min_duration_ms: float = 5.0,
) -> dict:
    """
    Classify voltage sags and swells from an amplitude envelope.

    Follows IEC 61000-4-11 / EN 50160 event classification:
    - Voltage sag:   envelope drops below nominal * sag_threshold
    - Voltage swell: envelope exceeds nominal * swell_threshold

    Parameters
    ----------
    envelope : np.ndarray
        Instantaneous amplitude envelope from hilbert_envelope().
        Units: V (peak). Converted internally to RMS for threshold comparison.
    fs : float
        Sampling frequency [Hz].
    nominal_rms : float
        Nominal grid RMS voltage [V]. Typically 230 V (EU) or 120 V (US).
    sag_threshold : float
        Sag threshold as fraction of nominal RMS. IEC 61000-4-30: 0.90.
        Default: 0.90.
    swell_threshold : float
        Swell threshold as fraction of nominal. EN 50160: 1.10.
        Default: 1.10.
    min_duration_ms : float
        Minimum event duration [ms]. Events shorter than this are
        classified as transients, not sags/swells. Default: 5.0 ms.

    Returns
    -------
    dict with keys:
        'sags'   : list of dicts {start_s, end_s, duration_ms, min_rms_v}
        'swells' : list of dicts {start_s, end_s, duration_ms, max_rms_v}
        'n_sags' : int
        'n_swells' : int

    Notes
    -----
    Envelope is in peak voltage. Convert to RMS: V_rms ≈ envelope / sqrt(2).
    This approximation is exact for pure sinusoids and accurate to ±2%
    for THD < 20% after bandpass filtering.
    """
    # Convert peak envelope to approximate RMS
    env_rms = envelope / np.sqrt(2)
    min_samples = int(min_duration_ms * 1e-3 * fs)

    sag_level = nominal_rms * sag_threshold
    swell_level = nominal_rms * swell_threshold

    def _extract_events(condition: np.ndarray, stat_fn) -> list:
        """Extract contiguous runs satisfying condition."""
        events = []
        in_event = False
        start = 0
        for i, val in enumerate(condition):
            if val and not in_event:
                start = i
                in_event = True
            elif not val and in_event:
                duration_samples = i - start
                if duration_samples >= min_samples:
                    events.append({
                        'start_s': start / fs,
                        'end_s': i / fs,
                        'duration_ms': duration_samples / fs * 1000,
                        'stat_value': float(stat_fn(env_rms[start:i])),
                    })
                in_event = False
        if in_event:
            duration_samples = len(condition) - start
            if duration_samples >= min_samples:
                events.append({
                    'start_s': start / fs,
                    'end_s': len(condition) / fs,
                    'duration_ms': duration_samples / fs * 1000,
                    'stat_value': float(stat_fn(env_rms[start:])),
                })
        return events

    sag_events = _extract_events(env_rms < sag_level, np.min)
    swell_events = _extract_events(env_rms > swell_level, np.max)

    # Rename stat_value for clarity
    for e in sag_events:
        e['min_rms_v'] = e.pop('stat_value')
    for e in swell_events:
        e['max_rms_v'] = e.pop('stat_value')

    return {
        'sags': sag_events,
        'swells': swell_events,
        'n_sags': len(sag_events),
        'n_swells': len(swell_events),
    }
