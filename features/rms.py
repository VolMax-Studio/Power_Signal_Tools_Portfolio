"""
rms.py — Windowed Root Mean Square
===================================
Computes RMS over a sliding window for 1-D or 2-D (multi-channel) arrays.

Physical context
----------------
For a pure sine wave v(t) = Vp * sin(2*pi*f*t), the true RMS is Vp / sqrt(2).
This holds only when the window contains an integer number of cycles, or when
the window is large enough that edge effects are negligible (window >> 1/f).
For non-periodic or transient signals, windowed RMS tracks instantaneous power
content without cycle-alignment assumptions.

API
---
windowed_rms(signal, window, step, axis) -> np.ndarray
"""

import numpy as np
from typing import Optional, Union


def windowed_rms(
    signal: np.ndarray,
    window: int,
    step: int = 1,
    axis: int = 0,
) -> np.ndarray:
    """
    Compute windowed RMS along the specified axis.

    Parameters
    ----------
    signal : np.ndarray
        1-D array (N,) for single channel, or 2-D array (N, C) for C channels.
        N is the number of samples, C is the number of channels.
    window : int
        Number of samples per RMS window. For a 50 Hz signal sampled at
        10 kHz, one full cycle = 200 samples.
    step : int, optional
        Stride between consecutive windows. step=1 gives a fully-overlapping
        rolling RMS. step=window gives non-overlapping (block) RMS.
        Default: 1.
    axis : int, optional
        Axis along which to slide the window. Default: 0 (time axis).

    Returns
    -------
    np.ndarray
        RMS values. Shape depends on input:
        - 1-D input (N,)     -> (M,) where M = floor((N - window) / step) + 1
        - 2-D input (N, C)   -> (M, C) if axis=0

    Raises
    ------
    ValueError
        If window > signal length along the specified axis, or step < 1.

    Examples
    --------
    >>> import numpy as np
    >>> fs = 10_000  # 10 kHz
    >>> t = np.arange(0, 0.1, 1/fs)          # 100 ms
    >>> v = 230 * np.sqrt(2) * np.sin(2*np.pi*50*t)  # 230 Vrms at 50 Hz
    >>> rms = windowed_rms(v, window=200, step=200)   # one window per cycle
    >>> np.allclose(rms, 230.0, atol=0.01)
    True
    """
    signal = np.asarray(signal, dtype=np.float64)

    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    n = signal.shape[axis]
    if window > n:
        raise ValueError(
            f"window ({window}) exceeds signal length ({n}) on axis {axis}"
        )

    # Move target axis to front for uniform indexing
    signal = np.moveaxis(signal, axis, 0)  # shape: (N, ...) or (N,)
    n = signal.shape[0]

    indices = np.arange(0, n - window + 1, step)
    n_windows = len(indices)

    if signal.ndim == 1:
        # Fast path: stride trick for 1-D
        shape = (n_windows, window)
        strides = (signal.strides[0] * step, signal.strides[0])
        windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
        result = np.sqrt(np.mean(windows ** 2, axis=1))
    else:
        # Multi-channel: iterate windows (N typically small enough)
        rest_shape = signal.shape[1:]
        result = np.empty((n_windows,) + rest_shape, dtype=np.float64)
        for i, start in enumerate(indices):
            chunk = signal[start : start + window]
            result[i] = np.sqrt(np.mean(chunk ** 2, axis=0))

    return result


def block_rms(signal: np.ndarray, fs: float, freq: float = 50.0) -> np.ndarray:
    """
    Convenience wrapper: non-overlapping RMS with window = one grid cycle.

    Parameters
    ----------
    signal : np.ndarray
        1-D time-domain voltage or current samples.
    fs : float
        Sampling frequency [Hz].
    freq : float
        Grid fundamental frequency [Hz]. Default: 50.0 (Europe/Serbia).

    Returns
    -------
    np.ndarray
        Per-cycle RMS values.

    Notes
    -----
    Window length is rounded to the nearest integer. For fs=10000, freq=50:
    window = 200 samples exactly. For fs=10000, freq=60: window = 167 samples
    (0.3% cycle-length error — acceptable for power monitoring).
    """
    window = int(round(fs / freq))
    return windowed_rms(signal, window=window, step=window)
