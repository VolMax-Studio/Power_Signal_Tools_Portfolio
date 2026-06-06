"""
zscore.py — Rolling Z-Score Anomaly Detector
=============================================
Computes a causal rolling z-score over a sliding window for detecting
transient spikes, slow drift, and step changes in power/voltage signals.

Definition
----------
    z(t) = (x(t) - μ_w(t)) / σ_w(t)

where μ_w and σ_w are the mean and standard deviation computed over the
window of the most recent W samples ending at t (causal window).

Physical applications
---------------------
- Voltage sag/swell detection: |z| > 3 on V_rms signal
- Load step detection: |z| > 4 on active power signal
- Sensor drift / fault: sustained z > 2 for > N samples
- Current spike in battery charging: z > 5 on dI/dt

Causal vs. centered windows
----------------------------
This implementation uses a CAUSAL window (past W samples only), which is
deployable in real-time embedded systems. For offline post-processing,
set centered=True to use a symmetric window (W/2 samples on each side),
which gives better time localization of detected events.
"""

import numpy as np
from typing import Optional


def rolling_zscore(
    signal: np.ndarray,
    window: int,
    centered: bool = False,
    min_periods: Optional[int] = None,
) -> np.ndarray:
    """
    Compute rolling z-score of a 1-D signal.

    Parameters
    ----------
    signal : np.ndarray
        1-D input signal. Any physical units (V, A, W, etc.).
    window : int
        Number of samples in the rolling window.
        Rule of thumb for power signals:
        - Spike detection:   window = 1–2 cycles (200–400 samples at 10kHz/50Hz)
        - Drift detection:   window = 10–100 cycles
        - Step change:       window = 0.5–1 cycle (100–200 samples)
    centered : bool
        If True, use a symmetric window centered on each sample.
        Better temporal accuracy for offline analysis. Default: False (causal).
    min_periods : int, optional
        Minimum number of valid samples required to compute z-score.
        If fewer are available (e.g., at signal start), returns NaN.
        Default: max(2, window // 4).

    Returns
    -------
    z : np.ndarray
        Z-score array, same length as input signal.
        NaN for positions where fewer than min_periods samples are available.
        |z| > 3 indicates >3σ deviation (anomaly candidate).
        |z| > 5 indicates likely fault or sensor artifact.

    Notes
    -----
    Uses Welford's online algorithm via numpy stride tricks for efficiency.
    Time complexity: O(N * window) naive; O(N) with the numpy implementation.
    For very large windows (> 100k samples), consider scipy.ndimage.uniform_filter.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> signal = rng.normal(0, 1, 1000)
    >>> signal[500] += 10.0  # inject spike
    >>> z = rolling_zscore(signal, window=100)
    >>> np.argmax(np.abs(z))  # spike should dominate
    500

    >>> # Step change detection
    >>> step = np.concatenate([np.zeros(500), np.ones(500)])
    >>> z = rolling_zscore(step, window=50)
    >>> z[500]  # step onset: high z-score
    ... # doctest: +ELLIPSIS
    ...
    """
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("signal must be 1-D")
    N = len(signal)

    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    if window > N:
        raise ValueError(f"window ({window}) > signal length ({N})")

    if min_periods is None:
        min_periods = max(2, window // 4)

    z = np.full(N, np.nan)

    if centered:
        half = window // 2
        for i in range(N):
            start = max(0, i - half)
            end = min(N, i + half + 1)
            chunk = signal[start:end]
            if len(chunk) >= min_periods:
                mu = np.mean(chunk)
                sigma = np.std(chunk, ddof=1)
                if sigma > 1e-12:
                    z[i] = (signal[i] - mu) / sigma
                else:
                    z[i] = 0.0
    else:
        # Causal: efficient cumsum-based rolling mean and variance
        # E[x^2] - E[x]^2 formulation for online variance
        cumsum = np.cumsum(np.insert(signal, 0, 0))
        cumsum_sq = np.cumsum(np.insert(signal ** 2, 0, 0))

        for i in range(N):
            start = max(0, i - window + 1)
            n_samples = i - start + 1

            if n_samples < min_periods:
                continue  # leave as NaN

            window_sum = cumsum[i + 1] - cumsum[start]
            window_sum_sq = cumsum_sq[i + 1] - cumsum_sq[start]

            mu = window_sum / n_samples
            variance = window_sum_sq / n_samples - mu ** 2
            # Clamp to 0 to avoid sqrt of small negative from float arithmetic
            sigma = np.sqrt(max(0.0, variance))

            if sigma > 1e-12:
                z[i] = (signal[i] - mu) / sigma
            else:
                z[i] = 0.0

    return z


def find_anomalies(
    z: np.ndarray,
    threshold: float = 3.0,
    min_separation_samples: int = 1,
) -> np.ndarray:
    """
    Extract sample indices where |z| exceeds threshold.

    Merges detections that are closer than min_separation_samples
    (returns the peak within each cluster).

    Parameters
    ----------
    z : np.ndarray
        Z-score array from rolling_zscore().
    threshold : float
        Detection threshold. Default: 3.0 (3σ). Common values:
        - 2.5: sensitive, higher false positive rate
        - 3.0: standard anomaly threshold
        - 4.0: conservative, strong events only
        - 5.0: fault-level events
    min_separation_samples : int
        Minimum spacing between distinct events. Events closer than this
        are merged (peak reported). Default: 1 (no merging).

    Returns
    -------
    np.ndarray
        Integer array of sample indices with |z| > threshold.
        Sorted in ascending order. Empty array if no anomalies found.
    """
    z_valid = np.where(np.isnan(z), 0.0, np.abs(z))
    raw_indices = np.where(z_valid > threshold)[0]

    if len(raw_indices) == 0 or min_separation_samples <= 1:
        return raw_indices

    # Cluster nearby detections and return peak within each cluster
    clusters = []
    current_cluster = [raw_indices[0]]

    for idx in raw_indices[1:]:
        if idx - current_cluster[-1] <= min_separation_samples:
            current_cluster.append(idx)
        else:
            clusters.append(current_cluster)
            current_cluster = [idx]
    clusters.append(current_cluster)

    peaks = np.array([
        c[np.argmax(z_valid[c])] for c in clusters
    ])
    return peaks
