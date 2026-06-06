# power_signal_tools

Importable Python library for power quality signal analysis. Domain-grounded utilities for grid monitoring, embedded signal processing, and power electronics validation.

The design priority: physically correct implementations with documented tolerances, not demo notebooks.

```python
from power_signal_tools.features import windowed_rms, compute_thd, compute_fft
from power_signal_tools.anomaly import rolling_zscore, hilbert_envelope
```

## Modules

### `features/`

| Module | Function | Description |
|---|---|---|
| `rms.py` | `windowed_rms(signal, window, step)` | Sliding-window RMS, 1-D and multi-channel |
| `rms.py` | `block_rms(signal, fs, freq)` | Per-cycle RMS (one window per grid cycle) |
| `thd.py` | `compute_thd(signal, fs, fundamental_hz)` | THD-F and THD-R via FFT harmonic extraction |
| `spectral.py` | `compute_fft(signal, fs)` | One-sided FFT with physical Hz axis and RMS magnitudes |
| `spectral.py` | `dominant_frequencies(result, n_peaks)` | N largest spectral peaks |
| `wavelet.py` | `dwt_decompose(signal, wavelet, level)` | Multi-level DWT decomposition |
| `wavelet.py` | `detect_transients(signal, fs, wavelet)` | Sub-cycle transient localization via detail coefficients |

### `anomaly/`

| Module | Function | Description |
|---|---|---|
| `zscore.py` | `rolling_zscore(signal, window)` | Causal rolling z-score for spike and drift detection |
| `zscore.py` | `find_anomalies(z, threshold)` | Extract sample indices where `|z| > threshold` |
| `envelope.py` | `hilbert_envelope(signal, fs)` | Instantaneous amplitude envelope via analytic signal |
| `envelope.py` | `detect_sags_swells(envelope, fs, nominal_rms)` | IEC 61000-4-11 sag/swell event classification |

## Quick start

```python
import numpy as np
from power_signal_tools.features.rms import block_rms
from power_signal_tools.features.thd import compute_thd
from power_signal_tools.features.spectral import compute_fft

fs = 10_000.0  # Hz
t = np.arange(0, 0.2, 1/fs)
v = 230*np.sqrt(2) * np.sin(2*np.pi*50*t)

# Per-cycle RMS
rms = block_rms(v, fs=fs, freq=50.0)   # array of 10 values ≈ 230.0 V

# THD
thd_f, thd_r, harmonics = compute_thd(v, fs=fs)
print(f"THD-F: {thd_f*100:.2f}%")

# Spectrum with physical axis
result = compute_fft(v, fs=fs)
# result.freqs_hz  → [0.0, 5.0, 10.0, ..., 5000.0] Hz
# result.magnitude → RMS amplitude at each frequency bin
```

```python
from power_signal_tools.anomaly.zscore import rolling_zscore, find_anomalies

# Detect current spikes in a measurement stream
z = rolling_zscore(current_signal, window=200)  # 200 samples = 1 cycle at 10kHz/50Hz
anomaly_indices = find_anomalies(z, threshold=3.5)
```

## Test suite

```bash
pip install -r requirements.txt
pytest tests/test_features.py -v
```

39 tests. All assertions are analytically verified against known signals
(pure sines, harmonic injections, step changes). Tolerances are documented
with physical justification in each test.

**Results on Python 3.12 / numpy 2.x / scipy 1.14:**
```
39 passed in 0.96s
```

## Design notes

**THD implementation:** Uses Hann window + nearest-bin harmonic extraction. Coherent gain correction applied so windowing does not attenuate the RMS estimate. THD-F matches IEC 61000-3-2 definition (ratio to fundamental); THD-R is bounded in [0, 1).

**Hilbert envelope prefilter:** The narrow Butterworth bandpass (default BW=20 Hz) uses `sosfiltfilt` (SOS form). For narrow fractional bandwidths (< 5%), `(b,a)` form butter is numerically unstable — gain errors of 3–10× observed. SOS form eliminates this. Edge settling time: ~800 samples (80 ms at 10 kHz) for 20 Hz bandwidth. For clean signals (THD < 5%), use `prefilter=False`.

**Wavelet fallback:** `dwt_decompose` uses PyWavelets if available; falls back to a numpy-only Haar implementation. For production use on db4/sym8, install PyWavelets.

**Rolling z-score causality:** Default mode is causal (past W samples only) — deployable in real-time embedded systems. Pass `centered=True` for offline analysis with better temporal localization.

## Scope

Covers: power quality disturbance analysis · NILM feature extraction · grid voltage monitoring · embedded sensor signal processing · battery current/voltage analysis.

Not in scope: streaming/real-time I/O, hardware drivers, communication protocols.

## License

MIT
