# VolMax Power Signal Tools

**Open-source Python tools for analyzing already-recorded electrical signals.**
A small signal-processing library plus a CSV-to-JSON command-line report.
It runs completely in software: no sensors, wiring, meter or network API needed.

Status: **0.2.0rc1 — release candidate.** Python 3.10+; MIT licensed.
Not a certified power-quality analyzer or a substitute for calibrated measurements.

## What it does

| Feature | Public Python API | Interpretation |
|---|---|---|
| Window and per-cycle RMS | windowed_rms, block_rms | Sampled-signal RMS in input units |
| Harmonic estimation | compute_thd | THD-F and modelled-harmonic THD-R, known fundamental |
| FFT | compute_fft, dominant_frequencies | Frequency-domain diagnostic (RMS-scaled) |
| Wavelets | dwt_decompose, detect_transients | Time-localized transient candidates |
| Anomaly scores | rolling_zscore, find_anomalies | Statistical departures from recent samples |
| Hilbert envelope | hilbert_envelope, detect_sags_swells | Offline narrowband amplitude-event candidates |
| File reporting | volmax-signal-report | Reproducible JSON output from CSV |

No historical duplicate packages: the only maintained implementation lives under
power_signal_tools/. The 39 original tests plus independent edge-case and CLI
tests import that package directly.

## Install

From a copy of this repository:

    python -m pip install -e ".[test,wavelets]"
    python -m pytest -q

To install just the runtime and the optional wavelet dependency:

    python -m pip install .

PyWavelets is optional: use pip install ".[wavelets]" for db4 and sym8 transforms.
Without it, only the basic Haar fallback is available.

## Run the fully synthetic demo

    python examples/generate_demo.py demo-synthetic.csv
    volmax-signal-report demo-synthetic.csv       --column voltage_v --fs 10000 --fundamental 50       --harmonics 10 --unit V --output demo-report.json

Or equivalently:

    python -m power_signal_tools demo-synthetic.csv       --column voltage_v --fs 10000 --fundamental 50       --unit V --output demo-report.json

The example is a mathematically constructed 50 Hz waveform with RMS
fundamental 230 V and third harmonic 10% of the fundamental. **It is not a
real measured voltage recording.** The expected results are:

| Metric | Expected |
|---|---:|
| Fundamental RMS | 230.000 V |
| THD-F | 0.100000 (10.000%) |
| THD-R (modelled harmonics) | 0.099503719 (9.95037%) |
| Total sample RMS | ~231.147 V |

The JSON report includes a SHA-256 fingerprint of the *input CSV bytes*,
the explicit sampling rate, signal units, fundamental frequency, harmonic
magnitudes, numeric results, and a limitations section. The hash binds the
calculation to those file bytes; it does **not** authenticate the sensor,
calibration, person, or time of acquisition.

## Python example

    import numpy as np
    from power_signal_tools import compute_thd, compute_fft, block_rms

    fs = 10_000
    t = np.arange(2000) / fs
    v = 230 * np.sqrt(2) * (
        np.sin(2*np.pi*50*t) + 0.1*np.sin(2*np.pi*150*t)
    )

    thd_f, thd_r, harmonics_rms = compute_thd(
        v, fs=fs, fundamental_hz=50, n_harmonics=10
    )
    print(thd_f, thd_r)  # ~0.1, ~0.099503719

    per_cycle_rms = block_rms(v, fs=fs, freq=50)
    spectrum = compute_fft(v, fs=fs)
    print(per_cycle_rms.mean(), spectrum.freqs_hz[np.argmax(spectrum.magnitude)])

## Measurement and numerical boundaries

- **Known sampling rate and grid fundamental required.** The tool does not
  infer timestamps, check sampling intervals, estimate grid frequency or
  reconstruct missing samples. Do not feed irregularly sampled CSV values
  as though they were uniform.
- **THD is computed by a joint least-squares harmonic fit** at the supplied
  fundamental and harmonics (1..H) with a DC column. This avoids nearest-FFT-
  bin errors for non-integer periods. The reported THD-R is defined as
  sqrt(sum of RMS harmonic powers for h>=2) divided by sqrt(sum of RMS powers
  for h>=1). DC, interharmonics and unexplained residuals are **excluded**.
  This is explicitly a modelled-harmonic ratio, not raw-signal RMS distortion.
- This is **not** an IEC 61000-4-7/4-30/EN 50160 / IEEE 519 compliant
  measurement implementation. No harmonic grouping/interharmonic grouping,
  prescribed observation windows, instrument calibration, uncertainty
  budget or conformity decision is provided.
- **Stationarity:** evaluate stable windows, not one long recording with
  changing load and frequency. The CLI limits input to 250,000 samples
  and the harmonic fit limits inputs similarly.
- **FFT** defaults to a Hann window and reports RMS bin magnitude. Peaks
  may be biased by leakage if the input is not bin-coherent; do not confuse
  the FFT peak amplitude with the fitted harmonic RMS.
- **Hilbert envelope** analysis is offline (filtfilt is noncausal) and has
  edge artifacts and narrowband assumptions. Its event labels are exploratory,
  not proof of grid-code noncompliance.
- **Wavelet and rolling z-score** produce anomaly *candidates*. Neither
  algorithm proves a fault classification.
- **Raw inputs** should have documented units, measurement provenance,
  sampling rate and an independent calibration record when available.
  These upstream properties cannot be reconstructed from a CSV alone.

## Software verification

Run:

    python -m pytest -q

The test suite includes pure sine-wave identities, multi-harmonic analytical
references, 50/60/50.2 Hz non-coherent windows, DC-offset invariance, Nyquist
bin scaling, NaN/Inf rejection, statistical anomaly regressions, API import
checks and a CSV-to-JSON end-to-end test. Tests run on *synthetic mathematical
signals*, not physical sensors.

A GitHub Actions workflow runs the same tests on supported Python versions.
See docs/VALIDATION.md for reproduction instructions and known open limits.

## Repository structure

    power_signal_tools/        # only canonical importable package
      features/                # RMS, harmonic fitting, FFT, wavelets
      anomaly/                 # z-score, envelope and event extraction
      cli.py                   # uniform CSV -> JSON report
    examples/generate_demo.py  # reproducible synthetic CSV
    tests/                     # analytical and CLI regression checks
    docs/VALIDATION.md         # independent reproduction steps
    LICENSE                    # MIT
    pyproject.toml             # build + dependencies + console script

## License

Copyright (c) 2026 Ivan Nestorov. Source code and original documentation
are provided under the [MIT License](LICENSE). Third-party Python libraries
are dependencies with their respective separate licenses; their source is
not bundled into this repository.

Developed by Ivan Nestorov / VolMax Studio Lab, Serbia.
