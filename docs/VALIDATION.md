# Technical validation notes — 0.2.0rc1

## Scope

This release candidate addresses an observable packaging/API split and a
peak-vs-RMS harmonic normalization defect. The earlier 39 tests covered
top-level repository modules, not the separately shipped package directory.
Those copies were consolidated and the actual package is now tested.

The independent synthetic reference waveform is

    x(t) = A*sin(2*pi*f*t) + 0.1*A*sin(2*pi*3*f*t) + C

with arbitrary amplitude A, DC offset C and known fundamental f.
Analytic results for harmonics 1 and 3:

    THD-F = 0.1
    THD-R = 0.1 / sqrt(1.01) = 0.0995037190209989

The previous second implementation returned THD-R about 0.14072 because
peak harmonic voltage was divided by total-signal RMS. The old FFT-bin
implementation also underestimated THD-F by about 10% relative at 2571
samples of a 50 Hz waveform sampled at 10 kHz, because the observation
window did not contain an integer number of 50 Hz cycles.

The current estimator jointly fits sine/cosine components for harmonic
frequencies and a DC column by numpy.linalg.lstsq. The input must be
sufficiently long, finite and sampled uniformly at an explicitly known
frequency. Invalid configurations fail closed.

## Reproduction (from a clean clone)

    python -m pip install -e ".[test,wavelets]"
    python -m pytest -q
    python examples/generate_demo.py /tmp/volmax-demo.csv
    python -m power_signal_tools /tmp/volmax-demo.csv       --column voltage_v --fs 10000 --fundamental 50       --unit V --output /tmp/volmax-report.json

Use a fresh output name if the report path already exists.

## Interpretation of tests

All passing tests establish only the software behavior tested under
specific numeric models. They do NOT establish:
- power analyzer compliance with any regulatory or IEC/IEEE measurement standard;
- error bounds on actual CT/ADC sensor measurements or grid installations;
- validation of timestamps, calibration, signal origin or physical assets;
- general accuracy under arbitrary frequency drift, switching harmonics,
  spectral leakage, missing samples, nonstationarity or severe noise;
- a causal/real-time guarantee for Hilbert-based analysis.

## Known limitations and follow-up engineering

1. harmonic fit assumes the specified fundamental accurately represents
   the recorded stationary segment; no phase-locked frequency tracking.
2. computational memory grows with samples x number of harmonics; the
   implementation explicitly caps harmonic windows at 250,000 samples.
3. source API changed from a historical, untested duplicate package tree;
   older names such as calculate_thd and rolling_rms in that tree are
   not preserved.
4. optional PyWavelets is required for db4/sym8; Haar fallback is simpler.
5. CI testing on GitHub is distinct from local testing; check the actual
   CI run before tagging a public release.

To check a built wheel, install it in a separate virtual environment and
run the same external-directory smoke test. Confirm that wheel metadata
contains the MIT license and that no repo-root features/ or anomaly/
duplicates are shipped.
