# Changelog

## 0.2.0rc1 (candidate — 2026-11-10)

- Consolidated the tested signal-processing modules into the installable
  power_signal_tools package and removed duplicate repo-root modules.
- Fixed THD-R's mixed peak/RMS basis by estimating harmonics on one RMS
  scale and defining the modelled-harmonic denominator explicitly.
- Replaced nearest-bin THD extraction with a joint least-squares harmonic
  fit, enabling analytically correct non-coherent windows at known f0.
- Added numeric validation and fixed high-DC-offset rolling z-score
  cancellation and FFT Nyquist-bin scaling.
- Added CSV-to-JSON CLI with an input file SHA-256 digest and explicit
  computation limitations; includes a deterministic synthetic example.
- Introduced pyproject.toml, MIT LICENSE, GitHub Actions CI, and an
  expanded analytical and input-boundary test suite.
- This is NOT a certified measurement tool and is not a published release.

## 0.1.0

- Original RMS, FFT, THD, wavelet, anomaly and envelope implementations.
- 39 analytical tests covered repo-root modules; installable package API
  and numerical THD-R differed from the tested sources.
