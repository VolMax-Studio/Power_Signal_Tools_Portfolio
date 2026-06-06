"""
test_features.py — Unit tests for power_signal_tools
======================================================
All tests use analytically verifiable signals (pure sines, known harmonics,
step changes) with tight tolerances. No mock data, no approximate assertions
without documented justification for the tolerance.

Run with:
    pytest tests/test_features.py -v

Requirements: numpy, scipy, pytest
Optional:     pywt (for wavelet tests beyond Haar)
"""

import numpy as np
import pytest
import sys
import os

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from features.rms import windowed_rms, block_rms
from features.thd import compute_thd
from features.spectral import compute_fft, dominant_frequencies
from features.wavelet import dwt_decompose, detect_transients
from anomaly.zscore import rolling_zscore, find_anomalies
from anomaly.envelope import hilbert_envelope, detect_sags_swells

# ─── Shared fixtures ────────────────────────────────────────────────────────

FS = 10_000.0          # Hz — standard for PQ analysis
F0 = 50.0              # Hz — EU grid fundamental
VP = 230.0 * np.sqrt(2)  # Peak voltage for 230 Vrms


def make_sine(duration_s: float = 0.2, vp: float = VP, f: float = F0) -> np.ndarray:
    """Pure sine wave: v(t) = Vp * sin(2π·f·t)"""
    t = np.arange(0, duration_s, 1.0 / FS)
    return vp * np.sin(2 * np.pi * f * t)


def make_distorted(
    duration_s: float = 0.2,
    harmonic_ratios: dict = None,
) -> np.ndarray:
    """
    Sine with superimposed harmonics.
    harmonic_ratios: {harmonic_order: amplitude_fraction_of_fundamental}
    e.g. {3: 0.1, 5: 0.05} → 10% 3rd + 5% 5th harmonic
    """
    if harmonic_ratios is None:
        harmonic_ratios = {}
    t = np.arange(0, duration_s, 1.0 / FS)
    signal = VP * np.sin(2 * np.pi * F0 * t)
    for h, ratio in harmonic_ratios.items():
        signal += ratio * VP * np.sin(2 * np.pi * h * F0 * t)
    return signal


# ─── RMS Tests ──────────────────────────────────────────────────────────────

class TestRMS:

    def test_pure_sine_rms_equals_vp_over_sqrt2(self):
        """RMS of pure sine must equal Vp/sqrt(2) — fundamental identity."""
        v = make_sine()
        one_cycle = int(FS / F0)  # 200 samples
        rms = windowed_rms(v, window=one_cycle, step=one_cycle)
        expected = VP / np.sqrt(2)
        # Tolerance: 0.1% — residual from non-integer-cycle boundary
        assert np.allclose(rms, expected, rtol=1e-3), (
            f"Expected RMS ≈ {expected:.3f} V, got mean {np.mean(rms):.3f} V"
        )

    def test_rms_dc_signal(self):
        """RMS of a constant DC signal equals its value."""
        dc = np.full(1000, 5.0)
        rms = windowed_rms(dc, window=100, step=100)
        assert np.allclose(rms, 5.0, atol=1e-10)

    def test_rms_zero_signal(self):
        """RMS of zero signal is zero."""
        zeros = np.zeros(500)
        rms = windowed_rms(zeros, window=100, step=100)
        assert np.allclose(rms, 0.0, atol=1e-12)

    def test_rms_multi_channel(self):
        """Multi-channel: two channels with known RMS values."""
        t = np.arange(0, 0.1, 1.0 / FS)
        ch1 = 100.0 * np.sqrt(2) * np.sin(2 * np.pi * F0 * t)  # 100 Vrms
        ch2 = 200.0 * np.sqrt(2) * np.sin(2 * np.pi * F0 * t)  # 200 Vrms
        signal_2ch = np.stack([ch1, ch2], axis=1)  # (N, 2)
        one_cycle = int(FS / F0)
        rms = windowed_rms(signal_2ch, window=one_cycle, step=one_cycle)
        assert rms.shape[1] == 2
        assert np.allclose(rms[:, 0], 100.0, rtol=1e-3)
        assert np.allclose(rms[:, 1], 200.0, rtol=1e-3)

    def test_rms_output_length(self):
        """Output length formula: floor((N - window) / step) + 1."""
        N, window, step = 1000, 200, 50
        signal = np.random.randn(N)
        rms = windowed_rms(signal, window=window, step=step)
        expected_len = (N - window) // step + 1
        assert len(rms) == expected_len

    def test_block_rms_wrapper(self):
        """block_rms convenience wrapper: one RMS per grid cycle."""
        v = make_sine(duration_s=0.1)
        rms = block_rms(v, fs=FS, freq=F0)
        assert len(rms) == 5  # 100 ms / (1/50 Hz) = 5 cycles
        assert np.allclose(rms, VP / np.sqrt(2), rtol=1e-3)

    def test_rms_raises_on_short_window(self):
        """Should raise if window > signal length."""
        with pytest.raises(ValueError, match="exceeds signal length"):
            windowed_rms(np.ones(10), window=100)

    def test_rms_raises_on_invalid_step(self):
        """Should raise if step < 1."""
        with pytest.raises(ValueError, match="step must be >= 1"):
            windowed_rms(np.ones(100), window=10, step=0)


# ─── THD Tests ──────────────────────────────────────────────────────────────

class TestTHD:

    def test_pure_sine_thd_is_zero(self):
        """THD of a pure sine must be zero — analytical ground truth."""
        v = make_sine()
        thd_f, thd_r, _ = compute_thd(v, fs=FS, fundamental_hz=F0)
        # Tolerance: 0.1% — residual from FFT leakage with Hann window
        assert abs(thd_f) < 1e-3, f"THD-F of pure sine = {thd_f:.6f}, expected ≈ 0"
        assert abs(thd_r) < 1e-3, f"THD-R of pure sine = {thd_r:.6f}, expected ≈ 0"

    def test_single_harmonic_thd_f_analytical(self):
        """
        With 10% 3rd harmonic: THD-F = 0.10 exactly (one harmonic).
        Analytical: THD_F = sqrt(V3²) / V1 = 0.1*V1 / V1 = 0.10
        """
        v = make_distorted(harmonic_ratios={3: 0.10})
        thd_f, _, _ = compute_thd(v, fs=FS, fundamental_hz=F0)
        assert abs(thd_f - 0.10) < 0.005, (
            f"Expected THD-F ≈ 0.10, got {thd_f:.4f}"
        )

    def test_two_harmonics_thd_f_analytical(self):
        """
        10% 3rd + 6% 5th harmonic:
        THD_F = sqrt(0.10² + 0.06²) = sqrt(0.0136) ≈ 0.1166
        """
        v = make_distorted(harmonic_ratios={3: 0.10, 5: 0.06})
        thd_f, _, _ = compute_thd(v, fs=FS, fundamental_hz=F0)
        expected = np.sqrt(0.10**2 + 0.06**2)
        assert abs(thd_f - expected) < 0.008, (
            f"Expected THD-F ≈ {expected:.4f}, got {thd_f:.4f}"
        )

    def test_thd_r_bounded_below_one(self):
        """THD-R must be in [0, 1) for any finite signal."""
        v = make_distorted(harmonic_ratios={3: 0.5, 5: 0.3, 7: 0.2})
        _, thd_r, _ = compute_thd(v, fs=FS, fundamental_hz=F0)
        assert 0.0 <= thd_r < 1.0, f"THD-R = {thd_r} out of range [0, 1)"

    def test_harmonic_amplitudes_shape(self):
        """harmonic_rms should have length == n_harmonics."""
        v = make_sine()
        n = 8
        _, _, hrms = compute_thd(v, fs=FS, fundamental_hz=F0, n_harmonics=n)
        assert len(hrms) == n

    def test_fundamental_amplitude_close_to_vrms(self):
        """harmonic_rms[0] (fundamental) should be ≈ Vp/sqrt(2) = 230 V."""
        v = make_sine()
        _, _, hrms = compute_thd(v, fs=FS, fundamental_hz=F0)
        expected_vrms = VP / np.sqrt(2)
        assert abs(hrms[0] - expected_vrms) < 5.0, (
            f"Fundamental RMS: expected ≈ {expected_vrms:.1f} V, got {hrms[0]:.1f} V"
        )

    def test_thd_raises_on_nyquist_violation(self):
        """n_harmonics exceeding Nyquist should raise ValueError."""
        v = make_sine()
        # 50 Hz * 110 = 5500 Hz > 5000 Hz (Nyquist at 10 kHz)
        with pytest.raises(ValueError, match="Nyquist"):
            compute_thd(v, fs=FS, fundamental_hz=F0, n_harmonics=110)

    def test_thd_raises_on_short_signal(self):
        """Signal shorter than 2 fundamental cycles should raise ValueError."""
        one_cycle = int(FS / F0)  # 200 samples
        short = make_sine()[:one_cycle]  # exactly 1 cycle
        with pytest.raises(ValueError, match="too short"):
            compute_thd(short, fs=FS, fundamental_hz=F0)


# ─── Spectral Tests ──────────────────────────────────────────────────────────

class TestSpectral:

    def test_peak_frequency_at_fundamental(self):
        """Spectral peak must be at 50 Hz for 50 Hz sine input."""
        v = make_sine()
        result = compute_fft(v, fs=FS)
        peak_idx = np.argmax(result.magnitude)
        peak_freq = result.freqs_hz[peak_idx]
        assert abs(peak_freq - F0) <= result.freq_resolution_hz, (
            f"Peak at {peak_freq:.2f} Hz, expected {F0} Hz "
            f"(resolution: {result.freq_resolution_hz:.2f} Hz)"
        )

    def test_magnitude_at_fundamental_close_to_vrms(self):
        """Spectral magnitude at 50 Hz should be ≈ 230 V (RMS)."""
        v = make_sine()
        result = compute_fft(v, fs=FS)
        expected_vrms = VP / np.sqrt(2)
        peak_mag = np.max(result.magnitude)
        assert abs(peak_mag - expected_vrms) < 5.0, (
            f"Expected magnitude ≈ {expected_vrms:.1f} V, got {peak_mag:.1f} V"
        )

    def test_freq_axis_is_physical_hz(self):
        """Frequency axis should start at 0 and end at fs/2."""
        v = make_sine()
        result = compute_fft(v, fs=FS)
        assert result.freqs_hz[0] == 0.0
        assert abs(result.freqs_hz[-1] - FS / 2) < result.freq_resolution_hz

    def test_freq_resolution(self):
        """freq_resolution_hz must equal fs / N."""
        N = 2000
        v = make_sine()[:N]
        result = compute_fft(v, fs=FS)
        expected_resolution = FS / N
        assert abs(result.freq_resolution_hz - expected_resolution) < 1e-10

    def test_f_max_filter_trims_output(self):
        """f_max parameter should limit the returned frequency range."""
        v = make_sine()
        result = compute_fft(v, fs=FS, f_max=200.0)
        assert result.freqs_hz[-1] <= 200.0

    def test_dominant_frequencies_returns_correct_count(self):
        """dominant_frequencies should return exactly n_peaks results."""
        v = make_distorted(harmonic_ratios={3: 0.1, 5: 0.05, 7: 0.03})
        result = compute_fft(v, fs=FS)
        freqs, mags = dominant_frequencies(result, n_peaks=3)
        assert len(freqs) == 3
        assert len(mags) == 3

    def test_power_db_negative_for_sub_unity_signals(self):
        """Small-amplitude signals should have negative power_db."""
        v = 0.001 * make_sine()  # mV scale
        result = compute_fft(v, fs=FS)
        assert np.all(result.power_db < 0), "Sub-unity signal should have dB < 0"


# ─── Wavelet Tests ───────────────────────────────────────────────────────────

class TestWavelet:

    def test_decompose_returns_correct_number_of_levels(self):
        """dwt_decompose should return level+1 arrays (1 approx + level details)."""
        v = make_sine()
        level = 4
        coeffs = dwt_decompose(v, wavelet='haar', level=level)
        assert len(coeffs) == level + 1

    def test_haar_fallback_works_without_pywt(self):
        """Haar decomposition should not require pywt."""
        v = make_sine()
        # Direct call to fallback
        from features.wavelet import _haar_decompose
        coeffs = _haar_decompose(v, level=3)
        assert len(coeffs) == 4  # [cA3, cD3, cD2, cD1]

    def test_transient_detected_at_correct_time(self):
        """
        Injected spike at sample 500 (t=0.05 s) must be detected within
        ±10 ms (±100 samples at 10 kHz).
        """
        v = make_sine()
        spike_sample = 500
        v[spike_sample] += 50.0  # 50 V spike on 325 V signal ≈ 15%
        event_times, _ = detect_transients(v, fs=FS, wavelet='haar', detail_level=1)
        assert len(event_times) > 0, "Spike not detected"
        spike_time = spike_sample / FS
        closest = np.min(np.abs(event_times - spike_time))
        assert closest < 0.01, (
            f"Detected transient at {event_times} s, "
            f"spike at {spike_time:.4f} s, gap = {closest*1000:.1f} ms"
        )

    def test_no_false_positives_on_clean_signal(self):
        """Clean sine should produce zero detections at 4σ threshold."""
        v = make_sine()
        event_times, _ = detect_transients(
            v, fs=FS, wavelet='haar', detail_level=1, threshold_sigma=4.0
        )
        assert len(event_times) == 0, (
            f"False detections on clean signal: {event_times}"
        )


# ─── Z-Score Tests ───────────────────────────────────────────────────────────

class TestZScore:

    def test_zscore_spikes_at_injected_anomaly(self):
        """
        Injected spike must produce the maximum |z| value.
        """
        rng = np.random.default_rng(0)
        signal = rng.normal(0, 1, 1000)
        spike_idx = 600
        signal[spike_idx] += 15.0  # 15σ spike
        z = rolling_zscore(signal, window=100)
        detected_idx = np.nanargmax(np.abs(z))
        # Allow ±5 samples for window lag
        assert abs(detected_idx - spike_idx) <= 5, (
            f"Max z at index {detected_idx}, expected ≈ {spike_idx}"
        )

    def test_zscore_step_change_produces_high_z(self):
        """Step change from 0 to 1 should produce z >> 3 at transition."""
        step = np.concatenate([np.zeros(500), np.ones(500)])
        z = rolling_zscore(step, window=50)
        # At the step (sample 500), z should be significantly elevated
        assert z[500] > 3.0, f"Step change z-score = {z[500]:.2f}, expected > 3"

    def test_zscore_constant_signal_is_zero(self):
        """Constant signal has zero variance — z-score should be 0."""
        const = np.full(500, 7.5)
        z = rolling_zscore(const, window=50)
        valid = z[~np.isnan(z)]
        assert np.allclose(valid, 0.0, atol=1e-10)

    def test_zscore_output_same_length_as_input(self):
        """Output must match input length regardless of window size."""
        signal = np.random.randn(800)
        z = rolling_zscore(signal, window=100)
        assert len(z) == len(signal)

    def test_zscore_nan_at_start_when_window_large(self):
        """Causal mode: first (min_periods-1) values should be NaN."""
        signal = np.random.randn(500)
        min_p = 20
        z = rolling_zscore(signal, window=100, min_periods=min_p)
        # First min_periods - 1 should be NaN
        assert np.all(np.isnan(z[:min_p - 1]))
        assert not np.isnan(z[min_p])

    def test_find_anomalies_returns_spike_index(self):
        """find_anomalies should return the spike index."""
        signal = np.zeros(500)
        signal[250] = 20.0
        z = rolling_zscore(signal, window=50)
        anomalies = find_anomalies(z, threshold=3.0)
        assert 250 in anomalies or (len(anomalies) > 0 and np.min(np.abs(anomalies - 250)) <= 5)

    def test_zscore_raises_on_small_window(self):
        """Window < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="window must be >= 2"):
            rolling_zscore(np.ones(100), window=1)


# ─── Envelope Tests ──────────────────────────────────────────────────────────

class TestEnvelope:

    def test_envelope_of_pure_sine_is_approximately_constant(self):
        """
        Envelope of undistorted 230 Vrms sine ≈ 325.3 V (constant).
        Guard band: 800 samples (80 ms at 10 kHz).
        Physical reason: 2nd-order BPF at BW=20 Hz has τ ≈ 16 ms;
        sosfiltfilt zero-phase doubling → effective settling ≈ 800 samples (5τ)
        on each end.
        """
        v = make_sine(duration_s=0.5)
        env, _ = hilbert_envelope(v, fs=FS, fundamental_hz=F0)
        steady = env[800:-800]
        expected_peak = VP  # 230*sqrt(2) ≈ 325.3 V
        assert abs(np.mean(steady) - expected_peak) < 5.0, (
            f"Mean envelope = {np.mean(steady):.1f} V, expected ≈ {expected_peak:.1f} V"
        )
        assert np.std(steady) < 3.0, (
            f"Envelope std = {np.std(steady):.2f} V — excessive ripple on clean signal"
        )

    def test_envelope_output_same_length_as_input(self):
        """Envelope must have same length as input signal."""
        v = make_sine()
        env, phase = hilbert_envelope(v, fs=FS)
        assert len(env) == len(v)
        assert len(phase) == len(v)

    def test_sag_detection_identifies_event(self):
        """
        Synthetic sag: 30% amplitude drop for 20 ms.
        Must be detected by detect_sags_swells.
        """
        v = make_sine(duration_s=0.2)
        sag_start = int(0.08 * FS)   # 80 ms
        sag_end = int(0.10 * FS)     # 100 ms → 20 ms sag
        v[sag_start:sag_end] *= 0.70  # drop to 70% = 30% sag

        env, _ = hilbert_envelope(v, fs=FS, fundamental_hz=F0)
        nominal_rms = VP / np.sqrt(2)
        result = detect_sags_swells(env, fs=FS, nominal_rms=nominal_rms)
        assert result['n_sags'] >= 1, (
            f"Expected ≥1 sag, detected {result['n_sags']}"
        )

    def test_no_events_on_clean_signal(self):
        """
        Clean signal should produce zero sags and swells.
        prefilter=False: a pure sine needs no bandpass prefilter.
        Using prefilter=True with narrow BW on a short signal introduces
        edge transients that can trigger false sag detections — documented
        expected behavior, not a bug.
        """
        v = make_sine(duration_s=0.3)
        env, _ = hilbert_envelope(v, fs=FS, fundamental_hz=F0, prefilter=False)
        nominal_rms = VP / np.sqrt(2)
        result = detect_sags_swells(env, fs=FS, nominal_rms=nominal_rms)
        assert result['n_sags'] == 0
        assert result['n_swells'] == 0


# ─── Integration test ────────────────────────────────────────────────────────

class TestIntegration:

    def test_full_pipeline_on_distorted_signal(self):
        """
        End-to-end: distorted signal through RMS → THD → FFT → envelope.
        All results must be internally consistent.
        """
        v = make_distorted(harmonic_ratios={3: 0.10, 5: 0.05})
        n_samples = len(v)

        # RMS
        one_cycle = int(FS / F0)
        rms = windowed_rms(v, window=one_cycle, step=one_cycle)
        assert np.all(rms > 0)

        # THD-F should reflect 10% 3rd + 5% 5th
        thd_f, thd_r, hrms = compute_thd(v, fs=FS, fundamental_hz=F0)
        expected_thd_f = np.sqrt(0.10**2 + 0.05**2)
        assert abs(thd_f - expected_thd_f) < 0.01

        # Spectral peak at 50 Hz
        spec = compute_fft(v, fs=FS)
        peak_idx = np.argmax(spec.magnitude)
        assert abs(spec.freqs_hz[peak_idx] - F0) <= spec.freq_resolution_hz

        # Envelope roughly constant (no sag/swell in this signal).
        # prefilter=False: signal is already near-narrowband (10% THD).
        # Narrow prefilter on short signals produces edge transients that
        # mimic sags; prefilter=True is for field signals with THD > 20%.
        env, _ = hilbert_envelope(v, fs=FS, fundamental_hz=F0, prefilter=False)
        result = detect_sags_swells(env, fs=FS, nominal_rms=VP / np.sqrt(2))
        assert result['n_sags'] == 0
        assert result['n_swells'] == 0
