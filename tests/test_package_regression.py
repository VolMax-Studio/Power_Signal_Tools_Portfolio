"""Checks the package users install, not historical repo-root modules."""
import numpy as np
import power_signal_tools as pst
from power_signal_tools.features.thd import compute_thd

FS = 10_000.0
F0 = 50.0

def _signal(v_peak=1.0):
    t = np.arange(0.0, 0.2, 1.0 / FS)
    return v_peak * (np.sin(2*np.pi*F0*t) + 0.1*np.sin(2*np.pi*3*F0*t))

def test_package_public_exports_resolve_to_canonical_implementation():
    assert callable(pst.windowed_rms)
    assert callable(pst.compute_thd)
    assert callable(pst.compute_fft)
    assert callable(pst.hilbert_envelope)
    assert pst.compute_thd is compute_thd
    assert pst.compute_thd.__module__ == "power_signal_tools.features.thd"

def test_thd_f_and_r_known_10_percent_third_harmonic():
    thd_f, thd_r, harmonic_rms = compute_thd(_signal(), fs=FS, fundamental_hz=F0)
    assert abs(thd_f - 0.1) < 2e-5
    assert abs(thd_r - 0.1 / np.sqrt(1.0 + 0.1**2)) < 2e-5
    assert abs(harmonic_rms[2] / harmonic_rms[0] - 0.1) < 2e-5

def test_thd_ratios_invariant_under_signal_scale():
    low = compute_thd(_signal(1.0), fs=FS, fundamental_hz=F0)
    high = compute_thd(_signal(230.0 * np.sqrt(2)), fs=FS, fundamental_hz=F0)
    np.testing.assert_allclose(low[:2],high[:2],rtol=1e-10,atol=1e-12)
