"""Independent analytical and input-boundary regression suite."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from power_signal_tools import compute_thd, compute_fft, windowed_rms, rolling_zscore
from power_signal_tools.cli import analyze_file, main


def wave(fs=10_000, n=2571, fundamental=50.0, third=0.1, dc=0.0):
    t = np.arange(n) / fs
    return dc + np.sin(2 * np.pi * fundamental * t) + third * np.sin(2 * np.pi * 3 * fundamental * t)


@pytest.mark.parametrize("n,f0,dc", [
    (2000,50.0,0.0),
    (2571,50.0,0.0),
    (2571,50.2,0.0),
    (2000,50.0,100.0),
    (2400,60.0,1.0),
])
def test_thd_recovers_analytic_harmonics_without_coherent_fft_bins(n,f0,dc):
    thdf, thdr, hrms = compute_thd(wave(n=n,fundamental=f0,dc=dc),fs=10000,fundamental_hz=f0)
    assert thdf == pytest.approx(0.1, abs=1e-10)
    assert thdr == pytest.approx(.1 / np.sqrt(1.01), abs=1e-10)
    assert hrms[2] == pytest.approx(0.1 / np.sqrt(2), abs=1e-10)


def test_thd_harmonic_denominator_excludes_dc_and_residual():
    x = wave(dc=700.0) + .03 * np.sin(2 * np.pi * 73 * np.arange(2571) / 10000)
    f, r, _ = compute_thd(x,fs=10000, fundamental_hz=50)
    assert 0.06 < f < 0.15
    assert 0.06 < r < 0.15


def test_thd_zero_signal_and_no_fundamental_are_explicit():
    f, r, _ = compute_thd(np.zeros(2000),fs=10000)
    assert np.isinf(f)
    assert r == 0.0
    t=np.arange(2000)/10000
    f,r,_=compute_thd(np.sin(2*np.pi*150*t),fs=10000)
    assert np.isinf(f)
    assert r == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs", [
    {"fs": 0}, {"fs": float("nan")}, {"fundamental_hz": -50},
    {"n_harmonics": 1}, {"n_harmonics": 110},
])
def test_thd_rejects_invalid_sample_configuration(kwargs):
    with pytest.raises(ValueError):
        compute_thd(wave(n=2000), **{"fs":10000, "fundamental_hz":50, **kwargs})


@pytest.mark.parametrize("values", [
    [], [1.0,2.0], [0.0]*2000,
])
def test_short_signal_or_configuration_guards(values):
    with pytest.raises(ValueError):
        compute_thd(np.array(values,dtype=float), fs=10000, fundamental_hz=50, n_harmonics=110 if len(values)==2000 else 10)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_input_rejected(bad):
    x=wave(n=2000);x[20]=bad
    with pytest.raises(ValueError, match="non-finite"):
        compute_thd(x,10000)
    with pytest.raises(ValueError, match="finite"):
        compute_fft(x,10000)
    with pytest.raises(ValueError, match="finite"):
        rolling_zscore(x,window=100)


def test_spectrum_nyquist_bin_has_correct_rms():
    x=(-1.0)**np.arange(2000)
    res=compute_fft(x,10000,window=np.ones(2000))
    assert res.magnitude[-1] == pytest.approx(1.0,abs=1e-12)
    assert res.freqs_hz[-1] == pytest.approx(5000)


@pytest.mark.parametrize("signal,fs", [
    (np.array([]),10000), (np.array([1.,2.]),10000),
    (np.arange(20.),0), (np.arange(20.),float("nan"))
])
def test_spectrum_rejects_invalid_data(signal,fs):
    with pytest.raises(ValueError):
        compute_fft(signal,fs)


def test_windowed_rms_input_bounds():
    with pytest.raises(ValueError):
        windowed_rms(np.ones(100),window=0)
    with pytest.raises(ValueError):
        windowed_rms(np.ones(100),window=2,step=0)
    with pytest.raises(ValueError):
        windowed_rms(np.array([1.,float("nan")]),window=2)


def test_large_dc_offset_does_not_change_zscore():
    rng=np.random.default_rng(19)
    x=rng.normal(size=1000)
    z1=rolling_zscore(x,window=100)
    z2=rolling_zscore(x+1e8,window=100)
    np.testing.assert_allclose(z1,z2,atol=3e-7,equal_nan=True)


def test_csv_report_e2e(tmp_path):
    infile=tmp_path/"sample.csv"
    outfile=tmp_path/"report.json"
    t=np.arange(2000)/10000
    x=230*np.sqrt(2)*(np.sin(2*np.pi*50*t)+0.1*np.sin(2*np.pi*150*t))
    with infile.open("w",newline="") as fd:
        writer=csv.writer(fd);writer.writerow(["time_s","signal_v"])
        writer.writerows(zip(t,x))
    assert main([str(infile),"--column","signal_v","--fs","10000",
                 "--fundamental","50","--unit","V","--output",str(outfile)])==0
    result=json.loads(outfile.read_text())
    assert result["schema"]=="volmax.power_signal_tools.report.v1"
    assert result["metrics"]["thd_f_fraction"]==pytest.approx(.1,abs=1e-10)
    assert result["metrics"]["thd_r_fraction_modelled_harmonics"]==pytest.approx(.1/np.sqrt(1.01),abs=1e-10)
    assert result["metrics"]["fundamental_rms"]==pytest.approx(230,abs=1e-9)
    assert len(result["input"]["sha256"])==64
    assert "certification" in result["limitations"][0]


def test_csv_report_fails_closed_on_non_numeric_and_short(tmp_path):
    infile=tmp_path/"bad.csv"
    infile.write_text("signal_v\n1\n2\nnope\n4\n")
    with pytest.raises(ValueError,match="invalid numeric"):
        analyze_file(infile,column="signal_v",fs=10000,fundamental=50)


def test_csv_report_fails_closed_on_missing_column(tmp_path):
    infile=tmp_path/"bad.csv"
    infile.write_text("voltage\n1\n2\n3\n4\n")
    with pytest.raises(ValueError,match="not found"):
        analyze_file(infile,column="signal_v",fs=10000,fundamental=50)


def test_module_cli_help_with_package_on_pythonpath(tmp_path):
    proc=subprocess.run([sys.executable,"-m","power_signal_tools","--help"],cwd=tmp_path,
                        text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                        env={**os.environ,"PYTHONPATH":str(Path(__file__).resolve().parents[1])})
    assert proc.returncode==0,proc.stderr
    assert "--fundamental" in proc.stdout
