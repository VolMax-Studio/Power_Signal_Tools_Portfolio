"""Command-line CSV to JSON signal report (software measurements only)."""
import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import sys

import numpy as np

from . import __version__
from .features.rms import windowed_rms
from .features.thd import compute_thd
from .features.spectral import compute_fft
from ._validation import positive_finite


def read_csv_column(path: Path, column: str) -> np.ndarray:
    """Read one finite numeric signal column without inferring sampling rate."""
    samples = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or column not in reader.fieldnames:
            raise ValueError(f"CSV column {column!r} not found; available: {reader.fieldnames}")
        if reader.fieldnames.count(column) != 1:
            raise ValueError("ambiguous duplicate CSV column")
        for row_number, row in enumerate(reader, start=2):
            try:
                val = float(row[column])
            except (ValueError, TypeError) as exc:
                raise ValueError(f"invalid numeric value in row {row_number}, column {column!r}") from exc
            if not np.isfinite(val):
                raise ValueError(f"non-finite value in row {row_number}")
            samples.append(val)
            if len(samples) > 250_000:
                raise ValueError("CSV exceeds 250000 samples; analyze shorter stationary segments")
    if len(samples) < 4:
        raise ValueError("CSV must contain at least 4 numeric signal samples")
    return np.asarray(samples, dtype=np.float64)


def analyze_file(path: Path, *, column: str, fs: float, fundamental: float,
                 harmonics: int = 10, unit: str = "unspecified") -> dict:
    """Analyze an evenly sampled column. No sensor validation is implied."""
    fs = positive_finite(fs, "fs")
    fundamental = positive_finite(fundamental, "fundamental")
    x = read_csv_column(path, column)
    f, r, hrms = compute_thd(x, fs=fs, fundamental_hz=fundamental,
                             n_harmonics=harmonics)
    spectrum = compute_fft(x, fs=fs)
    peak_idx = int(np.argmax(spectrum.magnitude[1:]) + 1)
    total_rms = float(windowed_rms(x, window=len(x), step=len(x))[0])
    result = {
        "schema": "volmax.power_signal_tools.report.v1",
        "tool_version": __version__,
        "input": {
            "filename": path.name,
            "sha256": sha256(path.read_bytes()).hexdigest(),
            "column": column,
            "unit": unit,
            "sampling_frequency_hz": fs,
            "fundamental_frequency_hz": fundamental,
            "samples": int(x.size),
            "duration_s": float(x.size / fs),
        },
        "metrics": {
            "total_sample_rms": total_rms,
            "thd_f_fraction": float(f),
            "thd_r_fraction_modelled_harmonics": float(r),
            "fundamental_rms": float(hrms[0]),
            "dominant_fft_bin_hz": float(spectrum.freqs_hz[peak_idx]),
            "harmonics": [
                {"order": i + 1, "frequency_hz": float((i + 1) * fundamental),
                 "rms": float(val)}
                for i, val in enumerate(hrms)
            ],
        },
        "limitations": [
            "Descriptive computational analysis only; no standards certification.",
            "Input authenticity, calibration, units and sampling regularity are not verified.",
            "The supplied fundamental frequency is fixed; no frequency tracking.",
            "THD-R uses only modelled harmonics, excluding DC, residual and interharmonics.",
            "A nonstationary signal should be divided into stationary windows.",
            "FFT peak is a discrete-bin observation and may suffer spectral leakage.",
        ],
    }
    if not np.isfinite(result["metrics"]["thd_f_fraction"]):
        result["metrics"]["thd_f_fraction"] = None
        result["limitations"].append("Fundamental amplitude below numerical threshold; THD-F undefined.")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="volmax-signal-report",
        description="Compute a bounded JSON report from a uniformly sampled CSV column.")
    parser.add_argument("csv", type=Path, help="Input CSV with header")
    parser.add_argument("--column", required=True, help="Numeric channel name")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate, Hz")
    parser.add_argument("--fundamental", type=float, default=50.0,
                        help="Actual fundamental frequency in Hz (default: 50)")
    parser.add_argument("--harmonics", type=int, default=10,
                        help="Highest harmonic order (default: 10)")
    parser.add_argument("--unit", default="unspecified", help="Channel unit (e.g. V or A)")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON filename")
    args = parser.parse_args(argv)
    try:
        if args.output.resolve() == args.csv.resolve():
            raise ValueError("output cannot overwrite the input CSV")
        if args.output.exists():
            raise ValueError("output file exists; choose a new filename")
        report = analyze_file(args.csv, column=args.column, fs=args.fs,
                              fundamental=args.fundamental, harmonics=args.harmonics,
                              unit=args.unit)
        output = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(f"Report written: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
