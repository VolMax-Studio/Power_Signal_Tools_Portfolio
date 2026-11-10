"""Generate reproducible SYNTHETIC 50 Hz + 10% third-harmonic CSV."""
import csv
from pathlib import Path
import sys

import numpy as np

fs = 10000
n = 2000
t = np.arange(n) / fs
voltage = 230.0 * np.sqrt(2.0) * (
    np.sin(2.0 * np.pi * 50.0 * t)
    + 0.1 * np.sin(2.0 * np.pi * 150.0 * t)
)
path = Path(sys.argv[1] if len(sys.argv) > 1 else "demo-synthetic.csv")
with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.writer(handle)
    writer.writerow(["timestamp_s", "voltage_v"])
    writer.writerows(zip(t, voltage))
print(f"Synthetic data written: {path}, {n} samples, fs={fs} Hz")
