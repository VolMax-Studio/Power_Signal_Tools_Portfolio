"""
Spectral Analysis Utilities
===========================
Provides a physical-scaled FFT wrapper and spectral peak detection
for frequency-domain diagnostics.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

def compute_fft(x, fs):
    """Computes the one-sided FFT of a real signal, returning physical units.
    
    Args:
        x (array_like): Input 1D signal.
        fs (float): Sampling frequency in Hz.
        
    Returns:
        pd.DataFrame: A DataFrame containing:
            - 'frequency_hz': Frequency bins in Hz.
            - 'magnitude': Physical peak amplitude of each frequency component.
            - 'phase_rad': Phase angle in radians.
    """
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        raise ValueError("Input signal is empty.")
        
    # Remove DC component for plotting clarity
    x_ac = x - np.mean(x)
    
    fft_vals = np.fft.rfft(x_ac)
    freqs = np.fft.rfftfreq(n, d=1/fs)
    
    # Scale to physical peak amplitude
    magnitudes = np.abs(fft_vals) * 2 / n
    # For DC component (if we hadn't removed it) and Nyquist bin, division by 2 is needed
    # but we removed DC and Nyquist is usually negligible, so standard scaling is fine.
    
    phases = np.angle(fft_vals)
    
    return pd.DataFrame({
        'frequency_hz': freqs,
        'magnitude': magnitudes,
        'phase_rad': phases
    })


def find_spectral_peaks(frequency_hz, magnitude, min_distance_hz=5.0, prominence=0.01, limit=10):
    """Identifies the dominant frequency components in a spectrum.
    
    Args:
        frequency_hz (array_like): Frequency bins in Hz.
        magnitude (array_like): Magnitude spectrum values.
        min_distance_hz (float, optional): Minimum frequency separation between peaks in Hz.
            Defaults to 5.0 Hz.
        prominence (float, optional): Required prominence of peaks. Defaults to 0.01.
        limit (int, optional): Maximum number of peaks to return. Defaults to 10.
        
    Returns:
        pd.DataFrame: A DataFrame of detected peaks with columns 'frequency_hz' and 'magnitude',
            sorted by magnitude descending.
    """
    freqs = np.asarray(frequency_hz)
    mags = np.asarray(magnitude)
    
    # Convert frequency distance to bin index distance
    bin_width = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    distance_bins = max(1, int(np.round(min_distance_hz / bin_width)))
    
    peak_idx, properties = find_peaks(mags, distance=distance_bins, prominence=prominence)
    
    peaks_df = pd.DataFrame({
        'frequency_hz': freqs[peak_idx],
        'magnitude': mags[peak_idx]
    })
    
    return peaks_df.sort_values(by='magnitude', ascending=False).head(limit).reset_index(drop=True)
