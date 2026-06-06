"""
THD (Total Harmonic Distortion) Utilities
=========================================
Provides functions to compute THD relative to the fundamental component (THD-F)
or relative to the total RMS (THD-R) of a signal.
"""

import numpy as np

def calculate_thd(x, fs, f_fundamental=None, method='fundamental', max_harmonics=10, search_window_hz=5.0):
    """Computes the Total Harmonic Distortion (THD) of a signal.
    
    Uses FFT to identify the fundamental component and its integer harmonics,
    optionally searching in a small frequency window to account for spectral leakage.
    
    Args:
        x (array_like): Input 1D signal array.
        fs (float): Sampling frequency in Hz.
        f_fundamental (float, optional): Fundamental frequency in Hz. If None,
            it is automatically estimated as the peak frequency in the spectrum.
        method (str, optional): 'fundamental' (THD-F, relative to V1) or
            'rms' (THD-R, relative to V_rms). Defaults to 'fundamental'.
        max_harmonics (int, optional): Maximum harmonic order to include (H). Defaults to 10.
        search_window_hz (float, optional): Search window width around harmonic frequencies
            to find the actual peak, in Hz. Defaults to 5.0 Hz.
            
    Returns:
        float: Total Harmonic Distortion (as a decimal fraction, e.g., 0.05 for 5% THD).
    """
    x = np.asarray(x)
    n = len(x)
    if n == 0:
        raise ValueError("Input signal is empty.")
        
    # Remove DC component
    x_ac = x - np.mean(x)
    
    # Compute FFT
    fft_vals = np.fft.rfft(x_ac)
    fft_freqs = np.fft.rfftfreq(n, d=1/fs)
    magnitudes = np.abs(fft_vals) * 2 / n  # Scale to physical amplitude
    
    # Step 1: Find or estimate fundamental frequency
    if f_fundamental is None:
        # Avoid DC bin (index 0) and find highest peak
        peak_idx = np.argmax(magnitudes[1:]) + 1
        f_fundamental = fft_freqs[peak_idx]
        
    if f_fundamental <= 0:
        raise ValueError("Estimated or provided fundamental frequency must be positive.")
        
    # Step 2: Extract fundamental amplitude (V1)
    bin_width = fs / n
    search_bins = int(np.ceil(search_window_hz / bin_width))
    
    def get_peak_amplitude(target_freq):
        idx_ideal = np.argmin(np.abs(fft_freqs - target_freq))
        start_idx = max(0, idx_ideal - search_bins)
        end_idx = min(len(magnitudes), idx_ideal + search_bins + 1)
        if start_idx >= end_idx:
            return 0.0
        return np.max(magnitudes[start_idx:end_idx])
        
    v1 = get_peak_amplitude(f_fundamental)
    if v1 <= 0:
        return 0.0
        
    # Step 3: Extract harmonic amplitudes (V2 to VH)
    harmonic_squares = []
    for h in range(2, max_harmonics + 1):
        vh = get_peak_amplitude(h * f_fundamental)
        harmonic_squares.append(vh ** 2)
        
    v_harmonics_sum = np.sqrt(np.sum(harmonic_squares))
    
    # Step 4: Calculate THD
    if method == 'fundamental':
        thd = v_harmonics_sum / v1
    elif method == 'rms':
        # V_rms of AC signal
        v_rms = np.sqrt(np.mean(np.square(x_ac)))
        if v_rms <= 0:
            return 0.0
        thd = v_harmonics_sum / v_rms
    else:
        raise ValueError(f"Unknown THD method: {method}. Use 'fundamental' or 'rms'.")
        
    return float(thd)
