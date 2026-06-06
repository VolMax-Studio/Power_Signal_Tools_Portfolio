"""
Hilbert Amplitude Envelope Utilities
====================================
Provides functions to extract the instantaneous amplitude envelope of a signal
using the Hilbert Transform, and detect voltage sags/swells (power fluctuations).
"""

import numpy as np
from scipy.signal import hilbert

def extract_envelope(x):
    """Extracts the instantaneous amplitude envelope of a signal using Hilbert Transform.
    
    Args:
        x (array_like): Input 1D signal.
        
    Returns:
        ndarray: Amplitude envelope of the signal (same size as x).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
        
    # Remove DC component for accurate Hilbert analysis
    mean_val = np.mean(x)
    x_ac = x - mean_val
    
    analytic_signal = hilbert(x_ac)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Restore DC offset to the envelope if applicable
    return amplitude_envelope + np.abs(mean_val)


def detect_sag_swell(x, fs, nominal_amplitude=None, window_cycles=5, sag_threshold=0.9, swell_threshold=1.1, f_grid=50.0):
    """Detects voltage sags and swells using the Hilbert amplitude envelope.
    
    Args:
        x (array_like): Input 1D signal (voltage or current wave).
        fs (float): Sampling frequency in Hz.
        nominal_amplitude (float, optional): The expected peak amplitude of the signal.
            If None, it is estimated using the median of the envelope.
        window_cycles (int, optional): The duration window in grid cycles (e.g. 5 cycles)
            to smooth the envelope and filter transients. Defaults to 5.
        sag_threshold (float, optional): Fractional threshold below nominal to flag a sag (e.g. 0.9).
            Defaults to 0.9.
        swell_threshold (float, optional): Fractional threshold above nominal to flag a swell (e.g. 1.1).
            Defaults to 1.1.
        f_grid (float, optional): Grid system frequency (50 Hz or 60 Hz). Defaults to 50.0.
        
    Returns:
        dict: A dictionary containing:
            - 'envelope': The raw Hilbert envelope.
            - 'smoothed_envelope': The smoothed envelope.
            - 'nominal_amplitude': Used nominal amplitude.
            - 'sag_mask': Boolean mask indicating active sag events.
            - 'swell_mask': Boolean mask indicating active swell events.
    """
    x = np.asarray(x, dtype=float)
    
    # Extract envelope
    env = extract_envelope(x)
    
    # Smooth the envelope using a rolling average filter based on system cycles
    samples_per_cycle = int(np.round(fs / f_grid))
    smoothing_window = max(1, samples_per_cycle * window_cycles)
    
    # Rolling mean smoothing
    kernel = np.ones(smoothing_window) / smoothing_window
    smoothed_env = np.convolve(env, kernel, mode='same')
    
    # Estimate nominal amplitude if not provided
    if nominal_amplitude is None:
        nominal_amplitude = float(np.median(smoothed_env))
        
    if nominal_amplitude <= 0:
        raise ValueError("Nominal amplitude must be positive.")
        
    normalized_env = smoothed_env / nominal_amplitude
    
    sag_mask = normalized_env < sag_threshold
    swell_mask = normalized_env > swell_threshold
    
    return {
        'envelope': env,
        'smoothed_envelope': smoothed_env,
        'nominal_amplitude': nominal_amplitude,
        'sag_mask': sag_mask,
        'swell_mask': swell_mask
    }
