"""
Discrete Wavelet Transform (DWT) Utilities
==========================================
Provides custom filterbank implementations of the Discrete Wavelet Transform (DWT)
supporting Haar and Daubechies 4 (db4) wavelets for multi-level decomposition
and transient detection, implemented with pure NumPy.
"""

import numpy as np

# Wavelet coefficient filterbanks (decomposition filters)
# g: Low-pass filter, h: High-pass filter
WAVELET_FILTERS = {
    'haar': {
        'g': np.array([0.7071067811865476, 0.7071067811865476]),
        'h': np.array([-0.7071067811865476, 0.7071067811865476])
    },
    'db4': {
        'g': np.array([0.3415063505269308, 0.5915063505269308, 0.1584936494730692, -0.0915063505269308]),
        'h': np.array([-0.0915063505269308, -0.1584936494730692, 0.5915063505269308, -0.3415063505269308])
    }
}

def dwt_step(x, wavelet='haar'):
    """Performs a 1-level Discrete Wavelet Transform (DWT) step.
    
    Args:
        x (ndarray): 1D input signal.
        wavelet (str): Wavelet type ('haar' or 'db4'). Defaults to 'haar'.
        
    Returns:
        tuple: (cA, cD) where:
            cA (ndarray): Approximation coefficients (low-pass, decimated).
            cD (ndarray): Detail coefficients (high-pass, decimated).
    """
    if wavelet not in WAVELET_FILTERS:
        raise ValueError(f"Unsupported wavelet: {wavelet}. Choose from {list(WAVELET_FILTERS.keys())}")
        
    filters = WAVELET_FILTERS[wavelet]
    g = filters['g']
    h = filters['h']
    
    # Pad input to handle boundary conditions (using periodic/reflect extension)
    pad_len = len(g) - 1
    x_padded = np.pad(x, pad_len, mode='reflect')
    
    # Convolution
    low_pass = np.convolve(x_padded, g, mode='valid')
    high_pass = np.convolve(x_padded, h, mode='valid')
    
    # Downsample by 2 (decimation)
    cA = low_pass[::2]
    cD = high_pass[::2]
    
    return cA, cD


def dwt_decomposition(x, wavelet='haar', level=3):
    """Performs multi-level Discrete Wavelet Transform (DWT) decomposition.
    
    Decomposes the signal hierarchically into approximation (cA) and detail (cD)
    coefficients up to the specified level.
    
    Args:
        x (array_like): 1D input signal.
        wavelet (str): Wavelet type ('haar' or 'db4'). Defaults to 'haar'.
        level (int): Decomposition level. Defaults to 3.
        
    Returns:
        list: A list of coefficients [cA_L, cD_L, cD_L-1, ..., cD_1] where L is the level.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
        
    coefs = []
    current_signal = x
    
    for l in range(level):
        cA, cD = dwt_step(current_signal, wavelet=wavelet)
        coefs.append(cD)
        current_signal = cA
        
    # Prepend the final approximation coefficients
    coefs.append(current_signal)
    
    # Reverse so the order is [cA_L, cD_L, cD_L-1, ..., cD_1]
    coefs.reverse()
    return coefs


def detect_transients_dwt(x, wavelet='haar', level=1, threshold_factor=3.0):
    """Detects high-frequency transients in a signal using Wavelet detail coefficients.
    
    Transients are identified as points where the absolute detail coefficients
    deviate significantly from the median absolute deviation (MAD).
    
    Args:
        x (array_like): 1D input signal.
        wavelet (str): Wavelet type ('haar' or 'db4'). Defaults to 'haar'.
        level (int): Wavelet decomposition level to check. Defaults to 1.
        threshold_factor (float): Multiplier for the threshold (e.g. 3 * MAD).
            Defaults to 3.0.
            
    Returns:
        tuple: (transient_indices, cD) where:
            transient_indices (ndarray): Indicies in the original signal where
                transients were detected (mapped back from decimated space).
            cD (ndarray): The detail coefficients at the specified level.
    """
    x = np.asarray(x, dtype=float)
    
    # Get multi-level decomposition
    coefs = dwt_decomposition(x, wavelet=wavelet, level=level)
    
    # Detail coefficients at the specified level are at index -level
    cD = coefs[-level]
    
    # Compute Median Absolute Deviation (MAD) of detail coefficients
    abs_cd = np.abs(cD)
    median = np.median(abs_cd)
    mad = np.median(np.abs(abs_cd - median))
    
    # Avoid division by zero
    mad = max(mad, 1e-8)
    
    # Threshold detection
    anomaly_indices_decimated = np.where(abs_cd > (median + threshold_factor * mad))[0]
    
    # Map back to original signal domain (scale index by 2^level)
    scale_factor = 2 ** level
    transient_indices = anomaly_indices_decimated * scale_factor
    
    # Clip to signal boundary
    transient_indices = transient_indices[transient_indices < len(x)]
    
    return transient_indices, cD
