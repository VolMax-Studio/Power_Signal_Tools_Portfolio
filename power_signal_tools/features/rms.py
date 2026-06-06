"""
RMS (Root Mean Square) Utilities
===============================
Provides functions to compute global and rolling RMS values for
single-channel and multi-channel signals.
"""

import numpy as np

def calculate_rms(x, axis=0):
    """Computes the global Root Mean Square (RMS) value along a specified axis.
    
    Args:
        x (array_like): Input signal array.
        axis (int, optional): Axis along which the RMS is computed. Defaults to 0.
        
    Returns:
        float or ndarray: Global RMS value.
    """
    x = np.asarray(x)
    return np.sqrt(np.mean(np.square(x), axis=axis))


def rolling_rms(x, window_size, padding='reflect'):
    """Computes the rolling RMS of a 1D signal or 2D multi-channel signal.
    
    For 2D inputs, the rolling window is applied along the time dimension (axis 0),
    treating each column as an independent channel.
    
    Args:
        x (array_like): Input 1D signal or 2D signal (shape: [samples, channels]).
        window_size (int): Size of the sliding window (must be positive).
        padding (str, optional): Padding mode for boundaries. Supported: 'reflect', 'zero', 'none'.
            Defaults to 'reflect'.
            
    Returns:
        ndarray: Rolling RMS signal. If padding is 'none', shape is reduced by window_size - 1.
    """
    x = np.asarray(x)
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
        
    is_1d = x.ndim == 1
    if is_1d:
        x = x[:, np.newaxis]
        
    samples, channels = x.shape
    
    # Square the signal
    squared = np.square(x)
    
    # Setup convolution kernel
    kernel = np.ones(window_size) / window_size
    
    results = []
    for col in range(channels):
        channel_data = squared[:, col]
        
        if padding == 'none':
            # Valid convolution
            mean_squared = np.convolve(channel_data, kernel, mode='valid')
        elif padding == 'zero':
            # Same convolution, zero padded
            mean_squared = np.convolve(channel_data, kernel, mode='same')
        elif padding == 'reflect':
            # Reflect pad to avoid edge attenuation
            pad_width = window_size // 2
            padded_data = np.pad(channel_data, pad_width, mode='reflect')
            conv = np.convolve(padded_data, kernel, mode='valid')
            
            # Trim to match original length
            if len(conv) > samples:
                conv = conv[:samples]
            elif len(conv) < samples:
                conv = np.pad(conv, (0, samples - len(conv)), mode='edge')
            mean_squared = conv
        else:
            raise ValueError(f"Unsupported padding mode: {padding}")
            
        results.append(np.sqrt(np.maximum(0.0, mean_squared)))
        
    out = np.column_stack(results)
    return out.squeeze() if is_1d else out
