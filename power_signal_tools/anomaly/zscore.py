"""
Rolling Z-Score Anomaly Detection
=================================
Provides functions to calculate rolling statistical Z-scores and
detect transient spikes or structural shifts in time-series data.
"""

import numpy as np
import pandas as pd

def calculate_rolling_zscore(x, window_size, center=True, threshold=3.0):
    """Computes the rolling Z-score and flags anomalies.
    
    Args:
        x (array_like): Input 1D signal.
        window_size (int): Size of the sliding window for rolling stats.
        center (bool, optional): If True, windows are centered. If False,
            windows look backward (causal). Defaults to True.
        threshold (float, optional): Multiplier for standard deviation (e.g. 3.0).
            Defaults to 3.0.
            
    Returns:
        tuple: (z_scores, anomaly_mask) where:
            z_scores (ndarray): Calculated rolling Z-score array (same size as x).
            anomaly_mask (ndarray): Boolean mask indicating if |Z| > threshold.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    if window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
        
    s = pd.Series(x)
    
    # Compute rolling mean and standard deviation
    rolling_mean = s.rolling(window_size, min_periods=1, center=center).mean().values
    rolling_std = s.rolling(window_size, min_periods=1, center=center).std().values
    
    # Handle NaNs or 0 standard deviation (e.g., flat signals)
    rolling_std = np.nan_to_num(rolling_std, nan=0.0)
    rolling_std[rolling_std == 0.0] = 1e-8
    
    # Calculate Z-score
    z_scores = (x - rolling_mean) / rolling_std
    
    # Flag anomalies
    anomaly_mask = np.abs(z_scores) > threshold
    
    return z_scores, anomaly_mask
