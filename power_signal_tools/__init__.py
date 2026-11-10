"""Power Signal Tools: software-only electrical signal analysis."""
from .features.rms import windowed_rms, block_rms
from .features.thd import compute_thd
from .features.spectral import SpectralResult, compute_fft, dominant_frequencies
from .features.wavelet import dwt_decompose, detect_transients
from .anomaly.zscore import rolling_zscore, find_anomalies
from .anomaly.envelope import hilbert_envelope, detect_sags_swells

__version__ = "0.2.0rc1"
__all__ = [
    "windowed_rms", "block_rms", "compute_thd", "SpectralResult",
    "compute_fft", "dominant_frequencies", "dwt_decompose",
    "detect_transients", "rolling_zscore", "find_anomalies",
    "hilbert_envelope", "detect_sags_swells",
]
