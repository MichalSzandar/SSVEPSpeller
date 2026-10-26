"""Signal processing shared by the offline analysis and the online loop.

Processing order (filter first, crop last, so filter edge effects fall outside
the analysis window):

    raw epochs (n_trials, n_channels, n_samples)
      -> notch(50 Hz)
      -> bandpass(...)[:, None]   or   filter_bank(...)
      -> crop(start = onset + latency, duration = window)
      -> classifier input (n_trials, n_bands, n_channels, n_window)

All filters work along the last axis (time) and accept any number of leading axes.
"""

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch


def notch(
    X: np.ndarray, fs: float, freq: float = 50.0, quality: float = 30.0
) -> np.ndarray:
    """Zero-phase notch filter removing mains interference at ``freq`` Hz."""
    b, a = iirnotch(w0=freq, Q=quality, fs=fs)
    return filtfilt(b, a, X, axis=-1)


def bandpass(
    X: np.ndarray, fs: float, low: float, high: float, order: int = 4
) -> np.ndarray:
    """Zero-phase band-pass filter between ``low`` and ``high`` Hz."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def filter_bank_weights(
    n_subbands: int, a: float = 1.25, b: float = 0.25
) -> np.ndarray:
    """Weights w(m), m = 1..n_subbands, for combining sub-band scores (Chen 2015)."""
    raise NotImplementedError


def filter_bank(X: np.ndarray, fs: float, n_subbands: int = 5) -> np.ndarray:
    """Split X into sub-bands (Chen 2015).

    (..., n_channels, n_samples) -> (..., n_subbands, n_channels, n_samples)
    """
    raise NotImplementedError


def crop(X: np.ndarray, fs: float, start: float, duration: float) -> np.ndarray:
    """Keep the samples in [start, start + duration) seconds along the last axis."""
    ns = int(round(duration * fs))
    start = int(round(start * fs))

    if start < 0 or start + ns > X.shape[-1]:
        raise ValueError(
            f"Crop window [{start/fs:.3f}, {(start+ns)/fs:.3f}) s "
            f"outside the signal length {X.shape[-1]/fs:.3f} s"
        )
    return X[..., start : start + ns]


def bad_trial_mask(X: np.ndarray, max_ptp: float) -> np.ndarray:
    """Boolean mask (n_trials,), True where any channel's peak-to-peak amplitude exceeds ``max_ptp`` µV."""
    raise NotImplementedError


def bad_channel_mask(X: np.ndarray, z_thresh: float = 3.0) -> np.ndarray:
    """Boolean mask (n_channels,), True for flat or abnormally noisy channels."""
    raise NotImplementedError
