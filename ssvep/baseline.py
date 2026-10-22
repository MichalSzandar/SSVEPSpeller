"""Machine-learning baseline: spectral features + LDA."""

import numpy as np


def psd_features(X: np.ndarray, fs: float, freqs: list[float], n_harmonics: int = 2) -> np.ndarray:
    """Log power at each target frequency and its harmonics, per channel.

    X : (n_trials, 1, n_channels, n_samples)  ->  (n_trials, n_features)
    """
    raise NotImplementedError


def make_psd_lda(fs: float, freqs: list[float], n_harmonics: int = 2):
    """Return a scikit-learn Pipeline: PSD features -> scaling -> shrinkage LDA."""
    raise NotImplementedError
