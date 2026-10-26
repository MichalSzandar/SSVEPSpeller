"""CCA-based SSVEP detection (training-free).

References: Lin et al. 2006 (CCA for SSVEP), Chen et al. 2015 (FBCCA).

Input convention for every classifier in this package:

    X : (n_trials, n_bands, n_channels, n_samples), already filtered and cropped

With n_bands == 1 this class is standard CCA; with filter-bank input it is FBCCA,
the sub-band scores being combined with filter-bank weights.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def reference_signals(
    freq: float, fs: float, n_samples: int, n_harmonics: int = 5
) -> np.ndarray:
    """Sine/cosine references for one stimulus frequency -> (2 * n_harmonics, n_samples).

    Rows: sin(2*pi*h*f*t) and cos(2*pi*h*f*t) for h = 1..n_harmonics, t = n / fs.
    """
    raise NotImplementedError


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between X (n_channels, n_samples) and Y (n_refs, n_samples)."""
    raise NotImplementedError


class CCA(BaseEstimator, ClassifierMixin):
    """Standard CCA (n_bands == 1) or FBCCA (filter-bank input)."""

    def __init__(self, freqs, fs, n_harmonics=5, fb_a=1.25, fb_b=0.25):
        self.freqs = freqs
        self.fs = fs
        self.n_harmonics = n_harmonics
        self.fb_a = fb_a
        self.fb_b = fb_b

    def fit(self, X, y=None):
        """Training-free; only sets ``classes_``."""
        raise NotImplementedError

    def decision_function(self, X):
        """Scores (n_trials, n_targets), one correlation per target.

        With n_bands > 1, sub-bands are combined as sum_m w(m) * rho_m**2 (Chen 2015).
        """
        raise NotImplementedError

    def predict(self, X):
        """Target index with the highest score, shape (n_trials,)."""
        raise NotImplementedError
