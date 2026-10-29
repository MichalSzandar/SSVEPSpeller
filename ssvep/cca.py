"""CCA-based SSVEP detection (training-free).

References: Lin et al. 2006 (CCA for SSVEP), Chen et al. 2015 (FBCCA).

Input convention for every classifier in this package:

    X : (n_trials, n_bands, n_channels, n_samples), already filtered and cropped

With n_bands == 1 this class is standard CCA; with filter-bank input it is FBCCA,
the sub-band scores being combined with filter-bank weights.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.linalg import orth, svd
from ssvep.preprocessing import filter_bank_weights


def reference_signals(
    freq: float, fs: float, n_samples: int, n_harmonics: int = 5
) -> np.ndarray:
    """Sine/cosine references for one stimulus frequency -> (2 * n_harmonics, n_samples).

    Rows: sin(2*pi*h*f*t) and cos(2*pi*h*f*t) for h = 1..n_harmonics, t = n / fs.
    """
    t = np.arange(n_samples) / fs
    refs = []
    for h in range(1, n_harmonics + 1):
        refs.append(np.sin(2 * np.pi * h * freq * t))
        refs.append(np.cos(2 * np.pi * h * freq * t))
    return np.array(refs)


def canonical_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    """Largest canonical correlation between X (n_channels, n_samples) and Y (n_refs, n_samples)."""

    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis=1, keepdims=True)
    X = X.T
    X = orth(X)

    Y = np.asarray(Y, dtype=np.float64)
    Y = Y - np.mean(Y, axis=1, keepdims=True)
    Y = Y.T
    Y = orth(Y)

    corr = svd(X.T @ Y, compute_uv=False)[0]

    return float(min(corr, 1.0))


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
        self.classes_ = np.arange(len(self.freqs))
        return self

    def decision_function(self, X):
        """Scores (n_trials, n_targets), one correlation per target.

        With n_bands > 1, sub-bands are combined as sum_m w(m) * rho_m**2 (Chen 2015).
        """
        n_trials = X.shape[0]
        n_bands = X.shape[1]
        n_targets = len(self.freqs)
        scores = np.zeros((n_trials, n_targets), dtype=np.float32)

        w = filter_bank_weights(n_bands, self.fb_a, self.fb_b)
        for i, freq in enumerate(self.freqs):
            Y = reference_signals(freq, self.fs, X.shape[-1], self.n_harmonics)
            for m in range(n_bands):
                rho = np.array(
                    [canonical_correlation(X[t, m], Y) for t in range(n_trials)]
                )
                if n_bands > 1:
                    scores[:, i] += w[m] * rho**2
                else:
                    scores[:, i] = rho

        return scores

    def predict(self, X):
        """Target index with the highest score, shape (n_trials,)."""
        return np.argmax(self.decision_function(X), axis=1)
