"""Task-related component analysis (Nakanishi et al. 2018).

For each target, TRCA learns a spatial filter that maximises the reproducibility of
the signal across training trials, and keeps the average training trial as a
template. A test trial is classified by correlating its filtered signal with each
filtered template.

Same input convention as cca.py: X is (n_trials, n_bands, n_channels, n_samples).

    ensemble=False, n_bands == 1  -> TRCA
    ensemble=True,  n_bands == 1  -> eTRCA   (filters of all targets stacked)
    ensemble=True,  n_bands >  1  -> FB-eTRCA
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def trca_spatial_filter(X: np.ndarray) -> np.ndarray:
    """Spatial filter for one target in one band.

    X : (n_trials, n_channels, n_samples), the training trials of one target.
    Returns w : (n_channels,).
    """
    raise NotImplementedError


class TRCA(BaseEstimator, ClassifierMixin):
    """TRCA / eTRCA, optionally on filter-bank input."""

    def __init__(self, ensemble=False, fb_a=1.25, fb_b=0.25):
        self.ensemble = ensemble
        self.fb_a = fb_a
        self.fb_b = fb_b

    def fit(self, X, y):
        """Learn ``templates_`` (n_targets, n_bands, n_channels, n_samples),
        ``filters_`` (n_bands, n_targets, n_channels) and ``classes_``.

        Test windows must have the same length as the training windows.
        """
        raise NotImplementedError

    def decision_function(self, X):
        """Scores (n_trials, n_targets): correlation between the filtered test trial
        and each filtered template, sub-bands combined with filter-bank weights."""
        raise NotImplementedError

    def predict(self, X):
        """Target index with the highest score, shape (n_trials,)."""
        raise NotImplementedError
