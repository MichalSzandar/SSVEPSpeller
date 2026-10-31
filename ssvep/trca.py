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
from ssvep.preprocessing import filter_bank_weights


def trca_spatial_filter(X: np.ndarray) -> np.ndarray:
    """Spatial filter for one target in one band.

    X : (n_trials, n_channels, n_samples), the training trials of one target. Needs at
        least 2 trials (ValueError otherwise); X itself is not modified.
    Returns w : (n_channels,).
    """
    n_trials, n_channels, n_samples = X.shape

    if n_trials < 2:
        raise ValueError(f"At least 2 trials required, got {n_trials}")

    X_centered = X - np.mean(X, axis=2, keepdims=True)  # center each trial

    Q = np.zeros((n_channels, n_channels))
    for i in range(n_trials):
        Q += X_centered[i] @ X_centered[i].T

    U = np.sum(X_centered, axis=0)
    S = U @ U.T - Q

    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Q) @ S)
    w = eigvecs[:, np.argmax(eigvals.real)].real
    return w / np.linalg.norm(w)


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
        self.classes_ = np.unique(y)
        self.templates_ =  np.zeros((len(self.classes_), X.shape[1], X.shape[2], X.shape[3]))
        self.filters_ = np.zeros((X.shape[1], len(self.classes_), X.shape[2]))

        for i, target in enumerate(self.classes_):
            X_target = X[y == target]
            self.templates_[i] = np.mean(X_target, axis=0)
            for m in range(X.shape[1]):
                self.filters_[m, i] = trca_spatial_filter(X_target[:, m])

        return self

    def decision_function(self, X):
        """Scores (n_trials, n_targets): correlation between the filtered test trial
        and each filtered template, sub-bands combined with filter-bank weights."""

        n_trials = X.shape[0]
        n_bands = X.shape[1]
        n_targets = len(self.classes_)
        scores = np.zeros((n_trials, n_targets), dtype=np.float32)

        w = filter_bank_weights(n_bands, self.fb_a, self.fb_b)
        for i in range(n_targets):
            for m in range(n_bands):
                if self.ensemble:
                    template_filtered = self.filters_[m] @ self.templates_[i, m]
                    trial_filtered = self.filters_[m] @ X[:, m]
                else:
                    template_filtered = self.filters_[m, i] @ self.templates_[i, m]
                    trial_filtered = self.filters_[m, i] @ X[:, m]

                rho = np.array(
                    [np.corrcoef(trial_filtered[t].ravel(), template_filtered.ravel())[0, 1] for t in range(n_trials)]
                )
                if n_bands > 1:
                    scores[:, i] += w[m] * rho
                else:
                    scores[:, i] = rho

        return scores


    def predict(self, X):
        """Label (entry of ``classes_``) with the highest score, shape (n_trials,)."""
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]
