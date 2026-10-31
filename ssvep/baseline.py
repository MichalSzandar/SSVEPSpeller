"""Machine-learning baseline: spectral features + LDA."""

import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline


def psd_features(
    X: np.ndarray, fs: float, freqs: list[float], n_harmonics: int = 2
) -> np.ndarray:
    """Log power at each target frequency and its harmonics, per channel.

    X : (n_trials, 1, n_channels, n_samples)  ->  (n_trials, n_features)
        Exactly one band (ValueError otherwise). Features are finite, also for a flat
        channel.
    """
    n_trials, n_bands, n_channels, n_samples = X.shape
    if n_bands != 1:
        raise ValueError(f"Expected 1 band, got {n_bands}")

    n_features = len(freqs) * n_harmonics * n_channels
    features = np.empty((n_trials, n_features), dtype=np.float64)

    # The whole window is one segment (shorter segments drop samples), zero-padded
    # to a 0.1 Hz grid so every target and harmonic is read at its own frequency.
    nfft = max(n_samples, int(round(10 * fs)))
    tiny = np.finfo(np.float64).tiny  # power of a flat channel is 0, and log(0) = -inf

    for i in range(n_trials):
        for j in range(n_channels):
            f, Pxx = welch(X[i, 0, j], fs=fs, nperseg=n_samples, nfft=nfft)
            for k, freq in enumerate(freqs):
                for h in range(1, n_harmonics + 1):
                    idx = np.argmin(np.abs(f - h * freq))
                    features[i, (k * n_harmonics + (h - 1)) * n_channels + j] = np.log(
                        max(Pxx[idx], tiny)
                    )

    return features


def make_psd_lda(fs: float, freqs: list[float], n_harmonics: int = 2):
    """Return a scikit-learn Pipeline: PSD features -> scaling -> shrinkage LDA."""

    return Pipeline(
        [
            (
                "psd",
                FunctionTransformer(
                    psd_features,
                    kw_args={"fs": fs, "freqs": freqs, "n_harmonics": n_harmonics},
                ),
            ),
            ("scaler", StandardScaler()),
            ("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )
