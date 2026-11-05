"""Signal processing shared by the offline analysis and the online loop.

Processing order (filter first, crop last, so filter edge effects fall outside
the analysis window):

    raw epochs (n_trials, n_channels, n_samples)
      -> notch(50 Hz)
      -> bandpass(...)[:, None]   or   filter_bank(...)
      -> crop(start = onset + latency, duration = window)
      -> classifier input (n_trials, n_bands, n_channels, n_window)

All filters work along the last axis (time) and accept any number of leading axes.

Artifact rejection (bad_channel_mask, bad_trial_mask) works on the classifier input
and only at training time: RejectArtifacts wraps a classifier so that its fit() skips
bad channels and bad trials, while every test trial is still classified.
"""

import numpy as np
from scipy.signal import butter, filtfilt, sosfiltfilt, iirnotch, cheb1ord, cheby1
from sklearn.base import BaseEstimator, ClassifierMixin, clone

STOPBANDS = [
    (4.0, 100.0),
    (10.0, 100.0),
    (16.0, 100.0),
    (24.0, 100.0),
    (32.0, 100.0),
    (40.0, 100.0),
    (48.0, 100.0),
    (56.0, 100.0),
    (64.0, 100.0),
    (72.0, 100.0),
]

PASSBANDS = [
    (6.0, 90.0),
    (14.0, 90.0),
    (22.0, 90.0),
    (30.0, 90.0),
    (38.0, 90.0),
    (46.0, 90.0),
    (54.0, 90.0),
    (62.0, 90.0),
    (70.0, 90.0),
    (78.0, 90.0),
]


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
    m_values = np.arange(1, n_subbands + 1)
    return 1.0 / m_values**a + b


def chebyshev_bandpass(
    X: np.ndarray,
    fs: float,
    passband: tuple[float, float],
    stopband: tuple[float, float],
    gpass: float = 3.0,
    gstop: float = 40.0,
    rp: float = 0.5,
) -> np.ndarray:
    """Zero-phase Chebyshev type I band-pass: keep ``passband``, attenuate beyond ``stopband`` (Hz).

    gpass: max loss in the passband (dB), gstop: min attenuation in the stopband (dB),
    rp: passband ripple (dB). Defaults follow Nakanishi's filterbank.m.
    """
    n, wn = cheb1ord(passband, stopband, gpass, gstop, analog=False, fs=fs)
    sos = cheby1(n, rp, wn, btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, X, axis=-1)


def filter_bank(X: np.ndarray, fs: float, n_subbands: int = 5) -> np.ndarray:
    """Split X into sub-bands (Chen 2015).

    (..., n_channels, n_samples) -> (..., n_subbands, n_channels, n_samples)
    """
    if n_subbands < 1 or n_subbands > len(PASSBANDS):
        raise ValueError(f"n_subbands must be in [1, {len(PASSBANDS)}]")

    subbands = []
    for m in range(n_subbands):
        subbands.append(
            chebyshev_bandpass(X, fs=fs, passband=PASSBANDS[m], stopband=STOPBANDS[m])
        )

    return np.stack(subbands, axis=-3)  # (..., n_subbands, n_channels, n_samples)


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


def trial_ptp_ratio(X: np.ndarray) -> np.ndarray:
    """How far each trial's worst channel swings above normal -> (n_trials,).

    X : (n_trials, n_channels, n_samples), band-passed. Each channel's peak-to-peak
    amplitude in a trial is divided by that channel's median over all trials; a trial's
    ratio is the largest one over its channels (1 = a typical trial, 10 = ten times the
    usual swing). Relative, so it needs no µV threshold: dry-electrode EEG here is a
    few µV, Benchmark tens of µV. A flat channel (median 0) counts as ratio 0.
    """
    ptp = np.ptp(X, axis=-1)  # (n_trials, n_channels)
    median = np.median(ptp, axis=0)  # (n_channels,)
    ratio = np.divide(
        ptp, median, out=np.zeros_like(ptp, dtype=float), where=median > 0
    )
    return ratio.max(axis=1)


def bad_trial_mask(X: np.ndarray, max_ratio: float = 5.0) -> np.ndarray:
    """Boolean mask (n_trials,), True where some channel's peak-to-peak amplitude is more
    than ``max_ratio`` times its median over the trials (see trial_ptp_ratio).

    X : (n_trials, n_channels, n_samples), band-passed. Needs mostly clean trials: the
    median is the reference, so it stops working when half the trials are bad.
    """
    return trial_ptp_ratio(X) > max_ratio


def bad_channel_mask(X: np.ndarray, max_ratio: float = 3.0) -> np.ndarray:
    """Boolean mask (n_channels,), True for abnormally noisy or flat channels.

    X : (n_trials, n_channels, n_samples), band-passed. A channel's level is the median
    over trials of its in-trial standard deviation (robust to a few artifact trials). It
    is bad if that level is more than ``max_ratio`` times, or less than 1/``max_ratio``
    of, the median level over channels. A ratio and not a z-score: with 8 channels the
    spread of the channels is too uncertain to standardise by.
    """
    level = np.median(X.std(axis=-1), axis=0)  # (n_channels,)
    typical = np.median(level)
    return (level > max_ratio * typical) | (level < typical / max_ratio)


class RejectArtifacts(BaseEstimator, ClassifierMixin):
    """Fit ``estimator`` without the bad channels and bad trials of the training data.

    Input convention of the classifiers: X (n_trials, n_bands, n_channels, n_samples);
    the masks look at band 0 (the single band, or the widest filter-bank sub-band).

    fit():   1. drop channels with bad_channel_mask(max_ratio=channel_ratio)
             2. on the remaining channels, drop trials with
                bad_trial_mask(max_ratio=trial_ratio), but keep at least ``min_trials``
                per class (the least bad ones), since TRCA needs 2 trials per target
             3. fit a clone of ``estimator`` on what is left
    predict() / decision_function(): every trial is classified, on the channels kept in
    fit. Online there is no skipping a selection, so test trials are never rejected.

    trial_ratio or channel_ratio set to None (or 0) switches that step off.
    Fitted attributes: ``channels_`` (indices of the kept channels), ``trials_kept_``
    (boolean mask over the training trials), ``estimator_``, ``classes_``.
    """

    def __init__(self, estimator, trial_ratio=5.0, channel_ratio=3.0, min_trials=2):
        self.estimator = estimator
        self.trial_ratio = trial_ratio
        self.channel_ratio = channel_ratio
        self.min_trials = min_trials

    def fit(self, X, y):
        y = np.asarray(y)
        if self.channel_ratio:
            self.channels_ = np.flatnonzero(
                ~bad_channel_mask(X[:, 0], max_ratio=self.channel_ratio)
            )
        else:
            self.channels_ = np.arange(X.shape[2])
        X = X[:, :, self.channels_]

        keep = np.ones(len(y), dtype=bool)
        if self.trial_ratio:
            ratio = trial_ptp_ratio(X[:, 0])
            keep = ratio <= self.trial_ratio
            for label in np.unique(y):
                idx = np.flatnonzero(y == label)
                if keep[idx].sum() < self.min_trials:
                    least_bad = idx[np.argsort(ratio[idx])[: self.min_trials]]
                    keep[least_bad] = True
        self.trials_kept_ = keep

        self.estimator_ = clone(self.estimator).fit(X[keep], y[keep])
        self.classes_ = self.estimator_.classes_
        return self

    def decision_function(self, X):
        """The wrapped classifier's scores, on the channels kept in fit."""
        return self.estimator_.decision_function(X[:, :, self.channels_])

    def predict(self, X):
        """The wrapped classifier's labels, on the channels kept in fit."""
        return self.estimator_.predict(X[:, :, self.channels_])
