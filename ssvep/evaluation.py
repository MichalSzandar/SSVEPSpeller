"""Metrics and cross-validation.

ITR (Wolpaw et al. 2002):
    B   = log2(N) + P*log2(P) + (1-P)*log2((1-P)/(N-1))   bits per selection
    ITR = B * 60 / T                                       bits per minute
N = number of targets, P = accuracy, T = seconds per selection (window + gaze shift).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from ssvep.preprocessing import crop


def itr(n_targets: int, accuracy: float, trial_time: float) -> float:
    """Information transfer rate in bits/min; 0 for accuracy at or below chance (1/N).

    n_targets : number of targets (N)
    accuracy  : proportion of correct selections (P)
    trial_time: seconds per selection (window + gaze shift) (T)

    Raises ValueError if any argument is out of range.
    """

    if n_targets <= 0:
        raise ValueError(f"n_targets must be positive, got {n_targets}")

    if accuracy < 0.0 or accuracy > 1.0:
        raise ValueError(f"accuracy must be in [0, 1], got {accuracy}")

    if trial_time <= 0.0:
        raise ValueError(f"trial_time must be positive, got {trial_time}")

    chance = 1.0 / n_targets
    if accuracy <= chance:
        return 0.0

    p = float(accuracy)
    if p >= 1.0:
        return np.log2(n_targets) * 60.0 / trial_time

    bits_per_selection = (
        np.log2(n_targets)
        + p * np.log2(p)
        + (1.0 - p) * np.log2((1.0 - p) / (n_targets - 1))
    )
    return bits_per_selection * 60.0 / trial_time


def leave_one_block_out(
    estimator, X: np.ndarray, y: np.ndarray, blocks: np.ndarray
) -> float:
    """Mean accuracy over folds, each fold leaving out one block.

    The estimator is cloned for every fold; the one passed in stays unfitted.
    """
    scores = cross_val_score(
        estimator, X, y, groups=blocks, cv=LeaveOneGroupOut(), error_score="raise"
    )
    return float(scores.mean())


def window_sweep(
    make_estimator,
    X: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    fs: float,
    start: float,
    windows: list[float],
    gaze_shift: float = 0.5,
) -> pd.DataFrame:
    """Evaluate one method at several window lengths.

    make_estimator : callable returning a fresh, unfitted estimator
    X              : (n_trials, n_bands, n_channels, n_samples), filtered but not cropped
    start          : window start in seconds (onset + latency)

    Returns a DataFrame with columns ``window, accuracy, itr``.
    """
    results = []
    for window in windows:
        X_window = crop(X, fs, start=start, duration=window)
        accuracy = leave_one_block_out(make_estimator(), X_window, y, blocks)
        trial_time = window + (gaze_shift if gaze_shift is not None else 0.0)
        results.append(
            {
                "window": window,
                "accuracy": accuracy,
                "itr": itr(len(np.unique(y)), accuracy, trial_time),
            }
        )
    return pd.DataFrame(results)
