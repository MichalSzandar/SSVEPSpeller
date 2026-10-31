"""Tests for ssvep.evaluation."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ssvep import config
from ssvep.cca import CCA
from ssvep.datasets import make_synthetic
from ssvep.evaluation import itr, leave_one_block_out, window_sweep
from ssvep.preprocessing import bandpass, crop
from ssvep.trca import TRCA

fs = config.BENCHMARK_FS


def filtered(X):
    """Raw synthetic epochs -> band-passed, uncropped (n_trials, 1, n_channels, n_samples)."""
    return bandpass(X, fs, *config.BANDPASS)[:, None]


def shuffled_synthetic(snr_db, seed, freqs=config.TARGET_FREQS):
    """make_synthetic, band-passed, with the trials in random order, so that the trials of
    one block are not next to each other."""
    X, y, blocks = make_synthetic(freqs=freqs, snr_db=snr_db, seed=seed)
    order = np.random.default_rng(seed).permutation(len(y))
    return filtered(X)[order], y[order], blocks[order]


class Memoriser(BaseEstimator, ClassifierMixin):
    """Answers correctly for trials it was trained on and -1 for every other trial."""

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.X_, self.y_ = X.copy(), y.copy()
        return self

    def predict(self, X):
        answers = np.full(len(X), -1)
        for i, x in enumerate(X):
            for x_train, y_train in zip(self.X_, self.y_):
                if np.array_equal(x, x_train):
                    answers[i] = y_train
        return answers


def test_itr_perfect_accuracy():
    # P = 1 -> B = log2(N) bits per selection, one selection per second.
    assert itr(n_targets=40, accuracy=1.0, trial_time=1.0) == pytest.approx(
        60 * np.log2(40)
    )


def test_itr_zero_accuracy():
    # P = 0 -> B = 0 bits per selection.
    assert itr(n_targets=40, accuracy=0.0, trial_time=1.0) == 0.0


def test_itr_raises_value_error_for_invalid_arguments():
    with pytest.raises(ValueError):
        itr(n_targets=0, accuracy=0.5, trial_time=1.0)
    with pytest.raises(ValueError):
        itr(n_targets=40, accuracy=-0.1, trial_time=1.0)
    with pytest.raises(ValueError):
        itr(n_targets=40, accuracy=1.1, trial_time=1.0)
    with pytest.raises(ValueError):
        itr(n_targets=40, accuracy=0.5, trial_time=0.0)


def test_itr_at_chance_is_zero():
    """N = 5, P = 0.2 -> 0 bits/min."""
    assert itr(n_targets=5, accuracy=0.2, trial_time=1.0) == 0.0


def test_itr_below_chance_is_zero():
    """N = 5, P = 0.0 or 0.1 -> 0 bits/min (the raw formula would be positive here)."""
    assert itr(n_targets=5, accuracy=0.0, trial_time=1.0) == 0.0
    assert itr(n_targets=5, accuracy=0.1, trial_time=1.0) == 0.0


def test_itr_increases_with_accuracy():
    """For fixed N and T, ITR is strictly increasing for P in (1/N, 1]."""
    values = [
        itr(n_targets=5, accuracy=p, trial_time=1.0) for p in np.linspace(0.21, 1, 80)
    ]
    assert np.all(np.diff(values) > 0)


# --- leave_one_block_out -------------------------------------------------------


def test_leave_one_block_out_on_easy_data(synthetic_easy):
    """CCA on high-SNR synthetic data -> accuracy close to 1.0."""
    X, y, blocks = synthetic_easy
    X = crop(filtered(X), fs, start=0.5, duration=1.0)
    assert leave_one_block_out(CCA(config.TARGET_FREQS, fs), X, y, blocks) > 0.95


def test_leave_one_block_out_matches_folds_done_by_hand():
    """The mean over blocks b of: a new model fitted on all trials outside block b, scored
    on the trials of block b. Returned as a float."""
    X, y, blocks = shuffled_synthetic(snr_db=-20.0, seed=4)
    X = crop(X, fs, start=0.5, duration=0.5)
    per_block = []
    for b in np.unique(blocks):
        test = blocks == b
        clf = TRCA(ensemble=True).fit(X[~test], y[~test])
        per_block.append(clf.score(X[test], y[test]))
    assert (
        len(set(per_block)) > 1
    ), "folds should differ, otherwise this test proves little"
    result = leave_one_block_out(TRCA(ensemble=True), X, y, blocks)
    assert isinstance(result, float)
    assert result == pytest.approx(np.mean(per_block))


def test_leave_one_block_out_never_tests_on_training_trials():
    """A classifier that only recognises the trials it was trained on scores exactly 0."""
    X, y, blocks = shuffled_synthetic(snr_db=0.0, seed=5)
    assert leave_one_block_out(Memoriser(), X, y, blocks) == 0.0


def test_leave_one_block_out_leaves_the_estimator_unfitted(synthetic_easy):
    """Every fold uses a clone, so the estimator passed in is not fitted afterwards."""
    X, y, blocks = synthetic_easy
    clf = TRCA()
    leave_one_block_out(clf, crop(filtered(X), fs, start=0.5, duration=0.5), y, blocks)
    assert not hasattr(clf, "templates_")


# --- window_sweep --------------------------------------------------------------


def test_window_sweep_returns_one_row_per_window():
    """A DataFrame with columns window, accuracy, itr: one row per window, in the order
    given."""
    X, y, blocks = shuffled_synthetic(snr_db=-10.0, seed=6)
    windows = [1.0, 0.25, 0.5]
    df = window_sweep(TRCA, X, y, blocks, fs, start=0.5, windows=windows)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["window", "accuracy", "itr"]
    assert df["window"].tolist() == windows


@pytest.mark.parametrize("gaze_shift", [None, 0.3])
def test_window_sweep_matches_lobo_on_each_window(gaze_shift):
    """Row for window T: accuracy = leave_one_block_out of a new estimator on the data
    cropped to [start, start + T); itr = itr(number of targets in y, accuracy,
    T + gaze_shift). gaze_shift defaults to 0.5 s. Here there are 3 targets, not 5."""
    target_freqs = [8.0, 10.0, 12.0]
    X, y, blocks = shuffled_synthetic(snr_db=-20.0, seed=7, freqs=target_freqs)
    windows = [0.25, 0.5, 1.0]

    def make_etrca():
        return TRCA(ensemble=True)

    if gaze_shift is None:
        df = window_sweep(make_etrca, X, y, blocks, fs, start=0.5, windows=windows)
        gaze_shift = 0.5
    else:
        df = window_sweep(
            make_etrca,
            X,
            y,
            blocks,
            fs,
            start=0.5,
            windows=windows,
            gaze_shift=gaze_shift,
        )
    expected = [
        leave_one_block_out(make_etrca(), crop(X, fs, 0.5, T), y, blocks)
        for T in windows
    ]
    assert len(set(expected)) == len(windows), "windows should differ in accuracy"
    assert df["accuracy"].tolist() == pytest.approx(expected)
    expected_itr = [itr(3, acc, T + gaze_shift) for acc, T in zip(expected, windows)]
    assert df["itr"].tolist() == pytest.approx(expected_itr)


def test_window_sweep_starts_at_start(synthetic_easy):
    """Windows begin at ``start``. From the stimulus onset (0.5 s in make_synthetic) the
    accuracy is high; a window inside the pre-stimulus noise is near chance."""
    X, y, blocks = synthetic_easy
    X = filtered(X)
    onset = window_sweep(TRCA, X, y, blocks, fs, start=0.5, windows=[0.25])
    before = window_sweep(TRCA, X, y, blocks, fs, start=0.0, windows=[0.25])
    assert onset["accuracy"].iloc[0] > 0.9
    assert before["accuracy"].iloc[0] < 0.6


def test_window_sweep_rejects_windows_longer_than_the_data():
    """A window running past the end of the epochs is a ValueError, never a silently
    shorter window."""
    X, y, blocks = shuffled_synthetic(snr_db=0.0, seed=8)
    with pytest.raises(ValueError):
        window_sweep(TRCA, X, y, blocks, fs, start=0.5, windows=[0.5, 10.0])
