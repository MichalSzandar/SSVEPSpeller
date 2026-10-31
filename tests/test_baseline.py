"""Tests for ssvep.baseline."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ssvep import config
from ssvep.baseline import make_psd_lda, psd_features
from ssvep.evaluation import leave_one_block_out
from ssvep.preprocessing import bandpass, crop

fs = config.BENCHMARK_FS
freqs = config.TARGET_FREQS


def prepare(X, duration):
    """Raw synthetic epochs -> single-band classifier input, from the onset at 0.5 s."""
    X = bandpass(X, fs, *config.BANDPASS)[:, None]
    return crop(X, fs, start=0.5, duration=duration)


def noise_trials(n_trials=12, n_channels=3, duration=1.0, seed=0):
    """Gaussian noise in classifier format (n_trials, 1, n_channels, n_samples)."""
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_trials, 1, n_channels, int(round(duration * fs))))


# --- psd_features --------------------------------------------------------------


@pytest.mark.parametrize("n_harmonics", [1, 2, 3])
def test_psd_features_shape(n_harmonics):
    """One feature per target frequency x harmonic x channel."""
    F = psd_features(noise_trials(n_channels=3), fs, freqs, n_harmonics=n_harmonics)
    assert F.shape == (12, len(freqs) * n_harmonics * 3)


def test_psd_features_are_log_power():
    """Scaling the signal by 10 multiplies every power by 100, so every feature shifts by
    the same nonzero constant (2 * log(10) in whatever log base is used)."""
    X = noise_trials()
    shift = psd_features(10 * X, fs, freqs) - psd_features(X, fs, freqs)
    assert np.all(np.abs(shift) > 0.5)
    assert np.allclose(shift, shift.flat[0], rtol=1e-6)


@pytest.mark.parametrize("duration", [0.25, 0.5, 1.0, 2.0])
def test_psd_features_measure_every_frequency_separately(duration):
    """No feature is a copy of another one: each target frequency and harmonic is read at
    its own frequency, also in short windows whose spectral bins are wider than the 1 Hz
    gap between 14 and 15 Hz. (The 5 targets with 2 harmonics share no frequency.)"""
    F = psd_features(
        noise_trials(n_trials=20, n_channels=2, duration=duration), fs, freqs
    )
    n = F.shape[1]
    copies = [
        (a, b)
        for a in range(n)
        for b in range(a + 1, n)
        if np.allclose(F[:, a], F[:, b])
    ]
    assert not copies, f"{len(copies)} pairs of features are identical"


def test_psd_features_of_a_trial_do_not_depend_on_other_trials():
    """Features are computed per trial: a trial gives the same features alone or in a batch."""
    X = noise_trials(seed=2)
    assert np.allclose(psd_features(X[3:4], fs, freqs), psd_features(X, fs, freqs)[3:4])


def test_psd_features_reject_filter_bank_input():
    """Input with more than one band is a ValueError, not silently reduced to band 1."""
    X = np.repeat(noise_trials(), 3, axis=1)
    with pytest.raises(ValueError):
        psd_features(X, fs, freqs)


def test_psd_features_flat_channel_gives_finite_features():
    """A flat channel (lost electrode contact) gives finite features: no -inf, no NaN."""
    X = noise_trials()
    X[:, :, 1] = 0.0
    assert np.all(np.isfinite(psd_features(X, fs, freqs)))


# --- make_psd_lda --------------------------------------------------------------


def test_make_psd_lda_is_a_scaled_shrinkage_lda_pipeline():
    """A Pipeline that can be cloned, ending with StandardScaler -> LDA with shrinkage."""
    clf = clone(make_psd_lda(fs, freqs))
    assert isinstance(clf, Pipeline)
    scaler, lda = clf.steps[-2][1], clf.steps[-1][1]
    assert isinstance(scaler, StandardScaler)
    assert isinstance(lda, LinearDiscriminantAnalysis)
    assert lda.shrinkage is not None


@pytest.mark.parametrize("duration", [0.5, 1.0, 2.0])
def test_psd_lda_easy_data(synthetic_easy, duration):
    """High-SNR synthetic data: leave-one-block-out accuracy > 0.9."""
    X, y, blocks = synthetic_easy
    X = prepare(X, duration)
    assert leave_one_block_out(make_psd_lda(fs, freqs), X, y, blocks) > 0.9


def test_psd_lda_noise_only_is_near_chance(synthetic_noise_only):
    """Leave-one-block-out accuracy well below 0.6 (chance is 1/5)."""
    X, y, blocks = synthetic_noise_only
    X = prepare(X, 1.0)
    assert leave_one_block_out(make_psd_lda(fs, freqs), X, y, blocks) < 0.6
