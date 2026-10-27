"""Tests for ssvep.cca and ssvep.trca."""

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from sklearn.base import clone
from sklearn.cross_decomposition import CCA as SklearnCCA

from ssvep import config
from ssvep.cca import CCA, canonical_correlation, reference_signals
from ssvep.preprocessing import bandpass, crop

fs = config.BENCHMARK_FS
freqs = config.TARGET_FREQS


def prepare(X, duration):
    """Raw synthetic epochs -> classifier input (n_trials, 1, n_channels, n_window).

    The synthetic signal starts at 0.5 s (the default ``pre`` of make_synthetic).
    """
    X = bandpass(X, fs, *config.BANDPASS)[:, None]
    return crop(X, fs, start=0.5, duration=duration)


def random_signals(n_rows, n_samples, seed):
    return np.random.default_rng(seed).normal(size=(n_rows, n_samples))


# --- reference_signals ---------------------------------------------------------


def test_reference_signals_shape():
    """reference_signals(10, 250, 500, n_harmonics=5).shape == (10, 500)."""
    assert reference_signals(10, 250, 500, n_harmonics=5).shape == (10, 500)


def test_reference_signals_contains_every_harmonic():
    """sin(2*pi*h*f*t) and cos(2*pi*h*f*t) for h = 1..n_harmonics are rows (any order)."""
    Y = reference_signals(12, fs, 250, n_harmonics=3)
    t = np.arange(250) / fs
    for h in range(1, 4):
        for wave in (np.sin, np.cos):
            expected = wave(2 * np.pi * h * 12 * t)
            assert np.isclose(Y, expected).all(axis=1).any(), f"{wave.__name__}, h={h}"


# --- canonical_correlation -----------------------------------------------------


def test_canonical_correlation_identical_signals_is_one():
    """X = Y (any full-rank signal) -> correlation ~1.0."""
    X = random_signals(3, 200, seed=0)
    assert np.isclose(canonical_correlation(X, X), 1.0)


def test_canonical_correlation_matches_sklearn():
    """Random X, Y: same value as corrcoef of sklearn CCA(n_components=1) components.

    sklearn's CCA is iterative, so it only agrees to ~1e-6.
    """
    X = random_signals(9, 250, seed=1)
    Y = random_signals(10, 250, seed=2)
    u, v = SklearnCCA(n_components=1).fit_transform(X.T, Y.T)
    expected = np.corrcoef(u[:, 0], v[:, 0])[0, 1]
    assert np.isclose(canonical_correlation(X, Y), expected, atol=1e-4)


def test_canonical_correlation_matches_subspace_angles():
    """Exact reference: cosine of the smallest principal angle between the centred signals."""
    X = random_signals(9, 250, seed=3)
    Y = random_signals(10, 250, seed=4)
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)
    expected = np.cos(subspace_angles(Xc.T, Yc.T).min())
    assert np.isclose(canonical_correlation(X, Y), expected)


def test_canonical_correlation_ignores_offset_and_gain():
    """A DC offset or a different gain per channel doesn't change the correlation."""
    X = random_signals(9, 250, seed=5)
    Y = reference_signals(10, fs, 250)
    gains = np.arange(1, 10)[:, None]
    assert np.isclose(
        canonical_correlation(gains * X + 100, Y), canonical_correlation(X, Y)
    )


def test_canonical_correlation_finds_sine_with_unknown_phase():
    """Channels carrying 12 Hz with different gains and phases (+ a little noise):
    ~1 against the 12 Hz references, low against the 8 Hz ones."""
    t = np.arange(250) / fs
    phases = np.array([0.3, 1.7, 4.0])[:, None]
    gains = np.array([1.0, 0.5, 2.0])[:, None]
    X = gains * np.sin(2 * np.pi * 12 * t + phases) + 0.05 * random_signals(3, 250, 6)
    assert canonical_correlation(X, reference_signals(12, fs, 250)) > 0.99
    assert canonical_correlation(X, reference_signals(8, fs, 250)) < 0.5


# --- CCA classifier ------------------------------------------------------------


def test_cca_can_be_cloned():
    """sklearn's clone() works, i.e. __init__ only stores its parameters."""
    assert clone(CCA(freqs, fs, n_harmonics=3)).n_harmonics == 3


def test_cca_fit_returns_self_and_sets_classes(synthetic_easy):
    """fit() returns the estimator itself and sets classes_ = 0..n_targets-1."""
    X, y, _ = synthetic_easy
    clf = CCA(freqs, fs)
    assert clf.fit(prepare(X, 1.0), y) is clf
    assert np.array_equal(clf.classes_, np.arange(len(freqs)))


def test_cca_easy_data(synthetic_easy):
    """Accuracy > 0.95 at a 1 s window; decision_function shape == (n_trials, n_targets)."""
    X, y, _ = synthetic_easy
    X = prepare(X, 1.0)
    clf = CCA(freqs, fs).fit(X, y)
    assert clf.decision_function(X).shape == (len(y), len(freqs))
    assert clf.predict(X).shape == y.shape
    assert clf.score(X, y) > 0.95


def test_cca_handles_changing_window_length(synthetic_easy):
    """One fitted estimator classifies 0.5 s and then 2 s windows."""
    X, y, _ = synthetic_easy
    clf = CCA(freqs, fs).fit(prepare(X, 0.5), y)
    for duration in (0.5, 2.0):
        assert clf.score(prepare(X, duration), y) > 0.95, f"{duration} s"


def test_cca_noise_only_is_near_chance(synthetic_noise_only):
    """Accuracy well below 0.6 (chance is 1/5)."""
    X, y, _ = synthetic_noise_only
    X = prepare(X, 1.0)
    assert CCA(freqs, fs).fit(X, y).score(X, y) < 0.6


def test_fbcca_identical_bands_match_single_band(synthetic_easy):
    """The same band copied 3 times (n_bands = 3) gives the same predictions as n_bands = 1."""
    X, y, _ = synthetic_easy
    X1 = prepare(X, 1.0)
    X3 = np.repeat(X1, 3, axis=1)
    clf = CCA(freqs, fs).fit(X3, y)
    assert clf.decision_function(X3).shape == (len(y), len(freqs))
    assert np.array_equal(clf.predict(X3), CCA(freqs, fs).fit(X1, y).predict(X1))


def test_fbcca_uses_every_band(synthetic_easy, synthetic_noise_only):
    """Band 1 is pure noise, band 2 carries the SSVEP: accuracy stays > 0.9."""
    X_noise, _, _ = synthetic_noise_only
    X_easy, y, _ = synthetic_easy
    X = np.concatenate([prepare(X_noise, 1.0), prepare(X_easy, 1.0)], axis=1)
    assert CCA(freqs, fs).fit(X, y).score(X, y) > 0.9


@pytest.mark.skip(reason="TODO")
def test_trca_and_etrca_easy_data(synthetic_easy):
    """Leave-one-block-out accuracy > 0.95 for TRCA(ensemble=False) and TRCA(ensemble=True)."""
