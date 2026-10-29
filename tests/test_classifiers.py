"""Tests for ssvep.cca and ssvep.trca."""

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from sklearn.base import clone
from sklearn.cross_decomposition import CCA as SklearnCCA

from ssvep import config
from ssvep.cca import CCA, canonical_correlation, reference_signals
from ssvep.datasets import make_synthetic
from ssvep.preprocessing import bandpass, crop, filter_bank

fs = config.BENCHMARK_FS
freqs = config.TARGET_FREQS


def prepare(X, duration):
    """Raw synthetic epochs -> classifier input (n_trials, 1, n_channels, n_window).

    The synthetic signal starts at 0.5 s (the default ``pre`` of make_synthetic).
    """
    X = bandpass(X, fs, *config.BANDPASS)[:, None]
    return crop(X, fs, start=0.5, duration=duration)


def prepare_fb(X, duration):
    """Raw synthetic epochs -> filter-bank input (n_trials, N_SUBBANDS, n_channels, n_window)."""
    X = filter_bank(X, fs, config.N_SUBBANDS)
    return crop(X, fs, start=0.5, duration=duration)


def add_alpha(X, amplitude, seed):
    """Add a 9.5-10.5 Hz rhythm: random frequency and phase per trial, random gain per channel."""
    rng = np.random.default_rng(seed)
    n_trials, n_channels, n_samples = X.shape
    t = np.arange(n_samples) / fs
    phase = rng.uniform(0, 2 * np.pi, size=(n_trials, 1, 1))
    gain = rng.uniform(0.5, 1.0, size=(n_trials, n_channels, 1))
    freq = rng.uniform(9.5, 10.5, size=(n_trials, 1, 1))
    return X + amplitude * gain * np.sin(2 * np.pi * freq * t + phase)


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


def test_cca_single_band_scores_are_correlations():
    """With n_bands = 1 the score is the canonical correlation itself (no weight, no square)."""
    X = np.random.default_rng(8).normal(size=(4, 1, 3, 250))
    scores = CCA(freqs, fs).fit(X).decision_function(X)
    for i, freq in enumerate(freqs):
        Y = reference_signals(freq, fs, 250)
        expected = [canonical_correlation(X[t, 0], Y) for t in range(4)]
        assert np.allclose(scores[:, i], expected)


@pytest.mark.parametrize(
    "fb_a, fb_b, weights",
    [
        (1.25, 0.25, [1.25, 0.67045, 0.50328]),  # Chen 2015
        (0.0, 0.0, [1.0, 1.0, 1.0]),
        (2.0, 0.5, [1.5, 0.75, 0.5 + 1 / 9]),
    ],
)
def test_fbcca_scores_are_weighted_squared_correlations(fb_a, fb_b, weights):
    """With n_bands = 3: score = sum over bands of w(m) * rho_m**2, where rho_m is band m's
    canonical correlation with the target's references and w(m) = m**(-fb_a) + fb_b."""
    X = np.random.default_rng(7).normal(size=(4, 3, 3, 250))
    scores = CCA(freqs, fs, fb_a=fb_a, fb_b=fb_b).fit(X).decision_function(X)
    for i, freq in enumerate(freqs):
        Y = reference_signals(freq, fs, 250)
        for t in range(4):
            rho = np.array([canonical_correlation(X[t, m], Y) for m in range(3)])
            assert np.isclose(scores[t, i], np.sum(weights * rho**2), rtol=1e-4)


@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_fbcca_filter_bank_easy_data(synthetic_easy, duration):
    """Input from filter_bank(): accuracy > 0.95 at 0.5 s and 1 s windows."""
    X, y, _ = synthetic_easy
    X = prepare_fb(X, duration)
    assert X.shape[1] == config.N_SUBBANDS
    assert CCA(freqs, fs).fit(X, y).score(X, y) > 0.95


def test_fbcca_filter_bank_noise_only_is_near_chance(synthetic_noise_only):
    """Accuracy well below 0.6 (chance is 1/5)."""
    X, y, _ = synthetic_noise_only
    X = prepare_fb(X, 1.0)
    assert CCA(freqs, fs).fit(X, y).score(X, y) < 0.6


def test_fbcca_resists_alpha_rhythm():
    """A strong ~10 Hz rhythm on top of a weak SSVEP fools single-band CCA towards the 10 Hz
    target. The higher sub-bands contain no alpha, so FBCCA keeps accuracy > 0.9 and beats
    CCA by > 0.3 (1 s window)."""
    X, y, _ = make_synthetic(snr_db=-10.0, n_blocks=10, seed=0)
    X = add_alpha(X, amplitude=2.0, seed=0)
    X_cca, X_fb = prepare(X, 1.0), prepare_fb(X, 1.0)
    acc_cca = CCA(freqs, fs).fit(X_cca, y).score(X_cca, y)
    acc_fbcca = CCA(freqs, fs).fit(X_fb, y).score(X_fb, y)
    assert acc_fbcca > 0.9, f"FBCCA {acc_fbcca:.2f}"
    assert acc_fbcca - acc_cca > 0.3, f"CCA {acc_cca:.2f}, FBCCA {acc_fbcca:.2f}"


@pytest.mark.skip(reason="TODO")
def test_trca_and_etrca_easy_data(synthetic_easy):
    """Leave-one-block-out accuracy > 0.95 for TRCA(ensemble=False) and TRCA(ensemble=True)."""
