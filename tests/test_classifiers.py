"""Tests for ssvep.cca and ssvep.trca."""

import numpy as np
import pytest
from scipy.linalg import subspace_angles
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.cross_decomposition import CCA as SklearnCCA
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score

from ssvep import config
from ssvep.cca import CCA, canonical_correlation, reference_signals
from ssvep.datasets import make_synthetic
from ssvep.preprocessing import bandpass, crop, filter_bank, filter_bank_weights
from ssvep.trca import TRCA, trca_spatial_filter

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


# --- trca_spatial_filter -------------------------------------------------------


def trca_trials(n_trials=5, n_channels=6, n_samples=250, seed=0):
    """Trials of one target, (n_trials, n_channels, n_samples): the same waveform in every
    trial with a random gain per channel, plus noise whose level differs between channels.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / fs
    source = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 20 * t + 1.0)
    gain = rng.normal(size=(n_channels, 1))
    noise_level = rng.uniform(0.5, 3.0, size=(n_channels, 1))
    return gain * source + noise_level * rng.normal(
        size=(n_trials, n_channels, n_samples)
    )


def reproducibility(w, X):
    """How alike the filtered trials y_i = w @ X[i] are: the sum of y_i . y_j over pairs of
    different trials, divided by the sum of y_i . y_i (time means removed)."""
    y = np.array([w @ x for x in X])
    y = y - y.mean(axis=1, keepdims=True)
    n = len(y)
    between = sum(y[i] @ y[j] for i in range(n) for j in range(n) if i != j)
    return between / sum(y[i] @ y[i] for i in range(n))


def cos_similarity(a, b):
    """|cos| of the angle between two filters: 1 means equal up to sign and scale."""
    return abs(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_trca_filter_shape():
    """One weight per channel."""
    assert trca_spatial_filter(trca_trials(n_channels=6)).shape == (6,)


def test_trca_filter_leaves_input_unchanged():
    """The caller's array is not modified, even when the input is a view into it."""
    data = trca_trials()[:, None] + 50.0  # (n_trials, n_bands=1, n_channels, n_samples)
    before = data.copy()
    trca_spatial_filter(data[:, 0])
    assert np.array_equal(data, before)


@pytest.mark.parametrize("snr", [1.0, 0.1])
def test_trca_filter_maximises_reproducibility(snr):
    """No other filter makes the filtered trials more alike. The reference is a numerical
    optimisation of `reproducibility`, started from every single channel."""
    X = trca_trials()
    X = snr * X + np.random.default_rng(1).normal(size=X.shape)
    returned = reproducibility(trca_spatial_filter(X), X)
    optimum = max(
        -minimize(lambda v: -reproducibility(v, X), start).fun
        for start in np.eye(X.shape[1])
    )
    assert returned >= optimum - 1e-6, f"returned {returned:.6f}, optimum {optimum:.6f}"


def test_trca_filter_cancels_noise_shared_by_two_channels():
    """ch0 = SSVEP + n, ch1 = n, ch2 = unrelated noise, where n is strong noise that changes
    from trial to trial. Only ch0 - ch1 is reproducible, so w is proportional to [1, -1, 0].
    """
    rng = np.random.default_rng(2)
    s = np.sin(2 * np.pi * 10 * np.arange(250) / fs)
    n = 3.0 * rng.normal(size=(5, 250))
    X = np.stack([s + n, n, rng.normal(size=(5, 250))], axis=1)
    X += 0.01 * rng.normal(size=X.shape)
    assert cos_similarity(trca_spatial_filter(X), np.array([1.0, -1.0, 0.0])) > 0.99


def test_trca_filter_ignores_offsets_and_overall_gain():
    """A different constant added to every trial and channel (DC drift between trials), or
    all data scaled by a constant, leaves the filter unchanged up to sign and scale."""
    X = trca_trials()
    w = trca_spatial_filter(X.copy())
    offsets = np.random.default_rng(3).normal(
        scale=100.0, size=(X.shape[0], X.shape[1], 1)
    )
    assert cos_similarity(trca_spatial_filter(X + offsets), w) > 0.9999
    assert cos_similarity(trca_spatial_filter(1e-3 * X), w) > 0.9999


def test_trca_filter_rejects_a_single_trial():
    """With one trial there are no pairs of trials to compare: ValueError."""
    with pytest.raises(ValueError):
        trca_spatial_filter(trca_trials(n_trials=1))


def test_trca_filter_flat_channel_gets_zero_weight():
    """A flat channel (lost electrode contact) gets a weight of ~0; no exception, no NaN."""
    X = trca_trials()
    X[:, 2] = 0.0
    w = trca_spatial_filter(X)
    assert np.all(np.isfinite(w))
    assert abs(w[2]) < 1e-6 * np.max(np.abs(w))


# --- TRCA classifier -----------------------------------------------------------


def lobo_accuracy(estimator, X, y, blocks):
    """Leave-one-block-out accuracy, computed by scikit-learn (independent of evaluation.py)."""
    cv = LeaveOneGroupOut()
    return cross_val_score(
        estimator, X, y, groups=blocks, cv=cv, error_score="raise"
    ).mean()


def random_trials(n_bands, seed, n_blocks=3, n_channels=4, n_samples=200):
    """Pure-noise data in classifier format: every target once per block, labels 0..4."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_blocks * len(freqs), n_bands, n_channels, n_samples))
    y = np.tile(np.arange(len(freqs)), n_blocks)
    return X, y


def corr(a, b):
    """Pearson correlation of two equally shaped arrays, all entries together."""
    return np.corrcoef(np.ravel(a), np.ravel(b))[0, 1]


def band_correlations(model, X_test):
    """r[t, k, m]: band m of test trial t vs. the template of target k, both filtered with
    target k's own filter (TRCA) or with the filters of all targets stacked (eTRCA)."""
    n_trials, n_bands = X_test.shape[:2]
    n_targets = len(model.classes_)
    r = np.zeros((n_trials, n_targets, n_bands))
    for t in range(n_trials):
        for k in range(n_targets):
            for m in range(n_bands):
                W = model.filters_[m] if model.ensemble else model.filters_[m, k]
                r[t, k, m] = corr(W @ X_test[t, m], W @ model.templates_[k, m])
    return r


def test_trca_can_be_cloned():
    """sklearn's clone() works, i.e. __init__ only stores its parameters."""
    params = clone(TRCA(ensemble=True, fb_a=2.0, fb_b=0.5)).get_params()
    assert params == {"ensemble": True, "fb_a": 2.0, "fb_b": 0.5}


@pytest.mark.parametrize("n_bands", [1, 3])
def test_trca_fit_returns_self_and_sets_attributes(n_bands):
    """fit() returns the estimator; templates_ is (n_targets, n_bands, n_channels, n_samples),
    filters_ is (n_bands, n_targets, n_channels) and classes_ = 0..n_targets-1."""
    X, y = random_trials(n_bands, seed=10)
    clf = TRCA()
    assert clf.fit(X, y) is clf
    assert clf.templates_.shape == (len(freqs), n_bands, 4, 200)
    assert clf.filters_.shape == (n_bands, len(freqs), 4)
    assert np.array_equal(clf.classes_, np.arange(len(freqs)))


def test_trca_templates_are_mean_training_trials():
    """templates_[k, m] is the average of target k's training trials in band m."""
    X, y = random_trials(n_bands=2, seed=11)
    clf = TRCA().fit(X, y)
    for k in range(len(freqs)):
        for m in range(2):
            expected = np.mean([X[i, m] for i in range(len(y)) if y[i] == k], axis=0)
            assert np.allclose(clf.templates_[k, m], expected)


def test_trca_filters_come_from_trca_spatial_filter():
    """filters_[m, k] is trca_spatial_filter() of target k's training trials in band m,
    up to sign and scale."""
    X, y = random_trials(n_bands=2, seed=12)
    clf = TRCA().fit(X, y)
    for k in range(len(freqs)):
        for m in range(2):
            trials = np.array([X[i, m] for i in range(len(y)) if y[i] == k])
            assert (
                cos_similarity(clf.filters_[m, k], trca_spatial_filter(trials)) > 0.9999
            )


@pytest.mark.parametrize("ensemble", [False, True])
def test_trca_single_band_scores_are_correlations(ensemble):
    """n_bands = 1: score[t, k] is the correlation r between the filtered test trial and the
    filtered template of target k (see band_correlations), with no weight and no square.
    """
    X, y = random_trials(n_bands=1, seed=13)
    X_test = np.random.default_rng(14).normal(size=(4, 1, 4, 200))
    clf = TRCA(ensemble=ensemble).fit(X, y)
    scores = clf.decision_function(X_test)
    assert scores.shape == (4, len(freqs))
    assert np.allclose(scores, band_correlations(clf, X_test)[:, :, 0])


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("fb_a, fb_b", [(1.25, 0.25), (2.0, 0.5)])
def test_fb_trca_scores_combine_bands(ensemble, fb_a, fb_b):
    """n_bands = 3: score = sum over bands of w(m) * f(r_m), where r_m is band m's correlation
    (see band_correlations) and w = filter_bank_weights(3, fb_a, fb_b). Both f(r) = r
    (Nakanishi's MATLAB code) and f(r) = sign(r) * r**2 (FBCCA style) are accepted."""
    X, y = random_trials(n_bands=3, seed=15)
    X_test = np.random.default_rng(16).normal(size=(4, 3, 4, 200))
    clf = TRCA(ensemble=ensemble, fb_a=fb_a, fb_b=fb_b).fit(X, y)
    scores = clf.decision_function(X_test)
    r = band_correlations(clf, X_test)
    w = filter_bank_weights(3, fb_a, fb_b)
    linear, signed_square = r @ w, (np.sign(r) * r**2) @ w
    assert np.allclose(scores, linear) or np.allclose(scores, signed_square)


def test_trca_predict_returns_labels():
    """predict() gives classes_[argmax of the scores]: a label, not a column index. Trained
    on targets 0, 2 and 4 only, it must never answer 1 or 3."""
    X, y = random_trials(n_bands=1, seed=17)
    keep = np.isin(y, [0, 2, 4])
    clf = TRCA().fit(X[keep], y[keep])
    X_test = np.random.default_rng(18).normal(size=(6, 1, 4, 200))
    assert np.array_equal(clf.classes_, [0, 2, 4])
    expected = clf.classes_[np.argmax(clf.decision_function(X_test), axis=1)]
    assert np.array_equal(clf.predict(X_test), expected)


def test_trca_rejects_test_windows_of_another_length():
    """Templates are compared sample by sample, so a shorter test window is a ValueError."""
    X, y = random_trials(n_bands=1, seed=19)
    clf = TRCA().fit(X, y)
    with pytest.raises(ValueError):
        clf.decision_function(X[:, :, :, :100])


@pytest.mark.parametrize("ensemble", [False, True])
@pytest.mark.parametrize("duration", [0.5, 1.0])
def test_trca_easy_data(synthetic_easy, ensemble, duration):
    """Single band: leave-one-block-out accuracy > 0.95 for TRCA and eTRCA."""
    X, y, blocks = synthetic_easy
    assert (
        lobo_accuracy(TRCA(ensemble=ensemble), prepare(X, duration), y, blocks) > 0.95
    )


def test_fb_etrca_easy_data(synthetic_easy):
    """Filter-bank input: leave-one-block-out accuracy > 0.95 for FB-eTRCA (1 s window)."""
    X, y, blocks = synthetic_easy
    assert lobo_accuracy(TRCA(ensemble=True), prepare_fb(X, 1.0), y, blocks) > 0.95


@pytest.mark.parametrize("ensemble", [False, True])
def test_trca_noise_only_is_near_chance(synthetic_noise_only, ensemble):
    """Leave-one-block-out accuracy well below 0.6 (chance is 1/5)."""
    X, y, blocks = synthetic_noise_only
    assert lobo_accuracy(TRCA(ensemble=ensemble), prepare(X, 1.0), y, blocks) < 0.6
