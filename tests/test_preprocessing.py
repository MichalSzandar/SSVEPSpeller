"""Tests for ssvep.preprocessing."""

import pytest

from ssvep import config
from ssvep.preprocessing import (
    bandpass,
    chebyshev_bandpass,
    crop,
    filter_bank,
    filter_bank_weights,
    notch,
)
import numpy as np

n_samples = 1000
fs = 250
time = np.arange(n_samples) / fs


def test_notch_removes_line_noise():
    """10 Hz + 50 Hz sine. After notch: 50 Hz amplitude drops > 20 dB, 10 Hz kept within ~1 %.
    Compare filtered sine to unfiltered sine in the passband (250-750 ms)."""

    X = np.sin(2 * np.pi * 10 * time) + 0.5 * np.sin(2 * np.pi * 50 * time)
    X = notch(X=X[None, None, :], fs=fs, freq=50.0, quality=30.0)[0, 0]

    assert np.allclose(X[250:750], np.sin(2 * np.pi * 10 * time[250:750]), atol=0.01)


def test_bandpass_keeps_shape_and_passband():
    """Output shape == input shape; a 12 Hz sine inside a 7-90 Hz band passes almost unchanged."""

    X = np.sin(2 * np.pi * 12 * time)[None, None, :]
    X = bandpass(
        X=X,
        fs=fs,
        low=7,
        high=90,
    )

    assert X.shape == (1, 1, 1000)
    assert np.allclose(
        X[:, :, 250:750], np.sin(2 * np.pi * 12 * time[250:750]), atol=0.01
    )


def sine(freq, seconds=8.0):
    return np.sin(2 * np.pi * freq * np.arange(int(seconds * fs)) / fs)


def gain_db(x, y):
    """Gain in dB from x to y along the last axis, measured 2 s away from both ends."""
    mid = slice(2 * fs, x.shape[-1] - 2 * fs)

    def rms(s):
        return np.sqrt(np.mean(s[..., mid] ** 2, axis=-1))

    return 20 * np.log10(rms(y) / rms(x))


def eeg_like_noise(n, seed):
    """Gaussian noise with a 1/f power spectrum, like background EEG."""
    spectrum = np.fft.rfft(np.random.default_rng(seed).normal(size=n))
    freqs = np.fft.rfftfreq(n, 1 / fs)
    freqs[0] = freqs[1]
    return np.fft.irfft(spectrum / np.sqrt(freqs), n=n)


def test_filter_bank_weights_match_chen_2015():
    """Defaults (a = 1.25, b = 0.25), 5 sub-bands: w ≈ [1.25, 0.670, 0.503, 0.427, 0.384]."""
    expected = [1.25, 0.670, 0.503, 0.427, 0.384]
    assert np.allclose(filter_bank_weights(5), expected, atol=1e-3)


@pytest.mark.parametrize(
    "n_subbands, a, b, expected",
    [
        (3, 1, 0, [1.0, 1 / 2, 1 / 3]),  # integer a and b
        (4, 0.0, 0.5, [1.5, 1.5, 1.5, 1.5]),  # a = 0: every sub-band counts the same
        (3, 2.0, 0.5, [1.5, 0.75, 0.5 + 1 / 9]),
        (1, 1.25, 0.25, [1.25]),
    ],
)
def test_filter_bank_weights_formula(n_subbands, a, b, expected):
    """w(m) = m**(-a) + b for m = 1..n_subbands, shape (n_subbands,)."""
    w = filter_bank_weights(n_subbands, a, b)
    assert w.shape == (n_subbands,)
    assert np.allclose(w, expected)


@pytest.mark.parametrize("n_subbands", [2, 5, 10])
def test_filter_bank_weights_decrease(n_subbands):
    """w(1) = 1 + b, the weights strictly decrease, and they stay above b."""
    w = filter_bank_weights(n_subbands, a=1.25, b=0.25)
    assert np.isclose(w[0], 1.25)
    assert np.all(np.diff(w) < 0)
    assert np.all(w > 0.25)


def test_chebyshev_bandpass_filters_each_row_along_time():
    """Output shape == input shape; filtering a (2, 3, n) array gives the same result
    as filtering each 1-D row on its own."""
    X = np.random.default_rng(0).normal(size=(2, 3, 1000))
    edges = dict(passband=(14.0, 90.0), stopband=(10.0, 100.0))
    Y = chebyshev_bandpass(X, fs, **edges)
    assert Y.shape == X.shape
    for i in range(2):
        for j in range(3):
            assert np.allclose(Y[i, j], chebyshev_bandpass(X[i, j], fs, **edges))


@pytest.mark.parametrize("rp", [0.5, 0.1])
@pytest.mark.parametrize(
    "passband, stopband",
    [((14.0, 90.0), (10.0, 100.0)), ((20.0, 40.0), (15.0, 50.0))],
)
def test_chebyshev_bandpass_passband_ripple(passband, stopband, rp):
    """Across the passband, edges included, the gain stays between -2*rp dB and 0 dB
    (the ripple, applied twice by the zero-phase filter)."""
    for freq in np.linspace(*passband, 7):
        x = sine(freq)
        g = gain_db(x, chebyshev_bandpass(x, fs, passband, stopband, rp=rp))
        assert -2 * rp - 0.05 <= g <= 0.05, f"{freq:.1f} Hz: {g:.2f} dB"


@pytest.mark.parametrize("gstop", [40.0, 60.0])
def test_chebyshev_bandpass_stopband_attenuation(gstop):
    """Frequencies at and beyond the stopband edges are attenuated by at least gstop dB."""
    for freq in [2.0, 6.0, 10.0, 100.0, 110.0, 120.0]:
        x = sine(freq)
        y = chebyshev_bandpass(x, fs, (14.0, 90.0), (10.0, 100.0), gstop=gstop)
        assert gain_db(x, y) <= -gstop, f"{freq} Hz"


@pytest.mark.parametrize("shape", [(9, 500), (4, 9, 500), (2, 4, 9, 500)])
@pytest.mark.parametrize("n_subbands", [1, 5, 10])
def test_filter_bank_shape(shape, n_subbands):
    """(..., n_channels, n_samples) -> (..., n_subbands, n_channels, n_samples),
    for any number of leading axes."""
    expected = shape[:-2] + (n_subbands,) + shape[-2:]
    assert filter_bank(np.zeros(shape), fs, n_subbands).shape == expected


def test_filter_bank_trials_and_channels_are_independent():
    """Filtering a batch equals filtering each trial alone, and a silent channel stays silent."""
    X = np.random.default_rng(1).normal(size=(3, 2, 1000))
    X[:, 1] = 0.0
    Y = filter_bank(X, fs)
    for i in range(3):
        assert np.allclose(Y[i], filter_bank(X[i], fs))
    assert np.allclose(Y[:, :, 1], 0.0)


@pytest.mark.parametrize("m", range(1, 11))
def test_filter_bank_keeps_harmonic_m_of_8hz_up_to_subband_m(m):
    """Sub-band k starts just below k * 8 Hz (Chen 2015). So 8*m Hz passes sub-bands
    1..m (loss < 1.5 dB) and is removed (> 40 dB down) from sub-band m+1 on."""
    x = sine(8.0 * m)
    gains = gain_db(x, filter_bank(x[None, :], fs, n_subbands=10)[:, 0])
    assert np.all(gains[:m] > -1.5), np.round(gains, 1)
    assert np.all(gains[m:] < -40.0), np.round(gains, 1)


@pytest.mark.parametrize("n_subbands", [0, 11])
def test_filter_bank_rejects_unsupported_n_subbands(n_subbands):
    """n_subbands outside 1..10 (the sub-bands of the reference design) raises ValueError."""
    with pytest.raises(ValueError):
        filter_bank(np.zeros((9, 500)), fs, n_subbands)


def test_filter_bank_edge_effects_stay_out_of_analysis_window():
    """A 6 s epoch filtered on its own matches the same stretch of a filtered 60 s
    recording inside the Benchmark analysis window (onset + latency, 1 s long):
    relative error < 2.5 % on EEG-like noise."""
    x = eeg_like_noise(60 * fs, seed=0)
    whole = filter_bank(x[None, :], fs)
    start = config.BENCHMARK_ONSET + config.BENCHMARK_LATENCY
    window = slice(int(round(start * fs)), int(round((start + 1.0) * fs)))
    for t0 in range(5, 55, 5):
        epoch = slice(t0 * fs, (t0 + 6) * fs)
        alone = filter_bank(x[None, epoch], fs)[..., window]
        expected = whole[..., epoch][..., window]
        error = np.sqrt(np.mean((alone - expected) ** 2) / np.mean(expected**2))
        assert error < 0.025, f"epoch at {t0} s: {100 * error:.1f} %"


def test_crop_length():
    """crop(X, fs=250, start=0.64, duration=1.0).shape[-1] == 250."""
    X = np.arange(n_samples)[None, None, :]
    X_cropped = crop(X, fs=fs, start=1.0, duration=2.0)
    assert X_cropped.shape[-1] == 500
    assert np.all(X_cropped == np.arange(250, 750)[None, None, :])
