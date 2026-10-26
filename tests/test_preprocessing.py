"""Tests for ssvep.preprocessing."""

import pytest

from ssvep.preprocessing import bandpass, crop, notch
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
    assert np.allclose(X[:, :, 250:750], np.sin(2 * np.pi * 12 * time[250:750]), atol=0.01)


@pytest.mark.skip(reason="TODO")
def test_filter_bank_shape():
    """(n_trials, n_channels, n_samples) -> (n_trials, n_subbands, n_channels, n_samples)."""


@pytest.mark.skip(reason="TODO")
def test_filter_bank_weights_decrease():
    """w(1) = 1 + b and the weights strictly decrease."""


def test_crop_length():
    """crop(X, fs=250, start=0.64, duration=1.0).shape[-1] == 250."""
    X = np.arange(n_samples)[None, None, :]
    X_cropped = crop(X, fs=fs, start=1.0, duration=2.0)
    assert X_cropped.shape[-1] == 500
    assert np.all(X_cropped == np.arange(250, 750)[None, None, :])
