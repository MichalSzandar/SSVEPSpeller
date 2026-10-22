"""Tests for ssvep.preprocessing."""

import pytest


@pytest.mark.skip(reason="TODO")
def test_notch_removes_line_noise():
    """10 Hz + 50 Hz sine. After notch: 50 Hz amplitude drops > 20 dB, 10 Hz kept within ~1 %."""


@pytest.mark.skip(reason="TODO")
def test_bandpass_keeps_shape_and_passband():
    """Output shape == input shape; a 12 Hz sine inside a 7-90 Hz band passes almost unchanged."""


@pytest.mark.skip(reason="TODO")
def test_filter_bank_shape():
    """(n_trials, n_channels, n_samples) -> (n_trials, n_subbands, n_channels, n_samples)."""


@pytest.mark.skip(reason="TODO")
def test_filter_bank_weights_decrease():
    """w(1) = 1 + b and the weights strictly decrease."""


@pytest.mark.skip(reason="TODO")
def test_crop_length():
    """crop(X, fs=250, start=0.64, duration=1.0).shape[-1] == 250."""
