import pytest

from ssvep.datasets import make_synthetic


@pytest.fixture
def synthetic_easy():
    """(X, y, blocks): high-SNR synthetic SSVEP, 250 Hz, config.TARGET_FREQS."""
    return make_synthetic(snr_db=0.0, seed=0)


@pytest.fixture
def synthetic_noise_only():
    """(X, y, blocks) where the signal is buried in noise: expect ~chance accuracy."""
    return make_synthetic(snr_db=-40.0, seed=1)
