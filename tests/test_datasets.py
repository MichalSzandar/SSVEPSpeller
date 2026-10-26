"""Tests for ssvep.datasets"""

import pytest
import numpy as np

from ssvep import config
from ssvep.datasets import benchmark_channel_indices, benchmark_target_indices, load_benchmark, make_synthetic

needs_benchmark = pytest.mark.skipif(
    not (config.BENCHMARK_DIR / "S1.mat").exists(), reason="Benchmark data not downloaded"
)


def test_make_synthetic_shapes():
    """5 targets x 6 blocks -> X (30, 8, 625) at 250 Hz with 0.5 s pre + 2 s; y and blocks have length 30."""
    X, y, blocks = make_synthetic(freqs=[8.0, 9.0, 10.0, 11.0, 12.0], fs=250, n_blocks=6, n_channels=8, duration=2.0)
    assert X.shape == (30, 8, 625)
    assert y.shape == (30,)
    assert blocks.shape == (30,)
    assert np.all(y >= 0) and np.all(y < 5)
    assert np.all(blocks >= 0) and np.all(blocks < 6)


@needs_benchmark
def test_load_benchmark_shapes():
    """S1, 5 targets, 9 channels -> X (30, 9, 1500), y in {0..4}, blocks in {0..5}."""
    X, y, blocks = load_benchmark(1)
    assert X.shape == (30, 9, 1500)
    assert y.shape == (30,)
    assert blocks.shape == (30,)
    assert np.all(y >= 0) and np.all(y < 5)
    assert np.all(blocks >= 0) and np.all(blocks < 6)
