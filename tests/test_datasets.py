"""Tests for ssvep.datasets"""

import pytest

from ssvep import config

needs_benchmark = pytest.mark.skipif(
    not (config.BENCHMARK_DIR / "S1.mat").exists(), reason="Benchmark data not downloaded"
)


@pytest.mark.skip(reason="TODO")
def test_make_synthetic_shapes():
    """5 targets x 6 blocks -> X (30, 8, 625) at 250 Hz with 0.5 s pre + 2 s; y and blocks have length 30."""


@pytest.mark.skip(reason="TODO")
@needs_benchmark
def test_load_benchmark_shapes():
    """S1, 5 targets, 9 channels -> X (30, 9, 1500), y in {0..4}, blocks in {0..5}."""
