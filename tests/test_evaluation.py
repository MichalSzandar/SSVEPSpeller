"""Tests for ssvep.evaluation."""

import numpy as np
import pytest

from ssvep.evaluation import itr


def test_itr_perfect_accuracy():
    # P = 1 -> B = log2(N) bits per selection, one selection per second.
    assert itr(n_targets=40, accuracy=1.0, trial_time=1.0) == pytest.approx(60 * np.log2(40))


@pytest.mark.skip(reason="TODO")
def test_itr_at_chance_is_zero():
    """N = 5, P = 0.2 -> 0 bits/min."""


@pytest.mark.skip(reason="TODO")
def test_itr_below_chance_is_zero():
    """N = 5, P = 0.0 -> 0 bits/min (the raw formula would be positive here)."""


@pytest.mark.skip(reason="TODO")
def test_itr_increases_with_accuracy():
    """For fixed N and T, ITR is strictly increasing for P in (1/N, 1]."""


@pytest.mark.skip(reason="TODO")
def test_leave_one_block_out_on_easy_data(synthetic_easy):
    """CCA on high-SNR synthetic data -> accuracy close to 1.0."""
