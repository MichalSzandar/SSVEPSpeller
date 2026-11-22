"""Tests for scripts/artifact_ratios.py: what the artifact rejection measures."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ssvep import config
from ssvep.datasets import make_synthetic

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import artifact_ratios  # noqa: E402

FS = config.BENCHMARK_FS
PRE = 0.5  # s before the synthetic SSVEP, the window start below
CHANNELS = [f"ch{c}" for c in range(9)]


def data(seed=0):
    X, y, blocks = make_synthetic(pre=PRE, snr_db=0.0, seed=seed)
    return X.copy(), y, blocks


def run(X, y, blocks, **kwargs):
    return artifact_ratios.ratios("S1", X, y, blocks, CHANNELS, FS, PRE, 1.0, **kwargs)


def test_clean_data_flags_nothing():
    trials, channels = run(*data())
    assert not trials["flagged"].any()
    assert trials["ratio"].max() < 5
    assert not channels["dropped"].any()
    assert channels["level_ratio"].between(1 / 3, 3).all()


def test_a_burst_on_one_channel_flags_that_trial_and_names_the_channel():
    X, y, blocks = data()
    rng = np.random.default_rng(1)
    start = int(PRE * FS) + 50  # inside the analysed window
    X[3, 2, start : start + 50] += 60 * rng.normal(size=50)
    trials, _ = run(X, y, blocks)
    assert trials.loc[trials["flagged"], "trial"].tolist() == [3]
    assert trials.loc[3, "worst_channel"] == "ch2"
    assert trials.loc[3, "ratio"] > 5


def test_a_noisy_channel_is_dropped_before_the_trial_check():
    """A channel 10x noisier in every trial: dropped, so it is never a trial's worst
    channel, and its swings don't flag any trial."""
    X, y, blocks = data()
    X[:, 5] *= 10
    trials, channels = run(X, y, blocks)
    assert channels.loc[channels["dropped"], "channel"].tolist() == ["ch5"]
    assert channels.loc[5, "level_ratio"] > 3
    assert "ch5" not in set(trials["worst_channel"])
    assert not trials["flagged"].any()


def test_thresholds_are_parameters():
    X, y, blocks = data()
    trials, _ = run(X, y, blocks, trial_ratio=1.0)  # every trial above its median
    assert trials["flagged"].any()


def test_trial_table_keeps_block_and_target():
    X, y, blocks = data()
    trials, _ = run(X, y, blocks)
    assert trials["block"].tolist() == blocks.tolist()
    assert trials["target"].tolist() == y.tolist()


def test_main_writes_trial_and_channel_tables(tmp_path):
    artifact_ratios.main(
        "--data synthetic --subjects 1 2 --window 1 --name r --out-dir".split()
        + [str(tmp_path)]
    )
    trials = pd.read_csv(tmp_path / "r_trials.csv")
    channels = pd.read_csv(tmp_path / "r_channels.csv")
    assert set(trials["subject"]) == {"synthetic1", "synthetic2"}
    assert len(trials) == 2 * 36
    assert len(channels) == 2 * len(config.BENCHMARK_CHANNELS)
