"""Tests for scripts/latency_sweep.py (synthetic data and synthetic recordings)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import latency_sweep  # noqa: E402


def test_one_table_with_a_latency_column(tmp_path):
    latency_sweep.main(
        "--data synthetic --subjects 1 --latencies 0 0.2 --windows 0.5 1 "
        "--methods CCA --name sweep --out-dir".split() + [str(tmp_path)]
    )
    table = pd.read_csv(tmp_path / "sweep.csv")
    assert list(table.columns) == [
        "subject",
        "method",
        "latency",
        "window",
        "accuracy",
        "itr",
    ]
    assert len(table) == 2 * 2
    assert sorted(table["latency"].unique()) == [0.0, 0.2]
    assert (table["accuracy"] > 0.9).all()


def test_the_window_moves_with_the_latency(tmp_path, ssvep_recording):
    """The SSVEP starts 1 s after the marker: a 0.5 s window at latency 0.14 s sees
    noise, at latency 1.14 s it sees the response."""
    path = tmp_path / "rec_20261116_120000.npz"
    ssvep_recording(path, n_blocks=3, delay=1.14, seed=3)
    table = latency_sweep.sweep(
        "recording", [0.14, 1.14], [0.5], ["CCA"], recordings=[str(path)]
    ).set_index("latency")
    assert table.loc[1.14, "accuracy"] > 0.9
    assert table.loc[0.14, "accuracy"] < 0.6


def test_recording_mode_needs_files():
    with pytest.raises(SystemExit):
        latency_sweep.main(["--latencies", "0.5"])
