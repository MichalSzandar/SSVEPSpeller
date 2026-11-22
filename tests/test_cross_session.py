"""Tests for scripts/cross_session.py: train on one recording, test on another."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import cross_session  # noqa: E402
from online import load_calibration  # noqa: E402


@pytest.fixture
def two_sessions(tmp_path, ssvep_recording):
    """Session A: the SSVEP 0.14 s after the markers. Session B: 1.14 s after them,
    as if its EEG timestamps led the markers by 1 s."""
    a, b = tmp_path / "a_20261110_120000.npz", tmp_path / "b_20261116_120000.npz"
    ssvep_recording(a, n_blocks=3, delay=0.14, seed=1)
    ssvep_recording(b, n_blocks=3, delay=1.14, seed=2)
    return a, b


def session(path, latency):
    return (*load_calibration([path]), latency)


@pytest.mark.parametrize("method", ["CCA", "eTRCA"])
def test_transfer_works_when_both_windows_cover_the_response(two_sessions, method):
    a, b = two_sessions
    accuracy = cross_session.transfer_accuracy(
        method, 1.0, session(a, 0.14), session(b, 1.14)
    )
    assert accuracy > 0.9


def test_each_recording_is_cut_at_its_own_latency(two_sessions):
    """eTRCA trained on A at 0.14 s: tested on B at 0.14 s it sees mostly noise, at
    B's own 1.14 s it sees the response it was trained on."""
    a, b = two_sessions
    same = cross_session.transfer_accuracy(
        "eTRCA", 0.5, session(a, 0.14), session(b, 0.14)
    )
    aligned = cross_session.transfer_accuracy(
        "eTRCA", 0.5, session(a, 0.14), session(b, 1.14)
    )
    assert aligned > 0.9
    assert same < 0.6


def test_test_epochs_too_short_for_the_latency_raise(two_sessions):
    a, b = two_sessions
    with pytest.raises(ValueError):
        cross_session.transfer_accuracy("CCA", 4.0, session(a, 0.14), session(b, 1.14))


def test_main_writes_within_and_cross_rows(two_sessions, tmp_path):
    a, b = two_sessions
    cross_session.main(
        [
            "--recordings",
            str(a),
            str(b),
            "--latencies",
            "0.14",
            "1.14",
            "--windows",
            "1",
            "--methods",
            "CCA",
            "eTRCA",
            "--name",
            "x",
            "--out-dir",
            str(tmp_path),
        ]
    )
    table = pd.read_csv(tmp_path / "x.csv")
    pairs = set(zip(table["train"], table["test"]))
    assert pairs == {
        (a.stem, a.stem),
        (b.stem, b.stem),
        (a.stem, b.stem),
        (b.stem, a.stem),
    }
    assert len(table) == 4 * 2  # pairs x methods, one window
    cross = table[table["train"] != table["test"]]
    assert set(zip(cross["train_latency"], cross["test_latency"])) == {
        (0.14, 1.14),
        (1.14, 0.14),
    }
    assert (table["accuracy"] > 0.9).all()


def test_main_needs_one_latency_per_recording(two_sessions):
    a, b = two_sessions
    with pytest.raises(SystemExit):
        cross_session.main(["--recordings", str(a), str(b), "--latencies", "0.7"])
