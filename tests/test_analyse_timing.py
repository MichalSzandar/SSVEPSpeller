"""Tests for scripts/analyse_timing.py: dropped frames and interval spread of flip logs."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_timing  # noqa: E402

R = 120.0
CUE, FLICKER, OFF = 0, 1, 2


def write_session_log(
    path, n_trials=3, n_flicker=50, skip=None, jitter_ms=0.05, seed=0
):
    """A session flip log as calibration.py saves it: per trial 20 cue flips, then
    ``n_flicker`` flicker flips and the stim_off flip, 1/R apart (plus jitter), with
    a 1.5 s gap before the next trial. ``skip`` = (trial, flip): that flicker flip
    lands one refresh late (a dropped frame)."""
    rng = np.random.default_rng(seed)
    times, trials, phases = [], [], []
    t = 100.0
    for trial in range(n_trials):
        for phase, n in [(CUE, 20), (FLICKER, n_flicker), (OFF, 1)]:
            for i in range(n):
                if skip == (trial, i) and phase == FLICKER:
                    t += 1 / R
                times.append(t + rng.normal(0, jitter_ms / 1000))
                trials.append(trial)
                phases.append(phase)
                t += 1 / R
        t += 1.5
    np.savez(
        path,
        flip_times=np.array(times),
        trial=np.array(trials),
        phase=np.array(phases),
        refresh_rate=R,
    )


def write_flicker_test(path, n=600, sd_ms=0.1, windowed=False, seed=0):
    rng = np.random.default_rng(seed)
    times = 50.0 + np.arange(n) / R + rng.normal(0, sd_ms / 1000, n)
    np.savez(
        path,
        flip_times=times,
        refresh_rate=R,
        freqs=[8.0],
        phases=[0.0],
        windowed=windowed,
        aborted=False,
    )


def test_kind_and_date_come_from_the_file_name():
    assert (
        analyse_timing.kind_of(Path("S02_20261116_200531_flips.npz")) == "calibration"
    )
    assert (
        analyse_timing.kind_of(Path("x_20261110_174135_online_flips.npz")) == "online"
    )
    assert analyse_timing.kind_of(Path("flicker_20261104_113747.npz")) == "flicker test"
    assert analyse_timing.date_of(Path("S02_20261116_200531_flips.npz")) == "2026-11-16"
    with pytest.raises(ValueError):
        analyse_timing.kind_of(Path("pilot_20261110_172435.npz"))


def test_session_log_counts_only_the_flicker_up_to_stim_off(tmp_path):
    """50 flicker flips + the stim_off flip = 50 intervals per trial; the cue flips and
    the gaps between trials don't count."""
    path = tmp_path / "s_20261116_120000_flips.npz"
    write_session_log(path, n_trials=3, n_flicker=50)
    intervals, rate, n_trials, windowed = analyse_timing.flicker_intervals(path)
    assert (len(intervals), rate, n_trials, windowed) == (150, R, 3, False)
    assert np.abs(intervals - 1 / R).max() < 0.001


def test_a_clean_session_has_no_dropped_frames(tmp_path):
    path = tmp_path / "s_20261116_120000_flips.npz"
    write_session_log(path)
    row = analyse_timing.summarize(path)
    assert row["n_dropped"] == 0 and row["verdict"] == "ok"
    assert row["mean_interval_ms"] == pytest.approx(1000 / R, abs=0.01)
    assert row["sd_interval_ms"] < 0.3
    assert row["kind"] == "calibration" and row["date"] == "2026-11-16"


def test_a_late_flip_is_one_dropped_frame(tmp_path):
    path = tmp_path / "s_20261116_120000_online_flips.npz"
    write_session_log(path, skip=(1, 10))
    row = analyse_timing.summarize(path)
    assert (row["n_dropped"], row["n_late_flips"]) == (1, 1)
    assert row["max_interval_ms"] == pytest.approx(2000 / R, abs=0.3)
    assert "1 dropped frame" in row["verdict"]
    assert row["kind"] == "online"


def test_flicker_tests_use_every_interval_and_judge_the_spread(tmp_path):
    clean, shaky = (
        tmp_path / "flicker_20261104_1.npz",
        tmp_path / "flicker_20261104_2.npz",
    )
    write_flicker_test(clean, sd_ms=0.05)
    write_flicker_test(shaky, sd_ms=1.0, seed=1)
    row = analyse_timing.summarize(clean)
    assert (row["n_intervals"], row["n_trials"], row["verdict"]) == (599, 1, "ok")
    assert "sd" in analyse_timing.summarize(shaky)["verdict"]


def test_a_windowed_flicker_test_is_not_judged(tmp_path):
    path = tmp_path / "flicker_20261027_1.npz"
    write_flicker_test(path, windowed=True)
    row = analyse_timing.summarize(path)
    assert row["windowed"] and row["verdict"] == "not valid (windowed)"


def test_main_writes_one_row_per_file_sorted_by_date(tmp_path):
    files = [
        tmp_path / "s_20261116_120000_flips.npz",
        tmp_path / "flicker_20261104_1.npz",
    ]
    write_session_log(files[0])
    write_flicker_test(files[1])
    analyse_timing.main([*map(str, files), "--name", "t", "--out-dir", str(tmp_path)])
    table = pd.read_csv(tmp_path / "t.csv")
    assert table["date"].tolist() == ["2026-11-04", "2026-11-16"]
    assert table["kind"].tolist() == ["flicker test", "calibration"]
