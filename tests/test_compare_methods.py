"""Smoke tests for scripts/compare_methods.py (synthetic data only, so they run in CI)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ssvep import config

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "compare_methods.py"
SPEC = importlib.util.spec_from_file_location("compare_methods", MODULE_PATH)
compare_methods = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare_methods)


def test_methods_have_a_band_count_and_a_fresh_estimator_factory():
    """Every METHODS entry is (n_subbands >= 1, factory(freqs, fs) -> new estimator)."""
    for name, (n_subbands, factory) in compare_methods.METHODS.items():
        assert isinstance(n_subbands, int) and n_subbands >= 1, name
        first = factory(config.TARGET_FREQS, config.BENCHMARK_FS)
        second = factory(config.TARGET_FREQS, config.BENCHMARK_FS)
        assert first is not second, name
        assert hasattr(first, "fit") and hasattr(first, "predict"), name


def test_every_method_has_a_plot_style():
    """A color and a marker per base method; a normalised variant needs its base method."""
    names = list(compare_methods.METHODS)
    bases = [n for n in names if not n.endswith(compare_methods.NORM_SUFFIX)]
    for name in names:
        assert name.removesuffix(compare_methods.NORM_SUFFIX) in bases, name
    assert len(compare_methods.COLORS) >= len(bases)
    assert len(compare_methods.MARKERS) >= len(bases)
    # Variants come after all base methods, so adding one never shifts a base's color
    assert names[: len(bases)] == bases


def test_synthetic_windows_start_at_the_signal_plus_latency():
    """Synthetic data: start = the 0.5 s of pre-stimulus noise, plus --latency if given."""
    for latency, start in [(None, 0.5), (0.1, 0.6)]:
        args = SimpleNamespace(data="synthetic", subjects=[1], latency=latency)
        ((subject, *_, fs, got),) = compare_methods.load_dataset(args)
        assert subject == "synthetic1"
        assert fs == config.BENCHMARK_FS
        assert got == pytest.approx(start)


def test_recordings_are_scored_with_their_own_frequencies(monkeypatch):
    """--data recording yields each file's session_start freqs (not config.TARGET_FREQS),
    the cap channel labels, EEG_FS, and the file name as the subject."""
    X = np.zeros((2, len(config.CAP_CHANNELS), 1500))
    monkeypatch.setattr(
        compare_methods,
        "load_recording",
        lambda path: (X, np.array([0, 1]), np.zeros(2)),
    )
    monkeypatch.setattr(compare_methods, "recording_freqs", lambda path: [8.0, 10.0])
    args = SimpleNamespace(data="recording", recordings=["dir/rec1.npz"], latency=None)
    ((subject, _, _, _, channels, freqs, fs, _),) = compare_methods.load_dataset(args)
    assert subject == "rec1"
    assert freqs == [8.0, 10.0]
    assert channels == config.CAP_CHANNELS
    assert fs == config.EEG_FS


def test_rejection_changes_nothing_on_clean_synthetic_data(tmp_path):
    """Default rejection and --trial-ratio 0 --channel-ratio 0 give the same accuracies on
    clean synthetic data (nothing to reject)."""
    common = "--data synthetic --subjects 1 --windows 1.0 --methods CCA-norm eTRCA"
    compare_methods.main(common.split() + ["--name", "on", "--out-dir", str(tmp_path)])
    off = "--trial-ratio 0 --channel-ratio 0 --name off"
    compare_methods.main(common.split() + off.split() + ["--out-dir", str(tmp_path)])
    on = pd.read_csv(tmp_path / "on.csv")
    off = pd.read_csv(tmp_path / "off.csv")
    pd.testing.assert_frame_equal(on, off)


def test_main_writes_one_csv_row_per_subject_method_window_and_a_figure(tmp_path):
    """--data synthetic: CSV with subject, method, window, accuracy, itr, plus a PNG."""
    args = "--data synthetic --subjects 1 2 --windows 0.5 1.0 --methods CCA eTRCA --name run"
    compare_methods.main(args.split() + ["--out-dir", str(tmp_path)])
    df = pd.read_csv(tmp_path / "run.csv")
    assert list(df.columns) == ["subject", "method", "window", "accuracy", "itr"]
    assert len(df) == 2 * 2 * 2
    assert set(df["method"]) == {"CCA", "eTRCA"}
    assert (df["accuracy"] > 0.9).all()
    assert (tmp_path / "run.png").stat().st_size > 0


def test_every_method_runs_end_to_end(tmp_path):
    """All methods, one subject, one window: each one is clearly above chance (1/6)."""
    args = "--data synthetic --subjects 1 --windows 1.0 --name all"
    compare_methods.main(args.split() + ["--out-dir", str(tmp_path)])
    df = pd.read_csv(tmp_path / "all.csv")
    assert set(df["method"]) == set(compare_methods.METHODS)
    assert (df["accuracy"] > 0.5).all(), df


def test_recording_mode_needs_files():
    """--data recording without --recordings stops with a usage error."""
    with pytest.raises(SystemExit):
        compare_methods.main(["--data", "recording"])
