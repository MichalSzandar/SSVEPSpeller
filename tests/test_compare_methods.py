"""Smoke tests for scripts/compare_methods.py (synthetic data only, so they run in CI)."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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
    assert len(compare_methods.COLORS) >= len(compare_methods.METHODS)
    assert len(compare_methods.MARKERS) >= len(compare_methods.METHODS)


def test_synthetic_windows_start_at_the_signal_plus_latency():
    """Synthetic data: start = the 0.5 s of pre-stimulus noise, plus --latency if given."""
    for latency, start in [(None, 0.5), (0.1, 0.6)]:
        args = SimpleNamespace(data="synthetic", subjects=[1], latency=latency)
        ((subject, *_, fs, got),) = compare_methods.load_dataset(args)
        assert subject == "synthetic1"
        assert fs == config.BENCHMARK_FS
        assert got == pytest.approx(start)


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
    """All methods, one subject, one window: each one is clearly above chance (0.2)."""
    args = "--data synthetic --subjects 1 --windows 1.0 --name all"
    compare_methods.main(args.split() + ["--out-dir", str(tmp_path)])
    df = pd.read_csv(tmp_path / "all.csv")
    assert set(df["method"]) == set(compare_methods.METHODS)
    assert (df["accuracy"] > 0.5).all(), df


def test_recording_mode_needs_files():
    """--data recording without --recordings stops with a usage error."""
    with pytest.raises(SystemExit):
        compare_methods.main(["--data", "recording"])
