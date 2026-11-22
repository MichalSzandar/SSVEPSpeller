"""Smoke tests for thesis/figures/make_results.py on small synthetic result tables."""

import importlib.util
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from ssvep import config

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "make_results", ROOT / "thesis" / "figures" / "make_results.py"
)
make_results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(make_results)

SESSIONS = ["pilot_20261110_172435", "S02_20261116_195733"]
METHODS = ["CCA", "FBCCA-norm", "eTRCA", "FB-eTRCA"]


def compare_table(subjects, windows, methods=METHODS, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for s in subjects:
        for m in methods:
            for w in windows:
                accuracy = float(np.clip(0.4 + 0.2 * w + rng.normal(0, 0.05), 0, 1))
                rows.append((s, m, w, accuracy, 60 * accuracy / (w + 0.5)))
    return pd.DataFrame(
        rows, columns=["subject", "method", "window", "accuracy", "itr"]
    )


def write_results(results: Path) -> None:
    """Every input table of make_results, small, plus two flip logs for timing."""
    results.mkdir()
    compare_table(["S1", "S2", "S3"], [0.5, 1.0]).to_csv(
        results / "benchmark_6.csv", index=False
    )
    compare_table(["S1", "S2"], [0.5, 1.0]).to_csv(
        results / "benchmark_40.csv", index=False
    )
    compare_table(SESSIONS, [0.5, 1.0, 2.0]).to_csv(results / "own.csv", index=False)
    latency = compare_table(SESSIONS, [1.0, 2.0])
    latency = pd.concat([latency.assign(latency=v) for v in [0.14, 0.7]])
    latency.to_csv(results / "own_latency.csv", index=False)
    rng = np.random.default_rng(1)
    for name, subjects in [("benchmark", ["S1", "S2", "S10"]), ("own", SESSIONS)]:
        pd.DataFrame(
            {
                "subject": np.repeat(subjects, 12),
                "ratio": rng.lognormal(0.3, 0.2, 12 * len(subjects)),
            }
        ).to_csv(results / f"ratios_{name}_trials.csv", index=False)

    period = 1 / config.REFRESH_RATE
    session = results / "s_20261116_120000_flips.npz"
    times = 10 + np.arange(61) * period
    np.savez(
        session,
        flip_times=times,
        trial=np.zeros(61, int),
        phase=np.r_[np.ones(60, int), 2],
        refresh_rate=config.REFRESH_RATE,
    )
    test = results / "flicker_20261104_1.npz"
    np.savez(test, flip_times=times, refresh_rate=config.REFRESH_RATE, windowed=False)
    rows = []
    for path, kind, date in [
        (session, "calibration", "2026-11-16"),
        (test, "flicker test", "2026-11-04"),
    ]:
        rows.append(
            {
                "file": str(path),
                "kind": kind,
                "date": date,
                "n_dropped": 0,
                "sd_interval_ms": 0.1,
                "windowed": False,
            }
        )
    pd.DataFrame(rows).to_csv(results / "timing.csv", index=False)


def test_every_figure_is_made_when_its_tables_exist(tmp_path):
    write_results(tmp_path / "results")
    made, skipped = make_results.main(
        ["--results", str(tmp_path / "results"), "--out-dir", str(tmp_path)]
    )
    assert set(made) == set(make_results.FIGURES)
    assert skipped == []
    for name in made:
        assert (tmp_path / f"{name}.pdf").stat().st_size > 1000


def test_figures_without_their_tables_are_skipped(tmp_path):
    (tmp_path / "results").mkdir()
    made, skipped = make_results.main(
        ["--results", str(tmp_path / "results"), "--out-dir", str(tmp_path)]
    )
    assert made == ["filter_bank"]  # needs no results
    assert any(line.startswith("own_vs_benchmark") for line in skipped)


def test_only_makes_the_named_figures(tmp_path):
    write_results(tmp_path / "results")
    made, _ = make_results.main(
        [
            "--results",
            str(tmp_path / "results"),
            "--out-dir",
            str(tmp_path),
            "--only",
            "own_methods",
        ]
    )
    assert made == ["own_methods"]


def test_sessions_are_numbered_by_date():
    labels = make_results.session_labels(list(reversed(SESSIONS)))
    assert labels == {
        "pilot_20261110_172435": "Session 1 (10 Nov)",
        "S02_20261116_195733": "Session 2 (16 Nov)",
    }


def test_the_analysis_driver_is_valid_bash():
    result = subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "run_analysis.sh")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
