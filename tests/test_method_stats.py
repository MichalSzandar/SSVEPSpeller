"""Tests for scripts/method_stats.py: statistics over subjects of a compare_methods table."""

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import method_stats  # noqa: E402


def table(values: dict[str, list[float]]) -> pd.DataFrame:
    """Subjects x methods table, subjects S1, S2, ..."""
    n = len(next(iter(values.values())))
    return pd.DataFrame(values, index=[f"S{i + 1}" for i in range(n)])


def long_table(rows) -> pd.DataFrame:
    """compare_methods format from (subject, method, window, accuracy, itr) tuples."""
    return pd.DataFrame(
        rows, columns=["subject", "method", "window", "accuracy", "itr"]
    )


# --- holm --------------------------------------------------------------------------


def test_holm_matches_a_hand_computed_example():
    """Sorted 0.005, 0.01, 0.03, 0.04 times 4, 3, 2, 1 -> 0.02, 0.03, 0.06, 0.04, then
    made non-decreasing (0.04 -> 0.06), back in the input order."""
    got = method_stats.holm([0.01, 0.04, 0.03, 0.005])
    np.testing.assert_allclose(got, [0.03, 0.06, 0.06, 0.02])


def test_holm_caps_at_one_and_skips_nan():
    """NaN stays NaN and does not count towards the number of tests (m = 2 here):
    0.2 * 2 = 0.4, and 0.3 * 1 = 0.3 is raised to 0.4 (non-decreasing); 0.6 * 2 = 1.2
    is capped at 1."""
    got = method_stats.holm([0.3, np.nan, 0.2])
    np.testing.assert_allclose(got[[0, 2]], [0.4, 0.4])
    assert np.isnan(got[1])
    np.testing.assert_allclose(method_stats.holm([0.7, 0.6]), [1.0, 1.0])


def test_holm_of_a_single_p_value_is_that_p_value():
    np.testing.assert_allclose(method_stats.holm([0.037]), [0.037])


# --- best window ---------------------------------------------------------------------


def test_best_window_has_the_highest_mean_itr_and_the_shorter_one_on_a_tie():
    rows = []
    for subject, offset in [("S1", 0.0), ("S2", 2.0)]:
        rows += [
            (subject, "A", 0.5, 0.6, 10 + offset),
            (subject, "A", 1.0, 0.9, 30 + offset),  # A peaks at 1.0 s
            (subject, "A", 2.0, 1.0, 20 + offset),
            (subject, "B", 0.5, 0.8, 25 + offset),  # B ties at 0.5 and 1.0 s
            (subject, "B", 1.0, 0.9, 25 + offset),
            (subject, "B", 2.0, 0.9, 12 + offset),
        ]
    best = method_stats.best_windows(long_table(rows)).set_index("method")
    assert best.loc["A", "window"] == 1.0
    assert best.loc["B", "window"] == 0.5
    assert best.loc["A", "itr_mean"] == pytest.approx(31.0)
    assert best.loc["A", "itr_sd"] == pytest.approx(np.std([30, 32], ddof=1))
    assert best.loc["A", "accuracy_mean"] == pytest.approx(0.9)
    assert best.loc["A", "n_subjects"] == 2
    assert list(best.index) == ["A", "B"]  # order of appearance


def test_wide_takes_each_method_at_its_own_window_and_drops_incomplete_subjects():
    rows = [
        ("S1", "A", 1.0, 0.5, 1.0),
        ("S1", "B", 2.0, 0.7, 2.0),
        ("S2", "A", 1.0, 0.6, 3.0),  # S2 has no B at 2.0 s
        ("S2", "B", 1.0, 0.9, 4.0),
    ]
    got = method_stats.wide(long_table(rows), "itr", {"A": 1.0, "B": 2.0})
    assert list(got.index) == ["S1"]
    assert got.loc["S1"].tolist() == [1.0, 2.0]


# --- friedman ------------------------------------------------------------------------


def test_friedman_with_the_same_ranking_in_every_subject():
    """10 subjects all rank A < B < C: chi2 = 12/(nk(k+1)) * sum(R^2) - 3n(k+1) = 20
    with rank sums 10, 20, 30, so Kendall's W = 20 / (10 * 2) = 1."""
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 0.4, 10)
    got = method_stats.friedman(table({"A": a, "B": a + 0.1, "C": a + 0.3}))
    assert got["statistic"] == pytest.approx(20.0)
    assert got["p"] == pytest.approx(chi2.sf(20.0, 2))
    assert got["kendall_w"] == pytest.approx(1.0)
    assert (got["n"], got["k"]) == (10, 3)


def test_friedman_when_nothing_differs():
    same = [0.5, 0.7, 1.0, 0.9]
    got = method_stats.friedman(table({"A": same, "B": same, "C": same}))
    assert (got["statistic"], got["p"], got["kendall_w"]) == (0.0, 1.0, 0.0)


def test_friedman_needs_three_methods():
    got = method_stats.friedman(table({"A": [0.1, 0.2], "B": [0.3, 0.4]}))
    assert np.isnan(got["p"])


# --- wilcoxon --------------------------------------------------------------------------


def test_pairwise_wilcoxon_finds_a_consistent_difference_and_corrects_for_pairs():
    rng = np.random.default_rng(1)
    a = rng.uniform(0.3, 0.6, 15)
    got = method_stats.pairwise_wilcoxon(
        table({"A": a, "B": a + 0.1, "C": a + rng.normal(0, 0.02, 15)})
    )
    assert list(zip(got["method_a"], got["method_b"])) == [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]
    ab = got.iloc[0]
    assert ab["median_diff"] == pytest.approx(0.1)
    assert ab["n_nonzero"] == 15
    assert ab["p"] < 0.001
    assert (got["p_holm"] >= got["p"] - 1e-12).all()
    np.testing.assert_allclose(got["p_holm"], method_stats.holm(got["p"]))


def test_pairwise_wilcoxon_of_identical_methods_gives_p_one():
    same = [0.5, 0.7, 1.0, 0.9, 0.8]
    got = method_stats.pairwise_wilcoxon(table({"A": same, "B": same}))
    assert got.iloc[0]["p"] == 1.0
    assert got.iloc[0]["statistic"] == 0.0


# --- run and main ----------------------------------------------------------------------


def synthetic_results(n_subjects=8, seed=2) -> pd.DataFrame:
    """Three methods, three windows; C is better than B than A for every subject."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_subjects):
        base = rng.uniform(0.3, 0.5)
        for m, gain in [("A", 0.0), ("B", 0.15), ("C", 0.3)]:
            for w in [0.5, 1.0, 2.0]:
                accuracy = min(1.0, base + gain + 0.1 * w)
                rows.append((f"S{s + 1}", m, w, accuracy, 60 * accuracy / (w + 0.5)))
    return long_table(rows)


def test_run_tests_every_window_and_the_best_windows():
    best, friedman, wilcoxon = method_stats.run(synthetic_results())
    assert set(friedman["comparison"]) == {"0.5", "1", "2", "best"}
    assert len(friedman) == 4 * 2  # comparisons x (accuracy, itr)
    assert len(wilcoxon) == 4 * 2 * comb(3, 2)
    at_best = friedman[
        (friedman["comparison"] == "best") & (friedman["metric"] == "itr")
    ]
    assert at_best["p"].iloc[0] < 0.01
    assert at_best["kendall_w"].iloc[0] == pytest.approx(1.0)
    assert list(best["method"]) == ["A", "B", "C"]


def test_main_writes_the_three_tables_next_to_the_input(tmp_path, capsys):
    synthetic_results().to_csv(tmp_path / "bench.csv", index=False)
    method_stats.main([str(tmp_path / "bench.csv")])
    for suffix in ["best", "friedman", "wilcoxon"]:
        assert (tmp_path / f"bench_{suffix}.csv").stat().st_size > 0
    wilcoxon = pd.read_csv(tmp_path / "bench_wilcoxon.csv")
    assert {"comparison", "metric", "method_a", "method_b", "p", "p_holm"} <= set(
        wilcoxon.columns
    )
    assert "Friedman" in capsys.readouterr().out
