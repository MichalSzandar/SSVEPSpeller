"""Statistics over subjects for a compare_methods.py result: which methods differ.

Usage:
    .venv/bin/python scripts/method_stats.py results/benchmark_6.csv

Input: a CSV with the columns subject, method, window, accuracy, itr (one row per
subject x method x window), as compare_methods.py writes it.

For every window, and once more at each method's best window, and for both measures
(accuracy and ITR):
  - a Friedman test over the methods, with the subjects as blocks (a repeated-measures
    test on ranks, so it needs no normal distribution), and Kendall's W as its effect
    size (0 = the methods rank randomly, 1 = every subject ranks them the same way);
  - two-sided Wilcoxon signed-rank tests for every pair of methods, with Holm's
    correction for the number of pairs in that family (one family = one window and
    one measure).
A method's best window is the one with the highest mean ITR over subjects (the shorter
one on a tie), as in the thesis (Sec. 4.6). In the "best" comparison, every subject
contributes its value at each method's best window.

Output, next to the input file:
  <stem>_best.csv      per method: best window, mean and SD of accuracy and ITR there
  <stem>_friedman.csv  per comparison and measure: chi2, p, Kendall's W, n, k
  <stem>_wilcoxon.csv  per comparison, measure and pair of methods: medians, median
                       difference (b - a), W statistic, p and Holm-corrected p
"""

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

METRICS = ["accuracy", "itr"]
BEST = "best"  # the comparison label for "each method at its best window"


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, in the input order.

    The i-th smallest of m p-values is multiplied by (m - i + 1); the adjusted values
    are then made non-decreasing in that order and capped at 1. NaN stays NaN and does
    not count towards m.
    """
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(~np.isnan(p))
    m = len(valid)
    if m == 0:
        return adjusted
    order = valid[np.argsort(p[valid], kind="stable")]
    scaled = p[order] * (m - np.arange(m))
    adjusted[order] = np.minimum(np.maximum.accumulate(scaled), 1.0)
    return adjusted


def method_order(df: pd.DataFrame) -> list[str]:
    """Methods in the order they first appear (compare_methods writes METHODS order)."""
    return list(dict.fromkeys(df["method"]))


def best_windows(df: pd.DataFrame) -> pd.DataFrame:
    """Per method: the window with the highest mean ITR over subjects (the shorter
    window on a tie), and mean and SD over subjects of accuracy and ITR there.

    Columns: method, window, accuracy_mean, accuracy_sd, itr_mean, itr_sd, n_subjects.
    """
    rows = []
    for method in method_order(df):
        part = df[df["method"] == method]
        mean_itr = part.groupby("window")["itr"].mean().sort_index()
        window = mean_itr.index[np.argmax(mean_itr.to_numpy())]  # first max: shortest
        at_best = part[part["window"] == window]
        rows.append(
            {
                "method": method,
                "window": window,
                "accuracy_mean": at_best["accuracy"].mean(),
                "accuracy_sd": at_best["accuracy"].std(ddof=1),
                "itr_mean": at_best["itr"].mean(),
                "itr_sd": at_best["itr"].std(ddof=1),
                "n_subjects": at_best["subject"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def wide(df: pd.DataFrame, metric: str, windows: dict[str, float]) -> pd.DataFrame:
    """Subjects x methods table of ``metric``, each method at ``windows[method]``.

    Only subjects that have a value for every method are kept (a paired test needs
    complete rows).
    """
    columns = {}
    for method, window in windows.items():
        part = df[(df["method"] == method) & (df["window"] == window)]
        columns[method] = part.set_index("subject")[metric]
    return pd.DataFrame(columns).dropna()


def friedman(table: pd.DataFrame) -> dict:
    """Friedman test over the columns (methods) of a subjects x methods table.

    Returns statistic (chi2), p, kendall_w, n (subjects) and k (methods). With fewer
    than 3 methods or 2 subjects there is no test (NaN). If every subject gives every
    method the same value, nothing differs: statistic 0, p 1, W 0.
    """
    n, k = table.shape
    result = {"statistic": np.nan, "p": np.nan, "kendall_w": np.nan, "n": n, "k": k}
    if k < 3 or n < 2:
        return result
    values = table.to_numpy(dtype=float)
    if np.all(values == values[:, :1]):  # all methods tie within every subject
        return {**result, "statistic": 0.0, "p": 1.0, "kendall_w": 0.0}
    statistic, p = friedmanchisquare(*values.T)
    return {
        **result,
        "statistic": float(statistic),
        "p": float(p),
        "kendall_w": float(statistic / (n * (k - 1))),
    }


def pairwise_wilcoxon(table: pd.DataFrame) -> pd.DataFrame:
    """Two-sided Wilcoxon signed-rank test for every pair of columns (a before b in
    column order), Holm-corrected over all pairs of the table.

    Zero differences are left out (scipy's default "wilcox" rule); a pair whose
    differences are all zero gets statistic 0 and p 1. Columns: method_a, method_b,
    median_a, median_b, median_diff (median of b - a), n_nonzero, statistic, p,
    p_holm.
    """
    rows = []
    for a, b in combinations(table.columns, 2):
        diff = table[b] - table[a]
        n_nonzero = int((diff != 0).sum())
        if n_nonzero == 0:
            statistic, p = 0.0, 1.0
        else:
            test = wilcoxon(table[b], table[a])
            statistic, p = float(test.statistic), float(test.pvalue)
        rows.append(
            {
                "method_a": a,
                "method_b": b,
                "median_a": table[a].median(),
                "median_b": table[b].median(),
                "median_diff": diff.median(),
                "n_nonzero": n_nonzero,
                "statistic": statistic,
                "p": p,
            }
        )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm(result["p"]) if len(result) else []
    return result


def run(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All statistics of a compare_methods table -> (best, friedman, wilcoxon).

    The friedman and wilcoxon tables have a ``comparison`` column (the window as a
    string, or "best") and a ``metric`` column; Holm's correction is applied within
    each comparison and metric.
    """
    best = best_windows(df)
    methods = method_order(df)
    comparisons = {
        f"{w:g}": {method: w for method in methods}
        for w in sorted(df["window"].unique())
    }
    comparisons[BEST] = dict(zip(best["method"], best["window"]))

    friedman_rows, wilcoxon_tables = [], []
    for label, windows in comparisons.items():
        for metric in METRICS:
            table = wide(df, metric, windows)
            friedman_rows.append(
                {"comparison": label, "metric": metric, **friedman(table)}
            )
            pairs = pairwise_wilcoxon(table)
            pairs.insert(0, "metric", metric)
            pairs.insert(0, "comparison", label)
            wilcoxon_tables.append(pairs)
    return best, pd.DataFrame(friedman_rows), pd.concat(wilcoxon_tables)


def print_report(best, friedman_table, wilcoxon_table) -> None:
    """Best windows, the Friedman tests at the best windows, and the pairs that differ
    there (Holm-corrected p < 0.05)."""
    print("Best window per method (highest mean ITR), mean ± SD over subjects:")
    for row in best.itertuples():
        print(
            f"  {row.method:<10} {row.window:>5g} s  accuracy "
            f"{row.accuracy_mean:.3f} ± {row.accuracy_sd:.3f}  ITR "
            f"{row.itr_mean:6.1f} ± {row.itr_sd:5.1f} bits/min  (n = {row.n_subjects})"
        )
    for metric in METRICS:
        f = friedman_table[
            (friedman_table["comparison"] == BEST)
            & (friedman_table["metric"] == metric)
        ].iloc[0]
        print(
            f"\nFriedman, {metric} at the best windows: chi2({f.k - 1:g}) = "
            f"{f.statistic:.1f}, p = {f.p:.3g}, Kendall's W = {f.kendall_w:.2f}"
        )
        pairs = wilcoxon_table[
            (wilcoxon_table["comparison"] == BEST)
            & (wilcoxon_table["metric"] == metric)
            & (wilcoxon_table["p_holm"] < 0.05)
        ]
        print(f"  {len(pairs)} pairs differ (Wilcoxon, Holm p < 0.05):")
        for row in pairs.itertuples():
            print(
                f"    {row.method_b} - {row.method_a}: median difference "
                f"{row.median_diff:+.3g}, p_holm = {row.p_holm:.3g}"
            )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("csv", type=Path, help="a compare_methods.py result")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.csv)
    best, friedman_table, wilcoxon_table = run(df)
    stem = args.csv.with_suffix("")
    best.to_csv(f"{stem}_best.csv", index=False)
    friedman_table.to_csv(f"{stem}_friedman.csv", index=False)
    wilcoxon_table.to_csv(f"{stem}_wilcoxon.csv", index=False)
    print_report(best, friedman_table, wilcoxon_table)
    print(f"\nSaved {stem}_best.csv, _friedman.csv and _wilcoxon.csv")


if __name__ == "__main__":
    main()
