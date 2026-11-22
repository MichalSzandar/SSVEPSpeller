"""Every result figure of the thesis, from the tables in results/.

Run from the repository root, after scripts/run_analysis.sh:
    .venv/bin/python thesis/figures/make_results.py
    .venv/bin/python thesis/figures/make_results.py --only benchmark_methods timing

Writes thesis/figures/<name>.pdf for every figure whose input tables exist, and says
which ones it skipped. The figures, with the result tables they read:
    benchmark_methods  accuracy and ITR vs. window, 35 Benchmark subjects, the six
                       speller targets, mean ± SEM (benchmark_6.csv)
    benchmark_40       the same with all 40 targets, for the check against
                       Nakanishi et al. 2018 (benchmark_40.csv)
    own_methods        accuracy and ITR vs. window, one row per own session (own.csv)
    own_vs_benchmark   accuracy vs. window of four methods: Benchmark mean ± SEM and
                       both own sessions (benchmark_6.csv, own.csv)
    own_latency        accuracy vs. latency, per session and window (own_latency.csv)
    artifact_ratios    every trial's artifact ratio per subject and session, with the
                       rejection threshold (ratios_benchmark_trials.csv,
                       ratios_own_trials.csv)
    timing             (a) flip intervals of the flicker in both sessions, (b) the
                       interval SD of every flicker test (timing.csv + the flip logs)
    filter_bank        frequency response of the band-pass and the five sub-bands
                       (needs no results; computed from ssvep/preprocessing.py)
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from ssvep import config  # noqa: E402
from ssvep.preprocessing import bandpass, filter_bank  # noqa: E402

sys.path.insert(0, str(config.ROOT / "scripts"))
import analyse_timing  # noqa: E402
from compare_methods import (  # noqa: E402
    COLORS,
    GRID,
    INK,
    INK_SECONDARY,
    MARKERS,
    METHODS,
    NORM_SUFFIX,
)

OUT_DIR = Path(__file__).parent
WIDTH = 5.9  # in, the text width of the thesis (15 cm)
STYLE = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.titlelocation": "left",
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK_SECONDARY,
    "axes.labelcolor": INK,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
    "savefig.bbox": "tight",
}
# Datasets, when one figure compares them (never the method colours)
SESSION_STYLES = [
    {"color": INK, "marker": "o", "linestyle": "-"},
    {"color": "#8f8d87", "marker": "s", "linestyle": "--"},
]
BENCHMARK_STYLE = {"color": COLORS[0], "marker": "D", "linestyle": "-"}
OWN_LATENCY = 0.7  # s, the latency of the own recordings in Ch. 4 and 6
COMPARED_METHODS = ["CCA", "FBCCA-norm", "eTRCA", "FB-eTRCA"]  # own_vs_benchmark


# --- helpers ---------------------------------------------------------------------


def method_style(method: str) -> dict:
    """Colour and marker of the base method; a normalised variant is dashed."""
    base = method.removesuffix(NORM_SUFFIX)
    i = list(METHODS).index(base)
    return {
        "color": COLORS[i],
        "marker": MARKERS[i],
        "linestyle": "--" if method != base else "-",
    }


def methods_in(df: pd.DataFrame) -> list[str]:
    """The methods of ``df`` in METHODS order (the order of the legends)."""
    present = set(df["method"])
    return [m for m in METHODS if m in present]


def session_labels(subjects) -> dict[str, str]:
    """Own recording name -> "Session i (d Mon)", numbered by the date in the name."""

    def date(name):
        match = re.search(r"(20\d{6})_(\d{6})", name)
        return match.group(0) if match else name

    labels = {}
    for i, name in enumerate(sorted(set(subjects), key=date), start=1):
        match = re.search(r"20\d{2}(\d{2})(\d{2})", name)
        when = ""
        if match:
            month = pd.Timestamp(2000, int(match.group(1)), 1).strftime("%b")
            when = f" ({int(match.group(2))} {month})"
        labels[name] = f"Session {i}{when}"
    return labels


def mean_lines(ax, df, x: str, y: str, band: bool = True) -> None:
    """One line per method: the mean of ``y`` over subjects at every ``x``, with a
    ± SEM band when there is more than one subject."""
    for method in methods_in(df):
        rows = df[df["method"] == method]
        stats = rows.groupby(x)[y].agg(["mean", "sem"]).sort_index()
        ax.plot(stats.index, stats["mean"], label=method, **method_style(method))
        if band and rows["subject"].nunique() > 1:
            ax.fill_between(
                stats.index,
                stats["mean"] - stats["sem"],
                stats["mean"] + stats["sem"],
                color=method_style(method)["color"],
                alpha=0.12,
                linewidth=0,
            )


def chance_line(ax, chance: float, x_right: float) -> None:
    ax.axhline(chance, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.annotate(
        "chance",
        (x_right, chance),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=INK_SECONDARY,
        fontsize=8,
    )


def legend_below(fig, axes, ncol=None) -> None:
    """One legend under all panels; up to 5 entries on one row, else ``ncol`` (4)."""
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if ncol is None:
        ncol = len(labels) if len(labels) <= 5 else 4
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=min(ncol, len(labels)),
        frameon=False,
        handlelength=2.6,  # long enough to show the dashes of a normalised variant
    )


def accuracy_itr(df: pd.DataFrame, chance: float, height: float = 3.3):
    """Two panels, accuracy and ITR vs. window, mean (± SEM) over subjects."""
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, height), layout="constrained")
    mean_lines(axes[0], df, "window", "accuracy")
    mean_lines(axes[1], df, "window", "itr")
    chance_line(axes[0], chance, df["window"].max())
    axes[0].set(xlabel="Window length (s)", ylabel="Accuracy", ylim=(0, 1.02))
    axes[1].set(xlabel="Window length (s)", ylabel="ITR (bits/min)", ylim=(0, None))
    axes[0].set_title("(a) Accuracy")
    axes[1].set_title("(b) Information transfer rate")
    fig.get_layout_engine().set(h_pad=0.1)
    legend_below(fig, axes)
    fig.get_layout_engine().set(rect=(0, 0.12, 1, 0.88))
    return fig


# --- the figures -----------------------------------------------------------------


def fig_benchmark_methods(results: Path):
    df = pd.read_csv(results / "benchmark_6.csv")
    return accuracy_itr(df, chance=1 / len(config.TARGET_FREQS))


def fig_benchmark_40(results: Path):
    df = pd.read_csv(results / "benchmark_40.csv")
    return accuracy_itr(df, chance=1 / 40)


def fig_own_methods(results: Path):
    df = pd.read_csv(results / "own.csv")
    labels = session_labels(df["subject"])
    sessions = sorted(labels, key=labels.get)
    fig, axes = plt.subplots(
        len(sessions),
        2,
        figsize=(WIDTH, 2.3 * len(sessions) + 0.6),
        sharex=True,
        layout="constrained",
        squeeze=False,
    )
    itr_top = 1.05 * df["itr"].max()  # one ITR scale for all sessions
    for row, subject in enumerate(sessions):
        part = df[df["subject"] == subject]
        mean_lines(axes[row, 0], part, "window", "accuracy", band=False)
        mean_lines(axes[row, 1], part, "window", "itr", band=False)
        chance_line(axes[row, 0], 1 / len(config.TARGET_FREQS), df["window"].max())
        axes[row, 0].set(ylabel="Accuracy", ylim=(0, 1.02))
        axes[row, 1].set(ylabel="ITR (bits/min)", ylim=(0, itr_top))
        axes[row, 0].set_title(f"{labels[subject]}: accuracy")
        axes[row, 1].set_title(f"{labels[subject]}: ITR")
    for ax in axes[-1]:
        ax.set_xlabel("Window length (s)")
    legend_below(fig, axes)
    fig.get_layout_engine().set(rect=(0, 0.08, 1, 0.92))
    return fig


def fig_own_vs_benchmark(results: Path):
    bench = pd.read_csv(results / "benchmark_6.csv")
    own = pd.read_csv(results / "own.csv")
    labels = session_labels(own["subject"])
    methods = [m for m in COMPARED_METHODS if m in set(own["method"])]
    n_cols = 2
    n_rows = int(np.ceil(len(methods) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(WIDTH, 2.2 * n_rows + 0.5),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    x_max = max(bench["window"].max(), own["window"].max())
    for ax, method in zip(axes.flat, methods):
        rows = bench[bench["method"] == method]
        stats = rows.groupby("window")["accuracy"].agg(["mean", "sem"])
        n = rows["subject"].nunique()
        ax.plot(
            stats.index,
            stats["mean"],
            label=f"Benchmark ({n} subjects)",
            **BENCHMARK_STYLE,
        )
        ax.fill_between(
            stats.index,
            stats["mean"] - stats["sem"],
            stats["mean"] + stats["sem"],
            color=BENCHMARK_STYLE["color"],
            alpha=0.12,
            linewidth=0,
        )
        for i, subject in enumerate(sorted(labels, key=labels.get)):
            part = own[(own["subject"] == subject) & (own["method"] == method)]
            part = part.sort_values("window")
            ax.plot(
                part["window"],
                part["accuracy"],
                label=labels[subject],
                **SESSION_STYLES[i % len(SESSION_STYLES)],
            )
        chance_line(ax, 1 / len(config.TARGET_FREQS), x_max)
        ax.set_title(method)
        ax.set_ylim(0, 1.02)
    for ax in axes.flat[len(methods) :]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("Window length (s)")
    legend_below(fig, axes, ncol=3)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    return fig


def fig_own_latency(results: Path):
    df = pd.read_csv(results / "own_latency.csv")
    labels = session_labels(df["subject"])
    sessions = sorted(labels, key=labels.get)
    windows = sorted(df["window"].unique())
    fig, axes = plt.subplots(
        len(sessions),
        len(windows),
        figsize=(WIDTH, 2.2 * len(sessions) + 0.6),
        sharex=True,
        sharey=True,
        layout="constrained",
        squeeze=False,
    )
    for row, subject in enumerate(sessions):
        for col, window in enumerate(windows):
            ax = axes[row, col]
            part = df[(df["subject"] == subject) & (df["window"] == window)]
            mean_lines(ax, part, "latency", "accuracy", band=False)
            ax.axvline(OWN_LATENCY, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
            chance_line(ax, 1 / len(config.TARGET_FREQS), df["latency"].max())
            ax.set_title(f"{labels[subject]}, {window:g} s window")
            ax.set_ylim(0, 1.02)
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")
    for ax in axes[-1]:
        ax.set_xlabel("Latency (s)")
    legend_below(fig, axes, ncol=6)
    fig.get_layout_engine().set(rect=(0, 0.07, 1, 0.93))
    return fig


def fig_artifact_ratios(results: Path):
    bench = pd.read_csv(results / "ratios_benchmark_trials.csv")
    own = pd.read_csv(results / "ratios_own_trials.csv")
    labels = session_labels(own["subject"])
    subjects = sorted(bench["subject"].unique(), key=lambda s: int(s.lstrip("S")))
    sessions = sorted(labels, key=labels.get)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 2.8),
        sharey=True,
        layout="constrained",
        gridspec_kw={"width_ratios": [len(subjects), max(8, 4 * len(sessions))]},
    )
    rng = np.random.default_rng(0)  # jitter only, so the points don't overlap
    for ax, table, names, ticks in [
        (axes[0], bench, subjects, [s.lstrip("S") for s in subjects]),
        (axes[1], own, sessions, [str(i + 1) for i in range(len(sessions))]),
    ]:
        for i, name in enumerate(names):
            ratio = table.loc[table["subject"] == name, "ratio"].to_numpy()
            x = i + rng.uniform(-0.25, 0.25, len(ratio))
            ax.scatter(x, ratio, s=6, color=INK, alpha=0.5, linewidths=0)
        ax.axhline(
            config.MAX_TRIAL_PTP_RATIO, color=COLORS[1], linewidth=1, linestyle="--"
        )
        ax.set_xticks(range(len(names)), ticks)
        ax.set_xlim(-0.7, len(names) - 0.3)
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    axes[0].tick_params(axis="x", labelsize=6)
    axes[0].set(xlabel="Benchmark subject", ylabel="Trial ratio (worst channel)")
    axes[1].set(xlabel="Session")
    axes[1].annotate(
        f"threshold {config.MAX_TRIAL_PTP_RATIO:g}",
        (len(sessions) - 0.3, config.MAX_TRIAL_PTP_RATIO),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=COLORS[1],
        fontsize=8,
    )
    axes[0].set_title("(a) Benchmark, 2 s window")
    axes[1].set_title("(b) Own, 2 s")
    return fig


def fig_timing(results: Path):
    table = pd.read_csv(results / "timing.csv")
    sessions = table[table["kind"] != "flicker test"]
    tests = table[(table["kind"] == "flicker test") & ~table["windowed"].astype(bool)]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(WIDTH, 2.6),
        layout="constrained",
        gridspec_kw={"width_ratios": [3, 2]},
    )

    ax = axes[0]
    period_ms = 1000 / config.REFRESH_RATE
    bins = np.arange(period_ms - 1.5, 2 * period_ms + 2.5, 0.1)
    for i, (date, part) in enumerate(sessions.groupby("date")):
        intervals = np.concatenate(
            [analyse_timing.flicker_intervals(config.ROOT / f)[0] for f in part["file"]]
        )
        style = SESSION_STYLES[i % len(SESSION_STYLES)]
        ms = np.clip(1000 * intervals, bins[0], bins[-1] - 1e-6)
        ax.hist(
            ms,
            bins=bins,
            histtype="step",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            label=f"Session {i + 1} ({len(intervals):,} intervals)",
        )
    ax.set_yscale("log")
    ax.axvline(2 * period_ms, color=INK_SECONDARY, linewidth=0.8, linestyle=":")
    ax.annotate(
        "one refresh\nmissed",
        (2 * period_ms, 1),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=8,
        color=INK_SECONDARY,
    )
    ax.set_xlim(period_ms - 1.5, 2 * period_ms + 2.5)
    ax.set(xlabel="Flip interval during the flicker (ms)", ylabel="Number of intervals")
    ax.set_title("(a) Recording sessions")
    ax.legend(frameon=False, loc="upper right")

    ax = axes[1]
    ok = tests["n_dropped"] == 0
    dates = pd.to_datetime(tests["date"])
    for mask, label, filled in [
        (ok, "no dropped frames", True),
        (~ok, "dropped frames", False),
    ]:
        ax.scatter(
            dates[mask],
            tests.loc[mask, "sd_interval_ms"],
            s=18,
            facecolors=INK if filled else "white",
            edgecolors=INK,
            linewidths=1,
            label=label,
            zorder=3,
        )
    ax.axhline(0.3, color=COLORS[1], linewidth=1, linestyle="--")
    ax.annotate(
        "limit 0.3 ms",
        (dates.max(), 0.3),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color=COLORS[1],
        fontsize=8,
    )
    ax.set_yscale("log")
    ax.set_ylim(0.05, 8)
    ax.set(ylabel="Flip interval SD (ms)", xlabel="Date of the flicker test")
    span_days = max(1, (dates.max() - dates.min()).days)
    ax.xaxis.set_major_locator(
        matplotlib.dates.DayLocator(interval=max(1, int(np.ceil(span_days / 3))))
    )
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %b"))
    ax.set_title("(b) Flicker tests")
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    return fig


def fig_filter_bank(results: Path):
    fs = config.EEG_FS
    n = 20 * fs  # 20 s impulse response: fine frequency grid
    impulse = np.zeros(n)
    impulse[n // 2] = 1.0
    freqs = np.fft.rfftfreq(n, 1 / fs)

    def gain_db(signal):
        return 20 * np.log10(np.maximum(np.abs(np.fft.rfft(signal)), 1e-6))

    fig, ax = plt.subplots(figsize=(WIDTH, 2.6), layout="constrained")
    ax.axvspan(
        min(config.TARGET_FREQS),
        max(config.TARGET_FREQS),
        color=GRID,
        alpha=0.6,
        linewidth=0,
        label="target frequencies",
    )
    subbands = filter_bank(impulse[None, :], fs, config.N_SUBBANDS)[:, 0]
    shades = plt.get_cmap("Blues")(np.linspace(0.45, 0.95, config.N_SUBBANDS))
    for m, (band, shade) in enumerate(zip(subbands, shades), start=1):
        ax.plot(freqs, gain_db(band), color=shade, label=f"sub-band {m}")
    ax.plot(
        freqs,
        gain_db(bandpass(impulse, fs, *config.BANDPASS)),
        color=INK,
        linestyle="--",
        label=f"band-pass {config.BANDPASS[0]:g}-" f"{config.BANDPASS[1]:g} Hz",
    )
    ax.set(
        xlim=(0, fs / 2),
        ylim=(-60, 5),
        xlabel="Frequency (Hz)",
        ylabel="Gain (dB), forward and backward",
    )
    ax.legend(frameon=False, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    return fig


FIGURES = {
    "benchmark_methods": (fig_benchmark_methods, ["benchmark_6.csv"]),
    "benchmark_40": (fig_benchmark_40, ["benchmark_40.csv"]),
    "own_methods": (fig_own_methods, ["own.csv"]),
    "own_vs_benchmark": (fig_own_vs_benchmark, ["benchmark_6.csv", "own.csv"]),
    "own_latency": (fig_own_latency, ["own_latency.csv"]),
    "artifact_ratios": (
        fig_artifact_ratios,
        ["ratios_benchmark_trials.csv", "ratios_own_trials.csv"],
    ),
    "timing": (fig_timing, ["timing.csv"]),
    "filter_bank": (fig_filter_bank, []),
}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--results", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="+", choices=list(FIGURES), default=None)
    args = parser.parse_args(argv)

    made, skipped = [], []
    for name in args.only or FIGURES:
        make, inputs = FIGURES[name]
        missing = [f for f in inputs if not (args.results / f).exists()]
        if missing:
            skipped.append(f"{name} (missing {', '.join(missing)})")
            continue
        with plt.rc_context(STYLE):
            fig = make(args.results)
            path = args.out_dir / f"{name}.pdf"
            fig.savefig(path)
            plt.close(fig)
        made.append(name)
        print(f"Saved {path}")
    for line in skipped:
        print(f"Skipped {line}")
    return made, skipped


if __name__ == "__main__":
    main()
