"""Offline comparison of all classifiers: accuracy and ITR vs. window length.

Usage:
    .venv/bin/python scripts/compare_methods.py --data synthetic
    .venv/bin/python scripts/compare_methods.py --data benchmark --subjects 1 2 3
    .venv/bin/python scripts/compare_methods.py --data recording --recordings data/recordings/*.npz

Output: results/<name>.csv (one row per subject x method x window) and
results/<name>.png (accuracy and ITR vs. window, mean ± SEM over subjects).

Every method is evaluated with leave-one-block-out cross-validation. ITR counts
window + config.GAZE_SHIFT seconds per selection.

Every method is wrapped in preprocessing.RejectArtifacts: bad channels and bad trials
are left out of each training fold (thresholds --trial-ratio, --channel-ratio; 0
switches a step off), and every test trial is classified. Own recordings are scored
with the frequencies they were recorded with (their session_start marker).
"""

import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ssvep import config
from ssvep.baseline import make_psd_lda
from ssvep.cca import CCA, NormalizedCCA
from ssvep.datasets import (
    load_benchmark,
    load_recording,
    make_synthetic,
    recording_freqs,
)
from ssvep.evaluation import window_sweep
from ssvep.preprocessing import (
    RejectArtifacts,
    bad_channel_mask,
    bad_trial_mask,
    bandpass,
    crop,
    filter_bank,
    notch,
)
from ssvep.trca import TRCA

# Method name -> (n_subbands, factory returning a fresh estimator).
# n_subbands == 1 means a single band-pass (config.BANDPASS) instead of a filter bank.
# Every factory takes (freqs, fs), also when the method needs neither.
METHODS = {
    "CCA": (1, lambda freqs, fs: CCA(freqs, fs, n_harmonics=config.N_HARMONICS)),
    "FBCCA": (
        config.N_SUBBANDS,
        lambda freqs, fs: CCA(
            freqs,
            fs,
            n_harmonics=config.N_HARMONICS,
            fb_a=config.FB_A,
            fb_b=config.FB_B,
        ),
    ),
    "TRCA": (1, lambda freqs, fs: TRCA()),
    "eTRCA": (1, lambda freqs, fs: TRCA(ensemble=True)),
    "FB-eTRCA": (
        config.N_SUBBANDS,
        lambda freqs, fs: TRCA(ensemble=True, fb_a=config.FB_A, fb_b=config.FB_B),
    ),
    "PSD+LDA": (1, lambda freqs, fs: make_psd_lda(fs, freqs)),
    "CCA-norm": (
        1,
        lambda freqs, fs: NormalizedCCA(freqs, fs, n_harmonics=config.N_HARMONICS),
    ),
    "FBCCA-norm": (
        config.N_SUBBANDS,
        lambda freqs, fs: NormalizedCCA(
            freqs,
            fs,
            n_harmonics=config.N_HARMONICS,
            fb_a=config.FB_A,
            fb_b=config.FB_B,
        ),
    ),
}
NORM_SUFFIX = "-norm"  # a normalised variant is drawn like its base method, dashed

# One color and marker per base method, by its position in METHODS, so a method keeps
# its look when only some methods are run. Categorical palette checked for color-vision
# deficiency; the markers tell the lines apart in grayscale print.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK_SECONDARY, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SYNTHETIC_PRE = 0.5  # s, noise before the synthetic SSVEP starts


def load_dataset(args):
    """Yield (subject_name, X, y, blocks, channels, freqs, fs, start) for every subject.

    ``channels`` are the channel labels of X, ``freqs`` the target frequencies that y
    indexes, ``start`` = stimulus onset + visual latency, in seconds.
    """
    if args.data == "synthetic":
        latency = 0.0 if args.latency is None else args.latency
        channels = [f"ch{c + 1}" for c in range(len(config.BENCHMARK_CHANNELS))]
        for subject in args.subjects:
            X, y, blocks = make_synthetic(pre=SYNTHETIC_PRE, seed=subject)
            yield (
                f"synthetic{subject}",
                X,
                y,
                blocks,
                channels,
                config.TARGET_FREQS,
                config.BENCHMARK_FS,
                SYNTHETIC_PRE + latency,
            )
        return

    latency = config.BENCHMARK_LATENCY if args.latency is None else args.latency
    start = config.BENCHMARK_ONSET + latency
    if args.data == "benchmark":
        for subject in args.subjects:
            X, y, blocks = load_benchmark(subject)
            yield (
                f"S{subject}",
                X,
                y,
                blocks,
                config.BENCHMARK_CHANNELS,
                config.TARGET_FREQS,
                config.BENCHMARK_FS,
                start,
            )
    else:
        for path in map(Path, args.recordings):
            X, y, blocks = load_recording(path)
            # Benchmark was notch-filtered during recording; own data still has mains.
            X = notch(X, config.EEG_FS, config.LINE_FREQ)
            yield (
                path.stem,
                X,
                y,
                blocks,
                config.CAP_CHANNELS,
                recording_freqs(path),
                config.EEG_FS,
                start,
            )


def prefilter(X, fs, n_subbands):
    """Raw epochs -> uncropped classifier input: one band-pass, or a filter bank."""
    if n_subbands == 1:
        return bandpass(X, fs, *config.BANDPASS)[:, None]
    return filter_bank(X, fs, n_subbands)


def with_rejection(make_estimator, trial_ratio, channel_ratio):
    """A fresh estimator from ``make_estimator()``, wrapped in RejectArtifacts."""
    return RejectArtifacts(
        make_estimator(), trial_ratio=trial_ratio, channel_ratio=channel_ratio
    )


def rejection_summary(X, channels, fs, start, window, trial_ratio, channel_ratio):
    """What RejectArtifacts would leave out if it trained on all trials -> one line.

    Measured like in fit (config.BANDPASS, window cropped at ``start``). The training
    folds of the cross-validation see a subset of the trials, so they can differ a bit.
    """
    X = crop(bandpass(X, fs, *config.BANDPASS), fs, start, window)
    keep = np.ones(X.shape[1], dtype=bool)
    if channel_ratio:
        keep = ~bad_channel_mask(X, max_ratio=channel_ratio)
    dropped = [c for c, k in zip(channels, keep) if not k]
    n_bad = (
        int(bad_trial_mask(X[:, keep], max_ratio=trial_ratio).sum())
        if trial_ratio
        else 0
    )
    return (
        f"channels dropped: {', '.join(dropped) or 'none'}; "
        f"trials flagged: {n_bad} of {len(X)} (all trials, {window:g} s window)"
    )


def compare(datasets, methods, windows, trial_ratio=None, channel_ratio=None):
    """Leave-one-block-out window sweep of every method on every subject.

    With ``trial_ratio`` or ``channel_ratio`` set, every method is wrapped in
    RejectArtifacts (training folds only). Returns a DataFrame with columns
    ``subject, method, window, accuracy, itr``.
    """
    print(f"{'accuracy':>23}  " + "  ".join(f"{w:>5g}s" for w in windows))
    tables = []
    for subject, X, y, blocks, channels, freqs, fs, start in datasets:
        if trial_ratio or channel_ratio:
            summary = rejection_summary(
                X, channels, fs, start, max(windows), trial_ratio, channel_ratio
            )
            print(f"{subject:>12} {summary}")
        filtered = (
            {}
        )  # n_subbands -> filtered epochs, shared by methods with that input
        for method in methods:
            n_subbands, factory = METHODS[method]
            if n_subbands not in filtered:
                filtered[n_subbands] = prefilter(X, fs, n_subbands)
            make_estimator = partial(factory, freqs, fs)
            if trial_ratio or channel_ratio:
                make_estimator = partial(
                    with_rejection, make_estimator, trial_ratio, channel_ratio
                )
            table = window_sweep(
                make_estimator,
                filtered[n_subbands],
                y,
                blocks,
                fs,
                start,
                windows,
                gaze_shift=config.GAZE_SHIFT,
            )
            table.insert(0, "method", method)
            table.insert(0, "subject", subject)
            tables.append(table)
            accuracies = "  ".join(f"{a:6.2f}" for a in table["accuracy"])
            print(f"{subject:>12} {method:<10}  {accuracies}")
    return pd.concat(tables, ignore_index=True)


def plot_results(df, path, chance=None, note=""):
    """Two panels, accuracy vs. window and ITR vs. window, one line per method.

    Lines are means over subjects, bands ± SEM (only with more than one subject).
    ``chance`` draws the chance level on the accuracy panel; ``note`` is appended to
    the title (e.g. how training data was cleaned).
    """
    n_subjects = df["subject"].nunique()
    style = {
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": GRID,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
        for ax, metric, label in [
            (axes[0], "accuracy", "Accuracy"),
            (axes[1], "itr", "ITR (bits/min)"),
        ]:
            for method in df["method"].unique():
                base = method.removesuffix(NORM_SUFFIX)
                i = list(METHODS).index(base)
                rows = df[df["method"] == method]
                stats = rows.groupby("window")[metric].agg(["mean", "sem"])
                ax.plot(
                    stats.index,
                    stats["mean"],
                    color=COLORS[i],
                    marker=MARKERS[i],
                    linestyle="--" if method != base else "-",
                    linewidth=2,
                    markersize=6,
                    label=method,
                )
                if n_subjects > 1:
                    ax.fill_between(
                        stats.index,
                        stats["mean"] - stats["sem"],
                        stats["mean"] + stats["sem"],
                        color=COLORS[i],
                        alpha=0.12,
                        linewidth=0,
                    )
            if metric == "accuracy":
                ax.set_ylim(0, 1.02)
                if chance is not None:
                    ax.axhline(chance, color=INK_SECONDARY, linewidth=1, linestyle="--")
                    ax.annotate(
                        "chance",
                        (df["window"].max(), chance),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="right",
                        color=INK_SECONDARY,
                    )
            else:
                ax.set_ylim(bottom=0)
            ax.set_xlabel("Window length (s)")
            ax.set_ylabel(label)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="outside lower center",
            # one row fits 6 entries; more go on two rows
            ncol=len(labels) if len(labels) <= 6 else (len(labels) + 1) // 2,
            frameon=False,
            handlelength=3,  # long enough to show the dashes of a normalised variant
        )
        subjects = f"{n_subjects} subject{'s' if n_subjects > 1 else ''}"
        spread = ", mean ± SEM" if n_subjects > 1 else ""
        fig.suptitle(
            f"Leave-one-block-out, {subjects}{spread}; "
            f"ITR includes a {config.GAZE_SHIFT:g} s gaze shift{note}",
            color=INK_SECONDARY,
            fontsize="medium",
        )
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)


def print_summary(df):
    """Mean accuracy per method and window, and each method's best mean ITR."""
    order = [m for m in METHODS if m in set(df["method"])]
    means = df.groupby(["method", "window"])[["accuracy", "itr"]].mean()
    accuracy = means["accuracy"].unstack().reindex(order)
    print(f"\nMean accuracy over {df['subject'].nunique()} subject(s):")
    print(accuracy.to_string(float_format=lambda a: f"{a:.2f}"))
    print("\nBest mean ITR:")
    for method in order:
        itr = means.loc[method, "itr"]
        print(f"  {method:<10} {itr.max():6.1f} bits/min at {itr.idxmax():g} s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", choices=["synthetic", "benchmark", "recording"], default="synthetic"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--recordings", nargs="+", default=[])
    parser.add_argument(
        "--windows",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    )
    parser.add_argument(
        "--methods", nargs="+", choices=list(METHODS), default=list(METHODS)
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=None,
        help="visual latency in s (default: Benchmark's 0.14 s; 0 for synthetic data)",
    )
    parser.add_argument(
        "--trial-ratio",
        type=float,
        default=config.MAX_TRIAL_PTP_RATIO,
        help="leave a training trial out if a channel swings more than this many times "
        "its median peak-to-peak (0 = keep all trials)",
    )
    parser.add_argument(
        "--channel-ratio",
        type=float,
        default=config.MAX_CHANNEL_SD_RATIO,
        help="leave a channel out if its sd is more than this many times, or less than "
        "1/this of, the typical channel's (0 = keep all channels)",
    )
    parser.add_argument(
        "--name", default=None, help="output file stem (default: derived from --data)"
    )
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.data == "recording" and not args.recordings:
        parser.error("--data recording needs --recordings")

    name = args.name or f"compare_{args.data}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = compare(
        load_dataset(args),
        args.methods,
        args.windows,
        trial_ratio=args.trial_ratio,
        channel_ratio=args.channel_ratio,
    )
    df.to_csv(args.out_dir / f"{name}.csv", index=False)
    cleaned = args.trial_ratio or args.channel_ratio
    plot_results(
        df,
        args.out_dir / f"{name}.png",
        chance=1 / len(config.TARGET_FREQS),
        note="; bad training trials/channels left out" if cleaned else "",
    )
    print_summary(df)
    print(f"\nSaved {args.out_dir / name}.csv and .png")


if __name__ == "__main__":
    main()
