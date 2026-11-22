"""What the artifact rejection sees: trial and channel ratios of every subject.

Usage:
    .venv/bin/python scripts/artifact_ratios.py --data benchmark --subjects 1 2 3 \\
        --window 2 --name ratios_benchmark
    .venv/bin/python scripts/artifact_ratios.py --data recording --recordings A.npz B.npz \\
        --latency 0.7 --window 2 --name ratios_own

The measures of preprocessing.RejectArtifacts, computed as if it trained on all trials
of a subject (a training fold sees a subset, so a fold can differ slightly). The
window is band-passed (config.BANDPASS) and cut at onset + latency, as in
compare_methods.rejection_summary.
  - channel level ratio: the channel's median in-trial SD over the median of all
    channels; a channel above --channel-ratio or below 1/--channel-ratio is dropped.
  - trial ratio, on the remaining channels: the largest, over channels, of the trial's
    peak-to-peak amplitude divided by that channel's median over trials; a trial above
    --trial-ratio is left out of training.

Output:
  results/<name>_trials.csv    subject, trial, block, target, ratio, worst_channel,
                               flagged
  results/<name>_channels.csv  subject, channel, level_ratio, dropped
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import compare_methods
from ssvep import config
from ssvep.preprocessing import bad_channel_mask, bandpass, crop


def ratios(
    subject: str,
    X: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    channels: list[str],
    fs: float,
    start: float,
    window: float,
    trial_ratio: float = config.MAX_TRIAL_PTP_RATIO,
    channel_ratio: float = config.MAX_CHANNEL_SD_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trial and channel tables (columns at the top of this file) of one subject.

    X : raw epochs (n_trials, n_channels, n_samples), as the loaders return them.
    """
    X = crop(bandpass(X, fs, *config.BANDPASS), fs, start, window)

    level = np.median(X.std(axis=-1), axis=0)  # (n_channels,)
    dropped = bad_channel_mask(X, max_ratio=channel_ratio)
    channel_table = pd.DataFrame(
        {
            "subject": subject,
            "channel": list(channels),
            "level_ratio": level / np.median(level),
            "dropped": dropped,
        }
    )

    kept = [c for c, d in zip(channels, dropped) if not d]
    ptp = np.ptp(X[:, ~dropped], axis=-1)  # (n_trials, n_kept)
    median = np.median(ptp, axis=0)
    per_channel = np.divide(
        ptp, median, out=np.zeros_like(ptp, dtype=float), where=median > 0
    )
    worst = per_channel.argmax(axis=1)
    trial_ratios = per_channel.max(axis=1)
    trial_table = pd.DataFrame(
        {
            "subject": subject,
            "trial": np.arange(len(X)),
            "block": blocks,
            "target": y,
            "ratio": trial_ratios,
            "worst_channel": [kept[i] for i in worst],
            "flagged": trial_ratios > trial_ratio,
        }
    )
    return trial_table, channel_table


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
        "--latency",
        type=float,
        default=None,
        help="visual latency in s (default as in compare_methods.py)",
    )
    parser.add_argument("--window", type=float, default=2.0, help="window length in s")
    parser.add_argument("--trial-ratio", type=float, default=config.MAX_TRIAL_PTP_RATIO)
    parser.add_argument(
        "--channel-ratio", type=float, default=config.MAX_CHANNEL_SD_RATIO
    )
    parser.add_argument("--name", default="ratios", help="output file stem")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.data == "recording" and not args.recordings:
        parser.error("--data recording needs --recordings")

    trial_tables, channel_tables = [], []
    for subject, X, y, blocks, channels, _, fs, start in compare_methods.load_dataset(
        args
    ):
        trials, chans = ratios(
            subject,
            X,
            y,
            blocks,
            channels,
            fs,
            start,
            args.window,
            args.trial_ratio,
            args.channel_ratio,
        )
        trial_tables.append(trials)
        channel_tables.append(chans)
        dropped = ", ".join(chans.loc[chans["dropped"], "channel"]) or "none"
        print(
            f"{subject:>24}: trial ratio median {trials['ratio'].median():.2f}, "
            f"max {trials['ratio'].max():.1f}; {trials['flagged'].sum()} of "
            f"{len(trials)} trials flagged; channels dropped: {dropped}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    trials = pd.concat(trial_tables, ignore_index=True)
    chans = pd.concat(channel_tables, ignore_index=True)
    trials.to_csv(args.out_dir / f"{args.name}_trials.csv", index=False)
    chans.to_csv(args.out_dir / f"{args.name}_channels.csv", index=False)
    print(
        f"\nLevel ratios of all channels: {chans['level_ratio'].min():.2f}-"
        f"{chans['level_ratio'].max():.2f} (limits {1 / args.channel_ratio:.2f}-"
        f"{args.channel_ratio:g})"
    )
    print(f"Saved {args.out_dir / args.name}_trials.csv and _channels.csv")


if __name__ == "__main__":
    main()
