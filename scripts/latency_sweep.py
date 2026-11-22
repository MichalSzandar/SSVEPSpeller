"""Accuracy and ITR of every method at several visual latencies.

Usage:
    .venv/bin/python scripts/latency_sweep.py --recordings A.npz B.npz \\
        --latencies 0.14 0.3 0.5 0.7 0.9 --windows 1 2 --name own_latency

The latency is where the analysis window starts after the onset marker. On the cap it
also absorbs any offset between the EEG and marker timestamps (thesis Sec. 3.7), so
the best latency can differ between sessions. For every latency, this runs the
leave-one-block-out window sweep of compare_methods.py (same preprocessing and
artifact rejection) and writes one table:

    results/<name>.csv  subject, method, latency, window, accuracy, itr

latency + window must fit inside the flicker (5 s for a calibration).
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import compare_methods
from ssvep import config


def sweep(
    data: str,
    latencies: list[float],
    windows: list[float],
    methods: list[str],
    recordings: list[str] = (),
    subjects: list[int] = (1,),
    trial_ratio: float = config.MAX_TRIAL_PTP_RATIO,
    channel_ratio: float = config.MAX_CHANNEL_SD_RATIO,
) -> pd.DataFrame:
    """compare_methods.compare at every latency -> one table with a latency column.

    ``data`` is "recording" (``recordings``) or "synthetic" (``subjects``).
    """
    tables = []
    for latency in latencies:
        print(f"\n--- latency {latency:g} s ---")
        args = SimpleNamespace(
            data=data,
            recordings=list(recordings),
            subjects=list(subjects),
            latency=latency,
        )
        table = compare_methods.compare(
            compare_methods.load_dataset(args),
            methods,
            windows,
            trial_ratio=trial_ratio,
            channel_ratio=channel_ratio,
        )
        table.insert(2, "latency", latency)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data", choices=["recording", "synthetic"], default="recording"
    )
    parser.add_argument("--recordings", nargs="+", default=[])
    parser.add_argument("--subjects", type=int, nargs="+", default=[1])
    parser.add_argument("--latencies", type=float, nargs="+", required=True)
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(compare_methods.METHODS),
        default=list(compare_methods.METHODS),
    )
    parser.add_argument("--trial-ratio", type=float, default=config.MAX_TRIAL_PTP_RATIO)
    parser.add_argument(
        "--channel-ratio", type=float, default=config.MAX_CHANNEL_SD_RATIO
    )
    parser.add_argument("--name", default="latency_sweep", help="output file stem")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)
    if args.data == "recording" and not args.recordings:
        parser.error("--data recording needs --recordings")

    table = sweep(
        args.data,
        args.latencies,
        args.windows,
        args.methods,
        recordings=args.recordings,
        subjects=args.subjects,
        trial_ratio=args.trial_ratio,
        channel_ratio=args.channel_ratio,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.csv"
    table.to_csv(path, index=False)

    print("\nAccuracy by latency (rows) and window (columns), per subject and method:")
    pivot = table.pivot_table(
        index=["subject", "method", "latency"], columns="window", values="accuracy"
    )
    print(pivot.to_string(float_format=lambda a: f"{a:.2f}"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
