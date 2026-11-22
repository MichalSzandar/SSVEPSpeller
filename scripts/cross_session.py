"""Train on one calibration, test on another: does a classifier carry over between
sessions?

Usage:
    .venv/bin/python scripts/cross_session.py --recordings A.npz B.npz --windows 1 2 3
    .venv/bin/python scripts/cross_session.py --recordings A.npz B.npz \\
        --latencies 0.7 0.2 --name cross_session_aligned

For every method and window, and every ordered pair (train, test) of the recordings:
  - train != test: an online.Decoder (the online preprocessing, classifier and
    artifact rejection) is fitted on all trials of ``train`` and classifies every trial
    of ``test``, as online with a calibration from another session;
  - train == test: the leave-one-block-out accuracy of compare_methods.py, the
    within-session reference.
Each recording is cut at its own latency (--latencies, one per recording; default
0.7 s for all), so a timestamp offset that only one session had can be compensated.

Output: results/<name>.csv with the columns train, test, method, window,
train_latency, test_latency, accuracy, itr (ITR with the window + gaze shift).
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import compare_methods
from online import Decoder, load_calibration
from ssvep import config
from ssvep.evaluation import itr

DEFAULT_LATENCY = 0.7  # s, the latency used with the cap (thesis Sec. 4.1)


def transfer_accuracy(
    method: str,
    window: float,
    train: tuple[np.ndarray, np.ndarray, float],
    test: tuple[np.ndarray, np.ndarray, float],
) -> float:
    """Accuracy on ``test`` of a Decoder fitted on ``train``.

    ``train`` and ``test`` are (X, y, latency): raw calibration epochs as
    load_calibration returns them, their labels, and the latency at which that
    recording's windows start.
    """
    X_train, y_train, latency_train = train
    X_test, y_test, latency_test = test
    fitted = Decoder(method, window=window, latency=latency_train).fit(X_train, y_train)
    # Same classifier, but the test epochs are cut at the test recording's latency
    tester = Decoder(method, window=window, latency=latency_test)
    tester.estimator_ = fitted.estimator_
    if X_test.shape[-1] < tester.n_samples:
        raise ValueError(
            f"Test epochs have {X_test.shape[-1]} samples, less than the "
            f"{tester.n_samples} needed for a {window:g} s window at latency "
            f"{latency_test:g} s"
        )
    predicted = tester.estimator_.predict(
        tester.prepare(X_test[..., : tester.n_samples])
    )
    return float(np.mean(predicted == y_test))


def within_session(path: Path, latency: float, methods, windows) -> pd.DataFrame:
    """compare_methods' leave-one-block-out sweep of one recording at ``latency``."""
    args = SimpleNamespace(data="recording", recordings=[str(path)], latency=latency)
    return compare_methods.compare(
        compare_methods.load_dataset(args),
        methods,
        windows,
        trial_ratio=config.MAX_TRIAL_PTP_RATIO,
        channel_ratio=config.MAX_CHANNEL_SD_RATIO,
    )


def cross_session(
    paths: list[Path], latencies: list[float], methods, windows
) -> pd.DataFrame:
    """The table described at the top of this file, for every ordered pair of
    ``paths`` (including each recording with itself)."""
    sessions = {
        path: (*load_calibration([path]), latency)
        for path, latency in zip(paths, latencies)
    }
    n_targets = len(config.TARGET_FREQS)
    rows = []
    for path, latency in zip(paths, latencies):
        within = within_session(path, latency, methods, windows)
        for row in within.itertuples():
            rows.append(
                {
                    "train": path.stem,
                    "test": path.stem,
                    "method": row.method,
                    "window": row.window,
                    "train_latency": latency,
                    "test_latency": latency,
                    "accuracy": row.accuracy,
                    "itr": row.itr,
                }
            )
    for train in paths:
        for test in paths:
            if train == test:
                continue
            for method in methods:
                for window in windows:
                    accuracy = transfer_accuracy(
                        method, window, sessions[train], sessions[test]
                    )
                    print(
                        f"train {train.stem} -> test {test.stem}  {method:<10} "
                        f"{window:g} s  {accuracy:.2f}"
                    )
                    rows.append(
                        {
                            "train": train.stem,
                            "test": test.stem,
                            "method": method,
                            "window": window,
                            "train_latency": sessions[train][2],
                            "test_latency": sessions[test][2],
                            "accuracy": accuracy,
                            "itr": itr(n_targets, accuracy, window + config.GAZE_SHIFT),
                        }
                    )
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--recordings", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--latencies",
        type=float,
        nargs="+",
        default=None,
        help=f"one per recording (default {DEFAULT_LATENCY:g} s for all)",
    )
    parser.add_argument("--windows", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(compare_methods.METHODS),
        default=list(compare_methods.METHODS),
    )
    parser.add_argument("--name", default="cross_session", help="output file stem")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)
    if len(args.recordings) < 2:
        parser.error("--recordings needs at least two calibrations")
    latencies = args.latencies or [DEFAULT_LATENCY] * len(args.recordings)
    if len(latencies) != len(args.recordings):
        parser.error("--latencies needs one value per recording")

    table = cross_session(args.recordings, latencies, args.methods, args.windows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.csv"
    table.to_csv(path, index=False)
    print("\nAccuracy (rows: method and window; columns: train -> test):")
    table["pair"] = table["train"] + " -> " + table["test"]
    pivot = table.pivot_table(
        index=["method", "window"], columns="pair", values="accuracy"
    )
    print(pivot.to_string(float_format=lambda a: f"{a:.2f}"))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
