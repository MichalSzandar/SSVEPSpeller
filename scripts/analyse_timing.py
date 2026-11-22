"""Stimulus timing of every session and flicker test: dropped frames, interval spread.

Usage:
    .venv/bin/python scripts/analyse_timing.py data/recordings/S02_*_flips.npz \\
        results/timing/flicker_*.npz --name timing

Two kinds of flip log:
  - a session log, <stem>_flips.npz (calibration.py) or <stem>_online_flips.npz
    (speller_gui.py): flip_times, trial, phase (0 cue/gaze, 1 flicker, 2 the stim_off
    flip) and refresh_rate. Only the flicker counts: in every trial, the intervals
    from its first flicker flip to its stim_off flip, as the GUI's own timing verdict
    (calibration.flicker_intervals) measures them.
  - a flicker test, results/timing/flicker_*.npz (calibration.py --test): one
    continuous flicker, so all of its intervals count. Its ``windowed`` flag marks a
    test in a window, whose timing is not valid.

Output: results/<name>.csv, one row per file: file, kind (calibration, online or
flicker test), date (YYYY-MM-DD from the file name), n_trials, n_intervals,
refresh_rate, mean_interval_ms, sd_interval_ms, min_interval_ms, max_interval_ms,
n_dropped (missed refreshes), n_late_flips (intervals with at least one), windowed,
verdict ("ok", or the problems of stimulus.timing_problems).
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ssvep import config, stimulus

FLICKER, OFF = 1, 2  # flip-log phases, as in scripts/calibration.py


def kind_of(path: Path) -> str:
    """calibration, online or flicker test, from the file name."""
    if path.name.startswith("flicker_"):
        return "flicker test"
    if path.name.endswith("_online_flips.npz"):
        return "online"
    if path.name.endswith("_flips.npz"):
        return "calibration"
    raise ValueError(f"{path.name} is not a flip log (*_flips.npz or flicker_*.npz)")


def date_of(path: Path) -> str:
    """YYYY-MM-DD from the first 8-digit date in the file name ("" if none)."""
    match = re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)
    return "-".join(match.groups()) if match else ""


def flicker_intervals(path: Path) -> tuple[np.ndarray, float, int, bool]:
    """Flip-to-flip intervals (s) of the flicker in a flip log -> (intervals,
    refresh_rate, n_trials, windowed).

    Session logs: per trial, the flicker flips and the stim_off flip, in time order.
    Flicker tests: all flips (n_trials 1).
    """
    with np.load(path, allow_pickle=False) as npz:
        times = np.asarray(npz["flip_times"], dtype=float)
        rate = float(npz["refresh_rate"])
        if kind_of(path) == "flicker test":
            windowed = bool(npz["windowed"]) if "windowed" in npz.files else False
            return np.diff(times), rate, 1, windowed
        trials = np.asarray(npz["trial"])
        phases = np.asarray(npz["phase"])

    parts = []
    trial_ids = np.unique(trials[phases == FLICKER])
    for trial in trial_ids:
        mask = (trials == trial) & ((phases == FLICKER) | (phases == OFF))
        parts.append(np.diff(np.sort(times[mask])))
    intervals = np.concatenate(parts) if parts else np.zeros(0)
    return intervals, rate, len(trial_ids), False


def summarize(path: Path) -> dict:
    """One row of the output table (columns at the top of this file)."""
    intervals, rate, n_trials, windowed = flicker_intervals(path)
    flips = np.r_[0.0, np.cumsum(intervals)]
    dropped = stimulus.dropped_frames(flips, rate)
    problems = stimulus.timing_problems(intervals, rate)
    return {
        "file": str(path),
        "kind": kind_of(path),
        "date": date_of(path),
        "n_trials": n_trials,
        "n_intervals": len(intervals),
        "refresh_rate": rate,
        "mean_interval_ms": 1000 * intervals.mean(),
        "sd_interval_ms": 1000 * intervals.std(),
        "min_interval_ms": 1000 * intervals.min(),
        "max_interval_ms": 1000 * intervals.max(),
        "n_dropped": int(dropped.sum()),
        "n_late_flips": int(np.count_nonzero(dropped)),
        "windowed": windowed,
        "verdict": (
            "not valid (windowed)" if windowed else ("; ".join(problems) or "ok")
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("files", type=Path, nargs="+", help="flip logs (.npz)")
    parser.add_argument("--name", default="timing", help="output file stem")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)

    table = pd.DataFrame([summarize(path) for path in args.files])
    table = table.sort_values(["date", "file"], ignore_index=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"{args.name}.csv"
    table.to_csv(path, index=False)
    for row in table.itertuples():
        print(
            f"{Path(row.file).name:<42} {row.kind:<12} {row.n_trials:>3} trials  "
            f"mean {row.mean_interval_ms:.3f} ms  sd {row.sd_interval_ms:.2f} ms  "
            f"max {row.max_interval_ms:5.1f} ms  dropped {row.n_dropped:>4}  "
            f"{row.verdict}"
        )
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
