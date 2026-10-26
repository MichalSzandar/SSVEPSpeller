"""Offline comparison of all classifiers: accuracy and ITR vs. window length.

Usage:
    .venv/bin/python scripts/compare_methods.py --data synthetic
    .venv/bin/python scripts/compare_methods.py --data benchmark --subjects 1 2 3
    .venv/bin/python scripts/compare_methods.py --data recording --recordings data/recordings/*.npz

Output: results/<name>.csv (one row per subject x method x window) and
results/<name>.png (accuracy and ITR vs. window, mean ± SEM over subjects).
"""

import argparse

from ssvep import config

# Method name -> (n_subbands, factory returning a fresh estimator).
# n_subbands == 1 means a single band-pass (config.BANDPASS) instead of a filter bank.
METHODS = {}


def load_dataset(args):
    """Yield (subject_name, X, y, blocks, fs, start) for every subject.

    ``start`` = stimulus onset + visual latency, in seconds.
    """
    raise NotImplementedError


def plot_results(df, path):
    """Two panels, accuracy vs. window and ITR vs. window, one line per method."""
    raise NotImplementedError


def main():
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
        "--name", default=None, help="output file stem (default: derived from --data)"
    )
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
