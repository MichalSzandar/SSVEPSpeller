"""Inspect live LSL streams.

Usage:
    .venv/bin/python scripts/check_lsl.py                       # list all streams
    .venv/bin/python scripts/check_lsl.py --type EEG --seconds 10
    .venv/bin/python scripts/check_lsl.py --type EEG --alpha    # eyes open/closed test

Prints the streams on the network, the channel labels of the selected stream, and
after recording its effective sampling rate and per-channel standard deviation.
"""

import argparse

import numpy as np
import pylsl


def list_streams(wait_time: float) -> list[pylsl.StreamInfo]:
    """All streams visible on the network."""
    raise NotImplementedError


def channel_labels(inlet: pylsl.StreamInlet) -> list[str]:
    """Channel labels from the stream's XML description."""
    raise NotImplementedError


def record(inlet: pylsl.StreamInlet, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (samples (n, n_channels), timestamps (n,))."""
    raise NotImplementedError


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--type", default=None, help="stream type to inspect, e.g. EEG")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--wait", type=float, default=2.0, help="seconds to search for streams")
    parser.add_argument("--alpha", action="store_true", help="eyes open/closed alpha test")
    args = parser.parse_args()
    raise NotImplementedError


if __name__ == "__main__":
    main()
