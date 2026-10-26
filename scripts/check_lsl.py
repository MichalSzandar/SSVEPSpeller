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
from scipy.signal import welch
from matplotlib import pyplot as plt
from ssvep import config


def list_streams(wait_time: float) -> list[pylsl.StreamInfo]:
    """All streams visible on the network."""
    return pylsl.resolve_streams(wait_time=wait_time)


def channel_labels(inlet: pylsl.StreamInlet) -> list[str]:
    """Channel labels from the stream's XML description."""
    xml = inlet.info().as_xml()
    labels = []
    for line in xml.splitlines():
        if "<label>" in line:
            label = line.strip().removeprefix("<label>").removesuffix("</label>")
            labels.append(label)
    return labels


def record(inlet: pylsl.StreamInlet, seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (samples (n, n_channels), timestamps (n,))."""
    inlet.flush()  # discard any old samples
    samples = []
    timestamps = []
    for _ in range(int(round(seconds * inlet.info().nominal_srate()))):
        sample, timestamp = inlet.pull_sample(timeout=1.0)
        if sample is None:
            break
        samples.append(sample)
        timestamps.append(timestamp)
    return np.array(samples), np.array(timestamps)


def print_stream_info(stream: pylsl.StreamInfo):
    """Print the name, type, channel count, nominal sampling rate, and source ID."""
    print(f"Stream: {stream.name()} ({stream.type()})")
    print(
        f"  channels: {stream.channel_count()}, srate: {stream.nominal_srate():.1f} Hz"
    )
    print(f"  source_id: {stream.source_id()}")


def stddev_by_channel(
    labels: list[str], samples: np.ndarray
) -> list[tuple[str, float]]:
    """Pair each channel label with its standard deviation."""
    if samples.size == 0:
        return []

    stddev = np.std(samples, axis=0)
    if np.ndim(stddev) == 0:
        stddev = np.asarray([stddev])

    if not labels:
        labels = [f"ch{i}" for i in range(len(stddev))]

    if len(labels) < len(stddev):
        labels = labels[: len(stddev)]
    elif len(labels) > len(stddev):
        labels = labels[: len(stddev)]

    return [(label, float(value)) for label, value in zip(labels, stddev)]


def select_stream(
    streams: list[pylsl.StreamInfo], stream_type: str | None
) -> pylsl.StreamInfo | None:
    """Pick a stream, preferring a data stream over marker streams when type is unspecified."""
    if stream_type is not None:
        matches = [s for s in streams if s.type() == stream_type]
        return matches[0] if matches else None

    data_streams = [s for s in streams if s.type().lower() not in {"marker", "markers"}]
    return data_streams[0] if data_streams else None


def print_recorded_info(
    samples: np.ndarray, timestamps: np.ndarray, labels: list[str] | None = None
):
    """Print the number of samples, effective sampling rate, and per-channel stddev."""
    print(f"Recorded {len(samples)} samples, {len(timestamps)} timestamps.")

    if len(samples) < 2 or len(timestamps) < 2:
        print(
            "Need at least 2 samples to compute effective sampling rate and stddev; this often means a marker stream was selected."
        )
        return

    span = timestamps[-1] - timestamps[0]
    if span <= 0:
        print("Cannot compute effective sampling rate: timestamps are not increasing.")
        return

    print(f"Effective sampling rate: {(len(samples) - 1) / span:.1f} Hz")

    label_list = labels or []
    for label, value in stddev_by_channel(label_list, samples):
        print(f"  {label}: {value:.6f}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--type", default=None, help="stream type to inspect, e.g. EEG")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--wait", type=float, default=2.0, help="seconds to search for streams"
    )
    parser.add_argument(
        "--alpha", action="store_true", help="eyes open/closed alpha test"
    )
    args = parser.parse_args()

    streams = list_streams(args.wait)
    selected_stream = select_stream(streams, args.type)
    if selected_stream is None:
        if args.type is not None:
            print(f"No streams found for type '{args.type}'.")
        else:
            print(
                "No data streams found. Pass --type EEG to inspect a specific stream."
            )
        return

    for stream in streams:
        print_stream_info(stream)

    inlet = pylsl.StreamInlet(selected_stream)

    ch_labels = channel_labels(inlet)
    print(f"Channel labels: {ch_labels}")
    print(
        f"\nRecording {args.seconds:.1f} seconds from stream {selected_stream.name()}..."
    )
    samples, timestamps = record(inlet, args.seconds)
    print_recorded_info(samples, timestamps, ch_labels)

    if args.alpha:
        print(
            "\nAlpha test: open your eyes for 10 seconds, then close them for 10 seconds."
        )
        input("Press Enter to start...")
        samples, timestamps = record(inlet, 10.0)
        fs_open, pxx_open = welch(
            samples, inlet.info().nominal_srate(), nperseg=256, axis=0
        )
        pxx_open = np.mean(pxx_open, axis=1)

        print("Now close your eyes for 10 seconds.")
        input("Press Enter to continue...")
        samples, timestamps = record(inlet, 10.0)
        fs_closed, pxx_closed = welch(
            samples, inlet.info().nominal_srate(), nperseg=256, axis=0
        )
        pxx_closed = np.mean(pxx_closed, axis=1)

        print("test finished, you can open your eyes again. \a")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(fs_open, pxx_open, label="Eyes Open")
        ax.semilogy(fs_closed, pxx_closed, label="Eyes Closed")
        ax.set_xlim(0, 50)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power Spectral Density")
        ax.set_title("Alpha Waves")
        ax.legend()
        ax.axvspan(8, 12, color="gray", alpha=0.2)
        fig.savefig(f"{config.RESULTS_DIR}/lsl_check/eyes_closed.png")
        plt.show()


if __name__ == "__main__":
    main()
