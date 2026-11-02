"""Record an EEG stream and the calibration markers to data/recordings/*.npz.

Usage:
    .venv/bin/python scripts/record.py --subject S01
    .venv/bin/python scripts/record.py --subject test --eeg-name FakeEEG

Start it before or right after scripts/calibration.py, which waits for a recorder on
the marker stream. It stops by itself TAIL s after the session_end marker, or on
Ctrl+C, and in both cases saves data/recordings/<subject>_<YYYYmmdd_HHMMSS>.npz in
the format described at the top of ssvep/acquisition.py. While recording it prints
every marker as it arrives and the sample count every STATUS_EVERY s.
"""

import argparse
from datetime import datetime
import json
import time

from ssvep import acquisition, config

TAIL = 1.0  # s of EEG kept after session_end; EEG chunks arrive later than markers
STATUS_EVERY = 2.0  # s between sample count prints


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", required=True)
    parser.add_argument("--eeg-type", default="EEG", help="LSL type of the EEG stream")
    parser.add_argument("--eeg-name", default=None, help="LSL name, if several match")
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="seconds to wait for each stream"
    )
    args = parser.parse_args()

    # Before recording, so a missing folder can't lose a finished session
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    path = config.RECORDINGS_DIR / f"{args.subject}_{stamp}.npz"

    # EEG first: calibration.py starts as soon as the marker inlet connects
    print(f"Looking for EEG stream of type {args.eeg_type}...")
    eeg_info = acquisition.find_stream(
        args.eeg_type, args.eeg_name, timeout=args.timeout
    )
    eeg_inlet = acquisition.open_eeg_inlet(eeg_info)
    print(
        f"Found EEG stream {eeg_info.name()} ({eeg_info.type()}) with "
        f"{eeg_info.channel_count()} channels at {eeg_info.nominal_srate()} Hz"
    )

    # The resolved info has no description, the inlet's has the labels
    channels = acquisition.channel_labels(eeg_inlet.info())
    print(f"Channel labels: {channels}")

    print(f"Looking for marker stream {config.MARKER_STREAM}...")
    marker_info = acquisition.find_stream(
        "Markers", config.MARKER_STREAM, timeout=args.timeout
    )
    marker_inlet = acquisition.open_marker_inlet(marker_info)
    print(f"Found marker stream {marker_info.name()} ({marker_info.type()})")

    recorder = acquisition.Recorder(
        eeg_inlet, marker_inlet, channels, fs=eeg_info.nominal_srate()
    )
    print(f"Recording to {path}...")
    session_end = None  # LSL time of the session_end marker, once it has arrived
    last_status = time.monotonic()
    try:
        while True:
            for marker, timestamp in recorder.pull():
                event = json.loads(marker)["event"]
                print(f"{recorder.n_samples} samples, marker {event}")
                if event == "session_end":
                    session_end = timestamp

            # Stop once the EEG covers TAIL s past session_end
            if (
                session_end is not None
                and recorder.eeg_times
                and recorder.eeg_times[-1] >= session_end + TAIL
            ):
                print("Session ended.")
                break

            if time.monotonic() - last_status >= STATUS_EVERY:
                print(f"{recorder.n_samples} samples")
                last_status = time.monotonic()

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Stopping on Ctrl+C...")
    finally:
        recorder.save(path)
        print(
            f"Saved {path} with {recorder.n_samples} samples and "
            f"{len(recorder.markers)} markers"
        )


if __name__ == "__main__":
    main()
