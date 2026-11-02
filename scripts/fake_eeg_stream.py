"""Synthetic EEG on LSL that answers the calibration markers with an SSVEP.

Usage:
    .venv/bin/python scripts/fake_eeg_stream.py
    .venv/bin/python scripts/fake_eeg_stream.py --snr -15 --latency 0.14

Publishes a regular-rate stream (type EEG, name FakeEEG, config.EEG_FS, labels
config.CAP_CHANNELS) of white Gaussian noise (NOISE_STD µV). While a calibration trial
flickers, every channel also carries the cued frequency and its harmonics, from
``latency`` s after the stim_on flip until ``latency`` s after stim_off, with the same
phase relative to stim_on in every trial (time-locked, like a real SSVEP). The recorder and the online
backend cannot tell it from the cap, so the pipeline can be tested without hardware.
Without a marker stream it sends noise only. Ctrl+C stops it.
"""

import argparse
import json
import time

import numpy as np
import pylsl

from ssvep import acquisition, config

LAG = 0.1  # s, every sample is made this long after its time, so markers arrive first
CHUNK = 0.02  # s, how often samples are pushed
NOISE_STD = 10.0  # µV


class FakeEEG:
    """Sample generator: white Gaussian noise plus an SSVEP during cued flicker.

    ``snr_db`` is the power of the SSVEP relative to the noise on an average channel.
    The per-channel gains are drawn once from ``seed``, so they stay fixed for a
    session; two generators with the same seed and the same calls give the same output.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        snr_db: float = -10.0,
        latency: float = config.BENCHMARK_LATENCY,
        n_harmonics: int = 3,
        seed: int = 0,
    ):
        self.n_channels = n_channels
        self.fs = fs
        self.snr_db = snr_db
        self.latency = latency
        self.n_harmonics = n_harmonics
        self.rng = np.random.default_rng(seed)
        self.gains = self.rng.uniform(0.5, 1.5, size=(n_channels, n_harmonics))
        # Amplitude A with mean power over channels A²/2 * sum_h gains² = noise * SNR
        signal_power = NOISE_STD**2 * 10 ** (snr_db / 10)
        self.amplitude = np.sqrt(2 * signal_power / np.mean(np.sum(self.gains**2, 1)))
        self.phases = np.zeros(0)  # rad, one per target, from session_start
        self.stim_on = None  # s, LSL time of the last stim_on marker
        self.stim_off = None  # s, LSL time of the last stim_off marker
        self.freq = None  # Hz, of the last stim_on marker
        self.phase = 0.0  # rad, of the last stim_on marker's target

    def on_marker(self, marker: str, timestamp: float) -> None:
        """Update the stimulus state from one JSON marker string.

        stim_on starts the response at its ``freq`` from ``timestamp + latency``;
        stim_off ends it at ``timestamp + latency``. session_start's ``phases`` (one per
        target, rad) set each target's starting phase (default 0). Other events are
        ignored.
        """
        marker_data = json.loads(marker)
        if marker_data["event"] == "stim_on":
            self.stim_on = timestamp + self.latency
            self.stim_off = None  # the previous trial's end
            self.freq = marker_data["freq"]
            target = marker_data["target"]
            self.phase = self.phases[target] if target < len(self.phases) else 0.0
        elif marker_data["event"] == "stim_off":
            self.stim_off = timestamp + self.latency
        elif marker_data["event"] == "session_start":
            self.phases = np.array(marker_data["phases"])

    def generate(self, times: np.ndarray) -> np.ndarray:
        """Samples for the increasing LSL times ``times`` (n,) -> (n, n_channels), µV."""
        # White Gaussian noise
        samples = self.rng.normal(0.0, NOISE_STD, size=(len(times), self.n_channels))
        # Add the SSVEP if a trial is ongoing
        if self.stim_on is not None:
            # No stim_off yet: the target is still flickering
            stim_off = np.inf if self.stim_off is None else self.stim_off
            # Only add the SSVEP to the times that are within the trial
            trial_mask = (times >= self.stim_on) & (times < stim_off)
            if np.any(trial_mask):
                trial_times = times[trial_mask] - self.stim_on  # s, relative to stim_on
                stimulus_phase = 2 * np.pi * self.freq * trial_times + self.phase
                for h in range(self.n_harmonics):
                    # Harmonic h + 1 of sin(stimulus_phase)
                    wave = np.sin((h + 1) * stimulus_phase)
                    samples[trial_mask] += (
                        self.amplitude * self.gains[:, h] * wave[:, None]
                    )
        return samples


def open_eeg_outlet(labels: list[str], fs: float, name: str) -> pylsl.StreamOutlet:
    """Regular-rate float32 EEG outlet with the channel labels in its description."""
    # The name doubles as source id, so inlets reconnect when the script restarts
    info = pylsl.StreamInfo(name, "EEG", len(labels), fs, pylsl.cf_float32, name)
    channels = info.desc().append_child("channels")
    for label in labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", "microvolts")
        channel.append_child_value("type", "EEG")
    return pylsl.StreamOutlet(info)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snr", type=float, default=-10.0, help="SSVEP SNR in dB")
    parser.add_argument("--latency", type=float, default=config.BENCHMARK_LATENCY)
    parser.add_argument("--name", default="FakeEEG", help="LSL stream name")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    labels = config.CAP_CHANNELS
    if not labels:
        raise SystemExit("config.CAP_CHANNELS is empty: fill in the cap's labels first")
    fs = config.EEG_FS
    fake = FakeEEG(len(labels), fs, args.snr, args.latency, seed=args.seed)
    outlet = open_eeg_outlet(labels, fs, args.name)
    # Looks for the markers in the background, so calibration can start at any time
    resolver = pylsl.ContinuousResolver(prop="name", value=config.MARKER_STREAM)
    marker_inlet = None
    print(f"Sending {args.name}: {len(labels)} channels at {fs} Hz, SNR {args.snr} dB")
    print(f"Noise only until marker stream {config.MARKER_STREAM} appears")

    start = pylsl.local_clock()
    n_sent = 0
    try:
        while True:
            if marker_inlet is None:
                found = resolver.results()
                if found:
                    marker_inlet = acquisition.open_marker_inlet(found[0])
                    print(f"Connected to marker stream {config.MARKER_STREAM}")
            if marker_inlet is not None:
                markers, timestamps = marker_inlet.pull_chunk(timeout=0.0)
                for sample, timestamp in zip(markers, timestamps):
                    fake.on_marker(sample[0], timestamp)

            # Sample i belongs to start + i / fs and is sent once it is LAG s old
            n_due = int((pylsl.local_clock() - LAG - start) * fs) - n_sent
            if n_due > 0:
                times = start + (n_sent + np.arange(n_due)) / fs
                # One timestamp (the newest sample's); LSL derives the rest from fs
                outlet.push_chunk(fake.generate(times), times[-1])
                n_sent += n_due

            time.sleep(CHUNK)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
