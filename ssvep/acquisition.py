"""LSL acquisition: stream lookup, EEG and marker inlets, a ring buffer, the recorder.

Used by scripts/record.py (calibration session -> .npz), scripts/fake_eeg_stream.py
(reads the markers) and later by the online backend (ring buffer).

Recording file (.npz), written by Recorder.save and read by datasets.load_recording:

    eeg           (n_channels, n_samples) float32, µV, continuous
    eeg_times     (n_samples,) float64, LSL time of every sample
    markers       (n_markers,) str, the JSON marker strings as received
    marker_times  (n_markers,) float64, LSL time of every marker
    channels      (n_channels,) str, channel labels
    fs            float, nominal sampling rate in Hz

Both inlets use clock synchronisation, so eeg_times and marker_times are on the
recorder's clock and can be compared directly.
"""

from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamInlet, resolve_byprop, proc_clocksync, proc_dejitter


def find_stream(
    stream_type: str, name: str | None = None, timeout: float = 5.0
) -> StreamInfo:
    """The first stream of type ``stream_type`` (and name ``name``, if given).

    Raises TimeoutError if no such stream appears within ``timeout`` seconds.
    """
    if name is None:
        stream = resolve_byprop("type", stream_type, timeout=timeout)
    else:
        stream = resolve_byprop("name", name, timeout=timeout)

    if not stream:
        raise TimeoutError(f"No {stream_type} stream named {name} found in {timeout}s")

    return stream[0]


def channel_labels(info: StreamInfo) -> list[str]:
    """Channel labels from the stream description; [] if it has none."""
    xml = info.as_xml()
    labels = []
    for line in xml.splitlines():
        if "<label>" in line:
            label = line.strip().removeprefix("<label>").removesuffix("</label>")
            labels.append(label)
    return labels


def open_eeg_inlet(info: StreamInfo) -> StreamInlet:
    """Inlet for a regular-rate stream: clock synchronisation and dejittering."""
    inlet = StreamInlet(
        info,
        max_buflen=3600,
        processing_flags=proc_clocksync | proc_dejitter,
        recover=True,
    )
    inlet.open_stream(timeout=5.0)
    inlet.time_correction(timeout=5.0)
    return inlet


def open_marker_inlet(info: StreamInfo) -> StreamInlet:
    """Inlet for an irregular-rate marker stream: clock synchronisation only."""
    inlet = StreamInlet(
        info, max_buflen=3600, processing_flags=proc_clocksync, recover=True
    )
    inlet.open_stream(timeout=5.0)
    inlet.time_correction(timeout=5.0)
    return inlet


class RingBuffer:
    """The most recent ``capacity`` samples of a stream, with their timestamps.

    The online backend appends every chunk it pulls and cuts trial windows by time.
    """

    def __init__(self, n_channels: int, capacity: int):
        self.n_channels = n_channels
        self.capacity = capacity
        self.data = np.zeros((capacity, n_channels), dtype=np.float32)
        self.times = np.zeros(capacity, dtype=np.float64)
        self.count = 0  # number of samples stored, at most capacity
        self.index = 0  # index of the next sample to overwrite, 0..capacity-1

    def __len__(self) -> int:
        """Number of samples stored, at most ``capacity``."""
        return self.count

    def append(self, samples, timestamps) -> None:
        """Add a chunk, oldest sample first: samples (n, n_channels), timestamps (n,).

        Accepts lists (as pylsl returns them) or arrays. When the buffer is full the
        oldest samples are overwritten; a chunk longer than ``capacity`` leaves only
        its last ``capacity`` samples.
        """
        if len(samples) == 0:
            return

        if len(samples) > self.capacity:
            samples = samples[-self.capacity :]
            timestamps = timestamps[-self.capacity :]

        end = self.index + len(samples)
        if end > self.capacity:
            # Wrap around: overwrite the oldest samples first
            n_wrap = end - self.capacity
            self.data[self.index : self.capacity, :] = samples[:-n_wrap]
            self.times[self.index : self.capacity] = timestamps[:-n_wrap]
            self.data[0:n_wrap, :] = samples[-n_wrap:]
            self.times[0:n_wrap] = timestamps[-n_wrap:]
        else:
            self.data[self.index : self.index + len(samples), :] = samples
            self.times[self.index : self.index + len(samples)] = timestamps

        self.index = (self.index + len(samples)) % self.capacity
        self.count = min(self.count + len(samples), self.capacity)

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        """Copy of everything stored, oldest first: data (n_channels, n), times (n,)."""

        if self.count < self.capacity:
            data = self.data[: self.count, :].copy()
            times = self.times[: self.count].copy()
        else:
            data = np.concatenate(
                (self.data[self.index :, :], self.data[: self.index, :]), axis=0
            )
            times = np.concatenate(
                (self.times[self.index :], self.times[: self.index]), axis=0
            )

        return data.T, times

    def window(self, t_start: float, n_samples: int) -> np.ndarray | None:
        """Copy of ``n_samples`` samples from the first one with timestamp >= ``t_start``,
        as (n_channels, n_samples).

        Returns None while those samples have not all arrived. Raises ValueError if
        ``t_start`` is before the oldest stored sample (that data is gone).
        """
        if self.count == 0:
            return None

        data, times = self.get()  # oldest first, already a copy
        if t_start < times[0]:
            raise ValueError(
                f"t_start {t_start} is before the oldest sample {times[0]}"
            )

        start_index = np.searchsorted(times, t_start, side="left")
        if start_index + n_samples > len(times):
            return None
        return data[:, start_index : start_index + n_samples]


class Recorder:
    """Keeps everything an EEG inlet and a marker inlet deliver, and saves it.

    The inlets only need ``pull_chunk(timeout=...)``, so tests can pass stand-ins.
    """

    def __init__(
        self,
        eeg_inlet: StreamInlet,
        marker_inlet: StreamInlet,
        channels: list[str],
        fs: float,
    ):
        self.eeg_inlet = eeg_inlet
        self.marker_inlet = marker_inlet
        self.channels = channels
        self.fs = fs
        self.eeg_data = []
        self.eeg_times = []
        self.markers = []
        self.marker_times = []

    @property
    def n_samples(self) -> int:
        """EEG samples recorded so far."""
        return len(self.eeg_data)

    def pull(self) -> list[tuple[str, float]]:
        """Fetch whatever both inlets have, without waiting.

        Returns the markers that arrived in this call as (marker, timestamp) pairs, so
        the caller can react to them (e.g. stop after ``session_end``).
        """
        marker_samples, timestamp = self.marker_inlet.pull_chunk(timeout=0.0)
        # Each marker sample is a one-channel list, so take its only string
        marker = [sample[0] for sample in marker_samples]
        self.markers.extend(marker)
        self.marker_times.extend(timestamp)

        eeg, eeg_times = self.eeg_inlet.pull_chunk(timeout=0.0)
        self.eeg_data.extend(eeg)
        self.eeg_times.extend(eeg_times)

        return list(zip(marker, timestamp))

    def save(self, path: Path) -> None:
        """Write the recording file described at the top of this module."""
        eeg = np.array(self.eeg_data, dtype=np.float32).T
        eeg_times = np.array(self.eeg_times, dtype=np.float64)
        markers = np.array(self.markers, dtype=str)
        marker_times = np.array(self.marker_times, dtype=np.float64)
        np.savez_compressed(
            path,
            eeg=eeg,
            eeg_times=eeg_times,
            markers=markers,
            marker_times=marker_times,
            channels=self.channels,
            fs=self.fs,
        )
