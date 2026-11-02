"""Tests for ssvep.acquisition."""

import json
import time
import uuid

import numpy as np
import pylsl
import pytest

from ssvep.acquisition import (
    Recorder,
    RingBuffer,
    channel_labels,
    find_stream,
    open_marker_inlet,
)


def unique_name():
    """A stream name no real device uses, so tests never pick up the cap or the GUI."""
    return f"pytest-{uuid.uuid4().hex[:8]}"


def labelled_info(name, labels):
    info = pylsl.StreamInfo(name, "EEG", len(labels), 250, pylsl.cf_float32, name)
    channels = info.desc().append_child("channels")
    for label in labels:
        channels.append_child("channel").append_child_value("label", label)
    return info


def chunk(start, n, n_channels=2, fs=250.0):
    """n samples whose value encodes (sample index, channel): 10 * index + channel."""
    index = np.arange(start, start + n)
    samples = 10.0 * index[:, None] + np.arange(n_channels)[None, :]
    return samples, index / fs


# --- find_stream / channel_labels / inlets --------------------------------------


def test_find_stream_times_out_with_timeout_error():
    """No stream of that type -> TimeoutError after about ``timeout`` seconds."""
    t0 = time.perf_counter()
    with pytest.raises(TimeoutError):
        find_stream("NoSuchType", name=unique_name(), timeout=0.3)
    assert time.perf_counter() - t0 < 3.0


def test_find_stream_finds_a_stream_by_type_and_name():
    """A stream published in this test is found by its type and name, and the result
    can open an inlet that connects."""
    name = unique_name()
    outlet = pylsl.StreamOutlet(labelled_info(name, ["O1", "Oz"]))
    info = find_stream("EEG", name=name, timeout=5.0)
    assert info.name() == name
    assert info.channel_count() == 2
    pylsl.StreamInlet(info).open_stream(timeout=5.0)  # TimeoutError if it can't
    del outlet


def test_channel_labels_reads_the_description():
    """Labels in the order of the stream description."""
    info = labelled_info(unique_name(), ["O1", "Oz", "O2"])
    assert channel_labels(info) == ["O1", "Oz", "O2"]


def test_channel_labels_without_description_is_empty():
    info = pylsl.StreamInfo(unique_name(), "EEG", 3, 250, pylsl.cf_float32, "x")
    assert channel_labels(info) == []


def test_marker_inlet_receives_markers_with_their_timestamps():
    """A marker pushed with an explicit LSL time arrives unchanged, with that time
    (same machine, clock-synced: within 10 ms)."""
    name = unique_name()
    info = pylsl.StreamInfo(
        name, "Markers", 1, pylsl.IRREGULAR_RATE, pylsl.cf_string, name
    )
    outlet = pylsl.StreamOutlet(info)
    inlet = open_marker_inlet(find_stream("Markers", name=name, timeout=5.0))
    assert isinstance(inlet, pylsl.StreamInlet)
    inlet.open_stream(timeout=5.0)
    time.sleep(0.2)
    marker = json.dumps({"event": "stim_on", "target": 2, "freq": 12.0})
    t_flip = pylsl.local_clock() - 0.05
    outlet.push_sample([marker], t_flip)
    sample, timestamp = inlet.pull_sample(timeout=5.0)
    assert sample == [marker]
    assert timestamp == pytest.approx(t_flip, abs=0.01)


# --- RingBuffer ----------------------------------------------------------------


def test_ring_buffer_starts_empty():
    buffer = RingBuffer(n_channels=2, capacity=5)
    data, times = buffer.get()
    assert len(buffer) == 0
    assert data.shape == (2, 0) and times.shape == (0,)


def test_ring_buffer_returns_channels_by_samples_in_time_order():
    """Input is (n, n_channels) like LSL; get() gives (n_channels, n), oldest first."""
    buffer = RingBuffer(n_channels=2, capacity=10)
    samples, times = chunk(0, 4)
    buffer.append(samples, times)
    data, got_times = buffer.get()
    assert len(buffer) == 4
    assert np.array_equal(data, samples.T)
    assert np.array_equal(got_times, times)


def test_ring_buffer_accepts_lists_like_pylsl():
    buffer = RingBuffer(n_channels=2, capacity=10)
    samples, times = chunk(0, 3)
    buffer.append(samples.tolist(), times.tolist())
    assert np.array_equal(buffer.get()[0], samples.T)


def test_ring_buffer_overwrites_the_oldest_samples():
    """Capacity 5, chunks of 3 and 4 samples: the last 5 samples remain, in order."""
    buffer = RingBuffer(n_channels=2, capacity=5)
    buffer.append(*chunk(0, 3))
    buffer.append(*chunk(3, 4))
    expected, expected_times = chunk(2, 5)
    data, times = buffer.get()
    assert len(buffer) == 5
    assert np.array_equal(data, expected.T)
    assert np.array_equal(times, expected_times)


def test_ring_buffer_many_small_chunks_wrap_around_correctly():
    buffer = RingBuffer(n_channels=2, capacity=7)
    for start in range(0, 40, 3):
        buffer.append(*chunk(start, 3))
    expected, expected_times = chunk(35, 7)
    data, times = buffer.get()
    assert np.array_equal(data, expected.T)
    assert np.array_equal(times, expected_times)


def test_ring_buffer_chunk_longer_than_capacity_keeps_its_end():
    buffer = RingBuffer(n_channels=2, capacity=5)
    buffer.append(*chunk(0, 12))
    expected, _ = chunk(7, 5)
    assert np.array_equal(buffer.get()[0], expected.T)


def test_ring_buffer_window_starts_at_first_sample_not_before_t_start():
    """t_start between samples -> starts at the next one; exactly on a sample -> that one."""
    buffer = RingBuffer(n_channels=2, capacity=50)
    samples, times = chunk(0, 30)
    buffer.append(samples, times)
    between = buffer.window(t_start=times[10] - 0.001, n_samples=5)
    exact = buffer.window(t_start=times[10], n_samples=5)
    assert between.shape == (2, 5)
    assert np.array_equal(between, samples[10:15].T)
    assert np.array_equal(exact, samples[10:15].T)


def test_ring_buffer_window_after_wraparound():
    buffer = RingBuffer(n_channels=2, capacity=8)
    for start in range(0, 30, 4):
        buffer.append(*chunk(start, 4))
    samples, times = chunk(0, 32)
    assert np.array_equal(buffer.window(times[26], 4), samples[26:30].T)


def test_ring_buffer_window_at_every_position_after_wraparound():
    """After wrapping, every window inside the stored range comes back whole and in
    order, including the windows that span the place where writing wrapped around."""
    buffer = RingBuffer(n_channels=2, capacity=8)
    for start in range(0, 30, 3):
        buffer.append(*chunk(start, 3))
    samples, times = chunk(0, 30)  # the buffer now holds samples 22..29
    for first in range(22, 27):
        window = buffer.window(times[first], 4)
        assert np.array_equal(window, samples[first : first + 4].T)


def test_ring_buffer_window_is_none_until_all_samples_arrived():
    buffer = RingBuffer(n_channels=2, capacity=50)
    samples, times = chunk(0, 12)
    buffer.append(samples, times)
    assert buffer.window(times[8], n_samples=5) is None
    assert buffer.window(times[8], n_samples=4) is not None
    buffer.append(*chunk(12, 1))
    assert np.array_equal(buffer.window(times[8], n_samples=5), chunk(8, 5)[0].T)


def test_ring_buffer_window_raises_when_the_start_was_overwritten():
    buffer = RingBuffer(n_channels=2, capacity=5)
    samples, times = chunk(0, 20)
    buffer.append(samples, times)
    with pytest.raises(ValueError):
        buffer.window(times[3], n_samples=2)


def test_ring_buffer_window_raises_before_the_first_sample_when_not_full():
    """A t_start earlier than the oldest stored sample also raises before the buffer
    is full, instead of silently starting the window later than asked."""
    buffer = RingBuffer(n_channels=2, capacity=50)
    samples, times = chunk(100, 20)
    buffer.append(samples, times)
    with pytest.raises(ValueError):
        buffer.window(times[0] - 0.1, n_samples=2)


def test_ring_buffer_returns_copies():
    """Changing what get() or window() returned does not change the buffer."""
    buffer = RingBuffer(n_channels=2, capacity=10)
    samples, times = chunk(0, 6)
    buffer.append(samples, times)
    buffer.get()[0][:] = -1
    buffer.window(times[0], 3)[:] = -1
    assert np.array_equal(buffer.get()[0], samples.T)


# --- Recorder ------------------------------------------------------------------


class FakeInlet:
    """Stands in for pylsl.StreamInlet: hands out prepared chunks, then empty ones."""

    def __init__(self, chunks):
        self.chunks = list(chunks)

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        return self.chunks.pop(0) if self.chunks else ([], [])


def eeg_chunk(start, n, n_channels=3):
    samples, times = chunk(start, n, n_channels)
    return samples.tolist(), (100.0 + times).tolist()


MARKERS = [
    json.dumps({"event": "stim_on", "target": 1}),
    json.dumps({"event": "stim_off"}),
]


def make_recorder():
    eeg = FakeInlet([eeg_chunk(0, 4), ([], []), eeg_chunk(4, 6)])
    markers = FakeInlet(
        [([], []), ([[MARKERS[0]]], [100.01]), ([[MARKERS[1]]], [100.03])]
    )
    return Recorder(eeg, markers, channels=["O1", "Oz", "O2"], fs=250.0)


def test_recorder_pull_returns_the_new_markers_and_counts_samples():
    recorder = make_recorder()
    assert recorder.pull() == [] and recorder.n_samples == 4
    assert recorder.pull() == [(MARKERS[0], 100.01)] and recorder.n_samples == 4
    assert recorder.pull() == [(MARKERS[1], 100.03)] and recorder.n_samples == 10
    assert recorder.pull() == [] and recorder.n_samples == 10


def test_recorder_save_writes_the_documented_file(tmp_path):
    """Keys, shapes and dtypes as in the module docstring; loads without allow_pickle."""
    recorder = make_recorder()
    for _ in range(4):
        recorder.pull()
    recorder.save(tmp_path / "rec.npz")
    with np.load(tmp_path / "rec.npz") as f:
        assert set(f.files) == {
            "eeg",
            "eeg_times",
            "markers",
            "marker_times",
            "channels",
            "fs",
        }
        samples, times = chunk(0, 10, n_channels=3)
        assert f["eeg"].dtype == np.float32 and f["eeg"].shape == (3, 10)
        assert np.array_equal(f["eeg"], samples.T.astype(np.float32))
        assert f["eeg_times"].dtype == np.float64
        assert np.allclose(f["eeg_times"], 100.0 + times)
        assert f["markers"].dtype.kind == "U" and f["markers"].tolist() == MARKERS
        assert f["marker_times"].dtype == np.float64
        assert np.allclose(f["marker_times"], [100.01, 100.03])
        assert f["channels"].tolist() == ["O1", "Oz", "O2"]
        assert float(f["fs"]) == 250.0


def test_recorder_without_markers_saves_an_empty_marker_list(tmp_path):
    recorder = Recorder(
        FakeInlet([eeg_chunk(0, 5)]), FakeInlet([]), ["O1", "Oz", "O2"], 250.0
    )
    recorder.pull()
    recorder.save(tmp_path / "rec.npz")
    with np.load(tmp_path / "rec.npz") as f:
        assert f["markers"].shape == (0,) and f["marker_times"].shape == (0,)
        assert f["eeg"].shape == (3, 5)
