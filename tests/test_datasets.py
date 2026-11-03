"""Tests for ssvep.datasets"""

import json

import pytest
import numpy as np

from ssvep import config
from ssvep.datasets import (
    benchmark_channel_indices,
    benchmark_target_indices,
    load_benchmark,
    load_recording,
    make_synthetic,
)

needs_benchmark = pytest.mark.skipif(
    not (config.BENCHMARK_DIR / "S1.mat").exists(),
    reason="Benchmark data not downloaded",
)


def test_make_synthetic_shapes():
    """5 targets x 6 blocks -> X (30, 8, 625) at 250 Hz with 0.5 s pre + 2 s; y and blocks have length 30."""
    X, y, blocks = make_synthetic(
        freqs=[8.0, 9.0, 10.0, 11.0, 12.0],
        fs=250,
        n_blocks=6,
        n_channels=8,
        duration=2.0,
    )
    assert X.shape == (30, 8, 625)
    assert y.shape == (30,)
    assert blocks.shape == (30,)
    assert np.all(y >= 0) and np.all(y < 5)
    assert np.all(blocks >= 0) and np.all(blocks < 6)


@needs_benchmark
def test_load_benchmark_shapes():
    """S1, config.TARGET_FREQS (6 targets), 9 channels -> X (36, 9, 1500), y in {0..5},
    blocks in {0..5}; every target appears once per block."""
    X, y, blocks = load_benchmark(1)
    n = 6 * len(config.TARGET_FREQS)
    assert X.shape == (n, 9, 1500)
    assert y.shape == (n,)
    assert blocks.shape == (n,)
    assert np.all(y >= 0) and np.all(y < len(config.TARGET_FREQS))
    assert np.all(blocks >= 0) and np.all(blocks < 6)
    for b in range(6):
        assert sorted(y[blocks == b].tolist()) == list(range(len(config.TARGET_FREQS)))


# --- load_recording ---------------------------------------------------------------------

FS = 250
T0 = 259178.0  # s, LSL time of the first sample: seconds since boot, far from 0
CHANNELS = ["O1", "Oz", "O2"]
STIM_S = 1.0  # s, not config.STIM_DURATION, so the epoch length must come from the file
PRE = 125  # samples, config.BENCHMARK_ONSET at 250 Hz
EPOCH = PRE + 250 + PRE  # 0.5 s before + 1 s flicker + 0.5 s after
ONSETS = [500, 1500, 2500, 3500]  # sample index of each stim_on
TARGETS = [1, 0, 0, 1]
BLOCKS = [0, 0, 1, 1]


def write_recording(
    path, n_samples=4500, onsets=ONSETS, incomplete=(), gap_at=None, channels=CHANNELS
):
    """Write a recording .npz in the Recorder format and return its EEG timestamps.

    Every sample holds where it came from, eeg[c, i] = 100000 * c + i, so a test can
    check exactly which samples an epoch contains. Each stim_on comes 1 ms before the
    sample at its onset index. ``gap_at``: the EEG jumps 0.4 s forward from that sample
    on (a dropout), so sample index and time no longer agree.
    """
    eeg = 100000.0 * np.arange(len(channels))[:, None] + np.arange(n_samples)
    eeg_times = T0 + np.arange(n_samples) / FS
    if gap_at is not None:
        eeg_times[gap_at:] += 0.4

    markers = []  # (time, JSON string)

    def add(t, event, **fields):
        markers.append((t, json.dumps({"event": event, **fields})))

    add(
        T0,
        "session_start",
        subject="test",
        freqs=[8.0, 10.0],
        phases=[0.0, 0.0],
        refresh_rate=120,
        n_blocks=2,
        cue_s=1.0,
        stim_s=STIM_S,
    )
    for trial, (onset, target, block) in enumerate(zip(onsets, TARGETS, BLOCKS)):
        t_on = eeg_times[onset] - 0.001
        info = dict(trial=trial, block=block, target=target, freq=[8.0, 10.0][target])
        if trial % 2 == 0:
            add(t_on - 1.5, "block_start", block=block)
        add(t_on - 1.0, "cue", **info)
        add(t_on, "stim_on", **info)
        add(
            t_on + STIM_S,
            "stim_off",
            **info,
            n_dropped=0,
            complete=trial not in incomplete,
        )
        if trial % 2 == 1:
            add(t_on + STIM_S + 0.01, "block_end", block=block)
    add(eeg_times[-1] - 0.5, "session_end", n_trials=len(onsets), aborted=False)
    markers.sort()

    np.savez_compressed(
        path,
        eeg=eeg.astype(np.float32),
        eeg_times=eeg_times,
        markers=np.array([m for _, m in markers], dtype=str),
        marker_times=np.array([t for t, _ in markers]),
        channels=channels,
        fs=float(FS),
    )
    return eeg_times


def expected_epoch(onset, channel):
    """The sample values of the epoch around ``onset`` on channel ``channel``."""
    return 100000.0 * channel + np.arange(onset - PRE, onset - PRE + EPOCH)


@pytest.fixture
def all_channels(monkeypatch):
    """Keep every channel of the file, whatever config.CAP_CHANNELS holds."""
    monkeypatch.setattr(config, "CAP_CHANNELS", [])


@pytest.mark.usefixtures("all_channels")
def test_load_recording_shapes_labels_blocks(tmp_path):
    """4 trials -> X (4, 3, 0.5 s + stim_s + 0.5 s); y = target, blocks = block of each stim_on."""
    write_recording(tmp_path / "rec.npz")
    X, y, blocks = load_recording(tmp_path / "rec.npz")
    assert X.shape == (4, 3, EPOCH)
    assert list(y) == TARGETS
    assert list(blocks) == BLOCKS
    assert np.issubdtype(y.dtype, np.integer) and np.issubdtype(
        blocks.dtype, np.integer
    )


@pytest.mark.usefixtures("all_channels")
def test_load_recording_epochs_start_0_5_s_before_stim_on(tmp_path):
    """Sample 0.5 s into each epoch is the first EEG sample at or after stim_on, on every channel."""
    write_recording(tmp_path / "rec.npz")
    X, _, _ = load_recording(tmp_path / "rec.npz")
    for k, onset in enumerate(ONSETS):
        for c in range(len(CHANNELS)):
            np.testing.assert_array_equal(X[k, c], expected_epoch(onset, c))


@pytest.mark.usefixtures("all_channels")
def test_load_recording_finds_onsets_by_timestamp(tmp_path):
    """A dropout in the EEG before a trial does not shift that trial's epoch."""
    write_recording(tmp_path / "rec.npz", gap_at=2000)
    X, _, _ = load_recording(tmp_path / "rec.npz")
    for k, onset in enumerate(ONSETS):
        np.testing.assert_array_equal(X[k, 0], expected_epoch(onset, 0))


@pytest.mark.usefixtures("all_channels")
def test_load_recording_drops_incomplete_trials(tmp_path):
    """A trial whose stim_off has complete=false is left out of X, y and blocks."""
    write_recording(tmp_path / "rec.npz", incomplete=(2,))
    X, y, blocks = load_recording(tmp_path / "rec.npz")
    assert X.shape[0] == 3
    assert list(y) == [1, 0, 1]
    assert list(blocks) == [0, 0, 1]
    np.testing.assert_array_equal(X[2, 0], expected_epoch(ONSETS[3], 0))


@pytest.mark.usefixtures("all_channels")
def test_load_recording_drops_trials_past_the_end(tmp_path):
    """A last trial whose epoch runs past the end of the EEG is dropped; the others are kept."""
    write_recording(tmp_path / "rec.npz", n_samples=ONSETS[3] + 300)
    X, y, _ = load_recording(tmp_path / "rec.npz")
    assert X.shape == (3, 3, EPOCH)
    assert list(y) == TARGETS[:3]


def test_load_recording_selects_cap_channels(tmp_path, monkeypatch):
    """With config.CAP_CHANNELS set, X holds only those channels, in that order."""
    monkeypatch.setattr(config, "CAP_CHANNELS", ["O2", "O1"])
    write_recording(tmp_path / "rec.npz")
    X, _, _ = load_recording(tmp_path / "rec.npz")
    assert X.shape == (4, 2, EPOCH)
    np.testing.assert_array_equal(X[0, 0], expected_epoch(ONSETS[0], 2))  # O2
    np.testing.assert_array_equal(X[0, 1], expected_epoch(ONSETS[0], 0))  # O1


def test_load_recording_missing_cap_channel_raises(tmp_path, monkeypatch):
    """A config.CAP_CHANNELS label that the recording lacks raises ValueError."""
    monkeypatch.setattr(config, "CAP_CHANNELS", ["O1", "Cz"])
    write_recording(tmp_path / "rec.npz")
    with pytest.raises(ValueError):
        load_recording(tmp_path / "rec.npz")
