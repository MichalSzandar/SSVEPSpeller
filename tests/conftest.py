import json

import numpy as np
import pytest

from ssvep import config
from ssvep.datasets import make_synthetic


@pytest.fixture
def synthetic_easy():
    """(X, y, blocks): high-SNR synthetic SSVEP, 250 Hz, config.TARGET_FREQS."""
    return make_synthetic(snr_db=0.0, seed=0)


@pytest.fixture
def synthetic_noise_only():
    """(X, y, blocks) where the signal is buried in noise: expect ~chance accuracy."""
    return make_synthetic(snr_db=-40.0, seed=1)


def write_ssvep_recording(
    path,
    n_blocks=3,
    stim=5.0,
    delay=config.BENCHMARK_LATENCY,
    seed=0,
    targets=None,
    marker_targets=None,
):
    """Write a recording in the Recorder format (top of ssvep/acquisition.py) of a
    synthetic calibration or online session -> the targets of its trials, in order.

    The CAP channels at config.EEG_FS: white noise (sd 1) and, from ``delay`` s after
    each stim_on marker for ``stim`` s, the target's sine and its 2nd harmonic,
    time-locked to that moment, with a fixed gain per channel (one head). ``delay``
    plays the visual latency plus any lead of the EEG timestamps over the markers.
    Each trial: 2 s cue, ``stim`` s flicker, 1 s after it. ``targets`` defaults to
    every target once per block in a random order; the markers carry
    ``marker_targets`` if given (a ground truth that differs from the EEG).
    """
    fs = config.EEG_FS
    freqs = config.TARGET_FREQS
    rng = np.random.default_rng(seed)
    gains = np.random.default_rng(12345).uniform(0.5, 1.5, len(config.CAP_CHANNELS))
    if targets is None:
        targets = np.concatenate([rng.permutation(len(freqs)) for _ in range(n_blocks)])
    targets = [int(t) for t in targets]
    blocks = np.arange(len(targets)) // len(freqs)
    logged = targets if marker_targets is None else [int(t) for t in marker_targets]

    t0 = 1000.0  # s, LSL time of the first sample
    trial_s = 2.0 + stim + 1.0
    n = int(round((1.0 + len(targets) * trial_s + 1.0) * fs))
    times = t0 + np.arange(n) / fs
    eeg = rng.normal(0.0, 1.0, (len(config.CAP_CHANNELS), n))
    markers = [
        (
            t0,
            {
                "event": "session_start",
                "freqs": freqs,
                "phases": [0.0] * len(freqs),
                "stim_s": stim,
                "cue_s": 2.0,
            },
        )
    ]
    for trial, target in enumerate(targets):
        onset = t0 + 1.0 + trial * trial_s + 2.0 - 0.001  # a flip between two samples
        info = {
            "trial": trial,
            "block": int(blocks[trial]),
            "target": logged[trial],
            "freq": freqs[logged[trial]],
        }
        markers.append((onset - 2.0, {"event": "cue", **info}))
        markers.append((onset, {"event": "stim_on", **info}))
        markers.append(
            (
                onset + stim,
                {"event": "stim_off", **info, "n_dropped": 0, "complete": True},
            )
        )
        since = times - onset - delay
        on = (since >= 0) & (since < stim)
        f = freqs[target]
        wave = np.sin(2 * np.pi * f * since[on]) + 0.5 * np.sin(
            4 * np.pi * f * since[on]
        )
        eeg[:, on] += gains[:, None] * wave
    markers.append((times[-1] - 0.05, {"event": "session_end"}))
    np.savez_compressed(
        path,
        eeg=eeg.astype(np.float32),
        eeg_times=times,
        markers=np.array([json.dumps(m) for _, m in markers], dtype=str),
        marker_times=np.array([t for t, _ in markers]),
        channels=np.array(config.CAP_CHANNELS),
        fs=float(fs),
    )
    return targets


@pytest.fixture
def ssvep_recording():
    """The function write_ssvep_recording(path, ...): a synthetic recording file."""
    return write_ssvep_recording
