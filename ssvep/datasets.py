"""Loading data into one common format.

Every loader returns the same triple:

    X      : (n_trials, n_channels, n_samples)  raw epochs in µV, *including* the
             pre-stimulus part (crop after filtering, see preprocessing.py)
    y      : (n_trials,)  target index 0..n_targets-1 (index into `freqs`)
    blocks : (n_trials,)  block/run index, used for leave-one-block-out CV
"""

import json
from pathlib import Path
from scipy.io import loadmat

import numpy as np

from ssvep import config


def benchmark_target_indices(freqs: list[float]) -> np.ndarray:
    """Return the Benchmark target index (0..39) of each frequency in `freqs`.

    Raises ValueError if a frequency is not in the dataset.
    """
    mat = loadmat(config.BENCHMARK_DIR / "Freq_Phase.mat")
    result = []
    for f in freqs:
        idx = np.where(np.isclose(mat["freqs"][0], f))[0]
        if len(idx) == 0:
            raise ValueError(f"Frequency {f} not found in Benchmark dataset")
        result.append(idx[0])

    return np.array(result)


def benchmark_channel_indices(channels: list[str]) -> np.ndarray:
    """Return the Benchmark channel index (0..63) of each channel in `channels`.

    Raises ValueError if a channel is not in the dataset.
    """
    result = [0] * len(channels)
    with open(config.BENCHMARK_DIR / "64-channels.loc") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 0:
                continue
            for ch in channels:
                if parts[-1].lower() == ch.lower():
                    result[channels.index(ch)] = int(parts[0]) - 1

    if len(result) != len(channels):
        raise ValueError(f"Some channels not found in Benchmark dataset: {channels}")

    return np.array(result)


def load_benchmark(
    subject: int,
    freqs: list[float] = config.TARGET_FREQS,
    channels: list[str] = config.BENCHMARK_CHANNELS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``S{subject}.mat`` from ``config.BENCHMARK_DIR``.

    Keeps only the targets in ``freqs`` (default ``config.TARGET_FREQS``) and the
    channels in ``channels`` (default ``config.BENCHMARK_CHANNELS``).
    Returns (X, y, blocks) as described at the top of this module.
    """
    mat = loadmat(config.BENCHMARK_DIR / f"S{subject}.mat")
    data = mat["data"]  # (channels, samples, targets, blocks)

    channel_indices = benchmark_channel_indices(channels)
    target_indices = benchmark_target_indices(freqs)

    n_channels = len(channel_indices)
    n_targets = len(target_indices)
    n_samples = data.shape[1]
    n_blocks = data.shape[3]

    # One axis at a time: (64, 1500, 40, 6) -> (9, 1500, 40, 6) -> (9, 1500, 5, 6)
    data = data[channel_indices][:, :, target_indices, :]

    # (channels, samples, targets, blocks) -> (blocks, targets, channels, samples),
    # then merge the first two axes into one trials axis.
    X = data.transpose(3, 2, 0, 1).reshape(n_blocks * n_targets, n_channels, n_samples)

    y = np.tile(np.arange(n_targets), n_blocks)
    blocks = np.repeat(np.arange(n_blocks), n_targets)
    return X, y, blocks


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an own calibration recording as (X, y, blocks).

    ``path`` is a .npz written by ``acquisition.Recorder`` (format at the top of
    ssvep/acquisition.py). Epochs are cut around the ``stim_on`` markers with the same
    pre-stimulus length as Benchmark (``config.BENCHMARK_ONSET``), so the same
    pipeline works on both. Like a Benchmark epoch, each one is that pre-stimulus
    part, the flicker (``stim_s`` from ``session_start``) and as long again after it.

    Only the ``config.CAP_CHANNELS`` channels are kept, in that order (all channels
    while it is empty). Trials cut short by ESC (``stim_off`` with ``complete`` false)
    and trials that are not fully in the recording are dropped.
    """
    with np.load(path) as npz:
        eeg = npz["eeg"]  # (channels, samples)
        eeg_times = npz["eeg_times"]
        markers = npz["markers"]
        marker_times = npz["marker_times"]
        channels = [str(c) for c in npz["channels"]]
        fs = float(npz["fs"])

    # Every marker is one JSON string -> array of dicts, so a boolean mask works on it
    markers = np.array([json.loads(m) for m in markers])

    # Keep the occipital channels, in the order of config.CAP_CHANNELS
    if config.CAP_CHANNELS:
        missing = [c for c in config.CAP_CHANNELS if c not in channels]
        if missing:
            raise ValueError(f"{path} has no channels {missing}")
        eeg = eeg[[channels.index(c) for c in config.CAP_CHANNELS]]

    stim_s = config.STIM_DURATION  # unless session_start says otherwise
    for m in markers:
        if m["event"] == "session_start":
            stim_s = m["stim_s"]
    pre_samples = int(round(config.BENCHMARK_ONSET * fs))
    # Pre-stimulus part, flicker, and as long again after it
    n_samples = pre_samples + int(round(stim_s * fs)) + pre_samples

    # Find the stim_on markers of the trials that were not cut short.
    incomplete = [
        m["trial"] for m in markers if m["event"] == "stim_off" and not m["complete"]
    ]
    stim_on_mask = np.array(
        [m["event"] == "stim_on" and m["trial"] not in incomplete for m in markers],
        dtype=bool,
    )
    stim_on_markers = markers[stim_on_mask]
    stim_on_marker_times = marker_times[stim_on_mask]

    # Cut epochs around the stim_on markers.
    X_epochs, y, blocks = [], [], []
    for i in range(len(stim_on_markers)):
        # LSL time -> index of the first EEG sample at or after the onset flip
        sample_idx = np.searchsorted(eeg_times, stim_on_marker_times[i])
        epoch_start = sample_idx - pre_samples
        epoch_end = epoch_start + n_samples
        if epoch_start < 0 or epoch_end > eeg.shape[1]:
            continue  # the recording started late or stopped early
        X_epochs.append(eeg[:, epoch_start:epoch_end])  # (channels, samples)
        y.append(stim_on_markers[i]["target"])
        blocks.append(stim_on_markers[i]["block"])

    if not X_epochs:
        raise ValueError(f"No complete trials in {path}")
    X_epochs = np.stack(X_epochs, axis=0)  # (trials, channels, samples)
    return X_epochs, np.array(y), np.array(blocks)


def make_synthetic(
    freqs: list[float] = config.TARGET_FREQS,
    fs: float = config.BENCHMARK_FS,
    n_blocks: int = 6,
    n_channels: int = len(config.BENCHMARK_CHANNELS),
    duration: float = 2.0,
    pre: float = 0.5,
    snr_db: float = -10.0,
    n_harmonics: int = 3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic SSVEP epochs, same format as ``load_benchmark``.

    Each trial: ``pre`` seconds of noise, then ``duration`` seconds in which every
    channel carries a sum of harmonics ``sin(2*pi*h*f*t + phi)`` of the target
    frequency (random amplitude and phase per channel and harmonic), plus Gaussian
    noise scaled to ``snr_db``. ``freqs`` defaults to ``config.TARGET_FREQS``.
    """
    n_noise_samples = int(round(pre * fs))
    n_targets = len(freqs)
    n_samples = int(round(duration * fs))
    rng = np.random.default_rng(seed)
    t_vec = np.arange(n_samples) / fs

    phases = rng.uniform(0.0, 2 * np.pi, size=n_harmonics)
    # (blocks, targets, channels, samples)
    X = np.zeros(
        (n_blocks, n_targets, n_channels, n_samples + n_noise_samples), dtype=np.float32
    )
    for b in range(n_blocks):
        for t, f in enumerate(freqs):
            for c in range(n_channels):
                # Random amplitude per channel and harmonic.
                amps = rng.uniform(0.5, 1.0, size=n_harmonics)

                signal = sum(
                    amp * np.sin(2 * np.pi * (h + 1) * f * t_vec + phase)
                    for h, (amp, phase) in enumerate(zip(amps, phases))
                )

                noise_std = np.sqrt(np.mean(signal**2)) / (10 ** (snr_db / 20))
                noise = rng.normal(0.0, noise_std, size=n_samples)
                X[b, t, c, n_noise_samples:] = signal + noise

                noise = rng.normal(0.0, noise_std, size=n_noise_samples)
                X[b, t, c, :n_noise_samples] = noise

    X = X.reshape(n_blocks * n_targets, n_channels, n_samples + n_noise_samples)
    y = np.tile(np.arange(n_targets), n_blocks)
    blocks = np.repeat(np.arange(n_blocks), n_targets)
    return X, y, blocks
