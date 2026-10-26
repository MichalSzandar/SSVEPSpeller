"""Loading data into one common format.

Every loader returns the same triple:

    X      : (n_trials, n_channels, n_samples)  raw epochs in µV, *including* the
             pre-stimulus part (crop after filtering, see preprocessing.py)
    y      : (n_trials,)  target index 0..n_targets-1 (index into `freqs`)
    blocks : (n_trials,)  block/run index, used for leave-one-block-out CV
"""

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
    freqs: list[float] | None = config.TARGET_FREQS,
    channels: list[str] | None = config.BENCHMARK_CHANNELS,
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

    Epochs are cut around the ``stim_on`` markers with the same pre-stimulus length
    as Benchmark (``config.BENCHMARK_ONSET``), so the same pipeline works on both.
    """
    raise NotImplementedError


def make_synthetic(
    freqs: list[float] | None = config.TARGET_FREQS,
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

    n_targets = len(freqs)
    n_samples = int(round((pre + duration) * fs))
    rng = np.random.default_rng(seed)

    # (blocks, targets, channels, samples)
    X = np.zeros((n_blocks, n_targets, n_channels, n_samples), dtype=np.float32)
    for b in range(n_blocks):
        for t, f in enumerate(freqs):
            for c in range(n_channels):
                # Random amplitude and phase per channel and harmonic.
                amps = rng.uniform(0.5, 1.0, size=n_harmonics)
                phases = rng.uniform(0.0, 2 * np.pi, size=n_harmonics)
                t_vec = np.arange(n_samples) / fs
                signal = sum(
                    amp * np.sin(2 * np.pi * (h + 1) * f * t_vec + phase)
                    for h, (amp, phase) in enumerate(zip(amps, phases))
                )
                noise_std = np.sqrt(np.mean(signal**2)) / (10 ** (snr_db / 20))
                noise = rng.normal(0.0, noise_std, size=n_samples)
                X[b, t, c, :] = signal + noise

    X = X.reshape(n_blocks * n_targets, n_channels, n_samples)
    y = np.tile(np.arange(n_targets), n_blocks)
    blocks = np.repeat(np.arange(n_blocks), n_targets)
    return X, y, blocks
