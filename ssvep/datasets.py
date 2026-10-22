"""Loading data into one common format.

Every loader returns the same triple:

    X      : (n_trials, n_channels, n_samples)  raw epochs in µV, *including* the
             pre-stimulus part (crop after filtering, see preprocessing.py)
    y      : (n_trials,)  target index 0..n_targets-1 (index into `freqs`)
    blocks : (n_trials,)  block/run index, used for leave-one-block-out CV
"""

from pathlib import Path

import numpy as np

from ssvep import config


def benchmark_target_indices(freqs: list[float]) -> np.ndarray:
    """Return the Benchmark target index (0..39) of each frequency in `freqs`.

    Raises ValueError if a frequency is not in the dataset.
    """
    raise NotImplementedError


def benchmark_channel_indices(names: list[str]) -> np.ndarray:
    """Return the row index of each channel name, read from ``64-channels.loc``."""
    raise NotImplementedError


def load_benchmark(
    subject: int,
    freqs: list[float] | None = None,
    channels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``S{subject}.mat`` from ``config.BENCHMARK_DIR``.

    Keeps only the targets in ``freqs`` (default ``config.TARGET_FREQS``) and the
    channels in ``channels`` (default ``config.BENCHMARK_CHANNELS``).
    Returns (X, y, blocks) as described at the top of this module.
    """
    raise NotImplementedError


def load_recording(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an own calibration recording as (X, y, blocks).

    Epochs are cut around the ``stim_on`` markers with the same pre-stimulus length
    as Benchmark (``config.BENCHMARK_ONSET``), so the same pipeline works on both.
    """
    raise NotImplementedError


def make_synthetic(
    freqs: list[float] | None = None,
    fs: float = 250,
    n_blocks: int = 6,
    n_channels: int = 8,
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
    raise NotImplementedError
