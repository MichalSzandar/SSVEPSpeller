"""Stimulus logic shared by the calibration and speller GUIs (no pyglet imports here).

Flicker uses the sampled-sinusoid method (Chen 2015, Wang 2017): on refresh i after
stimulus onset, target k is drawn with luminance

    L_k(i) = 0.5 * (1 + sin(2*pi * f_k * i / R + phi_k))      R = monitor refresh rate

so any frequency below R/2 can be shown, not only divisors of R.

The frame index i is taken from the flip timestamps, not from a frame counter, so a
dropped frame skips one sample instead of shifting the phase of the rest of the trial.
"""

import numpy as np


def luminance(
    freqs: list[float], phases: list[float], frames, refresh_rate: float
) -> np.ndarray:
    """Luminance in [0, 1] of every target on the given frames -> (n_targets, n_frames).

    freqs, phases : one per target (Hz, rad)
    frames        : frame index after onset (int) or indices (n_frames,)
    """
    freqs = np.asarray(freqs, dtype=float)[:, None]
    phases = np.asarray(phases, dtype=float)[:, None]
    t = np.atleast_1d(frames) / refresh_rate
    return 0.5 * (1.0 + np.sin(2 * np.pi * freqs * t + phases))


def gray_levels(lum: np.ndarray) -> np.ndarray:
    """Luminance in [0, 1] -> 8-bit gray levels 0..255 (no gamma correction)."""
    return np.round(255 * np.clip(lum, 0.0, 1.0)).astype(int)


def next_frame_index(last_flip: float, onset: float, refresh_rate: float) -> int:
    """Index of the frame shown on the refresh after ``last_flip`` (frame 0 at ``onset``)."""
    return int(round((last_flip - onset) * refresh_rate)) + 1


def grid_layout(
    n_targets: int, width: float, height: float, fill: float = 0.6
) -> tuple[np.ndarray, float]:
    """Centres of square targets on a near-square grid -> (centres (n_targets, 2), size).

    Rows are filled from the top, and a shorter last row is centred. Coordinates are
    pixels with the origin at the bottom left (pyglet convention). ``fill`` is the
    fraction of a grid cell covered by its square.
    """
    n_cols = int(np.ceil(np.sqrt(n_targets)))
    n_rows = int(np.ceil(n_targets / n_cols))
    cell = min(width / n_cols, height / n_rows)
    size = fill * cell

    centres = []
    for k in range(n_targets):
        row, col = divmod(k, n_cols)
        n_in_row = min(n_cols, n_targets - row * n_cols)
        x = width / 2 + (col - (n_in_row - 1) / 2) * cell
        y = height / 2 + ((n_rows - 1) / 2 - row) * cell
        centres.append((x, y))
    return np.array(centres), size


def trial_sequence(
    n_targets: int, n_blocks: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration order -> (targets, blocks), each (n_blocks * n_targets,).

    Every block shows each target exactly once, in a new random order.
    """
    targets = np.concatenate([rng.permutation(n_targets) for _ in range(n_blocks)])
    blocks = np.repeat(np.arange(n_blocks), n_targets)
    return targets, blocks


def dropped_frames(flip_times: np.ndarray, refresh_rate: float) -> np.ndarray:
    """Missed refreshes before each flip -> (n_flips - 1,) ints, 0 = flip on time."""
    intervals = np.diff(flip_times)
    return np.maximum(np.round(intervals * refresh_rate).astype(int) - 1, 0)


def timing_report(flip_times: np.ndarray, refresh_rate: float) -> dict:
    """Summary of a flip log: duration, measured refresh rate, interval stats, drops."""
    intervals = np.diff(flip_times)
    dropped = dropped_frames(flip_times, refresh_rate)
    return {
        "n_flips": len(flip_times),
        "duration_s": float(flip_times[-1] - flip_times[0]),
        "measured_rate_hz": float(1.0 / np.median(intervals)),
        "mean_interval_ms": float(1000 * np.mean(intervals)),
        "std_interval_ms": float(1000 * np.std(intervals)),
        "max_interval_ms": float(1000 * np.max(intervals)),
        "n_dropped": int(np.sum(dropped)),
        "n_late_flips": int(np.count_nonzero(dropped)),
    }
