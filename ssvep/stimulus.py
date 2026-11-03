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


def ring_layout(
    n_targets: int,
    width: float,
    height: float,
    slots: list[int] | None = None,
    fill: float = 0.6,
    margin: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Centres of square targets on a ring around the screen centre -> (centres
    (n_targets, 2), size).

    The middle of the screen stays free for the speller's typed text (Hex-o-Spell
    style), and every target is equally far from it. Slots are numbered clockwise,
    starting with the top-left one (with 6 targets: top left, top right, right,
    bottom right, bottom left, left). Target k is drawn in slot ``slots[k]`` (default:
    slot k). Coordinates are pixels with the origin at the bottom left (pyglet
    convention). ``fill`` is the fraction of the distance between neighbouring centres
    covered by a square, ``margin`` the free border as a fraction of the shorter side.
    """
    if n_targets < 2:
        raise ValueError("A ring needs at least 2 targets")
    if slots is None:
        slots = list(range(n_targets))
    if sorted(slots) != list(range(n_targets)):
        raise ValueError(f"slots must be a permutation of 0..{n_targets - 1}: {slots}")

    # Slot 0 is the first one clockwise from 12 o'clock, half a step past it
    step = 2 * np.pi / n_targets
    angles = np.pi / 2 + step / 2 - step * np.arange(n_targets)
    cos, sin = np.cos(angles), np.sin(angles)
    # Radius r of the largest ring whose squares (half side fill * r * sin(pi/n), from
    # the neighbour distance 2 * r * sin(pi/n)) stay inside the margins
    half_side = fill * np.sin(np.pi / n_targets)
    border = margin * min(width, height)
    r = min(
        (width / 2 - border) / (np.max(np.abs(cos)) + half_side),
        (height / 2 - border) / (np.max(np.abs(sin)) + half_side),
    )

    slot_centres = np.column_stack([width / 2 + r * cos, height / 2 + r * sin])
    return slot_centres[slots], 2 * half_side * r


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
