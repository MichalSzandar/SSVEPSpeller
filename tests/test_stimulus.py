"""Tests for ssvep.stimulus."""

import itertools

import numpy as np
import pytest

from ssvep.stimulus import (
    dropped_frames,
    gray_levels,
    grid_layout,
    luminance,
    next_frame_index,
    timing_report,
)

R = 120.0
PERIOD = 1.0 / R


def test_luminance_shape_and_range():
    """(n_targets, n_frames), values in [0, 1]; a single int frame gives one column."""
    freqs = [8.0, 10.0, 12.0, 14.0, 15.0]
    phases = [0.0] * 5

    lum = luminance(freqs, phases, np.arange(600), R)
    assert lum.shape == (5, 600)
    assert lum.min() >= 0.0 and lum.max() <= 1.0
    assert luminance(freqs, phases, 0, R).shape == (5, 1)


def test_luminance_phase_at_onset():
    """Frame 0 shows 0.5 * (1 + sin(phase))."""
    lum = luminance([10.0, 10.0, 10.0], [0.0, np.pi / 2, -np.pi / 2], 0, R)[:, 0]
    assert np.allclose(lum, [0.5, 1.0, 0.0])


def test_luminance_period_in_frames():
    """10 Hz at 120 Hz repeats every 12 frames, 15 Hz every 8."""
    lum = luminance([10.0, 15.0], [0.3, 1.1], np.arange(48), R)
    assert np.allclose(lum[0, :36], lum[0, 12:])
    assert np.allclose(lum[1, :40], lum[1, 8:])


@pytest.mark.parametrize("freq", [8.0, 9.4, 10.0, 12.0, 13.6, 14.0, 15.0])
def test_luminance_spectrum_peaks_at_target_frequency(freq):
    """The sampled sinusoid has its largest non-DC spectral peak at the target frequency,
    also for frequencies that are not divisors of the refresh rate."""
    frames = np.arange(int(5 * R))  # 5 s -> 0.2 Hz bins, all frequencies on a bin
    lum = luminance([freq], [0.0], frames, R)[0]
    spectrum = np.abs(np.fft.rfft(lum - lum.mean()))
    bins = np.fft.rfftfreq(len(frames), d=1 / R)
    assert bins[np.argmax(spectrum)] == pytest.approx(freq)


def test_gray_levels():
    """0 -> 0, 1 -> 255, 0.5 -> 128 (rounded), out-of-range values clipped, ints."""
    levels = gray_levels(np.array([0.0, 0.5, 1.0, -0.1, 1.2]))
    assert levels.tolist() == [0, 128, 255, 0, 255]
    assert np.issubdtype(levels.dtype, np.integer)


def test_next_frame_index_on_time_and_jitter():
    """Flips on time give consecutive frames; jitter below half a frame changes nothing."""
    onset = 1000.0
    assert next_frame_index(onset, onset, R) == 1
    assert next_frame_index(onset + 5 * PERIOD, onset, R) == 6
    assert next_frame_index(onset + 5 * PERIOD + 0.003, onset, R) == 6
    assert next_frame_index(onset + 5 * PERIOD - 0.003, onset, R) == 6


def test_next_frame_index_keeps_phase_after_dropped_frame():
    """Simulated flip loop with one dropped refresh: after the drop, the frame shown at
    every flip matches the wall-clock frame, so the flicker phase is not shifted."""
    rng = np.random.default_rng(0)
    refreshes = np.r_[np.arange(0, 50), np.arange(51, 100)]  # refresh 50 missed
    flip_times = 1000.0 + refreshes * PERIOD + rng.uniform(0, 0.002, len(refreshes))

    shown = [0]
    for t in flip_times[:-1]:
        shown.append(next_frame_index(t, flip_times[0], R))

    expected = refreshes.copy()
    expected[50] = 50  # drawn for refresh 50, but displayed one refresh late
    assert shown == expected.tolist()


def test_dropped_frames():
    """0 for on-time flips (with jitter); n - 1 for an interval of n refresh periods."""
    rng = np.random.default_rng(1)
    refreshes = np.r_[np.arange(0, 10), 11, 12, 15, 16]  # 1 and 2 refreshes missed
    flip_times = refreshes * PERIOD + rng.uniform(-0.001, 0.001, len(refreshes))

    dropped = dropped_frames(flip_times, R)
    assert dropped.shape == (len(flip_times) - 1,)
    assert dropped.tolist() == [0] * 9 + [1, 0, 2, 0]


def test_timing_report():
    """Measured rate, drop counts and interval stats of a log with one missed refresh."""
    refreshes = np.r_[np.arange(0, 120), np.arange(121, 241)]
    flip_times = 50.0 + refreshes * PERIOD

    report = timing_report(flip_times, R)
    assert report["n_flips"] == 240
    assert report["duration_s"] == pytest.approx(240 * PERIOD)
    assert report["measured_rate_hz"] == pytest.approx(R)
    assert report["mean_interval_ms"] == pytest.approx(1000 * 240 / 239 * PERIOD)
    assert report["max_interval_ms"] == pytest.approx(2000 * PERIOD)
    assert report["n_dropped"] == 1
    assert report["n_late_flips"] == 1


@pytest.mark.parametrize(
    "n_targets, width, height",
    [(n, 1920, 1200) for n in range(1, 10)] + [(5, 800, 600), (6, 1200, 1920)],
)
def test_grid_layout_fits_without_overlap(n_targets, width, height):
    """n centres; every square lies inside the window, squares don't overlap, and the
    whole grid is centred on the screen."""
    centres, size = grid_layout(n_targets, width, height)
    assert centres.shape == (n_targets, 2)
    assert size > 0

    x, y = centres[:, 0], centres[:, 1]
    assert np.all(x - size / 2 >= 0) and np.all(x + size / 2 <= width)
    assert np.all(y - size / 2 >= 0) and np.all(y + size / 2 <= height)
    for a, b in itertools.combinations(centres, 2):
        assert max(abs(a - b)) >= size
    assert (x.min() + x.max()) / 2 == pytest.approx(width / 2)
    assert (y.min() + y.max()) / 2 == pytest.approx(height / 2)


def test_grid_layout_rows_from_top():
    """5 targets: 3 on the top row, 2 centred below; target 0 is top left."""
    centres, _ = grid_layout(5, 1920, 1200)
    top, bottom = centres[:3], centres[3:]
    assert np.allclose(top[:, 1], top[0, 1]) and np.allclose(bottom[:, 1], bottom[0, 1])
    assert top[0, 1] > bottom[0, 1]
    assert np.all(np.diff(top[:, 0]) > 0) and np.all(np.diff(bottom[:, 0]) > 0)
    assert bottom[:, 0].mean() == pytest.approx(960)
