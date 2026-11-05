"""Tests for ssvep.stimulus."""

import itertools

import numpy as np
import pytest

from ssvep import config
from ssvep.stimulus import (
    centre_box,
    dropped_frames,
    gray_levels,
    luminance,
    refresh_indices,
    refreshes_between,
    ring_layout,
    timing_problems,
    timing_report,
    trial_sequence,
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


def test_refreshes_between():
    """Whole refresh periods from one flip to the next, rounded: jitter below half a
    period changes nothing, and two flips are always at least one refresh apart."""
    t = 1000.0
    assert refreshes_between(t, t + PERIOD, R) == 1
    assert refreshes_between(t, t + PERIOD + 0.003, R) == 1
    assert refreshes_between(t, t + PERIOD - 0.003, R) == 1
    assert refreshes_between(t, t + 2 * PERIOD, R) == 2
    assert refreshes_between(t, t + 3 * PERIOD - 0.002, R) == 3
    assert refreshes_between(t, t + 0.1 * PERIOD, R) == 1


def test_refresh_indices():
    """0 for the first flip, then the running sum of refreshes_between; this recovers the
    refreshes of a jittery log with missed refreshes. Empty and one-flip logs work."""
    rng = np.random.default_rng(2)
    refreshes = np.r_[0:20, 22:30, 31:40]  # refreshes 20, 21 and 30 missed
    flip_times = 5.0 + refreshes * PERIOD + rng.normal(0, 0.0005, len(refreshes))
    steps = [refreshes_between(a, b, R) for a, b in itertools.pairwise(flip_times)]

    indices = refresh_indices(flip_times, R)
    assert indices.tolist() == np.r_[0, np.cumsum(steps)].tolist()
    assert indices.tolist() == refreshes.tolist()
    assert refresh_indices(np.array([]), R).tolist() == []
    assert refresh_indices(np.array([5.0]), R).tolist() == [0]


def drawn_samples(flip_times):
    """The flicker sample drawn for each flip by the flip loop's bookkeeping (as in
    scripts/calibration.py, run_frames), replayed on the given flip times."""
    shown = []
    refresh = frame = 0
    for j in range(len(flip_times)):
        shown.append(frame)
        if j > 0:
            refresh += refreshes_between(flip_times[j - 1], flip_times[j], R)
        frame = refresh + 1
    return shown


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("onset_error_ms", [-3.9, -3.5, 0.0, 3.5])
def test_flip_loop_does_not_stutter_after_an_off_grid_onset(onset_error_ms, seed):
    """5 s of flicker without dropped frames, timestamp noise like a clean run on the
    thesis laptop (sd 0.2 ms, at most 0.9 ms), and the onset flip's timestamp a few ms
    off its refresh (once seen 3.2-3.9 ms early). After the onset no sample is repeated
    or skipped: flip j shows sample j, or j + 1 for every flip from the second interval
    on if the onset error made the first interval round up to two refreshes."""
    rng = np.random.default_rng(seed)
    refreshes = np.arange(600)
    noise = np.clip(rng.normal(0, 0.0002, len(refreshes)), -0.0009, 0.0009)
    flip_times = 1000.0 + refreshes * PERIOD + noise
    flip_times[0] = 1000.0 + onset_error_ms / 1000

    shift = np.array(drawn_samples(flip_times)) - refreshes
    assert shift[0] == shift[1] == 0
    assert np.all(shift[2:] == shift[2]) and shift[2] in (0, 1)


def test_flip_loop_keeps_phase_after_dropped_frame():
    """One refresh missed: the late flip shows the sample drawn for the missed refresh,
    and every later flip its own refresh's sample, so the phase is not shifted."""
    rng = np.random.default_rng(0)
    refreshes = np.r_[np.arange(0, 50), np.arange(51, 100)]  # refresh 50 missed
    flip_times = 1000.0 + refreshes * PERIOD + rng.uniform(0, 0.002, len(refreshes))

    expected = refreshes.copy()
    expected[50] = 50  # drawn for refresh 50, but displayed one refresh late
    assert drawn_samples(flip_times) == expected.tolist()


def test_flip_loop_does_not_skip_samples_on_a_slightly_slow_display():
    """30 s of flicker on a display refreshing at 119.9 Hz instead of 120 Hz: every flip
    shows the next sample, none is skipped or repeated."""
    flip_times = 1000.0 + np.arange(3600) / 119.9
    assert drawn_samples(flip_times) == list(range(3600))


def test_timing_problems_clean_run():
    """Intervals of one refresh with 0.1 ms jitter: no problems."""
    rng = np.random.default_rng(4)
    assert timing_problems(PERIOD + rng.normal(0, 0.0001, 2000), R) == []


def test_timing_problems_dropped_frames():
    """Intervals of 2 refreshes count as dropped frames ('1 dropped frame', then
    '3 dropped frames' once 2 more intervals of 2 refreshes are added)."""
    intervals = np.full(2000, PERIOD)
    intervals[10] = 2 * PERIOD
    assert timing_problems(intervals, R) == ["1 dropped frame"]
    intervals[20:22] = 2 * PERIOD
    assert timing_problems(intervals, R)[0] == "3 dropped frames"


def test_timing_problems_scattered_flips():
    """Intervals scattered by +-3 ms (sd ~1.7 ms) but none long enough to be a dropped
    frame: one problem, about the sd; with a 2 ms limit, none."""
    rng = np.random.default_rng(5)
    intervals = PERIOD + rng.uniform(-0.003, 0.003, 2000)
    problems = timing_problems(intervals, R)
    assert len(problems) == 1 and "sd" in problems[0]
    assert timing_problems(intervals, R, max_sd_ms=2.0) == []


def test_trial_sequence_balanced_blocks():
    """Each block holds every target exactly once; blocks run 0, 0, ..., 1, 1, ..."""
    targets, blocks = trial_sequence(5, 6, np.random.default_rng(0))
    assert targets.shape == blocks.shape == (30,)
    assert blocks.tolist() == [b for b in range(6) for _ in range(5)]
    for b in range(6):
        assert sorted(targets[blocks == b].tolist()) == [0, 1, 2, 3, 4]


def test_trial_sequence_is_shuffled_and_reproducible():
    """Not every block uses the same order; the same seed gives the same sequence."""
    targets, _ = trial_sequence(5, 6, np.random.default_rng(0))
    orders = {tuple(block) for block in targets.reshape(6, 5)}
    assert len(orders) > 1

    again, _ = trial_sequence(5, 6, np.random.default_rng(0))
    assert np.array_equal(targets, again)


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
    [(n, 1920, 1200) for n in range(2, 10)] + [(6, 800, 600), (6, 1200, 1920)],
)
def test_ring_layout_fits_without_overlap(n_targets, width, height):
    """n centres; every square lies inside the window, squares don't overlap, and all
    centres are equally far from the screen centre."""
    centres, size = ring_layout(n_targets, width, height)
    assert centres.shape == (n_targets, 2)
    assert size > 0

    x, y = centres[:, 0], centres[:, 1]
    assert np.all(x - size / 2 >= 0) and np.all(x + size / 2 <= width)
    assert np.all(y - size / 2 >= 0) and np.all(y + size / 2 <= height)
    for a, b in itertools.combinations(centres, 2):
        assert max(abs(a - b)) >= size
    radii = np.hypot(x - width / 2, y - height / 2)
    assert np.allclose(radii, radii[0])


def test_ring_layout_leaves_the_middle_free():
    """6 targets on 1920x1200: a 600 x 300 px box at the screen centre (room for the
    typed text) touches no square, and the targets are still large (> 200 px)."""
    centres, size = ring_layout(6, 1920, 1200)
    box_half = np.array([300, 150])
    gap = np.abs(centres - [960, 600]) - box_half - size / 2
    assert np.all(gap.max(axis=1) > 0)
    assert size > 200


def test_ring_layout_slots_clockwise_from_top_left():
    """Default slots for 6 targets: top left, top right, right, bottom right,
    bottom left, left (angles 120, 60, 0, -60, -120, 180 degrees)."""
    centres, _ = ring_layout(6, 1920, 1200)
    radius = np.hypot(*(centres[0] - [960, 600]))
    angles = np.radians([120, 60, 0, -60, -120, 180])
    expected = np.column_stack(
        [960 + radius * np.cos(angles), 600 + radius * np.sin(angles)]
    )
    assert np.allclose(centres, expected)


def test_ring_layout_puts_target_k_in_slot_k():
    """With ``slots``, target k is drawn where slot slots[k] is in the default layout."""
    slots = [0, 3, 1, 4, 2, 5]
    default, size = ring_layout(6, 1920, 1200)
    centres, size_slots = ring_layout(6, 1920, 1200, slots)
    assert size_slots == pytest.approx(size)
    for k, slot in enumerate(slots):
        assert np.allclose(centres[k], default[slot])


@pytest.mark.parametrize(
    "n_targets, slots",
    [(6, [0, 1, 2, 3, 4]), (6, [0, 0, 1, 2, 3, 4]), (6, [1, 2, 3, 4, 5, 6]), (1, None)],
)
def test_ring_layout_rejects_bad_input(n_targets, slots):
    """slots that are not a permutation of 0..n-1, or fewer than 2 targets: ValueError."""
    with pytest.raises(ValueError):
        ring_layout(n_targets, 1920, 1200, slots)


def test_config_slots_keep_neighbouring_frequencies_apart():
    """config.TARGET_SLOTS places every target; targets next to each other on the ring
    differ by at least twice the smallest spacing between any two target frequencies."""
    freqs = np.array(config.TARGET_FREQS)
    n = len(freqs)
    assert sorted(config.TARGET_SLOTS) == list(range(n))

    centres, _ = ring_layout(n, 1920, 1200, config.TARGET_SLOTS)
    dist = np.hypot(*(centres[:, None] - centres[None]).transpose(2, 0, 1))
    neighbour = np.isclose(dist, dist[dist > 0].min())
    min_spacing = np.diff(np.sort(freqs)).min()
    for a, b in zip(*np.nonzero(neighbour)):
        assert abs(freqs[a] - freqs[b]) >= 2 * min_spacing - 1e-9, (freqs[a], freqs[b])


def overlaps(box, centres, reach):
    """True if box (x, y, w, h) overlaps a square of half side ``reach`` around any
    centre (touching edges do not count)."""
    x, y, w, h = box
    cx, cy = centres[:, 0], centres[:, 1]
    return bool(
        np.any(
            (x < cx + reach)
            & (x + w > cx - reach)
            & (y < cy + reach)
            & (y + h > cy - reach)
        )
    )


@pytest.mark.parametrize(
    "width, height", [(1920, 1200), (800, 600), (1280, 720), (1200, 1920)]
)
def test_centre_box_is_centred_clear_and_maximal(width, height):
    """On the speller ring: the box is centred, has the requested aspect, keeps the
    clearance from every square and stays on screen, and growing it by 1 % breaks one
    of these."""
    centres, size = ring_layout(6, width, height, config.TARGET_SLOTS)
    clearance, aspect = 40.0, 1.4
    x, y, w, h = centre_box(centres, size, width, height, clearance, aspect)

    assert x + w / 2 == pytest.approx(width / 2)
    assert y + h / 2 == pytest.approx(height / 2)
    assert w / h == pytest.approx(aspect)
    reach = size / 2 + clearance - 1e-6
    assert not overlaps((x, y, w, h), centres, reach)
    assert x >= 0 and y >= 0 and x + w <= width and y + h <= height

    grown = (x - 0.005 * w, y - 0.005 * h, 1.01 * w, 1.01 * h)
    off_screen = grown[0] < 0 or grown[1] < 0
    assert overlaps(grown, centres, size / 2 + clearance) or off_screen


def test_centre_box_known_value():
    """One 100 px square 300 px right of the centre of a 1000 x 1000 screen, no
    clearance, aspect 2: the box may reach 250 px to the right, so it is 500 x 250."""
    centres = np.array([[800.0, 500.0]])
    box = centre_box(centres, 100.0, 1000, 1000, clearance=0.0, aspect=2.0)
    assert box == pytest.approx((250.0, 375.0, 500.0, 250.0))


def test_centre_box_without_room_raises():
    """A square on the screen centre leaves no room: ValueError."""
    with pytest.raises(ValueError):
        centre_box(np.array([[500.0, 500.0]]), 100.0, 1000, 1000, 10.0, 1.5)
