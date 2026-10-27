"""Calibration display. Current step: flicker test with a flip-timing log.

Usage:
    .venv/bin/python scripts/calibration.py                       # 10 s, all targets
    .venv/bin/python scripts/calibration.py --seconds 30 --screen 1
    .venv/bin/python scripts/calibration.py --windowed            # debugging only

Opens a fullscreen window and flickers config.TARGET_FREQS as sampled sinusoids.
Every flip is timestamped with pylsl.local_clock() once the buffer swap has finished.
At the end it prints the measured refresh rate and the number of dropped frames, and
saves the flip log to results/timing/flicker_<date>.npz.

Set the monitor to config.REFRESH_RATE in GNOME Settings -> Displays, then log out and
back in before running. Changing the rate live (or with xrandr) can leave GNOME Shell
on a mode it later loses track of, and it crashed once when this script started.
"""

import argparse
import gc
from datetime import datetime

import numpy as np
import pyglet
import pylsl
from pyglet import gl, shapes
from pyglet.libs.x11 import xrandr
from pyglet.window import key

from ssvep import config, stimulus

RATE_TOLERANCE = 0.02  # allowed relative error of the measured refresh rate
MIN_PROMPT_TIME = (
    1.0  # s of flips before SPACE is accepted, to measure the refresh rate
)
TEXT_COLOR = (160, 160, 160, 255)


class FlickerScene:
    """Square targets with frequency labels; their gray level is set every frame."""

    def __init__(self, window: pyglet.window.Window, freqs: list[float]):
        self.batch = pyglet.graphics.Batch()
        centres, size = stimulus.grid_layout(len(freqs), window.width, window.height)
        self.squares = []
        self.labels = []
        for (x, y), freq in zip(centres, freqs):
            square = shapes.Rectangle(
                x - size / 2,
                y - size / 2,
                size,
                size,
                color=(0, 0, 0),
                batch=self.batch,
            )
            label = pyglet.text.Label(
                f"{freq:g} Hz",
                x=x,
                y=y - size / 2 - 24,
                anchor_x="center",
                anchor_y="center",
                font_size=14,
                color=TEXT_COLOR,
                batch=self.batch,
            )
            self.squares.append(square)
            self.labels.append(label)

    def set_gray(self, levels: np.ndarray):
        """Set each target's gray level (0..255)."""
        for square, level in zip(self.squares, levels):
            square.color = (int(level), int(level), int(level))

    def draw(self):
        self.batch.draw()


def open_window(screen_index: int, windowed: bool) -> pyglet.window.Window:
    """Fullscreen vsync'd window on the given screen, or a small window for debugging."""
    # pyglet lists screens with XRRGetScreenResources, which makes the X server
    # re-probe the monitor. Right after a refresh-rate change in GNOME Settings this
    # crashed GNOME Shell (mutter 46, X11). The Current variant reads the cached
    # state without probing.
    xrandr.XRRGetScreenResources = xrandr.XRRGetScreenResourcesCurrent
    if windowed:
        return pyglet.window.Window(800, 600, caption="SSVEP calibration", vsync=True)
    screen = pyglet.display.get_display().get_screens()[screen_index]
    window = pyglet.window.Window(fullscreen=True, screen=screen, vsync=True)
    window.set_mouse_visible(False)
    return window


def flip(window: pyglet.window.Window) -> float:
    """Show the back buffer and return the LSL time once the swap has happened."""
    window.flip()
    gl.glFinish()  # flip() can return before the swap; glFinish waits for it
    return pylsl.local_clock()


def wait_for_space(window: pyglet.window.Window, text: str) -> np.ndarray:
    """Show ``text`` until SPACE (or ESC) is pressed -> flip times while waiting."""
    label = pyglet.text.Label(
        text,
        x=window.width / 2,
        y=window.height / 2,
        anchor_x="center",
        anchor_y="center",
        font_size=20,
        color=TEXT_COLOR,
    )
    keys = key.KeyStateHandler()
    window.push_handlers(keys)
    flip_times = []
    while not window.has_exit:
        window.dispatch_events()
        waited = flip_times[-1] - flip_times[0] if flip_times else 0.0
        if keys[key.SPACE] and waited >= MIN_PROMPT_TIME:
            break
        window.clear()
        label.draw()
        flip_times.append(flip(window))
    window.remove_handlers(keys)
    return np.array(flip_times)


def flicker(
    window: pyglet.window.Window,
    scene: FlickerScene,
    freqs: list[float],
    phases: list[float],
    refresh_rate: float,
    duration: float,
) -> np.ndarray:
    """Flicker all targets for ``duration`` s -> LSL time of every flip.

    The first flip is the stimulus onset. The last flip blanks the targets, so the
    final flicker frame stays on screen for exactly one refresh. Stops early on ESC.
    """
    n_frames = int(round(duration * refresh_rate))
    flip_times = []
    frame = 0
    gc.disable()  # a garbage-collection pause can cost a frame
    try:
        while frame < n_frames and not window.has_exit:
            window.dispatch_events()
            lum = stimulus.luminance(freqs, phases, frame, refresh_rate)[:, 0]
            scene.set_gray(stimulus.gray_levels(lum))
            window.clear()
            scene.draw()
            flip_times.append(flip(window))
            frame = stimulus.next_frame_index(
                flip_times[-1], flip_times[0], refresh_rate
            )
        scene.set_gray(np.zeros(len(freqs), dtype=int))
        window.clear()
        scene.draw()
        flip_times.append(flip(window))
    finally:
        gc.enable()
    return np.array(flip_times)


def print_report(report: dict):
    print(
        f"{report['n_flips']} flips in {report['duration_s']:.3f} s, "
        f"measured refresh rate {report['measured_rate_hz']:.2f} Hz"
    )
    print(
        f"Flip interval: mean {report['mean_interval_ms']:.3f} ms, "
        f"sd {report['std_interval_ms']:.3f} ms, max {report['max_interval_ms']:.3f} ms"
    )
    print(
        f"Dropped frames: {report['n_dropped']} "
        f"(in {report['n_late_flips']} late flips)"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="flicker duration")
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=config.REFRESH_RATE,
        help="expected monitor refresh rate in Hz",
    )
    parser.add_argument("--screen", type=int, default=0, help="screen for fullscreen")
    parser.add_argument(
        "--windowed",
        action="store_true",
        help="800x600 window for debugging; timing is not valid",
    )
    args = parser.parse_args()

    freqs, phases = config.TARGET_FREQS, config.TARGET_PHASES
    window = open_window(args.screen, args.windowed)
    scene = FlickerScene(window, freqs)

    prompt_flips = wait_for_space(window, "Flicker test: SPACE to start, ESC to quit")
    if window.has_exit:
        window.close()
        return

    measured = stimulus.timing_report(prompt_flips, args.refresh_rate)
    rate = measured["measured_rate_hz"]
    if abs(rate - args.refresh_rate) > RATE_TOLERANCE * args.refresh_rate:
        message = (
            f"Measured refresh rate {rate:.1f} Hz, expected {args.refresh_rate:g} Hz. "
            "Check the refresh rate in Settings -> Displays and that vsync is on."
        )
        if not args.windowed:
            window.close()
            raise SystemExit(message)
        print(f"Warning: {message}")

    flip_times = flicker(window, scene, freqs, phases, args.refresh_rate, args.seconds)
    aborted = window.has_exit
    window.close()
    if aborted:
        print("Aborted with ESC.")
    if len(flip_times) < 2:
        return

    print_report(stimulus.timing_report(flip_times, args.refresh_rate))

    out_dir = config.RESULTS_DIR / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flicker_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path,
        flip_times=flip_times,
        refresh_rate=args.refresh_rate,
        freqs=freqs,
        phases=phases,
        windowed=args.windowed,
        aborted=aborted,
    )
    print(f"Flip log saved to {path}")


if __name__ == "__main__":
    main()
