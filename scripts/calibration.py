"""Calibration session: cued SSVEP trials with LSL event markers.

Usage:
    .venv/bin/python scripts/calibration.py --subject S01            # full session
    .venv/bin/python scripts/calibration.py --no-recorder --blocks 1  # dry run
    .venv/bin/python scripts/calibration.py --test --seconds 30      # flicker timing test
    add --windowed for debugging (timing is not valid)

Protocol: config.N_BLOCKS blocks, each showing every target once in random order.
A trial is a red frame around the target (config.CUE_DURATION s) followed by the
flicker (config.STIM_DURATION s). The subject rests between blocks and starts the
next one with SPACE. ESC quits.

Markers go to the LSL stream config.MARKER_STREAM as JSON strings, each timestamped
with the LSL time of the flip that showed the event (stim_on = first flicker flip):
    session_start  subject, freqs, phases, refresh_rate, n_blocks, cue_s, stim_s
    block_start    block
    cue            trial, block, target, freq
    stim_on        trial, block, target, freq
    stim_off       trial, block, target, freq, n_dropped, complete
    block_end      block
    session_end    n_trials, aborted
``complete`` is false when ESC cut the flicker short. Before the window opens the
script waits for a recorder to connect to the stream (skip with --no-recorder). The
trial table is also saved to data/recordings/<subject>_<date>_trials.csv.

--test flickers all targets once and saves the flip log to results/timing/.

Set the monitor to config.REFRESH_RATE in GNOME Settings -> Displays, then log out and
back in before running. Changing the rate live (or with xrandr) can leave GNOME Shell
on a mode it later loses track of, and it crashed once when this script started.
"""

import argparse
import gc
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyglet
import pylsl
from pyglet import gl, shapes
from pyglet.libs.x11 import xrandr
from pyglet.window import key

from ssvep import config, stimulus

RATE_TOLERANCE = 0.02  # allowed relative error of the measured refresh rate
MIN_PROMPT_TIME = 1.0  # s of flips before SPACE is accepted (refresh-rate estimate)
TEXT_COLOR = (160, 160, 160, 255)
CUE_COLOR = (220, 30, 30)
CUE_GAP = 16  # px between a target and its cue frame
CUE_THICKNESS = 8  # px


class FlickerScene:
    """Square targets with frequency labels and a red cue frame."""

    def __init__(
        self,
        window: pyglet.window.Window,
        freqs: list[float],
        phases: list[float],
        refresh_rate: float,
    ):
        self.freqs = freqs
        self.phases = phases
        self.refresh_rate = refresh_rate
        self.batch = pyglet.graphics.Batch()
        self.centres, self.size = stimulus.grid_layout(
            len(freqs), window.width, window.height
        )
        half = self.size / 2
        self.squares = []
        self.labels = []
        for (x, y), freq in zip(self.centres, freqs):
            square = shapes.Rectangle(
                x - half,
                y - half,
                self.size,
                self.size,
                color=(0, 0, 0),
                batch=self.batch,
            )
            label = pyglet.text.Label(
                f"{freq:g} Hz",
                x=x,
                y=y - half - CUE_GAP - 30,
                anchor_x="center",
                anchor_y="center",
                font_size=14,
                color=TEXT_COLOR,
                batch=self.batch,
            )
            self.squares.append(square)
            self.labels.append(label)

        outer = self.size + 2 * CUE_GAP
        self.cue = shapes.Box(
            0, 0, outer, outer, CUE_THICKNESS, color=CUE_COLOR, batch=self.batch
        )
        self.cue.visible = False

    def set_frame(self, frame: int):
        """Set the targets' gray levels for flicker frame ``frame`` (0 = at rest)."""
        lum = stimulus.luminance(self.freqs, self.phases, frame, self.refresh_rate)
        for square, level in zip(self.squares, stimulus.gray_levels(lum[:, 0])):
            square.color = (int(level), int(level), int(level))

    def set_cue(self, target: int | None):
        """Put the cue frame around ``target``, or hide it (None)."""
        if target is None:
            self.cue.visible = False
            return
        x, y = self.centres[target]
        half = self.size / 2 + CUE_GAP
        self.cue.position = (x - half, y - half)
        self.cue.visible = True

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


def open_marker_outlet() -> pylsl.StreamOutlet:
    """LSL outlet for the JSON event markers (one string channel, irregular rate)."""
    info = pylsl.StreamInfo(
        config.MARKER_STREAM,
        "Markers",
        1,
        pylsl.IRREGULAR_RATE,
        pylsl.cf_string,
        "ssvep-gui",
    )
    return pylsl.StreamOutlet(info)


def wait_for_recorder(outlet: pylsl.StreamOutlet) -> bool:
    """Block until something reads the marker stream -> False on Ctrl+C."""
    print(f"Waiting for a recorder on LSL stream '{config.MARKER_STREAM}'")
    print("(start it now, or Ctrl+C to quit)...")
    try:
        while not outlet.wait_for_consumers(1.0):
            pass
    except KeyboardInterrupt:
        return False
    print("Recorder connected.")
    return True


def send_marker(outlet: pylsl.StreamOutlet, timestamp: float, event: str, **fields):
    """Push one JSON marker with the LSL time of the flip that showed the event."""
    outlet.push_sample([json.dumps({"event": event, **fields})], timestamp)


def flip(window: pyglet.window.Window) -> float:
    """Show the back buffer and return the LSL time once the swap has happened."""
    window.flip()
    gl.glFinish()  # flip() can return before the swap; glFinish waits for it
    return pylsl.local_clock()


def show_frame(window: pyglet.window.Window, scene: FlickerScene, frame: int) -> float:
    """Draw flicker frame ``frame`` (0 = at rest) and flip -> LSL time of the flip."""
    scene.set_frame(frame)
    window.clear()
    scene.draw()
    return flip(window)


def run_frames(
    window: pyglet.window.Window,
    scene: FlickerScene,
    n_frames: int,
    flickering: bool,
    on_onset=None,
) -> np.ndarray:
    """Show ``n_frames`` refreshes of flickering (or resting) targets -> flip times.

    The first flip is the onset; ``on_onset(t)`` is called right after it, so its
    marker goes out in real time. Stops early on ESC.
    """
    flip_times = []
    frame = 0
    while frame < n_frames and not window.has_exit:
        window.dispatch_events()
        flip_times.append(show_frame(window, scene, frame if flickering else 0))
        if len(flip_times) == 1 and on_onset is not None:
            on_onset(flip_times[0])
        frame = stimulus.next_frame_index(
            flip_times[-1], flip_times[0], scene.refresh_rate
        )
    return np.array(flip_times)


def wait_for_space(window: pyglet.window.Window, text: str) -> np.ndarray:
    """Show ``text`` until SPACE (or ESC) is pressed -> flip times while waiting."""
    label = pyglet.text.Label(
        text,
        x=window.width / 2,
        y=window.height / 2,
        width=int(0.8 * window.width),
        multiline=True,
        align="center",
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


def check_refresh_rate(
    window: pyglet.window.Window, flip_times: np.ndarray, expected: float, strict: bool
):
    """Compare the refresh rate measured from ``flip_times`` with ``expected``.

    On a mismatch: close the window and exit if ``strict``, otherwise warn.
    """
    rate = stimulus.timing_report(flip_times, expected)["measured_rate_hz"]
    if abs(rate - expected) <= RATE_TOLERANCE * expected:
        return
    message = (
        f"Measured refresh rate {rate:.1f} Hz, expected {expected:g} Hz. "
        "Check the refresh rate in Settings -> Displays and that vsync is on."
    )
    if strict:
        window.close()
        raise SystemExit(message)
    print(f"Warning: {message}")


def run_trial(
    window: pyglet.window.Window,
    scene: FlickerScene,
    outlet: pylsl.StreamOutlet,
    info: dict,
    n_cue: int,
    n_stim: int,
) -> dict | None:
    """Cue, flicker, then a resting flip -> trial-log row (None on ESC before onset).

    ``info`` holds the trial's marker fields: trial, block, target, freq.
    """
    scene.set_cue(info["target"])
    cue = run_frames(
        window, scene, n_cue, False, lambda t: send_marker(outlet, t, "cue", **info)
    )
    scene.set_cue(None)
    if window.has_exit:
        return None

    stim = run_frames(
        window, scene, n_stim, True, lambda t: send_marker(outlet, t, "stim_on", **info)
    )
    if len(stim) == 0:
        return None
    off = show_frame(window, scene, 0)
    n_dropped = int(stimulus.dropped_frames(np.r_[stim, off], scene.refresh_rate).sum())
    complete = (
        stimulus.next_frame_index(stim[-1], stim[0], scene.refresh_rate) >= n_stim
    )
    send_marker(outlet, off, "stim_off", **info, n_dropped=n_dropped, complete=complete)
    return {
        **info,
        "cue": cue[0],
        "stim_on": stim[0],
        "stim_off": off,
        "n_dropped": n_dropped,
        "complete": complete,
    }


def run_session(
    window: pyglet.window.Window,
    scene: FlickerScene,
    outlet: pylsl.StreamOutlet,
    n_blocks: int,
    cue_s: float,
    stim_s: float,
) -> list[dict]:
    """All blocks, with a SPACE-to-continue break before every block but the first
    -> trial-log rows. Stops early on ESC."""
    n_cue = int(round(cue_s * scene.refresh_rate))
    n_stim = int(round(stim_s * scene.refresh_rate))
    targets, blocks = stimulus.trial_sequence(
        len(scene.freqs), n_blocks, np.random.default_rng()
    )
    rows = []
    for block in range(n_blocks):
        if block > 0:
            wait_for_space(
                window, f"Break. Block {block + 1} of {n_blocks}: SPACE to start."
            )
            if window.has_exit:
                break
        send_marker(outlet, pylsl.local_clock(), "block_start", block=block)
        gc.collect()
        gc.disable()  # a garbage-collection pause can cost a frame
        try:
            for trial in np.flatnonzero(blocks == block):
                target = int(targets[trial])
                info = {
                    "trial": int(trial),
                    "block": block,
                    "target": target,
                    "freq": scene.freqs[target],
                }
                row = run_trial(window, scene, outlet, info, n_cue, n_stim)
                if row is not None:
                    rows.append(row)
                if window.has_exit:
                    break
        finally:
            gc.enable()
        if window.has_exit:
            break
        send_marker(outlet, pylsl.local_clock(), "block_end", block=block)
    return rows


def calibration(
    window: pyglet.window.Window,
    scene: FlickerScene,
    outlet: pylsl.StreamOutlet,
    args: argparse.Namespace,
):
    """Run the session, then close the window and save the trial table."""
    send_marker(
        outlet,
        pylsl.local_clock(),
        "session_start",
        subject=args.subject,
        freqs=scene.freqs,
        phases=scene.phases,
        refresh_rate=scene.refresh_rate,
        n_blocks=args.blocks,
        cue_s=args.cue,
        stim_s=args.stim,
    )
    rows = run_session(window, scene, outlet, args.blocks, args.cue, args.stim)
    aborted = window.has_exit
    send_marker(
        outlet, pylsl.local_clock(), "session_end", n_trials=len(rows), aborted=aborted
    )
    window.close()
    time.sleep(1.0)  # give the recorder time to receive the last markers

    if aborted:
        print("Aborted with ESC.")
    if not rows:
        return
    trials = pd.DataFrame(rows)
    print(
        f"{len(trials)} trials ({(~trials['complete']).sum()} incomplete), "
        f"{trials['n_dropped'].sum()} dropped frames in "
        f"{(trials['n_dropped'] > 0).sum()} trials"
    )
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RECORDINGS_DIR / (
        f"{args.subject}_{datetime.now():%Y%m%d_%H%M%S}_trials.csv"
    )
    trials.to_csv(path, index=False)
    print(f"Trial table saved to {path}")


def flicker_test(
    window: pyglet.window.Window, scene: FlickerScene, args: argparse.Namespace
):
    """Flicker all targets once, then close the window and save the flip log."""
    gc.disable()  # a garbage-collection pause can cost a frame
    try:
        n_frames = int(round(args.seconds * scene.refresh_rate))
        stim = run_frames(window, scene, n_frames, flickering=True)
        flip_times = np.r_[stim, show_frame(window, scene, 0)]
    finally:
        gc.enable()
    aborted = window.has_exit
    window.close()
    if aborted:
        print("Aborted with ESC.")
    if len(flip_times) < 2:
        return

    report = stimulus.timing_report(flip_times, scene.refresh_rate)
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

    out_dir = config.RESULTS_DIR / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"flicker_{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez(
        path,
        flip_times=flip_times,
        refresh_rate=scene.refresh_rate,
        freqs=scene.freqs,
        phases=scene.phases,
        windowed=args.windowed,
        aborted=aborted,
    )
    print(f"Flip log saved to {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", default="pilot", help="subject ID for the logs")
    parser.add_argument("--blocks", type=int, default=config.N_BLOCKS)
    parser.add_argument(
        "--cue", type=float, default=config.CUE_DURATION, help="cue duration in s"
    )
    parser.add_argument(
        "--stim", type=float, default=config.STIM_DURATION, help="flicker duration in s"
    )
    parser.add_argument(
        "--no-recorder",
        action="store_true",
        help="start without waiting for a recorder on the marker stream",
    )
    parser.add_argument(
        "--test", action="store_true", help="flicker timing test instead of a session"
    )
    parser.add_argument(
        "--seconds", type=float, default=10.0, help="flicker duration with --test"
    )
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

    outlet = None
    if not args.test:
        outlet = open_marker_outlet()
        if not args.no_recorder and not wait_for_recorder(outlet):
            return

    window = open_window(args.screen, args.windowed)
    scene = FlickerScene(
        window, config.TARGET_FREQS, config.TARGET_PHASES, args.refresh_rate
    )
    if args.test:
        text = "Flicker test\n\nSPACE to start, ESC to quit"
    else:
        text = (
            f"Calibration: {args.blocks} blocks of {len(scene.freqs)} trials.\n"
            "Look at the target in the red frame until it stops flickering.\n\n"
            "SPACE to start, ESC to quit"
        )
    prompt_flips = wait_for_space(window, text)
    if window.has_exit:
        window.close()
        return
    check_refresh_rate(window, prompt_flips, args.refresh_rate, not args.windowed)

    if args.test:
        flicker_test(window, scene, args)
    else:
        calibration(window, scene, outlet, args)


if __name__ == "__main__":
    main()
