"""Online speller GUI: runs the synchronous trials and draws what the backend sends.

Usage:
    .venv/bin/python scripts/speller_gui.py --subject S01        # with the backend
    .venv/bin/python scripts/speller_gui.py --demo --windowed    # GUI check, no backend
    add --windowed for debugging (timing is not valid)

The GUI is a thin view: the backend owns the speller state and the classifier, and the
GUI flickers the targets, shows the labels and text it gets, and timestamps every event.
The messages in both directions are described in ssvep/protocol.py.

One trial (one selection):
    gaze      --gaze s (config.GAZE_SHIFT), targets at rest with their labels; a red
              frame around the state's target if the state asks for a cue
    flicker   --stim s (config.ONLINE_STIM_DURATION), labels hidden, so the stimulus
              looks exactly like calibration; stim_on marker at the first flip
    wait      targets at rest until the backend answers (RESPONSE_TIMEOUT s at most,
              then the GUI pauses)
    feedback  --feedback s (config.FEEDBACK_DURATION), green frame around the selected
              target and the new text; the new labels appear after it
The operator starts with SPACE, pauses after the current trial with SPACE, and quits
with ESC. The backend can pause too (e.g. after a phrase).

Start the backend (it may also start after the GUI, which waits for its state
stream), then this script, then scripts/record.py, and press SPACE once the recorder
runs: session_start goes out on that SPACE. The trial table is saved to
data/recordings/<subject>_<date>_online.csv.

--demo replaces the backend with a stand-in that cues a random target in every trial
and selects it, to check the display and the flicker timing without EEG. Its markers
still go out, so it also works with fake_eeg_stream.py and record.py.

Set the monitor to config.REFRESH_RATE in GNOME Settings -> Displays, then log out and
back in before running (see scripts/calibration.py).
"""

import argparse
import gc
import time
from datetime import datetime

import numpy as np
import pandas as pd
import pyglet
import pylsl
from pyglet import shapes
from pyglet.window import key

from calibration import (
    CUE_GAP,
    CUE_THICKNESS,
    MIN_PROMPT_TIME,
    TEXT_COLOR,
    FlickerScene,
    check_refresh_rate,
    open_marker_outlet,
    open_window,
    run_frames,
    send_marker,
    show_frame,
)
from ssvep import acquisition, config, protocol, stimulus

RESPONSE_TIMEOUT = 3.0  # s after stim_off to wait for the backend's answer
REQUEST_EVERY = 0.5  # s between state_request markers until the first state arrives
FEEDBACK_COLOR = (40, 200, 60)
TYPED_COLOR = (235, 235, 235, 255)
FONT = "DejaVu Sans"  # has ␣ and ⌫; the default sans-serif (Noto Sans) has neither
BOX_CLEARANCE = 16  # px between the text box and a cue/feedback frame
BOX_ASPECT = 1.4  # width / height of the text box in the middle of the ring
LABEL_FILL = 0.8  # fraction of a square's side a label may cover
MINI_FILL = 1.0  # cell size in a nested label's ring, as in stimulus.ring_layout
MINI_MARGIN = 0.08
DEMO_BLOCK = 12  # trials between the demo's pauses


def fit_tail(label: pyglet.text.Label, text: str, max_height: float):
    """Show as much of the end of ``text`` in ``label`` as fits in ``max_height`` px,
    with "…" in front if the start is cut off."""
    label.text = text
    if label.content_height <= max_height:
        return
    # Smallest start index whose tail fits
    low, high = 1, len(text)
    while low < high:
        middle = (low + high) // 2
        label.text = "…" + text[middle:]
        if label.content_height <= max_height:
            high = middle
        else:
            low = middle + 1
    label.text = "…" + text[low:]


class SpellerScene(FlickerScene):
    """The calibration targets plus the speller's labels, text and feedback frame.

    A target's label is a string drawn on the square, or a list of one string per target
    drawn as a small ring on the square: entry k sits where target k sits on the screen.
    Labels are hidden while ``show_labels`` is false (during the flicker). The middle
    of the ring shows the phrase to copy, the typed text with a cursor, and a message.
    """

    def __init__(
        self,
        window: pyglet.window.Window,
        freqs: list[float],
        phases: list[float],
        slots: list[int],
        refresh_rate: float,
    ):
        super().__init__(window, freqs, phases, slots, refresh_rate, freq_labels=False)
        self.show_labels = True
        self.label_batch = pyglet.graphics.Batch()  # drawn over the squares
        self.target_labels = []
        # Black text on squares that are light at rest, white on dark ones
        rest = stimulus.luminance(freqs, phases, 0, refresh_rate)[:, 0]
        self.label_colors = [
            (0, 0, 0, 255) if level >= 128 else (255, 255, 255, 255)
            for level in stimulus.gray_levels(rest)
        ]

        outer = self.size + 2 * CUE_GAP
        self.feedback = shapes.Box(
            0, 0, outer, outer, CUE_THICKNESS, color=FEEDBACK_COLOR, batch=self.batch
        )
        self.feedback.visible = False

        x, y, w, h = stimulus.centre_box(
            self.centres,
            self.size,
            window.width,
            window.height,
            CUE_GAP + CUE_THICKNESS + BOX_CLEARANCE,
            BOX_ASPECT,
        )
        # Phrase at the top, typed text below it, message at the bottom of the box
        self.text_top = y + 0.68 * h
        self.text_height = 0.53 * h
        common = {
            "x": x,
            "width": int(w),
            "multiline": True,
            "font_name": FONT,
            "batch": self.batch,
        }
        self.phrase = pyglet.text.Label(
            "", y=y + h, anchor_y="top", font_size=w / 26, color=TEXT_COLOR, **common
        )
        self.typed = pyglet.text.Label(
            "",
            y=self.text_top,
            anchor_y="top",
            font_size=w / 20,
            color=TYPED_COLOR,
            **common,
        )
        self.message = pyglet.text.Label(
            "",
            y=y,
            anchor_y="bottom",
            align="center",
            font_size=w / 30,
            color=TEXT_COLOR,
            **common,
        )

    def _add_label(self, text: str, x: float, y: float, max_size: float, color):
        """Bold label centred on (x, y), shrunk to fit a ``max_size`` px square."""
        label = pyglet.text.Label(
            text,
            x=x,
            y=y,
            anchor_x="center",
            anchor_y="center",
            font_size=0.4 * max_size,
            font_name=FONT,
            weight="bold",
            color=color,
            batch=self.label_batch,
        )
        # Text size scales with the font size, so one measurement is enough
        scale = min(
            1.0,
            max_size / max(label.content_width, 1),
            max_size / max(label.content_height, 1),
        )
        if scale < 1.0:
            label.font_size *= scale
        self.target_labels.append(label)

    def set_labels(self, labels: list):
        """Replace the target labels (protocol format, one entry per target)."""
        for label in self.target_labels:
            label.delete()
        self.target_labels = []
        for (x, y), entry, color in zip(self.centres, labels, self.label_colors):
            if isinstance(entry, str):
                self._add_label(entry, x, y, LABEL_FILL * self.size, color)
                continue
            mini, cell = stimulus.ring_layout(
                len(entry), self.size, self.size, self.slots, MINI_FILL, MINI_MARGIN
            )
            left, bottom = x - self.size / 2, y - self.size / 2
            for (mx, my), item in zip(mini, entry):
                self._add_label(item, left + mx, bottom + my, cell, color)

    def set_text(self, phrase: str, text: str, message: str):
        """Phrase, typed text (its end, if it is too long) and message."""
        self.phrase.text = phrase
        fit_tail(self.typed, text + "_", self.text_height)
        self.set_message(message)

    def set_message(self, message: str):
        if self.message.text != message:  # a new text costs a layout
            self.message.text = message

    def show_state(self, state: dict):
        """Labels, phrase, text and message of a protocol state."""
        self.set_labels(state["labels"])
        self.set_text(state["phrase"], state["text"], state["message"])

    def set_feedback(self, target: int | None):
        """Put the green frame around ``target``, or hide it (None)."""
        if target is None:
            self.feedback.visible = False
            return
        x, y = self.centres[target]
        half = self.size / 2 + CUE_GAP
        self.feedback.position = (x - half, y - half)
        self.feedback.visible = True

    def draw(self):
        self.batch.draw()
        if self.show_labels:
            self.label_batch.draw()


class LslLink:
    """Markers to the backend (and the recorder), states from the backend."""

    def __init__(
        self, outlet: pylsl.StreamOutlet, inlet: pylsl.StreamInlet, n_targets: int
    ):
        self.outlet = outlet
        self.inlet = inlet
        self.n_targets = n_targets

    def send(self, timestamp: float, event: str, **fields):
        send_marker(self.outlet, timestamp, event, **fields)

    def receive(self) -> list[dict]:
        """The states that arrived since the last call, oldest first, without waiting.

        Invalid messages are reported and skipped, so a backend bug can't crash a
        session.
        """
        samples, _ = self.inlet.pull_chunk(timeout=0.0)
        states = []
        for sample in samples:
            try:
                states.append(protocol.parse_state(sample[0], self.n_targets))
            except ValueError as error:
                print(f"Ignoring invalid state from the backend: {error}")
        return states


class DemoLink:
    """Stands in for the backend (--demo): cues a random target in every trial, selects
    it and adds its frequency to the text; pauses every DEMO_BLOCK trials. Markers
    still go to the outlet."""

    def __init__(self, outlet: pylsl.StreamOutlet, freqs: list[float]):
        self.outlet = outlet
        self.freqs = freqs
        self.rng = np.random.default_rng()
        self.inbox = []
        self.text = ""

    def send(self, timestamp: float, event: str, **fields):
        send_marker(self.outlet, timestamp, event, **fields)
        if event == "state_request":
            self._reply(None, None, "Demo: look at the target in the red frame")
        elif event == "stim_off" and fields["complete"]:
            selected = fields["target"]
            self.text = (self.text + f"{self.freqs[selected]:g} ")[-80:]
            trial = fields["trial"]
            self._reply(
                trial,
                selected,
                f"Trial {trial}: selected {self.freqs[selected]:g} Hz",
                pause=(trial + 1) % DEMO_BLOCK == 0,
            )

    def _reply(self, trial, selected, message, pause=False):
        state = protocol.state_message(
            labels=[f"{freq:g} Hz" for freq in self.freqs],
            text=self.text,
            phrase="Demo spelling (no backend)",
            target=int(self.rng.integers(len(self.freqs))),
            cue=True,
            message=message,
            trial=trial,
            selected=selected,
            pause=pause,
        )
        self.inbox.append(protocol.parse_state(state, len(self.freqs)))

    def receive(self) -> list[dict]:
        states, self.inbox = self.inbox, []
        return states


class Session:
    """Start prompt, blocks of trials separated by pauses, and the trial table."""

    def __init__(
        self,
        window: pyglet.window.Window,
        scene: SpellerScene,
        link: LslLink | DemoLink,
        args: argparse.Namespace,
    ):
        self.window = window
        self.scene = scene
        self.link = link
        self.args = args
        rate = scene.refresh_rate
        self.n_gaze = int(round(args.gaze * rate))
        self.n_stim = int(round(args.stim * rate))
        self.n_feedback = int(round(args.feedback * rate))
        self.state = None  # newest state from the backend
        self.rows = []  # trial table, one row per flicker
        self.aborted = False  # ESC during a block
        self.space = False  # SPACE pressed since the last check
        window.push_handlers(on_key_press=self.on_key_press)

    def on_key_press(self, symbol, modifiers):
        # Not handled here, so the window's default handler still quits on ESC
        if symbol == key.SPACE:
            self.space = True

    def receive(self, show: bool) -> list[dict]:
        """States from the backend; the newest becomes ``self.state`` (and is drawn
        if ``show``)."""
        states = self.link.receive()
        if states:
            self.state = states[-1]
            if show:
                self.scene.show_state(self.state)
        return states

    def idle(self, message: str) -> np.ndarray:
        """Targets at rest with ``message`` until SPACE (or ESC) -> flip times.

        New states are drawn as they arrive. Until the first one, the backend is asked
        for it every REQUEST_EVERY s and SPACE does nothing.
        """
        if self.state is not None:
            self.scene.show_state(self.state)
        self.space = False
        last_request = -np.inf
        flip_times = []
        while not self.window.has_exit:
            self.window.dispatch_events()
            self.receive(show=True)
            if self.state is None:
                self.scene.set_message("Waiting for the backend...")
                now = pylsl.local_clock()
                if now - last_request >= REQUEST_EVERY:
                    self.link.send(now, "state_request")
                    last_request = now
            else:
                self.scene.set_message(message)
            waited = flip_times[-1] - flip_times[0] if flip_times else 0.0
            if self.space:
                self.space = False
                if self.state is not None and waited >= MIN_PROMPT_TIME:
                    break
            flip_times.append(show_frame(self.window, self.scene, 0))
        return np.array(flip_times)

    def wait_for_answer(self, trial: int) -> tuple[dict, float] | None:
        """Targets at rest until the backend's state for ``trial`` arrives -> (state,
        LSL time it arrived). None after RESPONSE_TIMEOUT s or on ESC. The labels on
        screen stay as they are."""
        start = pylsl.local_clock()
        while not self.window.has_exit:
            self.window.dispatch_events()
            answers = [s for s in self.receive(show=False) if s["trial"] == trial]
            if answers:
                return answers[-1], pylsl.local_clock()
            if show_frame(self.window, self.scene, 0) - start > RESPONSE_TIMEOUT:
                return None
        return None

    def run_trial(self, block: int) -> dict | None:
        """Gaze, flicker, wait, feedback -> the backend's answer (None on ESC or if it
        doesn't answer). Every flicker adds a row to the trial table."""
        target = self.state["target"]
        info = {
            "trial": len(self.rows),
            "block": block,
            "target": target,
            "freq": None if target is None else self.scene.freqs[target],
        }
        cue = target if self.state["cue"] else None
        self.scene.set_cue(cue)
        on_cue = None if cue is None else lambda t: self.link.send(t, "cue", **info)
        run_frames(self.window, self.scene, self.n_gaze, False, on_cue)
        self.scene.set_cue(None)
        if self.window.has_exit:
            return None

        self.scene.show_labels = False
        stim = run_frames(
            self.window,
            self.scene,
            self.n_stim,
            True,
            lambda t: self.link.send(t, "stim_on", **info),
        )
        self.scene.show_labels = True
        if len(stim) == 0:
            return None
        off = show_frame(self.window, self.scene, 0)
        refresh_rate = self.scene.refresh_rate
        n_dropped = int(stimulus.dropped_frames(np.r_[stim, off], refresh_rate).sum())
        complete = (
            stimulus.next_frame_index(stim[-1], stim[0], refresh_rate) >= self.n_stim
        )
        self.link.send(off, "stim_off", **info, n_dropped=n_dropped, complete=complete)
        row = {
            **info,
            "cued": cue is not None,
            "stim_on": stim[0],
            "stim_off": off,
            "n_dropped": n_dropped,
            "complete": complete,
            "answer": np.nan,
            "selected": None,
            "text": None,
        }
        self.rows.append(row)
        if not complete:
            return None

        gc.collect()  # while the backend classifies; collection is off in a block
        result = self.wait_for_answer(info["trial"])
        if result is None:
            if not self.window.has_exit:
                self.link.send(
                    pylsl.local_clock(), "no_answer", trial=info["trial"], block=block
                )
            return None
        answer, row["answer"] = result
        row["selected"], row["text"] = answer["selected"], answer["text"]

        self.scene.set_text(answer["phrase"], answer["text"], answer["message"])
        self.scene.set_feedback(answer["selected"])
        run_frames(
            self.window,
            self.scene,
            self.n_feedback,
            False,
            lambda t: self.link.send(
                t,
                "feedback",
                trial=info["trial"],
                block=block,
                selected=answer["selected"],
                text=answer["text"],
            ),
        )
        self.scene.set_feedback(None)
        self.scene.show_state(self.state)  # the answer, or a state sent after it
        return answer

    def run_block(self, block: int) -> str:
        """Trials until a pause -> text for the pause screen ("" after ESC)."""
        self.link.send(pylsl.local_clock(), "block_start", block=block)
        self.scene.set_message(self.state["message"])
        self.space = False
        gc.collect()
        gc.disable()  # a garbage-collection pause can cost a frame
        try:
            while True:
                answer = self.run_trial(block)
                if self.window.has_exit:
                    self.aborted = True
                    return ""
                if answer is None:
                    message = (
                        f"No answer from the backend for trial {len(self.rows) - 1}."
                    )
                    break
                if answer["pause"]:
                    message = answer["message"] or "Paused."
                    break
                if self.space:
                    message = "Paused."
                    break
        finally:
            gc.enable()
        self.link.send(pylsl.local_clock(), "block_end", block=block)
        return message

    def run(self):
        """The whole session; closes the window and saves the trial table."""
        flips = self.idle("Online speller\n\nSPACE to start, ESC to quit")
        if self.window.has_exit:
            self.window.close()
            return
        check_refresh_rate(
            self.window, flips, self.args.refresh_rate, not self.args.windowed
        )
        self.link.send(
            pylsl.local_clock(),
            "session_start",
            mode="online",
            subject=self.args.subject,
            demo=self.args.demo,
            freqs=self.scene.freqs,
            phases=self.scene.phases,
            refresh_rate=self.scene.refresh_rate,
            stim_s=self.args.stim,
            gaze_s=self.args.gaze,
            feedback_s=self.args.feedback,
            slots=self.scene.slots,
            screen_px=self.scene.screen,
            centres_px=self.scene.centres.round(1).tolist(),
            size_px=round(self.scene.size, 1),
        )
        block = 0
        while True:
            message = self.run_block(block)
            if self.window.has_exit:
                break
            block += 1
            self.idle(f"{message}\n\nSPACE to continue, ESC to quit")
            if self.window.has_exit:
                break
        self.link.send(
            pylsl.local_clock(),
            "session_end",
            n_trials=len(self.rows),
            aborted=self.aborted,
        )
        self.window.close()
        time.sleep(1.0)  # give the recorder time to receive the last markers
        self.save()

    def save(self):
        """Print a summary and write the trial table to data/recordings/."""
        if not self.rows:
            return
        trials = pd.DataFrame(self.rows)
        answered = trials["answer"].notna()
        delay = 1000 * (trials["answer"] - trials["stim_off"])
        print(
            f"{len(trials)} trials, {answered.sum()} answered, "
            f"{trials['n_dropped'].sum()} dropped frames in "
            f"{(trials['n_dropped'] > 0).sum()} trials"
        )
        if answered.any():
            print(
                f"Answer after stim_off: median {delay.median():.0f} ms, "
                f"max {delay.max():.0f} ms"
            )
        config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RECORDINGS_DIR / (
            f"{self.args.subject}_{datetime.now():%Y%m%d_%H%M%S}_online.csv"
        )
        trials.to_csv(path, index=False)
        print(f"Trial table saved to {path}")


def find_backend() -> pylsl.StreamInlet | None:
    """Wait for the backend's state stream and open an inlet -> None on Ctrl+C."""
    print(f"Waiting for the backend's LSL stream '{config.STATE_STREAM}'")
    print("(start it now, or Ctrl+C to quit)...")
    try:
        while True:
            try:
                info = acquisition.find_stream("Markers", config.STATE_STREAM, 1.0)
                break
            except TimeoutError:
                pass
    except KeyboardInterrupt:
        return None
    inlet = acquisition.open_marker_inlet(info)
    print("Backend found.")
    return inlet


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", default="pilot", help="subject ID for the logs")
    parser.add_argument(
        "--stim",
        type=float,
        default=config.ONLINE_STIM_DURATION,
        help="flicker duration in s",
    )
    parser.add_argument(
        "--gaze",
        type=float,
        default=config.GAZE_SHIFT,
        help="s with the new labels before the flicker",
    )
    parser.add_argument(
        "--feedback",
        type=float,
        default=config.FEEDBACK_DURATION,
        help="s with the selected target framed",
    )
    parser.add_argument(
        "--demo", action="store_true", help="stand-in backend instead of the real one"
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

    # The outlet goes first, so a backend waiting for the markers can find it
    outlet = open_marker_outlet()
    if args.demo:
        link = DemoLink(outlet, config.TARGET_FREQS)
    else:
        inlet = find_backend()
        if inlet is None:
            return
        link = LslLink(outlet, inlet, len(config.TARGET_FREQS))

    window = open_window(args.screen, args.windowed, caption="SSVEP speller")
    scene = SpellerScene(
        window,
        config.TARGET_FREQS,
        config.TARGET_PHASES,
        config.TARGET_SLOTS,
        args.refresh_rate,
    )
    Session(window, scene, link, args).run()


if __name__ == "__main__":
    main()
