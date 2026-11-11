"""Online backend: classifies every speller trial and drives the speller GUI.

Usage:
    .venv/bin/python scripts/online.py --method FBCCA-norm --calibration data/recordings/S01_*.npz
    .venv/bin/python scripts/online.py --method CCA --calibration data/recordings/e2e_*.npz --phrases "HELLO WORLD" --cue
    .venv/bin/python scripts/online.py --method FBCCA-norm --calibration A.npz --replay B.npz
    .venv/bin/python scripts/online.py --method FBCCA-norm --calibration A.npz --nlp gpt2

Start it first, then scripts/speller_gui.py and scripts/record.py (and
scripts/fake_eeg_stream.py instead of the cap). Ctrl+C stops it and saves the decision
log to data/recordings/<subject>_<date>_<nlp>_backend.csv (one row per answered trial).

--nlp picks the word suggestions (ssvep/language.py): none (the without-NLP condition),
unigram, or gpt2. The language model is loaded and warmed up before the GUI connects.

--replay feeds a recording through the same code without LSL or the GUI: the EEG in
small chunks, as if it arrived live, and every marker once the EEG has reached it. It
prints the accuracy against the recorded targets. Train and replay on different
recordings: a trained method scores its own training trials too well.

One trial (markers and states: ssvep/protocol.py):
    stim_on      the trial waits for its answer; its onset is the marker's LSL time
    stim_off     complete: the trial will be answered; incomplete (ESC): forget it
    EEG arrives  once the ring buffer holds the trial's epoch
                 [onset - PRE, onset + latency + window + POST]:
                 notch -> band-pass or filter bank -> crop [PRE + latency, + window]
                 -> classifier -> speller.select -> answer (trial, selected, new labels)
Epochs and processing are those of the offline comparison (datasets.load_recording,
scripts/compare_methods.py), so online the classifier sees what it was trained and
evaluated on.
"""

import argparse
from functools import partial
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pylsl

from compare_methods import METHODS, prefilter, with_rejection
from ssvep import acquisition, config, language, protocol
from ssvep.datasets import load_recording, recording_freqs
from ssvep.preprocessing import crop, notch
from ssvep.speller import Speller

PRE = config.BENCHMARK_ONSET  # s of EEG before the onset, as in load_recording's epochs
POST = 0.5  # s of EEG after the window, so the filters' edge effects stay outside it
BUFFER_SECONDS = 30.0  # s of EEG kept in the ring buffer
LOOP_SLEEP = 0.005  # s between two passes of the main loop


def cap_indices(labels: list[str]) -> list[int]:
    """Column of every config.CAP_CHANNELS channel in a stream whose channels are
    ``labels``, in CAP_CHANNELS order.

    Raises ValueError naming the CAP_CHANNELS that are not in ``labels``.
    """
    indices = []
    for c in config.CAP_CHANNELS:
        try:
            indices.append(labels.index(c))
        except ValueError:
            raise ValueError(f"CAP channel {c} is not in the list of labels")
    return indices


def load_calibration(
    paths: list[Path], freqs: list[float] = config.TARGET_FREQS
) -> tuple[np.ndarray, np.ndarray]:
    """Raw epochs and labels (X, y) of one or more calibration recordings, in order.

    X : (n_trials, len(CAP_CHANNELS), n_samples), as load_recording cuts them. Epochs
    of different lengths (recordings with different flicker durations) are all cut to
    the shortest. Raises ValueError if a recording was made with frequencies other than
    ``freqs``: its labels would point at other targets.
    """
    parts_X = []
    parts_y = []
    for path in paths:
        rec_freqs = recording_freqs(path)
        if rec_freqs != freqs:
            raise ValueError(
                f"Calibration recording {path} has frequencies {rec_freqs}, "
                f"not the expected {freqs}"
            )

        X_i, y_i, _ = load_recording(path)
        parts_X.append(X_i)
        parts_y.append(y_i)

    # Every epoch starts PRE s before its onset, so cutting the tail keeps them aligned
    n_samples = min(X_i.shape[-1] for X_i in parts_X)
    X = np.concatenate([X_i[..., :n_samples] for X_i in parts_X], axis=0)
    return X, np.concatenate(parts_y, axis=0)


class Decoder:
    """Preprocessing and classifier for one trial at a time: the steps of the offline
    comparison (scripts/compare_methods.py), for one method and one window.

    An epoch is raw EEG (n_channels, n_samples), the CAP_CHANNELS in µV, that starts
    PRE s before the onset sample (the first sample at or after the stim_on time), like
    load_recording's epochs.
    """

    def __init__(
        self,
        method: str,
        window: float = config.ONLINE_STIM_DURATION,
        latency: float = config.BENCHMARK_LATENCY,
        freqs: list[float] = config.TARGET_FREQS,
        fs: float = config.EEG_FS,
        trial_ratio: float = config.MAX_TRIAL_PTP_RATIO,
        channel_ratio: float = config.MAX_CHANNEL_SD_RATIO,
    ):
        """``method`` is a key of compare_methods.METHODS. ``trial_ratio`` and
        ``channel_ratio``: artifact rejection while fitting, as in compare_methods
        (0 switches that step off)."""
        self.method = method
        self.window = window
        self.latency = latency
        self.freqs = freqs
        self.fs = fs
        self.trial_ratio = trial_ratio
        self.channel_ratio = channel_ratio
        self.n_subbands, self.factory = METHODS[method]
        self.estimator_ = None  # the fitted classifier, set by fit

    @property
    def n_samples(self) -> int:
        """Epoch length: round((PRE + latency + window + POST) * fs) samples."""
        return int(round((PRE + self.latency + self.window + POST) * self.fs))

    def prepare(self, epochs: np.ndarray) -> np.ndarray:
        """Raw epochs (n_trials, n_channels, n_samples) -> classifier input
        (n_trials, n_bands, n_channels, window samples).

        Notch at config.LINE_FREQ, then band-pass or filter bank
        (compare_methods.prefilter with this method's n_subbands), then crop ``window``
        s starting PRE + latency s into the epoch.
        """
        notched = notch(epochs, self.fs, config.LINE_FREQ)
        prefiltered = prefilter(notched, self.fs, self.n_subbands)
        return crop(prefiltered, self.fs, PRE + self.latency, self.window)

    def fit(self, epochs: np.ndarray, y: np.ndarray) -> "Decoder":
        """Train on calibration epochs (load_calibration) -> self.

        Epochs longer than n_samples are first cut to n_samples, so the filters see
        exactly the length they see online. The classifier, stored in ``estimator_``,
        is a fresh METHODS[method] estimator wrapped in RejectArtifacts
        (compare_methods.with_rejection). Raises ValueError if the epochs are shorter
        than n_samples.
        """
        if epochs.shape[-1] < self.n_samples:
            raise ValueError(
                f"Calibration epochs have {epochs.shape[-1]} samples, "
                f"less than the {self.n_samples} needed for this window"
            )
        epochs = epochs[..., : self.n_samples]
        epochs = self.prepare(epochs)
        make_estimator = partial(self.factory, self.freqs, self.fs)
        self.estimator_ = with_rejection(
            make_estimator, self.trial_ratio, self.channel_ratio
        ).fit(epochs, y)
        return self

    def epoch(self, buffer: acquisition.RingBuffer, onset: float) -> np.ndarray | None:
        """The raw epoch of the trial whose stim_on came at LSL time ``onset`` ->
        (n_channels, n_samples), or None while its last sample hasn't arrived."""
        start = onset - PRE
        return buffer.window(start, self.n_samples)

    def classify(self, epoch: np.ndarray) -> tuple[int, np.ndarray]:
        """One raw epoch (n_channels, n_samples) -> (selected target, scores).

        ``scores`` (n_targets,) is the classifier's decision_function for this trial,
        and the selected target is the one with the highest score.
        """
        epoch = self.prepare(epoch[None, ...])  # add the trial axis
        scores = self.estimator_.decision_function(epoch)[0]  # remove the trial axis
        return int(np.argmax(scores)), scores


class Backend:
    """The speller behind the GUI protocol (ssvep/protocol.py), without any LSL.

    on_marker and poll return the JSON messages to push to the GUI, so main, replay and
    the tests all drive it the same way.
    """

    def __init__(
        self,
        decoder: Decoder,
        speller: Speller,
        buffer: acquisition.RingBuffer,
        phrases: list[str] = (),
        cue: bool = False,
    ):
        """``phrases``: copy-spelling phrases in order; none means free spelling.
        ``cue``: ask the GUI to frame the next target before each flicker."""
        self.decoder = decoder
        self.speller = speller
        self.buffer = buffer
        self.phrases = list(phrases)  # copy phrases still to come
        self.cue = cue
        self.pending = {}  # trial -> its marker fields, while it waits for an answer
        self.rows = []  # one dict per answered trial, the decision log
        self.next_phrase()

    def next_phrase(self) -> None:
        """Start copying the next phrase (taken off ``phrases``), or free spelling when
        none is left."""
        if self.phrases:
            self.speller.set_phrase(self.phrases.pop(0))
        else:
            self.speller.set_phrase("")

    def state(self, **fields) -> str:
        """JSON state of the speller as it is now (protocol.state_message).

        Its labels, text and phrase, its next_target() as ``target``, and ``cue``;
        ``fields`` (e.g. trial, selected, pause, message) are passed on.
        """
        state = dict(
            labels=self.speller.labels(),
            text=self.speller.text,
            phrase=self.speller.phrase,
            target=self.speller.next_target(),
            cue=self.cue,
        )
        state.update(fields)
        return protocol.state_message(**state)

    def on_marker(self, marker: str, timestamp: float) -> list[str]:
        """React to one GUI marker (JSON string, LSL time) -> messages to push.

        state_request: the current state. stim_on: the trial now waits for its answer,
        with onset ``timestamp``. stim_off: if complete, the trial may be answered; if
        not, it is forgotten. Other events: nothing.
        """
        fields = json.loads(marker)
        event = fields["event"]
        if event == "state_request":
            return [self.state()]
        if event == "stim_on":
            self.pending[fields["trial"]] = {
                **fields,
                "onset": timestamp,
                "complete": False,
            }
        elif event == "stim_off":
            trial = fields["trial"]
            if not fields["complete"]:
                self.pending.pop(trial, None)
            elif trial in self.pending:
                self.pending[trial]["complete"] = True
        return []

    def poll(self) -> list[str]:
        """Answer every trial whose complete stim_off has come and whose epoch is in
        the buffer -> messages to push.

        For each such trial: classify its epoch, pass the decision to speller.select,
        add a row to ``rows``, and answer with a state whose ``trial`` is that trial
        and ``selected`` the decision. If that selection finishes the phrase, the
        answer has ``pause`` true and is followed by a state for the next phrase
        (next_phrase). Every trial is answered at most once. A trial whose epoch has
        already left the buffer is lost: it is answered without a decision (selected
        None, its row's scores NaN), because the GUI waits for an answer.
        A row holds at least: trial, block, target (from stim_on, the ground truth
        for the log; never used to classify), selected, item (what speller.select
        returned), text, classify_ms, select_ms (speller.select, which runs the word
        suggestions when the text changes), and score_0 ... score_<n_targets - 1>.
        """
        messages = []
        for trial in sorted(self.pending):  # a separate list: trials are deleted below
            info = self.pending[trial]
            if not info["complete"]:
                continue

            try:
                epoch = self.decoder.epoch(self.buffer, info["onset"])
            except ValueError:  # that EEG has left the ring buffer
                selected = None
                scores = np.full(len(self.decoder.freqs), np.nan)
                classify_ms = np.nan
                message = "The EEG of this trial was lost"
            else:
                if epoch is None:
                    continue  # its last sample hasn't arrived yet
                start = time.perf_counter()
                selected, scores = self.decoder.classify(epoch)
                classify_ms = 1000 * (time.perf_counter() - start)
                message = ""

            start = time.perf_counter()
            item = self.speller.select(selected)
            select_ms = 1000 * (time.perf_counter() - start)
            del self.pending[trial]

            row = {
                "trial": trial,
                "block": info["block"],
                "target": info["target"],
                "selected": selected,
                "item": item,
                "text": self.speller.text,
                "classify_ms": classify_ms,
                "select_ms": select_ms,
            }
            for k, score in enumerate(scores):
                row[f"score_{k}"] = float(score)
            self.rows.append(row)

            # Built after select, so the answer shows the new labels and next target
            if self.speller.done:
                messages.append(
                    self.state(
                        trial=trial,
                        selected=selected,
                        pause=True,
                        message="Phrase done",
                    )
                )
                self.next_phrase()  # clears the text, so after the answer is built
                messages.append(self.state())
            else:
                messages.append(
                    self.state(trial=trial, selected=selected, message=message)
                )
        return messages


def replay(path: Path, backend: Backend, chunk: float = 0.02) -> list[str]:
    """Feed a recording through ``backend`` as if it arrived live -> every message the
    backend produced, in order.

    The CAP_CHANNELS columns of the EEG go into backend.buffer in chunks of ``chunk``
    s. After each chunk, every marker up to the time of the chunk's last sample goes
    to on_marker, then poll runs.
    """
    with np.load(path) as npz:
        eeg = np.asarray(npz["eeg"])  # (channels, samples)
        times = np.asarray(npz["eeg_times"])
        markers = np.asarray(npz["markers"])
        marker_times = np.asarray(npz["marker_times"])
        channels = [str(c) for c in np.asarray(npz["channels"])]
        fs = float(npz["fs"])

    samples = eeg[cap_indices(channels)].T  # (samples, channels), as LSL delivers it
    order = np.argsort(marker_times, kind="stable")  # recorded in arrival order
    markers, marker_times = markers[order], marker_times[order]
    step = max(1, int(round(chunk * fs)))

    messages = []
    next_marker = 0
    for start in range(0, len(samples), step):
        stop = min(start + step, len(samples))
        backend.buffer.append(samples[start:stop], times[start:stop])
        while (
            next_marker < len(markers) and marker_times[next_marker] <= times[stop - 1]
        ):
            messages += backend.on_marker(
                str(markers[next_marker]), float(marker_times[next_marker])
            )
            next_marker += 1
        messages += backend.poll()
    return messages


def wait_for_stream(stream_type: str, name: str | None = None) -> pylsl.StreamInfo:
    """The first matching LSL stream, however long it takes to appear."""
    print(f"Waiting for LSL stream {name or stream_type}...")
    while True:
        try:
            return acquisition.find_stream(stream_type, name, timeout=1.0)
        except TimeoutError:
            pass


def pump(backend: Backend, eeg_inlet, marker_inlet, outlet, cap: list[int]) -> None:
    """One pass of the online loop, without waiting: the EEG that has arrived goes into
    the buffer (its ``cap`` columns), the markers to on_marker, then poll runs. Every
    message goes out on ``outlet``.

    The inlets only need ``pull_chunk(timeout=...)`` and the outlet ``push_sample``, so
    tests can pass stand-ins.
    """
    samples, timestamps = eeg_inlet.pull_chunk(timeout=0.0)
    if samples:
        backend.buffer.append(np.asarray(samples)[:, cap], timestamps)

    messages = []
    markers, timestamps = marker_inlet.pull_chunk(timeout=0.0)
    for sample, timestamp in zip(markers, timestamps):
        messages += backend.on_marker(sample[0], timestamp)
    messages += backend.poll()

    for message in messages:
        outlet.push_sample([message])
        state = json.loads(message)
        if state["trial"] is not None:
            print(
                f"Trial {state['trial']}: selected {state['selected']}, "
                f"text {state['text']!r}"
            )


def serve(backend: Backend, eeg_type: str, eeg_name: str | None) -> None:
    """Serve the GUI over LSL until Ctrl+C."""
    # The GUI waits for this stream before it opens its window, so it comes first
    outlet = pylsl.StreamOutlet(
        pylsl.StreamInfo(
            config.STATE_STREAM,
            "Markers",
            1,
            pylsl.IRREGULAR_RATE,
            pylsl.cf_string,
            "ssvep-backend",
        )
    )
    marker_inlet = acquisition.open_marker_inlet(
        wait_for_stream("Markers", config.MARKER_STREAM)
    )
    eeg_inlet = acquisition.open_eeg_inlet(wait_for_stream(eeg_type, eeg_name))
    # The resolved info has no description, the inlet's has the labels
    cap = cap_indices(acquisition.channel_labels(eeg_inlet.info()))
    print("Running. Ctrl+C stops the backend and saves the decision log.")
    while True:
        pump(backend, eeg_inlet, marker_inlet, outlet, cap)
        time.sleep(LOOP_SLEEP)


def print_accuracy(rows: list[dict]) -> None:
    """Accuracy of the answered trials that have a recorded target."""
    scored = [row for row in rows if row["target"] is not None]
    if not scored:
        print("No answered trial has a recorded target: no accuracy.")
        return
    n_correct = sum(row["selected"] == row["target"] for row in scored)
    print(f"Accuracy: {n_correct}/{len(scored)} = {n_correct / len(scored):.1%}")


def save_log(rows: list[dict], path: Path) -> None:
    """Write the decision log as a CSV file, one row per answered trial."""
    if not rows:
        print("No answered trials: no decision log to save.")
        return
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved {len(rows)} decisions to {path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--subject", default="pilot", help="subject ID for the log")
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        required=True,
        help="calibration recording(s) to train on",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=config.ONLINE_STIM_DURATION,
        help="classifier window in s (at most the GUI's --stim)",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=config.BENCHMARK_LATENCY,
        help="visual latency in s: the window starts this long after the onset",
    )
    parser.add_argument(
        "--phrases",
        nargs="*",
        default=[],
        help="phrases to copy, in order (none: free spelling)",
    )
    parser.add_argument(
        "--cue", action="store_true", help="frame the next target before each trial"
    )
    parser.add_argument(
        "--nlp",
        choices=language.KINDS,
        default="none",
        help="word suggestions: none (the without-NLP condition), unigram or gpt2",
    )
    parser.add_argument("--eeg-type", default="EEG", help="LSL type of the EEG stream")
    parser.add_argument("--eeg-name", default=None, help="LSL name, if several match")
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="replay this recording instead of going online",
    )
    parser.add_argument(
        "--trial-ratio",
        type=float,
        default=config.MAX_TRIAL_PTP_RATIO,
        help="artifact rejection while training, as in compare_methods.py",
    )
    parser.add_argument(
        "--channel-ratio",
        type=float,
        default=config.MAX_CHANNEL_SD_RATIO,
        help="artifact rejection while training, as in compare_methods.py",
    )
    args = parser.parse_args(argv)
    if args.window > config.ONLINE_STIM_DURATION:
        print(
            f"Note: --window {args.window:g} s is longer than the default flicker "
            f"({config.ONLINE_STIM_DURATION:g} s); start the GUI with --stim >= it."
        )
    return args


def main(argv=None):
    """Train the decoder on the calibration recordings, then either replay a recording
    (print the accuracy) or serve the GUI over LSL until Ctrl+C. Both save the
    decision log (Backend.rows)."""

    args = parse_args(argv)
    if args.replay is not None:
        replay_freqs = recording_freqs(args.replay)
        if replay_freqs != config.TARGET_FREQS:
            raise ValueError(
                f"{args.replay} was recorded with frequencies {replay_freqs}, not "
                f"{config.TARGET_FREQS}: its targets would point at other frequencies"
            )

    # Before the session, so a missing folder can't lose its log
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    # The condition in the name, so the analysis can't mix up with and without NLP
    log_path = config.RECORDINGS_DIR / f"{args.subject}_{stamp}_{args.nlp}_backend.csv"

    print(f"Training {args.method} on {args.calibration} ...")
    X, y = load_calibration(args.calibration)
    decoder = Decoder(
        args.method,
        window=args.window,
        latency=args.latency,
        trial_ratio=args.trial_ratio,
        channel_ratio=args.channel_ratio,
    ).fit(X, y)
    fitted = decoder.estimator_
    dropped = [
        channel
        for i, channel in enumerate(config.CAP_CHANNELS)
        if i not in fitted.channels_
    ]
    print(
        f"Trained on {fitted.trials_kept_.sum()} of {len(y)} trials; "
        f"channels dropped: {', '.join(dropped) or 'none'}"
    )
    suggest = None
    if args.nlp != "none":
        print(f"Loading the {args.nlp} word suggestions ...")
        suggest = language.make_suggester(args.nlp)
        # Warm-up: the first call is slower; better now than in the first trial
        start = time.perf_counter()
        suggest("HELLO WOR")
        print(
            f"Word suggestions ready (warm-up {1000 * (time.perf_counter() - start):.0f} ms)"
        )
    speller = Speller(suggest=suggest)
    buffer = acquisition.RingBuffer(
        len(config.CAP_CHANNELS), int(BUFFER_SECONDS * config.EEG_FS)
    )
    backend = Backend(decoder, speller, buffer, phrases=args.phrases, cue=args.cue)

    try:
        if args.replay is None:
            serve(backend, args.eeg_type, args.eeg_name)
        else:
            replay(args.replay, backend)
            print_accuracy(backend.rows)
    except KeyboardInterrupt:
        print("Stopping on Ctrl+C...")
    finally:
        save_log(backend.rows, log_path)


if __name__ == "__main__":
    main()
