"""JSON messages between the speller GUI (scripts/speller_gui.py) and the online backend.

Both directions are LSL string streams (one channel, irregular rate) carrying one JSON
object per sample. Target indices are indices into config.TARGET_FREQS everywhere, like
the classifier's labels; the GUI draws target k in ring slot config.TARGET_SLOTS[k].

GUI -> backend: the markers on config.MARKER_STREAM, the same stream as in calibration,
so scripts/record.py and scripts/fake_eeg_stream.py work unchanged. Each marker is
timestamped with the LSL time of the flip that showed the event:

    session_start  mode ("online"), subject, freqs, phases, refresh_rate, stim_s,
                   gaze_s, feedback_s, slots, screen_px, centres_px, size_px
    state_request  the GUI has no state yet (repeated every 0.5 s until one arrives)
    block_start    block: the operator started or resumed spelling
    cue            trial, block, target, freq: red frame shown (only if the state asks)
    stim_on        trial, block, target, freq: first flicker flip
    stim_off       trial, block, target, freq, n_dropped, complete
    feedback       trial, block, selected, text: the answer is on screen
    no_answer      trial, block: no answer in time, the GUI has paused
    block_end      block: paused by the operator or the backend
    session_end    n_trials, aborted (ESC during a block)

``target`` and ``freq`` repeat the state's ``target`` (null if it has none). They are
ground truth for logging, offline analysis and the fake EEG stream; the classifier
must not use them.

Backend -> GUI: config.STATE_STREAM. Each state describes the whole view; fields left
out take the values in STATE_DEFAULTS, only ``labels`` is required:

    labels    one entry per target: a string drawn on the square, or a list of one
              string per target, drawn as a small ring on the square with entry k where
              target k sits on the screen (a preview of the next menu level). Both
              levels are in target order, not screen order: with TARGET_SLOTS
              [0, 3, 1, 4, 2, 5], ["A", "B", "C", "D", "E", "F"] reads A, C, E, B, D, F
              clockwise from the top left.
    text      typed text so far
    phrase    phrase to copy ("" in free spelling)
    target    target the subject should select next, or null
    cue       draw a red frame around ``target`` before the flicker
    message   short line under the text
    trial     the trial this state answers, null if it answers none
    selected  target the classifier chose in ``trial``, null for no decision
    pause     (answers only) stop after this trial until the operator presses SPACE

The backend answers every complete trial (stim_off with complete true) with a state
whose ``trial`` is that trial, also when it makes no decision. The GUI waits a few
seconds for it, then pauses. state_request is answered with the current state (trial
null). The GUI opens its marker outlet before it waits for the state stream, so the
backend must open its state outlet before it waits for the marker stream.

With an answer the GUI frames ``selected`` in green and shows the answer's text,
phrase and message for config.FEEDBACK_DURATION s, then the labels of the newest state
received. The next trial cues and logs that state's ``target``.
"""

import json

import numpy as np

from ssvep import config

STATE_DEFAULTS = {
    "text": "",
    "phrase": "",
    "target": None,
    "cue": False,
    "message": "",
    "trial": None,
    "selected": None,
    "pause": False,
}


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def validate_state(state: dict, n_targets: int = len(config.TARGET_FREQS)) -> dict:
    """Check a decoded state and fill in the defaults -> new dict with every field.

    Raises ValueError naming the first problem.
    """
    if not isinstance(state, dict):
        raise ValueError(f"A state must be a JSON object: {state!r}")
    if state.get("event") != "state":
        raise ValueError(f'"event" must be "state": {state.get("event")!r}')
    unknown = set(state) - set(STATE_DEFAULTS) - {"event", "labels"}
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")
    if "labels" not in state:
        raise ValueError('"labels" is missing')
    state = {**STATE_DEFAULTS, **state}

    labels = state["labels"]
    if not isinstance(labels, list) or len(labels) != n_targets:
        raise ValueError(f'"labels" must be a list of {n_targets} entries: {labels!r}')
    for entry in labels:
        if isinstance(entry, list):
            if len(entry) != n_targets or not all(isinstance(s, str) for s in entry):
                raise ValueError(
                    f"A nested label must be {n_targets} strings: {entry!r}"
                )
        elif not isinstance(entry, str):
            raise ValueError(
                f"A label must be a string or a list of strings: {entry!r}"
            )
    for field in ("text", "phrase", "message"):
        if not isinstance(state[field], str):
            raise ValueError(f'"{field}" must be a string: {state[field]!r}')
    for field in ("cue", "pause"):
        if not isinstance(state[field], bool):
            raise ValueError(f'"{field}" must be true or false: {state[field]!r}')
    for field in ("target", "selected"):
        value = state[field]
        if value is not None and not (_is_int(value) and 0 <= value < n_targets):
            raise ValueError(f'"{field}" must be null or 0..{n_targets - 1}: {value!r}')
    trial = state["trial"]
    if trial is not None and not (_is_int(trial) and trial >= 0):
        raise ValueError(f'"trial" must be null or an index >= 0: {trial!r}')
    return state


def parse_state(message: str, n_targets: int = len(config.TARGET_FREQS)) -> dict:
    """One JSON string from the state stream -> validated state with every field.

    Raises ValueError for invalid JSON or an invalid state.
    """
    return validate_state(json.loads(message), n_targets)


def _to_json(value):
    """json.dumps fallback for NumPy numbers and arrays (e.g. a classifier output)."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def state_message(n_targets: int = len(config.TARGET_FREQS), **fields) -> str:
    """JSON string for one state, for the backend to push: validated, defaults filled.

    NumPy numbers and arrays are converted to plain JSON. Raises ValueError for an
    invalid state.
    """
    plain = json.loads(json.dumps({"event": "state", **fields}, default=_to_json))
    return json.dumps(validate_state(plain, n_targets))
