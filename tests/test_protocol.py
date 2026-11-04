"""Tests for ssvep.protocol (speller GUI <-> backend messages)."""

import json

import numpy as np
import pytest

from ssvep.protocol import STATE_DEFAULTS, parse_state, state_message, validate_state

N = 6
LABELS = ["A", "B", "C", "D", "E", "F"]


def message(**fields):
    return json.dumps({"event": "state", "labels": LABELS, **fields})


def test_minimal_state_gets_all_defaults():
    """Only event + labels: every other field takes its default."""
    state = parse_state(message(), N)
    assert state == {"event": "state", "labels": LABELS, **STATE_DEFAULTS}


def test_full_state_round_trips():
    fields = {
        "labels": [list("ABCDEF"), "G", "H", "I", ["␣", "⌫", "x", "y", "z", "."], ""],
        "text": "HELLO WOR",
        "phrase": "HELLO WORLD",
        "target": 4,
        "cue": True,
        "message": "Look at L",
        "trial": 17,
        "selected": 0,
        "pause": False,
    }
    assert parse_state(json.dumps({"event": "state", **fields}), N) == {
        "event": "state",
        **fields,
    }


@pytest.mark.parametrize(
    "fields",
    [
        {"labels": LABELS[:5]},  # one label short
        {"labels": "ABCDEF"},  # not a list
        {"labels": [*LABELS[:5], 7]},  # not a string
        {"labels": [list("ABCDE"), *LABELS[1:]]},  # nested list one short
        {"labels": [[*"ABCDE", 1], *LABELS[1:]]},  # nested non-string
        {"selected": 6},
        {"selected": -1},
        {"selected": True},  # bool is not an index
        {"selected": 2.0},
        {"target": 6},
        {"trial": -1},
        {"trial": "3"},
        {"cue": 1},
        {"pause": "yes"},
        {"text": None},
        {"phrase": 5},
        {"message": ["hi"]},
        {"lables": LABELS},  # typo: unknown field
    ],
)
def test_invalid_states_raise(fields):
    state = {"event": "state", "labels": LABELS, **fields}
    with pytest.raises(ValueError):
        validate_state(state, N)


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        json.dumps([1, 2, 3]),
        json.dumps({"labels": LABELS}),  # no event
        json.dumps({"event": "stim_on", "labels": LABELS}),
        json.dumps({"event": "state"}),  # no labels
    ],
)
def test_parse_state_rejects_bad_messages(text):
    """Invalid JSON, a non-object, a wrong or missing event, or no labels: ValueError."""
    with pytest.raises(ValueError):
        parse_state(text, N)


def test_state_message_round_trip_and_defaults():
    text = state_message(n_targets=N, labels=LABELS, text="HI", trial=3, selected=1)
    state = parse_state(text, N)
    assert state["text"] == "HI" and state["trial"] == 3 and state["selected"] == 1
    assert state["phrase"] == "" and state["cue"] is False and state["target"] is None


def test_state_message_accepts_numpy_values():
    """Classifier outputs are NumPy ints and label arrays; they become plain JSON."""
    labels = np.array(LABELS)
    text = state_message(
        n_targets=N, labels=labels, selected=np.int64(2), trial=np.int32(5)
    )
    state = parse_state(text, N)
    assert state["labels"] == LABELS
    assert state["selected"] == 2 and state["trial"] == 5


def test_state_message_rejects_invalid_state():
    with pytest.raises(ValueError):
        state_message(n_targets=N, labels=LABELS, selected=9)
