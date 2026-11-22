"""Tests for scripts/analyse_online.py: results of the online runs and their replays."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssvep import acquisition, config
from ssvep.evaluation import itr
from ssvep.speller import Speller

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import analyse_online  # noqa: E402
import online  # noqa: E402

T_SEL = 3.7  # s between two onsets in the synthetic runs below


def test_common_prefix():
    assert analyse_online.common_prefix("HELLMLU", "HELLO") == 4
    assert analyse_online.common_prefix("", "HI") == 0
    assert analyse_online.common_prefix("HI ", "HI") == 2


def test_phrase_index_moves_on_after_each_finished_phrase():
    """A phrase ends with the decision that makes the text equal to it (a trailing
    space allowed); after the last phrase comes free spelling (-1)."""
    texts = ["", "H", "HI", "", "A", "B", "BX"]
    assert analyse_online.phrase_index(texts, ["hi", "a b"]) == [0, 0, 0, 1, 1, 1, 1]
    texts = ["", "H", "HI ", "", "A", "", "X"]
    assert analyse_online.phrase_index(texts, [" hi", "a"]) == [0, 0, 0, 1, 1, -1, -1]


def write_run(tmp_path, rows, name="r"):
    """GUI trial table + backend decision log of one run from rows of (target,
    selected, item, text); trial k starts at k * T_SEL s."""
    gui, backend = [], []
    for k, (target, selected, item, text) in enumerate(rows):
        onset = 100.0 + k * T_SEL
        gui.append(
            {
                "trial": k,
                "block": 0,
                "target": target,
                "freq": 8.0,
                "cued": target is not None,
                "stim_on": onset,
                "stim_off": onset + 2.0,
                "n_dropped": int(k == 1),
                "complete": True,
                "answer": onset + 2.7,
                "selected": selected,
                "text": text,
            }
        )
        backend.append(
            {
                "trial": k,
                "block": 0,
                "target": target,
                "selected": selected,
                "item": item,
                "text": text,
                "classify_ms": 10.0 + k,
                "select_ms": 30.0 if item else 0.001,
            }
        )
    pd.DataFrame(gui).to_csv(tmp_path / f"{name}_online.csv", index=False)
    pd.DataFrame(backend).to_csv(tmp_path / f"{name}_backend.csv", index=False)
    return {
        "run": name,
        "online": tmp_path / f"{name}_online.csv",
        "backend": tmp_path / f"{name}_backend.csv",
        "recording": None,
        "calibration": None,
        "method": "CCA",
        "window": 2.0,
        "latency": 0.7,
        "nlp": "gpt2",
        "fusion": False,
        "phrases": ["hi", "ac"],
        "include": True,
    }


# Copy "HI", then "AC" unfinished ("AB" typed).
# Targets and selections are target indices; None = no ground truth.
ROWS = [
    (1, 1, "", ""),  # open group F-J
    (2, 2, "H", "H"),  # H
    (1, 1, "", "H"),  # open group F-J
    (3, 2, "H", "HH"),  # wrong: H instead of I
    (5, 5, "", "HH"),  # open the word group
    (0, 3, "⌫", "H"),  # wrong target, but it corrects anyway (a lucky error)
    (5, 5, "", "H"),
    (0, 0, "#1", "HI "),  # accepts the word HI: phrase 1 done
    (0, 0, "", ""),  # phrase 2: open A-E
    (0, 0, "A", "A"),
    (5, 5, "", "A"),
    (1, 1, "#2", "A"),  # an empty word cell: no word accepted, text unchanged
    (0, 0, "", "A"),
    (None, 1, "B", "AB"),  # a trial without ground truth (target None)
]


def test_summarize_run_counts_accuracy_time_and_characters(tmp_path):
    run = write_run(tmp_path, ROWS)
    row, phrases = analyse_online.summarize_run(run)
    assert (row["decisions"], row["cued"], row["correct"]) == (14, 13, 11)
    assert row["accuracy"] == pytest.approx(11 / 13)
    assert row["t_sel_s"] == pytest.approx(T_SEL)
    assert row["itr"] == pytest.approx(itr(6, 11 / 13, T_SEL))
    assert row["answer_delay_s"] == pytest.approx(0.7)
    assert row["dropped_frames"] == 1
    assert row["words_accepted"] == 1  # the empty word cell doesn't count
    assert (row["phrases"], row["phrases_done"]) == (2, 1)
    assert row["classify_ms_max"] == pytest.approx(23.0)
    assert row["select_ms_median"] == pytest.approx(30.0)  # only text-changing calls
    assert row["condition"] == "gpt2"

    first, second = phrases
    assert (first["phrase"], first["done"], first["selections"]) == ("HI", True, 8)
    assert first["chars_correct"] == 2
    assert first["seconds"] == pytest.approx(8 * T_SEL)
    assert first["chars_per_min"] == pytest.approx(60 * 2 / (8 * T_SEL))
    assert first["selections_per_char"] == pytest.approx(8 / 2)
    assert first["min_selections_none"] == 4  # 2 selections per character
    assert (second["phrase"], second["done"], second["chars_correct"]) == (
        "AC",
        False,
        1,
    )
    assert np.isnan(second["selections_per_char"])
    assert row["chars_correct"] == 3
    assert row["minutes"] == pytest.approx((8 + 6) * T_SEL / 60)


def test_a_mismatch_between_gui_and_backend_is_counted(tmp_path):
    run = write_run(tmp_path, ROWS[:3])
    gui = pd.read_csv(run["online"])
    gui.loc[1, "selected"] = 4
    gui.to_csv(run["online"], index=False)
    row, _ = analyse_online.summarize_run(run)
    assert row["gui_backend_mismatches"] == 1


def test_load_runs_parses_the_runs_file(tmp_path):
    (tmp_path / "runs.csv").write_text(
        "run,online,backend,recording,calibration,method,window,latency,nlp,fusion,"
        "phrases,include,note\n"
        "r1,data/a_online.csv,data/a_backend.csv,,/abs/cal.npz,FBCCA-norm,2,0.6,gpt2,"
        "yes,never too rich| the sum ,no,x\n"
    )
    (run,) = analyse_online.load_runs(tmp_path / "runs.csv")
    assert run["online"] == config.ROOT / "data/a_online.csv"
    assert run["calibration"] == Path("/abs/cal.npz")
    assert run["recording"] is None
    assert (run["window"], run["latency"]) == (2.0, 0.6)
    assert run["fusion"] is True and run["include"] is False
    assert run["phrases"] == ["never too rich", "the sum"]


@pytest.fixture
def replayable_run(tmp_path, ssvep_recording):
    """A calibration, an online recording with 2 s trials (SSVEP 0.14 s after the
    markers), and the decision log of a CCA backend that ran it at latency 0.14."""
    calibration = tmp_path / "cal.npz"
    recording = tmp_path / "run.npz"
    ssvep_recording(calibration, n_blocks=3, seed=1)
    targets = ssvep_recording(recording, n_blocks=2, stim=2.0, seed=2)

    X, y = online.load_calibration([calibration])
    decoder = online.Decoder("CCA", window=1.0, latency=0.14).fit(X, y)
    buffer = acquisition.RingBuffer(len(config.CAP_CHANNELS), int(30 * config.EEG_FS))
    backend = online.Backend(decoder, Speller(), buffer)
    online.replay(recording, backend)
    pd.DataFrame(backend.rows).to_csv(tmp_path / "log.csv", index=False)
    run = {
        "run": "r",
        "recording": recording,
        "calibration": calibration,
        "backend": tmp_path / "log.csv",
        "method": "CCA",
        "window": 1.0,
        "latency": 0.14,
        "include": True,
    }
    return run, targets


def test_replay_at_the_run_latency_reproduces_the_log(replayable_run):
    run, targets = replayable_run
    row = analyse_online.replay_run(run, 0.14)
    assert row["same_as_log"] == 1.0
    assert row["compared"] == row["cued"] == len(targets)
    assert row["accuracy"] > 0.9
    assert (row["latency"], row["run_latency"]) == (0.14, 0.14)


def test_replay_at_another_latency_changes_the_decisions(replayable_run):
    """The response lasts from 0.14 to 2.14 s after each onset: a 1 s window at
    latency 2.2 s holds no response at all."""
    run, _ = replayable_run
    row = analyse_online.replay_run(run, 2.2)
    assert row["accuracy"] < 0.6
    assert row["same_as_log"] < 0.6


def test_a_fusion_run_is_compared_with_the_classifier_decisions(replayable_run):
    """A fusion log's ``selected`` is the fused decision; the replay (classifier only)
    must be compared with ``selected_eeg``."""
    run, _ = replayable_run
    log = pd.read_csv(run["backend"])
    log["selected_eeg"] = log["selected"]
    log["selected"] = (log["selected"] + 1) % 6
    log.to_csv(run["backend"], index=False)
    assert analyse_online.replay_run(run, 0.14)["same_as_log"] == 1.0


def test_main_writes_runs_phrases_and_replay_tables(tmp_path, replayable_run):
    replay_run, _ = replayable_run
    run = write_run(tmp_path, ROWS, name="logs")
    (tmp_path / "runs.csv").write_text(
        "run,online,backend,recording,calibration,method,window,latency,nlp,fusion,"
        "phrases,include,note\n"
        f"a,{run['online']},{run['backend']},,,CCA,2,0.7,gpt2,no,hi|ac,yes,\n"
        f"b,{run['online']},{replay_run['backend']},{replay_run['recording']},"
        f"{replay_run['calibration']},CCA,1,0.14,none,no,,no,\n"
    )
    analyse_online.main(
        [
            "--runs",
            str(tmp_path / "runs.csv"),
            "--replay-latencies",
            "0.14",
            "--name",
            "on",
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    runs = pd.read_csv(tmp_path / "out" / "on_runs.csv")
    assert runs["run"].tolist() == ["a", "b"]
    assert runs["include"].tolist() == [True, False]
    assert set(pd.read_csv(tmp_path / "out" / "on_phrases.csv")["run"]) == {"a"}
    replays = pd.read_csv(tmp_path / "out" / "on_replay.csv")
    assert replays["run"].tolist() == ["b"]  # run a has no recording to replay
