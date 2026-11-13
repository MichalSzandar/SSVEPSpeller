"""Tests for scripts/evaluate_fusion.py (toy scores and vocabulary, no EEG)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssvep import config, fusion
from ssvep.language import UnigramModel
from ssvep.speller import Speller

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # it imports online and keystroke_savings from there
SPEC = importlib.util.spec_from_file_location(
    "evaluate_fusion", SCRIPTS / "evaluate_fusion.py"
)
ef = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ef)

N = len(config.TARGET_FREQS)
TOY = {"HELLO": 5.0, "HELP": 3.0, "HE": 2.0, "THE": 50.0, "A": 40.0, "WORLD": 4.0}
PHRASES = ["HELLO", "THE WORLD"]


def perfect_scores(n_blocks=3):
    """Calibration scores of a perfect classifier: highest on the true target."""
    y = np.tile(np.arange(N), n_blocks)
    return 3.0 * np.eye(N)[y], y


def noisy_scores(n_blocks=10, seed=0):
    """Scores of a classifier that is right ~70 % of the time, every row different."""
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(N), n_blocks)
    return rng.normal(size=(len(y), N)) + 1.5 * np.eye(N)[y], y


def row_index(scores, row):
    (index,) = np.flatnonzero(np.all(scores == row, axis=1))
    return index


class FakeFusion:
    """Always decides ``choice``; records what decide was given."""

    def __init__(self, choice):
        self.choice = choice
        self.calls = []

    def decide(self, scores, speller):
        self.calls.append((np.array(scores), speller))
        return self.choice, np.full(N, 1 / N), np.eye(N)[self.choice]


# --- draw_scores ----------------------------------------------------------------------


def test_draw_scores_come_from_trials_of_the_target():
    scores, y = noisy_scores()
    rng = np.random.default_rng(0)
    drawn = [row_index(scores, ef.draw_scores(scores, y, 2, rng)) for _ in range(300)]
    assert set(drawn) == set(np.flatnonzero(y == 2))  # all of them, and only them


def test_draw_scores_without_a_target_come_from_any_trial():
    scores, y = noisy_scores(n_blocks=2)
    rng = np.random.default_rng(1)
    drawn = [
        row_index(scores, ef.draw_scores(scores, y, None, rng)) for _ in range(500)
    ]
    assert set(drawn) == set(range(len(y)))


def test_draw_scores_are_reproducible():
    scores, y = noisy_scores()

    def draws():
        rng = np.random.default_rng(7)
        return [ef.draw_scores(scores, y, t % N, rng) for t in range(20)]

    np.testing.assert_array_equal(draws(), draws())
    assert ef.draw_scores(scores, y, 0, np.random.default_rng(0)).shape == (N,)


# --- make_classify --------------------------------------------------------------------


def test_classify_without_fusion_is_the_argmax_of_the_drawn_scores():
    scores, y = perfect_scores()
    classify = ef.make_classify(scores, y, np.random.default_rng(0))
    speller = Speller()
    decisions = [classify(t, speller) for t in range(N)]
    assert decisions == list(range(N))
    assert all(type(d) is int for d in decisions)


def test_classify_without_fusion_keeps_the_classifiers_confusions():
    """A classifier that always says the next target up."""
    y = np.tile(np.arange(N), 3)
    scores = np.eye(N)[(y + 1) % N]
    classify = ef.make_classify(scores, y, np.random.default_rng(0))
    assert [classify(t, Speller()) for t in range(N)] == [(t + 1) % N for t in range(N)]


def test_classify_with_fusion_asks_it_with_the_drawn_scores_and_the_speller():
    scores, y = noisy_scores()
    fake = FakeFusion(choice=4)
    classify = ef.make_classify(scores, y, np.random.default_rng(0), fake)
    speller = Speller()
    assert classify(1, speller) == 4
    assert classify(None, speller) == 4
    (first_scores, first_speller), _ = fake.calls
    assert y[row_index(scores, first_scores)] == 1
    assert first_speller is speller


# --- run --------------------------------------------------------------------------------


def test_run_rows():
    scores, y = noisy_scores()
    settings = {ef.NO_FUSION: None, "fake": FakeFusion(choice=0)}
    results = ef.run(
        PHRASES, {"none": None}, scores, y, settings, reps=2, max_selections=40
    )
    assert len(results) == 1 * 2 * 2 * len(PHRASES)
    assert {
        "condition",
        "setting",
        "rep",
        "phrase_id",
        "n_chars",
        "selections",
        "done",
    } <= set(results.columns)
    fake = results[results["setting"] == "fake"]
    assert (fake["selections"] == 40).all() and not fake["done"].any()


def test_run_with_a_perfect_classifier_types_two_selections_per_character():
    scores, y = perfect_scores()
    results = ef.run(PHRASES, {"none": None}, scores, y, {ef.NO_FUSION: None})
    assert results["done"].all()
    assert (results["selections"] == 2 * results["n_chars"]).all()


def test_run_each_condition_spells_with_its_suggester():
    """With TOY, HELLO is suggested from the start: one group + one word cell."""
    scores, y = perfect_scores()
    suggesters = {"none": None, "unigram": UnigramModel(TOY)}
    results = ef.run(["HELLO"], suggesters, scores, y, {ef.NO_FUSION: None})
    selections = results.set_index("condition")["selections"]
    assert (selections["none"], selections["unigram"]) == (10, 2)


def test_run_settings_share_their_random_streams():
    """Two settings that decide alike make exactly the same selections, and a Fusion
    with floor 1 (an even prior) decides like the classifier alone."""
    scores, y = noisy_scores()
    settings = {
        ef.NO_FUSION: None,
        "eeg again": None,
        "even prior": fusion.Fusion(1.0, TOY, floor=1.0),
    }
    unigram = UnigramModel(TOY)
    results = ef.run(
        PHRASES,
        {"none": None, "unigram": unigram},
        scores,
        y,
        settings,
        reps=3,
        max_selections=200,
        seed=5,
    )
    table = results.pivot_table(
        index=["condition", "rep", "phrase_id"], columns="setting", values="selections"
    )
    assert (table[ef.NO_FUSION] == table["eeg again"]).all()
    assert (table[ef.NO_FUSION] == table["even prior"]).all()
    assert table[ef.NO_FUSION].nunique() > 1  # the repetitions do differ


def test_run_is_reproducible():
    scores, y = noisy_scores()

    def once(seed):
        return ef.run(
            PHRASES,
            {"none": None},
            scores,
            y,
            {ef.NO_FUSION: None},
            reps=2,
            max_selections=200,
            seed=seed,
        )

    pd.testing.assert_frame_equal(once(3), once(3))


# --- summarize --------------------------------------------------------------------------


def test_summarize():
    results = pd.DataFrame(
        {
            "condition": ["none"] * 4 + ["gpt2"] * 4,
            "setting": [ef.NO_FUSION, ef.NO_FUSION, "floor 0.3", "floor 0.3"] * 2,
            "rep": [0] * 8,
            "phrase_id": [0, 1] * 4,
            "n_chars": [5, 9] * 4,
            "selections": [10, 30, 8, 20, 6, 14, 6, 400],
            "done": [True, True, True, True, True, True, True, False],
        }
    )
    summary = ef.summarize(results).set_index(["condition", "setting"])
    assert len(summary) == 4
    assert summary.loc[("none", ef.NO_FUSION), "selections"] == 40
    assert summary.loc[("none", "floor 0.3"), "vs_eeg"] == pytest.approx(28 / 40)
    assert summary.loc[("none", ef.NO_FUSION), "vs_eeg"] == 1
    assert summary.loc[("gpt2", "floor 0.3"), "vs_eeg"] == pytest.approx(406 / 20)
    assert summary.loc[("gpt2", "floor 0.3"), "done"] == 0.5
    assert summary.loc[("none", "floor 0.3"), "selections_median"] == 14
    assert summary.loc[("none", "floor 0.3"), "runs"] == 2


# --- parse_args -------------------------------------------------------------------------


def test_parse_args_defaults():
    args = ef.parse_args(["--method", "FBCCA-norm", "--calibration", "a.npz", "b.npz"])
    assert args.calibration == [Path("a.npz"), Path("b.npz")]
    assert args.latency == pytest.approx(0.7)
    assert args.window == config.ONLINE_STIM_DURATION
    assert args.conditions == ["none", "unigram", "gpt2"]
    assert list(args.floors) == list(ef.FLOORS)
    assert args.limit > 0 and args.reps > 0
    assert args.name == "fusion_sim"
    with pytest.raises(SystemExit):
        ef.parse_args(["--method", "nonsense", "--calibration", "a.npz"])
