"""Tests for scripts/keystroke_savings.py (toy suggesters only, so they run in CI)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssvep.language import UnigramModel
from ssvep.speller import Speller

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "keystroke_savings.py"
SPEC = importlib.util.spec_from_file_location("keystroke_savings", MODULE_PATH)
ks = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ks)

TOY = {"HELLO": 5.0, "THE": 50.0, "THERE": 30.0, "WORLD": 4.0, "A": 40.0, "TO": 35.0}


def oracle_for(phrase):
    """A suggester that always offers the phrase word being typed (perfect LM)."""
    words = phrase.split(" ")

    def suggest(text):
        return [words[min(text.count(" "), len(words) - 1)]]

    return suggest


def useless(text):
    """Suggests a word that never fits."""
    return ["QQQQ"]


# --- load_phrases ---------------------------------------------------------------


def test_load_phrases_uppercases_and_cleans(tmp_path):
    path = tmp_path / "phrases.txt"
    path.write_bytes(b"my watch fell\r\n\r\n  I can  see\tSaturn \r\nok\n")
    assert ks.load_phrases(path) == ["MY WATCH FELL", "I CAN SEE SATURN", "OK"]


def test_load_phrases_are_typeable(tmp_path):
    """Every loaded phrase can be set on the speller (letters and spaces only)."""
    path = tmp_path / "phrases.txt"
    path.write_text("hello world\nthe quick brown fox\n", encoding="utf-8")
    speller = Speller()
    for phrase in ks.load_phrases(path):
        speller.set_phrase(phrase)


# --- decide -----------------------------------------------------------------------


def test_decide_perfect_classifier_always_picks_the_target():
    rng = np.random.default_rng(0)
    assert all(ks.decide(t % 6, 1.0, 6, rng) == t % 6 for t in range(300))


def test_decide_errors_are_uniform_over_the_other_targets():
    rng = np.random.default_rng(1)
    choices = [ks.decide(2, 0.0, 6, rng) for _ in range(3000)]
    counts = np.bincount(choices, minlength=6)
    assert counts[2] == 0
    others = np.delete(counts, 2)
    assert others.min() > 0.8 * 600 and others.max() < 1.2 * 600


def test_decide_accuracy_is_respected():
    rng = np.random.default_rng(2)
    hits = np.mean([ks.decide(4, 0.75, 6, rng) == 4 for _ in range(4000)])
    assert hits == pytest.approx(0.75, abs=0.03)


def test_decide_without_a_target_picks_any_target():
    rng = np.random.default_rng(3)
    choices = {ks.decide(None, 1.0, 6, rng) for _ in range(500)}
    assert choices == set(range(6))


# --- copy_phrase ------------------------------------------------------------------


def test_copy_letter_by_letter_takes_two_selections_per_character():
    result = ks.copy_phrase(Speller(), "HELLO WORLD")
    assert result == {"selections": 22, "done": True, "words": 0}


def test_copy_with_a_perfect_language_model_takes_two_selections_per_word():
    result = ks.copy_phrase(Speller(suggest=oracle_for("HELLO WORLD")), "HELLO WORLD")
    assert result == {"selections": 4, "done": True, "words": 2}


def test_copy_with_useless_suggestions_saves_nothing():
    result = ks.copy_phrase(Speller(suggest=useless), "HELLO WORLD")
    assert result["selections"] == 22 and result["words"] == 0


def test_copy_reuses_one_speller_across_phrases():
    speller = Speller()
    first = ks.copy_phrase(speller, "AB")
    second = ks.copy_phrase(speller, "AB")
    assert first == second == {"selections": 4, "done": True, "words": 0}


def test_copy_perfect_accuracy_matches_the_error_free_count():
    rng = np.random.default_rng(0)
    result = ks.copy_phrase(Speller(), "HELLO WORLD", accuracy=1.0, rng=rng)
    assert result["selections"] == 22 and result["done"]


def test_copy_with_errors_needs_more_selections_and_still_finishes():
    errors = [
        ks.copy_phrase(Speller(), "HELLO", 400, 0.8, np.random.default_rng(s))
        for s in range(20)
    ]
    assert all(r["done"] for r in errors)
    assert all(r["selections"] >= 10 for r in errors)
    assert np.mean([r["selections"] for r in errors]) > 10


def test_copy_with_errors_counts_only_words_that_were_typed():
    """Erroneous selections of an empty suggestion cell type nothing: no word."""
    for seed in range(10):
        rng = np.random.default_rng(seed)
        result = ks.copy_phrase(Speller(), "HELLO WORLD", 400, 0.5, rng)
        assert result["words"] == 0


def test_copy_with_errors_is_reproducible():
    def run():
        return ks.copy_phrase(Speller(), "HELLO", 400, 0.7, np.random.default_rng(5))

    assert run() == run()


def test_copy_stops_at_the_selection_limit():
    """A classifier that is always wrong never finishes; the loop is bounded."""
    rng = np.random.default_rng(0)
    result = ks.copy_phrase(Speller(), "HI", 50, accuracy=0.0, rng=rng)
    assert result["selections"] == 50 and not result["done"]


def test_copy_with_a_classify_function_asks_it_before_each_selection():
    seen = []

    def classify(target, speller):
        seen.append((target, speller.text, speller.group))
        return target

    result = ks.copy_phrase(Speller(), "AB", classify=classify)
    assert result == {"selections": 4, "done": True, "words": 0}
    speller = Speller()
    a_group = speller.slots.index(0)  # A is in the group drawn in slot 0
    assert seen[0] == (a_group, "", None)
    assert [text for _, text, _ in seen] == ["", "", "A", "A"]


def test_copy_with_a_classify_function_uses_its_decisions():
    """A classify function that always answers target 0 never types B."""
    result = ks.copy_phrase(Speller(), "B", 30, classify=lambda target, speller: 0)
    assert result["selections"] == 30 and not result["done"]


# --- TimedSuggest -----------------------------------------------------------------


class MemoSuggest:
    def __init__(self):
        self.seen = {}

    def __call__(self, text):
        self.seen[text] = ["A"]
        return ["A"]


def test_timed_suggest_passes_results_through_and_records_calls():
    timed = ks.TimedSuggest(MemoSuggest())
    assert timed("T") == ["A"]
    assert timed("T") == ["A"]
    assert timed("TO") == ["A"]
    assert [(text, hit) for text, _, hit in timed.calls] == [
        ("T", False),
        ("T", True),
        ("TO", False),
    ]
    assert all(ms >= 0 for _, ms, _ in timed.calls)


def test_timed_suggest_works_without_a_memo():
    timed = ks.TimedSuggest(useless)
    timed("X")
    assert timed.calls[0][2] is False


# --- evaluate, summarize, compare_pair --------------------------------------------


def test_evaluate_rows_and_keystroke_savings():
    phrases = ["HELLO WORLD", "THE"]
    results = ks.evaluate(phrases, {"none": None, "useless": useless})
    assert len(results) == 4
    assert list(results["baseline"]) == [22, 6, 22, 6]
    assert (results["ks"] == 0).all()
    assert results["done"].all()


def test_evaluate_with_the_unigram_saves_selections():
    phrases = ["HELLO WORLD", "THE", "HELLO THERE"]
    results = ks.evaluate(phrases, {"unigram": UnigramModel(TOY)})
    assert (results["ks"] > 0).all()
    assert (results["selections"] < results["baseline"]).all()
    # THE is the top suggestion on an empty text: one group + one item
    assert results.loc[results["phrase"] == "THE", "selections"].item() == 2


def test_summarize_totals_and_latency():
    phrases = ["HELLO WORLD", "THE"]
    timed = ks.TimedSuggest(UnigramModel(TOY))
    results = ks.evaluate(phrases, {"none": None, "unigram": timed})
    summary = ks.summarize(results, ks.latency_table({"unigram": timed}))
    none = summary.set_index("condition").loc["none"]
    unigram = summary.set_index("condition").loc["unigram"]
    assert none["selections"] == 28 and none["ks_total"] == 0
    assert np.isnan(none["ms_median"])
    assert unigram["ks_total"] == pytest.approx(1 - unigram["selections"] / 28)
    assert unigram["model_calls"] == len(timed.calls) > 0


def test_compare_pair_counts_and_p_value():
    results = pd.DataFrame(
        {
            "phrase_id": list(range(8)) * 2,
            "condition": ["a"] * 8 + ["b"] * 8,
            "selections": [10] * 8 + [8, 8, 8, 8, 8, 8, 10, 12],
        }
    )
    pair = ks.compare_pair(results, "a", "b")
    assert (pair["fewer"], pair["equal"], pair["more"]) == (6, 1, 1)
    assert pair["mean_diff"] == pytest.approx((6 * -2 + 2) / 8)
    assert 0 < pair["p_value"] < 1
    assert ks.compare_pair(results, "a", "missing") is None


def test_compare_pair_identical_conditions():
    results = pd.DataFrame(
        {"phrase_id": [0, 1, 0, 1], "condition": list("aabb"), "selections": [4] * 4}
    )
    assert ks.compare_pair(results, "a", "b")["p_value"] == 1.0


# --- simulate, summarize_sim ------------------------------------------------------


def test_simulate_rows_and_shared_randomness():
    phrases = ["HELLO", "THE"]
    sim = ks.simulate(
        phrases, {"none": None, "empty": lambda text: []}, (0.8, 1.0), reps=2, seed=1
    )
    assert len(sim) == 2 * 2 * 2 * 2
    # A suggester that never suggests anything behaves exactly like none, so with
    # the same seed the two conditions make exactly the same selections
    table = sim.pivot_table(
        index=["accuracy", "rep", "phrase_id"], columns="condition", values="selections"
    )
    assert (table["none"] == table["empty"]).all()
    perfect = sim[sim["accuracy"] == 1.0]
    assert (perfect["selections"] == 2 * perfect["n_chars"]).all()


def test_simulate_repetitions_take_different_random_paths():
    sim = ks.simulate(["HELLO"], {"none": None}, (0.7,), reps=6, seed=0)
    assert sim["selections"].nunique() > 1


def test_summarize_sim_time_and_savings():
    phrases = ["HELLO WORLD"]
    sim = ks.simulate(
        phrases, {"none": None, "oracle": oracle_for("HELLO WORLD")}, (1.0,), reps=1
    )
    summary = ks.summarize_sim(sim, selection_s=3.0).set_index("condition")
    assert summary.loc["none", "minutes_median"] == pytest.approx(22 * 3.0 / 60)
    assert summary.loc["oracle", "minutes_median"] == pytest.approx(4 * 3.0 / 60)
    assert summary.loc["none", "chars_per_min"] == pytest.approx(11 / (22 * 3 / 60))
    assert summary.loc["none", "saved_vs_none"] == 0
    assert summary.loc["oracle", "saved_vs_none"] == pytest.approx(1 - 4 / 22)
    assert (summary["done"] == 1).all()


def test_summarize_sim_counts_unfinished_phrases_as_time_without_characters():
    sim = ks.simulate(["HI"], {"none": None}, (0.0,), reps=1, max_selections=30)
    summary = ks.summarize_sim(sim, selection_s=2.0)
    assert summary["done"].item() == 0
    assert summary["chars_per_min"].item() == 0
    assert summary["minutes_median"].item() == pytest.approx(1.0)


def test_parse_args_defaults():
    args = ks.parse_args([])
    assert args.conditions == ["none", "unigram", "gpt2"]
    assert args.selection_s == pytest.approx(3.7)
    assert args.sim_phrases > 0
