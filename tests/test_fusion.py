"""Tests for ssvep.fusion (classifier x language-model prior), with a toy vocabulary."""

import numpy as np
import pytest
from scipy.special import log_softmax
from scipy.special import softmax as scipy_softmax

from ssvep import config, fusion
from ssvep.language import CHARS, UnigramModel
from ssvep.speller import BACK, BACKSPACE, SPACE, WORDS, Speller

SLOTS = config.TARGET_SLOTS
OTHER_SLOTS = [2, 5, 0, 3, 1, 4]  # a second permutation: catches slot/target mix-ups
N = len(SLOTS)

# Words that start with "HE": HE (2), HELLO (5), HELP (3), total 10
TOY = {"HE": 2.0, "HELLO": 5.0, "HELP": 3.0, "THE": 50.0, "THERE": 30.0, "A": 40.0}
TOY["WORLD"] = 4.0
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def item_key(char):
    return SPACE if char == " " else char


def by_target(slot_values, slots):
    """Values given per slot -> the same values in target order."""
    return np.array([slot_values[slots[k]] for k in range(len(slots))])


def speller_at(text, suggestions=(), group=None, slots=SLOTS):
    """A speller showing ``text`` with these suggestions, at level 1 (group None) or
    with the group in slot ``group`` open."""
    speller = Speller(slots=slots, suggest=lambda text: list(suggestions))
    speller.text = text
    speller.update_suggestions()
    speller.group = group
    return speller


# --- softmax ------------------------------------------------------------------------


def test_softmax_matches_scipy():
    scores = np.array([0.3, -1.2, 2.0, 0.0, 0.7, -0.4])
    np.testing.assert_allclose(fusion.softmax(scores), scipy_softmax(scores))
    np.testing.assert_allclose(fusion.softmax(scores, 0.5), scipy_softmax(scores / 0.5))


def test_softmax_of_many_trials_works_row_by_row():
    scores = np.random.default_rng(0).normal(size=(7, N))
    probs = fusion.softmax(scores, 2.0)
    assert probs.shape == (7, N)
    np.testing.assert_allclose(probs, scipy_softmax(scores / 2.0, axis=1))
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_softmax_does_not_overflow():
    probs = fusion.softmax(np.array([1000.0, 999.0, -1000.0]))
    assert np.all(np.isfinite(probs))
    np.testing.assert_allclose(probs, scipy_softmax([1.0, 0.0, -2000.0]))


def test_softmax_temperature_flattens_or_sharpens():
    scores = np.array([2.0, 1.0, 0.0])
    assert fusion.softmax(scores, 100.0).max() < 0.36  # almost even
    assert fusion.softmax(scores, 0.01).max() > 0.999  # almost one-hot


# --- fit_temperature ----------------------------------------------------------------


def mean_nll(scores, y, temperature):
    """Independent reference: mean negative log-likelihood of the true labels."""
    return -np.mean(log_softmax(scores / temperature, axis=1)[np.arange(len(y)), y])


def labelled_scores(true_temperature, n=4000, seed=0):
    """Scores, and labels drawn from softmax(scores / true_temperature): T is known."""
    rng = np.random.default_rng(seed)
    scores = rng.normal(0.0, 3.0, size=(n, N))
    probs = scipy_softmax(scores / true_temperature, axis=1)
    y = np.array([rng.choice(N, p=p) for p in probs])
    return scores, y


@pytest.mark.parametrize("true_temperature", [0.5, 2.0])
def test_fit_temperature_recovers_a_known_temperature(true_temperature):
    scores, y = labelled_scores(true_temperature)
    fitted = fusion.fit_temperature(scores, y)
    assert isinstance(fitted, float)
    assert fitted == pytest.approx(true_temperature, rel=0.1)


def test_fit_temperature_minimises_the_negative_log_likelihood():
    scores, y = labelled_scores(1.5, n=600, seed=1)
    fitted = fusion.fit_temperature(scores, y)
    best = mean_nll(scores, y, fitted)
    for factor in (0.5, 0.9, 0.99, 1.01, 1.1, 2.0):
        assert best <= mean_nll(scores, y, factor * fitted) + 1e-9


def test_fit_temperature_scales_with_the_scores():
    scores, y = labelled_scores(1.0, n=600, seed=2)
    assert fusion.fit_temperature(3 * scores, y) == pytest.approx(
        3 * fusion.fit_temperature(scores, y), rel=1e-3
    )


def test_fit_temperature_stays_within_the_bounds():
    scores, y = labelled_scores(2.0, n=600, seed=3)
    fitted = fusion.fit_temperature(scores, y, bounds=(0.5, 1.0))
    assert 0.5 <= fitted <= 1.0
    assert fitted == pytest.approx(1.0, rel=0.01)  # the best T inside (0.5, 1)


def test_fit_temperature_of_perfectly_separated_scores_is_small():
    """No best T: the smaller, the better. Any small T makes the true target ~certain."""
    y = np.tile(np.arange(N), 5)
    scores = 3.0 * np.eye(N)[y]
    fitted = fusion.fit_temperature(scores, y)
    assert fitted < 0.2
    assert mean_nll(scores, y, fitted) < 1e-6


# --- item_probs ---------------------------------------------------------------------


def test_item_probs_keys_and_total():
    probs = fusion.item_probs("HE", ["HELLO", "HELP"], TOY)
    assert set(probs) == set(LETTERS) | {SPACE, BACKSPACE, WORDS[0], WORDS[1]}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert set(fusion.item_probs("HE", [], TOY)) == set(LETTERS) | {SPACE, BACKSPACE}


def test_item_probs_without_suggestions_by_hand():
    """HE: HELLO (5) and HELP (3) continue with L, HE (2) is a word: SPACE."""
    probs = fusion.item_probs("HE", [], TOY, error=0.1)
    assert probs["L"] == pytest.approx(0.9 * 0.8)
    assert probs[SPACE] == pytest.approx(0.9 * 0.2)
    assert probs[BACKSPACE] == pytest.approx(0.1)
    assert sum(probs[c] for c in LETTERS if c != "L") == 0


def test_item_probs_suggested_words_take_their_words_share():
    """The user would accept HELLO or HELP instead of typing their L."""
    probs = fusion.item_probs("HE", ["HELLO", "HELP"], TOY, error=0.1)
    assert probs[WORDS[0]] == pytest.approx(0.9 * 0.5)
    assert probs[WORDS[1]] == pytest.approx(0.9 * 0.3)
    assert probs[SPACE] == pytest.approx(0.9 * 0.2)
    assert probs["L"] == 0


def test_item_probs_a_suggestion_outside_the_vocabulary_gets_nothing():
    probs = fusion.item_probs("HE", ["HEAVY", "HELLO"], TOY, error=0.1)
    assert probs[WORDS[0]] == 0
    assert probs[WORDS[1]] == pytest.approx(0.9 * 0.5)
    assert probs["L"] == pytest.approx(0.9 * 0.3)


def test_item_probs_use_only_the_word_being_typed():
    assert fusion.item_probs("SAY HE", [], TOY) == pytest.approx(
        fusion.item_probs("HE", [], TOY)
    )


def test_item_probs_at_the_start_of_a_word():
    """No word is empty, so SPACE gets nothing; first letters by word frequency."""
    probs = fusion.item_probs("HELLO ", [], TOY, error=0.1)
    total = sum(TOY.values())
    assert probs[SPACE] == 0
    assert probs["T"] == pytest.approx(0.9 * 80 / total)
    assert probs["H"] == pytest.approx(0.9 * 10 / total)


def test_item_probs_of_an_unknown_prefix_are_even():
    probs = fusion.item_probs("XQ", [], TOY, error=0.2)
    assert probs[BACKSPACE] == pytest.approx(0.2)
    for char in CHARS:
        assert probs[item_key(char)] == pytest.approx(0.8 / 27)


def test_item_probs_of_an_unknown_prefix_leave_the_word_cells_empty():
    """Suggestions outside the vocabulary (e.g. from the transformer) get nothing."""
    probs = fusion.item_probs("XQ", ["XQA", "XQB"], TOY, error=0.2)
    assert probs[WORDS[0]] == probs[WORDS[1]] == 0
    assert sum(probs.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("text", ["", "H", "HE", "HEL", "THE", "THE W", "QQ"])
def test_item_probs_without_suggestions_match_char_probs(text):
    error = 0.15
    probs = fusion.item_probs(text, [], TOY, error=error)
    expected = UnigramModel(TOY).char_probs(text)
    for char in CHARS:
        assert probs[item_key(char)] == pytest.approx((1 - error) * expected[char])


# --- target_prior -------------------------------------------------------------------


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_level_one_prior_adds_up_each_group(slots):
    """HE: L (0.72) is in K-O (slot 2); SPACE (0.18) and BACKSPACE (0.1) in slot 5."""
    speller = speller_at("HE", slots=slots)
    probs = fusion.item_probs("HE", [], TOY, error=0.1)
    prior = fusion.target_prior(speller, probs, error=0.1, floor=0.0)
    np.testing.assert_allclose(prior, by_target([0, 0, 0.72, 0, 0, 0.28], slots))


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_level_one_prior_with_suggestions(slots):
    """Every likely next item is now in the word group."""
    speller = speller_at("HE", ["HELLO", "HELP"], slots=slots)
    probs = fusion.item_probs("HE", ["HELLO", "HELP"], TOY, error=0.1)
    prior = fusion.target_prior(speller, probs, error=0.1, floor=0.0)
    np.testing.assert_allclose(
        prior, by_target([0, 0, 0, 0, 0, 1.0], slots), atol=1e-12
    )


def test_level_one_items_without_a_key_count_as_zero():
    speller = speller_at("")
    prior = fusion.target_prior(speller, {"A": 0.25, "Z": 0.75}, floor=0.0)
    np.testing.assert_allclose(prior, by_target([0.25, 0, 0, 0, 0.75, 0], SLOTS))


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_level_two_prior_gives_back_the_error_share(slots):
    """K-O open after HE: L takes all of 1 - error, BACK (slot 5) gets error."""
    speller = speller_at("HE", group=2, slots=slots)
    probs = fusion.item_probs("HE", [], TOY, error=0.1)
    prior = fusion.target_prior(speller, probs, error=0.1, floor=0.0)
    np.testing.assert_allclose(prior, by_target([0, 0.9, 0, 0, 0, 0.1], slots))


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_level_two_prior_shares_in_proportion(slots):
    """Word group after HE with HELLO, HELP (error 0.2): #1 .4, #2 .24, #3 empty,
    SPACE .16 and BACKSPACE .2 share 1 - error in proportion; BACK gets error."""
    speller = speller_at("HE", ["HELLO", "HELP"], group=5, slots=slots)
    probs = fusion.item_probs("HE", ["HELLO", "HELP"], TOY, error=0.2)
    prior = fusion.target_prior(speller, probs, error=0.2, floor=0.0)
    in_group = np.array([0.4, 0.24, 0.0, 0.16, 0.2])  # sums to 1
    np.testing.assert_allclose(prior, by_target([*0.8 * in_group, 0.2], slots))


def test_level_two_prior_of_a_group_without_likely_items_is_even():
    """A-E after HE (a wrong group): A-E share 1 - error evenly, BACK gets error."""
    speller = speller_at("HE", group=0)
    probs = fusion.item_probs("HE", [], TOY, error=0.1)
    prior = fusion.target_prior(speller, probs, error=0.1, floor=0.0)
    np.testing.assert_allclose(prior, by_target([0.18] * 5 + [0.1], SLOTS))


def test_level_two_prior_of_a_group_without_back():
    """U-Z has no BACK: its items share everything, evenly if all are 0."""
    probs = fusion.item_probs("", [], TOY, error=0.1)  # only W of U-Z starts a word
    prior = fusion.target_prior(speller_at("", group=4), probs, error=0.1, floor=0.0)
    np.testing.assert_allclose(prior, by_target([0, 0, 1.0, 0, 0, 0], SLOTS))
    probs = fusion.item_probs("HE", [], TOY, error=0.1)
    prior = fusion.target_prior(speller_at("HE", group=4), probs, floor=0.0)
    np.testing.assert_allclose(prior, np.full(N, 1 / N))


@pytest.mark.parametrize("group", [None, 0, 2, 4, 5])
def test_floor_mixes_in_an_even_prior(group):
    speller = speller_at("HE", ["HELLO"], group=group)
    probs = fusion.item_probs("HE", ["HELLO"], TOY)
    lm = fusion.target_prior(speller, probs, floor=0.0)
    floored = fusion.target_prior(speller, probs, floor=0.3)
    np.testing.assert_allclose(floored, 0.7 * lm + 0.3 / N)
    assert floored.sum() == pytest.approx(1.0)
    assert floored.min() >= 0.3 / N - 1e-12
    np.testing.assert_allclose(
        fusion.target_prior(speller, probs, floor=1.0), np.full(N, 1 / N)
    )


def test_target_prior_leaves_the_speller_alone():
    speller = speller_at("HE", ["HELLO"], group=5)
    fusion.target_prior(speller, fusion.item_probs("HE", ["HELLO"], TOY))
    assert (speller.text, speller.group, speller.suggestions) == ("HE", 5, ["HELLO"])


# --- fuse -----------------------------------------------------------------------------


def test_fuse_multiplies_and_normalises():
    eeg = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    prior = np.array([0.1, 0.4, 0.1, 0.2, 0.1, 0.1])
    expected = eeg * prior / np.sum(eeg * prior)
    np.testing.assert_allclose(fusion.fuse(eeg, prior), expected)


def test_fuse_with_an_even_prior_changes_nothing():
    eeg = np.array([0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    np.testing.assert_allclose(fusion.fuse(eeg, np.full(N, 1 / N)), eeg)


def test_fuse_many_trials_with_one_prior():
    eeg = scipy_softmax(np.random.default_rng(4).normal(size=(5, N)), axis=1)
    prior = np.array([0.1, 0.4, 0.1, 0.2, 0.1, 0.1])
    posterior = fusion.fuse(eeg, prior)
    assert posterior.shape == (5, N)
    for row, p in zip(eeg, posterior):
        np.testing.assert_allclose(p, fusion.fuse(row, prior))


# --- Fusion -------------------------------------------------------------------------


def test_fusion_stores_its_settings():
    f = fusion.Fusion(0.7, TOY, floor=0.25, error=0.05)
    assert (f.temperature, f.floor, f.error) == (0.7, 0.25, 0.05)
    assert f.vocabulary is TOY
    defaults = fusion.Fusion(1.0, TOY)
    assert (defaults.floor, defaults.error) == (fusion.FLOOR, fusion.ERROR)


@pytest.mark.parametrize("group", [None, 2, 5])
def test_fusion_prior_is_the_target_prior_of_the_current_state(group):
    speller = speller_at("HE", ["HELLO", "HELP"], group=group)
    f = fusion.Fusion(0.7, TOY, floor=0.25, error=0.05)
    probs = fusion.item_probs("HE", ["HELLO", "HELP"], TOY, error=0.05)
    expected = fusion.target_prior(speller, probs, error=0.05, floor=0.25)
    np.testing.assert_allclose(f.prior(speller), expected)


def test_decide_returns_the_argmax_of_the_posterior():
    speller = speller_at("HE")
    f = fusion.Fusion(0.7, TOY)
    scores = np.array([0.3, -1.2, 2.0, 0.0, 0.7, -0.4])
    selected, prior, posterior = f.decide(scores, speller)
    np.testing.assert_allclose(prior, f.prior(speller))
    np.testing.assert_allclose(
        posterior, fusion.fuse(fusion.softmax(scores, 0.7), prior)
    )
    assert selected == int(np.argmax(posterior))
    assert type(selected) is int


def test_decide_the_language_model_breaks_a_near_tie():
    """HE: the EEG slightly prefers A-E, but only K-O can continue the word."""
    speller = speller_at("HE")
    a_e, k_o = SLOTS.index(0), SLOTS.index(2)
    scores = np.zeros(N)
    scores[a_e], scores[k_o] = 1.0, 0.9
    assert np.argmax(scores) == a_e
    assert fusion.Fusion(1.0, TOY).decide(scores, speller)[0] == k_o


@pytest.mark.parametrize("group", [None, 2])
def test_decide_a_confident_eeg_decision_overrules_the_language_model(group):
    """95 % EEG confidence wins even for a target the language model rules out
    (the floor): here U-Z at level 1, and K at level 2 in K-O."""
    speller = speller_at("HE", group=group)
    target = SLOTS.index(4) if group is None else SLOTS.index(0)
    eeg = np.full(N, 0.01)
    eeg[target] = 0.95
    selected, _, _ = fusion.Fusion(1.0, TOY, floor=0.3).decide(np.log(eeg), speller)
    assert selected == target


def test_decide_leaves_the_speller_alone():
    speller = speller_at("HE", ["HELLO"], group=5)
    fusion.Fusion(1.0, TOY).decide(np.zeros(N), speller)
    assert (speller.text, speller.group, speller.suggestions) == ("HE", 5, ["HELLO"])
