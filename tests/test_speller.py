"""Tests for ssvep.speller (two-level menu state machine, no EEG needed)."""

import string

import numpy as np
import pytest

from ssvep import config, protocol
from ssvep.speller import BACK, BACKSPACE, MENU, SPACE, WORDS, Speller

SLOTS = config.TARGET_SLOTS
# A second permutation, so a test notices when slots and targets get mixed up
OTHER_SLOTS = [2, 5, 0, 3, 1, 4]

# Group 0 has no BACK, so a wrong selection of it can't be undone
TOY = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", BACK],
    ["L", "M", "N", "O", "P", BACK],
    ["Q", "R", "S", "T", "U", BACK],
    ["V", "W", "X", "Y", "Z", BACK],
    [*WORDS, SPACE, BACKSPACE, BACK],
]


def find(menu, item):
    """(group slot, item slot) of the first group that holds ``item``."""
    for g, group in enumerate(menu):
        if item in group:
            return g, group.index(item)
    raise LookupError(item)


def targets_for(speller, item):
    """The two targets that select ``item`` from level 1: its group, then the item."""
    g, i = find(speller.menu, item)
    return speller.slots.index(g), speller.slots.index(i)


def type_item(speller, item):
    """Select ``item`` from level 1 -> what select returned for the second selection."""
    group_target, item_target = targets_for(speller, item)
    speller.select(group_target)
    return speller.select(item_target)


def type_text(speller, text):
    for char in text:
        type_item(speller, SPACE if char == " " else char)


def recording(*words):
    """A suggest function that always returns ``words`` and records its calls."""

    def suggest(text):
        suggest.calls.append(text)
        return list(words)

    suggest.calls = []
    return suggest


def word_cell_labels(speller):
    """Labels of the WORDS cells, read from the level-1 preview of their group."""
    g, _ = find(speller.menu, WORDS[0])
    preview = speller.labels()[speller.slots.index(g)]
    return [preview[speller.slots.index(speller.menu[g].index(word))] for word in WORDS]


def state(speller):
    """The speller's fields as a decoded protocol state."""
    return {
        "event": "state",
        "labels": speller.labels(),
        "text": speller.text,
        "phrase": speller.phrase,
        "target": speller.next_target(),
    }


# --- the menu ----------------------------------------------------------------------


def test_menu_is_6_groups_of_6():
    assert len(MENU) == 6
    assert all(len(group) == 6 for group in MENU)


def test_menu_has_every_letter_space_backspace_and_word_cell_once():
    """26 letters, SPACE, BACKSPACE and the word cells once each; every other cell is
    BACK, at most one per group."""
    items = [item for group in MENU for item in group]
    needed = [*string.ascii_uppercase, SPACE, BACKSPACE, *WORDS]
    for item in needed:
        assert items.count(item) == 1, item
    assert set(items) - set(needed) <= {BACK}
    assert all(group.count(BACK) <= 1 for group in MENU)


# --- labels ------------------------------------------------------------------------


def where():
    """A menu whose labels say where they are: "g{group slot}i{item slot}"."""
    return [[f"g{g}i{i}" for i in range(6)] for g in range(6)]


def test_starts_empty_at_level_1():
    speller = Speller()
    assert speller.text == ""
    assert speller.phrase == ""
    assert speller.group is None
    assert speller.suggestions == []


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_level_1_labels_are_group_previews_in_target_order(slots):
    """Target k shows the group in slot slots[k]; inside it, entry j is the item in
    slot slots[j]."""
    labels = Speller(where(), slots).labels()
    assert labels == [[f"g{slots[k]}i{slots[j]}" for j in range(6)] for k in range(6)]


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_selecting_a_group_opens_it(slots):
    speller = Speller(where(), slots)
    assert speller.select(4) is None
    assert speller.group == slots[4]
    assert speller.labels() == [f"g{slots[4]}i{slots[k]}" for k in range(6)]


def test_labels_pass_protocol_validation_at_both_levels():
    speller = Speller(suggest=recording("hello"))
    protocol.validate_state(state(speller))
    speller.select(0)
    protocol.validate_state(state(speller))


# --- typing ------------------------------------------------------------------------


def test_a_letter_takes_two_selections_then_level_1_again():
    speller = Speller()
    group_target, item_target = targets_for(speller, "H")
    assert speller.select(group_target) is None
    assert speller.text == ""
    assert speller.select(item_target) == "H"
    assert speller.text == "H"
    assert speller.group is None


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_typing_words_and_spaces(slots):
    speller = Speller(slots=slots)
    type_text(speller, "HELLO WORLD")
    assert speller.text == "HELLO WORLD"


def test_space_types_a_space():
    speller = Speller()
    type_text(speller, "A")
    assert type_item(speller, SPACE) == SPACE
    assert speller.text == "A "


def test_backspace_deletes_the_last_character():
    speller = Speller()
    type_text(speller, "AB")
    assert type_item(speller, BACKSPACE) == BACKSPACE
    assert speller.text == "A"
    assert speller.group is None


def test_backspace_on_empty_text_does_nothing():
    speller = Speller()
    type_item(speller, BACKSPACE)
    assert speller.text == ""
    assert speller.group is None


def test_back_closes_the_group_without_typing():
    speller = Speller()
    type_text(speller, "A")
    g, _ = find(MENU, "F")
    speller.select(SLOTS.index(g))
    assert speller.select(SLOTS.index(MENU[g].index(BACK))) == BACK
    assert speller.text == "A"
    assert speller.group is None


def test_no_decision_changes_nothing():
    speller = Speller()
    type_text(speller, "A")
    assert speller.select(None) is None
    assert (speller.text, speller.group) == ("A", None)
    speller.select(2)
    assert speller.select(None) is None
    assert (speller.text, speller.group) == ("A", SLOTS[2])


# --- word suggestions --------------------------------------------------------------


def test_no_suggest_function_means_empty_word_cells():
    speller = Speller()
    assert speller.suggestions == []
    assert word_cell_labels(speller) == ["", "", ""]


def test_suggestions_are_uppercase_and_at_most_three():
    speller = Speller(suggest=recording("hello", "help", "held", "helmet"))
    assert speller.suggestions == ["HELLO", "HELP", "HELD"]
    assert word_cell_labels(speller) == ["HELLO", "HELP", "HELD"]


def test_missing_suggestions_leave_the_last_cells_empty():
    speller = Speller(suggest=recording("yes"))
    assert word_cell_labels(speller) == ["YES", "", ""]


def test_suggest_gets_the_new_text_after_every_change():
    suggest = recording("x")
    speller = Speller(suggest=suggest)
    assert suggest.calls == [""]
    for item, text in [("A", "A"), (SPACE, "A "), ("B", "A B"), (BACKSPACE, "A ")]:
        type_item(speller, item)
        assert suggest.calls[-1] == text


def test_drawing_never_calls_suggest():
    """labels() and next_target() reuse the stored suggestions (a language model is
    too slow to run on every redraw)."""
    suggest = recording("hello")
    speller = Speller(suggest=suggest)
    speller.set_phrase("hello")
    n_calls = len(suggest.calls)
    for _ in range(3):
        speller.labels()
        speller.next_target()
    assert len(suggest.calls) == n_calls


def test_selections_that_leave_the_text_as_it_is_never_call_suggest():
    """Closing a group or selecting an empty word cell gives nothing new to suggest."""
    suggest = recording("yes")
    speller = Speller(suggest=suggest)
    type_text(speller, "A")
    n_calls = len(suggest.calls)
    assert type_item(speller, BACK) == BACK
    assert type_item(speller, WORDS[2]) == WORDS[2]
    assert speller.text == "A"
    assert len(suggest.calls) == n_calls


def test_a_suggestion_replaces_the_word_being_typed():
    speller = Speller(suggest=recording("hello", "world"))
    type_text(speller, "SAY HEL")
    assert type_item(speller, WORDS[0]) == WORDS[0]
    assert speller.text == "SAY HELLO "
    type_item(speller, WORDS[1])
    assert speller.text == "SAY HELLO WORLD "


def test_a_suggestion_on_empty_text():
    speller = Speller(suggest=recording("hello"))
    type_item(speller, WORDS[0])
    assert speller.text == "HELLO "


def test_an_empty_word_cell_types_nothing():
    speller = Speller(suggest=recording("yes"))
    type_text(speller, "A")
    assert type_item(speller, WORDS[2]) == WORDS[2]
    assert speller.text == "A"
    assert speller.group is None


# --- copy-spelling -----------------------------------------------------------------


def test_set_phrase_uppercases_strips_and_resets():
    speller = Speller()
    type_text(speller, "XY")
    speller.select(1)
    speller.set_phrase("  hello world ")
    assert speller.phrase == "HELLO WORLD"
    assert speller.text == ""
    assert speller.group is None


def test_set_phrase_updates_the_suggestions():
    suggest = recording("hi")
    speller = Speller(suggest=suggest)
    type_text(speller, "AB")
    speller.set_phrase("hello")
    assert suggest.calls[-1] == ""


@pytest.mark.parametrize(
    "phrase",
    [
        "hello, world",
        "naïve",
        "3 cats",
        "tab\there",
        f"a{SPACE}b",
        f"a{BACKSPACE}b",
        f"a{BACK}b",
    ],
)
def test_set_phrase_rejects_what_the_menu_cannot_type(phrase):
    with pytest.raises(ValueError):
        Speller().set_phrase(phrase)


def test_set_phrase_rejects_a_space_without_space_in_the_menu():
    menu = [list(group) for group in TOY]
    menu[5][menu[5].index(SPACE)] = BACK
    speller = Speller(menu)
    speller.set_phrase("AB")
    with pytest.raises(ValueError):
        speller.set_phrase("A B")


def test_free_spelling_has_no_target_and_is_never_done():
    speller = Speller()
    assert speller.next_target() is None
    assert not speller.done
    type_text(speller, "HI")
    assert speller.next_target() is None
    assert not speller.done


def test_done_allows_one_trailing_space():
    speller = Speller()
    speller.set_phrase("HI")
    type_text(speller, "H")
    assert not speller.done
    type_text(speller, "I")
    assert speller.done
    assert speller.next_target() is None
    type_text(speller, " ")
    assert speller.done
    type_text(speller, " ")
    assert not speller.done


@pytest.mark.parametrize("slots", [SLOTS, OTHER_SLOTS])
def test_following_next_target_copies_in_two_selections_per_character(slots):
    phrase = "THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG"
    speller = Speller(slots=slots)
    speller.set_phrase(phrase)
    n_selections = 0
    while not speller.done and n_selections <= 2 * len(phrase):
        target = speller.next_target()
        assert target is not None, speller.text
        speller.select(target)
        n_selections += 1
    assert speller.text == phrase
    assert n_selections == 2 * len(phrase)


def test_first_target_is_the_group_of_the_first_letter():
    speller = Speller()
    speller.set_phrase("HELLO")
    group_target, item_target = targets_for(speller, "H")
    assert speller.next_target() == group_target
    speller.select(group_target)
    assert speller.next_target() == item_target


def test_a_wrong_letter_leads_to_backspace():
    speller = Speller()
    speller.set_phrase("HELLO")
    type_text(speller, "HX")
    group_target, item_target = targets_for(speller, BACKSPACE)
    assert speller.next_target() == group_target
    speller.select(group_target)
    assert speller.next_target() == item_target
    speller.select(item_target)
    assert speller.text == "H"
    assert speller.next_target() == targets_for(speller, "E")[0]


def test_an_extra_space_leads_to_backspace():
    speller = Speller()
    speller.set_phrase("HI THERE")
    type_text(speller, "HI  ")
    assert speller.next_target() == targets_for(speller, BACKSPACE)[0]


def test_a_wrong_group_leads_to_back():
    speller = Speller()
    speller.set_phrase("HELLO")
    g, _ = find(MENU, "A")  # H is not in A's group
    speller.select(SLOTS.index(g))
    back_target = SLOTS.index(MENU[g].index(BACK))
    assert speller.next_target() == back_target
    speller.select(back_target)
    assert (speller.text, speller.group) == ("", None)
    assert speller.next_target() == targets_for(speller, "H")[0]


def test_a_wrong_group_without_back_has_no_target():
    speller = Speller(TOY)
    speller.set_phrase("HI")
    speller.select(SLOTS.index(0))  # A-F, no BACK
    assert speller.next_target() is None


def test_backspace_from_inside_its_own_group():
    """After a mistake, at level 2 in the group that holds BACKSPACE, aim at it
    directly (not at BACK)."""
    speller = Speller()
    speller.set_phrase("AB")
    type_text(speller, "X")
    group_target, item_target = targets_for(speller, BACKSPACE)
    speller.select(group_target)
    assert speller.next_target() == item_target


def prefix_suggest(words):
    """Suggests the words that start with the word being typed, in list order."""

    def suggest(text):
        typed = text.split(" ")[-1]
        return [w for w in words if w.startswith(typed)]

    return suggest


def test_next_target_takes_a_suggestion_that_completes_the_word():
    speller = Speller(suggest=prefix_suggest(["THE", "QUICK", "BROWN", "FOX"]))
    speller.set_phrase("THE QUICK BROWN FOX")
    group_target, item_target = targets_for(speller, WORDS[0])  # "THE"
    assert speller.next_target() == group_target
    speller.select(group_target)
    assert speller.next_target() == item_target


def test_following_next_target_with_suggestions_saves_selections():
    """THE, QUICK and BROWN are suggested on an empty word (2 selections each); FOX is
    only 4th, so F is typed first and then FOX is suggested: 10 selections in all."""
    phrase = "THE QUICK BROWN FOX"
    speller = Speller(suggest=prefix_suggest(["THE", "QUICK", "BROWN", "FOX"]))
    speller.set_phrase(phrase)
    n_selections = 0
    while not speller.done and n_selections <= 2 * len(phrase):
        speller.select(speller.next_target())
        n_selections += 1
    assert speller.text == phrase + " "
    assert n_selections == 10


def test_a_suggestion_can_fix_a_mistake():
    """HX + HELLO -> "HELLO ", on track again: shorter than backspace + typing."""
    speller = Speller(suggest=recording("hello"))
    speller.set_phrase("HELLO")
    type_text(speller, "HX")
    group_target, item_target = targets_for(speller, WORDS[0])
    assert speller.next_target() == group_target
    speller.select(group_target)
    assert speller.next_target() == item_target


def test_a_suggestion_that_would_go_wrong_is_ignored():
    speller = Speller(suggest=recording("help"))
    speller.set_phrase("HELLO")
    type_text(speller, "HEL")
    assert speller.next_target() == targets_for(speller, "L")[0]


@pytest.mark.parametrize("seed", range(5))
def test_a_noisy_subject_still_finishes_the_phrase(seed):
    """One selection in five goes to a random target; following next_target (a random
    one when there is none) still ends with the phrase typed."""
    rng = np.random.default_rng(seed)
    speller = Speller()
    speller.set_phrase("HELLO WORLD")
    for _ in range(1000):
        if speller.done:
            break
        target = speller.next_target()
        if target is None or rng.random() < 0.2:
            target = int(rng.integers(6))
        speller.select(target)
    assert speller.done, speller.text


def test_copy_states_pass_protocol_validation():
    speller = Speller(suggest=prefix_suggest(["HELLO", "WORLD"]))
    speller.set_phrase("hello world")
    for _ in range(2 * len(speller.phrase)):
        if speller.done:
            break
        protocol.validate_state(state(speller))
        speller.select(speller.next_target())
    assert speller.done, speller.text
    protocol.validate_state(state(speller))
