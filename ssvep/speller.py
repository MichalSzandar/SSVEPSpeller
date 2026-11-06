"""Speller state machine: a two-level menu of 6 groups x 6 items (Hex-o-Spell style).

Level 1 shows the 6 groups; selecting one opens it. Level 2 shows the items of the open
group; selecting one acts on the text and returns to level 1. No EEG and no LSL here:
scripts/online.py feeds in the classifier's decisions and sends what this class shows
to the GUI (ssvep/protocol.py).

Two index spaces (config.TARGET_SLOTS):

    slot    position on the screen, clockwise from the top left. The menu is in slot
            order: group menu[s] is drawn in slot s, and at level 2 item menu[g][i] of
            the open group g is drawn in slot i, so the alphabet runs clockwise.
    target  index into config.TARGET_FREQS: what the classifier returns and what the
            protocol uses. Target k is drawn in slot slots[k].

Every method takes and returns targets; only the menu lookups use slots.

Items:

    "A" ... "Z"  type that letter
    SPACE        type a space
    BACKSPACE    delete the last character (nothing if the text is empty)
    BACK         close the group without typing (undoes a wrong group selection)
    WORDS[k]     accept the k-th word suggestion: it replaces the word being typed
                 (everything after the last space) and a space follows it. With fewer
                 than k + 1 suggestions the cell is empty and selecting it types nothing.

Copy-spelling: with a phrase set, next_target() says which target the subject should
select next. The GUI cues it (if asked) and logs it as ground truth, and the fake EEG
stream answers with its frequency.
"""

from collections.abc import Callable

from ssvep import config

SPACE = "␣"
BACKSPACE = "⌫"
BACK = "↶"
WORDS = ("#1", "#2", "#3")  # placeholders for the word suggestions, best first

# 26 letters + space + backspace + 3 suggestions = 31 items in 36 cells; the other 5
# cells are BACK. U-Z fill a whole group, so that group has none.
MENU = [
    ["A", "B", "C", "D", "E", BACK],
    ["F", "G", "H", "I", "J", BACK],
    ["K", "L", "M", "N", "O", BACK],
    ["P", "Q", "R", "S", "T", BACK],
    ["U", "V", "W", "X", "Y", "Z"],
    [*WORDS, SPACE, BACKSPACE, BACK],
]


class Speller:
    """Typed text, menu level and copy phrase of one speller session.

    Attributes:
        menu         6 groups of 6 items, slot order (module docstring)
        slots        target k is drawn in slot slots[k]
        text         typed text so far, uppercase
        phrase       phrase to copy, uppercase ("" in free spelling)
        group        slot of the open group at level 2, None at level 1
        suggestions  the words in the WORDS cells for the current text: uppercase, best
                     first, at most len(WORDS); [] without ``suggest``
    """

    def __init__(
        self,
        menu: list[list[str]] = MENU,
        slots: list[int] = config.TARGET_SLOTS,
        suggest: Callable[[str], list[str]] | None = None,
    ):
        """``suggest(text)`` returns suggested words for ``text``, best first (the
        language model, Sprint 3); None means no suggestions (the without-NLP
        condition)."""
        self.menu = menu
        self.slots = list(slots)
        self.suggest = suggest
        self.text = ""
        self.phrase = ""
        self.group = None
        self.suggestions = []
        self.update_suggestions()

    def update_suggestions(self) -> None:
        """Set ``suggestions`` from ``suggest(text)``.

        Called whenever the text changes, so ``suggest`` (slow for a language model)
        runs once per change, never just to draw the labels.
        """
        if self.suggest is None:
            self.suggestions = []
        else:
            words = self.suggest(self.text)[: len(WORDS)]
            self.suggestions = [word.upper() for word in words]

    def set_phrase(self, phrase: str) -> None:
        """Start copying ``phrase`` ("" for free spelling).

        Stores it uppercase without leading or trailing spaces, clears the text,
        returns to level 1 and updates the suggestions. Raises ValueError if the phrase
        has a character the menu cannot type: anything but the menu's letters, and
        spaces if the menu has SPACE.
        """
        phrase = phrase.strip().upper()
        items = {item for group in self.menu for item in group}
        for c in phrase:
            if c == " ":
                typeable = SPACE in items
            else:
                typeable = c.isalpha() and c in items
            if not typeable:
                raise ValueError(f"cannot type {c!r} with this menu")
        self.phrase = phrase
        self.text = ""
        self.group = None
        self.update_suggestions()

    def _item_label(self, item: str) -> str:
        """What the GUI draws for ``item``: the item itself, except WORDS[k]."""
        if item in WORDS:
            i = WORDS.index(item)
            return self.suggestions[i] if i < len(self.suggestions) else ""
        return item

    def _group_labels(self, group: list[str]) -> list[str]:
        """Labels of one group (slot order), in target order."""
        return [self._item_label(group[s]) for s in self.slots]

    def labels(self) -> list:
        """What the GUI draws on each target, in target order (``labels`` in
        ssvep/protocol.py).

        Level 1: every group as the list of its 6 item labels, also in target order
        (the preview of level 2). Level 2: the open group's item labels. An item's label
        is the item itself, except WORDS[k]: the k-th suggestion, or "" without one.
        """
        if self.group is None:
            return [self._group_labels(self.menu[s]) for s in self.slots]
        else:
            return self._group_labels(self.menu[self.group])

    def _text_with_word(self, word: str) -> str:
        """The text after accepting ``word``: it replaces the word being typed
        (everything after the last space), and a space follows it."""
        last_space = self.text.rfind(" ")  # -1 without a space: the slice is then empty
        return self.text[: last_space + 1] + word + " "

    def select(self, target: int | None) -> str | None:
        """Apply the classifier's decision ``target`` -> the item chosen at level 2.

        Level 1: open the group drawn on ``target`` and return None. Level 2: act on
        the item drawn on ``target`` (module docstring), return to level 1 and return
        the item. ``target`` None (no decision) changes nothing and returns None.
        """
        if target is None:
            return None
        slot = self.slots[target]
        if self.group is None:
            self.group = slot
            return None
        else:
            item = self.menu[self.group][slot]
            old_text = self.text
            if item in WORDS:
                i = WORDS.index(item)
                if i < len(self.suggestions):
                    self.text = self._text_with_word(self.suggestions[i])
            elif item == SPACE:
                self.text += " "
            elif item == BACKSPACE:
                self.text = self.text[:-1]
            elif item == BACK:
                pass
            else:
                self.text += item
            self.group = None
            if self.text != old_text:
                self.update_suggestions()
            return item

    @property
    def done(self) -> bool:
        """True once the text is the phrase, with or without one trailing space;
        always False in free spelling."""
        if self.phrase == "":
            return False
        return self.text == self.phrase or self.text == self.phrase + " "

    def next_target(self) -> int | None:
        """Target the subject should select next to copy the phrase.

        "On track" means the text is a start of the phrase followed by one space. The
        item to aim for is
          - a suggestion that leaves the text on track, if there is one;
          - else, while the text is on track, the phrase's next character (SPACE for a
            space);
          - else (a mistake) BACKSPACE.
        At level 1 the target is the group holding that item; at level 2 the item, if
        the open group holds it, else BACK. Returns None in free spelling, once the
        phrase is done, and at level 2 in a group that holds neither the item nor BACK.
        """
        if self.phrase == "" or self.done:
            return None
        goal = self.phrase + " "
        for i, suggestion in enumerate(self.suggestions):
            if goal.startswith(self._text_with_word(suggestion)):
                item = WORDS[i]
                break
        else:
            if goal.startswith(self.text):
                next_char = self.phrase[len(self.text)]
                item = SPACE if next_char == " " else next_char
            else:
                item = BACKSPACE
        if self.group is None:
            for g, group in enumerate(self.menu):
                if item in group:
                    return self.slots.index(g)
            return None
        else:
            group = self.menu[self.group]
            if item in group:
                return self.slots.index(group.index(item))
            elif BACK in group:
                return self.slots.index(group.index(BACK))
            else:
                return None
