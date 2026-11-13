"""Fusion of the classifier with the language model: P(target | EEG, typed text).

Speier, Arnold & Pouratian (2016): the classifier says how likely each target is from
the EEG, the language model how likely each target is from the text typed so far, and
Bayes' rule combines them:

    posterior(k)  ∝  softmax(scores / T)[k]  ·  prior(k)

    scores   the classifier's decision_function for one trial (n_targets,)
    T        a temperature, fitted on the calibration scores so that the softmax
             probabilities match how often the classifier is right
    prior    the language model's probability of each target in the speller's current
             state, with a floor: (1 - floor) * LM prior + floor / n_targets, so the
             EEG can always overrule the language model (names, typos, corrections)

The selected target is the one with the highest posterior.

The language-model prior comes from the "intended word" model. The user is typing some
vocabulary word W that starts with the current prefix, with probability proportional
to W's frequency. Next they select the item that gets them closer to W: W's word cell
if W is one of the suggestions, SPACE if W is the prefix itself, else W's next letter.
A fixed probability ``error`` goes to the correction items: BACKSPACE among the items,
and BACK at level 2 (the open group was the wrong one). Groups and cells then add up
the probabilities of their items (target_prior).

Item names are those of ssvep/speller.py: "A" ... "Z", SPACE, BACKSPACE, BACK and
WORDS[k].
"""

import numpy as np
from scipy.optimize import minimize_scalar

from ssvep.language import ALPHABET, split_text
from ssvep.speller import BACK, BACKSPACE, SPACE, WORDS, Speller

FLOOR = 0.3  # share of the prior spread evenly over the targets
ERROR = 0.1  # prior probability of a correction: BACKSPACE, and BACK at level 2


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """exp(scores / temperature), normalised to sum to 1 along the last axis.

    Works for one trial (n_targets,) and for many (n_trials, n_targets), and for any
    score size: scores of 1000 must not overflow. A higher temperature gives flatter
    probabilities, a lower one sharper.
    """
    z = scores / temperature
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def fit_temperature(
    scores: np.ndarray, y: np.ndarray, bounds: tuple[float, float] = (0.01, 100.0)
) -> float:
    """The temperature T within ``bounds`` that minimises the mean negative
    log-likelihood of the true labels, -mean(log softmax(scores / T)[i, y[i]]).

    scores : (n_trials, n_targets), the classifier's scores of calibration trials it
             was not trained on (cross-validated)
    y      : (n_trials,), the true targets

    Scaling all scores by c scales the fitted T by c. Perfectly separated scores have
    no best T (the smaller, the better): then the result is some small T that makes
    every true target almost certain.
    """

    def loss(t):
        return -np.mean(np.log(softmax(scores, t)[np.arange(len(y)), y]))

    return minimize_scalar(loss, bounds=bounds).x


def item_probs(
    text: str,
    suggestions: list[str],
    vocabulary: dict[str, float],
    error: float = ERROR,
) -> dict[str, float]:
    """The language model's probability of each item the user may select next.

    Keys: the 26 letters, SPACE, BACKSPACE and WORDS[k] for every k <
    len(suggestions) (the word cells that hold a word). The values sum to 1.

    BACKSPACE gets ``error``. The other 1 - ``error`` is split in proportion to the
    vocabulary frequencies of the words W that start with the prefix of ``text``
    (language.split_text; the context before it is not used). W's frequency goes to:
      - WORDS[k] if W is suggestions[k],
      - else SPACE if W is the prefix itself,
      - else the letter that follows the prefix in W.
    If no vocabulary word starts with the prefix (a typo, or a word the vocabulary
    lacks), the 1 - ``error`` is split evenly over the 26 letters and SPACE, as in
    UnigramModel.char_probs. A suggestion that is not in the vocabulary gets 0.

    Without suggestions, the letters and SPACE get (1 - error) times
    UnigramModel(vocabulary).char_probs(text) (with SPACE for " ").
    """
    probs = {c: 0.0 for c in ALPHABET}
    probs[SPACE] = 0.0
    probs.update({WORDS[k]: 0.0 for k in range(len(suggestions))})

    _, prefix = split_text(text)
    for word, freq in vocabulary.items():
        if not word.startswith(prefix):
            continue
        if word in suggestions:
            k = suggestions.index(word)
            probs[WORDS[k]] += freq
        elif word == prefix:
            probs[SPACE] += freq
        else:
            next_letter = word[len(prefix)]
            probs[next_letter] += freq

    total = sum(probs.values())
    if total > 0:
        for k in probs:
            probs[k] *= (1 - error) / total
    else:
        for key in list(ALPHABET) + [SPACE]:
            probs[key] = (1 - error) / 27

    probs[BACKSPACE] = error
    return probs


def target_prior(
    speller: Speller,
    probs: dict[str, float],
    error: float = ERROR,
    floor: float = FLOOR,
) -> np.ndarray:
    """The prior over the targets in the speller's current state -> (n_targets,),
    target order, sums to 1.

    ``probs``: item probabilities as item_probs returns them; an item without a key
    (BACK, an empty word cell) has probability 0.

    The language-model part, before the floor:
      - level 1: each target gets the summed probabilities of the items of the group
        drawn on it;
      - level 2: BACK gets ``error``, and the other items of the open group share the
        rest (1 - error, or all of it in a group without BACK) in proportion to their
        probabilities; evenly if those are all 0.
    Then (1 - floor) * that + floor / n_targets.

    Targets and slots as in Speller: target k shows menu[slots[k]] at level 1, and
    item slots[k] of the open group at level 2.
    """
    slots = speller.slots
    n_targets = len(slots)
    prior = np.zeros(n_targets, dtype=float)
    if speller.group is None:
        for target in range(n_targets):
            group = speller.menu[slots[target]]
            prior[target] = sum(probs.get(item, 0.0) for item in group)
    else:
        open_group = speller.menu[speller.group]
        items = [
            open_group[slots[target]] for target in range(n_targets)
        ]  # target order
        if BACK in items:
            prior[items.index(BACK)] = error
            remaining = 1 - error
        else:
            remaining = 1.0

        n_items = sum(item != BACK for item in items)
        total = sum(probs.get(item, 0.0) for item in items if item != BACK)
        for target, item in enumerate(items):
            if item == BACK:
                continue
            if total > 0:
                prior[target] = remaining * probs.get(item, 0.0) / total
            else:
                prior[target] = remaining / n_items

    return (1 - floor) * prior + floor / n_targets


def fuse(eeg_probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """eeg_probs * prior, normalised to sum to 1 along the last axis.

    Both are probabilities over the targets; either may also be (n_trials, n_targets)
    and is then broadcast like NumPy does.
    """
    return eeg_probs * prior / np.sum(eeg_probs * prior, axis=-1, keepdims=True)


class Fusion:
    """Classifier + language model, one trial at a time (scripts/online.py)."""

    def __init__(
        self,
        temperature: float,
        vocabulary: dict[str, float],
        floor: float = FLOOR,
        error: float = ERROR,
    ):
        """``temperature``: fit_temperature on the calibration scores.
        ``vocabulary``: word -> frequency, as language.load_vocabulary returns it.
        ``floor`` and ``error``: as in target_prior and item_probs. Stores all four
        under the same names."""
        self.temperature = temperature
        self.vocabulary = vocabulary
        self.floor = floor
        self.error = error

    def prior(self, speller: Speller) -> np.ndarray:
        """target_prior for the speller's current state (its text, suggestions and
        level) -> (n_targets,)."""
        probs = item_probs(
            speller.text, speller.suggestions, self.vocabulary, error=self.error
        )
        return target_prior(speller, probs, error=self.error, floor=self.floor)

    def decide(
        self, scores: np.ndarray, speller: Speller
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """One trial's classifier scores (n_targets,) and the speller in the state the
        trial was made in -> (selected target, prior, posterior).

        posterior = fuse(softmax(scores, temperature), prior(speller)); the selected
        target is its argmax, as a plain int.
        """
        prior = self.prior(speller)
        posterior = fuse(softmax(scores, self.temperature), prior)
        return int(np.argmax(posterior)), prior, posterior
