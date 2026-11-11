"""Word prediction for the speller: unigram prefix completion and transformer re-ranking.

Both suggesters are callables ``suggest(text) -> list[str]`` that plug into
``Speller(suggest=...)`` (ssvep/speller.py):

    UnigramModel         the k most frequent vocabulary words that start with the word
                         being typed (the baseline: no context)
    TransformerReranker  the unigram's best n candidates for that prefix, re-ordered by
                         a causal language model (distilgpt2) given the words typed
                         before it

``UnigramModel.char_probs`` gives P(next character | text) over the letters and space:
the language-model prior for the fusion with the classifier (Speier et al. 2016).

Text follows the speller's conventions: uppercase A-Z and single spaces. Suggestions
are uppercase too. The word being typed (the "prefix") is everything after the last
space; the "context" is everything up to and including that space.

wordfreq, transformers and torch are optional (the ``nlp`` extra and the CPU wheel of
torch). Importing this module must not need them, so the rest of the package and the
without-NLP condition run without them.
"""

from collections.abc import Callable

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = ALPHABET + " "  # what char_probs gives a probability for

DEFAULT_MODEL = "distilbert/distilgpt2"
KINDS = ("none", "unigram", "gpt2")  # make_suggester's choices, for --nlp


def split_text(text: str) -> tuple[str, str]:
    """Split the typed text into (context, prefix).

    prefix is the word being typed: everything after the last space ("" right after
    a space). context is everything before it, including that space.

        "" -> ("", "")      "HEL" -> ("", "HEL")
        "HELLO " -> ("HELLO ", "")      "HELLO WOR" -> ("HELLO ", "WOR")
    """
    index = text.rfind(" ")
    if index == -1:
        return "", text

    prefix = text[index + 1 :]
    context = text[: index + 1]

    return context, prefix


def load_vocabulary(n_words: int = 50_000, lang: str = "en") -> dict[str, float]:
    """The speller-typeable words among wordfreq's ``n_words`` most frequent words.

    Returns word -> wordfreq frequency, most frequent first. Words are uppercase and
    consist of A-Z only: words with digits, apostrophes, hyphens or accents are left
    out, so the result can have fewer than ``n_words`` entries.
    """
    from wordfreq import word_frequency, top_n_list

    words = top_n_list(lang, n_words)
    vocabulary = {}
    for word in words:
        if word.isalpha() and word.isascii():
            vocabulary[word.upper()] = word_frequency(word, lang)
    return vocabulary


class UnigramModel:
    """Prefix completion by word frequency: the baseline suggester."""

    def __init__(self, vocabulary: dict[str, float], k: int = 3):
        """``vocabulary``: word -> frequency, words uppercase A-Z, in any order
        (load_vocabulary, or a toy dict in the tests). ``k``: suggestions per call."""
        self.vocabulary = vocabulary
        self.k = k

    def candidates(self, prefix: str, n: int) -> list[str]:
        """The ``n`` most frequent vocabulary words that start with ``prefix``, most
        frequent first; equal frequencies keep the vocabulary's order.

        ``prefix`` itself is never a candidate: accepting it would only add a space,
        which SPACE already does. Fewer than ``n`` if fewer words match, [] if none.
        """
        candidates = [
            word
            for word in self.vocabulary
            if word.startswith(prefix) and word != prefix
        ]
        candidates.sort(key=lambda w: self.vocabulary[w], reverse=True)
        return candidates[:n]

    def __call__(self, text: str) -> list[str]:
        """Suggestions for ``text``: the best ``k`` candidates for its prefix."""
        _, prefix = split_text(text)
        return self.candidates(prefix, self.k)

    def char_probs(self, text: str) -> dict[str, float]:
        """P(next character | text) for every character in CHARS (27 keys).

        From the word frequencies, for the prefix of ``text``: a letter c gets the
        total frequency of the vocabulary words that start with prefix + c; " " gets
        the frequency of the prefix itself as a vocabulary word (0 if it is none, and
        for an empty prefix). Normalised to sum to 1. If that total is 0 (the prefix
        starts no vocabulary word: a typo, or a word the vocabulary lacks), every
        character gets 1/27.
        """
        _, prefix = split_text(text)
        total = 0.0
        probs = {}
        for c in CHARS:
            if c == " ":
                freq = self.vocabulary.get(prefix, 0.0)
            else:
                freq = sum(
                    freq
                    for word, freq in self.vocabulary.items()
                    if word.startswith(prefix + c)
                )
            probs[c] = freq
            total += freq

        if total > 0.0:
            for c in CHARS:
                probs[c] /= total
        else:
            for c in CHARS:
                probs[c] = 1.0 / len(CHARS)

        return probs


def model_text(text: str) -> str:
    """The typed text as normally written English, for the language model: lowercase,
    except the first letter and the word "I", which are capitals. The input's case
    doesn't matter; words keep their single spaces.

        "I LIVE IN NEW YORK" -> "I live in new york"
        "HELLO WORLD" -> "Hello world"      "THANK YOU AND I" -> "Thank you and I"
        "" -> ""
    """
    words = text.split(" ")
    if not words:
        return ""
    words[0] = words[0].capitalize()
    for i in range(1, len(words)):
        if words[i].upper() == "I":
            words[i] = "I"
        else:
            words[i] = words[i].lower()
    return " ".join(words)


class CausalLMScorer:
    """Scores candidate words given the typed context with a causal language model
    (Hugging Face transformers, CPU)."""

    def __init__(self, model_name: str = DEFAULT_MODEL, max_context_words: int = 20):
        """Load the tokenizer and the model (downloaded on first use, then cached) and
        put the model in evaluation mode. Only the last ``max_context_words`` words of
        the context are used."""
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
        )  # pylint: disable=import-error

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.max_context_words = max_context_words

    def __call__(self, context: str, words: list[str]) -> list[float]:
        """log P(word | context) for each word in ``words``, in order.

        The scored text is the model's beginning-of-text token followed by
        model_text(context + word), with the context cut to its last
        ``max_context_words`` words. A word's score is the sum of the log-probabilities
        of its own tokens, each given everything before it, so every score is <= 0.
        With an empty context the word starts the text. Upper- or lowercase input gives
        the same scores, and scoring several words at once gives the same scores as
        scoring each one alone. [] for no words.
        """
        import torch  # pylint: disable=import-error

        if not words:
            return []

        context_words = context.split()
        if len(context_words) > self.max_context_words:
            context_words = context_words[-self.max_context_words :]
        context = " ".join(context_words)

        scores = []
        for word in words:
            text = model_text(context + " " + word) if context else model_text(word)
            context_text = text[: len(context)]  # "I live in new"
            word_text = text[len(context) :]  # " york", or "The" with no context
            context_ids = [self.tokenizer.bos_token_id] + self.tokenizer.encode(
                context_text
            )
            word_ids = self.tokenizer.encode(word_text)
            input_ids = torch.tensor([context_ids + word_ids])
            # -100 = ignored by the loss, so only the word's own tokens are scored
            labels = torch.tensor([[-100] * len(context_ids) + word_ids])
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, labels=labels)
                loss = outputs.loss.item()  # mean over the word's tokens
            scores.append(-loss * len(word_ids))  # sum of their log-probabilities

        return scores


class TransformerReranker:
    """The unigram's candidates for the prefix, ordered by a language model's score
    given the context."""

    def __init__(
        self,
        unigram: UnigramModel,
        scorer: Callable[[str, list[str]], list[float]],
        n_candidates: int = 30,
        k: int = 3,
    ):
        """``scorer(context, words)`` returns one score per word, higher = more likely
        (a CausalLMScorer, or a fake one in the tests). ``n_candidates``: how many of
        the unigram's candidates to re-rank. ``k``: suggestions per call."""
        self.unigram = unigram
        self.scorer = scorer
        self.n_candidates = n_candidates
        self.k = k

    def __call__(self, text: str) -> list[str]:
        """The best ``k`` of ``unigram.candidates(prefix, n_candidates)``, highest
        score first; equal scores keep the unigram order.

        The scorer gets the context exactly as split_text returns it. Without
        candidates the result is [] and the scorer is not called.
        """
        context, prefix = split_text(text)
        candidates = self.unigram.candidates(prefix, self.n_candidates)
        if not candidates:
            return []
        scores = self.scorer(context, candidates)
        # Sort by score (descending) and return the top k
        sorted_pairs = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_pairs[: self.k]]


def make_suggester(kind: str, k: int = 3) -> Callable[[str], list[str]] | None:
    """The suggester for ``online.py --nlp``: "none" -> None (no suggestions),
    "unigram" -> UnigramModel on load_vocabulary(), "gpt2" -> TransformerReranker
    with that UnigramModel and a CausalLMScorer. Raises ValueError for any other
    ``kind``."""
    if kind == "none":
        return None
    if kind == "unigram":
        vocabulary = load_vocabulary()
        return UnigramModel(vocabulary, k)
    if kind == "gpt2":
        vocabulary = load_vocabulary()
        unigram = UnigramModel(vocabulary, k)
        scorer = CausalLMScorer()
        return TransformerReranker(unigram, scorer, n_candidates=30, k=k)
    raise ValueError(f"Unknown suggester kind {kind!r}, must be one of {KINDS}")
