import importlib.util
import math
import re
import subprocess
import sys

import pytest

from ssvep import language
from ssvep.language import (
    CHARS,
    KINDS,
    CausalLMScorer,
    TransformerReranker,
    UnigramModel,
    load_vocabulary,
    make_suggester,
    model_text,
    split_text,
)
from ssvep.speller import Speller

needs_wordfreq = pytest.mark.skipif(
    importlib.util.find_spec("wordfreq") is None,
    reason="wordfreq not installed (pip install -e '.[nlp]')",
)
needs_transformers = pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None
    or importlib.util.find_spec("torch") is None,
    reason="transformers/torch not installed (the nlp extra + CPU torch)",
)

# Deliberately not sorted by frequency; THERE and THEY tie (30.0), THERE first.
TOY = {
    "HELLO": 5.0,
    "THE": 100.0,
    "A": 90.0,
    "TO": 80.0,
    "HE": 50.0,
    "THERE": 30.0,
    "THEY": 30.0,
    "THEN": 20.0,
    "AN": 15.0,
    "TOO": 10.0,
    "HELP": 8.0,
    "WORLD": 7.0,
}


# --- split_text ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ("", "")),
        ("HEL", ("", "HEL")),
        ("HELLO ", ("HELLO ", "")),
        ("HELLO WOR", ("HELLO ", "WOR")),
        ("A B C", ("A B ", "C")),
        ("A B C ", ("A B C ", "")),
    ],
)
def test_split_text(text, expected):
    """The prefix is everything after the last space, the context everything up to and
    including it."""
    assert split_text(text) == expected


def test_split_text_parts_add_up_to_the_text():
    for text in ["", "X", "HELLO WORLD", "HELLO WORLD ", "A  B"]:
        context, prefix = split_text(text)
        assert context + prefix == text
        assert " " not in prefix


# --- model_text -----------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("", ""),
        ("HELLO", "Hello"),
        ("HELLO WORLD", "Hello world"),
        ("I LIVE IN NEW YORK", "I live in new york"),
        ("THANK YOU AND I", "Thank you and I"),
        ("I", "I"),
        ("A", "A"),
        ("WHAT IF I IT IS", "What if I it is"),
        ("HELLO ", "Hello "),
    ],
)
def test_model_text(text, expected):
    """Lowercase, except the first letter and the word I."""
    assert model_text(text) == expected


def test_model_text_ignores_the_input_case():
    for text in ["hello i am here", "Hello I Am Here", "HELLO I AM HERE"]:
        assert model_text(text) == "Hello I am here"


def test_model_text_only_capitalises_the_whole_word_i():
    """Words that merely start or end with I stay lowercase."""
    assert model_text("HI IN IT I") == "Hi in it I"


# --- UnigramModel.candidates ----------------------------------------------------


def test_candidates_most_frequent_first():
    model = UnigramModel(TOY)
    assert model.candidates("TH", 10) == ["THE", "THERE", "THEY", "THEN"]


def test_candidates_does_not_need_a_sorted_vocabulary():
    """HELLO (5) comes before HELP (8) in the dict, but HELP is more frequent."""
    assert UnigramModel(TOY).candidates("HEL", 10) == ["HELP", "HELLO"]


def test_candidates_never_include_the_prefix_itself():
    model = UnigramModel(TOY)
    assert model.candidates("THE", 10) == ["THERE", "THEY", "THEN"]
    assert model.candidates("HELLO", 10) == []


def test_candidates_limited_to_n():
    model = UnigramModel(TOY)
    assert model.candidates("TH", 2) == ["THE", "THERE"]
    assert model.candidates("TH", 0) == []


def test_candidates_ties_keep_vocabulary_order():
    """Equal frequencies keep the order of the vocabulary dict, not the alphabet."""
    model = UnigramModel({"ZB": 1.0, "ZC": 2.0, "ZA": 1.0})
    assert model.candidates("Z", 3) == ["ZC", "ZB", "ZA"]


def test_candidates_empty_prefix_gives_the_most_frequent_words():
    assert UnigramModel(TOY).candidates("", 3) == ["THE", "A", "TO"]


def test_candidates_unknown_prefix():
    assert UnigramModel(TOY).candidates("XYZ", 5) == []


def test_candidates_does_not_change_the_vocabulary():
    vocabulary = dict(TOY)
    model = UnigramModel(vocabulary)
    model.candidates("T", 5)
    model("HELLO T")
    assert vocabulary == TOY
    assert list(vocabulary) == list(TOY)


# --- UnigramModel.__call__ ------------------------------------------------------


def test_unigram_suggests_completions_of_the_word_being_typed():
    model = UnigramModel(TOY)
    assert model("HELLO TH") == ["THE", "THERE", "THEY"]
    assert model("TH") == ["THE", "THERE", "THEY"]


def test_unigram_after_a_space_suggests_frequent_words():
    assert UnigramModel(TOY)("HELLO ") == ["THE", "A", "TO"]
    assert UnigramModel(TOY)("") == ["THE", "A", "TO"]


def test_unigram_returns_at_most_k_words():
    assert UnigramModel(TOY, k=2)("T") == ["THE", "TO"]
    assert UnigramModel(TOY, k=5)("HEL") == ["HELP", "HELLO"]
    assert UnigramModel(TOY)("QQ") == []


def test_unigram_plugs_into_the_speller():
    """Speller(suggest=...) shows the unigram's suggestions after every change."""
    speller = Speller(suggest=UnigramModel(TOY))
    assert speller.suggestions == ["THE", "A", "TO"]
    speller.set_phrase("THEN")
    for _ in range(4):  # T, H: group + item each
        speller.select(speller.next_target())
    assert speller.text == "TH"
    assert speller.suggestions == ["THE", "THERE", "THEY"]


def _selections_to_copy(phrase, suggest, limit=200):
    """Selections a perfect user needs to copy ``phrase`` (bounded)."""
    speller = Speller(suggest=suggest)
    speller.set_phrase(phrase)
    for n in range(limit):
        if speller.done:
            return n
        speller.select(speller.next_target())
    raise AssertionError(f"{phrase!r} not copied within {limit} selections")


def test_unigram_suggestions_save_selections():
    """Copying with suggestions takes fewer selections than letter by letter."""
    phrase = "HELLO THERE"
    without = _selections_to_copy(phrase, None)
    with_unigram = _selections_to_copy(phrase, UnigramModel(TOY))
    assert without == 2 * len(phrase)  # every character is 2 selections
    assert with_unigram < without


# --- UnigramModel.char_probs ----------------------------------------------------


def test_char_probs_is_a_distribution_over_letters_and_space():
    model = UnigramModel(TOY)
    for text in ["", "T", "TH", "THE", "HELLO ", "HELLO WOR", "QQ"]:
        probs = model.char_probs(text)
        assert set(probs) == set(CHARS)
        assert len(CHARS) == 27
        assert all(p >= 0 for p in probs.values())
        assert math.isclose(sum(probs.values()), 1.0)


def test_char_probs_single_continuation():
    """Every vocabulary word starting with TH continues with E."""
    probs = UnigramModel(TOY).char_probs("TH")
    assert probs["E"] == pytest.approx(1.0)
    assert probs[" "] == 0.0


def test_char_probs_space_gets_the_prefix_as_a_word():
    """After THE: THE itself (100) -> space; THERE (30) -> R; THEY (30) -> Y;
    THEN (20) -> N."""
    probs = UnigramModel(TOY).char_probs("THE")
    total = 100 + 30 + 30 + 20
    assert probs[" "] == pytest.approx(100 / total)
    assert probs["R"] == pytest.approx(30 / total)
    assert probs["Y"] == pytest.approx(30 / total)
    assert probs["N"] == pytest.approx(20 / total)
    assert probs["A"] == 0.0


def test_char_probs_at_the_start_of_a_word():
    """Empty prefix: each letter gets the frequency of the words starting with it, and
    space gets nothing."""
    probs = UnigramModel(TOY).char_probs("HELLO ")
    t = 100 + 80 + 30 + 30 + 20 + 10  # THE TO THERE THEY THEN TOO
    a = 90 + 15  # A AN
    h = 5 + 50 + 8  # HELLO HE HELP
    w = 7  # WORLD
    total = t + a + h + w
    assert probs["T"] == pytest.approx(t / total)
    assert probs["A"] == pytest.approx(a / total)
    assert probs["H"] == pytest.approx(h / total)
    assert probs["W"] == pytest.approx(w / total)
    assert probs[" "] == 0.0
    assert probs == pytest.approx(UnigramModel(TOY).char_probs(""))


def test_char_probs_unknown_prefix_is_uniform():
    probs = UnigramModel(TOY).char_probs("QQ")
    assert all(p == pytest.approx(1 / 27) for p in probs.values())


def test_char_probs_only_looks_at_the_word_being_typed():
    model = UnigramModel(TOY)
    assert model.char_probs("HELLO TH") == pytest.approx(model.char_probs("TH"))


# --- TransformerReranker (with a fake scorer) -----------------------------------


class FakeScorer:
    """Fixed score per word (-10 for unknown words); records every call."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, context, words):
        self.calls.append((context, list(words)))
        return [self.scores.get(word, -10.0) for word in words]


def test_reranker_orders_candidates_by_score():
    """Unigram order for T: THE TO THERE THEY THEN TOO. The scorer prefers TOO, then
    THEN; the rest tie, so the third suggestion is the unigram's first, THE."""
    scorer = FakeScorer({"TOO": -1.0, "THEN": -2.0})
    reranker = TransformerReranker(UnigramModel(TOY), scorer, n_candidates=10)
    assert reranker("T") == ["TOO", "THEN", "THE"]


def test_reranker_only_considers_n_candidates():
    """With 2 candidates (THE, TO), TOO can't be suggested however well it scores."""
    scorer = FakeScorer({"TOO": 0.0, "TO": -1.0, "THE": -2.0})
    reranker = TransformerReranker(UnigramModel(TOY), scorer, n_candidates=2)
    assert reranker("T") == ["TO", "THE"]
    assert scorer.calls == [("", ["THE", "TO"])]


def test_reranker_passes_context_and_candidates_to_the_scorer():
    scorer = FakeScorer({})
    reranker = TransformerReranker(UnigramModel(TOY), scorer, n_candidates=3)
    reranker("HELLO WORLD T")
    assert scorer.calls == [("HELLO WORLD ", ["THE", "TO", "THERE"])]


def test_reranker_returns_at_most_k():
    scorer = FakeScorer({})
    assert len(TransformerReranker(UnigramModel(TOY), scorer, k=2)("T")) == 2
    assert TransformerReranker(UnigramModel(TOY), scorer, k=3)("T") == [
        "THE",
        "TO",
        "THERE",
    ]


def test_reranker_without_candidates_does_not_call_the_scorer():
    scorer = FakeScorer({})
    reranker = TransformerReranker(UnigramModel(TOY), scorer)
    assert reranker("HELLO QQ") == []
    assert reranker("HELLO") == []  # HELLO is the only H-E-L-L-O word: no completion
    assert not scorer.calls


def test_reranker_never_suggests_the_prefix_itself():
    scorer = FakeScorer({"THE": 0.0})
    assert "THE" not in TransformerReranker(UnigramModel(TOY), scorer)("THE")


# --- make_suggester -------------------------------------------------------------


def test_kinds():
    assert set(KINDS) == {"none", "unigram", "gpt2"}


def test_make_suggester_none():
    assert make_suggester("none") is None


def test_make_suggester_unknown_kind():
    with pytest.raises(ValueError):
        make_suggester("trigram")


def test_importing_language_needs_no_nlp_packages():
    """The without-NLP condition (and CI) must run without wordfreq/transformers/torch:
    importing the module may not import them."""
    code = (
        "import sys, ssvep.language; "
        "print(sorted(m for m in ('wordfreq', 'transformers', 'torch') "
        "if m in sys.modules))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"


# --- load_vocabulary (needs wordfreq) -------------------------------------------


@pytest.fixture(scope="module")
def vocabulary():
    return load_vocabulary()


@needs_wordfreq
def test_vocabulary_words_are_typeable(vocabulary):
    assert all(re.fullmatch("[A-Z]+", word) for word in vocabulary)


@needs_wordfreq
def test_vocabulary_most_frequent_first(vocabulary):
    freqs = list(vocabulary.values())
    assert all(f > 0 for f in freqs)
    assert all(a >= b for a, b in zip(freqs, freqs[1:]))
    assert next(iter(vocabulary)) == "THE"


@needs_wordfreq
def test_vocabulary_size_and_common_words(vocabulary):
    assert 40_000 <= len(vocabulary) <= 50_000
    for word in ["HELLO", "WORLD", "THE", "QUICK", "BROWN", "FOX"]:
        assert word in vocabulary
    small = load_vocabulary(1000)
    assert 800 <= len(small) <= 1000
    assert list(small) == list(vocabulary)[: len(small)]


@needs_wordfreq
def test_make_suggester_unigram():
    suggest = make_suggester("unigram")
    words = suggest("HELLO WOR")
    assert len(words) == 3
    assert all(word.startswith("WOR") and word != "WOR" for word in words)
    assert "WORLD" in words


# --- CausalLMScorer (needs transformers + torch; downloads distilgpt2 once) ------


@pytest.fixture(scope="module")
def scorer():
    return CausalLMScorer()


@needs_transformers
def test_scorer_one_log_probability_per_word(scorer):
    scores = scorer("HELLO ", ["WORLD", "THERE", "XYLOPHONE"])
    assert len(scores) == 3
    assert all(isinstance(s, float) and math.isfinite(s) and s <= 0 for s in scores)
    assert scorer("HELLO ", []) == []


@needs_transformers
@pytest.mark.parametrize(
    "context, likely, unlikely",
    [
        ("I LIVE IN NEW ", "YORK", "BANANA"),
        ("THANK ", "YOU", "TABLE"),
        ("THE UNITED ", "STATES", "PURPLE"),
        ("", "THE", "XYLOPHONE"),
    ],
)
def test_scorer_prefers_the_likely_word(scorer, context, likely, unlikely):
    s_likely, s_unlikely = scorer(context, [likely, unlikely])
    assert s_likely > s_unlikely


@needs_transformers
def test_scorer_uses_the_context(scorer):
    """The same word scores differently after different contexts."""
    (after_new,) = scorer("I LIVE IN NEW ", ["YORK"])
    (after_eat,) = scorer("I LIKE TO EAT ", ["YORK"])
    assert after_new > after_eat + 1.0


@needs_transformers
def test_scorer_batch_equals_one_by_one(scorer):
    """Words of different token lengths scored together give the same scores as alone
    (padding must not leak into the scores)."""
    words = ["YORK", "HAMPSHIRE", "BANANA", "A", "ANTIDISESTABLISHMENTARIANISM"]
    together = scorer("I LIVE IN NEW ", words)
    alone = [scorer("I LIVE IN NEW ", [word])[0] for word in words]
    assert together == pytest.approx(alone, abs=1e-3)


@needs_transformers
def test_scorer_multi_token_word_sums_its_tokens(scorer):
    """A long rare word costs more than a short common one in the same context."""
    common, rare = scorer("THIS IS A ", ["TEST", "ANTIDISESTABLISHMENTARIANISM"])
    assert rare < common - 5.0


@needs_transformers
def test_scorer_ignores_case(scorer):
    assert scorer("I LIVE IN NEW ", ["YORK"]) == pytest.approx(
        scorer("i live in new ", ["york"]), abs=1e-4
    )


@needs_transformers
def test_scorer_keeps_only_the_last_context_words():
    short = CausalLMScorer(max_context_words=2)
    a = short("BLAH BLAH CITY OF NEW ", ["YORK", "BANANA"])
    b = short("SOMETHING ELSE ENTIRELY OF NEW ", ["YORK", "BANANA"])
    assert a == pytest.approx(b, abs=1e-4)


@needs_transformers
@needs_wordfreq
def test_gpt2_reranker_uses_context(scorer):
    """Same prefix, different context: the language model changes the order."""
    reranker = TransformerReranker(UnigramModel(load_vocabulary()), scorer)
    assert reranker("I LIVE IN NEW Y")[0] == "YORK"
    assert reranker("THANK Y")[0] == "YOU"


@needs_transformers
@needs_wordfreq
def test_make_suggester_gpt2():
    suggest = make_suggester("gpt2")
    words = suggest("THANK Y")
    assert len(words) == 3
    assert words[0] == "YOU"
