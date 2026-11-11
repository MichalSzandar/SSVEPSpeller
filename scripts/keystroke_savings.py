"""Offline keystroke savings of the word suggestions on a standard phrase set.

Copies every phrase of the MacKenzie & Soukoreff (2003) phrase set with the real
Speller, following next_target() as the cue does online, once per suggestion condition
(the --nlp kinds of online.py: none, unigram, gpt2), and counts the selections.
Keystroke savings (KS) is the share of selections the suggestions save against typing
letter by letter (the "none" condition) on the same phrase:

    KS = 1 - selections with suggestions / selections without

Part 2 adds classifier errors: every decision is the cued target with probability
``accuracy`` and one of the other targets (uniformly) otherwise. Each phrase gets a
selection limit, and the time per selection (onset to onset, online) turns selections
into minutes and correct characters per minute.

Usage:
    .venv/bin/python scripts/keystroke_savings.py
    .venv/bin/python scripts/keystroke_savings.py --conditions none unigram --limit 50
    .venv/bin/python scripts/keystroke_savings.py --sim-phrases 0   # error-free only

The phrase set is phrases2.txt from https://www.yorku.ca/mack/PhraseSets.zip, saved
as data/phrases/mackenzie_soukoreff_2003.txt (500 phrases, letters and spaces).

Output (results/): <name>.csv (error-free, one row per phrase x condition),
<name>_latency.csv (one row per suggester call in the error-free part) and
<name>_sim.csv (one row per accuracy x repetition x phrase x condition), plus summary
tables on stdout.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from ssvep import config, language
from ssvep.speller import WORDS, Speller

PHRASES_PATH = config.DATA_DIR / "phrases" / "mackenzie_soukoreff_2003.txt"
ANSWER_DELAY = 0.7  # s from stim_off to the answer on the cap (Nov 10, latency 0.7)
# s from one flicker onset to the next: gaze shift, flicker, answer, feedback
SELECTION_S = (
    config.GAZE_SHIFT
    + config.ONLINE_STIM_DURATION
    + ANSWER_DELAY
    + config.FEEDBACK_DURATION
)
ACCURACIES = (0.75, 0.88)  # Nov 10 online: 72 % (replay, latency 0.7) to 88 %
MAX_SELECTIONS = 400  # per phrase, ~25 min at SELECTION_S
WARM_UP_TEXT = "HELLO WOR"


def load_phrases(path: Path) -> list[str]:
    """The phrases in ``path``, one per line: uppercase, with every run of whitespace
    collapsed to one space. Empty lines are skipped."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    phrases = [" ".join(line.split()).upper() for line in lines]
    return [phrase for phrase in phrases if phrase]


def decide(
    target: int | None, accuracy: float, n_targets: int, rng: np.random.Generator
) -> int:
    """A simulated classifier decision: ``target`` with probability ``accuracy``, else
    one of the other targets, uniformly. With ``target`` None (no correct choice) any
    target, uniformly."""
    if target is None:
        return int(rng.integers(n_targets))
    if rng.random() < accuracy:
        return target
    other = int(rng.integers(n_targets - 1))
    return other if other < target else other + 1  # skip the target itself


def copy_phrase(
    speller: Speller,
    phrase: str,
    max_selections: int = MAX_SELECTIONS,
    accuracy: float = 1.0,
    rng: np.random.Generator | None = None,
) -> dict:
    """Copy ``phrase`` with ``speller`` -> {"selections", "done", "words"}.

    Every selection is the cued target (next_target()) without ``rng``, else
    decide(target, accuracy, ...). Stops once the phrase is done or after
    ``max_selections`` selections. "words" counts the accepted suggestions.
    """
    speller.set_phrase(phrase)
    n_targets = len(speller.slots)
    words = 0
    for n in range(max_selections):
        if speller.done:
            return {"selections": n, "done": True, "words": words}
        target = speller.next_target()
        if rng is not None:
            target = decide(target, accuracy, n_targets, rng)
        old_text = speller.text
        item = speller.select(target)
        if item in WORDS and speller.text != old_text:
            words += 1
    return {"selections": max_selections, "done": speller.done, "words": words}


class TimedSuggest:
    """A suggester that records how long each call of the wrapped one takes.

    ``calls`` holds (text, ms, memo_hit) per call; memo_hit is True when the text was
    already in the wrapped suggester's ``seen`` memo (TransformerReranker).
    """

    def __init__(self, suggest: Callable[[str], list[str]]):
        self.suggest = suggest
        self.calls = []

    def __call__(self, text: str) -> list[str]:
        memo_hit = text in getattr(self.suggest, "seen", {})
        start = time.perf_counter()
        words = self.suggest(text)
        self.calls.append((text, 1000 * (time.perf_counter() - start), memo_hit))
        return words


def baselines(phrases: list[str], max_selections: int = MAX_SELECTIONS) -> list[int]:
    """Selections to copy each phrase letter by letter, without errors."""
    speller = Speller()
    return [copy_phrase(speller, p, max_selections)["selections"] for p in phrases]


def evaluate(
    phrases: list[str],
    suggesters: dict[str, Callable[[str], list[str]] | None],
    max_selections: int = MAX_SELECTIONS,
) -> pd.DataFrame:
    """Copy every phrase without errors in every condition -> one row per phrase x
    condition: phrase_id, phrase, n_chars, condition, selections, done, words,
    baseline (selections without suggestions) and ks (keystroke savings)."""
    base = baselines(phrases, max_selections)
    rows = []
    for condition, suggest in suggesters.items():
        speller = Speller(suggest=suggest)
        for i, phrase in enumerate(phrases):
            result = copy_phrase(speller, phrase, max_selections)
            rows.append(
                {
                    "phrase_id": i,
                    "phrase": phrase,
                    "n_chars": len(phrase),
                    "condition": condition,
                    **result,
                    "baseline": base[i],
                    "ks": 1 - result["selections"] / base[i],
                }
            )
    return pd.DataFrame(rows)


def simulate(
    phrases: list[str],
    suggesters: dict[str, Callable[[str], list[str]] | None],
    accuracies: tuple[float, ...] = ACCURACIES,
    reps: int = 1,
    max_selections: int = MAX_SELECTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    """Copy every phrase with simulated classifier errors -> one row per accuracy x
    repetition x phrase x condition: accuracy, rep, phrase_id, n_chars, condition,
    selections, done, words.

    All conditions of one (accuracy, rep, phrase) start from the same random seed, so
    they differ only where the suggestions change the path."""
    spellers = {name: Speller(suggest=s) for name, s in suggesters.items()}
    rows = []
    for a, accuracy in enumerate(accuracies):
        for rep in range(reps):
            for i, phrase in enumerate(phrases):
                for condition, speller in spellers.items():
                    rng = np.random.default_rng([seed, a, rep, i])
                    result = copy_phrase(speller, phrase, max_selections, accuracy, rng)
                    rows.append(
                        {
                            "accuracy": accuracy,
                            "rep": rep,
                            "phrase_id": i,
                            "n_chars": len(phrase),
                            "condition": condition,
                            **result,
                        }
                    )
    return pd.DataFrame(rows)


def latency_table(timed: dict[str, TimedSuggest]) -> pd.DataFrame:
    """One row per recorded suggester call: condition, text, ms, memo_hit."""
    rows = [
        {"condition": name, "text": text, "ms": ms, "memo_hit": hit}
        for name, suggest in timed.items()
        for text, ms, hit in suggest.calls
    ]
    return pd.DataFrame(rows, columns=["condition", "text", "ms", "memo_hit"])


def summarize(results: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    """One row per condition of evaluate()'s table: selections, selections per
    character, KS over all phrases (ks_total) and per phrase (median, quartiles),
    accepted words, and the suggester's latency on calls that ran the model (memo
    hits left out)."""
    rows = []
    for condition, group in results.groupby("condition", sort=False):
        calls = latency[latency["condition"] == condition]
        model = calls.loc[~calls["memo_hit"].astype(bool), "ms"]
        rows.append(
            {
                "condition": condition,
                "phrases": len(group),
                "done": group["done"].mean(),
                "selections": group["selections"].sum(),
                "per_char": group["selections"].sum() / group["n_chars"].sum(),
                "ks_total": 1 - group["selections"].sum() / group["baseline"].sum(),
                "ks_median": group["ks"].median(),
                "ks_q1": group["ks"].quantile(0.25),
                "ks_q3": group["ks"].quantile(0.75),
                "words": group["words"].sum(),
                "model_calls": len(model),
                "ms_median": model.median() if len(model) else np.nan,
                "ms_p95": model.quantile(0.95) if len(model) else np.nan,
                "ms_max": model.max() if len(model) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compare_pair(results: pd.DataFrame, a: str, b: str) -> dict | None:
    """Per-phrase selections of condition ``b`` against ``a`` -> phrases where b needs
    fewer / as many / more selections, the mean difference (b - a) and the Wilcoxon
    signed-rank p-value. None if either condition is missing."""
    table = results.pivot(index="phrase_id", columns="condition", values="selections")
    if a not in table or b not in table:
        return None
    diff = table[b] - table[a]
    p_value = wilcoxon(table[b], table[a]).pvalue if (diff != 0).any() else 1.0
    return {
        "fewer": int((diff < 0).sum()),
        "equal": int((diff == 0).sum()),
        "more": int((diff > 0).sum()),
        "mean_diff": float(diff.mean()),
        "p_value": float(p_value),
    }


def summarize_sim(sim: pd.DataFrame, selection_s: float = SELECTION_S) -> pd.DataFrame:
    """One row per accuracy x condition of simulate()'s table: share of phrases done,
    median selections, median and 90th-percentile minutes per phrase, correct
    characters per minute (characters of the finished phrases over all the time), and
    the selections saved against "none" at the same accuracy (if it was run)."""
    minutes = sim["selections"] * selection_s / 60
    rows = []
    for (accuracy, condition), group in sim.groupby(
        ["accuracy", "condition"], sort=False
    ):
        group_minutes = minutes[group.index]
        rows.append(
            {
                "accuracy": accuracy,
                "condition": condition,
                "runs": len(group),
                "done": group["done"].mean(),
                "selections_median": group["selections"].median(),
                "minutes_median": group_minutes.median(),
                "minutes_p90": group_minutes.quantile(0.9),
                "chars_per_min": group.loc[group["done"], "n_chars"].sum()
                / group_minutes.sum(),
                "selections_total": group["selections"].sum(),
            }
        )
    table = pd.DataFrame(rows)
    none = table[table["condition"] == "none"].set_index("accuracy")
    table["saved_vs_none"] = [
        (
            1 - row.selections_total / none.loc[row.accuracy, "selections_total"]
            if row.accuracy in none.index
            else np.nan
        )
        for row in table.itertuples()
    ]
    return table.drop(columns="selections_total")


def load_suggesters(kinds: list[str]) -> dict[str, TimedSuggest | None]:
    """language.make_suggester(kind) for every kind, warmed up and wrapped in
    TimedSuggest ("none" stays None)."""
    suggesters = {}
    for kind in kinds:
        suggest = language.make_suggester(kind)
        if suggest is not None:
            start = time.perf_counter()
            suggest(WARM_UP_TEXT)  # the first model run is slow
            print(f"{kind}: warm-up {1000 * (time.perf_counter() - start):.0f} ms")
            suggest = TimedSuggest(suggest)
        suggesters[kind] = suggest
    return suggesters


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--phrases", type=Path, default=PHRASES_PATH)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=language.KINDS,
        default=list(language.KINDS),
        help="suggestion conditions to compare (online.py --nlp)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="only the first N phrases (error-free)"
    )
    parser.add_argument(
        "--sim-phrases",
        type=int,
        default=100,
        help="phrases (the first N) for the simulation with errors; 0 skips it",
    )
    parser.add_argument("--accuracies", type=float, nargs="+", default=ACCURACIES)
    parser.add_argument(
        "--reps", type=int, default=2, help="simulation runs per phrase and accuracy"
    )
    parser.add_argument("--max-selections", type=int, default=MAX_SELECTIONS)
    parser.add_argument(
        "--selection-s",
        type=float,
        default=SELECTION_S,
        help="s per selection, onset to onset",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="keystroke_savings", help="output file stem")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    phrases = load_phrases(args.phrases)[: args.limit]
    print(f"{len(phrases)} phrases from {args.phrases}")
    suggesters = load_suggesters(args.conditions)
    pd.set_option("display.width", 200)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    results = evaluate(phrases, suggesters, args.max_selections)
    timed = {k: s for k, s in suggesters.items() if isinstance(s, TimedSuggest)}
    latency = latency_table(timed)
    print(f"\nError-free copying ({time.perf_counter() - start:.0f} s)")
    print(summarize(results, latency).round(3).to_string(index=False))
    pair = compare_pair(results, "unigram", "gpt2")
    if pair is not None:
        print(f"gpt2 vs unigram, per phrase: {pair}")
    results.to_csv(config.RESULTS_DIR / f"{args.name}.csv", index=False)
    latency.to_csv(config.RESULTS_DIR / f"{args.name}_latency.csv", index=False)

    if args.sim_phrases > 0:
        start = time.perf_counter()
        sim = simulate(
            phrases[: args.sim_phrases],
            suggesters,
            tuple(args.accuracies),
            args.reps,
            args.max_selections,
            args.seed,
        )
        print(
            f"\nWith classifier errors, {args.selection_s:g} s per selection, "
            f"at most {args.max_selections} selections per phrase "
            f"({time.perf_counter() - start:.0f} s)"
        )
        print(summarize_sim(sim, args.selection_s).round(3).to_string(index=False))
        sim.to_csv(config.RESULTS_DIR / f"{args.name}_sim.csv", index=False)
    print(f"\nSaved results/{args.name}*.csv")


if __name__ == "__main__":
    main()
