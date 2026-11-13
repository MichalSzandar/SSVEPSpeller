"""Offline check of the fusion (ssvep/fusion.py) before it goes online.

Copies phrases of the phrase set in simulation, like scripts/keystroke_savings.py, but
every decision comes from a real classifier score vector: the cross-validated scores
(online.Decoder.cross_val_scores) of a calibration trial whose target is the cued one,
drawn at random. Without fusion the decision is the highest score; with fusion it is
fusion.Fusion.decide. So the errors have the subject's real accuracy and confusions
(e.g. 15 -> 8 Hz), and the fusion sees the speller states a copy-spelling session
goes through, corrections included.

Every condition (--nlp kind) is copied without fusion and with fusion at each
--floors value. With the same --seed, every setting draws from the same random
streams (one per repetition x phrase), so the settings differ only in their decisions.

Usage:
    .venv/bin/python scripts/evaluate_fusion.py --method FBCCA-norm --latency 0.7 \\
        --calibration data/recordings/pilot_20261110_172435.npz
    .venv/bin/python scripts/evaluate_fusion.py ... --conditions none gpt2 --floors 0.2 0.3

Output: results/<name>.csv (one row per condition x setting x repetition x phrase) and
a summary table on stdout.
"""

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

import keystroke_savings as ks
from compare_methods import METHODS
from online import Decoder, load_calibration
from ssvep import config, language
from ssvep.fusion import Fusion, fit_temperature
from ssvep.speller import Speller

FLOORS = (0.1, 0.3, 0.5)
NO_FUSION = "eeg"  # the setting name of the decisions without fusion
LATENCY = 0.7  # s, the Nov 10 cap offset (online --latency 0.7)
PROGRESS_EVERY = 10  # phrases between two progress lines


def draw_scores(
    scores: np.ndarray, y: np.ndarray, target: int | None, rng: np.random.Generator
) -> np.ndarray:
    """The scores (n_targets,) of one calibration trial drawn at random from the trials
    whose label is ``target``; from all trials if ``target`` is None (no correct
    choice). ``scores``: (n_trials, n_targets), ``y``: (n_trials,)."""
    if target is None:
        rows = np.arange(len(y))
    else:
        rows = np.flatnonzero(y == target)
        if len(rows) == 0:
            raise ValueError(f"no calibration trial has target {target}")
    return scores[rng.choice(rows)]


def make_classify(
    scores: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    fusion: Fusion | None = None,
) -> Callable[[int | None, Speller], int]:
    """A ``classify(target, speller)`` for keystroke_savings.copy_phrase.

    Each call draws a score vector for the cued ``target`` (draw_scores, with ``rng``)
    and returns the decision as a plain int: the argmax of the scores without
    ``fusion``, else fusion.decide(scores, speller)'s selected target.
    """

    def classify(target: int | None, speller: Speller) -> int:
        trial = draw_scores(scores, y, target, rng)
        if fusion is None:
            return int(np.argmax(trial))
        selected, _, _ = fusion.decide(trial, speller)
        return int(selected)

    return classify


def run(
    phrases: list[str],
    suggesters: dict[str, Callable[[str], list[str]] | None],
    scores: np.ndarray,
    y: np.ndarray,
    fusions: dict[str, Fusion | None],
    reps: int = 1,
    max_selections: int = ks.MAX_SELECTIONS,
    seed: int = 0,
    progress: bool = False,
) -> pd.DataFrame:
    """Copy every phrase ``reps`` times in every condition (``suggesters``, as in
    keystroke_savings) with every setting (``fusions``: name -> Fusion, or None for
    the classifier alone) -> one row per condition x setting x rep x phrase:
    condition, setting, rep, phrase_id, n_chars, selections, done, words.

    The random stream of repetition r and phrase i is np.random.default_rng([seed, r,
    i]) in every condition and setting. ``progress`` prints a line every
    PROGRESS_EVERY phrases.
    """
    spellers = {name: Speller(suggest=s) for name, s in suggesters.items()}
    rows = []
    start = time.perf_counter()
    for rep in range(reps):
        for i, phrase in enumerate(phrases):
            for condition, speller in spellers.items():
                for setting, fusion in fusions.items():
                    rng = np.random.default_rng([seed, rep, i])
                    classify = make_classify(scores, y, rng, fusion)
                    result = ks.copy_phrase(
                        speller, phrase, max_selections, classify=classify
                    )
                    rows.append(
                        {
                            "condition": condition,
                            "setting": setting,
                            "rep": rep,
                            "phrase_id": i,
                            "n_chars": len(phrase),
                            **result,
                        }
                    )
            if progress and ((i + 1) % PROGRESS_EVERY == 0 or i + 1 == len(phrases)):
                print(
                    f"  rep {rep + 1}/{reps}: {i + 1}/{len(phrases)} phrases "
                    f"({time.perf_counter() - start:.0f} s)"
                )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """One row per condition x setting: runs, done (share of runs that finished),
    selections (total over all runs, an unfinished run counting its max_selections),
    selections_median (per run), and vs_eeg: selections / the same condition's
    selections without fusion (setting NO_FUSION), so < 1 means the fusion helps."""
    summary = (
        results.groupby(["condition", "setting"], sort=False)
        .agg(
            runs=("selections", "size"),
            done=("done", "mean"),
            selections=("selections", "sum"),
            selections_median=("selections", "median"),
        )
        .reset_index()
    )
    eeg = summary[summary["setting"] == NO_FUSION].set_index("condition")["selections"]
    summary["vs_eeg"] = summary["selections"] / summary["condition"].map(eeg)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    """--method (online.py's choices), --calibration (one or more paths), --window
    (default config.ONLINE_STIM_DURATION), --latency (default 0.7, the Nov 10 cap
    offset), --conditions (default none unigram gpt2), --floors (default FLOORS),
    --phrases (default ks.PHRASES_PATH), --limit (phrases, default 100), --reps
    (default 2), --max-selections (default ks.MAX_SELECTIONS), --seed (default 0),
    --name (default "fusion_sim")."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--method", required=True, choices=list(METHODS))
    parser.add_argument(
        "--calibration",
        type=Path,
        nargs="+",
        required=True,
        help="calibration recording(s): the classifier's scores come from these",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=config.ONLINE_STIM_DURATION,
        help="classifier window in s",
    )
    parser.add_argument(
        "--latency",
        type=float,
        default=LATENCY,
        help="visual latency in s: the window starts this long after the onset",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=language.KINDS,
        default=list(language.KINDS),
        help="suggestion conditions (online.py --nlp)",
    )
    parser.add_argument(
        "--floors",
        type=float,
        nargs="+",
        default=list(FLOORS),
        help="one fusion setting per floor (share of the prior spread evenly)",
    )
    parser.add_argument("--phrases", type=Path, default=ks.PHRASES_PATH)
    parser.add_argument(
        "--limit", type=int, default=100, help="only the first N phrases"
    )
    parser.add_argument("--reps", type=int, default=2, help="runs per phrase")
    parser.add_argument("--max-selections", type=int, default=ks.MAX_SELECTIONS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default="fusion_sim", help="output file stem")
    return parser.parse_args(argv)


def main(argv=None):
    """Get the cross_val_scores of a Decoder on the calibration and fit the
    temperature (print it, with the cross-validated accuracy); load the vocabulary,
    the suggesters (ks.load_suggesters) and the first --limit phrases; run with
    NO_FUSION and one Fusion per floor; save results/<name>.csv; print summarize."""
    args = parse_args(argv)
    pd.set_option("display.width", 200)

    print(f"Cross-validating {args.method} on {args.calibration} ...")
    X, y = load_calibration(args.calibration)
    decoder = Decoder(args.method, window=args.window, latency=args.latency)
    scores = decoder.cross_val_scores(X, y)
    accuracy = np.mean(scores.argmax(axis=1) == y)
    temperature = fit_temperature(scores, y)
    print(
        f"Cross-validated accuracy {accuracy:.1%} on {len(y)} trials; "
        f"temperature {temperature:.4g}"
    )

    vocabulary = language.load_vocabulary()
    suggesters = ks.load_suggesters(args.conditions)
    phrases = ks.load_phrases(args.phrases)[: args.limit]
    fusions = {NO_FUSION: None}
    for floor in args.floors:
        fusions[f"floor {floor:g}"] = Fusion(temperature, vocabulary, floor=floor)
    print(
        f"{len(phrases)} phrases x {args.reps} reps x {len(suggesters)} conditions "
        f"x {len(fusions)} settings"
    )

    start = time.perf_counter()
    results = run(
        phrases,
        suggesters,
        scores,
        y,
        fusions,
        args.reps,
        args.max_selections,
        args.seed,
        progress=True,
    )
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / f"{args.name}.csv"
    results.to_csv(path, index=False)

    print(
        f"\nAt most {args.max_selections} selections per phrase "
        f"({time.perf_counter() - start:.0f} s)"
    )
    print(summarize(results).round(3).to_string(index=False))
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
