"""Results of the online copy-spelling runs, and replays at other latencies.

Usage:
    .venv/bin/python scripts/analyse_online.py --runs data/online_runs.csv
    .venv/bin/python scripts/analyse_online.py --runs data/online_runs.csv \\
        --replay-latencies 0.14 0.5 0.6 0.7

The runs file lists every online run, one row each, because the logs alone don't say
how a run was configured. Columns:
    run          a short label
    online       the GUI's trial table (<stem>_online.csv)
    backend      the backend's decision log (<stem>_<condition>_backend.csv)
    recording    the EEG recording of the run (.npz), for --replay-latencies
    calibration  the calibration the backend was trained on (.npz)
    method, window, latency, nlp, fusion (yes/no)   the online.py options
    phrases      the copy phrases in order, separated by "|"
    include      yes: part of the results; no: kept for the replays only
    note         free text
online.py writes these options next to every new decision log (<log>.json).

Output:
  results/<name>_runs.csv     one row per run: cued selections and accuracy, median
      time per selection (onset to onset), ITR with that time, answer delay after the
      flicker, dropped frames, classification and suggestion times, words accepted,
      phrases done, correctly typed characters and characters per minute
  results/<name>_phrases.csv  one row per copy phrase: selections, accuracy, done,
      correct characters, time, characters per minute, selections per character, and
      the selections the phrase needs without errors and without suggestions
  results/<name>_replay.csv   (with --replay-latencies) one row per run and latency:
      the accuracy if the run's recording had been classified at that latency, and the
      share of decisions identical to the logged ones (1.0 at the run's own latency
      shows the replay reproduces the run)

Definitions: a phrase's time runs from its first flicker onset to its last one, plus
one median time per selection (the last selection's cycle). Correct characters are
the phrase's length if it was finished, else the length of the longest correct start
of the typed text. Free spelling after the last phrase has no ground truth and counts
for neither accuracy nor characters per minute.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import keystroke_savings
from online import BUFFER_SECONDS, Backend, Decoder, load_calibration, replay
from ssvep import acquisition, config
from ssvep.evaluation import itr
from ssvep.speller import Speller

YES = {"yes", "y", "true", "1"}


def load_runs(path: Path) -> list[dict]:
    """The runs file as a list of dicts: paths as Path (a relative path is relative to
    the repository root), window and latency as float, fusion and include as bool,
    phrases as a list of strings."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    runs = []
    for row in table.to_dict("records"):
        row["phrases"] = [p.strip() for p in row["phrases"].split("|") if p.strip()]
        for key in ["online", "backend", "recording", "calibration"]:
            row[key] = config.ROOT / row[key] if row.get(key) else None
        row["window"] = float(row["window"])
        row["latency"] = float(row["latency"])
        row["fusion"] = row.get("fusion", "").strip().lower() in YES
        row["include"] = row.get("include", "yes").strip().lower() in YES
        runs.append(row)
    return runs


def normalise(phrase: str) -> str:
    """A phrase as the speller stores it: uppercase, no spaces at the ends."""
    return phrase.strip().upper()


def common_prefix(a: str, b: str) -> int:
    """Length of the longest common start of ``a`` and ``b``."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def phrase_index(texts: list[str], phrases: list[str]) -> list[int]:
    """Which copy phrase each decision belongs to -> one index per decision, -1 for
    free spelling after the last phrase.

    ``texts`` are the typed texts after each decision, in order. A phrase ends with
    the decision after which the text equals it, with or without one trailing space
    (Speller.done); the next decision belongs to the next phrase.
    """
    phrases = [normalise(p) for p in phrases]
    index, current = [], 0
    for text in texts:
        index.append(current if current < len(phrases) else -1)
        if current < len(phrases) and text in (
            phrases[current],
            phrases[current] + " ",
        ):
            current += 1
    return index


def read_logs(run: dict) -> pd.DataFrame:
    """The run's decision log joined with the GUI's trial table on the trial number.

    One row per decision of the backend, with its target, selected, item, text,
    classify_ms (and select_ms, if logged), and the GUI's stim_on, stim_off, answer
    and n_dropped of that trial (NaN for a decision the GUI never received), and
    ``accepted``: whether the decision accepted a suggested word (a word cell that
    changed the text).
    """
    backend = pd.read_csv(run["backend"])
    gui = pd.read_csv(run["online"])
    gui = gui[["trial", "stim_on", "stim_off", "answer", "n_dropped", "selected"]]
    log = backend.merge(
        gui.rename(columns={"selected": "gui_selected"}), on="trial", how="left"
    )
    log["text"] = log["text"].fillna("").astype(str)
    log["item"] = log["item"].fillna("").astype(str)
    log = log.sort_values("trial", ignore_index=True)
    # A word cell only accepts a word if it held one: an empty cell changes nothing
    changed = log["text"] != log["text"].shift(fill_value="")
    log["accepted"] = log["item"].str.startswith("#") & changed
    return log


def summarize_run(run: dict) -> tuple[dict, list[dict]]:
    """One run -> (its row of <name>_runs.csv, its rows of <name>_phrases.csv)."""
    log = read_logs(run)
    gui = pd.read_csv(run["online"])
    n_targets = len(config.TARGET_FREQS)
    t_sel = float(np.median(np.diff(gui["stim_on"].to_numpy())))

    scored = log[log["target"].notna() & log["selected"].notna()]
    n_correct = int((scored["selected"] == scored["target"]).sum())
    accuracy = n_correct / len(scored) if len(scored) else np.nan
    delay = (gui["answer"] - gui["stim_off"]).dropna()
    both = log[log["gui_selected"].notna()]
    mismatches = int((both["gui_selected"] != both["selected"]).sum())

    log["phrase"] = phrase_index(log["text"].tolist(), run["phrases"])
    phrase_rows = []
    for p, phrase in enumerate(run["phrases"]):
        part = log[log["phrase"] == p]
        if part.empty:
            continue
        target_text = normalise(phrase)
        done = part["text"].iloc[-1] in (target_text, target_text + " ")
        chars = (
            len(target_text)
            if done
            else common_prefix(part["text"].iloc[-1], target_text)
        )
        onsets = part["stim_on"].dropna()
        seconds = float(onsets.iloc[-1] - onsets.iloc[0] + t_sel)
        cued = part[part["target"].notna() & part["selected"].notna()]
        phrase_rows.append(
            {
                "run": run["run"],
                "phrase": target_text,
                "done": done,
                "selections": len(part),
                "cued": len(cued),
                "correct": int((cued["selected"] == cued["target"]).sum()),
                "accuracy": (
                    (cued["selected"] == cued["target"]).mean() if len(cued) else np.nan
                ),
                "chars_correct": chars,
                "seconds": seconds,
                "chars_per_min": 60 * chars / seconds,
                "selections_per_char": len(part) / len(target_text) if done else np.nan,
                "words_accepted": int(part["accepted"].sum()),
                "min_selections_none": keystroke_savings.baselines([target_text])[0],
            }
        )

    chars = sum(r["chars_correct"] for r in phrase_rows)
    seconds = sum(r["seconds"] for r in phrase_rows)
    changed = (
        log.loc[log["select_ms"] > 0.1, "select_ms"]
        if "select_ms" in log
        else pd.Series(dtype=float)
    )
    row = {
        "run": run["run"],
        "include": run["include"],
        "condition": f"{run['nlp']}-fusion" if run["fusion"] else run["nlp"],
        "method": run["method"],
        "window": run["window"],
        "latency": run["latency"],
        "decisions": len(log),
        "cued": len(scored),
        "correct": n_correct,
        "accuracy": accuracy,
        "t_sel_s": t_sel,
        "itr": itr(n_targets, accuracy, t_sel) if len(scored) else np.nan,
        "answer_delay_s": float(delay.median()),
        "answer_delay_min_s": float(delay.min()),
        "dropped_frames": int(gui["n_dropped"].sum()),
        "classify_ms_median": float(log["classify_ms"].median()),
        "classify_ms_max": float(log["classify_ms"].max()),
        "select_ms_median": float(changed.median()) if len(changed) else np.nan,
        "select_ms_max": float(changed.max()) if len(changed) else np.nan,
        "words_accepted": int(log["accepted"].sum()),
        "phrases": len(run["phrases"]),
        "phrases_done": sum(r["done"] for r in phrase_rows),
        "chars_correct": chars,
        "minutes": seconds / 60,
        "chars_per_min": 60 * chars / seconds if seconds else np.nan,
        "gui_backend_mismatches": mismatches,
    }
    return row, phrase_rows


def replay_run(run: dict, latency: float) -> dict:
    """The run's recording replayed through the online backend at ``latency``,
    trained on the run's calibration -> one row of <name>_replay.csv.

    The classifier's decision does not depend on the speller's state, so the replay
    spells freely and is scored against the targets in the recorded markers. A fusion
    run is compared with the classifier's own decisions in its log (selected_eeg).
    """
    X, y = load_calibration([run["calibration"]])
    decoder = Decoder(run["method"], window=run["window"], latency=latency).fit(X, y)
    buffer = acquisition.RingBuffer(
        len(config.CAP_CHANNELS), int(BUFFER_SECONDS * config.EEG_FS)
    )
    backend = Backend(decoder, Speller(), buffer)
    replay(run["recording"], backend)

    rows = pd.DataFrame(backend.rows)
    scored = rows[rows["target"].notna() & rows["selected"].notna()]
    logged = pd.read_csv(run["backend"])
    logged_column = "selected_eeg" if "selected_eeg" in logged else "selected"
    logged = logged[["trial", logged_column]].rename(columns={logged_column: "logged"})
    both = rows.merge(logged, on="trial")
    both = both[both["selected"].notna() & both["logged"].notna()]
    return {
        "run": run["run"],
        "include": run["include"],
        "method": run["method"],
        "window": run["window"],
        "run_latency": run["latency"],
        "latency": latency,
        "cued": len(scored),
        "accuracy": float((scored["selected"] == scored["target"]).mean()),
        "compared": len(both),
        "same_as_log": float((both["selected"] == both["logged"]).mean()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=Path, required=True, help="the runs file (CSV)")
    parser.add_argument(
        "--replay-latencies",
        type=float,
        nargs="*",
        default=[],
        help="also replay every run's recording at these latencies (s)",
    )
    parser.add_argument("--name", default="online", help="output file stem")
    parser.add_argument("--out-dir", type=Path, default=config.RESULTS_DIR)
    args = parser.parse_args(argv)

    runs = load_runs(args.runs)
    run_rows, phrase_rows = [], []
    for run in runs:
        row, phrases = summarize_run(run)
        run_rows.append(row)
        phrase_rows += phrases
    run_table = pd.DataFrame(run_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_table.to_csv(args.out_dir / f"{args.name}_runs.csv", index=False)
    pd.DataFrame(phrase_rows).to_csv(
        args.out_dir / f"{args.name}_phrases.csv", index=False
    )
    for r in run_table.itertuples():
        print(
            f"{r.run:<14} {r.condition:<12} latency {r.latency:g}  {r.correct}/"
            f"{r.cued} = {r.accuracy:.1%}  {r.t_sel_s:.2f} s/selection  ITR "
            f"{r.itr:5.1f}  {r.chars_per_min:4.2f} chars/min  {r.phrases_done}/"
            f"{r.phrases} phrases  answer +{r.answer_delay_s:.2f} s  dropped "
            f"{r.dropped_frames}"
        )
        if r.gui_backend_mismatches:
            print(f"  WARNING: {r.gui_backend_mismatches} GUI/backend mismatches")

    if args.replay_latencies:
        replay_rows = []
        for run in runs:
            if run["recording"] is None or run["calibration"] is None:
                continue
            for latency in args.replay_latencies:
                row = replay_run(run, latency)
                replay_rows.append(row)
                print(
                    f"replay {row['run']:<14} latency {latency:g}: accuracy "
                    f"{row['accuracy']:.1%} of {row['cued']}, same as logged "
                    f"{row['same_as_log']:.1%}"
                )
        pd.DataFrame(replay_rows).to_csv(
            args.out_dir / f"{args.name}_replay.csv", index=False
        )
    print(f"\nSaved {args.out_dir / args.name}_runs.csv, _phrases.csv", end="")
    print(", _replay.csv" if args.replay_latencies else "")


if __name__ == "__main__":
    main()
