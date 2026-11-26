# Design and Implementation of a Real-Time Brain–Computer Interface for Predictive EEG-Based Communication Using Machine Learning and Natural Language Processing

An SSVEP speller with word prediction. Six targets flicker at different frequencies
(8.0, 9.4, 10.8, 12.2, 13.6 and 15.0 Hz) on a ring around the typed text. When the user
looks at a target, the EEG over the visual cortex follows its frequency, and a
classifier decides which target was chosen. Each target opens a group of letters or
picks an item in it, so a character takes two selections. A language model suggests
whole words, which saves selections.

The repository has three parts, and all of them use the `ssvep` Python package for
filtering, classification and the speller logic:

- **Offline analysis**: compares the classifiers (CCA, FBCCA, TRCA, eTRCA, FB-eTRCA,
  PSD+LDA and the normalised CCA-norm and FBCCA-norm) on the Tsinghua SSVEP Benchmark
  dataset and on our own recordings, with leave-one-block-out cross-validation.
- **Calibration**: a flicker GUI that runs cued trials while the EEG and event markers
  are recorded over Lab Streaming Layer (LSL).
- **Online speller**: a GUI and a separate backend process that classifies each trial
  as it arrives, runs the speller and suggests words (unigram or distilgpt2).

Everything can run without the cap: a fake EEG stream responds to the GUI's markers
with a synthetic SSVEP.

## Requirements

- Python 3.11 or newer (developed on 3.12).
- Linux. The stimulus windows were tested on GNOME on X11.
- A monitor that runs at 120 Hz (`config.REFRESH_RATE`), for the flicker.
- For real EEG: a BrainAccess MIDI cap with dry electrodes at 250 Hz, streamed over LSL
  by BrainAccess Board. The system uses 8 channels, `config.CAP_CHANNELS`: P7, P3, Pz,
  P4, P8, O1, Oz, O2.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # the package, plus pytest, pylint, black
.venv/bin/pip install -e ".[nlp]"        # word prediction: transformers, wordfreq
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e ".[analysis]"   # only for the notebooks: jupyterlab, seaborn, mne
```

`torch` is not in the extras. Install the CPU wheel as shown, otherwise pip downloads
several GB of CUDA libraries. `torch` and `transformers` are only needed for
`--nlp gpt2`. The first time it runs, it downloads distilgpt2 from the Hugging Face
Hub.

All data goes into `data/` and all results into `results/`. Both are gitignored.

### Benchmark dataset

Tsinghua SSVEP Benchmark (Wang et al. 2017): 35 subjects, 40 targets, 64 channels,
250 Hz. The scripts expect the files in `data/benchmark/`. You need `7z` (package
`p7zip-full`).

```bash
mkdir -p data/benchmark && cd data/benchmark
base=https://bci.med.tsinghua.edu.cn/upload/yijun
wget -c $base/Freq_Phase.mat $base/64-channels.loc $base/Readme.txt $base/Sub_info.txt
for s in $(seq 1 35); do wget -c $base/S$s.mat.7z && 7z x S$s.mat.7z; done   # ~105 MB each
```

A few subjects are enough to try the scripts. `scripts/run_analysis.sh` needs all 35.

### Phrase set

The keystroke-savings and fusion simulations copy the phrases of MacKenzie & Soukoreff
(2003):

```bash
mkdir -p data/phrases && cd data/phrases
wget https://www.yorku.ca/mack/PhraseSets.zip
unzip -p PhraseSets.zip phrases2.txt > mackenzie_soukoreff_2003.txt
```

## Quick start without the cap

```bash
# Offline comparison on synthetic data (no download needed)
.venv/bin/python scripts/compare_methods.py --data synthetic

# Speller GUI with a stand-in backend that selects a random target in every trial
.venv/bin/python scripts/speller_gui.py --demo --windowed
```

The whole online loop can also run on the fake EEG stream. Run each command in its own
terminal:

```bash
.venv/bin/python scripts/fake_eeg_stream.py
.venv/bin/python scripts/record.py --subject test --eeg-name FakeEEG
.venv/bin/python scripts/calibration.py --subject test --blocks 2 --windowed
# when the calibration is done, record.py saves data/recordings/test_<stamp>.npz and exits
.venv/bin/python scripts/online.py --subject test --method FBCCA-norm \
    --calibration data/recordings/test_<stamp>.npz --phrases "hello world" --cue --nlp unigram
.venv/bin/python scripts/speller_gui.py --subject test --windowed
.venv/bin/python scripts/record.py --subject test --eeg-name FakeEEG   # then SPACE in the GUI
```

`--windowed` opens an 800×600 window instead of full screen. The flicker timing is not
valid in a window, so use it only to try things out.

Every script lists its options with `--help`, and its docstring describes its inputs
and outputs.

## Running a session with the cap

Before a session, set the monitor to 120 Hz in GNOME Settings → Displays, then log out
and back in. Changing the rate while logged in (or with `xrandr`) has crashed GNOME
Shell. Start BrainAccess Board and its LSL stream.

### 1. Check the EEG stream

```bash
.venv/bin/python scripts/check_lsl.py                          # list the streams on the network
.venv/bin/python scripts/check_lsl.py --type EEG --seconds 10  # sampling rate, per-channel SD
.venv/bin/python scripts/check_lsl.py --type EEG --alpha       # eyes open/closed: alpha peak at O1/O2
```

A channel with a very large or very small SD usually has poor contact.

### 2. Test the flicker timing

```bash
.venv/bin/python scripts/calibration.py --test --seconds 30
```

This flickers every target and saves the flip log to `results/timing/`. Record only
if it reports that the timing is OK (no dropped frames).

### 3. Record a calibration

Run each command in its own terminal:

```bash
.venv/bin/python scripts/record.py --subject S01
.venv/bin/python scripts/calibration.py --subject S01
```

A session has 10 blocks (`--blocks`). Each block shows every target once, in random
order: a 2 s cue (red frame), then 5 s of flicker. The subject starts each block with
SPACE and can rest in between; ESC quits. The whole session takes about 7 minutes
without the rests. `record.py` stops by itself after the session ends.

Files in `data/recordings/`:

| File | Written by | Contents |
|---|---|---|
| `S01_<stamp>.npz` | `record.py` | EEG, LSL timestamps, markers, channel names (format at the top of [ssvep/acquisition.py](ssvep/acquisition.py)) |
| `S01_<stamp>_trials.csv` | `calibration.py` | one row per trial |
| `S01_<stamp>_flips.npz` | `calibration.py` | the time of every screen flip |

`<stamp>` is `YYYYmmdd_HHMMSS`. The two scripts start at slightly different times, so
their stamps differ.

### 4. Choose the classifier settings

```bash
.venv/bin/python scripts/compare_methods.py --data recording \
    --recordings data/recordings/S01_<stamp>.npz --windows 1 1.5 2 2.5 3 --name s01
.venv/bin/python scripts/latency_sweep.py --recordings data/recordings/S01_<stamp>.npz \
    --latencies 0.14 0.3 0.5 0.7 0.9 --windows 1 2 --name s01_latency
```

`latency_sweep.py` checks where the analysis window should start after the stimulus
onset. Normally that is the visual latency, 0.14 s. But the cap's timestamps were
once about 0.5 s ahead of the markers. If accuracy keeps rising with the latency, pass
the best latency to `online.py --latency`. Check this in every session.

FBCCA-norm is the method used online. The TRCA methods score well within a
calibration but carried over poorly to online trials.

### 5. Run the online speller

Run each command in its own terminal, in this order:

```bash
.venv/bin/python scripts/online.py --subject S01 --method FBCCA-norm \
    --calibration data/recordings/S01_<stamp>.npz --phrases "hello world" --cue --nlp gpt2
.venv/bin/python scripts/speller_gui.py --subject S01
.venv/bin/python scripts/record.py --subject S01        # then press SPACE in the GUI
```

The backend loads the calibration, trains the classifier and warms up the language
model, then waits for the GUI. In the GUI, SPACE starts and pauses the session and ESC
quits. Stop the backend with Ctrl+C, which saves its decision log.

Backend options:

| Option | Meaning |
|---|---|
| `--method` | classifier, as in `compare_methods.py` |
| `--window` | classifier window in s, 2 by default. It must not be longer than the GUI's `--stim`, so a 3 s window needs `speller_gui.py --stim 3` too. |
| `--latency` | start of the window after the onset, 0.14 s by default (see step 4) |
| `--phrases` | phrases to copy, in order. Without it the speller runs in free spelling. |
| `--cue` | frame the next correct target in red before each trial |
| `--nlp` | word suggestions: `none`, `unigram` (most frequent words with the typed prefix) or `gpt2` (the unigram candidates re-ranked by distilgpt2) |
| `--fusion` | pick the target by combining the classifier scores with the language model's probabilities, instead of by the classifier alone (`--floor` sets how much the EEG can overrule the language model) |
| `--eeg-name` | pick the EEG stream by name, e.g. `FakeEEG` when the cap is streaming too |

One trial: 0.5 s with the labels shown (a red frame with `--cue`), 2 s of flicker,
a wait for the backend's answer, then 0.5 s with the selected target framed in green.
With the default 2 s window, the answer comes at least latency + 0.5 s after the
flicker ends. The backend waits for 0.5 s of EEG after the window, so that filter edge
effects stay out of it.

Files in `data/recordings/`:

| File | Written by | Contents |
|---|---|---|
| `S01_<stamp>_online.csv` | `speller_gui.py` | one row per trial: cued target, flicker times, dropped frames, selection, text |
| `S01_<stamp>_online_flips.npz` | `speller_gui.py` | the time of every screen flip |
| `S01_<stamp>_<condition>_backend.csv` | `online.py` | one row per answered trial: selection, classifier scores, text, compute times |
| `S01_<stamp>_<condition>_backend.json` | `online.py` | the run's options |
| `S01_<stamp>.npz` | `record.py` | the EEG, which can be replayed (below) |

`<condition>` is the `--nlp` kind, with `-fusion` added when `--fusion` is used.

### 6. Log the run

`scripts/analyse_online.py` reads the list of online runs from `data/online_runs.csv`,
one row per run:

```
run,online,backend,recording,calibration,method,window,latency,nlp,fusion,phrases,include,note
```

The first five columns are the files above. The settings come from the backend's
`.json`. `phrases` are separated by `|`. `include` is `yes` for runs that go into the
results and `no` for runs that are kept only for replays.

## Offline analysis

All scripts write to `results/` (`--name` sets the file stem).

| Script | What it does |
|---|---|
| `compare_methods.py` | accuracy and ITR of every method against window length, with leave-one-block-out CV. `--data synthetic\|benchmark\|recording`. `--targets all` uses all 40 Benchmark targets. `--trial-ratio 0 --channel-ratio 0` switches artifact rejection off. |
| `method_stats.py` | Friedman and Holm-corrected Wilcoxon tests between the methods, from a `compare_methods.py` CSV |
| `latency_sweep.py` | the same comparison at several window start times |
| `cross_session.py` | trains on one recording and tests on another, for every pair |
| `artifact_ratios.py` | the trial and channel ratios that artifact rejection uses, per subject |
| `keystroke_savings.py` | copies the phrase set with the real speller, with and without suggestions, then again with simulated classifier errors |
| `evaluate_fusion.py` | the same simulation, with decisions from a calibration's cross-validated classifier scores, with and without fusion |
| `analyse_online.py` | accuracy, ITR, characters per minute and timing of every run in `data/online_runs.csv`. `--replay-latencies` re-classifies each run at other latencies. |
| `analyse_timing.py` | dropped frames and flip-interval spread of flip logs |

Examples:

```bash
.venv/bin/python scripts/compare_methods.py --data benchmark --subjects 1 2 3
.venv/bin/python scripts/method_stats.py results/compare_benchmark.csv
.venv/bin/python scripts/compare_methods.py --data benchmark --methods CCA FBCCA --windows 0.5 1 2 --name cca_fbcca
.venv/bin/python scripts/cross_session.py --recordings A.npz B.npz --windows 1 2 3
.venv/bin/python scripts/keystroke_savings.py --conditions none unigram gpt2
.venv/bin/python scripts/evaluate_fusion.py --method FBCCA-norm --calibration A.npz --conditions none gpt2
.venv/bin/python scripts/analyse_online.py --runs data/online_runs.csv --replay-latencies 0.14 0.5
```

### Replay a recording

```bash
.venv/bin/python scripts/online.py --method FBCCA-norm --calibration A.npz --replay B.npz
```

This feeds recording `B` through the online backend as if it arrived live, without
LSL or the GUI. It prints the accuracy against the recorded targets and saves the
decision log. Both calibrations and online sessions can be replayed. Train and replay
on different recordings, because a trained method scores its own training trials too
well.

### All results at once

```bash
scripts/run_analysis.sh                   # every step
scripts/run_analysis.sh own online        # only these steps
FORCE=1 scripts/run_analysis.sh own       # redo a step whose results exist
```

The steps are `benchmark` (35 subjects, the 6 speller frequencies, ~25 min),
`benchmark40` (all 40 targets, ~40 min), `own` (our own calibrations), `online` (the
runs in `data/online_runs.csv`), `timing` (flip logs) and `figures`
(`thesis/figures/make_results.py`). The recordings and latency used for the own
sessions are set at the top of the script.

### Notebooks

`notebooks/` has three notebooks that explore the Benchmark data and try CCA and
FBCCA on it. They need the `analysis` extra: `.venv/bin/jupyter lab`.

## How the parts fit together

```
┌──────────────────────────┐  LSL "SSVEP-Markers" (JSON)  ┌───────────────────────────────┐
│ GUI process (pyglet)     │ ───────────────────────────▶ │ Backend process (online.py)   │
│ • flicker, frame clock   │                              │ • ring buffer of the EEG      │
│ • draws labels and text  │ ◀─────────────────────────── │ • filtering + classifier      │
└──────────────────────────┘  LSL "SSVEP-State" (JSON)    │ • speller state + suggestions │
                                                          └───────────────▲───────────────┘
                     BrainAccess cap or fake_eeg_stream.py ── LSL "EEG" ──┘
```

- The GUI only draws and timestamps events, so the classifier and the language model
  never delay a frame. The backend owns the speller state and tells the GUI what to
  show. The messages are described in [ssvep/protocol.py](ssvep/protocol.py).
- Calibration uses the same marker stream and the same target positions, so
  `record.py` and `fake_eeg_stream.py` work the same in both.
- The backend cuts each trial's epoch from the ring buffer and filters it with the same
  code as the offline analysis ([ssvep/preprocessing.py](ssvep/preprocessing.py)), so
  online trials match what the classifier was trained and evaluated on.

### Speller menu

The menu has 6 groups of 6 cells, drawn on the 6 targets. The first selection opens a
group and the second picks a cell in it:

| Group | Cells |
|---|---|
| 1–4 | A–E, F–J, K–O, P–T, each with ↶ (back) |
| 5 | U–Z |
| 6 | 3 word suggestions, ␣ (space), ⌫ (backspace), ↶ (back) |

Back closes a wrongly opened group. Accepting a suggestion replaces the word being
typed and adds a space. Right after accepting a word, ⌫ removes the whole word. In
copy-spelling, the speller knows the shortest correct next selection, including
corrections after a mistake. The GUI can cue it, and the logs keep it as ground truth.
The classifier never sees it.

## Project layout

```
ssvep/
  config.py          constants: sampling rates, channels, target frequencies, timing
  datasets.py        Benchmark, own recordings, synthetic data -> (X, y, blocks)
  preprocessing.py   notch, band-pass, filter bank, crop, artifact rejection
  cca.py             CCA, FBCCA, normalised CCA
  trca.py            TRCA, eTRCA, FB-eTRCA
  baseline.py        PSD features + LDA
  evaluation.py      ITR, leave-one-block-out CV, window sweep
  stimulus.py        flicker luminance, target layout, trial order, frame timing
  acquisition.py     LSL inlets, ring buffer, recorder
  protocol.py        JSON messages between the speller GUI and the backend
  speller.py         speller state machine: menu, typed text, copy-spelling target
  language.py        word suggestions: unigram, distilgpt2 re-ranking
  fusion.py          classifier scores x language-model prior
scripts/
  check_lsl.py           inspect a live LSL stream
  fake_eeg_stream.py     synthetic EEG stream that reacts to the markers
  record.py              record EEG + markers to .npz
  calibration.py         calibration GUI and flicker timing test
  online.py              online backend, and replay of a recording
  speller_gui.py         online speller GUI (--demo runs without a backend)
  compare_methods.py     offline comparison of the classifiers
  method_stats.py        statistical tests between methods
  latency_sweep.py       comparison at several window start times
  cross_session.py       train on one session, test on another
  artifact_ratios.py     what artifact rejection sees
  keystroke_savings.py   simulated copy-spelling with and without suggestions
  evaluate_fusion.py     simulated copy-spelling with and without fusion
  analyse_online.py      results of the online runs
  analyse_timing.py      stimulus timing from the flip logs
  run_analysis.sh        every analysis step, then the figures
notebooks/           exploring Benchmark, CCA and FBCCA
tests/               pytest, on synthetic data. Tests that need the Benchmark files are skipped without them.
thesis/              LaTeX source of the thesis
```

## Development

```bash
.venv/bin/pytest -q
.venv/bin/pylint ssvep scripts --disable=all --enable=E,F --disable=E1121 --score=no
.venv/bin/black --check ssvep scripts tests
```

GitHub Actions runs these three checks on every push to `main` and on pull requests.
