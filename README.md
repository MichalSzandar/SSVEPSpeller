# Design and Implementation of a Real-Time Brain–Computer Interface for Predictive EEG-Based Communication Using Machine Learning and Natural Language Processing

Engineering thesis: an SSVEP speller. The user looks at one of six targets that
flicker at different frequencies, the EEG over the visual cortex follows that
frequency, and a classifier decides which target was chosen. A language model
suggests words, so fewer selections are needed per word.

The `ssvep` package holds the signal processing, the classifiers and the speller
logic. The offline analysis, the calibration app and the online speller all use the
same code.

## Status

As of 6 November 2026. The speller runs end to end, but not yet with the cap on.

| Part | State |
|---|---|
| Offline analysis on the Benchmark dataset: filtering, CCA, FBCCA, TRCA, eTRCA, FB-eTRCA, PSD+LDA baseline, normalised CCA and FBCCA, ITR, leave-one-block-out cross-validation | done for S1–S3; all 35 subjects still to run |
| Calibration app: sampled-sinusoid flicker at 120 Hz, cue → flicker → rest protocol, LSL markers timestamped at the frame flip, dropped-frame log | done |
| Acquisition over LSL: inlets, ring buffer, recorder, fake EEG stream for testing without the cap | done |
| Bad-trial and bad-channel rejection, on training trials | done |
| Pilot recordings with the cap: 3 calibration sessions, one subject | done |
| Speller GUI: synchronous trials on the calibration ring, labels and typed text from the backend, JSON protocol over LSL | done |
| Speller state machine: two-level menu, copy-spelling ground truth, cells for word suggestions | done |
| Online backend: same processing as offline, decision log, replay mode | done; tested in replay and over LSL with the fake EEG stream, not yet with the real GUI or the cap |
| Word prediction: unigram baseline, distilgpt2 re-ranking | planned |
| Online copy-spelling experiment, with and without word prediction | planned |

## Next steps

Code freeze is on 15 November 2026.

**First online session (by 8 November)**

1. Run the backend with the real GUI and the fake EEG stream, the one pair of parts not
   yet tested together.
2. Choose the online window. The default is 2 s, but the pilots point to 3 s (see
   [Pilot recordings](#pilot-recordings)). In a simulation with the real speller
   (uniform errors, no word suggestions), typing "HELLO" takes a median of about 6
   minutes at 2 s and 59 % accuracy, and about 2 minutes at 3 s and 75 %.
3. Check electrode contact, record a new 10-block calibration, then run a cued accuracy
   test and free spelling in the same session. The goal is to spell a five-letter word
   with the cap on.
4. Use that session to tune two timing settings: the visual latency
   (`compare_methods.py --latency`, 0 to 0.3 s), and `POST` in `scripts/online.py`, the
   0.5 s of EEG the backend waits for after the window, which delays every answer.

**Word prediction (9–11 November)**

5. Add `ssvep/language.py` with unigram prefix completion (`wordfreq`), and a backend
   option that passes it to `Speller(suggest=...)`, so that sessions can run with or
   without prediction.
6. Measure keystroke savings offline on a standard phrase set (MacKenzie & Soukoreff
   2003).
7. Add distilgpt2 re-ranking, compare it with the unigram baseline, and check that it
   runs fast enough between trials.
8. If time allows, combine the classifier scores with the language model's
   probabilities. This is the first item to drop if the schedule slips.

**Online experiment (12–14 November)**

9. Write the analysis of the online logs: selection accuracy, ITR, correct characters
   per minute and keystroke savings. Run it on the first online session, so that
   problems with the logs show up early.
10. Do a full dry run, then run the experiment: calibration, then copy-spelling with
    and without word prediction, in counterbalanced order. Set a time limit per phrase,
    because a low-accuracy session may not finish one. Recording other people depends
    on ethics approval and photosensitive epilepsy screening; without them, this
    becomes a single-subject case study.

**Code freeze (15 November)**

11. Tag `v1.0` and back up `data/`, which is not in git.

**Writing (16 November – 6 December)**

12. Download all 35 Benchmark subjects (S1–S3 so far), run the comparison and the
    statistical tests on them, analyse the online logs, make the figures, and send the
    draft to the supervisor by 6 December.

## Results so far

### Benchmark dataset

Subjects S1–S3, the speller's 6 targets (8.0, 9.4, 10.8, 12.2,
13.6, 15.0 Hz), 9 parieto-occipital channels, leave-one-block-out cross-validation
(6 folds). The window starts 0.14 s after stimulus onset (visual latency). ITR counts
the window plus 0.5 s for the gaze shift. Values are means over the 3 subjects, so
they are only a first check; chance level is 16.7 %.

| Method | Accuracy at 1 s | Best ITR (bits/min) | Window at best ITR |
|---|---|---|---|
| CCA | 85.2 % | 69.8 | 1.25 s |
| FBCCA | 98.1 % | 97.6 | 1.0 s |
| TRCA | 100 % | 121.8 | 0.5 s |
| eTRCA | 100 % | 150.1 | 0.5 s |
| FB-eTRCA | 100 % | 158.8 | 0.25 s |
| PSD+LDA | 45.4 % | 22.3 | 1.0 s |
| CCA-norm | 94.4 % | 87.2 | 1.0 s |
| FBCCA-norm | 98.1 % | 97.6 | 1.0 s |

`-norm` means each target's score is z-scored with statistics from the training
trials, so that low-frequency targets are not favoured on weak data.

To reproduce: `scripts/compare_methods.py --data benchmark --subjects 1 2 3` (see
[Usage](#usage)).

### Pilot recordings

Three calibration sessions of one subject with the BrainAccess cap, 6 blocks each, 8
parietal and occipital channels. The evaluation is the same as above, with artifact
rejection on the training folds. Values are means over the 3 sessions; chance level is
16.7 %.

| Method | Accuracy at 2 s | at 3 s | at 4 s | Best ITR (bits/min) | Window at best ITR |
|---|---|---|---|---|---|
| CCA | 51.9 % | 57.4 % | 63.9 % | 13.2 | 2.5 s |
| FBCCA | 50.9 % | 59.3 % | 71.3 % | 14.4 | 4.0 s |
| TRCA | 40.7 % | 47.2 % | 48.1 % | 7.7 | 1.0 s |
| eTRCA | 45.4 % | 40.7 % | 46.3 % | 12.8 | 0.5 s |
| FB-eTRCA | 28.7 % | 30.6 % | 31.5 % | 5.9 | 1.0 s |
| PSD+LDA | 28.7 % | 16.7 % | 25.0 % | 2.2 | 2.0 s |
| CCA-norm | 51.9 % | 69.4 % | 73.1 % | 17.9 | 3.0 s |
| FBCCA-norm | 59.3 % | 75.0 % | 78.7 % | 21.5 | 3.0 s |

- Accuracy is far below Benchmark. The cap has dry electrodes and no PO channels.
- FBCCA-norm is the best method here, and normalisation helps much more than on
  Benchmark. The TRCA methods, the best on Benchmark, fall behind with 6 training
  blocks. Calibration now defaults to 10 blocks, which may help them.
- Sessions differ a lot: at 3 s, FBCCA-norm reaches 58.3 %, 77.8 % and 88.9 %.

To reproduce: the `--data recording` command under [Usage](#usage).

### Replay through the online backend

`scripts/online.py --replay` feeds a recording through the backend as if it arrived
live. Here the backend trained on two pilot sessions and replayed the third, with a 2 s
window:

| Method | Session 1 | Session 2 | Session 3 |
|---|---|---|---|
| CCA | 38.9 % | 50.0 % | 55.6 % |
| FBCCA | 36.1 % | 58.3 % | 52.8 % |
| FBCCA-norm | 44.4 % | 58.3 % | 72.2 % |
| eTRCA | 13.9 % | 38.9 % | 38.9 % |

- CCA and FBCCA match `compare_methods.py` with artifact rejection off exactly, so the
  online path processes trials the same way as the offline one.
- eTRCA falls well below its results within one session (38.9 %, 44.4 %, 52.8 %), so
  online sessions should be calibrated in the same sitting.

Over LSL with the fake EEG stream and a scripted stand-in for the GUI, the backend
copy-spelled two short phrases with all 8 selections correct. Answers arrived about
0.78 s after the flicker ended, of which classification took 12–28 ms.

## Hardware

- BrainAccess MIDI EEG cap: 16 dry electrodes, 250 Hz, streaming over LSL. The system
  uses 8 of them, `config.CAP_CHANNELS`: P7, P3, Pz, P4, P8, O1, Oz, O2.
- Monitor with 60 Hz and 120 Hz refresh rates; the stimulus runs at 120 Hz

## Setup

Python 3.11 or newer (developed on 3.12). The stimulus window has been tested on
Linux (GNOME on X11).

```bash
python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # package + pytest, pylint, black
.venv/bin/pip install -e ".[analysis]"   # notebooks: jupyterlab, seaborn, mne
.venv/bin/pip install -e ".[nlp]"        # word prediction: transformers, wordfreq
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

`torch` is left out of the extras on purpose. Install the CPU wheel as shown above,
otherwise pip downloads several GB of CUDA libraries.

### Benchmark dataset

The offline analysis uses the Tsinghua SSVEP Benchmark dataset (Wang et al. 2017):
35 subjects, 40 targets (8–15.8 Hz), 64 channels, 250 Hz, 6 blocks. The files go into
`data/benchmark/`, which is gitignored.

```bash
mkdir -p data/benchmark && cd data/benchmark
base=https://bci.med.tsinghua.edu.cn/upload/yijun
wget -c $base/Freq_Phase.mat $base/64-channels.loc $base/Readme.txt $base/Sub_info.txt
for s in 1 2 3; do wget -c $base/S$s.mat.7z && 7z x S$s.mat.7z; done   # up to S35, ~105 MB each
```

## Usage

Every script lists its options with `--help`.

### Compare the classifiers offline

```bash
.venv/bin/python scripts/compare_methods.py --data synthetic                # no download needed
.venv/bin/python scripts/compare_methods.py --data benchmark --subjects 1 2 3
.venv/bin/python scripts/compare_methods.py --data benchmark --methods CCA FBCCA --windows 0.5 1 2
.venv/bin/python scripts/compare_methods.py --data recording --recordings data/recordings/pilot_*.npz --windows 0.5 1 1.5 2 2.5 3 4 --name pilot
```

The script writes `results/<name>.csv`, with one row per subject × method × window, and
`results/<name>.png`, a plot of accuracy and ITR against window length. The name
defaults to `compare_<data>`. `--trial-ratio 0 --channel-ratio 0` switches artifact
rejection off.

### Check the EEG stream

```bash
.venv/bin/python scripts/check_lsl.py                          # list the streams on the network
.venv/bin/python scripts/check_lsl.py --type EEG --seconds 10  # sampling rate, per-channel std
.venv/bin/python scripts/check_lsl.py --type EEG --alpha       # eyes open/closed: alpha peak at O1/O2
```

### Flicker timing test and calibration

Before running these, set the monitor to 120 Hz (`config.REFRESH_RATE`) in the
display settings, then log out and back in.

```bash
.venv/bin/python scripts/calibration.py --test --seconds 30       # flip log -> results/timing/
.venv/bin/python scripts/calibration.py --no-recorder --blocks 1  # dry run without EEG
.venv/bin/python scripts/calibration.py --subject S01             # full session
```

A session has 10 blocks by default (`--blocks`, `config.N_BLOCKS`), about 7 minutes of
flicker and cues. Each block shows every target once, in random order: a 2 s cue (red
frame) followed by 5 s of flicker. The subject starts each block with SPACE; ESC quits.
Event markers go to the LSL stream `SSVEP-Markers` as JSON strings, each timestamped
with the flip that showed the event.

The 6 targets sit on a ring around the screen centre, where the speller shows the
typed text, in the same positions the speller uses. `config.TARGET_SLOTS` decides
which frequency goes where: clockwise from the top left, 8.0, 10.8, 13.6, 9.4, 12.2 and
15.0 Hz, so targets next to each other are at least 2.8 Hz apart.

### Record a calibration session

Run each command in its own terminal:

```bash
.venv/bin/python scripts/fake_eeg_stream.py                   # only without the cap: synthetic EEG with an SSVEP on cued trials
.venv/bin/python scripts/record.py --subject S01              # add --eeg-name FakeEEG for the fake stream
.venv/bin/python scripts/calibration.py --subject S01
```

The recorder saves `data/recordings/<subject>_<date>_<time>.npz`. Its format is
described at the top of [ssvep/acquisition.py](ssvep/acquisition.py).
`compare_methods.py --data recording --recordings data/recordings/*.npz` then runs the
same comparison on your own data.

### Online speller

The GUI runs the synchronous trials and draws what the backend sends; the backend
classifies and owns the speller state. Run each command in its own terminal:

```bash
.venv/bin/python scripts/fake_eeg_stream.py              # only without the cap
.venv/bin/python scripts/online.py --method FBCCA-norm --calibration data/recordings/S01_*.npz --phrases "hello world" --cue
.venv/bin/python scripts/speller_gui.py --subject S01    # waits for the backend
.venv/bin/python scripts/record.py --subject S01         # then press SPACE in the GUI
```

- `--phrases` gives the phrases to copy, in order. Without it, the speller runs in free
  spelling.
- `--cue` frames the next target before each flicker.
- `--eeg-name FakeEEG` picks the fake stream if the cap is streaming too.
- `--window` is the classifier window, 2 s by default. It must not be longer than the
  GUI's `--stim`, so a 3 s window needs `online.py --window 3` and
  `speller_gui.py --stim 3`.

One trial: 0.5 s with the labels shown (a red frame if the backend cues a target),
2 s of flicker with the labels hidden (so the stimulus looks the same as in
calibration), then the answer: a green frame around the selected target and the new
text for 0.5 s, then the new labels. The answer comes at least 0.64 s after a 2 s
flicker ends, because the backend waits for the visual latency and for 0.5 s of EEG
after the window, which keeps filter edge effects out of it. SPACE starts and pauses,
ESC quits.

Each session leaves these files in `data/recordings/`:

- `<subject>_<date>_online.csv`: the GUI's trial table
- `<subject>_<date>_backend.csv`: the backend's decisions and scores, saved on Ctrl+C
- `<subject>_<date>.npz`: the recording, which loads with `load_recording` like a
  calibration session

`speller_gui.py --demo --windowed` runs without a backend: a stand-in cues a random
target in every trial and selects it. On the real screen (without `--windowed`) it
prints the dropped frames, the same check as `calibration.py --test`.

### Replay a recording

```bash
.venv/bin/python scripts/online.py --method FBCCA-norm --window 3 --calibration A.npz B.npz --replay C.npz
```

This runs a recording through the same backend code without LSL or the GUI, prints the
accuracy against the recorded targets, and saves the decision log. Train and replay on
different recordings, because a trained method scores its own training trials too well.

## How it works

### Signal processing

Filtering happens before cropping, so filter edge effects fall outside the analysis
window.

```
raw epochs (n_trials, n_channels, n_samples), µV, including 0.5 s before stimulus onset
  -> notch filter, 50 Hz
  -> band-pass 7–90 Hz, or a 5-band Chebyshev filter bank (Chen et al. 2015)
  -> crop [onset + 0.14 s, onset + 0.14 s + window]
  -> classifier input (n_trials, n_bands, n_channels, n_window)
```

Every classifier is a scikit-learn estimator that takes this 4-D input. Labels are
indices into `config.TARGET_FREQS`. With one band, `CCA` and `TRCA` are the plain
methods; with filter-bank input they become FBCCA and FB-(e)TRCA.

### Online system

```
┌──────────────────────────┐  LSL "SSVEP-Markers" (JSON)  ┌───────────────────────────────┐
│ GUI process (pyglet)     │ ───────────────────────────▶ │ Backend process               │
│ • flicker, frame clock   │                              │ • acquisition ring buffer     │
│ • draws labels and text  │ ◀─────────────────────────── │ • preprocessing + classifier  │
└──────────────────────────┘  LSL "SSVEP-State" (JSON)    │ • speller state + NLP         │
                                                          └───────────────▲───────────────┘
                               BrainAccess cap ── LSL "EEG" ──────────────┘
```

The GUI only draws, so model inference never delays a frame. The backend owns the
speller state and tells the GUI what to show. The messages in both directions are
described in [ssvep/protocol.py](ssvep/protocol.py).

For each trial, the backend cuts `[onset - 0.5 s, onset + 0.14 s + window + 0.5 s]` from
the ring buffer and applies the processing above. Calibration epochs are cut to the same
length before training, so the filters see the same signal length online and in
training. The backend answers a trial only when its flicker has finished and its whole
epoch has arrived.

### Speller

The menu has 6 groups of 6 cells, drawn on the 6 targets. The first selection opens a
group and the second picks a cell in it, so a character takes 2 selections:

| Group | Cells |
|---|---|
| 1–4 | A–E, F–J, K–O, P–T, each with ↶ (back) |
| 5 | U–Z |
| 6 | 3 word suggestions, ␣ (space), ⌫ (backspace), ↶ (back) |

Back closes a wrongly opened group in one selection. Accepting a suggestion replaces the
word being typed and adds a space, so a word of any length costs 2 selections. In
copy-spelling, `Speller.next_target()` gives the shortest correct next selection,
including corrections after a mistake. The GUI can cue it, and the logs keep it as ground
truth; the classifier never uses it.

## Project layout

```
ssvep/
  config.py          constants: sampling rates, channels, target frequencies, timing
  datasets.py        Benchmark, own recordings, synthetic data -> (X, y, blocks)
  preprocessing.py   notch, band-pass, filter bank, crop, bad trials/channels
  cca.py             reference signals, CCA / FBCCA
  trca.py            TRCA, eTRCA, FB-eTRCA
  baseline.py        PSD features + LDA
  evaluation.py      ITR, leave-one-block-out, window sweep
  stimulus.py        flicker luminance, target layout, trial order, frame timing
  acquisition.py     LSL inlets, ring buffer, recorder
  protocol.py        JSON messages between the speller GUI and the backend
  speller.py         speller state machine: menu, typed text, copy-spelling target
scripts/
  check_lsl.py       inspect a live LSL stream
  compare_methods.py offline comparison -> results/*.csv, *.png
  calibration.py     calibration GUI and flicker timing test
  fake_eeg_stream.py synthetic EEG stream that reacts to the markers
  record.py          record EEG + markers to .npz
  online.py          online backend, and replay of a recording
  speller_gui.py     online speller GUI (--demo runs without a backend)
notebooks/           exploring Benchmark, CCA and FBCCA on Benchmark
tests/               pytest; uses synthetic data, Benchmark tests are skipped without the files
```

## Development

```bash
.venv/bin/pytest -q
.venv/bin/pylint ssvep scripts --disable=all --enable=E,F --disable=E1121 --score=no
.venv/bin/black --check ssvep scripts tests
```

GitHub Actions runs these three checks on every push to `main` and on pull requests.

## Thesis topic

### Engineering aspect / research aspect
The thesis focuses on the engineering design and implementation of a real-time data processing pipeline capable of acquiring, filtering, and classifying steady-state visual evoked potentials (SSVEP) from raw EEG streams. The research aspect involves evaluating different spatial filtering techniques (e.g., TRCA, CCA) and machine learning algorithms, combined with an NLP-based predictive text engine, to maximize the Information Transfer Rate (ITR) and classification accuracy.

### Purpose of the thesis
The main purpose of this thesis is to develop a fully functional, non-invasive BCI software system that allows users with severe motor dysfunctions to communicate efficiently. The system aims to significantly reduce user fatigue and improve communication speed by integrating intelligent word prediction (NLP) with real-time bioelectric signal classification.

### Tasks to be performed / scope of the thesis
1. Review of current literature on BCI systems, SSVEP paradigms, and real-time EEG signal processing methods.
2. Design and implementation of a graphical user interface with precise stimulation timing.
3. Implementation of an API for real-time hardware integration and EEG data acquisition.
4. Development of a signal processing pipeline, including artifact removal and temporal/spatial filtering.
5. Training and evaluation of machine learning / statistical classification algorithms (e.g., CCA, TRCA) using offline data.
6. Integration of a Natural Language Processing (NLP) model for real-time predictive text generation to enhance the Information Transfer Rate.
7. System integration, conducting online experiments, and evaluating the system's overall performance (Accuracy, ITR).

### Bibliography
[1] Chen, X., Wang, Y., Nakanishi, M., Gao, X., Jung, T.-P., & Gao, S. "High-speed spelling with a noninvasive brain–computer interface". Proceedings of the National Academy of Sciences, 112(44), E6058-E6067. https://doi.org/10.1073/pnas.1508080112 (2015)

[2] Chen, X., Chen, Z., Gao, S., & Gao, X. "A high-ITR SSVEP-based BCI speller". Brain-Computer Interfaces, 1(3-4), 181-191. https://www.researchgate.net/publication/269998531_A_high-ITR_SSVEP-based_BCI_speller (2014)

[3] Liu, B., Huang, X., Wang, Y., Chen, X., & Gao, X. "BETA: A Large Benchmark Database Toward SSVEP-BCI Application". Frontiers in Neuroscience, 14, 627. https://doi.org/10.3389/fnins.2020.00627 (2020)

[4] Nakanishi, M., Wang, Y., Chen, X., Wang, Y.-T., Gao, X., & Jung, T.-P. "Enhancing Detection of SSVEPs for a High-Speed Brain Speller Using Task-Related Component Analysis". IEEE Transactions on Biomedical Engineering, 65(1), 104-112. https://doi.org/10.1109/TBME.2017.2694818 (2018)

### Other references used in the code
- Lin, Z., Zhang, C., Wu, W., & Gao, X. "Frequency recognition based on canonical correlation analysis for SSVEP-based BCIs". IEEE Transactions on Biomedical Engineering, 53(12), 2610-2614. https://doi.org/10.1109/TBME.2006.886577 (2006)
- Wang, Y., Chen, X., Gao, X., & Gao, S. "A Benchmark Dataset for SSVEP-Based Brain–Computer Interfaces". IEEE Transactions on Neural Systems and Rehabilitation Engineering, 25(10), 1746-1752. https://doi.org/10.1109/TNSRE.2016.2627556 (2017)
- Wolpaw, J. R., Birbaumer, N., McFarland, D. J., Pfurtscheller, G., & Vaughan, T. M. "Brain–computer interfaces for communication and control". Clinical Neurophysiology, 113(6), 767-791. https://doi.org/10.1016/S1388-2457(02)00057-3 (2002)
