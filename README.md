# Design and Implementation of a Real-Time Brain–Computer Interface for Predictive EEG-Based Communication Using Machine Learning and Natural Language Processing

Engineering thesis: an SSVEP speller. The user looks at one of several targets that
flicker at different frequencies, the EEG over the visual cortex follows that
frequency, and a classifier decides which target was chosen. A language model
suggests words, so fewer selections are needed per word.

The `ssvep` package holds the signal processing and the classifiers. The offline
analysis, the calibration app and the online speller all use the same code.

## Status

| Part | State |
|---|---|
| Offline analysis on the Benchmark dataset: filtering, CCA, FBCCA, TRCA, eTRCA, FB-eTRCA, PSD+LDA baseline, ITR, leave-one-block-out cross-validation | done |
| Calibration app: sampled-sinusoid flicker at 120 Hz, cue → flicker → rest protocol, LSL markers timestamped at the frame flip, dropped-frame log | done |
| Acquisition over LSL: inlets, ring buffer, recorder, fake EEG stream for testing without the cap | in progress |
| Bad-trial and bad-channel rejection | planned |
| Online backend and speller GUI | planned |
| Word prediction: unigram baseline, distilgpt2 re-ranking | planned |
| Online copy-spelling experiment, with and without word prediction | planned |

## Offline results so far

Benchmark dataset, subjects S1–S3, 5 targets (8, 10, 12, 14, 15 Hz), 9
parieto-occipital channels, leave-one-block-out cross-validation (6 folds). The
window starts 0.14 s after stimulus onset (visual latency). ITR counts the window plus
0.5 s for the gaze shift. Values are means over the 3 subjects, so they are only a
first check; chance level is 20 %.

| Method | Accuracy at 1 s | Best ITR (bits/min) | Window at best ITR |
|---|---|---|---|
| CCA | 88.9 % | 70.3 | 0.75 s |
| FBCCA | 95.6 % | 91.9 | 0.75 s |
| TRCA | 97.8 % | 94.2 | 0.5 s |
| eTRCA | 100 % | 123.7 | 0.25 s |
| FB-eTRCA | 100 % | 133.8 | 0.5 s |
| PSD+LDA | 63.3 % | 28.3 | 1.0 s |

To reproduce: `scripts/compare_methods.py --data benchmark --subjects 1 2 3` (see
[Usage](#usage)).

## Hardware

- BrainAccess 32-channel EEG cap (dry electrodes, 250 Hz), streaming over LSL
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
```

The script writes `results/compare_<data>.csv`, with one row per subject × method ×
window, and `results/compare_<data>.png`, a plot of accuracy and ITR against window
length.

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

A session has 6 blocks. Each block shows every target once, in random order: a 2 s
cue (red frame) followed by 5 s of flicker. The subject starts each block with SPACE;
ESC quits. Event markers go to the LSL stream `SSVEP-Markers` as JSON strings, each
timestamped with the flip that showed the event.

### Record a calibration session (in progress)

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

### Online system (planned)

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
speller state and tells the GUI what to show.

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
scripts/
  check_lsl.py       inspect a live LSL stream
  compare_methods.py offline comparison -> results/*.csv, *.png
  calibration.py     calibration GUI and flicker timing test
  fake_eeg_stream.py synthetic EEG stream that reacts to the markers
  record.py          record EEG + markers to .npz
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
