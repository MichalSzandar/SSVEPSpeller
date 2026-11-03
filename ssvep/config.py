"""Project-wide constants shared by offline analysis, calibration and the online speller."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RECORDINGS_DIR = DATA_DIR / "recordings"

# --- Benchmark dataset (Wang et al.) -----------------------------------------
BENCHMARK_DIR = DATA_DIR / "benchmark"
BENCHMARK_FS = 250  # Hz
BENCHMARK_ONSET = 0.5  # s, stimulus onset inside each 6 s epoch
BENCHMARK_LATENCY = 0.14  # s, visual latency used by Chen 2015 / Nakanishi 2018
# The 9 parieto-occipital channels used by most Benchmark papers
BENCHMARK_CHANNELS = ["Pz", "PO5", "PO3", "POz", "PO4", "PO6", "O1", "Oz", "O2"]

# --- Hardware ------------------------------------------------------------------
EEG_FS = 250  # Hz, BrainAccess
LINE_FREQ = 50.0  # Hz, mains frequency
REFRESH_RATE = 120  # Hz, monitor refresh rate
# TODO: cap electrodes over the occipital area
CAP_CHANNELS: list[str] = []

# --- Stimulus ------------------------------------------------------------------
TARGET_FREQS = [8.0, 9.4, 10.8, 12.2, 13.6, 15.0]  # Hz, all on the Benchmark grid
TARGET_PHASES = [0.0] * len(TARGET_FREQS)  # rad, one per target (JFPM)
# Targets sit on a ring around the typed text (Hex-o-Spell style), slots numbered
# clockwise from the top left. Target k is drawn in slot TARGET_SLOTS[k], which puts
# 8.0, 10.8, 13.6, 9.4, 12.2, 15.0 Hz clockwise: spatial neighbours are >= 2.8 Hz apart.
TARGET_SLOTS = [0, 3, 1, 4, 2, 5]

# --- Calibration protocol --------------------------------------------------------
N_BLOCKS = 6  # every block shows each target once, in random order
CUE_DURATION = 2.0  # s, red frame on the next target
STIM_DURATION = 5.0  # s, flicker per trial
MARKER_STREAM = "SSVEP-Markers"  # LSL stream with the GUI's JSON event markers

# --- Processing ----------------------------------------------------------------
BANDPASS = (7.0, 90.0)  # Hz, single band for non-filter-bank methods
N_HARMONICS = 5  # harmonics in the CCA reference signals
N_SUBBANDS = 5  # filter-bank size (Chen 2015)
FB_A, FB_B = 1.25, 0.25  # filter-bank weight parameters (Chen 2015)
GAZE_SHIFT = 0.5  # s, added to the window length when computing ITR
