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
TARGET_FREQS = [8.0, 10.0, 12.0, 14.0, 15.0]  # Hz
TARGET_PHASES = [0.0] * len(TARGET_FREQS)  # rad, one per target (JFPM)

# --- Processing ----------------------------------------------------------------
BANDPASS = (7.0, 90.0)  # Hz, single band for non-filter-bank methods
N_HARMONICS = 5  # harmonics in the CCA reference signals
N_SUBBANDS = 5  # filter-bank size (Chen 2015)
FB_A, FB_B = 1.25, 0.25  # filter-bank weight parameters (Chen 2015)
GAZE_SHIFT = 0.5  # s, added to the window length when computing ITR
