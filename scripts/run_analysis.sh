#!/bin/bash
# Every analysis of the thesis results (Ch. 6), then the figures.
#
# Usage, from the repository root:
#   scripts/run_analysis.sh                 all steps
#   scripts/run_analysis.sh benchmark own   only these steps
#   FORCE=1 scripts/run_analysis.sh own     redo a step even if its results exist
#
# Steps (results go to results/, figures to thesis/figures/):
#   benchmark    35 Benchmark subjects, the 6 speller targets, all methods, with and
#                without artifact rejection; statistics; artifact ratios   (~25 min)
#   benchmark40  35 subjects, all 40 targets, CCA/FBCCA/TRCA family: the check
#                against Nakanishi et al. 2018                              (~40 min)
#   own          both own calibrations: methods vs. window, latency sweep,
#                cross-session transfer, artifact ratios                    (~5 min)
#   online       every online run (data/online_runs.csv), replayed at other
#                latencies                                                  (~3 min)
#   timing       flip logs of the own sessions and the flicker tests        (seconds)
#   figures      thesis/figures/make_results.py: every figure whose results exist
# A step is skipped if its main result file exists, unless FORCE=1.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python
OUT=results
# The own dataset (thesis Sec. 4.1): two 10-block calibrations, both of the author
SESSION1=data/recordings/pilot_20261110_172435.npz  # Nov 10, EEG timestamps ~0.5 s early
SESSION2=data/recordings/S02_20261116_195733.npz    # Nov 16
OWN_LATENCY=0.7          # s, used with the cap (Sec. 4.1)
SESSION2_ALIGNED=0.2     # s, Nov 16 latency that matches Nov 10 at 0.7 s minus the 0.5 s lead
SUBJECTS=$(seq -s ' ' 1 35)

needed() {  # needed <file>: true if the step that makes <file> must run
  [ "${FORCE:-0}" = 1 ] || [ ! -f "$1" ]
}

step_benchmark() {
  local missing
  missing=$(for s in $SUBJECTS; do [ -f data/benchmark/S$s.mat ] || echo -n "S$s "; done)
  if [ -n "$missing" ]; then
    echo "Missing Benchmark files: $missing(see PLAN.md, Tools: Datasets)"; exit 1
  fi
  if needed $OUT/benchmark_6.csv; then
    $PY scripts/compare_methods.py --data benchmark --subjects $SUBJECTS --name benchmark_6
  fi
  if needed $OUT/benchmark_6_raw.csv; then
    $PY scripts/compare_methods.py --data benchmark --subjects $SUBJECTS \
      --trial-ratio 0 --channel-ratio 0 --name benchmark_6_raw
  fi
  $PY scripts/method_stats.py $OUT/benchmark_6.csv
  if needed $OUT/ratios_benchmark_trials.csv; then
    $PY scripts/artifact_ratios.py --data benchmark --subjects $SUBJECTS --window 2 \
      --name ratios_benchmark
  fi
}

step_benchmark40() {
  if needed $OUT/benchmark_40.csv; then
    $PY scripts/compare_methods.py --data benchmark --targets all --subjects $SUBJECTS \
      --methods CCA FBCCA TRCA eTRCA FB-eTRCA --windows 0.25 0.5 0.75 1.0 \
      --name benchmark_40
  fi
}

step_own() {
  local windows="0.5 1 1.5 2 2.5 3 4"
  if needed $OUT/own.csv; then
    $PY scripts/compare_methods.py --data recording --recordings $SESSION1 $SESSION2 \
      --latency $OWN_LATENCY --windows $windows --name own
  fi
  if needed $OUT/own_raw.csv; then
    $PY scripts/compare_methods.py --data recording --recordings $SESSION1 $SESSION2 \
      --latency $OWN_LATENCY --windows $windows --trial-ratio 0 --channel-ratio 0 \
      --name own_raw
  fi
  if needed $OUT/own_latency.csv; then
    $PY scripts/latency_sweep.py --recordings $SESSION1 $SESSION2 \
      --latencies 0.14 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --windows 1 2 \
      --methods CCA FBCCA CCA-norm FBCCA-norm eTRCA FB-eTRCA --name own_latency
  fi
  if needed $OUT/cross_session.csv; then
    $PY scripts/cross_session.py --recordings $SESSION1 $SESSION2 --windows 1 2 3 \
      --name cross_session
  fi
  if needed $OUT/cross_session_aligned.csv; then
    $PY scripts/cross_session.py --recordings $SESSION1 $SESSION2 --windows 1 2 3 \
      --latencies $OWN_LATENCY $SESSION2_ALIGNED --name cross_session_aligned
  fi
  if needed $OUT/ratios_own_trials.csv; then
    $PY scripts/artifact_ratios.py --data recording --recordings $SESSION1 $SESSION2 \
      --latency $OWN_LATENCY --window 2 --name ratios_own
  fi
}

step_online() {
  if needed $OUT/online_runs.csv; then
    $PY scripts/analyse_online.py --runs data/online_runs.csv \
      --replay-latencies 0.14 0.5 0.6 0.7 --name online
  fi
}

step_timing() {
  # The own sessions' flip logs (calibration and online) and every flicker test
  $PY scripts/analyse_timing.py data/recordings/pilot_20261110_*flips.npz \
    data/recordings/S02_20261116_*flips.npz results/timing/flicker_*.npz --name timing
}

step_figures() {
  $PY thesis/figures/make_results.py
}

steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(benchmark benchmark40 own online timing figures)
for step in "${steps[@]}"; do
  case $step in
    benchmark | benchmark40 | own | online | timing | figures)
      echo "=== $step ==="; "step_$step" ;;
    *) echo "Unknown step: $step"; exit 1 ;;
  esac
done
