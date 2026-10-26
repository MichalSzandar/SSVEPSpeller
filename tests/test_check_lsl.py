import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_lsl.py"
SPEC = importlib.util.spec_from_file_location("check_lsl", MODULE_PATH)
check_lsl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_lsl)


def test_stddev_by_channel_pairs_labels_and_values():
    labels = ["O1", "Oz"]
    samples = np.array([
        [0.10, 0.20],
        [0.40, 0.50],
        [0.30, 0.60],
    ])

    pairs = check_lsl.stddev_by_channel(labels, samples)

    assert pairs[0][0] == "O1"
    assert pairs[1][0] == "Oz"
    assert np.isclose(pairs[0][1], np.std(samples[:, 0]))
    assert np.isclose(pairs[1][1], np.std(samples[:, 1]))


def test_select_stream_prefers_non_marker_without_type():
    marker = SimpleNamespace(type=lambda: "Markers", name=lambda: "MarkerStream")
    eeg = SimpleNamespace(type=lambda: "EEG", name=lambda: "LSLExampleAmp")

    assert check_lsl.select_stream([marker, eeg], None) is eeg


def test_print_recorded_info_handles_short_recording(capsys):
    check_lsl.print_recorded_info(np.array([1.0]), np.array([0.0]), ["Marker"])

    captured = capsys.readouterr().out
    assert "at least 2 samples" in captured.lower()
