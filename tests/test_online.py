"""Tests for scripts/online.py: decoder, backend and replay, on synthetic EEG (no LSL)."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ssvep import acquisition, config, protocol
from ssvep.preprocessing import RejectArtifacts, crop, notch
from ssvep.speller import Speller

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # online.py imports compare_methods from there
SPEC = importlib.util.spec_from_file_location("online", SCRIPTS / "online.py")
online = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(online)
import compare_methods  # found through SCRIPTS on sys.path

FS = config.EEG_FS
T0 = 259178.0  # s, LSL time of the first sample: seconds since boot, far from 0
CAP = config.CAP_CHANNELS
SLOTS = config.TARGET_SLOTS
N_TARGETS = len(config.TARGET_FREQS)
LATENCY = config.BENCHMARK_LATENCY
WINDOW = 1.0  # s, classifier window in most tests
# One gain per CAP channel, the same in every session (one head, many recordings)
GAINS = dict(zip(CAP, np.random.default_rng(12345).uniform(0.5, 1.5, len(CAP))))


class Session:
    """Continuous synthetic EEG and the speller GUI's markers, built trial by trial.

    White noise (sd 1) on every channel. During a trial every CAP channel also carries
    GAINS[channel] * sin(2*pi*f*(t - onset - LATENCY)) from LATENCY s after stim_on
    until LATENCY s after stim_off: a time-locked SSVEP, like the fake EEG stream's.
    Other channels carry a strong, constant sine at the last target's frequency, so a
    decoder that reads them picks that target. Each stim_on comes 1 ms before a sample,
    as a flip falls between two samples.
    """

    def __init__(self, channels=CAP, seed=0, stim=2.0, freqs=config.TARGET_FREQS):
        self.channels = list(channels)
        self.rng = np.random.default_rng(seed)
        self.stim = stim
        self.freqs = list(freqs)
        self.gains = np.array([GAINS.get(c, 0.0) for c in self.channels])
        self.not_cap = np.array([c not in CAP for c in self.channels])
        self.eeg = np.zeros((0, len(self.channels)))  # (samples, channels)
        self.markers = []  # (LSL time, JSON string), in time order
        self.n_trials = 0
        self.mark(
            T0,
            "session_start",
            mode="online",
            freqs=self.freqs,
            phases=[0.0] * len(self.freqs),
            stim_s=stim,
        )
        self.mark(T0 + 0.5, "block_start", block=0)
        self.add(1.0)

    @property
    def times(self):
        return T0 + np.arange(len(self.eeg)) / FS

    def mark(self, t, event, **fields):
        self.markers.append((t, json.dumps({"event": event, **fields})))

    def add(self, seconds, freq=None, onset=None):
        """Append ``seconds`` of EEG; with ``freq``, the SSVEP of a trial whose stim_on
        came at LSL time ``onset``."""
        n = int(round(seconds * FS))
        t = T0 + (len(self.eeg) + np.arange(n)) / FS
        x = self.rng.normal(0.0, 1.0, (n, len(self.channels)))
        x[:, self.not_cap] += 20 * np.sin(2 * np.pi * self.freqs[-1] * t)[:, None]
        if freq is not None:
            since = t - onset - LATENCY
            on = (since >= 0) & (since < self.stim)
            x[on] += np.sin(2 * np.pi * freq * since[on])[:, None] * self.gains
        self.eeg = np.concatenate([self.eeg, x])

    def trial(self, target, marker_target=None, complete=True, gaze=1.0, after=0.5):
        """Append ``gaze`` s of noise, a flicker of ``target``'s frequency and ``after``
        s more -> the stim_on time. The markers carry ``marker_target`` if given (a
        wrong ground truth), else ``target``."""
        self.add(gaze)
        onset = T0 + len(self.eeg) / FS - 0.001
        logged = int(target if marker_target is None else marker_target)
        info = dict(
            trial=self.n_trials, block=0, target=logged, freq=self.freqs[logged]
        )
        self.mark(onset - gaze / 2, "cue", **info)
        self.mark(onset, "stim_on", **info)
        self.mark(onset + self.stim, "stim_off", **info, n_dropped=0, complete=complete)
        self.add(self.stim + after, self.freqs[target], onset)
        self.n_trials += 1
        return onset

    def save(self, path):
        """Write a recording in the Recorder format (top of ssvep/acquisition.py)."""
        markers = self.markers + [
            (self.times[-1] - 0.1, json.dumps({"event": "block_end", "block": 0})),
            (self.times[-1] - 0.05, json.dumps({"event": "session_end"})),
        ]
        np.savez_compressed(
            path,
            eeg=self.eeg.T.astype(np.float32),
            eeg_times=self.times,
            markers=np.array([m for _, m in markers], dtype=str),
            marker_times=np.array([t for t, _ in markers]),
            channels=np.array(self.channels),
            fs=float(FS),
        )


class Feeder:
    """Feeds a Session (CAP channels only) to a Backend like the live loop: the EEG in
    chunks of 5 samples, then every marker up to the chunk's last sample, then poll."""

    def __init__(self, backend, session):
        self.backend = backend
        self.session = session
        self.n_samples = 0  # samples fed so far
        self.n_markers = 0  # markers fed so far

    def run(self, until=np.inf):
        """Feed everything up to LSL time ``until`` -> the backend's messages."""
        times, eeg, markers = self.session.times, self.session.eeg, self.session.markers
        end = int(np.searchsorted(times, until, side="right"))
        messages = []
        while self.n_samples < end:
            stop = min(self.n_samples + 5, end)
            self.backend.buffer.append(
                eeg[self.n_samples : stop], times[self.n_samples : stop]
            )
            self.n_samples = stop
            while (
                self.n_markers < len(markers)
                and markers[self.n_markers][0] <= times[stop - 1]
            ):
                t, marker = markers[self.n_markers]
                messages += self.backend.on_marker(marker, t)
                self.n_markers += 1
            messages += self.backend.poll()
        return messages


def calibration(n_blocks=4, seed=1, stim=2.0):
    """(X, y) of a synthetic calibration session, cut like load_recording: from PRE s
    before the onset sample, the flicker and PRE s more."""
    session = Session(seed=seed, stim=stim)
    y = np.tile(np.arange(N_TARGETS), n_blocks)
    onsets = [session.trial(target, after=1.0) for target in y]
    pre = int(round(online.PRE * FS))
    n = pre + int(round(stim * FS)) + pre
    starts = np.searchsorted(session.times, onsets) - pre
    X = np.stack([session.eeg[s : s + n].T for s in starts])
    return X, y


def make_backend(phrases=(), cue=False, window=WINDOW):
    X, y = calibration(n_blocks=2)
    decoder = online.Decoder("CCA", window=window).fit(X, y)
    buffer = acquisition.RingBuffer(len(CAP), int(online.BUFFER_SECONDS * FS))
    return online.Backend(decoder, Speller(), buffer, phrases, cue)


def answers(messages):
    """The decoded states among ``messages`` that answer a trial."""
    states = [protocol.parse_state(m) for m in messages]
    return [s for s in states if s["trial"] is not None]


# --- cap_indices -------------------------------------------------------------------


def test_cap_indices_pick_the_cap_channels_in_order():
    labels = ["Fz", *reversed(CAP), "contact_O1", "Accel_x"]
    indices = online.cap_indices(labels)
    assert [labels[i] for i in indices] == CAP


def test_cap_indices_name_the_missing_channels():
    labels = [c for c in CAP if c != CAP[1]]
    with pytest.raises(ValueError, match=CAP[1]):
        online.cap_indices(labels)


# --- load_calibration --------------------------------------------------------------


def test_load_calibration_joins_recordings_in_order(tmp_path):
    first = Session(channels=[*CAP, "Fz"], seed=1, stim=3.0)
    for target in [0, 1, 2]:
        first.trial(target, after=1.0)
    first.save(tmp_path / "first.npz")
    second = Session(channels=["Fz", *reversed(CAP)], seed=2, stim=2.0)
    for target in [5, 4]:
        second.trial(target, after=1.0)
    second.save(tmp_path / "second.npz")

    X, y = online.load_calibration([tmp_path / "first.npz", tmp_path / "second.npz"])
    pre = int(round(online.PRE * FS))
    assert X.shape == (5, len(CAP), pre + int(2.0 * FS) + pre)  # the shorter epochs
    assert y.tolist() == [0, 1, 2, 5, 4]


def test_load_calibration_rejects_other_frequencies(tmp_path):
    session = Session(freqs=[8.0, 10.0, 12.0, 14.0, 15.0, 9.0])
    session.trial(0, after=1.0)
    session.save(tmp_path / "old.npz")
    with pytest.raises(ValueError):
        online.load_calibration([tmp_path / "old.npz"])


# --- Decoder -----------------------------------------------------------------------


def test_epoch_length():
    decoder = online.Decoder("CCA", window=1.5, latency=0.1, fs=250)
    assert decoder.n_samples == round((online.PRE + 0.1 + 1.5 + online.POST) * 250)


@pytest.mark.parametrize("method", ["CCA", "FBCCA"])
def test_prepare_is_the_offline_pipeline(method):
    """notch -> compare_methods.prefilter -> crop at PRE + latency, as offline."""
    decoder = online.Decoder(method, window=WINDOW)
    X = np.random.default_rng(0).normal(0.0, 10.0, (3, len(CAP), decoder.n_samples))
    n_subbands = compare_methods.METHODS[method][0]
    expected = crop(
        compare_methods.prefilter(notch(X, FS, config.LINE_FREQ), FS, n_subbands),
        FS,
        online.PRE + LATENCY,
        WINDOW,
    )
    prepared = decoder.prepare(X)
    assert prepared.shape == (3, n_subbands, len(CAP), int(WINDOW * FS))
    np.testing.assert_allclose(prepared, expected)


def test_fit_returns_the_decoder_with_artifact_rejection():
    """Like the offline comparison: the classifier is wrapped in RejectArtifacts with
    the decoder's thresholds."""
    X, y = calibration(n_blocks=2)
    decoder = online.Decoder("CCA", window=WINDOW, trial_ratio=4.0, channel_ratio=2.5)
    assert decoder.fit(X, y) is decoder
    assert isinstance(decoder.estimator_, RejectArtifacts)
    assert decoder.estimator_.trial_ratio == 4.0
    assert decoder.estimator_.channel_ratio == 2.5


@pytest.mark.parametrize("method", ["CCA", "FBCCA-norm", "eTRCA"])
def test_classify_recovers_the_targets(method):
    X, y = calibration(seed=1)
    decoder = online.Decoder(method, window=WINDOW).fit(X, y)
    X_test, y_test = calibration(n_blocks=1, seed=2)
    for epoch, target in zip(X_test[..., : decoder.n_samples], y_test):
        selected, scores = decoder.classify(epoch)
        assert selected == target
        assert np.shape(scores) == (N_TARGETS,)
        assert np.argmax(scores) == selected


def test_fit_cuts_long_epochs_to_the_online_length():
    """Fitting on 3 s calibration epochs gives the same classifier as fitting on those
    epochs cut to the online length first (the filters then see the same signal)."""
    X, y = calibration(seed=1)
    X_test, _ = calibration(n_blocks=1, seed=2)
    long = online.Decoder("eTRCA", window=WINDOW).fit(X, y)
    short = online.Decoder("eTRCA", window=WINDOW).fit(X[..., : long.n_samples], y)
    for epoch in X_test[..., : long.n_samples]:
        np.testing.assert_allclose(
            long.classify(epoch)[1], short.classify(epoch)[1], rtol=1e-10
        )


def test_fit_rejects_epochs_shorter_than_online():
    X, y = calibration(n_blocks=2)
    decoder = online.Decoder("CCA", window=WINDOW)
    with pytest.raises(ValueError):
        decoder.fit(X[..., : decoder.n_samples - 1], y)


def test_epoch_waits_for_its_last_sample_then_cuts_like_load_recording():
    """The epoch starts PRE s before the first sample at or after the onset. Every
    sample holds 1000 * channel + its index, so the test sees exactly which ones."""
    decoder = online.Decoder("CCA", window=WINDOW)
    n = decoder.n_samples
    pre = int(round(online.PRE * FS))
    k = 1000  # index of the first sample after the onset
    last = k - pre + n - 1  # index of the epoch's last sample
    data = 1000.0 * np.arange(2) + np.arange(last + 50)[:, None]
    times = T0 + np.arange(last + 50) / FS
    onset = times[k] - 0.001

    buffer = acquisition.RingBuffer(2, 5000)
    buffer.append(data[:last], times[:last])
    assert decoder.epoch(buffer, onset) is None
    buffer.append(data[last:], times[last:])
    epoch = decoder.epoch(buffer, onset)
    assert epoch.shape == (2, n)
    np.testing.assert_array_equal(epoch, data[k - pre : k - pre + n].T)


# --- Backend -----------------------------------------------------------------------


def test_state_request_gets_the_current_state():
    backend = make_backend(phrases=["hello", "world"], cue=True)
    messages = backend.on_marker(json.dumps({"event": "state_request"}), T0)
    assert len(messages) == 1
    state = protocol.parse_state(messages[0])
    speller = backend.speller
    assert state["trial"] is None
    assert state["labels"] == speller.labels()
    assert (state["phrase"], state["text"]) == ("HELLO", "")
    assert state["target"] is not None
    assert state["target"] == speller.next_target()
    assert state["cue"] is True


def test_no_phrases_means_free_spelling():
    state = protocol.parse_state(make_backend().state())
    assert (state["phrase"], state["target"], state["cue"]) == ("", None, False)


def test_state_passes_extra_fields_on():
    message = make_backend().state(trial=3, selected=2, pause=True, message="Hi")
    state = protocol.parse_state(message)
    assert (state["trial"], state["selected"]) == (3, 2)
    assert (state["pause"], state["message"]) == (True, "Hi")


def test_a_trial_is_answered_once_its_epoch_has_arrived():
    """stim_off comes 1.2 s after the onset; the epoch ends 1.64 s after it."""
    backend = make_backend()
    session = Session(stim=1.2)
    onset = session.trial(3)
    feeder = Feeder(backend, session)
    epoch_end = onset + LATENCY + WINDOW + online.POST
    assert answers(feeder.run(until=epoch_end - 0.05)) == []
    got = answers(feeder.run())
    assert [(s["trial"], s["selected"]) for s in got] == [(0, 3)]
    assert backend.poll() == []


def test_no_answer_before_stim_off():
    """The epoch has arrived 1.64 s after the onset, stim_off comes after 2 s."""
    backend = make_backend()
    session = Session(stim=2.0)
    onset = session.trial(1)
    feeder = Feeder(backend, session)
    assert answers(feeder.run(until=onset + 1.9)) == []
    assert [s["selected"] for s in answers(feeder.run())] == [1]


def test_the_decision_comes_from_the_eeg_not_from_the_marker():
    backend = make_backend()
    session = Session()
    session.trial(2, marker_target=4)
    got = answers(Feeder(backend, session).run())
    assert [s["selected"] for s in got] == [2]
    assert backend.rows[0]["target"] == 4


def test_an_incomplete_trial_gets_no_answer():
    backend = make_backend()
    session = Session()
    session.trial(1, complete=False)
    session.trial(2)
    got = answers(Feeder(backend, session).run())
    assert [s["trial"] for s in got] == [1]
    assert len(backend.rows) == 1
    assert backend.speller.group == SLOTS[2]  # only trial 1 reached the speller


def test_a_stim_off_for_an_unknown_trial_is_ignored():
    """E.g. the backend was started after that trial's stim_on."""
    backend = make_backend()
    for complete in [True, False]:
        marker = json.dumps({"event": "stim_off", "trial": 9, "complete": complete})
        assert backend.on_marker(marker, T0) == []
    assert backend.pending == {}
    assert backend.poll() == []


def test_a_trial_whose_epoch_left_the_buffer_is_answered_without_a_decision():
    """The GUI waits for an answer to every complete trial, so a lost one gets one too,
    and the speller stays where it was."""
    backend = make_backend()
    backend.buffer = acquisition.RingBuffer(len(CAP), int(0.4 * FS))  # shorter than PRE
    session = Session()
    session.trial(1)
    got = answers(Feeder(backend, session).run())
    assert [(s["trial"], s["selected"]) for s in got] == [(0, None)]
    assert backend.speller.group is None
    assert backend.poll() == []
    assert len(backend.rows) == 1
    assert backend.rows[0]["selected"] is None
    assert np.isnan(backend.rows[0]["score_0"])


def test_the_answer_shows_the_speller_after_the_selection():
    backend = make_backend(phrases=["hi"])
    session = Session()
    session.trial(5)
    (answer,) = answers(Feeder(backend, session).run())
    speller = backend.speller
    assert speller.group == SLOTS[5]
    assert answer["labels"] == speller.labels()
    assert all(isinstance(label, str) for label in answer["labels"])  # level 2
    assert (answer["phrase"], answer["target"]) == ("HI", speller.next_target())
    assert answer["pause"] is False


def test_rows_log_every_answered_trial():
    backend = make_backend()
    session = Session()
    for target in [0, 4]:
        session.trial(target)
    Feeder(backend, session).run()
    rows = backend.rows
    assert [row["trial"] for row in rows] == [0, 1]
    assert [row["target"] for row in rows] == [0, 4]
    assert [row["selected"] for row in rows] == [0, 4]
    for row in rows:
        assert {"block", "item", "text", "classify_ms"} <= set(row)
        scores = [row[f"score_{k}"] for k in range(N_TARGETS)]
        assert np.argmax(scores) == row["selected"]


def copy_spell(backend, n_trials):
    """Closed loop, like the fake EEG stream: every trial flickers the target the
    speller expects next -> decoded states of all messages."""
    session = Session()
    feeder = Feeder(backend, session)
    messages = []
    for _ in range(n_trials):
        session.trial(backend.speller.next_target())
        messages += feeder.run()
    return [protocol.parse_state(m) for m in messages]


def test_copy_spelling_a_phrase_pauses_then_starts_the_next():
    backend = make_backend(phrases=["hi", "ok"])
    states = copy_spell(backend, 4)
    got = [s for s in states if s["trial"] is not None]
    assert [s["pause"] for s in got] == [False, False, False, True]
    assert (got[-1]["text"], got[-1]["phrase"]) == ("HI", "HI")
    following = states[states.index(got[-1]) + 1]
    assert following["trial"] is None
    assert (following["phrase"], following["text"]) == ("OK", "")
    assert following["target"] == backend.speller.next_target()


def test_after_the_last_phrase_comes_free_spelling():
    backend = make_backend(phrases=["a"])
    states = copy_spell(backend, 2)
    assert states[-2]["pause"] is True
    assert (states[-1]["trial"], states[-1]["phrase"]) == (None, "")


# --- replay ------------------------------------------------------------------------


def test_replay_answers_every_complete_trial_from_the_eeg(tmp_path):
    """A 43 s recording (longer than the ring buffer) with extra channels in between
    the CAP channels; trial 6 was cut short by ESC."""
    channels = ["Fz", CAP[3], "contact_O1", *[c for c in CAP if c != CAP[3]], "Accel_x"]
    session = Session(channels=channels, seed=5)
    targets = [0, 1, 2, 3, 4, 5, 5, 4, 3, 2, 1, 0]
    for trial, target in enumerate(targets):
        session.trial(target, complete=trial != 6)
    session.save(tmp_path / "rec.npz")

    backend = make_backend()
    got = answers(online.replay(tmp_path / "rec.npz", backend))
    complete = [trial for trial in range(len(targets)) if trial != 6]
    assert [s["trial"] for s in got] == complete
    assert [s["selected"] for s in got] == [targets[trial] for trial in complete]
    assert [row["selected"] for row in backend.rows] == [targets[t] for t in complete]


# --- the online loop and main ------------------------------------------------------


class FakeInlet:
    """Stands in for pylsl.StreamInlet: hands out prepared chunks, then empty ones."""

    def __init__(self, *chunks):
        self.chunks = list(chunks)

    def pull_chunk(self, timeout=0.0, max_samples=1024):
        return self.chunks.pop(0) if self.chunks else ([], [])


class FakeOutlet:
    """Stands in for pylsl.StreamOutlet: keeps what was pushed."""

    def __init__(self):
        self.pushed = []

    def push_sample(self, sample):
        self.pushed.append(sample)


def test_pump_feeds_the_cap_columns_and_markers_in_and_pushes_the_answers():
    """The columns that aren't CAP channels carry a strong sine at another target, so
    reading them would give the wrong decision."""
    channels = ["Fz", *CAP, "Accel_x"]
    cap = online.cap_indices(channels)
    session = Session(channels=channels, seed=5)
    session.trial(3)
    eeg = FakeInlet((session.eeg.tolist(), session.times.tolist()))
    markers = FakeInlet(
        ([[m] for _, m in session.markers], [t for t, _ in session.markers])
    )
    outlet = FakeOutlet()
    backend = make_backend()

    online.pump(backend, eeg, markers, outlet, cap)
    got = answers([sample[0] for sample in outlet.pushed])
    assert [(s["trial"], s["selected"]) for s in got] == [(0, 3)]

    n_pushed = len(outlet.pushed)
    online.pump(backend, eeg, markers, outlet, cap)  # nothing new has arrived
    assert len(outlet.pushed) == n_pushed


def write_calibration(path):
    session = Session(seed=1)
    for target in np.tile(np.arange(N_TARGETS), 2):
        session.trial(target, after=1.0)
    session.save(path)


def main_args(calibration, *more):
    return [
        "--subject",
        "S9",
        "--method",
        "CCA",
        "--window",
        str(WINDOW),
        "--calibration",
        str(calibration),
        *more,
    ]


def test_main_replay_prints_the_accuracy_and_saves_the_log(
    tmp_path, monkeypatch, capsys
):
    """The last trial's recorded target is wrong on purpose, so 3 of 4 are correct."""
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "logs")
    write_calibration(tmp_path / "calibration.npz")
    session = Session(seed=2)
    for target in [0, 1, 2]:
        session.trial(target)
    session.trial(3, marker_target=4)
    session.save(tmp_path / "recording.npz")

    online.main(
        main_args(
            tmp_path / "calibration.npz", "--replay", str(tmp_path / "recording.npz")
        )
    )
    assert "3/4" in capsys.readouterr().out
    (log,) = (tmp_path / "logs").glob("S9_*_backend.csv")
    table = pd.read_csv(log)
    assert table["trial"].tolist() == [0, 1, 2, 3]
    assert table["selected"].tolist() == [0, 1, 2, 3]
    assert table["target"].tolist() == [0, 1, 2, 4]


@pytest.mark.parametrize(
    "freqs", [[8.0, 10.0, 12.0, 14.0, 15.0, 9.0], config.TARGET_FREQS[::-1]]
)
def test_main_refuses_a_replay_recorded_with_other_frequencies(
    tmp_path, monkeypatch, freqs
):
    """Its targets would index other frequencies (also when the same frequencies are in
    another order), so the accuracy would mean nothing."""
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "logs")
    write_calibration(tmp_path / "calibration.npz")
    session = Session(freqs=freqs)
    session.trial(0)
    session.save(tmp_path / "recording.npz")
    with pytest.raises(ValueError):
        online.main(
            main_args(
                tmp_path / "calibration.npz",
                "--replay",
                str(tmp_path / "recording.npz"),
            )
        )


def test_main_saves_the_log_on_ctrl_c(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RECORDINGS_DIR", tmp_path / "logs")
    write_calibration(tmp_path / "calibration.npz")

    def interrupted(backend, eeg_type, eeg_name):
        backend.rows.append({"trial": 0, "selected": 1})
        raise KeyboardInterrupt

    monkeypatch.setattr(online, "serve", interrupted)
    online.main(main_args(tmp_path / "calibration.npz"))
    (log,) = (tmp_path / "logs").glob("S9_*_backend.csv")
    assert pd.read_csv(log)["selected"].tolist() == [1]
