"""Tests for scripts/fake_eeg_stream.py."""

import importlib.util
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pylsl
import pytest
from scipy.signal import welch

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fake_eeg_stream.py"
SPEC = importlib.util.spec_from_file_location("fake_eeg_stream", MODULE_PATH)
fake = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fake)

FS = 250.0


def times(start, duration):
    """LSL times of the samples in [start, start + duration)."""
    return start + np.arange(int(round(duration * FS))) / FS


def marker(event, **fields):
    return json.dumps({"event": event, **fields})


def peak_frequency(x, f_max):
    """Frequency (Hz) of the largest spectral peak between 5 Hz and ``f_max``, per
    channel (``f_max`` below the 2nd harmonic, so harmonic strength doesn't matter).
    x : (n, n_channels)."""
    f, p = welch(x, fs=FS, nperseg=len(x), nfft=8 * len(x), axis=0)
    band = (f >= 5) & (f <= f_max)
    return f[band][np.argmax(p[band], axis=0)]


def band_power(x, freq):
    """Mean power per channel within 0.5 Hz of ``freq``. x : (n, n_channels)."""
    f, p = welch(x, fs=FS, nperseg=len(x), nfft=8 * len(x), axis=0)
    return p[np.abs(f - freq) <= 0.5].mean(axis=0)


# One-sided power spectral density of white noise with standard deviation NOISE_STD.
NOISE_PSD = 2 * fake.NOISE_STD**2 / FS


def test_generate_shape_and_noise_level():
    """(n, n_channels) samples; with no flicker it is noise of about NOISE_STD µV."""
    x = fake.FakeEEG(n_channels=4, fs=FS, seed=1).generate(times(0.0, 4.0))
    assert x.shape == (1000, 4)
    assert np.allclose(x.std(axis=0), fake.NOISE_STD, rtol=0.15)


def test_same_seed_same_output():
    a, b = (fake.FakeEEG(3, FS, seed=7) for _ in range(2))
    for g in (a, b):
        g.on_marker(marker("stim_on", trial=0, block=0, target=2, freq=12.0), 1.0)
    assert np.array_equal(a.generate(times(0.0, 2.0)), b.generate(times(0.0, 2.0)))


def test_ssvep_at_the_cued_frequency_only_during_the_trial():
    """stim_on at 10 s, stim_off at 15 s (latency 0.14 s): a 12 Hz peak on every channel
    inside the trial; before and after it, 12 Hz holds only white-noise power."""
    g = fake.FakeEEG(n_channels=3, fs=FS, snr_db=10.0, latency=0.14, seed=2)
    before = g.generate(times(6.0, 4.0))
    g.on_marker(marker("stim_on", trial=0, block=0, target=2, freq=12.0), 10.0)
    g.on_marker(marker("stim_off", trial=0, block=0, target=2, freq=12.0), 15.0)
    during = g.generate(times(10.0, 6.0))[int(0.14 * FS) : int(5.14 * FS)]
    after = g.generate(times(16.0, 4.0))
    assert np.allclose(peak_frequency(during, f_max=18.0), 12.0, atol=0.25)
    assert np.all(band_power(during, 12.0) > 50 * NOISE_PSD)
    assert np.all(band_power(before, 12.0) < 10 * NOISE_PSD)
    assert np.all(band_power(after, 12.0) < 10 * NOISE_PSD)


def test_response_starts_latency_after_stim_on_and_ends_latency_after_stim_off():
    """latency 0.5 s: the half second after stim_on (and after the end of the response)
    is noise, the half second after that carries the SSVEP."""
    g = fake.FakeEEG(n_channels=2, fs=FS, snr_db=20.0, latency=0.5, seed=3)
    g.on_marker(marker("stim_on", trial=0, block=0, target=0, freq=8.0), 20.0)
    g.on_marker(marker("stim_off", trial=0, block=0, target=0, freq=8.0), 22.0)
    x = g.generate(times(20.0, 3.0))
    power = (x**2).mean(axis=1)
    noise = fake.NOISE_STD**2
    assert power[: int(0.5 * FS) - 2].mean() < 2 * noise
    assert power[int(0.5 * FS) + 2 : int(1.0 * FS)].mean() > 20 * noise
    assert power[int(2.5 * FS) + 2 :].mean() < 2 * noise


def test_uses_the_frequency_from_the_marker():
    """The marker's ``freq`` decides, also for a frequency outside config.TARGET_FREQS."""
    g = fake.FakeEEG(n_channels=2, fs=FS, snr_db=10.0, latency=0.0, seed=4)
    g.on_marker(marker("stim_on", trial=0, block=0, target=3, freq=9.3), 0.0)
    x = g.generate(times(0.0, 4.0))
    assert np.allclose(peak_frequency(x, f_max=14.0), 9.3, atol=0.15)


def test_response_is_time_locked_to_stim_on():
    """Two trials of one target at onsets 10 s and 23.37 s: the responses relative to
    each onset are the same waveform (what TRCA templates rely on)."""
    g = fake.FakeEEG(n_channels=2, fs=FS, snr_db=30.0, latency=0.14, seed=5)
    segments = []
    for onset in (10.0, 23.37):
        g.on_marker(marker("stim_on", trial=0, block=0, target=1, freq=10.0), onset)
        segments.append(g.generate(onset + 0.14 + np.arange(250) / FS))
        g.on_marker(
            marker("stim_off", trial=0, block=0, target=1, freq=10.0), onset + 2
        )
    for channel in range(2):
        r = np.corrcoef(segments[0][:, channel], segments[1][:, channel])[0, 1]
        assert r > 0.99


def test_session_start_phases_shift_the_response():
    """Target 1 with phase pi (from session_start) gives the inverted waveform of
    phase 0 (one harmonic, so the sign flip is exact)."""
    responses = []
    for phase in (0.0, np.pi):
        g = fake.FakeEEG(
            n_channels=2, fs=FS, snr_db=30.0, latency=0.0, n_harmonics=1, seed=6
        )
        g.on_marker(marker("session_start", phases=[0.0, phase, 0.0]), 0.0)
        g.on_marker(marker("stim_on", trial=0, block=0, target=1, freq=10.0), 5.0)
        responses.append(g.generate(times(5.0, 1.0)))
    for channel in range(2):
        assert (
            np.corrcoef(responses[0][:, channel], responses[1][:, channel])[0, 1]
            < -0.99
        )


def test_other_markers_are_ignored():
    """cue, block_start, ... do not start a response."""
    g = fake.FakeEEG(n_channels=2, fs=FS, snr_db=10.0, latency=0.0, seed=8)
    quiet = fake.FakeEEG(n_channels=2, fs=FS, snr_db=10.0, latency=0.0, seed=8)
    g.on_marker(marker("block_start", block=0), 0.0)
    g.on_marker(marker("cue", trial=0, block=0, target=2, freq=12.0), 0.0)
    assert np.array_equal(g.generate(times(0.0, 2.0)), quiet.generate(times(0.0, 2.0)))


def test_eeg_outlet_describes_the_stream():
    """Type EEG, regular rate fs, float32, one channel per label, labels in the XML."""
    labels = ["O1", "Oz", "O2"]
    outlet = fake.open_eeg_outlet(labels, FS, name="pytest-fake")
    info = outlet.get_info()
    assert info.type() == "EEG"
    assert info.nominal_srate() == pytest.approx(FS)
    assert info.channel_count() == 3
    assert info.channel_format() == pylsl.cf_float32
    root = ElementTree.fromstring(info.as_xml())
    assert [e.text for e in root.iter("label")] == labels
