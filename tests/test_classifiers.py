"""Tests for ssvep.cca and ssvep.trca."""

import pytest


@pytest.mark.skip(reason="TODO")
def test_reference_signals_shape():
    """reference_signals(10, 250, 500, n_harmonics=5).shape == (10, 500)."""


@pytest.mark.skip(reason="TODO")
def test_canonical_correlation_identical_signals_is_one():
    """X = Y (any full-rank signal) -> correlation ~1.0."""


@pytest.mark.skip(reason="TODO")
def test_canonical_correlation_matches_sklearn():
    """Random X, Y: your value == corrcoef of sklearn CCA(n_components=1).fit_transform(X.T, Y.T)."""


@pytest.mark.skip(reason="TODO")
def test_cca_easy_data(synthetic_easy):
    """Accuracy > 0.95 at a 1 s window; decision_function shape == (n_trials, n_targets)."""


@pytest.mark.skip(reason="TODO")
def test_cca_noise_only_is_near_chance(synthetic_noise_only):
    """Accuracy well below 0.6 (chance is 1/5)."""


@pytest.mark.skip(reason="TODO")
def test_trca_and_etrca_easy_data(synthetic_easy):
    """Leave-one-block-out accuracy > 0.95 for TRCA(ensemble=False) and TRCA(ensemble=True)."""
