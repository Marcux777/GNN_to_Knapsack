"""
Tests for custom OOD sampling utilities.
"""

import numpy as np

from experiments.pipelines.evaluate_ood_pipeline import sample_from_distribution


def test_uniform_distribution_respects_bounds():
    rng = np.random.default_rng(0)
    samples = sample_from_distribution(rng, "uniform", size=256, low=5, high=25)
    assert samples.min() >= 5
    assert samples.max() <= 25
    assert samples.dtype == np.float32


def test_power_distribution_is_heavy_tailed():
    rng = np.random.default_rng(123)
    samples = sample_from_distribution(rng, "power", size=512, low=1, high=100, alpha=2.0)
    assert samples.min() >= 1
    assert samples.max() <= 100
    assert np.median(samples) < 5  # heavy tail concentrates near the minimum
