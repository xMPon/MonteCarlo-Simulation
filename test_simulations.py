"""Tests for Monte Carlo simulation behavior and reproducibility."""

from __future__ import annotations
from simulations import estimate_pi


def test_estimate_pi_is_reproducible_with_seed() -> None:
    # The same seed and sample size should always produce the same result
    first = estimate_pi(samples=50_000, seed=42)
    second = estimate_pi(samples=50_000, seed=42)
    assert first == second


def test_estimate_pi_is_reasonably_accurate() -> None:
    # With enough samples, the estimate should be close to the true value of pi
    pi_hat = estimate_pi(samples=200_000, seed=123)
    assert abs(pi_hat - 3.141592653589793) < 0.02


def test_estimate_pi_raises_for_non_positive_samples() -> None:
    # The function should raise a ValueError for non-positive sample sizes
    try:
        estimate_pi(samples=0)
    except ValueError as exc:
        assert "samples must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive samples")
