"""Tests for Monte Carlo simulation behavior and reproducibility."""

from __future__ import annotations

from simulations import estimate_pi


def test_estimate_pi_is_reproducible_with_seed() -> None:
    first = estimate_pi(samples=50_000, seed=42)
    second = estimate_pi(samples=50_000, seed=42)
    assert first == second


def test_estimate_pi_is_reasonably_accurate() -> None:
    pi_hat = estimate_pi(samples=200_000, seed=123)
    assert abs(pi_hat - 3.141592653589793) < 0.02


def test_estimate_pi_raises_for_non_positive_samples() -> None:
    try:
        estimate_pi(samples=0)
    except ValueError as exc:
        assert "samples must be a positive integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-positive samples")
