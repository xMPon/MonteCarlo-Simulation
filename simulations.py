"""Core Monte Carlo simulation functions for the project."""

from __future__ import annotations
import numpy as np


def sample_unit_square(
    samples: int, seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Draw random points in the unit square and flag those inside the quarter circle.

    Args:
        samples: Number of random points to generate. Must be positive.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of (x, y, inside) where inside is a boolean mask of the points
        satisfying x^2 + y^2 <= 1.
    """
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate random (x, y) points in the unit square
    x = rng.random(samples)
    y = rng.random(samples)

    return x, y, (x * x + y * y) <= 1.0


def estimate_pi_from_mask(inside: np.ndarray) -> float:
    """Convert a quarter-circle hit mask into an estimate of pi."""
    # The quarter circle covers pi/4 of the unit square, so scale the hit rate by 4
    return 4.0 * float(np.mean(inside))


def estimate_pi(samples: int, seed: int | None = None) -> float:
    """Estimate pi using random points sampled in a unit square.

    Args:
        samples: Number of random points to generate. Must be positive.
        seed: Optional random seed for reproducibility.

    Returns:
        Estimated value of pi.
    """
    _, _, inside = sample_unit_square(samples, seed)
    return estimate_pi_from_mask(inside)
