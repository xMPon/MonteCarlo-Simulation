"""Core Monte Carlo simulation functions for the project."""

from __future__ import annotations
import numpy as np


def estimate_pi(samples: int, seed: int | None = None) -> float:
    """Estimate pi using random points sampled in a unit square.

    Args:
        samples: Number of random points to generate. Must be positive.
        seed: Optional random seed for reproducibility.

    Returns:
        Estimated value of pi.
    """
    if samples <= 0:
        raise ValueError("samples must be a positive integer")

    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate random (x, y) points in the unit square
    x = rng.random(samples)
    y = rng.random(samples)
    # Count how many points fall inside the quarter circle
    inside_quarter_circle = (x * x + y * y) <= 1.0

    # Estimate pi as 4 times the ratio inside the quarter circle
    return 4.0 * float(np.mean(inside_quarter_circle))
