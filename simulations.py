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

    rng = np.random.default_rng(seed)

    # Generate points and count how many fall in the quarter circle.
    x = rng.random(samples)
    y = rng.random(samples)
    inside_quarter_circle = (x * x + y * y) <= 1.0

    return 4.0 * float(np.mean(inside_quarter_circle))
