"""CLI entry point for running sample Monte Carlo simulations."""

from __future__ import annotations
import argparse
from simulations import estimate_pi
import numpy as np
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo sample simulations")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of simulation samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    return parser.parse_args()


def main() -> None:
    """Execute the sample Monte Carlo simulation and show a plot."""
    args = parse_args()
    # Set up random number generator
    rng = np.random.default_rng(args.seed)
    # Generate random (x, y) points in the unit square
    x = rng.random(args.samples)
    y = rng.random(args.samples)
    # Check which points fall inside the quarter circle
    inside = (x * x + y * y) <= 1.0
    # Estimate pi using the ratio of points inside the quarter circle
    pi_hat = 4.0 * float(np.mean(inside))
    print(f"Estimated pi with {args.samples} samples: {pi_hat:.6f}")

    # Plot the points and the quarter circle
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x[~inside], y[~inside], color="red", s=1, label="Outside quarter circle", alpha=0.5)
    ax.scatter(x[inside], y[inside], color="blue", s=1, label="Inside quarter circle", alpha=0.5)
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linewidth=2, linestyle='--', label='Quarter circle')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Monte Carlo Pi Estimation\nEstimated   {pi_hat:.6f}')
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the main simulation and plotting routine
    main()