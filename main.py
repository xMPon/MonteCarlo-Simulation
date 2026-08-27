"""CLI entry point for running sample Monte Carlo simulations."""

from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
from simulations import estimate_pi_from_mask, sample_unit_square


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo sample simulations")
    parser.add_argument(
        "--samples", type=int, default=100_000, help="Number of simulation samples"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducible runs"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Print the estimate without opening a plot"
    )
    return parser.parse_args()


def main() -> None:
    """Execute the sample Monte Carlo simulation and show a plot."""
    args = parse_args()
    # Draw the sample points and estimate pi from the hit mask
    x, y, inside = sample_unit_square(args.samples, args.seed)
    pi_hat = estimate_pi_from_mask(inside)
    print(f"Estimated pi with {args.samples} samples: {pi_hat:.6f}")

    if args.no_plot:
        return

    # Plot the points and the quarter circle
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        x[~inside], y[~inside], color="red", s=1, alpha=0.5, label="Outside quarter circle"
    )
    ax.scatter(
        x[inside], y[inside], color="blue", s=1, alpha=0.5, label="Inside quarter circle"
    )
    circle = plt.Circle(
        (0, 0), 1, color="black", fill=False, linewidth=2, linestyle="--", label="Quarter circle"
    )
    ax.add_patch(circle)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Monte Carlo Pi Estimation\nEstimated pi = {pi_hat:.6f}")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Run the main simulation and plotting routine
    main()
