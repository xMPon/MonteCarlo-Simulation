"""CLI entry point for running sample Monte Carlo simulations."""

from __future__ import annotations

import argparse

from src.simulations import estimate_pi


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Monte Carlo sample simulations")
    parser.add_argument("--samples", type=int, default=100_000, help="Number of simulation samples")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    return parser.parse_args()


def main() -> None:
    """Execute the sample Monte Carlo simulation."""
    args = parse_args()
    pi_hat = estimate_pi(samples=args.samples, seed=args.seed)
    print(f"Estimated pi with {args.samples} samples: {pi_hat:.6f}")


if __name__ == "__main__":
    main()
