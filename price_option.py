"""CLI for pricing a European option by Monte Carlo and checking it against Black-Scholes."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from finance import Estimate, black_scholes, calibrate_gbm, price_european


def load_close_prices(csv_path: Path) -> np.ndarray:
    """Read a close-price series from a CSV written by download_index_data.py.

    Handles both layouts: the multi-row header yfinance emits, and the single
    Close column that the FRED fallback produces.
    """
    frame = pd.read_csv(csv_path, index_col=0)
    # yfinance writes two extra header rows (ticker, then a blank); drop non-numeric rows
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError(f"no usable Close prices found in {csv_path}")
    return close.to_numpy(dtype=float)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Price a European option by Monte Carlo and compare to Black-Scholes",
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Current underlying price")
    parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Option strike. With --calibrate this is read as a percentage of spot, "
             "so 100 is at-the-money and 110 is 10%% out of the money for a call.",
    )
    parser.add_argument(
        "--rate", type=float, default=0.03, help="Risk-free rate, e.g. 0.03 for 3%%"
    )
    parser.add_argument("--vol", type=float, default=0.20, help="Annualised volatility")
    parser.add_argument("--years", type=float, default=1.0, help="Time to expiry in years")
    parser.add_argument("--paths", type=int, default=200_000, help="Number of simulated paths")
    parser.add_argument("--kind", choices=["call", "put"], default="call", help="Option type")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs")
    parser.add_argument(
        "--calibrate",
        type=Path,
        metavar="CSV",
        help="Calibrate spot and volatility from a price CSV, e.g. data/gspc.csv",
    )
    return parser.parse_args()


def main() -> int:
    """Run the pricing comparison. Returns a process exit code."""
    args = parse_args()
    spot, vol = args.spot, args.vol

    if args.calibrate:
        if not args.calibrate.exists():
            print(f"{args.calibrate} not found. Run: python download_index_data.py")
            return 1
        prices = load_close_prices(args.calibrate)
        mu, vol = calibrate_gbm(prices)
        spot = float(prices[-1])
        print(f"Calibrated from {args.calibrate} ({len(prices)} observations)")
        print(f"  last close      {spot:,.2f}")
        print(f"  historical drift {mu:.2%}  (not used for pricing)")
        print(f"  volatility       {vol:.2%}")
        print()

    # Absolute strike normally; a percentage of spot when calibrating, since the
    # index level is not known until the data has been read
    strike = args.strike / 100.0 * spot if args.calibrate else args.strike
    exact = black_scholes(spot, strike, args.rate, vol, args.years, kind=args.kind)

    print(f"{args.kind.capitalize()}: spot {spot:,.2f}, strike {strike:,.2f}, "
          f"rate {args.rate:.2%}, vol {vol:.2%}, {args.years}y, {args.paths:,} paths")
    print(f"Black-Scholes (exact): {exact:.4f}")
    print()

    variants: list[tuple[str, dict]] = [
        ("plain", {}),
        ("antithetic", {"antithetic": True}),
        ("control variate", {"control_variate": True}),
        ("both", {"antithetic": True, "control_variate": True}),
    ]

    header = f"{'method':<18}{'price':>12}{'std error':>12}{'error vs BS':>14}{'SE reduction':>14}"
    print(header)
    print("-" * len(header))

    baseline: Estimate | None = None
    results: dict[str, Estimate] = {}
    for label, options in variants:
        est = price_european(
            spot, strike, args.rate, vol, args.years,
            paths=args.paths, kind=args.kind, seed=args.seed, **options,
        )
        results[label] = est
        if baseline is None:
            baseline = est
        speedup = f"{baseline.stderr / est.stderr:>12.1f}x" if est.stderr > 0 else f"{'n/a':>13}"
        print(f"{label:<18}{est.value:>12.4f}{est.stderr:>12.4f}"
              f"{est.value - exact:>+14.4f}{speedup:>14}")

    print()
    print("SE reduction is relative to the plain estimator at the same path count.")
    print("A correct estimator's error vs BS should sit within about 2 standard errors.")

    if results["both"].stderr > results["control variate"].stderr:
        print()
        print("Note: combining the two is worse here than the control variate alone.")
        print("That is expected, not a bug. The control already absorbs the exposure")
        print("to the terminal price that antithetic sampling exploits, which flips")
        print("the antithetic pairs from negatively to positively correlated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
