"""CLI for path-dependent options, where Monte Carlo is the only practical method."""

from __future__ import annotations
import argparse
from exotics import (
    barrier_closed_form,
    continuity_corrected_barrier,
    geometric_asian_closed_form,
    price_asian,
    price_barrier,
)
from finance import black_scholes


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Price Asian and barrier options by Monte Carlo",
    )
    parser.add_argument(
        "--product", choices=["asian", "barrier"], default="asian", help="Which payoff to price"
    )
    parser.add_argument("--spot", type=float, default=100.0, help="Current underlying price")
    parser.add_argument("--strike", type=float, default=100.0, help="Option strike")
    parser.add_argument("--rate", type=float, default=0.05, help="Risk-free rate")
    parser.add_argument("--vol", type=float, default=0.25, help="Annualised volatility")
    parser.add_argument("--years", type=float, default=1.0, help="Time to expiry in years")
    parser.add_argument("--paths", type=int, default=200_000, help="Number of simulated paths")
    parser.add_argument("--kind", choices=["call", "put"], default="call", help="Option type")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Monitoring dates (default 12 for Asian, 252 for barrier)",
    )
    parser.add_argument(
        "--barrier", type=float, default=90.0, help="Barrier level (barrier product only)"
    )
    parser.add_argument(
        "--knock", choices=["in", "out"], default="out", help="Barrier knock direction"
    )
    parser.add_argument(
        "--side", choices=["down", "up"], default="down", help="Barrier side relative to spot"
    )
    return parser.parse_args()


def run_asian(args: argparse.Namespace) -> None:
    """Price an Asian option and show what the geometric control variate buys."""
    steps = args.steps or 12
    common = (args.spot, args.strike, args.rate, args.vol, args.years)

    exact_geometric = geometric_asian_closed_form(*common, steps=steps, kind=args.kind)
    vanilla = black_scholes(*common, kind=args.kind)

    print(f"Asian {args.kind}: spot {args.spot:,.2f}, strike {args.strike:,.2f}, "
          f"vol {args.vol:.2%}, {args.years}y, {steps} averaging dates, {args.paths:,} paths")
    print(f"Vanilla {args.kind} for reference:   {vanilla:.4f}")
    print(f"Geometric Asian (exact):          {exact_geometric:.4f}")
    print()

    header = f"{'estimator':<34}{'price':>11}{'std error':>12}{'SE reduction':>14}"
    print(header)
    print("-" * len(header))

    geometric = price_asian(*common, steps=steps, paths=args.paths, kind=args.kind,
                            averaging="geometric", seed=args.seed)
    gap = abs(geometric.value - exact_geometric) / geometric.stderr
    print(f"{'geometric (has a closed form)':<34}{geometric.value:>11.4f}"
          f"{geometric.stderr:>12.4f}{'n/a':>14}")

    plain = price_asian(*common, steps=steps, paths=args.paths, kind=args.kind, seed=args.seed)
    print(f"{'arithmetic, plain':<34}{plain.value:>11.4f}{plain.stderr:>12.4f}{1.0:>13.0f}x")

    controlled = price_asian(*common, steps=steps, paths=args.paths, kind=args.kind,
                             seed=args.seed, control_variate=True)
    ratio = plain.stderr / controlled.stderr
    print(f"{'arithmetic, geometric control':<34}{controlled.value:>11.4f}"
          f"{controlled.stderr:>12.4f}{ratio:>13.0f}x")

    print()
    print(f"The geometric leg sits {gap:.1f} standard errors from its closed form, which is")
    print("what validates the simulation. The arithmetic option has no closed form, so")
    print("the geometric price is reused as a control variate: the two averages are")
    print(f"nearly perfectly correlated, cutting the standard error by {ratio:.0f}x.")


def run_barrier(args: argparse.Namespace) -> None:
    """Price a barrier option and show the discrete-monitoring bias converging."""
    steps = args.steps or 252
    common = (args.spot, args.strike, args.barrier, args.rate, args.vol, args.years)
    label = f"{args.side}-and-{args.knock} {args.kind}"
    options = dict(kind=args.kind, knock=args.knock, side=args.side)

    continuous = barrier_closed_form(*common, **options)
    print(f"{label}: spot {args.spot:,.2f}, strike {args.strike:,.2f}, "
          f"barrier {args.barrier:,.2f}, vol {args.vol:.2%}, {args.years}y, {args.paths:,} paths")
    print(f"Continuously monitored (closed form): {continuous:.4f}")
    print()

    header = (f"{'monitoring dates':<18}{'MC (discrete)':>15}{'BGK-corrected':>15}"
              f"{'gap':>10}{'raw bias':>11}")
    print(header)
    print("-" * len(header))

    grid = sorted({25, 100, steps})
    for count in grid:
        est = price_barrier(*common, steps=count, paths=args.paths, seed=args.seed, **options)
        shifted = continuity_corrected_barrier(args.barrier, args.vol, args.years, count, args.side)
        corrected = barrier_closed_form(
            args.spot, args.strike, shifted, args.rate, args.vol, args.years, **options
        )
        gap = (est.value - corrected) / est.stderr
        print(f"{count:<18}{est.value:>15.4f}{corrected:>15.4f}"
              f"{gap:>8.1f}SE{est.value - continuous:>+11.4f}")

    print()
    print("A discretely monitored barrier is missed between observation dates, so a")
    print("knock-out is worth more than the continuous formula says. That raw bias")
    print("shrinks as O(1/sqrt(dates)). The Broadie-Glasserman-Kou correction shifts")
    print("the barrier away from spot to account for it, and the corrected column")
    print("should agree with the simulation to within a couple of standard errors.")


def main() -> int:
    """Dispatch to the requested product. Returns a process exit code."""
    args = parse_args()
    if args.vol <= 0 or args.years <= 0:
        print("vol and years must be positive")
        return 1
    if args.product == "asian":
        run_asian(args)
    else:
        if args.side == "down" and args.barrier >= args.spot:
            print(f"a down barrier must sit below spot ({args.barrier} >= {args.spot})")
            return 1
        if args.side == "up" and args.barrier <= args.spot:
            print(f"an up barrier must sit above spot ({args.barrier} <= {args.spot})")
            return 1
        run_barrier(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
