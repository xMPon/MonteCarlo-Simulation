"""Financial Monte Carlo: GBM paths, European option pricing, variance reduction.

Prices are simulated under the risk-neutral measure, so the drift used for
pricing is the risk-free rate, not the historical drift. `calibrate_gbm`
recovers the *historical* drift and volatility from a price series; only the
volatility should be fed back into pricing.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

TRADING_DAYS = 252


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo estimate together with its standard error."""

    value: float
    stderr: float

    @property
    def ci95(self) -> tuple[float, float]:
        """Return the 95% confidence interval around the estimate."""
        half_width = 1.96 * self.stderr
        return self.value - half_width, self.value + half_width

    def __str__(self) -> str:
        low, high = self.ci95
        return f"{self.value:.4f} +/- {1.96 * self.stderr:.4f}  (95% CI {low:.4f} to {high:.4f})"


def calibrate_gbm(prices, trading_days: int = TRADING_DAYS) -> tuple[float, float]:
    """Estimate annualised drift and volatility from a series of prices.

    Args:
        prices: One-dimensional sequence of positive prices in time order.
        trading_days: Periods per year used to annualise (252 for daily data).

    Returns:
        Tuple of (mu, sigma), both annualised. mu is the arithmetic drift,
        converted from the log-return mean via mu = m + sigma ** 2 / 2.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be one-dimensional")
    if prices.size < 2:
        raise ValueError("need at least two prices to compute a return")
    if np.any(prices <= 0):
        raise ValueError("prices must be positive")

    log_returns = np.diff(np.log(prices))
    sigma = float(log_returns.std(ddof=1) * np.sqrt(trading_days))
    # Convert the mean log return into an arithmetic drift
    mu = float(log_returns.mean() * trading_days + 0.5 * sigma**2)
    return mu, sigma


def simulate_gbm_paths(
    spot: float,
    drift: float,
    volatility: float,
    years: float,
    steps: int,
    paths: int,
    seed: int | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion price paths.

    Returns:
        Array of shape (paths, steps + 1); column 0 is the spot for every path.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive integers")

    dt = years / steps
    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((paths, steps))
    # Exact GBM increment in log space, so the discretisation introduces no bias
    increments = (drift - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * shocks
    log_paths = np.concatenate([np.zeros((paths, 1)), np.cumsum(increments, axis=1)], axis=1)
    return spot * np.exp(log_paths)


def simulate_terminal_prices(
    spot: float,
    drift: float,
    volatility: float,
    years: float,
    paths: int,
    seed: int | None = None,
    antithetic: bool = False,
) -> np.ndarray:
    """Sample terminal GBM prices directly, skipping the intermediate path.

    With antithetic=True each normal draw Z is paired with -Z, and paths is
    rounded up to the next even number so the pairing is exact.
    """
    if spot <= 0:
        raise ValueError("spot must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    if paths <= 0:
        raise ValueError("paths must be a positive integer")

    rng = np.random.default_rng(seed)
    if antithetic:
        half = (paths + 1) // 2
        base = rng.standard_normal(half)
        shocks = np.concatenate([base, -base])
    else:
        shocks = rng.standard_normal(paths)

    return spot * np.exp(
        (drift - 0.5 * volatility**2) * years + volatility * np.sqrt(years) * shocks
    )


def _payoff(terminal: np.ndarray, strike: float, kind: str) -> np.ndarray:
    """Return the European payoff for each terminal price."""
    if kind == "call":
        return np.maximum(terminal - strike, 0.0)
    if kind == "put":
        return np.maximum(strike - terminal, 0.0)
    raise ValueError("kind must be 'call' or 'put'")


def price_european(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    years: float,
    paths: int,
    kind: str = "call",
    seed: int | None = None,
    antithetic: bool = False,
    control_variate: bool = False,
) -> Estimate:
    """Price a European option by Monte Carlo, with its standard error.

    Args:
        spot: Current underlying price.
        strike: Option strike.
        rate: Continuously compounded risk-free rate, also the risk-neutral drift.
        volatility: Annualised volatility.
        years: Time to expiry in years.
        paths: Number of simulated paths.
        kind: Either "call" or "put".
        seed: Optional random seed for reproducibility.
        antithetic: Pair each draw with its negative to cut variance.
        control_variate: Use the discounted terminal price as a control, whose
            expectation under the risk-neutral measure is exactly the spot.
    """
    if strike <= 0:
        raise ValueError("strike must be positive")

    terminal = simulate_terminal_prices(
        spot, rate, volatility, years, paths, seed=seed, antithetic=antithetic
    )
    discount = np.exp(-rate * years)
    values = discount * _payoff(terminal, strike, kind)

    if control_variate:
        # E[discounted terminal price] = spot exactly, so this control adds no bias
        control = discount * terminal
        control_var = control.var(ddof=1)
        if control_var > 0:
            # Optimal coefficient estimated from the same sample (standard practice)
            beta = -np.cov(values, control, ddof=1)[0, 1] / control_var
            values = values + beta * (control - spot)

    if antithetic:
        # Antithetic draws are negatively correlated in pairs, so a naive
        # standard error over all draws would be wrong. Average each pair,
        # then treat the pair means as the independent sample.
        half = values.size // 2
        values = 0.5 * (values[:half] + values[half:])

    return Estimate(
        value=float(values.mean()),
        stderr=float(values.std(ddof=1) / np.sqrt(values.size)),
    )


def black_scholes(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    years: float,
    kind: str = "call",
) -> float:
    """Closed-form Black-Scholes price, the exact benchmark for the simulation."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    if years <= 0 or volatility <= 0:
        # Degenerate case: the option is worth its discounted intrinsic value
        forward = spot if years <= 0 else spot * np.exp(rate * years)
        intrinsic = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        return float(np.exp(-rate * max(years, 0.0)) * intrinsic)

    sqrt_t = volatility * np.sqrt(years)
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * years) / sqrt_t
    d2 = d1 - sqrt_t
    if kind == "call":
        return float(spot * norm.cdf(d1) - strike * np.exp(-rate * years) * norm.cdf(d2))
    return float(strike * np.exp(-rate * years) * norm.cdf(-d2) - spot * norm.cdf(-d1))
