"""Path-dependent option pricing: Asian and barrier payoffs.

These are the payoffs Monte Carlo actually earns its keep on. An arithmetic
Asian option has no closed form, and a discretely monitored barrier option has
no exact one either. Both are still checkable:

- The *geometric* Asian option does have an exact discrete closed form, which
  serves both as a validation oracle and as a control variate for the
  arithmetic version.
- The continuously monitored barrier option has the Reiner-Rubinstein closed
  form. Discrete monitoring differs from it by a known O(1/sqrt(steps)) bias,
  which the Broadie-Glasserman-Kou barrier shift corrects for.

Paths are generated in chunks so that memory stays bounded: a fine barrier
monitoring grid over many paths would otherwise be tens of gigabytes.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from finance import Estimate, black_scholes

# Broadie-Glasserman-Kou barrier shift constant, -zeta(1/2)/sqrt(2*pi)
BGK_BETA = 0.5826
DEFAULT_CHUNK = 50_000


@dataclass
class _Accumulator:
    """Streaming mean/variance for one or two correlated payoff series."""

    n: int = 0
    sum_a: float = 0.0
    sum_aa: float = 0.0
    sum_g: float = 0.0
    sum_gg: float = 0.0
    sum_ag: float = 0.0

    def update(self, a: np.ndarray, g: np.ndarray | None = None) -> None:
        """Fold one chunk of payoffs into the running totals."""
        self.n += a.size
        self.sum_a += float(a.sum())
        self.sum_aa += float((a * a).sum())
        if g is not None:
            self.sum_g += float(g.sum())
            self.sum_gg += float((g * g).sum())
            self.sum_ag += float((a * g).sum())

    def _moments(self) -> tuple[float, float, float, float, float]:
        """Return (mean_a, var_a, mean_g, var_g, cov_ag) with Bessel correction."""
        n = self.n
        mean_a = self.sum_a / n
        mean_g = self.sum_g / n
        var_a = (self.sum_aa - n * mean_a**2) / (n - 1)
        var_g = (self.sum_gg - n * mean_g**2) / (n - 1)
        cov = (self.sum_ag - n * mean_a * mean_g) / (n - 1)
        return mean_a, var_a, mean_g, var_g, cov

    def estimate(self) -> Estimate:
        """Plain estimate, ignoring any control series."""
        mean_a, var_a, _, _, _ = self._moments()
        return Estimate(mean_a, float(np.sqrt(max(var_a, 0.0) / self.n)))

    def controlled_estimate(self, control_mean: float) -> Estimate:
        """Estimate adjusted by the control series, whose exact mean is known."""
        mean_a, var_a, mean_g, var_g, cov = self._moments()
        if var_g <= 0:
            return self.estimate()
        beta = -cov / var_g
        value = mean_a + beta * (mean_g - control_mean)
        # Var(A + beta*(G - E[G])) collapses to var_a - cov^2 / var_g
        var_adj = max(var_a - cov**2 / var_g, 0.0)
        return Estimate(float(value), float(np.sqrt(var_adj / self.n)))


def _validate(spot: float, strike: float, volatility: float, years: float,
              steps: int, paths: int) -> None:
    """Shared argument validation for the path-dependent pricers."""
    if spot <= 0 or strike <= 0:
        raise ValueError("spot and strike must be positive")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")
    if years <= 0:
        raise ValueError("years must be positive")
    if steps <= 0 or paths <= 0:
        raise ValueError("steps and paths must be positive integers")


def _path_chunks(spot: float, drift: float, volatility: float, years: float,
                 steps: int, paths: int, seed: int | None, chunk: int):
    """Yield GBM path chunks of shape (rows, steps), excluding the t=0 column.

    NumPy fills normal draws row-major from the generator stream, so the
    concatenated chunks hold exactly the same draws as one big call. Prices
    therefore agree across chunk sizes to floating-point rounding (the running
    sums are added in a different order), not bit-for-bit.
    """
    dt = years / steps
    rng = np.random.default_rng(seed)
    log_drift = (drift - 0.5 * volatility**2) * dt
    diffusion = volatility * np.sqrt(dt)

    remaining = paths
    while remaining > 0:
        rows = min(chunk, remaining)
        shocks = rng.standard_normal((rows, steps))
        log_paths = np.cumsum(log_drift + diffusion * shocks, axis=1)
        yield spot * np.exp(log_paths)
        remaining -= rows


def geometric_asian_closed_form(
    spot: float, strike: float, rate: float, volatility: float,
    years: float, steps: int, kind: str = "call",
) -> float:
    """Exact price of a geometric-average Asian option monitored at `steps` dates.

    The log of a geometric average of lognormals is itself normal, which is
    what makes this exactly solvable where the arithmetic average is not.
    """
    _validate(spot, strike, volatility, years, steps, 1)
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")

    m = steps
    # Mean and variance of log G over m equally spaced monitoring dates
    mean_log = np.log(spot) + (rate - 0.5 * volatility**2) * years * (m + 1) / (2 * m)
    var_log = volatility**2 * years * (m + 1) * (2 * m + 1) / (6 * m**2)
    sd_log = np.sqrt(var_log)

    if sd_log <= 0:
        forward = np.exp(mean_log)
        intrinsic = max(forward - strike, 0.0) if kind == "call" else max(strike - forward, 0.0)
        return float(np.exp(-rate * years) * intrinsic)

    d1 = (mean_log - np.log(strike)) / sd_log + sd_log
    d2 = d1 - sd_log
    expected = np.exp(mean_log + 0.5 * var_log)
    discount = np.exp(-rate * years)
    if kind == "call":
        return float(discount * (expected * norm.cdf(d1) - strike * norm.cdf(d2)))
    return float(discount * (strike * norm.cdf(-d2) - expected * norm.cdf(-d1)))


def price_asian(
    spot: float, strike: float, rate: float, volatility: float, years: float,
    steps: int = 12, paths: int = 200_000, kind: str = "call",
    averaging: str = "arithmetic", seed: int | None = None,
    control_variate: bool = False, chunk: int = DEFAULT_CHUNK,
) -> Estimate:
    """Price an Asian option by Monte Carlo over `steps` monitoring dates.

    Args:
        averaging: "arithmetic" (no closed form) or "geometric" (checkable).
        control_variate: Use the geometric Asian as a control for the
            arithmetic one. The two are highly correlated, so this is far more
            effective here than the terminal-price control is for a vanilla.
    """
    _validate(spot, strike, volatility, years, steps, paths)
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if averaging not in ("arithmetic", "geometric"):
        raise ValueError("averaging must be 'arithmetic' or 'geometric'")
    if control_variate and averaging != "arithmetic":
        raise ValueError("the control variate applies to arithmetic averaging only")

    discount = np.exp(-rate * years)
    accumulator = _Accumulator()

    for prices in _path_chunks(spot, rate, volatility, years, steps, paths, seed, chunk):
        arithmetic = prices.mean(axis=1)
        geometric = np.exp(np.log(prices).mean(axis=1))
        average = arithmetic if averaging == "arithmetic" else geometric

        payoff = (average - strike) if kind == "call" else (strike - average)
        values = discount * np.maximum(payoff, 0.0)

        if control_variate:
            control_payoff = (geometric - strike) if kind == "call" else (strike - geometric)
            accumulator.update(values, discount * np.maximum(control_payoff, 0.0))
        else:
            accumulator.update(values)

    if control_variate:
        exact = geometric_asian_closed_form(spot, strike, rate, volatility, years, steps, kind)
        return accumulator.controlled_estimate(exact)
    return accumulator.estimate()


def barrier_closed_form(
    spot: float, strike: float, barrier: float, rate: float, volatility: float,
    years: float, kind: str = "call", knock: str = "out", side: str = "down",
) -> float:
    """Reiner-Rubinstein price for a *continuously* monitored barrier option.

    Args:
        kind: "call" or "put".
        knock: "in" (option activates on touch) or "out" (option dies on touch).
        side: "down" (barrier below spot) or "up" (barrier above spot).
    """
    if spot <= 0 or strike <= 0 or barrier <= 0:
        raise ValueError("spot, strike and barrier must be positive")
    if volatility <= 0 or years <= 0:
        raise ValueError("volatility and years must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if knock not in ("in", "out"):
        raise ValueError("knock must be 'in' or 'out'")
    if side not in ("down", "up"):
        raise ValueError("side must be 'down' or 'up'")

    vanilla = black_scholes(spot, strike, rate, volatility, years, kind)

    # Already knocked out (or in) before the option starts
    if (side == "down" and spot <= barrier) or (side == "up" and spot >= barrier):
        return vanilla if knock == "in" else 0.0

    mu = (rate - 0.5 * volatility**2) / volatility**2
    sqrt_t = volatility * np.sqrt(years)
    phi = 1.0 if kind == "call" else -1.0
    eta = 1.0 if side == "down" else -1.0
    discount = np.exp(-rate * years)

    x1 = np.log(spot / strike) / sqrt_t + (1 + mu) * sqrt_t
    x2 = np.log(spot / barrier) / sqrt_t + (1 + mu) * sqrt_t
    y1 = np.log(barrier**2 / (spot * strike)) / sqrt_t + (1 + mu) * sqrt_t
    y2 = np.log(barrier / spot) / sqrt_t + (1 + mu) * sqrt_t
    ratio = barrier / spot

    a = phi * spot * norm.cdf(phi * x1) - phi * strike * discount * norm.cdf(phi * (x1 - sqrt_t))
    b = phi * spot * norm.cdf(phi * x2) - phi * strike * discount * norm.cdf(phi * (x2 - sqrt_t))
    # The reflected terms carry the barrier's image about the log-price axis
    reflect_spot = spot * ratio ** (2 * (mu + 1))
    reflect_strike = strike * discount * ratio ** (2 * mu)
    c = phi * reflect_spot * norm.cdf(eta * y1) - phi * reflect_strike * norm.cdf(
        eta * (y1 - sqrt_t)
    )
    d = phi * reflect_spot * norm.cdf(eta * y2) - phi * reflect_strike * norm.cdf(
        eta * (y2 - sqrt_t)
    )

    above = strike > barrier
    if side == "down" and kind == "call":
        knock_in = c if above else a - b + d
    elif side == "up" and kind == "call":
        knock_in = a if above else b - c + d
    elif side == "down" and kind == "put":
        knock_in = b - c + d if above else a
    else:
        knock_in = a - b + d if above else c

    knock_in = float(min(max(knock_in, 0.0), vanilla))
    return knock_in if knock == "in" else vanilla - knock_in


def continuity_corrected_barrier(
    barrier: float, volatility: float, years: float, steps: int, side: str = "down"
) -> float:
    """Shift a barrier so the continuous formula approximates discrete monitoring.

    Broadie-Glasserman-Kou: a barrier monitored at `steps` discrete dates behaves
    like a continuous barrier moved away from the spot by exp(+/- beta*sigma*sqrt(dt)).
    """
    if side not in ("down", "up"):
        raise ValueError("side must be 'down' or 'up'")
    sign = -1.0 if side == "down" else 1.0
    return float(barrier * np.exp(sign * BGK_BETA * volatility * np.sqrt(years / steps)))


def price_barrier(
    spot: float, strike: float, barrier: float, rate: float, volatility: float,
    years: float, steps: int = 252, paths: int = 200_000, kind: str = "call",
    knock: str = "out", side: str = "down", seed: int | None = None,
    chunk: int = DEFAULT_CHUNK,
) -> Estimate:
    """Price a discretely monitored barrier option by Monte Carlo.

    The barrier is checked at each of the `steps` monitoring dates, which is
    what a real contract specifies. The result is therefore biased relative to
    the continuous closed form by O(1/sqrt(steps)); see
    `continuity_corrected_barrier` for the standard adjustment.
    """
    _validate(spot, strike, volatility, years, steps, paths)
    if barrier <= 0:
        raise ValueError("barrier must be positive")
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    if knock not in ("in", "out"):
        raise ValueError("knock must be 'in' or 'out'")
    if side not in ("down", "up"):
        raise ValueError("side must be 'down' or 'up'")

    discount = np.exp(-rate * years)
    accumulator = _Accumulator()

    for prices in _path_chunks(spot, rate, volatility, years, steps, paths, seed, chunk):
        if side == "down":
            touched = prices.min(axis=1) <= barrier
        else:
            touched = prices.max(axis=1) >= barrier

        terminal = prices[:, -1]
        payoff = (terminal - strike) if kind == "call" else (strike - terminal)
        payoff = np.maximum(payoff, 0.0)

        alive = touched if knock == "in" else ~touched
        accumulator.update(discount * np.where(alive, payoff, 0.0))

    return accumulator.estimate()
