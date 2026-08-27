"""Tests for the financial Monte Carlo layer, benchmarked against Black-Scholes."""

from __future__ import annotations
import numpy as np
import pytest
from finance import (
    black_scholes,
    calibrate_gbm,
    price_european,
    simulate_gbm_paths,
    simulate_terminal_prices,
)

# A single at-the-money contract reused across the pricing tests
SPOT, STRIKE, RATE, VOL, YEARS = 100.0, 100.0, 0.03, 0.20, 1.0


def test_calibrate_recovers_known_parameters() -> None:
    # Simulate a long path with known parameters, then calibrate back to them
    mu, sigma = 0.08, 0.25
    path = simulate_gbm_paths(
        spot=100.0, drift=mu, volatility=sigma, years=40.0, steps=40 * 252, paths=1, seed=1
    )[0]
    mu_hat, sigma_hat = calibrate_gbm(path)
    # Volatility is estimated far more precisely than drift, so tolerances differ
    assert sigma_hat == pytest.approx(sigma, abs=0.01)
    assert mu_hat == pytest.approx(mu, abs=0.05)


def test_gbm_paths_have_expected_shape_and_start_at_spot() -> None:
    paths = simulate_gbm_paths(
        spot=100.0, drift=0.05, volatility=0.2, years=1.0, steps=12, paths=50, seed=3
    )
    assert paths.shape == (50, 13)
    assert np.all(paths[:, 0] == 100.0)
    assert np.all(paths > 0)


def test_terminal_price_mean_matches_risk_neutral_forward() -> None:
    # Under the risk-neutral measure, E[S_T] = spot * exp(rate * years)
    terminal = simulate_terminal_prices(SPOT, RATE, VOL, YEARS, paths=400_000, seed=5)
    expected = SPOT * np.exp(RATE * YEARS)
    stderr = terminal.std(ddof=1) / np.sqrt(terminal.size)
    assert abs(terminal.mean() - expected) < 4 * stderr


def test_antithetic_draws_are_exactly_mirrored() -> None:
    terminal = simulate_terminal_prices(SPOT, RATE, VOL, YEARS, paths=1000, seed=5, antithetic=True)
    # Each pair of log-returns should sum to twice the deterministic drift term
    log_moves = np.log(terminal / SPOT)
    half = log_moves.size // 2
    drift_term = (RATE - 0.5 * VOL**2) * YEARS
    assert np.allclose(log_moves[:half] + log_moves[half:], 2 * drift_term)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_monte_carlo_price_agrees_with_black_scholes(kind: str) -> None:
    exact = black_scholes(SPOT, STRIKE, RATE, VOL, YEARS, kind=kind)
    estimate = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=200_000, kind=kind, seed=11)
    # The closed-form price must sit inside the simulation's 95% confidence interval
    low, high = estimate.ci95
    assert low <= exact <= high


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("strike", [80.0, 100.0, 125.0])
def test_agreement_holds_across_moneyness(kind: str, strike: float) -> None:
    exact = black_scholes(SPOT, strike, RATE, VOL, YEARS, kind=kind)
    estimate = price_european(
        SPOT, strike, RATE, VOL, YEARS, paths=200_000, kind=kind, seed=17,
        antithetic=True, control_variate=True,
    )
    low, high = estimate.ci95
    assert low <= exact <= high


def test_put_call_parity_is_exact_for_shared_draws() -> None:
    # Sharing a seed makes both legs use identical terminal prices, so parity
    # becomes an exact algebraic identity rather than a statistical one.
    paths = 200_000
    call = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=paths, kind="call", seed=23).value
    put = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=paths, kind="put", seed=23).value
    terminal = simulate_terminal_prices(SPOT, RATE, VOL, YEARS, paths, seed=23)
    discount = np.exp(-RATE * YEARS)
    assert call - put == pytest.approx(discount * (terminal.mean() - STRIKE), abs=1e-12)


def test_put_call_parity_matches_theory_within_sampling_error() -> None:
    # The simulated parity converges on spot - strike * exp(-rate * years), but
    # only to within the sampling error of the mean terminal price. Deriving the
    # tolerance from that standard error keeps this from being a coin flip.
    paths = 200_000
    call = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=paths, kind="call", seed=23).value
    put = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=paths, kind="put", seed=23).value
    terminal = simulate_terminal_prices(SPOT, RATE, VOL, YEARS, paths, seed=23)
    discount = np.exp(-RATE * YEARS)
    stderr = discount * terminal.std(ddof=1) / np.sqrt(paths)
    theoretical = SPOT - STRIKE * discount
    assert abs((call - put) - theoretical) < 4 * stderr


def test_antithetic_sampling_reduces_standard_error() -> None:
    plain = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=100_000, seed=29)
    reduced = price_european(
        SPOT, STRIKE, RATE, VOL, YEARS, paths=100_000, seed=29, antithetic=True
    )
    assert reduced.stderr < plain.stderr


def test_control_variate_reduces_standard_error() -> None:
    plain = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=100_000, seed=31)
    reduced = price_european(
        SPOT, STRIKE, RATE, VOL, YEARS, paths=100_000, seed=31, control_variate=True
    )
    assert reduced.stderr < plain.stderr


def test_pricing_is_reproducible_with_seed() -> None:
    first = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=50_000, seed=37)
    second = price_european(SPOT, STRIKE, RATE, VOL, YEARS, paths=50_000, seed=37)
    assert first == second


def test_black_scholes_matches_known_values() -> None:
    # Reference values for spot=100, strike=100, rate=5%, vol=20%, 1 year
    assert black_scholes(100.0, 100.0, 0.05, 0.20, 1.0, "call") == pytest.approx(10.4506, abs=1e-4)
    assert black_scholes(100.0, 100.0, 0.05, 0.20, 1.0, "put") == pytest.approx(5.5735, abs=1e-4)


def test_expired_option_is_worth_intrinsic_value() -> None:
    assert black_scholes(120.0, 100.0, 0.05, 0.20, 0.0, "call") == pytest.approx(20.0)
    assert black_scholes(80.0, 100.0, 0.05, 0.20, 0.0, "put") == pytest.approx(20.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": -1.0},
        {"strike": 0.0},
        {"volatility": -0.1},
        {"years": 0.0},
        {"paths": 0},
        {"kind": "straddle"},
    ],
)
def test_invalid_inputs_raise_value_error(kwargs: dict) -> None:
    args = {
        "spot": SPOT, "strike": STRIKE, "rate": RATE,
        "volatility": VOL, "years": YEARS, "paths": 1000,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        price_european(**args)


def test_calibrate_rejects_bad_series() -> None:
    with pytest.raises(ValueError, match="at least two prices"):
        calibrate_gbm([100.0])
    with pytest.raises(ValueError, match="must be positive"):
        calibrate_gbm([100.0, -5.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        calibrate_gbm([[100.0, 101.0], [102.0, 103.0]])
