"""Tests for path-dependent pricing, where the oracles are subtler than Black-Scholes.

An arithmetic Asian option and a discretely monitored barrier option have no
exact closed form, so each test below leans on something that does hold exactly:
the geometric Asian formula, the AM-GM inequality, in-out parity, or the
Broadie-Glasserman-Kou continuity correction.
"""

from __future__ import annotations
import numpy as np
import pytest
from exotics import (
    barrier_closed_form,
    continuity_corrected_barrier,
    geometric_asian_closed_form,
    price_asian,
    price_barrier,
)
from finance import black_scholes

SPOT, STRIKE, RATE, VOL, YEARS = 100.0, 100.0, 0.05, 0.25, 1.0


# --- Asian options ---------------------------------------------------------


def test_geometric_asian_simulation_matches_closed_form() -> None:
    exact = geometric_asian_closed_form(SPOT, STRIKE, RATE, VOL, YEARS, steps=12)
    est = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=200_000, averaging="geometric", seed=7
    )
    low, high = est.ci95
    assert low <= exact <= high


@pytest.mark.parametrize("kind", ["call", "put"])
def test_geometric_closed_form_matches_simulation_for_both_kinds(kind: str) -> None:
    exact = geometric_asian_closed_form(SPOT, STRIKE, RATE, VOL, YEARS, steps=12, kind=kind)
    est = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=200_000,
        averaging="geometric", kind=kind, seed=8,
    )
    low, high = est.ci95
    assert low <= exact <= high


def test_geometric_control_variate_slashes_standard_error() -> None:
    plain = price_asian(SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=100_000, seed=9)
    controlled = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=100_000, seed=9, control_variate=True
    )
    # The two averages are almost perfectly correlated, so expect a large win
    assert controlled.stderr < plain.stderr / 10


def test_control_variate_does_not_shift_the_price() -> None:
    plain = price_asian(SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=200_000, seed=11)
    controlled = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=200_000, seed=11, control_variate=True
    )
    # Both estimate the same quantity; they must agree within the looser error bar
    assert abs(controlled.value - plain.value) < 4 * plain.stderr


def test_arithmetic_asian_is_worth_at_least_the_geometric_one() -> None:
    # AM-GM holds pathwise, so the arithmetic call dominates the geometric call
    arithmetic = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=100_000, seed=13
    ).value
    geometric = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=100_000, averaging="geometric", seed=13
    ).value
    assert arithmetic >= geometric


def test_asian_call_is_cheaper_than_the_vanilla() -> None:
    # Averaging damps the terminal variance, so the Asian must be worth less
    asian = price_asian(
        SPOT, STRIKE, RATE, VOL, YEARS, steps=12, paths=100_000, seed=17, control_variate=True
    ).value
    assert asian < black_scholes(SPOT, STRIKE, RATE, VOL, YEARS, "call")


def test_chunk_size_does_not_change_the_price() -> None:
    common = dict(steps=12, paths=60_000, seed=5)
    small = price_asian(SPOT, STRIKE, RATE, VOL, YEARS, chunk=10_000, **common)
    large = price_asian(SPOT, STRIKE, RATE, VOL, YEARS, chunk=60_000, **common)
    # Identical draws, but the running sums accumulate in a different order,
    # so agreement is to floating-point rounding rather than bit-for-bit
    assert small.value == pytest.approx(large.value, rel=1e-12)
    assert small.stderr == pytest.approx(large.stderr, rel=1e-12)


# --- Barrier options -------------------------------------------------------

BARRIER_CASES = [
    ("down", "out", "call", 90.0),
    ("down", "in", "call", 90.0),
    ("up", "out", "call", 120.0),
    ("up", "in", "call", 120.0),
    ("down", "out", "put", 85.0),
    ("up", "out", "put", 115.0),
]


@pytest.mark.parametrize("side,knock,kind,barrier", BARRIER_CASES)
def test_barrier_simulation_matches_continuity_corrected_formula(
    side: str, knock: str, kind: str, barrier: float
) -> None:
    steps = 100
    est = price_barrier(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, steps=steps, paths=150_000,
        kind=kind, knock=knock, side=side, seed=13,
    )
    # Discrete monitoring is biased against the continuous formula, so shift the
    # barrier by the BGK correction before comparing
    shifted = continuity_corrected_barrier(barrier, VOL, YEARS, steps, side)
    corrected = barrier_closed_form(
        SPOT, STRIKE, shifted, RATE, VOL, YEARS, kind=kind, knock=knock, side=side
    )
    assert abs(est.value - corrected) < 4 * est.stderr


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("side,barrier", [("down", 90.0), ("up", 115.0)])
def test_closed_form_in_out_parity(kind: str, side: str, barrier: float) -> None:
    knock_in = barrier_closed_form(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, kind=kind, knock="in", side=side
    )
    knock_out = barrier_closed_form(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, kind=kind, knock="out", side=side
    )
    vanilla = black_scholes(SPOT, STRIKE, RATE, VOL, YEARS, kind)
    assert knock_in + knock_out == pytest.approx(vanilla)


@pytest.mark.parametrize("side,barrier", [("down", 1e-8), ("up", 1e8)])
def test_unreachable_barrier_reduces_to_the_vanilla(side: str, barrier: float) -> None:
    vanilla = black_scholes(SPOT, STRIKE, RATE, VOL, YEARS, "call")
    knock_out = barrier_closed_form(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, knock="out", side=side
    )
    knock_in = barrier_closed_form(SPOT, STRIKE, barrier, RATE, VOL, YEARS, knock="in", side=side)
    assert knock_out == pytest.approx(vanilla, abs=1e-6)
    assert knock_in == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("side,barrier", [("down", 105.0), ("up", 95.0)])
def test_already_breached_barrier_is_settled_at_inception(side: str, barrier: float) -> None:
    # Spot is already through the barrier, so the knock event has occurred
    vanilla = black_scholes(SPOT, STRIKE, RATE, VOL, YEARS, "call")
    assert barrier_closed_form(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, knock="out", side=side
    ) == pytest.approx(0.0)
    assert barrier_closed_form(
        SPOT, STRIKE, barrier, RATE, VOL, YEARS, knock="in", side=side
    ) == pytest.approx(vanilla)


def test_simulated_in_and_out_sum_to_the_simulated_vanilla() -> None:
    # Sharing a seed makes this an exact identity: every path lands in exactly
    # one of the two legs, so the knocked-in and knocked-out values must add up
    kwargs = dict(steps=100, paths=100_000, side="down", seed=21)
    knock_out = price_barrier(SPOT, STRIKE, 90.0, RATE, VOL, YEARS, knock="out", **kwargs).value
    knock_in = price_barrier(SPOT, STRIKE, 90.0, RATE, VOL, YEARS, knock="in", **kwargs).value
    # An unreachable barrier reproduces the vanilla payoff on the same paths
    vanilla_mc = price_barrier(
        SPOT, STRIKE, 1e-8, RATE, VOL, YEARS, knock="out", **kwargs
    ).value
    assert knock_out + knock_in == pytest.approx(vanilla_mc, rel=1e-12)


def test_discretisation_bias_shrinks_as_monitoring_gets_finer() -> None:
    # A discretely monitored knock-out is worth more than a continuous one,
    # and the gap should narrow as the monitoring grid is refined
    continuous = barrier_closed_form(SPOT, STRIKE, 90.0, RATE, VOL, YEARS, knock="out")
    gaps = []
    for steps in (25, 400):
        est = price_barrier(
            SPOT, STRIKE, 90.0, RATE, VOL, YEARS, steps=steps, paths=100_000, seed=23
        )
        gaps.append(est.value - continuous)
    assert gaps[0] > gaps[1] > 0


def test_barrier_pricing_is_reproducible_with_seed() -> None:
    kwargs = dict(steps=50, paths=50_000, seed=29)
    first = price_barrier(SPOT, STRIKE, 90.0, RATE, VOL, YEARS, **kwargs)
    second = price_barrier(SPOT, STRIKE, 90.0, RATE, VOL, YEARS, **kwargs)
    assert first == second


# --- Input validation ------------------------------------------------------


def test_control_variate_rejected_for_geometric_averaging() -> None:
    with pytest.raises(ValueError, match="arithmetic averaging only"):
        price_asian(
            SPOT, STRIKE, RATE, VOL, YEARS, paths=1000,
            averaging="geometric", control_variate=True,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"spot": 0.0},
        {"strike": -1.0},
        {"volatility": -0.1},
        {"years": 0.0},
        {"steps": 0},
        {"paths": 0},
        {"kind": "digital"},
        {"averaging": "harmonic"},
    ],
)
def test_asian_rejects_invalid_inputs(kwargs: dict) -> None:
    args = {
        "spot": SPOT, "strike": STRIKE, "rate": RATE, "volatility": VOL,
        "years": YEARS, "steps": 12, "paths": 1000,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        price_asian(**args)


@pytest.mark.parametrize(
    "kwargs",
    [{"barrier": 0.0}, {"knock": "sideways"}, {"side": "diagonal"}, {"kind": "digital"}],
)
def test_barrier_rejects_invalid_inputs(kwargs: dict) -> None:
    args = {
        "spot": SPOT, "strike": STRIKE, "barrier": 90.0, "rate": RATE,
        "volatility": VOL, "years": YEARS, "steps": 12, "paths": 1000,
    }
    args.update(kwargs)
    with pytest.raises(ValueError):
        price_barrier(**args)


def test_continuity_correction_moves_the_barrier_away_from_spot() -> None:
    down = continuity_corrected_barrier(90.0, VOL, YEARS, 100, "down")
    up = continuity_corrected_barrier(110.0, VOL, YEARS, 100, "up")
    assert down < 90.0
    assert up > 110.0
    # Finer monitoring means a smaller shift
    assert continuity_corrected_barrier(90.0, VOL, YEARS, 10_000, "down") > down
    with pytest.raises(ValueError):
        continuity_corrected_barrier(90.0, VOL, YEARS, 100, "sideways")


def test_geometric_closed_form_rejects_bad_kind() -> None:
    with pytest.raises(ValueError, match="kind must be"):
        geometric_asian_closed_form(SPOT, STRIKE, RATE, VOL, YEARS, steps=12, kind="digital")


def test_arithmetic_average_beats_geometric_pathwise() -> None:
    # Guard the inequality the AM-GM test above relies on, directly on the paths
    rng = np.random.default_rng(0)
    sample = rng.lognormal(mean=0.0, sigma=0.4, size=(5000, 12))
    assert np.all(sample.mean(axis=1) >= np.exp(np.log(sample).mean(axis=1)) - 1e-12)
