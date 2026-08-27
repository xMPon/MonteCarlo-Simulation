# MonteCarlo-Simulation

Jupyter-first Python project for building and validating Monte Carlo simulations.

## Quick Start

1. Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the sample simulation (from the project root):

```powershell
python main.py --samples 100000 --seed 42
```

Add `--no-plot` to print the estimate without opening a matplotlib window
(useful on headless machines and in CI).

4. Run tests:

```powershell
pytest
```

5. Open the notebook example:

```powershell
jupyter lab notebooks/pi_estimation_example.ipynb
```

## Project Layout

- `main.py` — CLI for the pi-estimation demo: arguments, simulation, plot
- `simulations.py` — core sampling and estimation primitives
- `finance.py` — GBM paths, European option pricing, Black-Scholes benchmark
- `exotics.py` — Asian and barrier options, with their closed-form oracles
- `price_option.py` — CLI for vanilla pricing and variance-reduction comparison
- `price_exotic.py` — CLI for Asian and barrier pricing
- `test_simulations.py`, `test_finance.py`, `test_exotics.py` — unit tests
- `download_index_data.py` — fetches historical S&P 500 data into `data/`
- `notebooks/` — Jupyter examples
- `data/` — downloaded market data (generated, not tracked in git)

## Downloading Market Index Data

To fetch historical S&P 500 data for use in your simulations:

```powershell
python download_index_data.py
```

This tries Yahoo Finance first (`^GSPC`, `^SPX`, then `SPY` via `yfinance`).
If every Yahoo attempt fails it falls back to FRED's `SP500` series, which needs
no API key but provides **daily close prices only, going back about 10 years** —
enough to keep the project runnable, not a full substitute for the Yahoo OHLCV
history. The result is written to `data/`, which is git-ignored, and the script
exits non-zero if all sources fail.

> Stooq, named as the fallback in earlier versions of this README, now sits
> behind a JavaScript bot check and no longer works as a scripted CSV source.

## Option Pricing

Price a European option by Monte Carlo and check it against the closed form:

```powershell
python price_option.py --seed 42
python price_option.py --kind put --strike 90 --years 0.5 --seed 42
```

Output compares four estimators at the same path count:

```
Black-Scholes (exact): 9.4134

method                   price   std error   error vs BS  SE reduction
----------------------------------------------------------------------
plain                   9.4281      0.0318       +0.0147          1.0x
antithetic              9.4375      0.0237       +0.0241          1.3x
control variate         9.4297      0.0130       +0.0163          2.4x
both                    9.4279      0.0182       +0.0145          1.7x
```

Black-Scholes is the exact benchmark, so `error vs BS` is the simulation's true
error. It should sit within roughly two standard errors — that is what the test
suite asserts, by checking the closed-form price falls inside the simulation's
95% confidence interval across strikes and both option types.

Calibrate the volatility from real index data instead of passing it directly:

```powershell
python download_index_data.py
python price_option.py --calibrate data/gspc.csv --seed 42
```

With `--calibrate`, spot is the last close and `--strike` is read as a
percentage of it (100 = at-the-money). The historical drift is reported but
deliberately not used for pricing: European options are priced under the
risk-neutral measure, where the drift is the risk-free rate.

### On combining variance reduction

Antithetic sampling and the control variate each cut the standard error, but
combining them is *worse* than the control variate alone. This is a real effect
rather than a bug. The control variate absorbs exposure to the terminal price,
which is exactly what antithetic sampling exploits: the antithetic pairs go from
a correlation of -0.44 on the raw payoff to +0.95 after the control is applied,
and antithetic sampling only helps when that correlation is negative.

## Path-Dependent Options

Vanilla options are a warm-up: Black-Scholes prices them exactly, so Monte Carlo
is only ever checking itself. Asian and barrier options are where simulation is
genuinely the practical method — and where validation gets interesting, because
the obvious oracle no longer exists.

### Asian options

```powershell
python price_exotic.py --seed 42
```

```
Vanilla call for reference:   12.3360
Geometric Asian (exact):          6.9907

estimator                               price   std error  SE reduction
-----------------------------------------------------------------------
geometric (has a closed form)          7.0078      0.0229           n/a
arithmetic, plain                      7.3309      0.0238            1x
arithmetic, geometric control          7.3132      0.0008           29x
```

An *arithmetic* Asian option has no closed form. A *geometric* one does, because
the log of a geometric average of lognormals is itself normal. That single fact
does double duty: the geometric price validates the simulation, and then serves
as a control variate for the arithmetic version. The two averages are nearly
perfectly correlated, which is why the standard error drops by **29x** — far more
than the 2.4x the terminal-price control manages on a vanilla.

### Barrier options

```powershell
python price_exotic.py --product barrier --seed 42
python price_exotic.py --product barrier --side up --barrier 115 --knock in --kind put
```

```
Continuously monitored (closed form): 9.1112

monitoring dates    MC (discrete)  BGK-corrected       gap   raw bias
---------------------------------------------------------------------
25                        10.2902        10.2824     0.2SE    +1.1790
100                         9.7614         9.7505     0.3SE    +0.6502
252                         9.5957         9.5272     2.1SE    +0.4844
```

Here the closed form and the simulation are pricing *different contracts*. The
Reiner-Rubinstein formula assumes the barrier is monitored continuously; a real
contract checks it daily. A discretely monitored knock-out survives breaches
that happen between observation dates, so it is worth systematically more — the
`raw bias` column, shrinking as O(1/sqrt(dates)) rather than vanishing.

Naively comparing the two and calling the difference a bug would be a mistake.
The Broadie-Glasserman-Kou correction shifts the barrier away from spot by
`exp(±0.5826 * sigma * sqrt(dt))`, which reconciles them to within a couple of
standard errors across all four side/kind combinations and both knock directions.

Paths are generated in chunks, because a fine monitoring grid over many paths
does not fit in memory otherwise: 3,200 steps x 400,000 paths is 9.5 GB in one
array.

## Next Steps

- Report the standard error for the pi estimator too, so `simulations.py`
  matches the `Estimate` convention `finance.py` already uses
- Add VaR and CVaR, both parametric and bootstrapped from historical returns
- Add Greeks by pathwise and likelihood-ratio methods, checked against
  Black-Scholes analytic deltas
- Replace constant volatility with a fitted stochastic-volatility or jump model,
  since the calibration already shows returns are not lognormal
- Add CI checks for tests and linting
