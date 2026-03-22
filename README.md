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

3. Run sample simulation:

```powershell
cd .\MonteCarlo-Simulation
python -m src.main --samples 100000 --seed 42
```

4. Run tests:

```powershell
pytest -v
```

5. Open notebook example:

- Use `notebooks/pi_estimation_example.ipynb` for an interactive run.

## Project Layout

- `src/` core simulation logic and CLI entry point
- `tests/` unit tests
- `notebooks/` Jupyter examples
- `data/` generated outputs (kept out of git except placeholder)

## Next Steps

- Add financial simulations (option pricing, VaR, portfolio paths)
- Add plotting utilities and benchmark notebook
- Add CI checks for tests and linting
