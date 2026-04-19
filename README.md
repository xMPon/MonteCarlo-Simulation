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


3. Run sample simulation (from project root):

```powershell
python run_simulation.py --samples 100000 --seed 42
```


> **Note:** The project now uses a flat structure. Use the root-level CLI script (`run_simulation.py`) for launching simulations. No `src/` directory is needed.

4. Run tests:

```powershell
pytest -v
```

5. Open notebook example:

- Use `notebooks/pi_estimation_example.ipynb` for an interactive run.

## Project Layout

- `run_simulation.py` — root CLI script for launching simulations
- `main.py` — CLI entry point and sample simulation logic
- `simulations.py` — core simulation logic
- `test_simulations.py` — unit tests

## Requirements

- The code requires `numpy` (covered by `requirements.txt`). If you see an ImportError, ensure dependencies are installed.

## Missing/To Improve

- No documentation or usage instructions were present in the original README (now added).
- There was no CLI or script at the root for easy launching (now provided as `run_simulation.py`).
- The code assumes `numpy` is installed (requirements.txt covers this, but there is no runtime check in code).
- `notebooks/` Jupyter examples
- `data/` generated outputs (kept out of git except placeholder)

## Next Steps

- Add financial simulations (option pricing, VaR, portfolio paths)
- Add plotting utilities and benchmark notebook
- Add CI checks for tests and linting

## Downloading Market Index Data

To fetch historical S&P 500 data for use in your simulations, run the provided script:

```powershell
python download_index_data.py
```

This will attempt to download S&P 500 data from Yahoo Finance (using yfinance) and save it in the `data/` folder. If Yahoo fails, it will automatically try to fetch the data from Stooq as a fallback.

You can then load the resulting CSV in your analysis or simulations.
