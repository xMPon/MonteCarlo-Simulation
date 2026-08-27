"""Download historical S&P 500 data, falling back to FRED if Yahoo Finance fails."""

from __future__ import annotations
from pathlib import Path
import pandas as pd
import yfinance as yf

# Ensure data directory exists inside MonteCarlo-Simulation
PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"

# Yahoo Finance tickers to try, in order of preference
YAHOO_SYMBOLS = ["^GSPC", "^SPX", "SPY"]
# FRED serves the daily S&P 500 close as CSV with no API key. It carries only
# the last ~10 years and a single close column, so it is a fallback, not a peer.
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500"
START_DATE = "2000-01-01"


def save(df: pd.DataFrame, name: str) -> Path:
    """Write a dataframe to the data directory and return its path."""
    DATA_DIR.mkdir(exist_ok=True)
    csv_path = DATA_DIR / f"{name}.csv"
    df.to_csv(csv_path)
    return csv_path


def download_from_yahoo() -> Path | None:
    """Try each Yahoo symbol in turn, returning the saved path on the first success."""
    for symbol in YAHOO_SYMBOLS:
        print(f"Trying to download data for {symbol} from Yahoo Finance...")
        try:
            # end is left unset so yfinance downloads through today
            df = yf.download(symbol, start=START_DATE, progress=False)
        except Exception as exc:
            print(f"  Yahoo request for {symbol} failed: {exc}")
            continue
        if df is None or df.empty:
            print(f"  No data returned for {symbol}.")
            continue
        return save(df, symbol.replace("^", "").lower())
    return None


def download_from_fred() -> Path | None:
    """Fall back to FRED's S&P 500 series (daily close only, ~10 years of history)."""
    print("Falling back to FRED...")
    try:
        df = pd.read_csv(FRED_URL, parse_dates=["observation_date"], index_col="observation_date")
    except Exception as exc:
        print(f"  FRED request failed: {exc}")
        return None

    # FRED writes "." for market holidays; drop those rows and name the column Close
    df = df.rename(columns={"SP500": "Close"})
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    df.index.name = "Date"

    if df.empty:
        print("  No usable data returned from FRED.")
        return None
    print(f"  Note: FRED provides close prices only, from {df.index.min().date()}.")
    return save(df.loc[START_DATE:], "spx")


def main() -> int:
    """Download index data from the first working source. Returns a process exit code."""
    csv_path = download_from_yahoo() or download_from_fred()
    if csv_path is None:
        print("All data sources failed. No file was written.")
        return 1
    print(f"Saved index data to {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
