import yfinance as yf
from pathlib import Path

# Ensure data directory exists inside MonteCarlo-Simulation
project_dir = Path(__file__).parent
data_dir = project_dir / "data"
data_dir.mkdir(exist_ok=True)

# Try multiple S&P 500 symbols with yfinance
symbols = ["^GSPC", "^SPX", "SPY"]
start_date = "2000-01-01"
end_date = "2026-01-01"

success = False
for symbol in symbols:
    print(f"Trying to download data for {symbol}...")
    # Attempt to download historical data for the symbol
    df = yf.download(symbol, start=start_date, end=end_date)
    if not df.empty:
        # Save the data to a CSV file named after the symbol
        csv_path = data_dir / f"{symbol.replace('^','').lower()}.csv"
        df.to_csv(csv_path)
        print(f"Downloaded {symbol} data to {csv_path}")
        success = True
        break
    else:
        print(f"Download failed or no data found for {symbol}.")