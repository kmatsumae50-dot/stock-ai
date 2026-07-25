from __future__ import annotations

import pandas as pd


CANONICAL_COLUMNS = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume",
    "Dividends": "dividends",
    "Stock Splits": "stock_splits",
}


def fetch_daily_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV data from Yahoo Finance."""
    import yfinance as yf

    raw = yf.Ticker(ticker).history(start=start, end=end, interval="1d", auto_adjust=False)
    if raw.empty:
        raise ValueError(f"No OHLCV data returned for {ticker}.")

    data = raw.rename(columns=CANONICAL_COLUMNS).reset_index()
    data = data.rename(columns={"Date": "date"})
    data["date"] = pd.to_datetime(data["date"]).dt.tz_localize(None)
    data["ticker"] = ticker

    columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dividends",
        "stock_splits",
    ]
    available_columns = [column for column in columns if column in data.columns]
    return data[available_columns].sort_values(["ticker", "date"]).reset_index(drop=True)
