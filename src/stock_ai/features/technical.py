from __future__ import annotations

import pandas as pd


BASE_COLUMNS = {"date", "ticker", "open", "high", "low", "close", "volume"}

TECHNICAL_DISPLAY_COLUMNS = (
    "return_1d",
    "volume_change_1d",
    "intraday_return",
    "high_low_range",
    "ma_5",
    "ma_25",
    "ma_75",
    "close_vs_ma_5",
    "close_vs_ma_25",
    "close_vs_ma_75",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "volatility_20",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
)


def validate_ohlcv(data: pd.DataFrame) -> None:
    missing = BASE_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")


def available_technical_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in TECHNICAL_DISPLAY_COLUMNS if column in data.columns]


def add_technical_features(
    data: pd.DataFrame,
    moving_average_windows: tuple[int, ...] = (5, 25, 75),
    rsi_window: int = 14,
    volatility_window: int = 20,
    bollinger_window: int = 20,
) -> pd.DataFrame:
    """Add only past-and-current-day technical features."""
    validate_ohlcv(data)
    result = data.sort_values(["ticker", "date"]).copy()
    grouped = result.groupby("ticker", group_keys=False)

    result["return_1d"] = grouped["close"].pct_change()
    result["volume_change_1d"] = grouped["volume"].pct_change()
    result["intraday_return"] = result["close"] / result["open"] - 1
    result["high_low_range"] = result["high"] / result["low"] - 1
    result["close_position"] = (result["close"] - result["low"]) / (result["high"] - result["low"])
    result["close_position"] = result["close_position"].replace([float("inf"), -float("inf")], pd.NA)

    for window in moving_average_windows:
        ma = grouped["close"].rolling(window=window, min_periods=window).mean()
        ma = ma.reset_index(level=0, drop=True)
        result[f"ma_{window}"] = ma
        result[f"close_vs_ma_{window}"] = result["close"] / ma - 1

    result[f"volatility_{volatility_window}"] = grouped["return_1d"].rolling(
        window=volatility_window,
        min_periods=volatility_window,
    ).std().reset_index(level=0, drop=True)

    bb_mean = grouped["close"].rolling(
        window=bollinger_window,
        min_periods=bollinger_window,
    ).mean().reset_index(level=0, drop=True)
    bb_std = grouped["close"].rolling(
        window=bollinger_window,
        min_periods=bollinger_window,
    ).std().reset_index(level=0, drop=True)
    result["bb_middle"] = bb_mean
    result["bb_upper"] = bb_mean + 2 * bb_std
    result["bb_lower"] = bb_mean - 2 * bb_std
    result["bb_percent_b"] = (result["close"] - result["bb_lower"]) / (
        result["bb_upper"] - result["bb_lower"]
    )

    result[f"rsi_{rsi_window}"] = grouped["close"].transform(
        lambda close: _rsi(close, window=rsi_window)
    )

    result["macd"] = grouped["close"].transform(_macd_line)
    result["macd_signal"] = grouped["close"].transform(_macd_signal)
    result["macd_hist"] = result["macd"] - result["macd_signal"]

    return result.replace([float("inf"), -float("inf")], pd.NA)


def _rsi(close: pd.Series, window: int) -> pd.Series:
    diff = close.diff()
    gain = diff.clip(lower=0)
    loss = -diff.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd_line(close: pd.Series) -> pd.Series:
    ema_fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    return ema_fast - ema_slow


def _macd_signal(close: pd.Series) -> pd.Series:
    macd = _macd_line(close)
    return macd.ewm(span=9, adjust=False, min_periods=9).mean()
