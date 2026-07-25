import pandas as pd

from stock_ai.cross_ticker.similarity import (
    cross_ticker_direction,
    find_cross_ticker_similar_patterns,
    summarize_cross_ticker_patterns,
)


def _frame(ticker: str, dates: pd.DatetimeIndex, returns: list[float]) -> pd.DataFrame:
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": [ticker] * len(dates),
            "close": close,
            "return_1d": returns,
            "volume_change_1d": returns,
            "rsi_14": [55.0 + value * 100 for value in returns],
            "macd": returns,
            "volatility_20": [abs(value) for value in returns],
            "sector_strength": returns,
            "narrative_bullish_pressure": [0.3 if value > 0 else 0.0 for value in returns],
            "narrative_bearish_pressure": [0.3 if value < 0 else 0.0 for value in returns],
        }
    )


def test_find_cross_ticker_similar_patterns_returns_peer_windows_and_toyota_returns():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    pattern = [0.01, -0.005, 0.015, -0.002, 0.004] * 4
    target_returns = [0.002] * 100 + pattern
    target = _frame("7203.T", dates, target_returns)

    peer_returns = [0.001] * 120
    peer_returns[40:60] = pattern
    peer_returns[60:80] = [0.01] * 20
    peer = _frame("TSLA", dates, peer_returns)

    result = find_cross_ticker_similar_patterns(
        target,
        {"TSLA": peer},
        target_ticker="7203.T",
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
    )

    assert not result.empty
    assert result.loc[0, "ticker"] == "TSLA"
    assert result.loc[0, "similar_start_date"] == dates[40]
    assert result.loc[0, "similar_end_date"] == dates[59]
    assert "peer_return_5d_after" in result.columns
    assert "toyota_return_5d_after" in result.columns
    assert result.loc[0, "similar_features"]

    summary = summarize_cross_ticker_patterns(result, forward_days=(1, 5, 20))
    assert list(summary["horizon"]) == ["1D", "5D", "20D"]
    assert summary.loc[summary["horizon"] == "5D", "count"].iloc[0] > 0
    assert cross_ticker_direction(summary) in {"up", "down", "mixed", "unknown"}
