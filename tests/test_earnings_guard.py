from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from stock_ai.backtest.overnight_portfolio import simulate_overnight_strategy
from stock_ai.events.earnings_guard import (
    batch_check_earnings_guard,
    check_earnings_risk,
    fetch_next_earnings_date,
)


def test_fetch_next_earnings_date_with_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "earnings_cache.json"

    # Mock yfinance Ticker calendar
    mock_ticker = MagicMock()
    mock_ticker.calendar = {"Earnings Date": [date(2026, 11, 5)]}

    with patch("yfinance.Ticker", return_value=mock_ticker):
        e_date = fetch_next_earnings_date("7203.T", cache_path=cache_path, force_refresh=True)
        assert e_date == "2026-11-05"
        assert cache_path.exists()

    # Second call should read from cache without yfinance
    with patch("yfinance.Ticker", side_effect=RuntimeError("Should not be called")):
        cached_date = fetch_next_earnings_date("7203.T", cache_path=cache_path, force_refresh=False)
        assert cached_date == "2026-11-05"


def test_fetch_next_earnings_date_safe_fallback(tmp_path: Path) -> None:
    cache_path = tmp_path / "empty_cache.json"
    with patch("yfinance.Ticker", side_effect=Exception("API down")):
        result = fetch_next_earnings_date("9999.T", cache_path=cache_path, force_refresh=True)
        assert result is None


def test_check_earnings_risk_logic() -> None:
    # Same day earnings
    risk, ed = check_earnings_risk("7203.T", ref_date="2026-11-05", earnings_date="2026-11-05")
    assert risk is True
    assert ed == "2026-11-05"

    # Day before earnings (Thursday -> Friday)
    risk, _ = check_earnings_risk("7203.T", ref_date="2026-11-04", earnings_date="2026-11-05")
    assert risk is True

    # Friday before Monday earnings (weekend gap risk)
    risk, _ = check_earnings_risk("7203.T", ref_date="2026-11-06", earnings_date="2026-11-09")
    assert risk is True

    # Safe future date (more than 3 days)
    risk, _ = check_earnings_risk("7203.T", ref_date="2026-10-16", earnings_date="2026-11-05")
    assert risk is False

    # Already passed earnings date
    risk, _ = check_earnings_risk("7203.T", ref_date="2026-11-10", earnings_date="2026-11-05")
    assert risk is False

    # No earnings date known
    risk, ed = check_earnings_risk("7203.T", ref_date="2026-11-05", earnings_date=None, cache_path=None)
    # When cache is None and yf returns None:
    with patch("stock_ai.events.earnings_guard.fetch_next_earnings_date", return_value=None):
        risk, ed = check_earnings_risk("7203.T", ref_date="2026-11-05", earnings_date=None)
        assert risk is False
        assert ed is None


def test_batch_check_earnings_guard() -> None:
    with patch("stock_ai.events.earnings_guard.fetch_next_earnings_date") as mock_fetch:
        mock_fetch.side_effect = lambda t, **kw: "2026-11-05" if t == "7203.T" else None
        res = batch_check_earnings_guard(["7203.T", "8306.T"], ref_date="2026-11-05")
        assert res["7203.T"]["earnings_risk"] is True
        assert res["7203.T"]["status"] == "⚠️ 決算警戒"
        assert res["8306.T"]["earnings_risk"] is False
        assert res["8306.T"]["status"] == "✅ 通常"


def test_strategy_b_earnings_guard_skips_risky_ticker() -> None:
    """Verify that Paper Trader Strategy B excludes earnings_risk == True tickers."""
    test_dates = [pd.Timestamp("2026-11-05")]

    # Two Q5 tickers: 7203.T (higher p_up but has earnings_risk) and 8306.T (clean)
    df = pd.DataFrame([
        {
            "date": pd.Timestamp("2026-11-05"),
            "ticker": "7203.T",
            "auc": 0.62,
            "quantile": "Q5",
            "win_rate": 0.65,
            "p_up": 0.75,
            "next_open_return": 0.01,
            "earnings_risk": True,
        },
        {
            "date": pd.Timestamp("2026-11-05"),
            "ticker": "8306.T",
            "auc": 0.60,
            "quantile": "Q5",
            "win_rate": 0.62,
            "p_up": 0.65,
            "next_open_return": 0.01,
            "earnings_risk": False,
        },
    ])

    sim = simulate_overnight_strategy(df, test_dates, strategy_name="elite_top2_long")
    logs = sim["trade_logs"]
    assert len(logs) == 1
    picks = logs[0]["long_picks"]

    # 7203.T must be excluded because earnings_risk is True, leaving only 8306.T
    assert "7203.T" not in picks
    assert "8306.T" in picks
