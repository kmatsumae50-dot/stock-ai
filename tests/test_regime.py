import pandas as pd

from stock_ai.regime.rule_based import (
    REGIME_BULL,
    REGIME_PANIC,
    add_regime_labels,
    classify_latest_regime,
)


def _base_frame(periods: int = 90) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series([100 + index * 0.4 for index in range(periods)], dtype="float")
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * periods,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": [1_000_000] * periods,
            "return_1d": close.pct_change(),
            "rsi_14": [62.0] * periods,
            "macd_hist": [0.2] * periods,
            "volatility_20": [0.01] * periods,
            "close_vs_ma_25": [0.04] * periods,
        }
    )


def test_classify_latest_regime_detects_bull_trend():
    state = classify_latest_regime(_base_frame())

    assert state.name == REGIME_BULL
    assert "rsi" in state.features
    assert state.metrics["ma_deviation"] == 0.04


def test_classify_latest_regime_detects_panic():
    data = _base_frame()
    last_index = data.index[-1]
    data.loc[last_index, "close"] = data.loc[last_index - 1, "close"] * 0.92
    data.loc[last_index, "return_1d"] = -0.08
    data.loc[last_index, "price_change_5d"] = -0.12
    data.loc[last_index, "volume"] = 2_500_000
    data.loc[last_index, "rsi_14"] = 25.0
    data.loc[last_index, "macd_hist"] = -0.4
    data.loc[last_index, "volatility_20"] = 0.04
    data.loc[last_index, "close_vs_ma_25"] = -0.08

    state = classify_latest_regime(data)

    assert state.name == REGIME_PANIC
    assert state.level == "High stress"


def test_add_regime_labels_adds_regime_column():
    result = add_regime_labels(_base_frame())

    assert "regime" in result.columns
    assert result["regime"].iloc[-1] == REGIME_BULL
