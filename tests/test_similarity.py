import pandas as pd

from stock_ai.similarity.chart_similarity import (
    build_pattern_comparison_frame,
    find_similar_patterns,
    summarize_forward_returns,
)


def test_find_similar_patterns_returns_top_matches_and_forward_returns():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    pattern = [0.01, -0.005, 0.015, -0.002, 0.004] * 4
    returns = [0.002] * 100
    returns[30:50] = pattern
    returns[80:100] = pattern
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))

    volume_change = [0.01] * 100
    volume_change[30:50] = pattern
    volume_change[80:100] = pattern
    rsi = [50.0] * 100
    rsi[30:50] = [55 + value * 100 for value in pattern]
    rsi[80:100] = [55 + value * 100 for value in pattern]
    macd = [0.0] * 100
    macd[30:50] = pattern
    macd[80:100] = pattern
    volatility = [0.01] * 100
    volatility[30:50] = [abs(value) for value in pattern]
    volatility[80:100] = [abs(value) for value in pattern]

    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 100,
            "close": close,
            "return_1d": returns,
            "volume_change_1d": volume_change,
            "rsi_14": rsi,
            "macd": macd,
            "volatility_20": volatility,
        }
    )

    result = find_similar_patterns(
        data,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
    )

    assert len(result) <= 5
    assert not result.empty
    assert "similar_start_date" in result.columns
    assert "similar_end_date" in result.columns
    assert "return_20d_after" in result.columns
    assert result["similarity"].max() <= 1.0
    assert result.loc[0, "similar_start_date"] == dates[30]
    assert result.loc[0, "similar_end_date"] == dates[49]
    assert result.loc[0, "similar_features"]
    assert result.loc[0, "similar_feature_scores"]

    summary = summarize_forward_returns(result, forward_days=(1, 5, 20))
    assert list(summary["horizon"]) == ["1D", "5D", "20D"]
    assert summary.loc[summary["horizon"] == "20D", "count"].iloc[0] > 0

    comparison = build_pattern_comparison_frame(
        data,
        result.iloc[0],
        window_size=20,
        forward_days=20,
    )
    assert not comparison.empty
    assert {"Current", "Matched past"} == set(comparison["series"])
    assert comparison[comparison["series"] == "Current"]["day"].max() == 19
    assert comparison[comparison["phase"] == "future"]["day"].max() == 39


def test_find_similar_patterns_prioritizes_same_regime_when_requested():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    pattern = [0.01, -0.005, 0.015, -0.002, 0.004] * 4
    near_pattern = [value * 0.8 for value in pattern]
    returns = [0.002] * 100
    returns[5:25] = near_pattern
    returns[30:50] = pattern
    returns[80:100] = pattern
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))

    volume_change = [0.01] * 100
    volume_change[5:25] = near_pattern
    volume_change[30:50] = pattern
    volume_change[80:100] = pattern
    rsi = [50.0] * 100
    rsi[5:25] = [55 + value * 90 for value in near_pattern]
    rsi[30:50] = [55 + value * 100 for value in pattern]
    rsi[80:100] = [55 + value * 100 for value in pattern]
    macd = [0.0] * 100
    macd[5:25] = near_pattern
    macd[30:50] = pattern
    macd[80:100] = pattern
    volatility = [0.01] * 100
    volatility[5:25] = [abs(value) for value in near_pattern]
    volatility[30:50] = [abs(value) for value in pattern]
    volatility[80:100] = [abs(value) for value in pattern]
    regime = ["Sideways"] * 100
    regime[5:25] = ["Bull Trend"] * 20
    regime[30:50] = ["Bear Trend"] * 20
    regime[80:100] = ["Bull Trend"] * 20

    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 100,
            "close": close,
            "return_1d": returns,
            "volume_change_1d": volume_change,
            "rsi_14": rsi,
            "macd": macd,
            "volatility_20": volatility,
            "regime": regime,
        }
    )

    result = find_similar_patterns(
        data,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
        regime_column="regime",
        prefer_regime="Bull Trend",
    )

    assert not result.empty
    assert result.loc[0, "regime"] == "Bull Trend"
    assert result.loc[0, "regime_match"]


def test_find_similar_patterns_prioritizes_same_event_state_when_requested():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    pattern = [0.01, -0.005, 0.015, -0.002, 0.004] * 4
    near_pattern = [value * 0.8 for value in pattern]
    returns = [0.002] * 100
    returns[5:25] = near_pattern
    returns[30:50] = pattern
    returns[80:100] = pattern
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))

    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 100,
            "close": close,
            "return_1d": returns,
            "volume_change_1d": returns,
            "rsi_14": [55.0] * 100,
            "macd": returns,
            "volatility_20": [abs(value) for value in returns],
            "earnings_event_state": ["normal"] * 100,
            "earnings_event_state_code": [0] * 100,
        }
    )
    data.loc[5:24, "earnings_event_state"] = "post_earnings"
    data.loc[5:24, "earnings_event_state_code"] = 2
    data.loc[80:99, "earnings_event_state"] = "post_earnings"
    data.loc[80:99, "earnings_event_state_code"] = 2

    result = find_similar_patterns(
        data,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
        event_state_column="earnings_event_state",
        prefer_event_state="post_earnings",
    )

    assert not result.empty
    assert result.loc[0, "event_state"] == "post_earnings"
    assert result.loc[0, "event_state_match"]


def test_find_similar_patterns_prioritizes_same_narrative_when_requested():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    pattern = [0.01, -0.005, 0.015, -0.002, 0.004] * 4
    near_pattern = [value * 0.8 for value in pattern]
    returns = [0.002] * 100
    returns[5:25] = near_pattern
    returns[30:50] = pattern
    returns[80:100] = pattern
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))
    narrative = ["none"] * 100
    narrative[5:25] = ["ev"] * 20
    narrative[30:50] = ["yen_weakness"] * 20
    narrative[80:100] = ["ev"] * 20

    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 100,
            "close": close,
            "return_1d": returns,
            "volume_change_1d": returns,
            "rsi_14": [55.0] * 100,
            "macd": returns,
            "volatility_20": [abs(value) for value in returns],
            "dominant_narrative": narrative,
            "dominant_narrative_code": [1] * 100,
        }
    )

    result = find_similar_patterns(
        data,
        window_size=20,
        top_n=5,
        forward_days=(1, 5, 20),
        narrative_column="dominant_narrative",
        prefer_narrative="ev",
    )

    assert not result.empty
    assert result.loc[0, "narrative"] == "ev"
    assert result.loc[0, "narrative_match"]
