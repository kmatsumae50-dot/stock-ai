import pandas as pd

from stock_ai.external.market_data import (
    AUTO_SECTOR_SPECS,
    ExternalMarketSpec,
    build_auto_sector_feature_frame,
    build_external_market_context,
    build_external_market_feature_frame,
    merge_external_feature_frames,
    specs_for_symbols,
)


def _market_frame(periods: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    close = pd.Series([100.0 + index * 0.5 for index in range(periods)])
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["USDJPY=X"] * periods,
            "open": close * 0.999,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": [1_000_000] * periods,
        }
    )


def test_build_external_market_feature_frame_adds_prefixed_features():
    spec = ExternalMarketSpec(symbol="USDJPY=X", prefix="usdjpy", name="USDJPY")

    result = build_external_market_feature_frame(_market_frame(), spec)

    assert "usdjpy_return_1d" in result.columns
    assert "usdjpy_ma_25" in result.columns
    assert "usdjpy_rsi_14" in result.columns
    assert "usdjpy_volatility_20" in result.columns
    assert "usdjpy_trend_direction" in result.columns


def test_merge_external_feature_frames_uses_left_join_and_forward_fill_only():
    stock = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [100.0, 101.0, 102.0],
        }
    )
    external = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "usdjpy_return_1d": [0.01],
        }
    )

    result = merge_external_feature_frames(stock, [external])

    assert pd.isna(result.loc[0, "usdjpy_return_1d"])
    assert result.loc[1, "usdjpy_return_1d"] == 0.01
    assert result.loc[2, "usdjpy_return_1d"] == 0.01


def test_build_auto_sector_feature_frame_adds_sector_aggregates():
    frames = {}
    for spec in AUTO_SECTOR_SPECS[:2]:
        frames[spec.prefix] = build_external_market_feature_frame(_market_frame(), spec)

    result = build_auto_sector_feature_frame(frames)

    assert "sector_mean_return" in result.columns
    assert "sector_volatility" in result.columns
    assert "sector_trend" in result.columns
    assert "sector_strength" in result.columns
    assert result["sector_trend"].dropna().iloc[-1] == 1


def test_build_external_market_context_summarizes_usdjpy_and_market_trend():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "usdjpy_return_1d": [0.01],
            "usdjpy_close_vs_ma_25": [0.02],
            "usdjpy_trend_direction": [1],
            "n225_return_1d": [0.015],
            "n225_trend_direction": [1],
            "topix_return_1d": [0.012],
            "topix_trend_direction": [1],
            "honda_return_1d": [0.02],
            "honda_trend_direction": [1],
            "tesla_return_1d": [-0.03],
            "tesla_trend_direction": [-1],
            "sector_mean_return": [0.01],
            "sector_trend": [1],
            "sector_strength": [0.03],
        }
    )

    context = build_external_market_context(data)

    assert context.usdjpy_status == "Yen weakening"
    assert context.market_trend == "Positive"
    assert context.auto_sector_trend == "Positive"
    assert context.tesla_trend == "Negative"
    assert context.honda_trend == "Positive"
    assert "exporters" in context.summary
    assert "Auto sector" in context.summary


def test_specs_for_symbols_keeps_known_symbols_and_accepts_future_extensions():
    specs = specs_for_symbols(["USDJPY=X", "TSLA", "CL=F"])

    assert specs[0].prefix == "usdjpy"
    assert specs[1].prefix == "tesla"
    assert specs[2].prefix == "cl_f"


def test_specs_for_symbols_contains_default_auto_sector_symbols():
    prefixes = {spec.prefix for spec in specs_for_symbols(None)}

    assert {"honda", "nissan", "tesla", "byd"}.issubset(prefixes)


def test_build_external_market_feature_frame_strictly_lags_foreign_markets():
    # US series (TSLA) must be shifted by 1 to prevent lookahead leak at 15:30 JST close
    tesla_spec = ExternalMarketSpec(symbol="TSLA", prefix="tesla", name="Tesla", is_foreign=True)
    raw = _market_frame(10)
    raw["close"] = [100.0 + i * 2.0 for i in range(10)]

    unlagged = build_external_market_feature_frame(raw, tesla_spec, lag_foreign=False)
    lagged = build_external_market_feature_frame(raw, tesla_spec, lag_foreign=True)

    # In unlagged, row 1 has return between row 0 and row 1
    # In lagged (shift 1), row 1 is NaN, and row 2 has the return of row 1
    assert not pd.isna(unlagged.loc[1, "tesla_return_1d"])
    assert pd.isna(lagged.loc[1, "tesla_return_1d"])
    assert lagged.loc[2, "tesla_return_1d"] == unlagged.loc[1, "tesla_return_1d"]


def test_build_external_market_feature_frame_does_not_lag_domestic_japanese_markets():
    # Japanese domestic (closes at 15:30 JST) is not lagged
    n225_spec = ExternalMarketSpec(symbol="^N225", prefix="n225", name="Nikkei 225", is_foreign=False)
    raw = _market_frame(10)

    result = build_external_market_feature_frame(raw, n225_spec, lag_foreign=True)
    assert not pd.isna(result.loc[1, "n225_return_1d"])

