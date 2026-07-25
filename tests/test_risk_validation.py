import pandas as pd

from stock_ai.evaluation.risk_validation import (
    RiskValidationConfig,
    apply_bucket_mode,
    bucket_for_score,
    build_risk_validation_frame,
    compare_bucket_modes,
    high_risk_days,
    risk_bucket,
    risk_score_distribution,
    score_percentile,
    summarize_risk_buckets,
)


def test_risk_bucket_boundaries_are_stable():
    assert risk_bucket(0) == "Low"
    assert risk_bucket(33) == "Low"
    assert risk_bucket(34) == "Medium"
    assert risk_bucket(66) == "Medium"
    assert risk_bucket(67) == "High"
    assert risk_bucket(100) == "High"


def test_build_risk_validation_frame_and_bucket_summaries():
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    close = [100.0 + index * 0.1 for index in range(100)]
    close[71] = close[70] * 1.04
    close[81] = close[80] * 0.96
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 100,
            "close": close,
            "volume": [100.0] * 100,
            "volume_change_1d": [0.0] * 100,
            "rsi_14": [50.0] * 100,
            "macd_hist": [0.0] * 100,
            "volatility_20": [0.02] * 100,
            "positive_news_shock": [0.0] * 100,
            "negative_news_shock": [0.0] * 100,
            "importance_weighted_sentiment": [0.0] * 100,
            "sentiment_trend_3d": [0.0] * 100,
            "sector_news_strength": [0.0] * 100,
        }
    )
    data.loc[70, ["volume_change_1d", "rsi_14", "macd_hist", "positive_news_shock"]] = [
        0.6,
        70.0,
        0.2,
        1.0,
    ]
    data.loc[70, ["importance_weighted_sentiment", "sentiment_trend_3d", "sector_news_strength"]] = [
        0.6,
        0.5,
        0.4,
    ]
    data.loc[80, ["volume_change_1d", "rsi_14", "macd_hist", "volatility_20", "negative_news_shock"]] = [
        0.6,
        30.0,
        -0.2,
        0.04,
        1.0,
    ]
    data.loc[80, ["importance_weighted_sentiment", "sentiment_trend_3d", "sector_news_strength"]] = [
        -0.6,
        -0.5,
        -0.4,
    ]

    frame = build_risk_validation_frame(
        data,
        config=RiskValidationConfig(min_history_rows=20),
    )
    surge_summary = summarize_risk_buckets(frame, risk_type="surge")
    crash_summary = summarize_risk_buckets(frame, risk_type="crash")

    assert not frame.empty
    assert set(surge_summary["bucket"]) == {"Low", "Medium", "High"}
    assert set(crash_summary["bucket"]) == {"Low", "Medium", "High"}
    assert not high_risk_days(frame, risk_type="surge").empty
    assert not high_risk_days(frame, risk_type="crash").empty


def test_risk_validation_can_merge_saved_multiclass_signal_by_date():
    dates = pd.date_range("2024-01-01", periods=30, freq="D")
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 30,
            "close": [100.0 + index for index in range(30)],
            "volume_change_1d": [0.0] * 30,
            "rsi_14": [50.0] * 30,
            "macd_hist": [0.0] * 30,
            "volatility_20": [0.02] * 30,
        }
    )
    predictions = pd.DataFrame(
        {
            "date": [dates[20]],
            "ticker": ["7203.T"],
            "predicted_class": [4],
            "probability_surge": [0.40],
            "probability_up": [0.50],
            "probability_flat": [0.05],
            "probability_down": [0.03],
            "probability_crash": [0.02],
        }
    )

    frame = build_risk_validation_frame(
        data,
        config=RiskValidationConfig(min_history_rows=5),
        multiclass_predictions=predictions,
    )
    row = frame[pd.to_datetime(frame["date"]) == dates[20]].iloc[0]

    assert row["multiclass_signal_used"]
    assert row["surge_risk_score"] >= 45


def test_quantile_bucket_modes_create_data_driven_high_bucket():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "ticker": ["7203.T"] * 20,
            "surge_risk_score": list(range(20)),
            "crash_risk_score": list(range(20)),
            "return_1d": [0.01] * 20,
            "return_5d": [0.02] * 20,
            "return_20d": [0.03] * 20,
            "future_max_return_20d": [0.05] * 20,
            "future_min_return_20d": [-0.04] * 20,
            "actual_up_1d": [True] * 20,
            "actual_down_1d": [False] * 20,
            "actual_surge_1d": [False] * 20,
            "actual_crash_1d": [False] * 20,
            "multiclass_signal_used": [False] * 20,
        }
    )

    calibrated = apply_bucket_mode(frame, "quantile_60_90")
    summary = summarize_risk_buckets(frame, risk_type="surge", bucket_mode="quantile_60_90")
    comparison = compare_bucket_modes(frame, risk_type="surge")
    distribution = risk_score_distribution(frame)

    assert (calibrated["surge_risk_bucket"] == "High").sum() == 2
    assert summary.loc[summary["bucket"] == "High", "count"].iloc[0] == 2
    assert set(comparison["bucket_mode"]) == {"fixed_33_66", "quantile_50_85", "quantile_60_90"}
    assert set(distribution["risk_type"]) == {"surge", "crash"}


def test_score_percentile_and_bucket_for_arbitrary_current_score():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=20, freq="D"),
            "ticker": ["7203.T"] * 20,
            "surge_risk_score": list(range(20)),
            "crash_risk_score": list(range(20)),
        }
    )

    assert score_percentile(frame, risk_type="surge", score=19) == 1.0
    assert score_percentile(frame, risk_type="crash", score=9) == 0.5
    assert bucket_for_score(frame, risk_type="surge", score=19, bucket_mode="quantile_60_90") == "High"
    assert bucket_for_score(frame, risk_type="surge", score=12, bucket_mode="quantile_60_90") == "Medium"
    assert bucket_for_score(frame, risk_type="surge", score=10, bucket_mode="fixed_33_66") == "Low"
