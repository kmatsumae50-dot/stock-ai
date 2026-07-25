import pandas as pd

from stock_ai.evaluation.today_signal_validation import (
    TodaySignalRuleConfig,
    build_today_signal_validation_frame,
    build_config_adoption_recommendations,
    build_signal_improvement_suggestions,
    compare_today_signal_configs,
    current_signal_context_text,
    current_signal_improvement_text,
    describe_today_signal_config,
    load_today_signal_experiments,
    load_today_signal_config,
    summarize_today_signals,
)


def test_today_signal_validation_frame_and_summary():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    binary = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 6,
            "prediction_direction": ["up", "down", "up", "up", "down", "up"],
            "predicted_class": [1, 0, 1, 1, 0, 1],
            "probability_up": [0.62, 0.30, 0.51, 0.64, 0.40, 0.49],
            "probability_down": [0.38, 0.70, 0.49, 0.36, 0.60, 0.51],
        }
    )
    risk = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 6,
            "surge_risk_score": [80, 20, 20, 85, 20, 80],
            "crash_risk_score": [20, 90, 90, 85, 20, 80],
            "return_1d": [0.01, -0.02, -0.01, 0.02, -0.01, 0.0],
            "return_5d": [0.03, -0.04, -0.02, 0.04, -0.02, 0.0],
            "return_20d": [0.05, -0.06, -0.03, 0.06, -0.03, 0.01],
            "future_min_return_20d": [-0.01, -0.08, -0.05, -0.02, -0.04, -0.03],
            "future_max_return_20d": [0.07, 0.02, 0.03, 0.08, 0.01, 0.04],
        }
    )
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 6,
            "regime": ["Bull Trend", "High Volatility", "High Volatility", "High Volatility", "Sideways", "High Volatility"],
            "anomaly_score": [10, 20, 20, 30, 10, 55],
        }
    )

    frame = build_today_signal_validation_frame(
        binary,
        risk,
        data=data,
        bucket_mode="fixed_33_66",
    )
    summary = summarize_today_signals(frame)

    assert set(frame["today_signal"]) >= {"Bullish", "High Risk", "Caution", "Watch Only"}
    assert summary.loc[summary["today_signal"] == "Bullish", "count"].iloc[0] >= 1
    assert summary.loc[summary["today_signal"] == "High Risk", "average_return_5d"].iloc[0] < 0
    assert "Historical High Risk validation" in current_signal_context_text(summary, "High Risk")


def test_signal_improvement_suggestions_detect_common_issues():
    summary = pd.DataFrame(
        {
            "today_signal": ["Bullish", "Bearish", "Neutral", "Caution", "High Risk", "Watch Only"],
            "count": [40, 10, 130, 20, 35, 15],
            "average_return_1d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "average_return_5d": [-0.01, 0.01, 0.0, 0.0, 0.01, 0.0],
            "average_return_20d": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "up_rate_1d": [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            "up_rate_5d": [0.48, 0.52, 0.50, 0.50, 0.50, 0.50],
            "max_drawdown_20d": [-0.05, -0.05, -0.05, -0.05, -0.01, -0.05],
            "max_upside_20d": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        }
    )

    suggestions = build_signal_improvement_suggestions(summary, min_samples=20)

    bullish = suggestions[suggestions["today_signal"] == "Bullish"].iloc[0]
    high_risk = suggestions[suggestions["today_signal"] == "High Risk"].iloc[0]
    neutral = suggestions[suggestions["today_signal"] == "Neutral"].iloc[0]
    bearish = suggestions[suggestions["today_signal"] == "Bearish"].iloc[0]

    assert bullish["priority"] == "High"
    assert "too loose" in bullish["suggestion"]
    assert "bullish_up_probability_threshold" in bullish["config_keys"]
    assert high_risk["priority"] == "High"
    assert "oversensitive" in high_risk["suggestion"]
    assert "high_crash_percentile_threshold" in high_risk["config_keys"]
    assert neutral["priority"] == "Medium"
    assert "conservative" in neutral["suggestion"]
    assert neutral["config_keys"] == "neutral_probability_gap_threshold"
    assert bearish["priority"] == "Low"
    assert "Sample size" in bearish["issue"]
    assert "Signal improvement note" in current_signal_improvement_text(suggestions, "Bullish")


def test_today_signal_config_loads_yaml_and_falls_back(tmp_path):
    config_path = tmp_path / "today_signal.yaml"
    config_path.write_text(
        "\n".join(
            [
                "rules:",
                "  bullish_up_probability_threshold: 0.62",
                "  bearish_down_probability_threshold: 0.63",
                "  neutral_probability_gap_threshold: 0.05",
            ]
        ),
        encoding="utf-8",
    )

    config = load_today_signal_config(config_path)
    description = describe_today_signal_config(config_path)
    missing_config = load_today_signal_config(tmp_path / "missing.yaml")

    assert config.bullish_up_probability_threshold == 0.62
    assert config.bearish_down_probability_threshold == 0.63
    assert config.neutral_probability_gap_threshold == 0.05
    assert config.high_crash_percentile_threshold == 0.90
    assert description["status"] == "configured"
    assert missing_config.bullish_up_probability_threshold == 0.58


def test_today_signal_config_experiments_compare_candidates(tmp_path):
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    binary = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 8,
            "prediction_direction": ["up", "up", "down", "down", "up", "down", "up", "down"],
            "predicted_class": [1, 1, 0, 0, 1, 0, 1, 0],
            "probability_up": [0.59, 0.53, 0.40, 0.47, 0.61, 0.44, 0.52, 0.35],
            "probability_down": [0.41, 0.47, 0.60, 0.53, 0.39, 0.56, 0.48, 0.65],
        }
    )
    risk = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * 8,
            "surge_risk_score": [70, 45, 20, 15, 80, 20, 35, 10],
            "crash_risk_score": [20, 30, 75, 55, 80, 70, 35, 85],
            "return_1d": [0.01, 0.00, -0.02, -0.01, 0.03, -0.02, 0.01, -0.03],
            "return_5d": [0.03, 0.01, -0.04, -0.02, 0.04, -0.03, 0.01, -0.05],
            "return_20d": [0.05, 0.02, -0.08, -0.04, 0.06, -0.05, 0.02, -0.09],
            "future_min_return_20d": [-0.01, -0.01, -0.09, -0.05, -0.02, -0.07, -0.02, -0.10],
            "future_max_return_20d": [0.06, 0.04, 0.02, 0.03, 0.08, 0.02, 0.05, 0.01],
        }
    )
    experiments_path = tmp_path / "today_signal_experiments.yaml"
    experiments_path.write_text(
        "\n".join(
            [
                "experiments:",
                "  current: {}",
                "  aggressive:",
                "    bullish_up_probability_threshold: 0.52",
                "    bearish_down_probability_threshold: 0.52",
                "    high_crash_percentile_threshold: 0.80",
                "    high_surge_percentile_threshold: 0.80",
            ]
        ),
        encoding="utf-8",
    )

    experiments = load_today_signal_experiments(
        experiments_path,
        current_config=TodaySignalRuleConfig(),
    )
    comparison, distribution, changes, recommendation = compare_today_signal_configs(
        binary,
        risk,
        bucket_mode="quantile_60_90",
        experiments=experiments,
    )
    adoption = build_config_adoption_recommendations(comparison)

    assert set(comparison["config_name"]) == {"current", "aggressive"}
    assert set(distribution["config_name"]) == {"current", "aggressive"}
    assert "aggressive" in set(changes["config_name"])
    assert recommendation["config_name"] in {"current", "aggressive"}
    assert comparison["validation_rows"].min() == len(binary)
    assert set(adoption["config_name"]) == {"current", "aggressive"}
    assert set(adoption["adoption_rating"]).issubset(
        {
            "Recommended",
            "Candidate",
            "Too Conservative",
            "Too Aggressive",
            "Insufficient Samples",
            "Not Recommended",
        }
    )
    assert adoption["adoption_reason"].astype(str).str.len().min() > 0
