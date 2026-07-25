import pandas as pd

from stock_ai.evaluation.offline_model_experiments import (
    CandidateConfig,
    OfflineExperimentConfig,
    _best_threshold,
    _binary_metrics,
    _filter_feature_group,
    classify_experiment_conclusions,
    run_offline_model_experiments,
    write_offline_experiment_reports,
)


def test_binary_metrics_include_balanced_accuracy_and_recalls():
    actual = pd.Series([1, 1, 0, 0])
    predicted = pd.Series([1, 0, 0, 0])

    metrics = _binary_metrics(actual, predicted)

    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert metrics["up_recall"] == 0.5
    assert metrics["down_recall"] == 1.0
    assert metrics["up_precision"] == 1.0
    assert metrics["down_precision"] == 2 / 3


def test_threshold_is_selected_from_validation_inputs_only():
    validation_actual = pd.Series([1, 1, 0, 0])
    validation_probability = pd.Series([0.45, 0.55, 0.40, 0.60])

    threshold = _best_threshold(
        validation_actual,
        validation_probability,
        metric="balanced_accuracy",
    )

    assert 0.35 <= threshold <= 0.65
    assert threshold != 0.5


def test_feature_group_selection_does_not_require_feature_generation_changes():
    columns = [
        "close",
        "ma_5",
        "usdjpy_return_1d",
        "sector_mean_return",
        "news_count_1d",
        "avg_sentiment_3d",
        "days_to_earnings",
        "dominant_narrative_code",
    ]

    assert "usdjpy_return_1d" not in _filter_feature_group(columns, "no_external_sector_features")
    assert "news_count_1d" not in _filter_feature_group(columns, "no_news_features")
    technical = _filter_feature_group(columns, "technical_price_only")
    assert "close" in technical
    assert "ma_5" in technical
    assert "news_count_1d" not in technical
    assert _filter_feature_group(columns, "external_market_only") == [
        "usdjpy_return_1d",
        "sector_mean_return",
    ]


def test_run_offline_model_experiments_builds_candidate_summary_and_report(tmp_path):
    data = _toy_frame(420)
    config = OfflineExperimentConfig(
        ticker="7203.T",
        train_years=1,
        test_step_months=1,
        min_train_rows=30,
        valid_fraction=0.25,
    )
    candidates = [
        CandidateConfig("current_model"),
        CandidateConfig("balanced_accuracy_optimized_threshold", threshold_mode="balanced_accuracy"),
    ]

    result = run_offline_model_experiments(data, config=config, candidates=candidates)
    paths = write_offline_experiment_reports(result, tmp_path, run_date="2026-06-23")

    assert set(result.summary["candidate_name"]) == {
        "current_model",
        "balanced_accuracy_optimized_threshold",
    }
    assert {"accuracy", "balanced_accuracy", "up_recall", "down_recall"}.issubset(result.summary.columns)
    assert result.summary["sample_count"].sum() > 0
    assert paths["csv"].endswith("offline_model_experiments_2026-06-23.csv")
    assert paths["json"].endswith("offline_model_experiments_2026-06-23.json")


def test_classify_experiment_conclusions_marks_not_ready_for_weak_results():
    summary = pd.DataFrame(
        [
            {
                "candidate_name": "current_model",
                "balanced_accuracy": 0.50,
                "up_recall": 0.30,
                "confidence_mean": 0.52,
                "model_minus_always_up": 0.0,
                "model_minus_always_down": 0.0,
                "model_minus_previous_day_direction": 0.0,
                "model_minus_moving_average_trend_only": 0.0,
                "threshold_mode": "fixed_0_50",
            }
        ]
    )

    conclusions = classify_experiment_conclusions(summary)

    assert "no_candidate_beats_baseline" in conclusions
    assert "probability_calibration_needed" in conclusions
    assert "production_change_not_ready" in conclusions


def _toy_frame(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    index = range(rows)
    close = pd.Series([100 + i * 0.1 + (1 if i % 7 == 0 else -0.5) for i in index])
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": [1_000_000 + i * 100 for i in index],
            "return_1d": close.pct_change().fillna(0.0),
            "ma_5": close.rolling(5, min_periods=1).mean(),
            "ma_25": close.rolling(25, min_periods=1).mean(),
            "close_vs_ma_5": close / close.rolling(5, min_periods=1).mean() - 1,
            "usdjpy_return_1d": [0.001 if i % 3 == 0 else -0.001 for i in index],
            "sector_mean_return": [0.002 if i % 4 == 0 else -0.001 for i in index],
            "news_count_1d": [1 if i % 11 == 0 else 0 for i in index],
            "avg_sentiment_3d": [0.2 if i % 13 == 0 else 0.0 for i in index],
            "regime": ["uptrend" if i % 2 == 0 else "range" for i in index],
        }
    )
