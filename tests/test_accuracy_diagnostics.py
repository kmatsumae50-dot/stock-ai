import pandas as pd

from stock_ai.evaluation.accuracy_diagnostics import (
    anomaly_accuracy_frame,
    baseline_accuracy_frame,
    binary_class_performance_frame,
    build_binary_diagnostic_frame,
    build_multiclass_diagnostic_frame,
    build_walk_forward_binary_diagnostic_frame,
    confidence_distribution_frame,
    confidence_threshold_frame,
    multiclass_flat_summary_frame,
    multiclass_confusion_matrix,
    multiclass_performance_frame,
    period_summary_frame,
    regime_accuracy_frame,
    source_comparison_frame,
    source_breakdown_frame,
    summarize_evaluation_targets,
)


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "ticker": ["7203.T"] * 6,
            "close": [100.0, 102.0, 101.0, 104.0, 100.0, 103.0],
            "ma_5": [100.0, 101.0, 101.0, 102.0, 101.5, 102.0],
            "ma_25": [100.0, 100.5, 100.8, 101.0, 101.2, 101.4],
            "regime": ["Bull Trend", "Bull Trend", "Bear Trend", "Bear Trend", "High Volatility", "Range"],
            "anomaly_score": [5, 12, 35, 70, 25, 10],
        }
    )


def _binary_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "ticker": ["7203.T"] * 6,
            "target": ["next_close_binary"] * 6,
            "model_name": ["lightgbm_binary"] * 6,
            "prediction_type": ["live", "live", "backfill", "live", "backfill", "live"],
            "predicted_class": [1, 1, 0, 0, 1, 1],
            "confidence": [0.52, 0.56, 0.61, 0.66, 0.72, 0.80],
            "probability_up": [0.52, 0.56, 0.39, 0.34, 0.72, 0.80],
            "probability_down": [0.48, 0.44, 0.61, 0.66, 0.28, 0.20],
            "top_features": ['[{"feature": "return_1d"}]'] * 6,
        }
    )


def test_baseline_accuracy_and_unconfirmed_rows_are_handled():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    baselines = baseline_accuracy_frame(frame)
    summary = summarize_evaluation_targets(frame, pd.DataFrame())

    assert summary.binary_rows == 6
    assert summary.binary_confirmed_rows == 5
    assert summary.binary_pending_rows == 1
    assert set(baselines["baseline"]) >= {"always_up", "always_down", "random_baseline"}
    assert baselines["sample_count"].min() > 0


def test_confidence_thresholds_handle_sparse_buckets():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    confidence = confidence_threshold_frame(frame)

    assert set(confidence["threshold"]) >= {0.5, 0.7}
    high = confidence[confidence["threshold"] == 0.7].iloc[0]
    assert high["sample_count"] == 1
    assert high["coverage"] == 0.2


def test_regime_and_anomaly_accuracy_are_calculated():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    regime = regime_accuracy_frame(frame)
    anomaly = anomaly_accuracy_frame(frame)

    assert "Bull Trend" in set(regime["regime"])
    assert {"low", "medium", "high"}.issubset(set(anomaly["anomaly_level"]))
    assert "up_accuracy" in regime.columns
    assert "down_accuracy" in anomaly.columns


def test_binary_class_precision_recall_style_rows_are_calculated():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    class_rows = binary_class_performance_frame(frame)

    assert set(class_rows["metric"]) == {"predicted_class_accuracy", "actual_class_recall"}
    assert set(class_rows["class"]) == {"up", "down"}
    assert class_rows["sample_count"].sum() > 0


def test_multiclass_performance_and_confusion_matrix_are_calculated():
    predictions = _binary_predictions().copy()
    predictions["target"] = "next_close_multiclass"
    predictions["model_name"] = "lightgbm_multiclass"
    predictions["predicted_class"] = [3, 1, 3, 0, 4, 2]
    frame = build_multiclass_diagnostic_frame(predictions, _features())

    perf = multiclass_performance_frame(frame)
    confusion = multiclass_confusion_matrix(frame)

    assert set(perf["class"]) == {"crash", "down", "flat", "up", "surge"}
    assert {"precision", "recall", "support"}.issubset(perf.columns)
    assert confusion.loc["up"].sum() >= 0
    assert int(confusion.to_numpy().sum()) == 5


def test_live_and_backfill_can_be_separated():
    all_frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    live_frame = build_binary_diagnostic_frame(_binary_predictions(), _features(), prediction_type="live")
    source = source_breakdown_frame(all_frame)

    assert len(live_frame) == 4
    assert set(source["source"]) == {"live", "backfill"}
    assert int(source.loc[source["source"] == "backfill", "rows"].iloc[0]) == 2


def test_walk_forward_predictions_can_use_accuracy_diagnostics():
    predictions = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "ticker": ["7203.T"] * 5,
            "predicted_class": [1, 0, 1, 0, 1],
            "actual_class": [1, 0, 0, 1, 1],
            "probability_up": [0.7, 0.4, 0.6, 0.45, 0.8],
            "probability_down": [0.3, 0.6, 0.4, 0.55, 0.2],
            "confidence": [0.7, 0.6, 0.6, 0.55, 0.8],
            "next_close_return": [0.01, -0.01, -0.02, 0.03, 0.02],
        }
    )

    frame = build_walk_forward_binary_diagnostic_frame(predictions, _features())
    baselines = baseline_accuracy_frame(frame)

    assert set(frame["prediction_type"]) == {"historical_walk_forward"}
    assert int((frame["status"] == "confirmed").sum()) == 5
    assert "previous_day_direction" in set(baselines["baseline"])


def test_source_comparison_includes_balanced_accuracy_and_recalls():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    live = build_binary_diagnostic_frame(_binary_predictions(), _features(), prediction_type="live")

    comparison = source_comparison_frame({"all": frame, "live": live}, min_sample_warning=10)

    assert set(comparison["source"]) == {"all", "live"}
    assert {"balanced_accuracy", "up_recall", "down_recall"}.issubset(comparison.columns)
    assert set(comparison["warning"]) == {"low sample"}


def test_period_summary_handles_sparse_periods():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())

    summary = period_summary_frame(frame, reference_date="2024-01-06", min_sample_warning=10)

    assert {"2024-", "Last 3 months", "Last 6 months"}.issubset(set(summary["period"]))
    assert "always_down" in summary.columns
    assert summary["sample_count"].max() <= 5


def test_confidence_distribution_and_multiclass_flat_summary():
    frame = build_binary_diagnostic_frame(_binary_predictions(), _features())
    distribution = confidence_distribution_frame(frame)

    predictions = _binary_predictions().copy()
    predictions["target"] = "next_close_multiclass"
    predictions["model_name"] = "lightgbm_multiclass"
    predictions["predicted_class"] = [2, 2, 3, 2, 1, 2]
    multiclass_frame = build_multiclass_diagnostic_frame(predictions, _features())
    flat = multiclass_flat_summary_frame(multiclass_frame)

    assert set(distribution["metric"]) == {"probability_up", "probability_down", "confidence"}
    assert int(distribution.loc[distribution["metric"] == "confidence", "count_ge_55"].iloc[0]) >= 1
    assert int(flat["flat_count"].iloc[0]) >= 1
    assert flat["flat_rate"].iloc[0] is not None
