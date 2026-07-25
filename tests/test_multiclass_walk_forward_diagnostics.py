import numpy as np
import pandas as pd

from stock_ai.evaluation.multiclass_walk_forward_diagnostics import (
    MulticlassWalkForwardConfig,
    binary_direction_comparison_frame,
    class_distribution_frame,
    confidence_bucket_frame,
    context_summary_frame,
    direction_collapsed_comparison_frame,
    multiclass_class_metrics_frame,
    multiclass_confusion_matrix_frame,
    multiclass_source_summary_frame,
    probability_summary_frame,
    run_multiclass_walk_forward_diagnostics,
    write_multiclass_walk_forward_reports,
)


class FlatBiasedModel:
    classes_ = np.array([0, 1, 2, 3, 4])

    def predict_proba(self, features):
        signal = pd.to_numeric(features.get("signal"), errors="coerce").fillna(0.0).to_numpy()
        probabilities = np.tile(np.array([0.10, 0.15, 0.42, 0.23, 0.10]), (len(features), 1))
        probabilities[signal > 0.8] = np.array([0.04, 0.08, 0.38, 0.35, 0.15])
        probabilities[signal < -0.8] = np.array([0.15, 0.34, 0.36, 0.10, 0.05])
        return probabilities / probabilities.sum(axis=1).reshape(-1, 1)


def _toy_features(rows: int = 520) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return_pattern = np.array([-0.06, -0.018, -0.004, 0.004, 0.018, 0.06, 0.0])
    returns = np.resize(return_pattern, rows)
    close = 100.0 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "open": close * 0.995,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000 + np.arange(rows),
            "signal": np.resize(np.array([-1.0, -0.4, 0.0, 0.4, 1.0]), rows),
            "ma_5": pd.Series(close).rolling(5, min_periods=1).mean(),
            "ma_25": pd.Series(close).rolling(25, min_periods=1).mean(),
            "regime": np.resize(["Bull Trend", "Bear Trend", "High Volatility", "Sideways"], rows),
            "anomaly_score": np.resize([5, 35, 70, 20], rows),
            "anomaly_level": np.resize(["low", "medium", "high", "low"], rows),
        }
    )


def _manual_multiclass_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-01", periods=6),
            "ticker": ["7203.T"] * 6,
            "target_close_date": pd.bdate_range("2024-01-02", periods=6),
            "prediction_type": ["historical_multiclass_walk_forward"] * 6,
            "status": ["confirmed"] * 6,
            "actual_class": [0, 1, 2, 3, 4, 3],
            "predicted_class": [2, 2, 2, 3, 2, 1],
            "actual_return": [-0.07, -0.02, 0.0, 0.02, 0.06, 0.015],
            "confidence": [0.44, 0.48, 0.51, 0.58, 0.54, 0.49],
            "probability_crash": [0.10, 0.12, 0.08, 0.04, 0.03, 0.08],
            "probability_down": [0.20, 0.20, 0.16, 0.10, 0.09, 0.49],
            "probability_flat": [0.44, 0.48, 0.51, 0.20, 0.54, 0.23],
            "probability_up": [0.18, 0.12, 0.15, 0.58, 0.19, 0.15],
            "probability_surge": [0.08, 0.08, 0.10, 0.08, 0.15, 0.05],
            "regime": ["Panic", "Bear Trend", "Sideways", "Bull Trend", "Panic", "Bull Trend"],
            "anomaly_level": ["high", "medium", "low", "low", "high", "low"],
        }
    )


def _model_factory(train_data, valid_data, feature_columns, target_column):
    assert target_column == "target_next_close_class"
    assert feature_columns
    assert len(train_data) > 0
    assert len(valid_data) > 0
    return FlatBiasedModel()


def test_historical_multiclass_summary_class_metrics_and_confusion_matrix():
    result = run_multiclass_walk_forward_diagnostics(
        _toy_features(),
        MulticlassWalkForwardConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=3,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        model_factory=_model_factory,
    )

    assert not result.predictions.empty
    assert "Historical all" in set(result.summary["period"])
    assert set(result.class_metrics["class"]) == {"crash", "down", "flat", "up", "surge"}
    historical_confusion = result.confusion_matrix[
        result.confusion_matrix["period"].eq("Historical all")
    ]
    assert int(historical_confusion["count"].sum()) == len(result.predictions)


def test_flat_bias_distribution_and_confidence_buckets_are_safe():
    frame = _manual_multiclass_frame()
    summary = multiclass_source_summary_frame({"historical": frame}, min_sample_warning=10)
    distribution = class_distribution_frame(historical_frame=frame)
    buckets = confidence_bucket_frame(frame, source="historical", period="Historical all")
    probabilities = probability_summary_frame(frame, source="historical", period="Historical all")

    historical = summary.iloc[0]
    assert historical["predicted_flat_ratio"] == 4 / 6
    assert historical["actual_up_down_predicted_flat_count"] == 3
    assert {"historical_actual", "historical_predicted"}.issubset(set(distribution["dataset"]))
    assert int(buckets.loc[buckets["threshold"] == 0.60, "sample_count"].iloc[0]) == 0
    assert {"flat", "confidence", "max_probability"}.issubset(set(probabilities["metric"]))


def test_direction_collapsed_and_binary_reference_metrics():
    multiclass = direction_collapsed_comparison_frame(
        _manual_multiclass_frame(),
        source="historical",
        period="Historical all",
    )
    binary = binary_direction_comparison_frame(
        pd.DataFrame(
            {
                "status": ["confirmed", "confirmed", "confirmed"],
                "actual_class": [1, 0, 1],
                "predicted_class": [1, 0, 0],
            }
        ),
        source="historical",
        period="Historical all",
    )

    flat_excluded = multiclass[multiclass["mode"].eq("flat_excluded")].iloc[0]
    assert flat_excluded["coverage"] < 1.0
    assert int(binary["sample_count"].iloc[0]) == 3
    assert binary["model"].iloc[0] == "binary"


def test_live_and_historical_frames_are_separated():
    historical = _manual_multiclass_frame()
    live = historical.head(2).copy()
    live["prediction_type"] = "live"
    result = run_multiclass_walk_forward_diagnostics(
        _toy_features(),
        MulticlassWalkForwardConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        live_frame=live,
        model_factory=_model_factory,
    )

    assert {"historical", "live", "historical + live"}.issubset(set(result.summary["source"]))
    assert "live_actual" in set(result.class_distribution["dataset"])


def test_report_writing_does_not_create_sqlite_rows(tmp_path):
    result = run_multiclass_walk_forward_diagnostics(
        _toy_features(),
        MulticlassWalkForwardConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        model_factory=_model_factory,
    )

    paths = write_multiclass_walk_forward_reports(result, tmp_path, run_date="2026-06-23")

    assert set(paths) == {"csv", "json"}
    assert (tmp_path / "multiclass_walk_forward_diagnostics_2026-06-23.csv").exists()
    assert not list(tmp_path.rglob("*.db"))


def test_class_metrics_and_confusion_helpers_accept_manual_frames():
    frame = _manual_multiclass_frame()

    class_metrics = multiclass_class_metrics_frame(
        frame,
        source="historical",
        period="Historical all",
    )
    confusion = multiclass_confusion_matrix_frame(
        frame,
        source="historical",
        period="Historical all",
    )

    assert set(class_metrics["class"]) == {"crash", "down", "flat", "up", "surge"}
    assert int(confusion["count"].sum()) == len(frame)


def test_regime_and_anomaly_context_summary_is_calculated():
    context = context_summary_frame(
        _manual_multiclass_frame(),
        source="historical",
        period="Historical all",
    )

    assert {"regime", "anomaly"}.issubset(set(context["context_type"]))
    assert {"accuracy", "macro_f1", "predicted_flat_ratio"}.issubset(context.columns)
