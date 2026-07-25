import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.evaluation.binary_up_detection_experiments import (
    BinaryUpCandidate,
    BinaryUpExperimentConfig,
    _best_threshold,
    _confidence_row,
    _summary_row,
    default_up_detection_candidates,
    run_binary_up_detection_experiments,
    write_binary_up_detection_reports,
)


class SignalModel:
    def predict_proba(self, features):
        signal = pd.to_numeric(features.get("signal"), errors="coerce").fillna(0.0).to_numpy()
        probability_up = np.clip(0.50 + signal * 0.18, 0.05, 0.95)
        return np.column_stack([1.0 - probability_up, probability_up])


def _model_factory(train_data, valid_data, feature_columns, target_column, candidate):
    assert len(train_data) > len(valid_data)
    assert target_column == "target_next_close_binary"
    assert target_column not in feature_columns
    assert candidate.name
    return SignalModel()


def test_up_detection_metrics_include_balanced_accuracy_and_recalls():
    rows = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=4),
            "ticker": ["7203.T"] * 4,
            "actual_class": [1, 1, 0, 0],
            "predicted_class": [1, 0, 0, 0],
            "confidence": [0.7, 0.6, 0.8, 0.9],
            "baseline_previous_day_direction": [1, 1, 0, 1],
            "baseline_moving_average_trend": [1, 0, 0, 0],
        }
    )

    row = _summary_row(rows, "candidate", "Historical all", BinaryUpExperimentConfig(min_sample_warning=1))

    assert row["accuracy"] == 0.75
    assert row["balanced_accuracy"] == 0.75
    assert row["up_recall"] == 0.5
    assert row["down_recall"] == 1.0
    assert row["predicted_up_ratio"] == 0.25


def test_threshold_is_selected_from_validation_inputs_only():
    validation_actual = pd.Series([1, 1, 0, 0])
    validation_probability = pd.Series([0.44, 0.58, 0.42, 0.62])
    test_actual_that_must_not_be_used = pd.Series([1, 0, 1, 0])

    threshold = _best_threshold(validation_actual, validation_probability, metric="balanced_accuracy")

    assert 0.35 <= threshold <= 0.65
    assert len(test_actual_that_must_not_be_used) == 4
    assert threshold != 0.5


def test_confidence_bucket_counts_hits_and_misses():
    rows = pd.DataFrame(
        {
            "confidence": [0.56, 0.70, 0.52],
            "is_correct": [True, False, True],
        }
    )

    result = _confidence_row("candidate", "Historical all", 0.55, rows)

    assert result["sample_count"] == 2
    assert result["hit_count"] == 1
    assert result["miss_count"] == 1
    assert result["hit_rate"] == 0.5


def test_run_binary_up_detection_experiments_builds_periods_constraints_and_hybrid_rows(tmp_path):
    result = run_binary_up_detection_experiments(
        _toy_features(),
        config=BinaryUpExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
            min_sample_warning=10,
        ),
        candidates=[
            BinaryUpCandidate("current_model"),
            BinaryUpCandidate(
                "up_recall_mild_threshold",
                threshold_mode="up_recall_mild",
            ),
            BinaryUpCandidate(
                "previous_day_direction_hybrid",
                hybrid_mode="previous_day_when_weak",
            ),
        ],
        model_factory=_model_factory,
    )

    paths = write_binary_up_detection_reports(result, tmp_path, run_date="2026-07-08")

    assert {"current_model", "up_recall_mild_threshold", "previous_day_direction_hybrid"}.issubset(
        set(result.summary["candidate_name"])
    )
    assert "Historical all" in set(result.summary["period"])
    assert {"accuracy", "balanced_accuracy", "up_recall", "down_recall"}.issubset(result.summary.columns)
    assert result.summary["constraint_passed"].isin([True, False]).all()
    assert result.confidence_buckets["threshold"].isin([0.55, 0.60, 0.65]).all()
    assert result.fold_details["threshold_source"].eq("validation_only").all()
    assert (result.predictions["hybrid_mode"] == "previous_day_when_weak").any()
    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    assert Path(paths["md"]).exists()
    assert not list(tmp_path.rglob("*.db"))


def test_report_json_contains_failure_pattern_sections(tmp_path):
    live = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-06-01", periods=3),
            "ticker": ["7203.T"] * 3,
            "target": ["next_close_binary"] * 3,
            "model_name": ["lightgbm_binary"] * 3,
            "prediction_type": ["live"] * 3,
            "predicted_class": [0, 1, 0],
            "actual_class": [1, 0, 0],
            "actual_return": [0.01, -0.02, -0.01],
            "status": ["confirmed"] * 3,
            "is_correct": [False, False, True],
            "confidence": [0.56, 0.57, 0.53],
            "probability_up": [0.44, 0.57, 0.47],
            "probability_down": [0.56, 0.43, 0.53],
            "return_1d": [-0.01, 0.01, -0.01],
            "ma_5": [100, 101, 100],
            "ma_25": [101, 100, 101],
        }
    )
    result = run_binary_up_detection_experiments(
        _toy_features(),
        config=BinaryUpExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=12,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=[BinaryUpCandidate("current_model")],
        live_frame=live,
        model_factory=_model_factory,
    )

    paths = write_binary_up_detection_reports(result, tmp_path, run_date="2026-07-08")
    payload = json.loads(Path(paths["json"]).read_text(encoding="utf-8"))

    assert payload["failure_patterns"]
    assert payload["failure_common_features"]
    assert "current_live_saved" in set(result.summary["candidate_name"])


def test_default_candidate_set_contains_requested_safe_groups():
    names = {candidate.name for candidate in default_up_detection_candidates()}

    assert "class_weight_up_boost_mild" in names
    assert "balanced_accuracy_optimized_threshold" in names
    assert "previous_day_direction_hybrid" in names
    assert "technical_only" in names
    assert "news_excluded" in names


def _toy_features(rows: int = 460) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    index = np.arange(rows)
    signal = np.sin(index / 7)
    returns = np.where(signal > 0.15, 0.006, -0.005)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + index,
            "signal": signal,
            "return_1d": pd.Series(close).pct_change().fillna(0.0),
            "ma_5": pd.Series(close).rolling(5, min_periods=1).mean(),
            "ma_25": pd.Series(close).rolling(25, min_periods=1).mean(),
            "rsi_14": np.resize([35, 45, 55, 65], rows),
            "usdjpy_return_1d": np.resize([0.001, -0.001], rows),
            "sector_mean_return": np.resize([0.002, -0.001, 0.0], rows),
            "news_count_1d": np.resize([0, 1, 0, 0], rows),
            "avg_sentiment_3d": np.resize([0.1, -0.1, 0.0], rows),
            "regime": np.resize(["Bull Trend", "Bear Trend", "Range"], rows),
            "anomaly_score": np.resize([5, 45, 80], rows),
            "anomaly_level": np.resize(["Low", "Medium", "High"], rows),
        }
    )
