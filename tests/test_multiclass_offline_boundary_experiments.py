import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.evaluation.multiclass_offline_boundary_experiments import (
    BoundaryCandidate,
    BoundaryExperimentConfig,
    default_boundary_candidates,
    run_multiclass_offline_boundary_experiments,
    write_boundary_experiment_reports,
)


class CandidateAwareModel:
    def __init__(self, classes):
        self.classes_ = np.array(sorted(classes), dtype=int)

    def predict_proba(self, features):
        signal = pd.to_numeric(features.get("signal"), errors="coerce").fillna(0.0).to_numpy()
        rows = []
        for value in signal:
            weights = []
            for class_id in self.classes_:
                if class_id == 2:
                    weight = 0.40
                elif class_id in (3, 4) and value > 0.5:
                    weight = 0.35
                elif class_id in (0, 1) and value < -0.5:
                    weight = 0.35
                else:
                    weight = 0.12
                weights.append(weight)
            row = np.asarray(weights, dtype=float)
            rows.append(row / row.sum())
        return np.vstack(rows)


def _model_factory(train_data, valid_data, feature_columns, target_column, candidate):
    assert feature_columns
    assert target_column == "target_offline_multiclass"
    assert target_column not in feature_columns
    assert candidate.name
    return CandidateAwareModel(train_data[target_column].unique())


def _toy_features(rows: int = 420) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return_pattern = np.array([-0.055, -0.021, -0.007, -0.002, 0.002, 0.007, 0.021, 0.055])
    returns = np.resize(return_pattern, rows)
    close = 100 * np.cumprod(1 + returns)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "volume": 1_000_000 + np.arange(rows),
            "signal": np.resize([-1.0, -0.4, 0.0, 0.4, 1.0], rows),
            "ma_5": pd.Series(close).rolling(5, min_periods=1).mean(),
            "ma_25": pd.Series(close).rolling(25, min_periods=1).mean(),
            "regime": np.resize(["Bull Trend", "Bear Trend", "High Volatility"], rows),
            "anomaly_score": np.resize([5, 45, 80], rows),
            "anomaly_level": np.resize(["Low", "Medium", "High"], rows),
        }
    )


def _candidate_subset() -> list[BoundaryCandidate]:
    wanted = {"current_multiclass", "practical_3_0_5", "quantile_5class", "3class_down_flat_up"}
    return [candidate for candidate in default_boundary_candidates() if candidate.name in wanted]


def test_boundary_experiments_create_candidate_and_period_summary():
    result = run_multiclass_offline_boundary_experiments(
        _toy_features(),
        BoundaryExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=_candidate_subset(),
        model_factory=_model_factory,
    )

    assert {"current_multiclass", "practical_3_0_5", "quantile_5class"}.issubset(
        set(result.summary["candidate_name"])
    )
    assert "Historical all" in set(result.summary["period"])
    assert {"macro_f1", "predicted_flat_ratio", "direction_coverage"}.issubset(result.summary.columns)
    assert not result.class_metrics.empty
    assert not result.confusion_matrix.empty


def test_quantile_boundaries_are_recorded_per_fold_from_train_rows():
    result = run_multiclass_offline_boundary_experiments(
        _toy_features(),
        BoundaryExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=[candidate for candidate in default_boundary_candidates() if candidate.name == "quantile_5class"],
        model_factory=_model_factory,
    )

    details = result.fold_details
    assert not details.empty
    thresholds = json.loads(details["label_thresholds"].iloc[0])
    assert {"crash_threshold", "down_threshold", "up_threshold", "surge_threshold"}.issubset(
        thresholds
    )
    assert details["train_rows"].min() >= 40


def test_direction_and_flat_bias_metrics_are_available():
    result = run_multiclass_offline_boundary_experiments(
        _toy_features(),
        BoundaryExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=_candidate_subset(),
        model_factory=_model_factory,
    )

    direction_modes = set(result.direction_comparison["mode"])
    assert {"flat_excluded", "flat_as_neutral", "actual_direction_only"}.issubset(direction_modes)
    assert result.summary["predicted_flat_ratio"].notna().any()
    assert result.confidence_buckets["threshold"].isin([0.40, 0.45, 0.50, 0.55, 0.60]).all()


def test_three_class_candidate_has_tail_caveat():
    result = run_multiclass_offline_boundary_experiments(
        _toy_features(),
        BoundaryExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=[candidate for candidate in default_boundary_candidates() if candidate.name == "3class_down_flat_up"],
        model_factory=_model_factory,
    )

    row = result.summary[result.summary["candidate_name"].eq("3class_down_flat_up")].iloc[0]
    assert row["class_count"] == 3
    assert "tail recall is not directly comparable" in str(row["caveats"])


def test_reports_are_written_without_sqlite_side_effects(tmp_path):
    result = run_multiclass_offline_boundary_experiments(
        _toy_features(),
        BoundaryExperimentConfig(
            ticker="7203.T",
            train_years=1,
            test_step_months=6,
            min_train_rows=40,
            start_date="2021-01-01",
        ),
        candidates=_candidate_subset(),
        model_factory=_model_factory,
    )

    paths = write_boundary_experiment_reports(result, tmp_path, run_date="2026-06-24")

    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    assert not list(tmp_path.rglob("*.db"))
