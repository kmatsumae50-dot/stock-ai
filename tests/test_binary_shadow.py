from pathlib import Path

import pandas as pd

from stock_ai.evaluation import binary_shadow
from stock_ai.evaluation.binary_shadow import (
    build_binary_shadow_predictions,
    compare_shadow_with_live,
    evaluate_shadow_predictions,
    save_shadow_predictions,
    summarize_shadow_lifecycle,
    summarize_shadow_predictions,
)


class StaticModel:
    def __init__(self, probability_up: float):
        self.probability_up = probability_up

    def predict_proba(self, features):
        return [[1.0 - self.probability_up, self.probability_up] for _ in range(len(features))]


def test_shadow_predict_saves_csv_json_without_sqlite_and_replaces_duplicate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        binary_shadow,
        "load_model_bundle",
        lambda _path: {"model": StaticModel(0.40), "feature_columns": ["signal"]},
    )
    monkeypatch.setattr(
        binary_shadow,
        "_train_balanced_binary_model",
        lambda _train, _valid: (StaticModel(0.62), ["signal"]),
    )
    data = _features(12)

    predictions = build_binary_shadow_predictions(
        data,
        ticker="7203.T",
        production_model_path=tmp_path / "unused.joblib",
        run_timestamp="2026-07-08T10:00:00",
    )
    paths = save_shadow_predictions(predictions, tmp_path)
    updated = build_binary_shadow_predictions(
        data,
        ticker="7203.T",
        production_model_path=tmp_path / "unused.joblib",
        run_timestamp="2026-07-08T11:00:00",
    )
    save_shadow_predictions(updated, tmp_path)
    rows = pd.read_csv(paths["csv"])

    assert Path(paths["csv"]).exists()
    assert Path(paths["json"]).exists()
    assert not list(tmp_path.rglob("*.db"))
    assert set(rows["candidate_name"]) == {"production_current", "class_weight_balanced"}
    assert len(rows) == 2
    assert rows["source"].eq("shadow").all()
    assert rows["run_timestamp"].eq("2026-07-08T11:00:00").all()


def test_shadow_evaluate_updates_pending_to_hit_or_miss(tmp_path, monkeypatch):
    shadow = pd.DataFrame(
        [
            {
                "run_timestamp": "2026-07-08T10:00:00",
                "ticker": "7203.T",
                "feature_date": "2026-07-07",
                "target_close_date": "2026-07-08",
                "candidate_name": "production_current",
                "predicted_direction": "up",
                "probability_up": 0.6,
                "probability_down": 0.4,
                "confidence": 0.6,
                "source": "shadow",
                "model_note": "test",
                "actual_return": "",
                "actual_direction": "",
                "result": "pending",
                "evaluated_at": "",
                "caveat": "",
            }
        ]
    )
    shadow_path = tmp_path / "binary_shadow_predictions.csv"
    shadow.to_csv(shadow_path, index=False)
    result = evaluate_shadow_predictions(
        shadow_path,
        _features(12),
        ticker="7203.T",
        output_dir=tmp_path,
        live_predictions=pd.DataFrame(),
        run_date="2026-07-08",
    )

    row = result.predictions.iloc[0]
    assert row["result"] in {"hit", "miss"}
    assert row["actual_direction"] in {"up", "down"}
    assert result.summary["confirmed_count"].iloc[0] == 1
    assert Path(result.saved_paths["md"]).exists()


def test_shadow_summary_metrics_and_confidence_bucket():
    rows = pd.DataFrame(
        {
            "candidate_name": ["production_current"] * 4,
            "feature_date": pd.date_range("2026-07-01", periods=4).strftime("%Y-%m-%d"),
            "predicted_direction": ["up", "up", "down", "down"],
            "actual_direction": ["up", "down", "down", "up"],
            "result": ["hit", "miss", "hit", "miss"],
            "confidence": [0.6, 0.7, 0.52, 0.51],
        }
    )

    summary = summarize_shadow_predictions(rows)

    assert summary["confirmed_count"].iloc[0] == 4
    assert summary["accuracy"].iloc[0] == 0.5
    assert summary["balanced_accuracy"].iloc[0] == 0.5
    assert summary["up_recall"].iloc[0] == 0.5
    assert summary["down_recall"].iloc[0] == 0.5
    assert summary["confidence_ge_55_count"].iloc[0] == 2
    assert summary["confidence_ge_55_hit_count"].iloc[0] == 1


def test_shadow_can_compare_with_existing_live_prediction():
    shadow = pd.DataFrame(
        {
            "feature_date": ["2026-07-08"],
            "candidate_name": ["class_weight_balanced"],
            "predicted_direction": ["up"],
            "confidence": [0.61],
            "result": ["pending"],
        }
    )
    live = pd.DataFrame(
        {
            "date": ["2026-07-08"],
            "target": ["next_close_binary"],
            "predicted_class": [0],
            "confidence": [0.55],
        }
    )

    comparison = compare_shadow_with_live(shadow, live)

    assert comparison["production_live_direction"].iloc[0] == "down"
    assert not bool(comparison["direction_matches_live"].iloc[0])


def test_target_actual_missing_stays_pending(tmp_path):
    shadow = pd.DataFrame(
        [
            {
                "run_timestamp": "2026-07-08T10:00:00",
                "ticker": "7203.T",
                "feature_date": "2026-07-09",
                "target_close_date": "2026-07-10",
                "candidate_name": "class_weight_balanced",
                "predicted_direction": "up",
                "probability_up": 0.6,
                "probability_down": 0.4,
                "confidence": 0.6,
                "source": "shadow",
                "model_note": "test",
                "actual_return": "",
                "actual_direction": "",
                "result": "pending",
                "evaluated_at": "",
                "caveat": "",
            }
        ]
    )
    shadow_path = tmp_path / "binary_shadow_predictions.csv"
    shadow.to_csv(shadow_path, index=False)
    result = evaluate_shadow_predictions(
        shadow_path,
        _features(12),
        ticker="7203.T",
        output_dir=tmp_path,
        live_predictions=pd.DataFrame(),
        run_date="2026-07-08",
    )

    assert result.predictions["result"].iloc[0] == "pending"
    assert result.summary["pending_count"].iloc[0] == 1


def test_shadow_lifecycle_distinguishes_ready_pending_and_generation_eligibility():
    feature_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09", "2026-07-10"]),
            "ticker": ["7203.T"] * 3,
            "close": [2500.0, 2525.0, 2510.0],
        }
    )
    predictions = pd.DataFrame(
        [_shadow_row_for_test("production_current"), _shadow_row_for_test("class_weight_balanced")]
    )

    summary = summarize_shadow_lifecycle(predictions, feature_data, ticker="7203.T")

    assert summary["total_rows"] == 2
    assert summary["pending_count"] == 2
    assert summary["confirmed_count"] == 0
    assert summary["rows_ready_for_evaluation"] == 2
    assert summary["stale_pending_count"] == 2
    assert summary["duplicate_count"] == 0
    assert summary["next_generation_eligible"] is True
    assert summary["generation_mode"] == "manual_opt_in"
    assert summary["sqlite_saved"] is False


def test_shadow_evaluation_dry_run_does_not_persist(tmp_path):
    feature_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09"]),
            "ticker": ["7203.T", "7203.T"],
            "close": [2500.0, 2525.0],
        }
    )
    shadow_path = tmp_path / "input.csv"
    pd.DataFrame([_shadow_row_for_test("production_current")]).to_csv(shadow_path, index=False)
    before = shadow_path.read_bytes()

    result = evaluate_shadow_predictions(
        shadow_path,
        feature_data,
        ticker="7203.T",
        output_dir=tmp_path / "output",
        persist=False,
    )

    assert result.predictions.iloc[0]["result"] == "hit"
    assert result.saved_paths == {}
    assert shadow_path.read_bytes() == before
    assert not (tmp_path / "output").exists()


def test_shadow_evaluation_is_idempotent_and_preserves_other_tickers(tmp_path):
    feature_data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-08", "2026-07-09"]),
            "ticker": ["7203.T", "7203.T"],
            "close": [2500.0, 2525.0],
        }
    )
    toyota = _shadow_row_for_test("production_current")
    other = _shadow_row_for_test("production_current")
    other["ticker"] = "9999.T"
    shadow_path = tmp_path / "binary_shadow_predictions.csv"
    pd.DataFrame([toyota, other]).to_csv(shadow_path, index=False)

    first = evaluate_shadow_predictions(shadow_path, feature_data, ticker="7203.T", output_dir=tmp_path)
    evaluated_at = first.predictions.iloc[0]["evaluated_at"]
    second = evaluate_shadow_predictions(shadow_path, feature_data, ticker="7203.T", output_dir=tmp_path)
    persisted = pd.read_csv(shadow_path)

    assert second.predictions.iloc[0]["evaluated_at"] == evaluated_at
    assert set(persisted["ticker"]) == {"7203.T", "9999.T"}


def _shadow_row_for_test(candidate_name: str) -> dict[str, object]:
    return {
        "run_timestamp": "2026-07-08T16:48:55",
        "ticker": "7203.T",
        "feature_date": "2026-07-08",
        "target_close_date": "2026-07-09",
        "candidate_name": candidate_name,
        "predicted_direction": "up",
        "probability_up": 0.55,
        "probability_down": 0.45,
        "confidence": 0.55,
        "source": "shadow",
        "model_note": "test",
        "actual_return": None,
        "actual_direction": "",
        "result": "pending",
        "evaluated_at": "",
        "caveat": "shadow only",
    }


def _features(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-24", periods=rows)
    close = pd.Series([100 + index for index in range(rows)], dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "open": close - 0.5,
            "high": close + 0.5,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
            "signal": [1 if index % 2 else -1 for index in range(rows)],
            "return_1d": close.pct_change().fillna(0.0),
            "ma_5": close.rolling(5, min_periods=1).mean(),
            "ma_25": close.rolling(25, min_periods=1).mean(),
        }
    )
