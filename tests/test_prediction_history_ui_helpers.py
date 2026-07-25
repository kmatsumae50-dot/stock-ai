import pandas as pd

from stock_ai.app.main import (
    _apply_prediction_history_record_scope,
    _prediction_history_scope_frame,
    _prediction_history_scope_summary,
)


def _sample_prediction_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-07",
                    "2026-07-07",
                    "2026-07-08",
                    "2026-07-08",
                    "2026-06-02",
                    "2026-05-01",
                ]
            ),
            "ticker": ["7203.T"] * 6,
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_binary",
                "next_close_multiclass",
                "next_close_multiclass",
                "next_close_binary",
            ],
            "model_name": [
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_multiclass_quantile",
                "lightgbm_binary",
            ],
            "prediction_type": ["live", "live", "live", "live", "live", "backfill"],
            "created_at": pd.to_datetime(
                [
                    "2026-07-07 10:30",
                    "2026-07-07 10:31",
                    "2026-07-08 10:30",
                    "2026-07-08 10:31",
                    "2026-06-02 07:53",
                    "2026-06-01 06:26",
                ]
            ),
        }
    )


def test_prediction_history_scope_summary_separates_total_latest_and_production_rows():
    summary = _prediction_history_scope_summary(_sample_prediction_rows())

    assert summary["total_rows"] == 6
    assert summary["live_rows"] == 5
    assert summary["backfill_rows"] == 1
    assert summary["production_live_rows"] == 4
    assert summary["binary_production_live_rows"] == 2
    assert summary["multiclass_production_live_rows"] == 2
    assert summary["experimental_live_rows"] == 1
    assert summary["latest_feature_date"] == pd.Timestamp("2026-07-08")
    assert summary["latest_feature_date_rows"] == 2
    assert summary["latest_feature_date_production_rows"] == 2


def test_prediction_history_record_scope_defaults_to_production_live_only():
    rows = _sample_prediction_rows()

    production = _apply_prediction_history_record_scope(rows, "Production live")
    experimental = _apply_prediction_history_record_scope(rows, "Experimental / variants")
    backfill = _apply_prediction_history_record_scope(rows, "Backfill only")

    assert len(production) == 4
    assert set(production["model_name"]) == {"lightgbm_binary", "lightgbm_multiclass"}
    assert len(experimental) == 1
    assert experimental.iloc[0]["model_name"] == "lightgbm_multiclass_quantile"
    assert len(backfill) == 1
    assert backfill.iloc[0]["prediction_type"] == "backfill"


def test_prediction_history_scope_frame_explains_record_counts():
    frame = _prediction_history_scope_frame(_sample_prediction_rows())

    by_scope = frame.set_index("Scope")
    assert by_scope.loc["All saved rows", "Rows"] == 6
    assert by_scope.loc["Production live rows", "Rows"] == 4
    assert by_scope.loc["Experimental live rows", "Rows"] == 1
    assert by_scope.loc["Backfill rows", "Rows"] == 1
    assert "normal daily operation" in by_scope.loc["Production live rows", "Meaning"]
