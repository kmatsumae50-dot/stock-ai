import pandas as pd

from stock_ai.backtest.prediction_history import (
    diagnose_missing_live_dates,
    evaluate_prediction_history,
    missing_prediction_dates,
    summarize_prediction_operations,
)
from stock_ai.config import load_settings


def test_evaluate_prediction_history_matches_saved_predictions_to_next_close():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "target": ["next_close_binary"] * 3,
            "model_name": ["lightgbm_binary"] * 3,
            "predicted_class": [1, 0, 1],
            "probability_up": [0.70, 0.40, 0.65],
            "probability_down": [0.30, 0.60, 0.35],
            "confidence": [0.70, 0.60, 0.65],
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "close": [100.0, 101.0, 99.0],
        }
    )

    history, metrics = evaluate_prediction_history(predictions, prices)

    assert metrics["total_predictions"] == 3
    assert metrics["settled_predictions"] == 2
    assert metrics["pending_predictions"] == 1
    assert metrics["correct_predictions"] == 2
    assert metrics["accuracy"] == 1.0
    assert metrics["up_prediction_accuracy"] == 1.0
    assert metrics["down_prediction_accuracy"] == 1.0
    assert not metrics["confidence_accuracy"].empty
    assert history.loc[history["date"] == pd.Timestamp("2024-01-03"), "status"].iloc[0] == "pending"


def test_prediction_operations_find_missing_live_dates_and_latest_save():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 3,
            "target": ["next_close_binary"] * 3,
            "model_name": ["lightgbm_binary"] * 3,
            "predicted_class": [1, 0, 1],
            "prediction_type": ["live", "backfill", "live"],
            "created_at": pd.to_datetime(
                ["2024-01-01 17:00", "2024-01-03 17:00", "2024-01-04 17:00"]
            ),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    missing = missing_prediction_dates(predictions[predictions["prediction_type"] == "live"], prices)
    operations = summarize_prediction_operations(predictions, prices, ticker="7203.T", lookback_trading_days=4)

    assert missing == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert operations["missing_prediction_dates"] == [
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert operations["missing_prediction_count"] == 2
    assert operations["backfilled_missing_dates"] == [pd.Timestamp("2024-01-03")]
    assert operations["backfilled_missing_count"] == 1
    assert operations["backfillable_missing_dates"] == [pd.Timestamp("2024-01-02")]
    assert operations["backfillable_missing_count"] == 1
    assert operations["first_backfillable_missing_date"] == pd.Timestamp("2024-01-02")
    assert operations["latest_prediction_date"] == pd.Timestamp("2024-01-04")
    assert operations["latest_saved_at"] == pd.Timestamp("2024-01-04 17:00")


def test_missing_live_dates_ignore_dates_before_operation_start():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "predicted_class": [1],
            "prediction_type": ["live"],
            "created_at": pd.to_datetime(["2024-01-04 17:00"]),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    operations = summarize_prediction_operations(
        predictions,
        prices,
        ticker="7203.T",
        lookback_trading_days=4,
        live_prediction_start_date="2024-01-04",
    )

    assert operations["missing_prediction_dates"] == []
    assert operations["missing_prediction_count"] == 0
    assert operations["pre_start_excluded_dates"] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert operations["pre_start_excluded_count"] == 3
    assert operations["ignored_pre_start_missing_dates"] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]
    assert operations["ignored_pre_start_missing_count"] == 3
    assert operations["live_prediction_start_date"] == pd.Timestamp("2024-01-04")


def test_missing_live_dates_warn_after_operation_start():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "predicted_class": [1],
            "prediction_type": ["live"],
            "created_at": pd.to_datetime(["2024-01-04 17:00"]),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    operations = summarize_prediction_operations(
        predictions,
        prices,
        ticker="7203.T",
        lookback_trading_days=4,
        live_prediction_start_date="2024-01-03",
    )

    assert operations["missing_prediction_dates"] == [pd.Timestamp("2024-01-03")]
    assert operations["missing_prediction_count"] == 1
    assert operations["pre_start_excluded_dates"] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]
    assert operations["pre_start_excluded_count"] == 2
    assert operations["ignored_pre_start_missing_dates"] == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
    ]


def test_missing_live_dates_default_behavior_without_operation_start_is_unchanged():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "predicted_class": [1],
            "prediction_type": ["live"],
            "created_at": pd.to_datetime(["2024-01-04 17:00"]),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 4,
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    missing = missing_prediction_dates(
        predictions,
        prices,
        ticker="7203.T",
        lookback_trading_days=4,
    )

    assert missing == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-02"),
        pd.Timestamp("2024-01-03"),
    ]


def test_missing_live_date_diagnostics_detects_data_not_available_at_run_time():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-04"]),
            "ticker": ["7203.T"] * 3,
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_binary",
            ],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass", "lightgbm_binary"],
            "prediction_type": ["live", "live", "live"],
            "created_at": pd.to_datetime(
                ["2024-01-03 09:00", "2024-01-03 09:01", "2024-01-04 17:00"]
            ),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )

    diagnostics = diagnose_missing_live_dates(
        predictions,
        prices,
        [pd.Timestamp("2024-01-03")],
        ticker="7203.T",
    )

    row = diagnostics.iloc[0]
    assert bool(row["in_feature_data"]) is True
    assert bool(row["any_prediction_exists"]) is False
    assert bool(row["binary_live_exists"]) is False
    assert bool(row["multiclass_live_exists"]) is False
    assert row["previous_live_prediction_date"] == pd.Timestamp("2024-01-02")
    assert row["previous_live_saved_at"] == pd.Timestamp("2024-01-03 09:01")
    assert row["next_live_prediction_date"] == pd.Timestamp("2024-01-04")
    assert row["inferred_reason"] == "data_not_available_at_run_time"


def test_missing_live_date_diagnostics_distinguishes_target_and_backfill_cases():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "ticker": ["7203.T"] * 3,
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_binary",
            ],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass", "lightgbm_binary"],
            "prediction_type": ["live", "live", "backfill"],
            "created_at": pd.to_datetime(
                ["2024-01-03 17:00", "2024-01-04 17:00", "2024-01-05 17:00"]
            ),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"]),
            "ticker": ["7203.T"] * 3,
            "close": [100.0, 101.0, 102.0],
        }
    )

    diagnostics = diagnose_missing_live_dates(
        predictions,
        prices,
        [
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-04"),
            pd.Timestamp("2024-01-05"),
        ],
        ticker="7203.T",
    ).set_index("missing_date")

    assert diagnostics.loc[pd.Timestamp("2024-01-03"), "inferred_reason"] == "multiclass_missing_only"
    assert diagnostics.loc[pd.Timestamp("2024-01-04"), "inferred_reason"] == "binary_missing_only"
    assert diagnostics.loc[pd.Timestamp("2024-01-05"), "inferred_reason"] == "backfill_only"
    assert bool(diagnostics.loc[pd.Timestamp("2024-01-05"), "backfill_exists"]) is True


def test_missing_live_date_diagnostics_marks_known_missed_for_matching_ticker():
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "prediction_type": ["live"],
            "created_at": pd.to_datetime(["2024-01-03 09:00"]),
        }
    )
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "ticker": ["7203.T"] * 2,
            "close": [100.0, 101.0],
        }
    )

    diagnostics = diagnose_missing_live_dates(
        predictions,
        prices,
        [pd.Timestamp("2024-01-03")],
        ticker="7203.T",
        known_missed_live_dates=[
            {
                "date": "2024-01-03",
                "ticker": "7203.T",
                "reason_code": "data_not_available_at_run_time",
                "reason": "Data was not available at run time.",
                "action": "do_not_create_live_prediction",
                "allow_backfill": True,
            }
        ],
    )

    row = diagnostics.iloc[0]
    assert bool(row["known_missed"]) is True
    assert row["known_reason_code"] == "data_not_available_at_run_time"
    assert row["known_action"] == "do_not_create_live_prediction"
    assert bool(row["known_allow_backfill"]) is True


def test_missing_live_date_diagnostics_ignores_known_missed_for_other_ticker():
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03"]),
            "ticker": ["7203.T"],
            "close": [100.0],
        }
    )

    diagnostics = diagnose_missing_live_dates(
        pd.DataFrame(),
        prices,
        [pd.Timestamp("2024-01-03")],
        ticker="7203.T",
        known_missed_live_dates=[
            {
                "date": "2024-01-03",
                "ticker": "9999.T",
                "reason_code": "data_not_available_at_run_time",
                "action": "do_not_create_live_prediction",
                "allow_backfill": True,
            }
        ],
    )

    assert bool(diagnostics.iloc[0]["known_missed"]) is False


def test_july_known_missed_dates_are_post_start_and_keep_existing_dates():
    settings = load_settings()
    known_rows = settings.operations.known_missed_live_dates
    known_by_date = {row.date: row for row in known_rows}
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-06-04", "2026-07-13"]),
            "ticker": ["7203.T", "7203.T"],
            "close": [100.0, 110.0],
        }
    )
    operations = summarize_prediction_operations(
        pd.DataFrame(),
        prices,
        ticker="7203.T",
        lookback_trading_days=2,
        live_prediction_start_date=settings.operations.live_prediction_start_date,
    )
    diagnostics = diagnose_missing_live_dates(
        pd.DataFrame(),
        prices,
        [pd.Timestamp("2026-07-13")],
        ticker="7203.T",
        known_missed_live_dates=known_rows,
    )

    assert {"2026-06-15", "2026-06-23", "2026-07-13", "2026-07-21"} <= set(known_by_date)
    assert known_by_date["2026-07-13"].reason_code == "task_scheduler_not_run"
    assert known_by_date["2026-07-13"].action == "do_not_create_live_prediction"
    assert known_by_date["2026-07-13"].allow_backfill is True
    assert known_by_date["2026-07-21"].reason_code == "prediction_generation_failed"
    assert known_by_date["2026-07-21"].action == "do_not_create_live_prediction"
    assert known_by_date["2026-07-21"].allow_backfill is True
    assert pd.Timestamp("2026-07-13") in operations["missing_prediction_dates"]
    assert pd.Timestamp("2026-07-13") not in operations["pre_start_excluded_dates"]
    assert pd.Timestamp("2026-06-04") in operations["pre_start_excluded_dates"]
    assert bool(diagnostics.iloc[0]["known_missed"]) is True
    assert diagnostics.iloc[0]["known_reason_code"] == "task_scheduler_not_run"
    assert diagnostics.iloc[0]["known_action"] == "do_not_create_live_prediction"
