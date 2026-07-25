import pandas as pd

from stock_ai.backtest.append_only_history import (
    build_append_only_health_visibility,
    build_append_only_feature_date_run_summary,
    build_append_only_run_detail,
    build_append_only_run_snapshot_compare,
    summarize_append_only_history_for_ui,
)
from stock_ai.storage.sqlite_store import (
    init_database,
    upsert_predictions,
    write_append_only_prediction_run,
)


def test_append_only_history_summary_handles_empty_tables(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)

    summary = summarize_append_only_history_for_ui(db_path, ticker="7203.T")

    assert summary["run_summary"]["total_runs"] == 0
    assert summary["record_summary"]["total_records"] == 0
    assert summary["feature_date_summary"].empty
    assert summary["evaluation_summary"]["pending_count"] == 0
    assert summary["comparison"]["append_official_count"] == 0


def test_append_only_health_visibility_handles_empty_tables(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)

    visibility = build_append_only_health_visibility(db_path, ticker="7203.T")

    assert visibility["visibility_status"] == "append_only_empty"
    assert "append_only_not_health_source" in visibility["visibility_classifications"]
    assert visibility["prediction_runs_count"] == 0
    assert visibility["prediction_records_count"] == 0
    assert visibility["append_only_vs_snapshot_matches"] is False
    assert visibility["mismatch_count"] == 0


def test_append_only_history_summary_reads_one_manual_dual_write_run(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    write_append_only_prediction_run(
        db_path,
        snapshot,
        ticker="7203.T",
        run_id="manual-dual-write",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        data_freshness_status="OK",
        latest_raw_date="2026-07-09",
        latest_feature_date="2026-07-09",
        expected_latest_trading_date="2026-07-09",
        stale_at_run_time=False,
        is_official=True,
        is_latest=True,
        notes="manual controlled append-only dual-write",
    )

    summary = summarize_append_only_history_for_ui(db_path, ticker="7203.T")

    assert summary["run_summary"]["total_runs"] == 1
    assert summary["run_summary"]["latest_run_id"] == "manual-dual-write"
    assert summary["run_summary"]["latest_run_type"] == "manual_live_dual_write"
    assert summary["record_summary"]["total_records"] == 2
    assert summary["record_summary"]["official_records"] == 2
    assert summary["record_summary"]["binary_records"] == 1
    assert summary["record_summary"]["multiclass_records"] == 1
    assert summary["evaluation_summary"]["pending_count"] == 2
    assert summary["comparison"]["matches"] is True


def test_append_only_health_visibility_reports_match_status(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    write_append_only_prediction_run(
        db_path,
        snapshot,
        ticker="7203.T",
        run_id="manual-dual-write",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        is_official=True,
        is_latest=True,
    )

    visibility = build_append_only_health_visibility(db_path, ticker="7203.T")

    assert visibility["visibility_status"] == "append_only_available"
    assert "append_only_latest_matches_snapshot" in visibility["visibility_classifications"]
    assert "append_only_manual_only" in visibility["visibility_classifications"]
    assert "append_only_not_health_source" in visibility["visibility_classifications"]
    assert visibility["prediction_runs_count"] == 1
    assert visibility["prediction_records_count"] == 2
    assert visibility["latest_official_records_count"] == 2
    assert visibility["append_only_vs_snapshot_matches"] is True
    assert visibility["mismatch_count"] == 0
    assert visibility["pending_count"] == 2
    assert visibility["hit_count"] == 0
    assert visibility["miss_count"] == 0
    assert visibility["unevaluated_count"] == 2


def test_append_only_history_summary_counts_multiple_runs_for_same_feature_date(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    for run_id, confidence_delta in (("first-run", 0.0), ("second-run", 0.01)):
        rows = snapshot.copy()
        rows["confidence"] = rows["confidence"] + confidence_delta
        write_append_only_prediction_run(
            db_path,
            rows,
            ticker="7203.T",
            run_id=run_id,
            run_type="manual_live_dual_write",
            command_name="prediction-append-only-dual-write",
            source="live",
            is_official=True,
            is_latest=True,
        )

    summary = summarize_append_only_history_for_ui(db_path, ticker="7203.T")
    feature_summary = summary["feature_date_summary"]

    assert summary["run_summary"]["total_runs"] == 2
    assert summary["record_summary"]["total_records"] == 4
    assert len(feature_summary) == 1
    assert feature_summary.loc[0, "feature_date"] == "2026-07-09"
    assert feature_summary.loc[0, "run_count"] == 2
    assert feature_summary.loc[0, "run_ids"] == "first-run, second-run"
    assert feature_summary.loc[0, "record_count"] == 4
    assert feature_summary.loc[0, "binary_records"] == 2
    assert feature_summary.loc[0, "multiclass_records"] == 2


def test_append_only_health_visibility_reports_mismatch_as_info(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    mismatched = snapshot.copy()
    mismatched["confidence"] = [0.10, 0.20]
    write_append_only_prediction_run(
        db_path,
        mismatched,
        ticker="7203.T",
        run_id="mismatch-run",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        is_official=True,
        is_latest=True,
    )

    visibility = build_append_only_health_visibility(db_path, ticker="7203.T")

    assert visibility["visibility_status"] == "append_only_available"
    assert "append_only_mismatch_detected_info" in visibility["visibility_classifications"]
    assert "append_only_not_health_source" in visibility["visibility_classifications"]
    assert visibility["append_only_vs_snapshot_matches"] is False
    assert visibility["mismatch_count"] == 2


def test_append_only_run_detail_reads_selected_run_records(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    write_append_only_prediction_run(
        db_path,
        snapshot,
        ticker="7203.T",
        run_id="selected-run",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        is_official=True,
        is_latest=True,
    )

    detail = build_append_only_run_detail(db_path, ticker="7203.T", run_id="selected-run")

    assert detail["run"]["run_id"] == "selected-run"
    assert len(detail["records"]) == 2
    assert len(detail["record_table"]) == 2
    assert detail["run_summary"]["record_count"] == 2
    assert detail["run_summary"]["official_record_count"] == 2
    assert detail["run_summary"]["binary_record_count"] == 1
    assert detail["run_summary"]["multiclass_record_count"] == 1
    assert detail["run_summary"]["pending_count"] == 2
    assert detail["snapshot_compare"]["matches"] is True
    assert detail["snapshot_compare"]["matched_count"] == 2


def test_append_only_run_snapshot_compare_reports_value_mismatch(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _official_snapshot()
    upsert_predictions(snapshot, db_path)
    mismatched = snapshot.copy()
    mismatched["confidence"] = [0.10, 0.20]
    write_append_only_prediction_run(
        db_path,
        mismatched,
        ticker="7203.T",
        run_id="mismatch-run",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        is_official=True,
        is_latest=True,
    )

    comparison = build_append_only_run_snapshot_compare(
        db_path,
        ticker="7203.T",
        run_id="mismatch-run",
    )

    assert comparison["matches"] is False
    assert comparison["matched_count"] == 0
    assert comparison["mismatch_count"] == 2
    assert comparison["value_mismatch_count"] == 2
    assert {row["reason"] for row in comparison["rows"]} == {"value_mismatch"}
    assert all(row["label_match"] for row in comparison["rows"])
    assert all(row["direction_match"] for row in comparison["rows"])
    assert not any(row["confidence_match"] for row in comparison["rows"])


def test_append_only_feature_date_run_summary_handles_empty_records():
    summary = build_append_only_feature_date_run_summary(
        pd.DataFrame(),
        pd.DataFrame(),
    )

    assert summary.empty
    assert "run_ids" in summary.columns
    assert "official_run_ids" in summary.columns


def _official_snapshot():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09"]),
            "ticker": ["7203.T", "7203.T"],
            "target": ["next_close_binary", "next_close_multiclass"],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass"],
            "predicted_class": [0, 2],
            "confidence": [0.53, 0.48],
            "prediction_type": ["live", "live"],
        }
    )
