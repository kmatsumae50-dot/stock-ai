import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from stock_ai.storage.sqlite_store import (
    append_prediction_record,
    append_prediction_records,
    compare_append_only_official_to_snapshot,
    create_prediction_run,
    evaluate_prediction_records,
    filter_official_prediction_snapshot_records,
    get_append_only_candidate_records_from_snapshot,
    get_latest_live_record_by_feature_date,
    get_latest_official_prediction_records,
    get_or_create_prediction_run,
    get_prediction_run,
    get_records_for_feature_date,
    get_run_history_for_feature_date,
    init_database,
    is_official_prediction_snapshot_record,
    list_official_latest_prediction_snapshot_records,
    list_prediction_records,
    list_prediction_runs,
    read_latest_predictions,
    read_prediction_history,
    summarize_append_only_records,
    update_prediction_record_evaluation,
    upsert_predictions,
    write_append_only_prediction_run,
)


def test_upsert_predictions(tmp_path):
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "target": ["next_close"],
            "model_name": ["lightgbm"],
            "predicted_class": [3],
            "confidence": [0.62],
            "probability_surge": [0.05],
            "probability_up": [0.62],
            "probability_flat": [0.20],
            "probability_down": [0.38],
            "probability_crash": [0.02],
            "top_features": ['[{"feature": "rsi_14", "importance": 10.0}]'],
        }
    )

    assert upsert_predictions(predictions, db_path) == 1
    latest = read_latest_predictions(db_path)

    assert len(latest) == 1
    assert latest.loc[0, "ticker"] == "7203.T"
    assert latest.loc[0, "predicted_class"] == 3
    assert latest.loc[0, "probability_surge"] == 0.05
    assert latest.loc[0, "probability_flat"] == 0.20
    assert latest.loc[0, "probability_crash"] == 0.02
    assert not isinstance(latest.loc[0, "predicted_class"], bytes)


def test_upsert_predictions_stores_numpy_integer_as_sqlite_integer(tmp_path):
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "target": ["next_close"],
            "model_name": ["lightgbm"],
            "predicted_class": [np.int64(4)],
            "confidence": [0.71],
        }
    )

    upsert_predictions(predictions, db_path)
    latest = read_latest_predictions(db_path)

    assert latest.loc[0, "predicted_class"] == 4
    assert not isinstance(latest.loc[0, "predicted_class"], bytes)
    assert pd.isna(latest.loc[0, "probability_up"])
    assert pd.isna(latest.loc[0, "probability_crash"])


def test_read_prediction_history_can_filter_rows(tmp_path):
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["7203.T", "6758.T"],
            "target": ["next_close_binary", "next_close_binary"],
            "model_name": ["lightgbm_binary", "lightgbm_binary"],
            "predicted_class": [1, 0],
            "confidence": [0.62, 0.58],
            "probability_up": [0.62, 0.42],
            "probability_down": [0.38, 0.58],
        }
    )

    upsert_predictions(predictions, db_path)
    history = read_prediction_history(
        db_path,
        ticker="7203.T",
        target="next_close_binary",
        model_name="lightgbm_binary",
    )

    assert len(history) == 1
    assert history.loc[0, "ticker"] == "7203.T"
    assert history.loc[0, "predicted_class"] == 1


def test_upsert_predictions_overwrites_same_prediction_key_and_tracks_type(tmp_path):
    db_path = tmp_path / "market.db"
    first = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "predicted_class": [1],
            "confidence": [0.62],
            "probability_up": [0.62],
            "probability_down": [0.38],
            "prediction_type": ["backfill"],
        }
    )
    second = first.copy()
    second["predicted_class"] = [0]
    second["confidence"] = [0.71]
    second["probability_up"] = [0.29]
    second["probability_down"] = [0.71]
    second["prediction_type"] = ["live"]

    assert upsert_predictions(first, db_path) == 1
    assert upsert_predictions(second, db_path) == 1

    history = read_prediction_history(
        db_path,
        ticker="7203.T",
        target="next_close_binary",
        model_name="lightgbm_binary",
    )

    assert len(history) == 1
    assert history.loc[0, "predicted_class"] == 0
    assert history.loc[0, "prediction_type"] == "live"


def test_append_only_schema_init_is_idempotent(tmp_path):
    db_path = tmp_path / "market.db"

    init_database(db_path)
    init_database(db_path)

    with sqlite3.connect(db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }

    assert {"predictions", "prediction_runs", "prediction_records"} <= tables


def test_append_only_empty_tables_are_readable(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)

    runs = list_prediction_runs(db_path)
    records = list_prediction_records(db_path)
    summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert runs.empty
    assert records.empty
    assert summary["append_only_tables_exist"] is True
    assert summary["prediction_runs_count"] == 0
    assert summary["prediction_records_count"] == 0


def test_append_only_helpers_read_inserted_runs_and_records(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)
    _insert_append_only_run(
        db_path,
        run_id="run-1",
        run_timestamp="2026-07-08T10:30:00",
        run_type="scheduled_live",
        source="live",
    )
    _insert_append_only_record(
        db_path,
        record_id="record-1",
        run_id="run-1",
        feature_date="2026-07-08",
        source="live",
        is_official=1,
        is_latest=1,
        confidence=0.62,
    )

    run = get_prediction_run(db_path, "run-1")
    runs = list_prediction_runs(db_path, ticker="7203.T")
    records = list_prediction_records(db_path, ticker="7203.T", source="live")
    feature_records = get_records_for_feature_date(
        db_path,
        "2026-07-08",
        ticker="7203.T",
    )
    run_history = get_run_history_for_feature_date(
        db_path,
        "2026-07-08",
        ticker="7203.T",
    )

    assert run is not None
    assert run["run_type"] == "scheduled_live"
    assert len(runs) == 1
    assert len(records) == 1
    assert len(feature_records) == 1
    assert len(run_history) == 1
    assert run_history.loc[0, "record_record_id"] == "record-1"
    assert run_history.loc[0, "run_run_id"] == "run-1"


def test_append_only_latest_official_and_latest_live_selection(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)
    _insert_append_only_run(
        db_path,
        run_id="run-live-old",
        run_timestamp="2026-07-08T10:30:00",
        run_type="scheduled_live",
        source="live",
    )
    _insert_append_only_run(
        db_path,
        run_id="run-live-new",
        run_timestamp="2026-07-08T11:00:00",
        run_type="manual_live",
        source="live",
    )
    _insert_append_only_run(
        db_path,
        run_id="run-backfill",
        run_timestamp="2026-07-08T12:00:00",
        run_type="backfill",
        source="backfill",
    )
    _insert_append_only_record(
        db_path,
        record_id="record-live-old",
        run_id="run-live-old",
        feature_date="2026-07-08",
        source="live",
        is_official=1,
        confidence=0.51,
        created_at="2026-07-08T10:31:00",
    )
    _insert_append_only_record(
        db_path,
        record_id="record-live-new",
        run_id="run-live-new",
        feature_date="2026-07-08",
        source="live",
        is_official=1,
        confidence=0.64,
        created_at="2026-07-08T11:01:00",
    )
    _insert_append_only_record(
        db_path,
        record_id="record-backfill",
        run_id="run-backfill",
        feature_date="2026-07-08",
        source="backfill",
        is_official=0,
        confidence=0.72,
        created_at="2026-07-08T12:01:00",
    )

    official = get_latest_official_prediction_records(db_path, ticker="7203.T")
    latest_live = get_latest_live_record_by_feature_date(
        db_path,
        "2026-07-08",
        ticker="7203.T",
    )
    summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert len(official) == 1
    assert official.loc[0, "record_id"] == "record-live-new"
    assert len(latest_live) == 1
    assert latest_live.loc[0, "record_id"] == "record-live-new"
    assert summary["source_counts"] == {"live": 2, "backfill": 1}
    assert summary["run_type_counts"] == {
        "backfill": 1,
        "manual_live": 1,
        "scheduled_live": 1,
    }


def test_append_only_schema_does_not_change_prediction_snapshot_upsert(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "ticker": ["7203.T"],
            "target": ["next_close_binary"],
            "model_name": ["lightgbm_binary"],
            "predicted_class": [1],
            "confidence": [0.62],
            "prediction_type": ["live"],
        }
    )

    assert upsert_predictions(predictions, db_path) == 1
    snapshot = read_prediction_history(db_path, ticker="7203.T")
    summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert len(snapshot) == 1
    assert summary["existing_predictions_count"] == 1
    assert summary["prediction_runs_count"] == 0
    assert summary["prediction_records_count"] == 0


def test_create_prediction_run_and_get_or_create_are_idempotent(tmp_path):
    db_path = tmp_path / "market.db"

    run_id = create_prediction_run(
        db_path,
        run_id="manual-run-1",
        run_timestamp="2026-07-09T10:30:00",
        ticker="7203.T",
        run_type="manual_live",
        command_name="prediction-append-only-write-check",
        source="live",
        working_directory="C:/tmp/stock-ai",
    )
    same_run_id = get_or_create_prediction_run(
        db_path,
        run_id="manual-run-1",
        run_timestamp="2026-07-09T10:30:00",
        ticker="7203.T",
        run_type="manual_live",
        source="live",
    )
    runs = list_prediction_runs(db_path, ticker="7203.T")

    assert run_id == "manual-run-1"
    assert same_run_id == "manual-run-1"
    assert len(runs) == 1
    assert runs.loc[0, "run_type"] == "manual_live"


def test_append_prediction_record_prevents_duplicates_within_same_run(tmp_path):
    db_path = tmp_path / "market.db"
    create_prediction_run(
        db_path,
        run_id="run-dup",
        ticker="7203.T",
        run_type="manual_live",
        source="live",
    )

    first_count = append_prediction_record(
        db_path,
        record_id="record-dup-1",
        run_id="run-dup",
        ticker="7203.T",
        feature_date="2026-07-09",
        target="next_close_binary",
        model_name="lightgbm_binary",
        model_type="binary",
        predicted_label=1,
        predicted_direction="up",
        confidence=0.6,
        source="live",
    )
    duplicate_count = append_prediction_record(
        db_path,
        record_id="record-dup-2",
        run_id="run-dup",
        ticker="7203.T",
        feature_date="2026-07-09",
        target="next_close_binary",
        model_name="lightgbm_binary",
        model_type="binary",
        predicted_label=1,
        predicted_direction="up",
        confidence=0.6,
        source="live",
    )
    records = list_prediction_records(db_path)

    assert first_count == 1
    assert duplicate_count == 0
    assert len(records) == 1
    assert records.loc[0, "record_id"] == "record-dup-1"


def test_append_prediction_records_keeps_same_feature_date_across_runs(tmp_path):
    db_path = tmp_path / "market.db"
    for run_id in ("run-a", "run-b"):
        create_prediction_run(
            db_path,
            run_id=run_id,
            ticker="7203.T",
            run_type="manual_live",
            source="live",
        )
    records = pd.DataFrame(
        [
            {
                "record_id": "record-a",
                "run_id": "run-a",
                "ticker": "7203.T",
                "feature_date": "2026-07-09",
                "target": "next_close_binary",
                "model_name": "lightgbm_binary",
                "model_type": "binary",
                "predicted_label": "1",
                "predicted_direction": "up",
                "confidence": 0.61,
                "source": "live",
            },
            {
                "record_id": "record-b",
                "run_id": "run-b",
                "ticker": "7203.T",
                "feature_date": "2026-07-09",
                "target": "next_close_binary",
                "model_name": "lightgbm_binary",
                "model_type": "binary",
                "predicted_label": "0",
                "predicted_direction": "down",
                "confidence": 0.59,
                "source": "live",
            },
        ]
    )

    inserted = append_prediction_records(db_path, records)
    feature_records = get_records_for_feature_date(db_path, "2026-07-09", ticker="7203.T")

    assert inserted == 2
    assert len(feature_records) == 2
    assert set(feature_records["run_id"]) == {"run-a", "run-b"}


def test_append_only_sources_and_latest_official_are_separated(tmp_path):
    db_path = tmp_path / "market.db"
    for run_id, run_type, source in (
        ("run-live", "manual_live", "live"),
        ("run-backfill", "backfill", "backfill"),
        ("run-shadow", "shadow", "shadow"),
        ("run-experiment", "experiment", "experiment"),
    ):
        create_prediction_run(
            db_path,
            run_id=run_id,
            ticker="7203.T",
            run_type=run_type,
            source=source,
        )
        append_prediction_record(
            db_path,
            record_id=f"record-{source}",
            run_id=run_id,
            ticker="7203.T",
            feature_date="2026-07-09",
            target="next_close_binary",
            model_name="lightgbm_binary",
            predicted_label=1,
            predicted_direction="up",
            confidence=0.6,
            source=source,
            is_official=source == "live",
            is_latest=source == "live",
        )

    live_latest = get_latest_live_record_by_feature_date(
        db_path,
        "2026-07-09",
        ticker="7203.T",
    )
    official = get_latest_official_prediction_records(db_path, ticker="7203.T")
    summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert len(live_latest) == 1
    assert live_latest.loc[0, "source"] == "live"
    assert len(official) == 1
    assert official.loc[0, "source"] == "live"
    assert summary["source_counts"] == {
        "backfill": 1,
        "experiment": 1,
        "live": 1,
        "shadow": 1,
    }


def test_mark_latest_official_records_scopes_update(tmp_path):
    db_path = tmp_path / "market.db"
    create_prediction_run(
        db_path,
        run_id="run-latest",
        ticker="7203.T",
        run_type="manual_live",
        source="live",
    )
    for record_id, feature_date in (("old-record", "2026-07-08"), ("new-record", "2026-07-09")):
        append_prediction_record(
            db_path,
            record_id=record_id,
            run_id="run-latest",
            ticker="7203.T",
            feature_date=feature_date,
            target="next_close_binary",
            model_name="lightgbm_binary",
            source="live",
            is_official=True,
            is_latest=record_id == "old-record",
        )

    from stock_ai.storage.sqlite_store import mark_latest_official_records

    updated = mark_latest_official_records(
        db_path,
        ticker="7203.T",
        target="next_close_binary",
        model_name="lightgbm_binary",
        record_ids=["new-record"],
    )
    records = list_prediction_records(db_path, ticker="7203.T")

    assert updated >= 1
    latest = records[records["is_latest"] == 1]
    assert len(latest) == 1
    assert latest.iloc[0]["record_id"] == "new-record"


def test_update_and_evaluate_prediction_records(tmp_path):
    db_path = tmp_path / "market.db"
    create_prediction_run(
        db_path,
        run_id="run-eval",
        ticker="7203.T",
        run_type="manual_live",
        source="live",
    )
    append_prediction_record(
        db_path,
        record_id="record-eval",
        run_id="run-eval",
        ticker="7203.T",
        feature_date="2026-07-09",
        target_close_date="2026-07-10",
        target="next_close_binary",
        model_name="lightgbm_binary",
        predicted_label=1,
        predicted_direction="up",
        source="live",
    )

    updated = evaluate_prediction_records(
        db_path,
        pd.DataFrame(
            {
                "date": ["2026-07-10"],
                "actual_return": [0.012],
                "actual_direction": ["up"],
                "actual_label": ["up"],
            }
        ),
        ticker="7203.T",
    )
    records = list_prediction_records(db_path, ticker="7203.T")

    assert updated == 1
    assert records.loc[0, "result"] == "hit"
    assert records.loc[0, "actual_direction"] == "up"
    assert records.loc[0, "actual_return"] == 0.012

    manual_update = update_prediction_record_evaluation(
        db_path,
        record_id="record-eval",
        actual_return=-0.01,
        actual_direction="down",
        actual_label="down",
        result="miss",
        evaluated_at="2026-07-11T00:00:00",
    )
    updated_record = list_prediction_records(db_path, ticker="7203.T").iloc[0]

    assert manual_update == 1
    assert updated_record["result"] == "miss"
    assert updated_record["actual_direction"] == "down"


def test_write_append_only_prediction_run_uses_snapshot_without_changing_predictions(tmp_path):
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09"]),
            "ticker": ["7203.T", "7203.T"],
            "target": ["next_close_binary", "next_close_multiclass"],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass"],
            "predicted_class": [1, 2],
            "confidence": [0.62, 0.48],
            "probability_up": [0.62, 0.12],
            "probability_down": [0.38, 0.10],
            "prediction_type": ["live", "live"],
        }
    )
    upsert_predictions(predictions, db_path)
    before_snapshot = read_prediction_history(db_path, ticker="7203.T")

    result = write_append_only_prediction_run(
        db_path,
        before_snapshot,
        ticker="7203.T",
        run_id="run-write",
        run_type="manual_live",
        command_name="prediction-append-only-write-check",
        source="live",
    )
    after_snapshot = read_prediction_history(db_path, ticker="7203.T")
    records = list_prediction_records(db_path, ticker="7203.T")

    assert result["inserted_records"] == 2
    assert result["duplicate_records"] == 0
    assert len(records) == 2
    assert len(after_snapshot) == len(before_snapshot)
    assert after_snapshot[["date", "target", "model_name", "predicted_class"]].equals(
        before_snapshot[["date", "target", "model_name", "predicted_class"]]
    )


def test_filter_official_prediction_snapshot_records_includes_only_current_production():
    snapshot = _mixed_prediction_snapshot()

    included, excluded = filter_official_prediction_snapshot_records(
        snapshot,
        ticker="7203.T",
    )

    assert len(included) == 2
    assert set(zip(included["target"], included["model_name"], strict=True)) == {
        ("next_close_binary", "lightgbm_binary"),
        ("next_close_multiclass", "lightgbm_multiclass"),
    }
    assert set(excluded["excluded_reason"]) == {
        "stale_feature_date",
        "non_official_target_model",
        "non_live_source_backfill",
        "non_live_source_shadow",
        "non_live_source_experiment",
        "ticker_mismatch",
    }


def test_filter_official_prediction_snapshot_record_reports_reason():
    snapshot = _mixed_prediction_snapshot()
    official = snapshot.iloc[0].to_dict()
    experiment = snapshot[snapshot["model_name"] == "lightgbm_multiclass_quantile"].iloc[0].to_dict()

    is_official, official_reason = is_official_prediction_snapshot_record(
        official,
        ticker="7203.T",
        latest_feature_date="2026-07-09",
    )
    is_excluded, excluded_reason = is_official_prediction_snapshot_record(
        experiment,
        ticker="7203.T",
        latest_feature_date="2026-07-09",
    )

    assert is_official is True
    assert official_reason == "official_production_current"
    assert is_excluded is False
    assert excluded_reason == "stale_feature_date"


def test_append_only_candidate_records_from_snapshot_returns_included_only():
    candidates = get_append_only_candidate_records_from_snapshot(
        _mixed_prediction_snapshot(),
        ticker="7203.T",
    )

    assert len(candidates) == 2
    assert candidates["official_filter_status"].tolist() == ["included", "included"]


def test_list_official_latest_prediction_snapshot_records_reads_existing_snapshot(tmp_path):
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09", "2026-06-01"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_multiclass",
            ],
            "model_name": [
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_multiclass_quantile",
            ],
            "predicted_class": [0, 2, 3],
            "confidence": [0.53, 0.48, 0.44],
            "prediction_type": ["live", "live", "live"],
        }
    )
    upsert_predictions(predictions, db_path)

    official = list_official_latest_prediction_snapshot_records(db_path, ticker="7203.T")

    assert len(official) == 2
    assert set(official["model_name"]) == {"lightgbm_binary", "lightgbm_multiclass"}


def test_write_append_only_prediction_run_with_official_filter_writes_only_included_records(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _mixed_prediction_snapshot()
    included, _excluded = filter_official_prediction_snapshot_records(snapshot, ticker="7203.T")

    result = write_append_only_prediction_run(
        db_path,
        included,
        ticker="7203.T",
        run_id="official-filter-run",
        run_type="manual_live",
        command_name="prediction-append-only-write-check",
        source="live",
        is_official=True,
        is_latest=True,
    )
    records = list_prediction_records(db_path, ticker="7203.T")

    assert result["inserted_records"] == 2
    assert len(records) == 2
    assert records["is_official"].tolist() == [1, 1]
    assert records["is_latest"].tolist() == [1, 1]
    assert set(records["model_name"]) == {"lightgbm_binary", "lightgbm_multiclass"}


def test_append_only_write_check_cli_filters_and_respects_dry_run(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "market.db"
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09", "2026-06-01"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_multiclass",
            ],
            "model_name": [
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_multiclass_quantile",
            ],
            "predicted_class": [0, 2, 3],
            "confidence": [0.53, 0.48, 0.44],
            "prediction_type": ["live", "live", "live"],
        }
    )
    upsert_predictions(predictions, db_path)

    dry_run = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-write-check",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
            "--dry-run",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    dry_summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert "raw latest snapshot candidates: 3" in dry_run.stdout
    assert "included official candidates: 2" in dry_run.stdout
    assert "excluded candidates: 1" in dry_run.stdout
    assert "reason=stale_feature_date" in dry_run.stdout
    assert dry_summary["prediction_runs_count"] == 0
    assert dry_summary["prediction_records_count"] == 0

    write = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-write-check",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
            "--write",
            "--run-id",
            "cli-filter-test",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    write_summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert "record candidates: 2" in write.stdout
    assert "inserted records: 2" in write.stdout
    assert write_summary["prediction_runs_count"] == 1
    assert write_summary["prediction_records_count"] == 2
    assert write_summary["existing_predictions_count"] == 3


def test_append_only_compare_matches_official_snapshot_after_dual_write(tmp_path):
    db_path = tmp_path / "market.db"
    snapshot = _dual_write_prediction_snapshot()
    upsert_predictions(snapshot, db_path)
    latest_snapshot = read_latest_predictions(db_path)
    included, _excluded = filter_official_prediction_snapshot_records(
        latest_snapshot,
        ticker="7203.T",
    )

    before = compare_append_only_official_to_snapshot(db_path, ticker="7203.T")
    result = write_append_only_prediction_run(
        db_path,
        included,
        ticker="7203.T",
        run_id="dual-write-compare-test",
        run_type="manual_live_dual_write",
        command_name="prediction-append-only-dual-write",
        source="live",
        is_official=True,
        is_latest=True,
    )
    after = compare_append_only_official_to_snapshot(db_path, ticker="7203.T")

    assert before["snapshot_official_count"] == 2
    assert before["snapshot_excluded_count"] == 1
    assert before["append_official_count"] == 0
    assert before["matches"] is False
    assert result["inserted_records"] == 2
    assert after["snapshot_official_count"] == 2
    assert after["snapshot_excluded_count"] == 1
    assert after["append_official_count"] == 2
    assert after["matches"] is True
    assert {row["reason"] for row in after["rows"]} == {"match"}


def test_append_only_dual_write_cli_is_dry_run_by_default_and_idempotent(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "market.db"
    upsert_predictions(_dual_write_prediction_snapshot(), db_path)

    dry_run = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-dual-write",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    dry_summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert "mode: dry-run" in dry_run.stdout
    assert "included official candidates: 2" in dry_run.stdout
    assert "excluded candidates: 1" in dry_run.stdout
    assert dry_summary["prediction_runs_count"] == 0
    assert dry_summary["prediction_records_count"] == 0

    write = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-dual-write",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
            "--write",
            "--idempotency-key",
            "cli-dual-write-test",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    repeat = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-dual-write",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
            "--write",
            "--idempotency-key",
            "cli-dual-write-test",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    compare = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-compare",
            "--ticker",
            "7203.T",
            "--database-path",
            str(db_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    write_summary = summarize_append_only_records(db_path, ticker="7203.T")

    assert "mode: write" in write.stdout
    assert "record candidates: 2" in write.stdout
    assert "inserted records: 2" in write.stdout
    assert "inserted records: 0" in repeat.stdout
    assert "duplicate records ignored: 2" in repeat.stdout
    assert "matches: true" in compare.stdout
    assert write_summary["prediction_runs_count"] == 1
    assert write_summary["prediction_records_count"] == 2
    assert write_summary["existing_predictions_count"] == 3


def _insert_append_only_run(
    db_path,
    *,
    run_id,
    run_timestamp,
    run_type,
    source,
):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_runs
                (
                    run_id,
                    run_timestamp,
                    ticker,
                    run_type,
                    command_name,
                    source,
                    working_directory,
                    data_freshness_status,
                    latest_raw_date,
                    latest_feature_date,
                    expected_latest_trading_date,
                    stale_at_run_time,
                    notes
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                run_timestamp,
                "7203.T",
                run_type,
                "predict-binary",
                source,
                "C:/tmp/stock-ai",
                "OK",
                "2026-07-08",
                "2026-07-08",
                "2026-07-08",
                0,
                None,
            ),
        )


def _insert_append_only_record(
    db_path,
    *,
    record_id,
    run_id,
    feature_date,
    source,
    is_official,
    confidence,
    is_latest=0,
    created_at="2026-07-08T10:31:00",
):
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_records
                (
                    record_id,
                    run_id,
                    ticker,
                    feature_date,
                    target_close_date,
                    target,
                    model_name,
                    model_type,
                    predicted_label,
                    predicted_direction,
                    probability_up,
                    probability_down,
                    confidence,
                    source,
                    is_official,
                    is_latest,
                    result,
                    created_at
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_id,
                run_id,
                "7203.T",
                feature_date,
                "2026-07-09",
                "close_binary",
                "lightgbm_binary",
                "binary",
                "up",
                "up",
                0.62,
                0.38,
                confidence,
                source,
                is_official,
                is_latest,
                "pending",
                created_at,
            ),
        )


def _mixed_prediction_snapshot():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2026-07-09",
                    "2026-07-09",
                    "2026-06-01",
                    "2026-07-09",
                    "2026-07-09",
                    "2026-07-09",
                    "2026-07-09",
                    "2026-07-09",
                ]
            ),
            "ticker": [
                "7203.T",
                "7203.T",
                "7203.T",
                "7203.T",
                "7203.T",
                "7203.T",
                "7203.T",
                "6758.T",
            ],
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_multiclass",
                "next_close_multiclass",
                "next_close_binary",
                "next_close_binary",
                "next_close_binary",
                "next_close_binary",
            ],
            "model_name": [
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_multiclass_quantile",
                "lightgbm_multiclass_experiment",
                "lightgbm_binary",
                "lightgbm_binary",
                "lightgbm_binary",
                "lightgbm_binary",
            ],
            "predicted_class": [0, 2, 3, 1, 1, 1, 1, 1],
            "confidence": [0.53, 0.48, 0.44, 0.41, 0.62, 0.63, 0.64, 0.70],
            "prediction_type": [
                "live",
                "live",
                "live",
                "live",
                "backfill",
                "shadow",
                "experiment",
                "live",
            ],
            "created_at": [
                "2026-07-09 04:16:38",
                "2026-07-09 04:16:44",
                "2026-06-01 00:00:00",
                "2026-07-09 04:00:00",
                "2026-07-09 04:01:00",
                "2026-07-09 04:02:00",
                "2026-07-09 04:03:00",
                "2026-07-09 04:04:00",
            ],
        }
    )


def _dual_write_prediction_snapshot():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-07-09", "2026-07-09", "2026-06-01"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "target": [
                "next_close_binary",
                "next_close_multiclass",
                "next_close_multiclass",
            ],
            "model_name": [
                "lightgbm_binary",
                "lightgbm_multiclass",
                "lightgbm_multiclass_quantile",
            ],
            "predicted_class": [0, 2, 3],
            "confidence": [0.53, 0.48, 0.44],
            "prediction_type": ["live", "live", "live"],
            "created_at": [
                "2026-07-09 04:16:38",
                "2026-07-09 04:16:44",
                "2026-06-01 00:00:00",
            ],
        }
    )
