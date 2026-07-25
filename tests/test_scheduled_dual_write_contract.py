from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

from stock_ai.backtest.append_only_policy import (
    BACKFILL,
    EXPERIMENT,
    MANUAL_LIVE_DUAL_WRITE,
    SHADOW,
    build_official_selection_diagnostics_from_db,
)
from stock_ai.backtest.scheduled_dual_write import (
    CATCH_UP_AFTER_MISSED_SCHEDULE,
    MANUAL,
    RETRY,
    assess_scheduled_official_eligibility,
    build_prediction_idempotency_key,
    build_scheduled_dual_write_contract,
    build_scheduled_idempotency_key,
    run_scheduled_contract_check,
)
from stock_ai.storage.sqlite_store import (
    init_database,
    list_prediction_records,
    list_prediction_runs,
    upsert_predictions,
)


TICKER = "7203.T"
FEATURE_DATE = "2026-07-16"


def test_scheduled_complete_contract_is_eligible():
    eligibility = assess_scheduled_official_eligibility(_contract(), _snapshot())

    assert eligibility["eligible_for_official"] is True
    assert eligibility["complete_target_set"] is True
    assert eligibility["present_slots"] == ["binary", "multiclass"]
    assert eligibility["missing_slots"] == []
    assert eligibility["duplicate_slots"] == []
    assert eligibility["selected_official_candidate"] is True


@pytest.mark.parametrize(
    ("run_type", "source", "execution_mode"),
    [
        (MANUAL_LIVE_DUAL_WRITE, "live", MANUAL),
        (BACKFILL, "backfill", MANUAL),
        (SHADOW, "shadow", MANUAL),
        (EXPERIMENT, "experiment", MANUAL),
    ],
)
def test_non_scheduled_contracts_are_not_eligible(run_type, source, execution_mode):
    contract = _contract(
        run_type=run_type,
        source=source,
        execution_mode=execution_mode,
    )
    records = _snapshot(source=source)

    eligibility = assess_scheduled_official_eligibility(contract, records)

    assert eligibility["eligible_for_official"] is False
    assert "run_type_not_scheduled_live_dual_write" in eligibility["reason_codes"]


def test_partial_contract_is_not_eligible():
    contract = _contract(multiclass_success=False)
    records = _snapshot().query("target == 'next_close_binary'").reset_index(drop=True)

    eligibility = assess_scheduled_official_eligibility(contract, records)

    assert eligibility["eligible_for_official"] is False
    assert eligibility["attempt_status"] == "partial"
    assert eligibility["missing_slots"] == ["multiclass"]
    assert "multiclass_prediction_failed" in eligibility["reason_codes"]


def test_both_prediction_failures_are_reported_without_candidates():
    contract = _contract(binary_success=False, multiclass_success=False)

    eligibility = assess_scheduled_official_eligibility(contract, pd.DataFrame())

    assert eligibility["eligible_for_official"] is False
    assert eligibility["attempt_status"] == "failed"
    assert eligibility["missing_slots"] == ["binary", "multiclass"]
    assert "binary_prediction_failed" in eligibility["reason_codes"]
    assert "multiclass_prediction_failed" in eligibility["reason_codes"]


def test_stale_or_wrong_feature_date_is_not_eligible():
    stale = assess_scheduled_official_eligibility(
        _contract(stale_at_run_time=True),
        _snapshot(),
    )
    wrong_date = assess_scheduled_official_eligibility(
        _contract(expected_feature_date="2026-07-15"),
        _snapshot(),
    )

    assert stale["eligible_for_official"] is False
    assert "stale_at_run_time" in stale["reason_codes"]
    assert wrong_date["eligible_for_official"] is False
    assert "feature_date_mismatch" in wrong_date["reason_codes"]


def test_old_experiment_model_is_visible_but_excluded_from_slots():
    records = pd.concat(
        [
            _snapshot(),
            pd.DataFrame(
                [
                    {
                        **_snapshot().iloc[1].to_dict(),
                        "model_name": "lightgbm_multiclass_experiment",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    eligibility = assess_scheduled_official_eligibility(_contract(), records)

    assert eligibility["eligible_for_official"] is True
    assert eligibility["complete_target_set"] is True
    assert eligibility["ineligible_records"][0]["reason_code"] == "non_production_model_slot"
    assert "ineligible_records_ignored" in eligibility["reason_codes"]


def test_scheduled_idempotency_is_stable_for_retry_and_catch_up():
    scheduled = build_scheduled_idempotency_key(TICKER, FEATURE_DATE)
    retry = build_prediction_idempotency_key(
        ticker=TICKER,
        feature_date=FEATURE_DATE,
        run_type="scheduled_live_dual_write",
    )
    catch_up = _contract(execution_mode=CATCH_UP_AFTER_MISSED_SCHEDULE).idempotency_key

    assert scheduled == retry == catch_up
    assert scheduled == (
        "scheduled_live_dual_write:7203.T:2026-07-16:binary_multiclass"
    )


def test_manual_idempotency_defaults_to_distinct_keys():
    first = build_prediction_idempotency_key(
        ticker=TICKER,
        feature_date=FEATURE_DATE,
        run_type=MANUAL_LIVE_DUAL_WRITE,
    )
    second = build_prediction_idempotency_key(
        ticker=TICKER,
        feature_date=FEATURE_DATE,
        run_type=MANUAL_LIVE_DUAL_WRITE,
    )

    assert first != second


def test_contract_notes_include_required_metadata():
    contract = _contract(
        execution_mode=CATCH_UP_AFTER_MISSED_SCHEDULE,
        task_last_result=0,
    )
    payload = json.loads(contract.as_notes_json())

    assert payload["schema"] == "prediction_append_only_scheduled_contract/v1"
    assert payload["contract"]["feature_date"] == FEATURE_DATE
    assert payload["contract"]["execution_mode"] == CATCH_UP_AFTER_MISSED_SCHEDULE
    assert payload["contract"]["task_last_result"] == 0


def test_dry_run_does_not_change_any_database_rows(tmp_path):
    db_path = _database_with_snapshot(tmp_path)
    before = _database_state(db_path)

    result = run_scheduled_contract_check(
        db_path,
        _contract(),
        production_database_path=db_path,
    )

    assert result["mode"] == "dry-run"
    assert result["database_modified"] is False
    assert result["eligibility"]["eligible_for_official"] is True
    assert result["before"]["prediction_runs_count"] == 0
    assert result["after"]["prediction_runs_count"] == 0
    assert before == _database_state(db_path)


def test_temp_database_write_keeps_snapshot_and_flags_unchanged(tmp_path):
    db_path = _database_with_snapshot(tmp_path)
    snapshot_before = _prediction_rows(db_path)

    result = run_scheduled_contract_check(
        db_path,
        _contract(),
        write=True,
        production_database_path=tmp_path / "production.db",
    )
    records = list_prediction_records(db_path, ticker=TICKER)

    assert result["database_modified"] is True
    assert result["existing_predictions_unchanged"] is True
    assert result["after"]["prediction_runs_count"] == 1
    assert result["after"]["prediction_records_count"] == 2
    assert result["eligibility"]["selected_official_candidate"] is True
    assert records["is_official"].astype(int).sum() == 0
    assert records["is_latest"].astype(int).sum() == 0
    assert snapshot_before == _prediction_rows(db_path)


def test_partial_retry_reuses_logical_run_and_becomes_complete(tmp_path):
    db_path = _database_with_snapshot(tmp_path)
    partial_contract = _contract(multiclass_success=False)
    partial = run_scheduled_contract_check(
        db_path,
        partial_contract,
        write=True,
        production_database_path=tmp_path / "production.db",
    )
    retry_contract = _contract(execution_mode=RETRY)
    retry = run_scheduled_contract_check(
        db_path,
        retry_contract,
        write=True,
        production_database_path=tmp_path / "production.db",
    )
    diagnostics = build_official_selection_diagnostics_from_db(db_path, ticker=TICKER)
    retry_run = list_prediction_runs(db_path, ticker=TICKER).iloc[0]
    retry_notes = json.loads(retry_run["notes"])

    assert partial["after"]["prediction_runs_count"] == 1
    assert partial["after"]["prediction_records_count"] == 1
    assert partial["eligibility"]["eligible_for_official"] is False
    assert retry["run_id"] == partial["run_id"]
    assert retry["after"]["prediction_runs_count"] == 1
    assert retry["after"]["prediction_records_count"] == 2
    assert retry["eligibility"]["attempt_status"] == "retry_complete"
    assert retry["eligibility"]["selected_official_candidate"] is True
    assert diagnostics["selected_official_candidate_count"] == 1
    assert diagnostics["feature_dates"][0]["selected_official_run_id"] == retry["run_id"]
    assert len(retry_notes["attempts"]) == 2
    assert retry_notes["contract"]["execution_mode"] == RETRY


def test_f2_selector_rejects_complete_run_with_stale_f3_contract(tmp_path):
    db_path = _database_with_snapshot(tmp_path)
    result = run_scheduled_contract_check(
        db_path,
        _contract(stale_at_run_time=True, data_freshness_status="WARN"),
        write=True,
        production_database_path=tmp_path / "production.db",
    )
    diagnostics = build_official_selection_diagnostics_from_db(db_path, ticker=TICKER)
    scheduled = diagnostics["feature_dates"][0]["scheduled_run_diagnostics"][0]

    assert result["eligibility"]["eligible_for_official"] is False
    assert diagnostics["selected_official_candidate_count"] == 0
    assert scheduled["is_complete"] is True
    assert scheduled["contract_metadata_available"] is True
    assert scheduled["contract_metadata_eligible"] is False
    assert "stale_at_run_time" in scheduled["reason_codes"]


def test_production_database_write_is_refused(tmp_path):
    db_path = _database_with_snapshot(tmp_path)

    with pytest.raises(ValueError, match="production database"):
        run_scheduled_contract_check(
            db_path,
            _contract(),
            write=True,
            production_database_path=db_path,
        )


def test_cli_defaults_to_dry_run_and_does_not_modify_database(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    db_path = _database_with_snapshot(tmp_path)
    before = _database_state(db_path)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-scheduled-contract-check",
            "--ticker",
            TICKER,
            "--feature-date",
            FEATURE_DATE,
            "--expected-feature-date",
            FEATURE_DATE,
            "--database-path",
            str(db_path),
            "--data-freshness-status",
            "OK",
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "mode: dry-run" in result.stdout
    assert "eligible for official: true" in result.stdout
    assert "contract selected official candidate: true" in result.stdout
    assert "selected official run id in DB: none" in result.stdout
    assert "database modified: false" in result.stdout
    assert before == _database_state(db_path)


def _contract(**overrides):
    values = {
        "ticker": TICKER,
        "feature_date": FEATURE_DATE,
        "expected_feature_date": FEATURE_DATE,
        "scheduled_for": "2026-07-16T10:30:00+09:00",
        "execution_started_at": "2026-07-16T10:30:01+09:00",
        "execution_completed_at": "2026-07-16T10:30:30+09:00",
        "latest_raw_date": FEATURE_DATE,
        "latest_feature_date": FEATURE_DATE,
        "expected_latest_trading_date": FEATURE_DATE,
    }
    values.update(overrides)
    return build_scheduled_dual_write_contract(**values)


def _snapshot(*, source="live"):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([FEATURE_DATE, FEATURE_DATE]),
            "ticker": [TICKER, TICKER],
            "target": ["next_close_binary", "next_close_multiclass"],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass"],
            "predicted_class": [1, 2],
            "confidence": [0.61, 0.48],
            "prediction_type": [source, source],
        }
    )


def _database_with_snapshot(tmp_path):
    db_path = tmp_path / "market.db"
    init_database(db_path)
    upsert_predictions(_snapshot(), db_path)
    return db_path


def _database_state(db_path):
    with sqlite3.connect(db_path) as connection:
        return {
            "predictions": connection.execute(
                "SELECT * FROM predictions ORDER BY date, target, model_name"
            ).fetchall(),
            "runs": connection.execute(
                "SELECT * FROM prediction_runs ORDER BY run_id"
            ).fetchall(),
            "records": connection.execute(
                "SELECT * FROM prediction_records ORDER BY record_id"
            ).fetchall(),
        }


def _prediction_rows(db_path):
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            "SELECT * FROM predictions ORDER BY date, target, model_name"
        ).fetchall()
