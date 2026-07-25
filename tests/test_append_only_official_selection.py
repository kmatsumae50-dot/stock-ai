import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stock_ai.backtest.append_only_policy import (
    BACKFILL,
    DRY_RUN,
    EXPERIMENT,
    MANUAL_LIVE_CHECK,
    MANUAL_LIVE_DUAL_WRITE,
    RUN_TYPE_POLICIES,
    SCHEDULED_LIVE_DUAL_WRITE,
    SHADOW,
    assess_append_only_run_completeness,
    build_official_selection_diagnostics,
    classify_append_only_run_type,
    classify_prediction_model_slot,
    get_official_candidate_records_by_feature_date,
    get_production_model_slots,
    is_production_model_slot,
    select_first_complete_scheduled_run,
)
from stock_ai.storage.sqlite_store import (
    append_prediction_record,
    create_prediction_run,
    init_database,
    upsert_predictions,
)


TICKER = "7203.T"
FEATURE_DATE = "2026-07-09"


def test_run_type_policies_and_production_slots_are_explicit():
    assert set(RUN_TYPE_POLICIES) == {
        SCHEDULED_LIVE_DUAL_WRITE,
        MANUAL_LIVE_DUAL_WRITE,
        MANUAL_LIVE_CHECK,
        DRY_RUN,
        BACKFILL,
        SHADOW,
        EXPERIMENT,
    }
    scheduled = classify_append_only_run_type(SCHEDULED_LIVE_DUAL_WRITE)
    manual = classify_append_only_run_type(MANUAL_LIVE_DUAL_WRITE)
    slots = get_production_model_slots()

    assert scheduled.eligible_for_official is True
    assert scheduled.eligible_for_live_metrics is True
    assert scheduled.requires_complete_target_set is True
    assert scheduled.allowed_sources == ("live",)
    assert manual.eligible_for_official is False
    assert manual.audit_only is True
    assert [(slot.slot_name, slot.target, slot.model_name) for slot in slots] == [
        ("binary", "next_close_binary", "lightgbm_binary"),
        ("multiclass", "next_close_multiclass", "lightgbm_multiclass"),
    ]
    assert classify_prediction_model_slot("next_close_binary", "lightgbm_binary") == "binary"
    assert is_production_model_slot("next_close_multiclass", "lightgbm_multiclass") is True
    assert is_production_model_slot("next_close_multiclass", "experimental_model") is False


def test_complete_scheduled_run_is_selected_with_separate_target_records():
    runs = _runs(_run("scheduled-1", "2026-07-09T10:30:00Z"))
    records = _records_for_run("scheduled-1")

    assessment = assess_append_only_run_completeness(
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
        run_id="scheduled-1",
    )
    selection = select_first_complete_scheduled_run(
        runs,
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
    )
    selected_records = get_official_candidate_records_by_feature_date(
        runs,
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
    )

    assert assessment["is_complete"] is True
    assert assessment["present_slots"] == ["binary", "multiclass"]
    assert assessment["missing_slots"] == []
    assert assessment["duplicate_slots"] == []
    assert selection["selected_run"]["run_id"] == "scheduled-1"
    assert set(selected_records["record_id"]) == {
        "scheduled-1-binary",
        "scheduled-1-multiclass",
    }


@pytest.mark.parametrize(
    "run_type",
    [MANUAL_LIVE_DUAL_WRITE, MANUAL_LIVE_CHECK, BACKFILL, SHADOW, EXPERIMENT],
)
def test_non_scheduled_run_types_are_never_selected(run_type):
    source = run_type if run_type in {BACKFILL, SHADOW, EXPERIMENT} else "live"
    runs = _runs(_run("non-scheduled", "2026-07-09T10:30:00Z", run_type, source))
    records = _records_for_run("non-scheduled", source=source)

    diagnostics = build_official_selection_diagnostics(
        runs,
        records,
        ticker=TICKER,
    )

    assert diagnostics["selected_official_candidate_count"] == 0
    assert diagnostics["feature_dates"][0]["selected_official_run_id"] is None
    assert "no_scheduled_official_candidate" in diagnostics["classifications"]


def test_partial_scheduled_run_is_not_selected():
    runs = _runs(_run("partial", "2026-07-09T10:30:00Z"))
    records = pd.DataFrame([_record("partial", "binary")])

    diagnostics = build_official_selection_diagnostics(runs, records, ticker=TICKER)
    feature = diagnostics["feature_dates"][0]

    assert feature["selected_official_run_id"] is None
    assert feature["partial_scheduled_run_count"] == 1
    assert feature["missing_slots"] == ["multiclass"]


def test_retry_complete_run_is_selected_after_partial_scheduled_run():
    runs = _runs(
        _run("first-partial", "2026-07-09T10:30:00Z"),
        _run("retry-complete", "2026-07-09T10:35:00Z"),
    )
    records = pd.DataFrame(
        [
            _record("first-partial", "binary"),
            *_records_for_run("retry-complete").to_dict(orient="records"),
        ]
    )

    diagnostics = build_official_selection_diagnostics(runs, records, ticker=TICKER)
    feature = diagnostics["feature_dates"][0]

    assert feature["selected_official_run_id"] == "retry-complete"
    assert feature["complete_scheduled_run_count"] == 1
    assert feature["partial_scheduled_run_count"] == 1
    assert "retry_complete_run_selected" in feature["reason_codes"]


def test_first_complete_scheduled_run_wins_and_latest_can_differ():
    runs = _runs(
        _run("scheduled-first", "2026-07-09T10:30:00Z"),
        _run("scheduled-later", "2026-07-09T10:40:00Z"),
        _run("manual-latest", "2026-07-09T10:50:00Z", MANUAL_LIVE_DUAL_WRITE),
    )
    records = pd.concat(
        [
            _records_for_run("scheduled-first"),
            _records_for_run("scheduled-later"),
            _records_for_run("manual-latest"),
        ],
        ignore_index=True,
    )

    diagnostics = build_official_selection_diagnostics(runs, records, ticker=TICKER)
    feature = diagnostics["feature_dates"][0]

    assert feature["selected_official_run_id"] == "scheduled-first"
    assert feature["latest_run_id"] == "manual-latest"
    assert feature["complete_scheduled_run_count"] == 2
    assert "multiple_complete_scheduled_runs" in feature["reason_codes"]
    assert "latest_run_differs_from_official_candidate" in feature["reason_codes"]


def test_non_production_record_is_excluded_without_hiding_complete_production_slots():
    runs = _runs(_run("scheduled", "2026-07-09T10:30:00Z"))
    records = pd.concat(
        [
            _records_for_run("scheduled"),
            pd.DataFrame(
                [
                    {
                        **_record("scheduled", "multiclass"),
                        "record_id": "scheduled-experiment",
                        "model_name": "lightgbm_multiclass_experiment",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    selection = select_first_complete_scheduled_run(
        runs,
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
    )
    assessment = selection["selected_assessment"]

    assert selection["selected_run"]["run_id"] == "scheduled"
    assert assessment["is_complete"] is True
    assert assessment["ineligible_records"][0]["reason_code"] == "non_production_model_slot"
    assert set(selection["selected_records"]["record_id"]) == {
        "scheduled-binary",
        "scheduled-multiclass",
    }


def test_duplicate_target_is_detected_and_run_is_not_selected():
    runs = _runs(_run("duplicate", "2026-07-09T10:30:00Z"))
    duplicate_binary = {
        **_record("duplicate", "binary"),
        "record_id": "duplicate-binary-second",
    }
    records = pd.DataFrame(
        [
            *_records_for_run("duplicate").to_dict(orient="records"),
            duplicate_binary,
        ]
    )

    assessment = assess_append_only_run_completeness(
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
        run_id="duplicate",
    )
    selection = select_first_complete_scheduled_run(
        runs,
        records,
        ticker=TICKER,
        feature_date=FEATURE_DATE,
    )

    assert assessment["is_complete"] is False
    assert assessment["duplicate_slots"] == ["binary"]
    assert "duplicate_binary_slot" in assessment["reason_codes"]
    assert selection["selected_run"] is None


def test_same_timestamp_selection_is_deterministic_by_run_id():
    runs = _runs(
        _run("run-b", "2026-07-09T10:30:00Z"),
        _run("run-a", "2026-07-09T10:30:00Z"),
    )
    records = pd.concat(
        [_records_for_run("run-b"), _records_for_run("run-a")],
        ignore_index=True,
    )

    diagnostics = build_official_selection_diagnostics(runs, records, ticker=TICKER)
    feature = diagnostics["feature_dates"][0]

    assert feature["selected_official_run_id"] == "run-a"
    assert feature["latest_run_id"] == "run-b"


def test_snapshot_comparison_and_current_flags_are_diagnostics_only():
    runs = _runs(_run("scheduled", "2026-07-09T10:30:00Z"))
    records = _records_for_run("scheduled", is_official=0, is_latest=0)
    snapshot = _snapshot()

    diagnostics = build_official_selection_diagnostics(
        runs,
        records,
        ticker=TICKER,
        snapshot=snapshot,
    )
    feature = diagnostics["feature_dates"][0]

    assert feature["selected_official_run_id"] == "scheduled"
    assert feature["selected_records_currently_official"] is False
    assert feature["selected_records_currently_latest"] is False
    assert feature["snapshot_matches"] is True
    assert feature["snapshot_compare_status"] == "match"
    assert records["is_official"].tolist() == [0, 0]
    assert records["is_latest"].tolist() == [0, 0]


def test_read_only_cli_does_not_change_append_only_or_snapshot_rows(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    db_path = tmp_path / "market.db"
    init_database(db_path)
    upsert_predictions(_snapshot(), db_path)
    create_prediction_run(
        db_path,
        run_id="manual-only",
        run_timestamp="2026-07-09T10:30:00Z",
        ticker=TICKER,
        run_type=MANUAL_LIVE_DUAL_WRITE,
        source="live",
        latest_feature_date=FEATURE_DATE,
    )
    for slot in ("binary", "multiclass"):
        record = _record("manual-only", slot, is_official=1, is_latest=1)
        append_prediction_record(
            db_path,
            run_id=record["run_id"],
            ticker=record["ticker"],
            feature_date=record["feature_date"],
            target=record["target"],
            model_name=record["model_name"],
            source=record["source"],
            record_id=record["record_id"],
            predicted_label=record["predicted_label"],
            confidence=record["confidence"],
            is_official=True,
            is_latest=True,
        )
    before = _database_state(db_path)

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "prediction-append-only-official-selection-check",
            "--ticker",
            TICKER,
            "--database-path",
            str(db_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    after = _database_state(db_path)

    assert "mode: read-only" in result.stdout
    assert "selected official run id: none" in result.stdout
    assert '"manual_only"' in result.stdout
    assert "current DB official record count: 2" in result.stdout
    assert before == after


def _run(
    run_id,
    run_timestamp,
    run_type=SCHEDULED_LIVE_DUAL_WRITE,
    source="live",
):
    return {
        "run_id": run_id,
        "run_timestamp": run_timestamp,
        "ticker": TICKER,
        "run_type": run_type,
        "source": source,
        "latest_feature_date": FEATURE_DATE,
    }


def _runs(*rows):
    return pd.DataFrame(rows)


def _record(run_id, slot, *, source="live", is_official=0, is_latest=0):
    if slot == "binary":
        target = "next_close_binary"
        model_name = "lightgbm_binary"
        predicted_label = "1"
        confidence = 0.61
    else:
        target = "next_close_multiclass"
        model_name = "lightgbm_multiclass"
        predicted_label = "2"
        confidence = 0.48
    return {
        "record_id": f"{run_id}-{slot}",
        "run_id": run_id,
        "ticker": TICKER,
        "feature_date": FEATURE_DATE,
        "target": target,
        "model_name": model_name,
        "model_type": slot,
        "predicted_label": predicted_label,
        "source": source,
        "confidence": confidence,
        "is_official": is_official,
        "is_latest": is_latest,
    }


def _records_for_run(run_id, *, source="live", is_official=0, is_latest=0):
    return pd.DataFrame(
        [
            _record(
                run_id,
                "binary",
                source=source,
                is_official=is_official,
                is_latest=is_latest,
            ),
            _record(
                run_id,
                "multiclass",
                source=source,
                is_official=is_official,
                is_latest=is_latest,
            ),
        ]
    )


def _snapshot():
    return pd.DataFrame(
        {
            "date": pd.to_datetime([FEATURE_DATE, FEATURE_DATE]),
            "ticker": [TICKER, TICKER],
            "target": ["next_close_binary", "next_close_multiclass"],
            "model_name": ["lightgbm_binary", "lightgbm_multiclass"],
            "predicted_class": [1, 2],
            "confidence": [0.61, 0.48],
            "prediction_type": ["live", "live"],
        }
    )


def _database_state(db_path):
    with sqlite3.connect(db_path) as connection:
        return {
            "runs": connection.execute(
                "SELECT * FROM prediction_runs ORDER BY run_id"
            ).fetchall(),
            "records": connection.execute(
                "SELECT * FROM prediction_records ORDER BY record_id"
            ).fetchall(),
            "predictions": connection.execute(
                "SELECT * FROM predictions ORDER BY date, target, model_name"
            ).fetchall(),
        }
