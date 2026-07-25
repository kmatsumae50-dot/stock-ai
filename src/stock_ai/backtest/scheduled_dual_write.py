from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from stock_ai.backtest.append_only_policy import (
    PRODUCTION_SLOT_NAMES,
    SCHEDULED_LIVE_DUAL_WRITE,
    classify_prediction_model_slot,
    select_first_complete_scheduled_run,
)


ON_SCHEDULE = "on_schedule"
DELAYED_START = "delayed_start"
CATCH_UP_AFTER_MISSED_SCHEDULE = "catch_up_after_missed_schedule"
MANUAL = "manual"
RETRY = "retry"
UNKNOWN = "unknown"

EXECUTION_MODES = (
    ON_SCHEDULE,
    DELAYED_START,
    CATCH_UP_AFTER_MISSED_SCHEDULE,
    MANUAL,
    RETRY,
    UNKNOWN,
)
OFFICIAL_EXECUTION_MODES = (
    ON_SCHEDULE,
    DELAYED_START,
    CATCH_UP_AFTER_MISSED_SCHEDULE,
    RETRY,
)
CONTRACT_SCHEMA = "prediction_append_only_scheduled_contract/v1"
TARGET_SET_VERSION = "binary_multiclass"


@dataclass(frozen=True)
class ScheduledDualWriteContract:
    ticker: str
    feature_date: str
    expected_feature_date: str
    scheduled_for: str
    execution_started_at: str
    execution_completed_at: str
    run_type: str
    execution_mode: str
    command_name: str
    source: str
    binary_success: bool
    multiclass_success: bool
    data_freshness_status: str
    stale_at_run_time: bool
    latest_raw_date: str | None
    latest_feature_date: str | None
    expected_latest_trading_date: str | None
    task_name: str | None
    task_last_result: int | None
    notes: str | None
    idempotency_key: str
    target_set_version: str = TARGET_SET_VERSION

    def __post_init__(self) -> None:
        if not self.ticker.strip():
            raise ValueError("ticker must not be empty")
        if not self.command_name.strip():
            raise ValueError("command_name must not be empty")
        if not self.idempotency_key.strip():
            raise ValueError("idempotency_key must not be empty")
        if self.execution_mode not in EXECUTION_MODES:
            raise ValueError(f"Unsupported execution_mode: {self.execution_mode}")
        object.__setattr__(self, "feature_date", _date_text(self.feature_date))
        object.__setattr__(
            self,
            "expected_feature_date",
            _date_text(self.expected_feature_date),
        )
        for field_name in (
            "scheduled_for",
            "execution_started_at",
            "execution_completed_at",
        ):
            object.__setattr__(self, field_name, _timestamp_text(getattr(self, field_name)))
        for field_name in (
            "latest_raw_date",
            "latest_feature_date",
            "expected_latest_trading_date",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                _date_text(value) if value is not None else None,
            )

    def as_metadata(self) -> dict[str, Any]:
        return asdict(self)

    def as_notes_json(self) -> str:
        return json.dumps(
            {
                "schema": CONTRACT_SCHEMA,
                "contract": self.as_metadata(),
                "attempts": [self.as_metadata()],
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def build_scheduled_idempotency_key(
    ticker: str,
    feature_date: str | pd.Timestamp,
    *,
    target_set_version: str = TARGET_SET_VERSION,
) -> str:
    return ":".join(
        (
            SCHEDULED_LIVE_DUAL_WRITE,
            ticker.strip(),
            _date_text(feature_date),
            target_set_version,
        )
    )


def build_prediction_idempotency_key(
    *,
    ticker: str,
    feature_date: str | pd.Timestamp,
    run_type: str,
    explicit_key: str | None = None,
) -> str:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    if run_type == SCHEDULED_LIVE_DUAL_WRITE:
        return build_scheduled_idempotency_key(ticker, feature_date)
    return f"{run_type}:{ticker.strip()}:{_date_text(feature_date)}:{uuid4().hex}"


def run_id_from_idempotency_key(idempotency_key: str) -> str:
    safe = "".join(
        character if character.isalnum() else "_" for character in idempotency_key
    )
    return "_".join(part for part in safe.split("_") if part)


def build_scheduled_dual_write_contract(
    *,
    ticker: str,
    feature_date: str | pd.Timestamp,
    expected_feature_date: str | pd.Timestamp,
    scheduled_for: str | pd.Timestamp | None = None,
    execution_started_at: str | pd.Timestamp | None = None,
    execution_completed_at: str | pd.Timestamp | None = None,
    run_type: str = SCHEDULED_LIVE_DUAL_WRITE,
    execution_mode: str = ON_SCHEDULE,
    command_name: str = "prediction-append-only-scheduled-contract-check",
    source: str = "live",
    binary_success: bool = True,
    multiclass_success: bool = True,
    data_freshness_status: str = "OK",
    stale_at_run_time: bool = False,
    latest_raw_date: str | pd.Timestamp | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    expected_latest_trading_date: str | pd.Timestamp | None = None,
    task_name: str | None = "stock-ai daily predict",
    task_last_result: int | None = None,
    notes: str | None = None,
    idempotency_key: str | None = None,
) -> ScheduledDualWriteContract:
    now = _timestamp_text(None)
    final_key = build_prediction_idempotency_key(
        ticker=ticker,
        feature_date=feature_date,
        run_type=run_type,
        explicit_key=idempotency_key,
    )
    return ScheduledDualWriteContract(
        ticker=ticker,
        feature_date=_date_text(feature_date),
        expected_feature_date=_date_text(expected_feature_date),
        scheduled_for=_timestamp_text(scheduled_for or now),
        execution_started_at=_timestamp_text(execution_started_at or now),
        execution_completed_at=_timestamp_text(execution_completed_at or now),
        run_type=run_type,
        execution_mode=execution_mode,
        command_name=command_name,
        source=source.strip().lower(),
        binary_success=bool(binary_success),
        multiclass_success=bool(multiclass_success),
        data_freshness_status=str(data_freshness_status or "UNKNOWN").strip().upper(),
        stale_at_run_time=bool(stale_at_run_time),
        latest_raw_date=latest_raw_date,
        latest_feature_date=latest_feature_date,
        expected_latest_trading_date=expected_latest_trading_date,
        task_name=task_name,
        task_last_result=task_last_result,
        notes=notes,
        idempotency_key=final_key,
    )


def assess_scheduled_official_eligibility(
    contract: ScheduledDualWriteContract,
    records: pd.DataFrame,
    *,
    records_written_successfully: bool = True,
) -> dict[str, Any]:
    slot_records: dict[str, list[dict[str, Any]]] = {
        slot_name: [] for slot_name in PRODUCTION_SLOT_NAMES
    }
    ineligible_records: list[dict[str, Any]] = []
    matching_records = _matching_contract_records(records, contract)

    for row in matching_records.to_dict(orient="records"):
        source = _record_source(row, contract.source)
        slot_name = classify_prediction_model_slot(row.get("target"), row.get("model_name"))
        if source != "live":
            ineligible_records.append(_ineligible_record(row, "non_live_source"))
            continue
        if slot_name is None:
            ineligible_records.append(_ineligible_record(row, "non_production_model_slot"))
            continue
        slot_records[slot_name].append(row)

    present_slots = [slot for slot in PRODUCTION_SLOT_NAMES if slot_records[slot]]
    missing_slots = [slot for slot in PRODUCTION_SLOT_NAMES if not slot_records[slot]]
    duplicate_slots = [slot for slot in PRODUCTION_SLOT_NAMES if len(slot_records[slot]) > 1]
    complete_target_set = not missing_slots and not duplicate_slots
    delayed_or_catch_up = contract.execution_mode in {
        DELAYED_START,
        CATCH_UP_AFTER_MISSED_SCHEDULE,
    }

    metadata = assess_scheduled_contract_metadata(contract)
    blocking_reason_codes = list(metadata["blocking_reason_codes"])
    informational_reason_codes: list[str] = []
    for slot in missing_slots:
        blocking_reason_codes.append(f"missing_{slot}_slot")
    for slot in duplicate_slots:
        blocking_reason_codes.append(f"duplicate_{slot}_slot")
    if ineligible_records:
        informational_reason_codes.append("ineligible_records_ignored")
        informational_reason_codes.extend(
            sorted({row["reason_code"] for row in ineligible_records})
        )
    if not records_written_successfully:
        blocking_reason_codes.append("record_write_incomplete")

    eligible = not blocking_reason_codes
    reason_codes = [*blocking_reason_codes, *informational_reason_codes]
    reason_codes.append(
        "eligible_scheduled_official_candidate"
        if eligible
        else "not_eligible_scheduled_official_candidate"
    )
    return {
        "eligible_for_official": eligible,
        "complete_target_set": complete_target_set,
        "present_slots": present_slots,
        "missing_slots": missing_slots,
        "duplicate_slots": duplicate_slots,
        "reason_codes": _deduplicate(reason_codes),
        "execution_mode": contract.execution_mode,
        "delayed_or_catch_up": delayed_or_catch_up,
        "selected_official_candidate": eligible,
        "records_written_successfully": bool(records_written_successfully),
        "ineligible_records": ineligible_records,
        "attempt_status": _attempt_status(contract, complete_target_set),
    }


def assess_scheduled_contract_metadata(
    contract: ScheduledDualWriteContract,
) -> dict[str, Any]:
    reason_codes: list[str] = []
    if contract.run_type != SCHEDULED_LIVE_DUAL_WRITE:
        reason_codes.append("run_type_not_scheduled_live_dual_write")
    if contract.source != "live":
        reason_codes.append("run_source_not_live")
    if contract.execution_mode not in OFFICIAL_EXECUTION_MODES:
        reason_codes.append("execution_mode_not_official_eligible")
    if contract.feature_date != contract.expected_feature_date:
        reason_codes.append("feature_date_mismatch")
    if contract.stale_at_run_time:
        reason_codes.append("stale_at_run_time")
    if contract.data_freshness_status != "OK":
        reason_codes.append("data_freshness_not_ok")
    if not contract.binary_success:
        reason_codes.append("binary_prediction_failed")
    if not contract.multiclass_success:
        reason_codes.append("multiclass_prediction_failed")
    return {
        "metadata_available": True,
        "eligible": not reason_codes,
        "blocking_reason_codes": _deduplicate(reason_codes),
    }


def assess_persisted_scheduled_contract(
    run: dict[str, Any],
    *,
    feature_date: str | pd.Timestamp,
) -> dict[str, Any]:
    notes = run.get("notes")
    if notes is None or not str(notes).strip():
        return {
            "metadata_available": False,
            "eligible": True,
            "blocking_reason_codes": [],
            "reason_codes": ["scheduled_contract_metadata_not_available_legacy"],
        }
    try:
        payload = json.loads(str(notes))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {
            "metadata_available": False,
            "eligible": True,
            "blocking_reason_codes": [],
            "reason_codes": ["scheduled_contract_metadata_not_available_legacy"],
        }
    if payload.get("schema") != CONTRACT_SCHEMA or not isinstance(
        payload.get("contract"),
        dict,
    ):
        return {
            "metadata_available": False,
            "eligible": True,
            "blocking_reason_codes": [],
            "reason_codes": ["scheduled_contract_metadata_not_available_legacy"],
        }
    try:
        contract = ScheduledDualWriteContract(**payload["contract"])
    except (TypeError, ValueError) as error:
        return {
            "metadata_available": True,
            "eligible": False,
            "blocking_reason_codes": ["scheduled_contract_metadata_invalid"],
            "reason_codes": [
                "scheduled_contract_metadata_invalid",
                f"scheduled_contract_error_{type(error).__name__}",
            ],
        }
    metadata = assess_scheduled_contract_metadata(contract)
    reason_codes = list(metadata["blocking_reason_codes"])
    if contract.feature_date != _date_text(feature_date):
        reason_codes.append("persisted_contract_feature_date_mismatch")
    eligible = not reason_codes
    return {
        "metadata_available": True,
        "eligible": eligible,
        "blocking_reason_codes": _deduplicate(reason_codes),
        "reason_codes": [
            *_deduplicate(reason_codes),
            "scheduled_contract_metadata_eligible"
            if eligible
            else "scheduled_contract_metadata_ineligible",
        ],
    }


def run_scheduled_contract_check(
    database_path: str | Path,
    contract: ScheduledDualWriteContract,
    *,
    write: bool = False,
    production_database_path: str | Path | None = None,
) -> dict[str, Any]:
    from stock_ai.storage.sqlite_store import (
        filter_official_prediction_snapshot_records,
        get_prediction_run,
        list_prediction_records,
        list_prediction_runs,
        read_latest_predictions,
        summarize_append_only_records,
        update_prediction_run_notes,
        write_append_only_prediction_run,
    )

    db_path = Path(database_path)
    if write and production_database_path is not None:
        production_path = Path(production_database_path)
        if db_path.resolve() == production_path.resolve():
            raise ValueError("Scheduled contract --write is not allowed for the production database.")

    before = summarize_append_only_records(db_path, ticker=contract.ticker)
    snapshot_before = read_latest_predictions(db_path)
    raw_candidates = _matching_contract_records(snapshot_before, contract)
    included, excluded = filter_official_prediction_snapshot_records(
        snapshot_before,
        ticker=contract.ticker,
        latest_feature_date=contract.feature_date,
    )
    write_candidates = _apply_slot_success_contract(included, contract)
    preflight = assess_scheduled_official_eligibility(contract, write_candidates)
    run_id = run_id_from_idempotency_key(contract.idempotency_key)
    existing_run = get_prediction_run(db_path, run_id)
    contract_notes = _merge_contract_attempt_notes(
        existing_run.get("notes") if existing_run else None,
        contract,
    )
    write_result: dict[str, Any] | None = None

    if write:
        write_result = write_append_only_prediction_run(
            db_path,
            write_candidates,
            ticker=contract.ticker,
            run_type=contract.run_type,
            command_name=contract.command_name,
            source=contract.source,
            run_id=run_id,
            run_timestamp=contract.execution_started_at,
            working_directory=str(Path.cwd()),
            data_freshness_status=contract.data_freshness_status,
            latest_raw_date=contract.latest_raw_date,
            latest_feature_date=contract.latest_feature_date,
            expected_latest_trading_date=contract.expected_latest_trading_date,
            stale_at_run_time=contract.stale_at_run_time,
            is_official=False,
            is_latest=False,
            notes=contract_notes,
        )
        if existing_run is not None:
            update_prediction_run_notes(
                db_path,
                run_id=run_id,
                notes=contract_notes,
            )

    after = summarize_append_only_records(db_path, ticker=contract.ticker)
    run_records = list_prediction_records(db_path, ticker=contract.ticker, run_id=run_id)
    actual_eligibility = (
        assess_scheduled_official_eligibility(
            contract,
            run_records,
            records_written_successfully=_records_cover_candidates(
                run_records,
                write_candidates,
            ),
        )
        if write
        else preflight
    )
    runs = list_prediction_runs(db_path, ticker=contract.ticker)
    records = list_prediction_records(db_path, ticker=contract.ticker)
    selection = select_first_complete_scheduled_run(
        runs,
        records,
        ticker=contract.ticker,
        feature_date=contract.feature_date,
    )
    selected_run = selection.get("selected_run")
    selected_run_id = str(selected_run.get("run_id")) if selected_run else None
    current_selected = bool(
        actual_eligibility["eligible_for_official"] and selected_run_id == run_id
    )
    actual_eligibility["selected_official_candidate"] = current_selected if write else bool(
        preflight["eligible_for_official"]
    )

    return {
        "database_path": str(db_path.resolve()),
        "mode": "write" if write else "dry-run",
        "database_modified": bool(
            int(after["prediction_runs_count"]) != int(before["prediction_runs_count"])
            or int(after["prediction_records_count"])
            != int(before["prediction_records_count"])
        ),
        "run_id": run_id,
        "contract": contract.as_metadata(),
        "raw_snapshot_candidates": int(len(raw_candidates)),
        "official_filtered_candidates": int(len(included)),
        "write_candidates": int(len(write_candidates)),
        "excluded_candidates": int(len(excluded)),
        "preflight": preflight,
        "eligibility": actual_eligibility,
        "selected_official_run_id": selected_run_id,
        "before": before,
        "after": after,
        "write_result": write_result,
        "existing_predictions_unchanged": int(before["existing_predictions_count"])
        == int(after["existing_predictions_count"]),
        "flags_updated": False,
    }


def _apply_slot_success_contract(
    candidates: pd.DataFrame,
    contract: ScheduledDualWriteContract,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates.copy()
    keep = pd.Series(True, index=candidates.index)
    if not contract.binary_success:
        keep &= ~candidates.apply(
            lambda row: classify_prediction_model_slot(row.get("target"), row.get("model_name"))
            == "binary",
            axis=1,
        )
    if not contract.multiclass_success:
        keep &= ~candidates.apply(
            lambda row: classify_prediction_model_slot(row.get("target"), row.get("model_name"))
            == "multiclass",
            axis=1,
        )
    return candidates[keep].copy().reset_index(drop=True)


def _merge_contract_attempt_notes(
    existing_notes: object,
    contract: ScheduledDualWriteContract,
) -> str:
    attempts: list[dict[str, Any]] = []
    if existing_notes is not None and str(existing_notes).strip():
        try:
            payload = json.loads(str(existing_notes))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        if payload.get("schema") == CONTRACT_SCHEMA:
            raw_attempts = payload.get("attempts")
            if isinstance(raw_attempts, list):
                attempts.extend(item for item in raw_attempts if isinstance(item, dict))
            elif isinstance(payload.get("contract"), dict):
                attempts.append(payload["contract"])
    latest = contract.as_metadata()
    if latest not in attempts:
        attempts.append(latest)
    return json.dumps(
        {"schema": CONTRACT_SCHEMA, "contract": latest, "attempts": attempts},
        ensure_ascii=False,
        sort_keys=True,
    )


def _matching_contract_records(
    records: pd.DataFrame,
    contract: ScheduledDualWriteContract,
) -> pd.DataFrame:
    if records.empty:
        return records.copy()
    rows = records.copy()
    if "ticker" in rows:
        rows = rows[rows["ticker"].astype(str).eq(contract.ticker)]
    date_column = "feature_date" if "feature_date" in rows else "date"
    if date_column in rows:
        dates = rows[date_column].map(lambda value: _optional_date_text(value))
        rows = rows[dates.eq(contract.feature_date)]
    return rows.copy().reset_index(drop=True)


def _record_source(row: dict[str, Any], fallback: str) -> str:
    for key in ("snapshot_source", "prediction_type", "source"):
        value = row.get(key)
        if value is not None and not pd.isna(value) and str(value).strip():
            return str(value).strip().lower()
    return fallback


def _ineligible_record(row: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "target": row.get("target"),
        "model_name": row.get("model_name"),
        "source": _record_source(row, "unknown"),
        "reason_code": reason_code,
    }


def _records_cover_candidates(records: pd.DataFrame, candidates: pd.DataFrame) -> bool:
    if candidates.empty:
        return True
    if records.empty:
        return False
    actual = {
        (str(row.target), str(row.model_name))
        for row in records.itertuples(index=False)
    }
    expected = {
        (str(row.target), str(row.model_name))
        for row in candidates.itertuples(index=False)
    }
    return expected <= actual


def _attempt_status(
    contract: ScheduledDualWriteContract,
    complete_target_set: bool,
) -> str:
    if not contract.binary_success and not contract.multiclass_success:
        return "failed"
    if not complete_target_set or not contract.binary_success or not contract.multiclass_success:
        return "partial"
    if contract.execution_mode == RETRY:
        return "retry_complete"
    if contract.execution_mode == CATCH_UP_AFTER_MISSED_SCHEDULE:
        return "catch_up_complete"
    return "complete"


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _date_text(value: object) -> str:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"Invalid date value: {value}")
    return parsed.strftime("%Y-%m-%d")


def _optional_date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _date_text(value)


def _timestamp_text(value: object | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat()
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"Invalid timestamp value: {value}")
    return parsed.isoformat()
