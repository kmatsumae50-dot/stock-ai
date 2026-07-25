from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isclose
from pathlib import Path
from typing import Any

import pandas as pd


SCHEDULED_LIVE_DUAL_WRITE = "scheduled_live_dual_write"
MANUAL_LIVE_DUAL_WRITE = "manual_live_dual_write"
MANUAL_LIVE_CHECK = "manual_live_check"
DRY_RUN = "dry_run"
BACKFILL = "backfill"
SHADOW = "shadow"
EXPERIMENT = "experiment"


@dataclass(frozen=True)
class RunTypePolicy:
    run_type: str
    eligible_for_official: bool
    eligible_for_live_metrics: bool
    audit_only: bool
    expected_targets: tuple[str, ...]
    requires_complete_target_set: bool
    allowed_sources: tuple[str, ...]
    priority: int


@dataclass(frozen=True)
class ProductionModelSlot:
    slot_name: str
    target: str
    model_name: str
    model_type: str
    allowed_source: str = "live"


PRODUCTION_MODEL_SLOTS = (
    ProductionModelSlot(
        slot_name="binary",
        target="next_close_binary",
        model_name="lightgbm_binary",
        model_type="binary",
    ),
    ProductionModelSlot(
        slot_name="multiclass",
        target="next_close_multiclass",
        model_name="lightgbm_multiclass",
        model_type="multiclass",
    ),
)

PRODUCTION_TARGETS = tuple(slot.target for slot in PRODUCTION_MODEL_SLOTS)
PRODUCTION_SLOT_NAMES = tuple(slot.slot_name for slot in PRODUCTION_MODEL_SLOTS)

RUN_TYPE_POLICIES = {
    SCHEDULED_LIVE_DUAL_WRITE: RunTypePolicy(
        run_type=SCHEDULED_LIVE_DUAL_WRITE,
        eligible_for_official=True,
        eligible_for_live_metrics=True,
        audit_only=False,
        expected_targets=PRODUCTION_TARGETS,
        requires_complete_target_set=True,
        allowed_sources=("live",),
        priority=10,
    ),
    MANUAL_LIVE_DUAL_WRITE: RunTypePolicy(
        run_type=MANUAL_LIVE_DUAL_WRITE,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=PRODUCTION_TARGETS,
        requires_complete_target_set=True,
        allowed_sources=("live",),
        priority=20,
    ),
    MANUAL_LIVE_CHECK: RunTypePolicy(
        run_type=MANUAL_LIVE_CHECK,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=PRODUCTION_TARGETS,
        requires_complete_target_set=False,
        allowed_sources=("live",),
        priority=30,
    ),
    DRY_RUN: RunTypePolicy(
        run_type=DRY_RUN,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=(),
        requires_complete_target_set=False,
        allowed_sources=(),
        priority=90,
    ),
    BACKFILL: RunTypePolicy(
        run_type=BACKFILL,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=(),
        requires_complete_target_set=False,
        allowed_sources=("backfill",),
        priority=60,
    ),
    SHADOW: RunTypePolicy(
        run_type=SHADOW,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=(),
        requires_complete_target_set=False,
        allowed_sources=("shadow",),
        priority=70,
    ),
    EXPERIMENT: RunTypePolicy(
        run_type=EXPERIMENT,
        eligible_for_official=False,
        eligible_for_live_metrics=False,
        audit_only=True,
        expected_targets=(),
        requires_complete_target_set=False,
        allowed_sources=("experiment",),
        priority=80,
    ),
}

UNKNOWN_RUN_TYPE_POLICY = RunTypePolicy(
    run_type="unknown",
    eligible_for_official=False,
    eligible_for_live_metrics=False,
    audit_only=True,
    expected_targets=(),
    requires_complete_target_set=False,
    allowed_sources=(),
    priority=999,
)


def classify_append_only_run_type(run_type: object) -> RunTypePolicy:
    normalized = str(run_type or "").strip().lower()
    return RUN_TYPE_POLICIES.get(normalized, UNKNOWN_RUN_TYPE_POLICY)


def get_production_model_slots() -> tuple[ProductionModelSlot, ...]:
    return PRODUCTION_MODEL_SLOTS


def classify_prediction_model_slot(target: object, model_name: object) -> str | None:
    normalized_target = str(target or "").strip()
    normalized_model = str(model_name or "").strip()
    for slot in PRODUCTION_MODEL_SLOTS:
        if slot.target == normalized_target and slot.model_name == normalized_model:
            return slot.slot_name
    return None


def is_production_model_slot(target: object, model_name: object) -> bool:
    return classify_prediction_model_slot(target, model_name) is not None


def assess_append_only_run_completeness(
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str | pd.Timestamp,
    run_id: str,
) -> dict[str, Any]:
    normalized_date = _date_text(feature_date)
    matching = _matching_records(
        records,
        ticker=ticker,
        feature_date=normalized_date,
        run_id=run_id,
    )
    slot_records: dict[str, list[dict[str, Any]]] = {
        slot_name: [] for slot_name in PRODUCTION_SLOT_NAMES
    }
    ineligible_records: list[dict[str, Any]] = []

    for row in matching.to_dict(orient="records"):
        source = str(row.get("source") or "").strip().lower()
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
    selected_record_ids = {
        slot: str(slot_records[slot][0].get("record_id"))
        for slot in PRODUCTION_SLOT_NAMES
        if len(slot_records[slot]) == 1
    }
    is_complete = not missing_slots and not duplicate_slots

    reason_codes: list[str] = []
    if matching.empty:
        reason_codes.append("no_records")
    for slot in missing_slots:
        reason_codes.append(f"missing_{slot}_slot")
    for slot in duplicate_slots:
        reason_codes.append(f"duplicate_{slot}_slot")
    if ineligible_records:
        reason_codes.append("ineligible_records_ignored")
        reason_codes.extend(
            sorted({str(row["reason_code"]) for row in ineligible_records})
        )
    reason_codes.append("complete_run" if is_complete else "partial_run")

    return {
        "run_id": run_id,
        "ticker": ticker,
        "feature_date": normalized_date,
        "is_complete": is_complete,
        "record_count": int(len(matching)),
        "present_slots": present_slots,
        "missing_slots": missing_slots,
        "duplicate_slots": duplicate_slots,
        "selected_record_ids": selected_record_ids,
        "ineligible_records": ineligible_records,
        "reason_codes": _deduplicate(reason_codes),
    }


def select_first_complete_scheduled_run(
    runs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str | pd.Timestamp,
) -> dict[str, Any]:
    normalized_date = _date_text(feature_date)
    feature_runs = _runs_for_feature_date(runs, records, ticker=ticker, feature_date=normalized_date)
    scheduled = feature_runs[
        _column(feature_runs, "run_type").astype(str).eq(SCHEDULED_LIVE_DUAL_WRITE)
    ].copy()
    candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    assessments: list[dict[str, Any]] = []

    for run in scheduled.to_dict(orient="records"):
        from stock_ai.backtest.scheduled_dual_write import (
            assess_persisted_scheduled_contract,
        )

        run_id = str(run.get("run_id") or "")
        assessment = assess_append_only_run_completeness(
            records,
            ticker=ticker,
            feature_date=normalized_date,
            run_id=run_id,
        )
        run_source = str(run.get("source") or "").strip().lower()
        source_allowed = run_source in RUN_TYPE_POLICIES[
            SCHEDULED_LIVE_DUAL_WRITE
        ].allowed_sources
        contract_metadata = assess_persisted_scheduled_contract(
            run,
            feature_date=normalized_date,
        )
        assessment["run_source_allowed"] = source_allowed
        assessment["contract_metadata_available"] = contract_metadata[
            "metadata_available"
        ]
        assessment["contract_metadata_eligible"] = contract_metadata["eligible"]
        assessment["eligible_for_official"] = bool(
            assessment["is_complete"]
            and source_allowed
            and contract_metadata["eligible"]
        )
        if not source_allowed:
            assessment["reason_codes"] = _deduplicate(
                [*assessment["reason_codes"], "run_source_not_allowed"]
            )
        assessment["reason_codes"] = _deduplicate(
            [*assessment["reason_codes"], *contract_metadata["reason_codes"]]
        )
        assessments.append(assessment)
        if assessment["eligible_for_official"]:
            candidates.append((run, assessment))

    candidates.sort(key=lambda item: _earliest_run_sort_key(item[0]))
    selected_run = candidates[0][0] if candidates else None
    selected_assessment = candidates[0][1] if candidates else None
    selected_ids = set(
        (selected_assessment or {}).get("selected_record_ids", {}).values()
    )
    selected_records = records[
        _column(records, "record_id").astype(str).isin(selected_ids)
    ].copy()
    selected_records = selected_records.sort_values(
        [column for column in ("target", "model_name", "record_id") if column in selected_records],
        kind="stable",
    ).reset_index(drop=True) if not selected_records.empty else selected_records

    return {
        "selected_run": selected_run,
        "selected_records": selected_records,
        "selected_assessment": selected_assessment,
        "scheduled_run_assessments": assessments,
    }


def get_official_candidate_records_by_feature_date(
    runs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str | pd.Timestamp,
) -> pd.DataFrame:
    selection = select_first_complete_scheduled_run(
        runs,
        records,
        ticker=ticker,
        feature_date=feature_date,
    )
    return selection["selected_records"].copy()


def build_official_selection_diagnostics(
    runs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    ticker: str,
    snapshot: pd.DataFrame | None = None,
) -> dict[str, Any]:
    ticker_runs = _filter_equal(runs, "ticker", ticker)
    ticker_records = _filter_equal(records, "ticker", ticker)
    feature_dates = _feature_dates(ticker_runs, ticker_records)
    date_diagnostics = [
        _feature_date_diagnostics(
            ticker_runs,
            ticker_records,
            ticker=ticker,
            feature_date=feature_date,
            snapshot=snapshot,
        )
        for feature_date in feature_dates
    ]

    run_types = _column(ticker_runs, "run_type").fillna("").astype(str)
    scheduled_run_count = int(run_types.eq(SCHEDULED_LIVE_DUAL_WRITE).sum())
    manual_run_count = int(run_types.isin((MANUAL_LIVE_DUAL_WRITE, MANUAL_LIVE_CHECK)).sum())
    selected_count = sum(
        1 for item in date_diagnostics if item["selected_official_run_id"] is not None
    )
    classifications: list[str] = []
    if ticker_runs.empty:
        classifications.append("no_append_only_runs")
    if scheduled_run_count == 0 and manual_run_count > 0:
        classifications.append("manual_only")
    if selected_count == 0:
        classifications.append("no_scheduled_official_candidate")
    else:
        classifications.append("scheduled_official_candidate_available")

    latest_run = _latest_run(ticker_runs)
    return {
        "ticker": ticker,
        "run_type_policies": {
            name: asdict(policy) for name, policy in RUN_TYPE_POLICIES.items()
        },
        "production_model_slots": [asdict(slot) for slot in PRODUCTION_MODEL_SLOTS],
        "feature_date_count": len(feature_dates),
        "scheduled_run_count": scheduled_run_count,
        "manual_run_count": manual_run_count,
        "complete_scheduled_run_count": sum(
            int(item["complete_scheduled_run_count"]) for item in date_diagnostics
        ),
        "partial_scheduled_run_count": sum(
            int(item["partial_scheduled_run_count"]) for item in date_diagnostics
        ),
        "selected_official_candidate_count": selected_count,
        "latest_run_id": latest_run.get("run_id") if latest_run else None,
        "latest_run_type": latest_run.get("run_type") if latest_run else None,
        "classifications": classifications,
        "feature_dates": date_diagnostics,
        "read_only": True,
    }


def build_official_selection_diagnostics_from_db(
    path: str | Path,
    *,
    ticker: str,
) -> dict[str, Any]:
    from stock_ai.storage.sqlite_store import (
        list_prediction_records,
        list_prediction_runs,
        read_latest_predictions,
    )

    database_path = Path(path)
    runs = list_prediction_runs(database_path, ticker=ticker)
    records = list_prediction_records(database_path, ticker=ticker)
    snapshot = read_latest_predictions(database_path)
    if not snapshot.empty and "ticker" in snapshot:
        snapshot = snapshot[snapshot["ticker"].astype(str).eq(ticker)].reset_index(drop=True)
    return build_official_selection_diagnostics(
        runs,
        records,
        ticker=ticker,
        snapshot=snapshot,
    )


def _feature_date_diagnostics(
    runs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str,
    snapshot: pd.DataFrame | None,
) -> dict[str, Any]:
    feature_runs = _runs_for_feature_date(
        runs,
        records,
        ticker=ticker,
        feature_date=feature_date,
    )
    selection = select_first_complete_scheduled_run(
        feature_runs,
        records,
        ticker=ticker,
        feature_date=feature_date,
    )
    assessments = selection["scheduled_run_assessments"]
    selected_run = selection["selected_run"]
    selected_records = selection["selected_records"]
    latest_run = _latest_run(feature_runs)

    run_types = _column(feature_runs, "run_type").fillna("").astype(str)
    scheduled_count = int(run_types.eq(SCHEDULED_LIVE_DUAL_WRITE).sum())
    manual_count = int(run_types.isin((MANUAL_LIVE_DUAL_WRITE, MANUAL_LIVE_CHECK)).sum())
    complete_count = sum(1 for item in assessments if item["eligible_for_official"])
    partial_count = scheduled_count - complete_count
    selected_run_id = str(selected_run.get("run_id")) if selected_run else None
    selected_ids = (
        selection["selected_assessment"].get("selected_record_ids", {})
        if selection["selected_assessment"]
        else {}
    )

    feature_records = _filter_date_equal(records, "feature_date", feature_date)
    feature_records = _filter_equal(feature_records, "ticker", ticker)
    current_official = feature_records[_truthy_series(feature_records, "is_official")]
    current_latest = feature_records[_truthy_series(feature_records, "is_latest")]
    selected_current_official = (
        bool(_truthy_series(selected_records, "is_official").all())
        if not selected_records.empty
        else None
    )
    selected_current_latest = (
        bool(_truthy_series(selected_records, "is_latest").all())
        if not selected_records.empty
        else None
    )
    snapshot_comparison = _compare_selected_records_to_snapshot(
        selected_records,
        snapshot if snapshot is not None else pd.DataFrame(),
        ticker=ticker,
        feature_date=feature_date,
    )

    reason_codes: list[str] = []
    if scheduled_count == 0 and manual_count > 0:
        reason_codes.append("manual_only")
    if selected_run_id is None:
        reason_codes.append("no_scheduled_official_candidate")
    else:
        reason_codes.append("first_complete_scheduled_run_selected")
    if partial_count:
        reason_codes.append("partial_scheduled_runs_present")
    if complete_count > 1:
        reason_codes.append("multiple_complete_scheduled_runs")
    if selected_run_id and latest_run and str(latest_run.get("run_id")) != selected_run_id:
        reason_codes.append("latest_run_differs_from_official_candidate")
    if selected_run_id and _has_earlier_partial_run(selected_run, feature_runs, assessments):
        reason_codes.append("retry_complete_run_selected")
    if any(item["duplicate_slots"] for item in assessments):
        reason_codes.append("duplicate_slots_detected")
    if any(item["ineligible_records"] for item in assessments):
        reason_codes.append("ineligible_records_present")

    missing_slots = (
        selection["selected_assessment"]["missing_slots"]
        if selection["selected_assessment"]
        else _latest_scheduled_missing_slots(feature_runs, assessments)
    )
    duplicate_slots = sorted(
        {slot for assessment in assessments for slot in assessment["duplicate_slots"]}
    )
    return {
        "feature_date": feature_date,
        "scheduled_run_count": scheduled_count,
        "manual_run_count": manual_count,
        "complete_scheduled_run_count": complete_count,
        "partial_scheduled_run_count": partial_count,
        "selected_official_run_id": selected_run_id,
        "latest_run_id": latest_run.get("run_id") if latest_run else None,
        "latest_run_type": latest_run.get("run_type") if latest_run else None,
        "selected_binary_record_id": selected_ids.get("binary"),
        "selected_multiclass_record_id": selected_ids.get("multiclass"),
        "missing_slots": missing_slots,
        "duplicate_slots": duplicate_slots,
        "reason_codes": _deduplicate(reason_codes),
        "scheduled_run_diagnostics": assessments,
        "current_db_official_record_count": int(len(current_official)),
        "current_db_latest_record_count": int(len(current_latest)),
        "current_db_official_record_ids": _record_ids(current_official),
        "current_db_latest_record_ids": _record_ids(current_latest),
        "selected_records_currently_official": selected_current_official,
        "selected_records_currently_latest": selected_current_latest,
        **snapshot_comparison,
    }


def _runs_for_feature_date(
    runs: pd.DataFrame,
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str,
) -> pd.DataFrame:
    ticker_runs = _filter_equal(runs, "ticker", ticker)
    ticker_records = _filter_equal(records, "ticker", ticker)
    date_records = _filter_date_equal(ticker_records, "feature_date", feature_date)
    record_run_ids = set(_column(date_records, "run_id").dropna().astype(str))
    latest_dates = _column(ticker_runs, "latest_feature_date").map(_optional_date_text)
    mask = _column(ticker_runs, "run_id").astype(str).isin(record_run_ids) | latest_dates.eq(
        feature_date
    )
    return ticker_runs[mask].copy().reset_index(drop=True)


def _feature_dates(runs: pd.DataFrame, records: pd.DataFrame) -> list[str]:
    dates = {
        value
        for value in _column(records, "feature_date").map(_optional_date_text)
        if value is not None
    }
    dates.update(
        value
        for value in _column(runs, "latest_feature_date").map(_optional_date_text)
        if value is not None
    )
    return sorted(dates, reverse=True)


def _matching_records(
    records: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str,
    run_id: str,
) -> pd.DataFrame:
    rows = _filter_equal(records, "ticker", ticker)
    rows = _filter_equal(rows, "run_id", run_id)
    if rows.empty:
        return rows
    date_values = _column(rows, "feature_date").map(_optional_date_text)
    return rows[date_values.eq(feature_date)].copy().reset_index(drop=True)


def _filter_equal(rows: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    if rows.empty or column not in rows:
        return rows.iloc[0:0].copy()
    return rows[rows[column].astype(str).eq(str(value))].copy().reset_index(drop=True)


def _filter_date_equal(rows: pd.DataFrame, column: str, value: object) -> pd.DataFrame:
    if rows.empty or column not in rows:
        return rows.iloc[0:0].copy()
    normalized = rows[column].map(_optional_date_text)
    return rows[normalized.eq(_optional_date_text(value))].copy().reset_index(drop=True)


def _column(rows: pd.DataFrame, column: str) -> pd.Series:
    if column in rows:
        return rows[column]
    return pd.Series(index=rows.index, dtype="object")


def _truthy_series(rows: pd.DataFrame, column: str) -> pd.Series:
    return _column(rows, column).map(_is_truthy).astype(bool)


def _is_truthy(value: object) -> bool:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _earliest_run_sort_key(run: dict[str, Any]) -> tuple[int, pd.Timestamp, str]:
    timestamp = pd.to_datetime(run.get("run_timestamp"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        return (1, pd.Timestamp.max.tz_localize("UTC"), str(run.get("run_id") or ""))
    return (0, timestamp, str(run.get("run_id") or ""))


def _latest_run(runs: pd.DataFrame) -> dict[str, Any] | None:
    if runs.empty:
        return None
    values = runs.to_dict(orient="records")
    valid = [row for row in values if _earliest_run_sort_key(row)[0] == 0]
    if not valid:
        return sorted(values, key=lambda row: str(row.get("run_id") or ""), reverse=True)[0]
    latest_timestamp = max(_earliest_run_sort_key(row)[1] for row in valid)
    tied = [row for row in valid if _earliest_run_sort_key(row)[1] == latest_timestamp]
    return sorted(tied, key=lambda row: str(row.get("run_id") or ""), reverse=True)[0]


def _latest_scheduled_missing_slots(
    feature_runs: pd.DataFrame,
    assessments: list[dict[str, Any]],
) -> list[str]:
    if not assessments:
        return list(PRODUCTION_SLOT_NAMES)
    by_run = {str(item["run_id"]): item for item in assessments}
    scheduled = feature_runs[
        _column(feature_runs, "run_type").astype(str).eq(SCHEDULED_LIVE_DUAL_WRITE)
    ]
    latest = _latest_run(scheduled)
    if latest is None:
        return list(PRODUCTION_SLOT_NAMES)
    return list(by_run.get(str(latest.get("run_id")), {}).get("missing_slots", []))


def _has_earlier_partial_run(
    selected_run: dict[str, Any],
    feature_runs: pd.DataFrame,
    assessments: list[dict[str, Any]],
) -> bool:
    selected_key = _earliest_run_sort_key(selected_run)
    by_run = {str(item["run_id"]): item for item in assessments}
    for run in feature_runs.to_dict(orient="records"):
        if str(run.get("run_type")) != SCHEDULED_LIVE_DUAL_WRITE:
            continue
        if _earliest_run_sort_key(run) >= selected_key:
            continue
        assessment = by_run.get(str(run.get("run_id")))
        if assessment is not None and not assessment["eligible_for_official"]:
            return True
    return False


def _compare_selected_records_to_snapshot(
    selected_records: pd.DataFrame,
    snapshot: pd.DataFrame,
    *,
    ticker: str,
    feature_date: str,
) -> dict[str, Any]:
    if selected_records.empty:
        return {
            "snapshot_matches": None,
            "snapshot_compare_status": "no_selected_candidate",
            "snapshot_mismatch_count": 0,
            "snapshot_compare_rows": [],
        }
    if snapshot.empty:
        return {
            "snapshot_matches": False,
            "snapshot_compare_status": "snapshot_not_available",
            "snapshot_mismatch_count": int(len(selected_records)),
            "snapshot_compare_rows": [],
        }

    snapshot_rows = _filter_equal(snapshot, "ticker", ticker)
    if "date" in snapshot_rows:
        snapshot_dates = snapshot_rows["date"].map(_optional_date_text)
        snapshot_rows = snapshot_rows[snapshot_dates.eq(feature_date)].copy()
    compare_rows: list[dict[str, Any]] = []
    for record in selected_records.to_dict(orient="records"):
        matches = snapshot_rows[
            _column(snapshot_rows, "target").astype(str).eq(str(record.get("target")))
            & _column(snapshot_rows, "model_name").astype(str).eq(str(record.get("model_name")))
        ]
        if matches.empty:
            compare_rows.append(
                {
                    "record_id": record.get("record_id"),
                    "slot": classify_prediction_model_slot(
                        record.get("target"), record.get("model_name")
                    ),
                    "match": False,
                    "reason": "missing_snapshot_record",
                }
            )
            continue
        snapshot_row = matches.iloc[0]
        label_match = str(record.get("predicted_label")) == str(snapshot_row.get("predicted_class"))
        confidence_match = _optional_float_matches(
            record.get("confidence"), snapshot_row.get("confidence")
        )
        compare_rows.append(
            {
                "record_id": record.get("record_id"),
                "slot": classify_prediction_model_slot(
                    record.get("target"), record.get("model_name")
                ),
                "match": bool(label_match and confidence_match),
                "reason": "match" if label_match and confidence_match else "value_mismatch",
            }
        )
    mismatch_count = sum(1 for row in compare_rows if not row["match"])
    return {
        "snapshot_matches": mismatch_count == 0 and len(compare_rows) == len(PRODUCTION_MODEL_SLOTS),
        "snapshot_compare_status": "match" if mismatch_count == 0 else "mismatch",
        "snapshot_mismatch_count": mismatch_count,
        "snapshot_compare_rows": compare_rows,
    }


def _optional_float_matches(left: object, right: object) -> bool:
    if left is None or right is None or pd.isna(left) or pd.isna(right):
        return left is None and right is None
    return isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _ineligible_record(row: dict[str, Any], reason_code: str) -> dict[str, Any]:
    return {
        "record_id": row.get("record_id"),
        "target": row.get("target"),
        "model_name": row.get("model_name"),
        "source": row.get("source"),
        "reason_code": reason_code,
    }


def _record_ids(rows: pd.DataFrame) -> list[str]:
    return [str(value) for value in _column(rows, "record_id").dropna().tolist()]


def _date_text(value: str | pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _optional_date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    try:
        return _date_text(value)
    except (TypeError, ValueError):
        return None


def _deduplicate(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))
