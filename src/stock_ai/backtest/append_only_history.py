from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.storage.sqlite_store import (
    compare_append_only_official_to_snapshot,
    filter_official_prediction_snapshot_records,
    list_prediction_records,
    list_prediction_runs,
    read_latest_predictions,
    summarize_append_only_records,
)


def summarize_append_only_history_for_ui(
    database_path: str | Path,
    *,
    ticker: str,
) -> dict[str, Any]:
    """Build read-only append-only prediction history summaries for UI display."""
    summary = summarize_append_only_records(database_path, ticker=ticker)
    runs = list_prediction_runs(database_path, ticker=ticker)
    records = list_prediction_records(database_path, ticker=ticker)
    comparison = compare_append_only_official_to_snapshot(database_path, ticker=ticker)

    return {
        "summary": summary,
        "runs": runs,
        "records": records,
        "run_summary": _run_summary(runs),
        "record_summary": _record_summary(records),
        "feature_date_summary": _feature_date_summary(records, runs),
        "evaluation_summary": _evaluation_summary(records),
        "comparison": comparison,
        "comparison_rows": pd.DataFrame(comparison.get("rows", [])),
    }


def build_append_only_health_visibility(
    database_path: str | Path,
    *,
    ticker: str,
) -> dict[str, Any]:
    """Build a read-only, non-blocking append-only health visibility summary."""
    summary = summarize_append_only_history_for_ui(database_path, ticker=ticker)
    raw_summary = summary["summary"]
    run_summary = summary["run_summary"]
    evaluation_summary = summary["evaluation_summary"]
    comparison = summary["comparison"]
    comparison_rows = summary["comparison_rows"]
    classifications = _append_only_visibility_classifications(
        raw_summary,
        run_summary,
        comparison,
    )
    mismatch_count = int(comparison.get("snapshot_official_count", 0)) - int(
        comparison.get("append_official_count", 0)
    )
    if not comparison_rows.empty and "match" in comparison_rows.columns:
        mismatch_count = int((~comparison_rows["match"].fillna(False).astype(bool)).sum())

    return {
        "visibility_status": classifications[0],
        "visibility_classifications": classifications,
        "append_only_tables_exist": bool(raw_summary.get("append_only_tables_exist")),
        "prediction_runs_count": int(raw_summary.get("prediction_runs_count", 0)),
        "prediction_records_count": int(raw_summary.get("prediction_records_count", 0)),
        "existing_predictions_count": int(raw_summary.get("existing_predictions_count", 0)),
        "latest_run_id": run_summary.get("latest_run_id"),
        "latest_run_timestamp": run_summary.get("latest_run_timestamp"),
        "latest_run_type": run_summary.get("latest_run_type"),
        "latest_feature_date": run_summary.get("latest_feature_date"),
        "latest_official_records_count": int(raw_summary.get("official_records_count", 0)),
        "append_only_vs_snapshot_matches": bool(comparison.get("matches")),
        "mismatch_count": int(max(mismatch_count, 0)),
        "pending_count": int(evaluation_summary.get("pending_count", 0)),
        "hit_count": int(evaluation_summary.get("hit_count", 0)),
        "miss_count": int(evaluation_summary.get("miss_count", 0)),
        "unevaluated_count": int(evaluation_summary.get("unevaluated_count", 0)),
        "latest_compare_status": (
            "matches" if comparison.get("matches") else "mismatch_or_not_available"
        ),
        "not_health_source": True,
        "manual_only": True,
    }


def build_append_only_run_detail(
    database_path: str | Path,
    *,
    ticker: str,
    run_id: str | None,
) -> dict[str, Any]:
    """Return one append-only run with its records and audit-only comparison."""
    runs = list_prediction_runs(database_path, ticker=ticker)
    records = list_prediction_records(database_path, ticker=ticker)
    if runs.empty or not run_id:
        return _empty_run_detail(runs, records)

    selected_runs = runs[runs["run_id"].astype(str).eq(str(run_id))].copy()
    selected_records = records[records["run_id"].astype(str).eq(str(run_id))].copy()
    if selected_runs.empty:
        return _empty_run_detail(runs, records)

    return {
        "run": selected_runs.iloc[0].to_dict(),
        "records": selected_records.reset_index(drop=True),
        "run_summary": build_append_only_run_summary(selected_records),
        "record_table": build_append_only_run_record_table(selected_records),
        "snapshot_compare": build_append_only_run_snapshot_compare(
            database_path,
            ticker=ticker,
            run_id=run_id,
        ),
        "feature_date_run_summary": build_append_only_feature_date_run_summary(
            records,
            runs,
        ),
        "available_runs": runs,
    }


def build_append_only_run_summary(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "record_count": 0,
            "official_record_count": 0,
            "latest_record_count": 0,
            "binary_record_count": 0,
            "multiclass_record_count": 0,
            "pending_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "unevaluated_count": 0,
            "targets_included": "",
            "model_names_included": "",
        }
    evaluation = _evaluation_summary(records)
    return {
        "record_count": int(len(records)),
        "official_record_count": _truthy_count(records, "is_official"),
        "latest_record_count": _truthy_count(records, "is_latest"),
        "binary_record_count": int(records["model_type"].eq("binary").sum()),
        "multiclass_record_count": int(records["model_type"].eq("multiclass").sum()),
        "pending_count": evaluation["pending_count"],
        "hit_count": evaluation["hit_count"],
        "miss_count": evaluation["miss_count"],
        "unevaluated_count": evaluation["unevaluated_count"],
        "targets_included": _joined_unique(records["target"]),
        "model_names_included": _joined_unique(records["model_name"]),
    }


def build_append_only_run_record_table(records: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "record_id",
        "feature_date",
        "target_close_date",
        "target",
        "model_name",
        "model_type",
        "predicted_label",
        "predicted_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "source",
        "is_official",
        "is_latest",
        "actual_return",
        "actual_label",
        "actual_direction",
        "result",
        "evaluated_at",
        "notes",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)
    available = [column for column in columns if column in records.columns]
    return records[available].copy().reset_index(drop=True)


def build_append_only_run_snapshot_compare(
    database_path: str | Path,
    *,
    ticker: str,
    run_id: str,
) -> dict[str, Any]:
    """Compare selected run records with current official prediction snapshot."""
    run_records = list_prediction_records(database_path, ticker=ticker, run_id=run_id)
    snapshot = read_latest_predictions(database_path)
    if not snapshot.empty:
        snapshot = snapshot[snapshot["ticker"].astype(str).eq(str(ticker))].reset_index(drop=True)
    official_snapshot, _excluded = filter_official_prediction_snapshot_records(
        snapshot,
        ticker=ticker,
    )
    rows = _compare_run_records_to_snapshot(run_records, official_snapshot)
    matched = sum(1 for row in rows if row["match"])
    mismatches = [row for row in rows if not row["match"]]
    return {
        "run_id": run_id,
        "snapshot_official_count": int(len(official_snapshot)),
        "run_record_count": int(len(run_records)),
        "matched_count": int(matched),
        "mismatch_count": int(len(mismatches)),
        "missing_in_append_only_count": int(
            sum(1 for row in mismatches if row["reason"] == "missing_in_append_only")
        ),
        "missing_in_snapshot_count": int(
            sum(1 for row in mismatches if row["reason"] == "missing_in_snapshot")
        ),
        "value_mismatch_count": int(
            sum(1 for row in mismatches if row["reason"] == "value_mismatch")
        ),
        "matches": bool(rows and not mismatches),
        "rows": rows,
    }


def build_append_only_feature_date_run_summary(
    records: pd.DataFrame,
    runs: pd.DataFrame,
) -> pd.DataFrame:
    return _feature_date_summary(records, runs)


def _append_only_visibility_classifications(
    raw_summary: dict[str, Any],
    run_summary: dict[str, Any],
    comparison: dict[str, Any],
) -> list[str]:
    classifications: list[str] = []
    if not raw_summary.get("append_only_tables_exist"):
        classifications.append("append_only_not_initialized")
    elif int(raw_summary.get("prediction_records_count", 0)) <= 0:
        classifications.append("append_only_empty")
    else:
        classifications.append("append_only_available")
        if comparison.get("matches"):
            classifications.append("append_only_latest_matches_snapshot")
        else:
            classifications.append("append_only_mismatch_detected_info")
    if run_summary.get("latest_run_type") == "manual_live_dual_write":
        classifications.append("append_only_manual_only")
    classifications.append("append_only_not_health_source")
    return classifications


def _run_summary(runs: pd.DataFrame) -> dict[str, Any]:
    if runs.empty:
        return {
            "total_runs": 0,
            "latest_run_id": None,
            "latest_run_timestamp": None,
            "latest_run_type": None,
            "latest_source": None,
            "latest_ticker": None,
            "latest_feature_date": None,
            "latest_data_freshness_status": None,
            "latest_stale_at_run_time": None,
            "latest_notes": None,
        }
    ordered = runs.copy()
    ordered["_run_timestamp"] = pd.to_datetime(
        ordered.get("run_timestamp"),
        errors="coerce",
    )
    ordered = ordered.sort_values(
        ["_run_timestamp", "created_at", "run_id"],
        ascending=[False, False, False],
    )
    latest = ordered.iloc[0]
    return {
        "total_runs": int(len(ordered)),
        "latest_run_id": latest.get("run_id"),
        "latest_run_timestamp": latest.get("run_timestamp"),
        "latest_run_type": latest.get("run_type"),
        "latest_source": latest.get("source"),
        "latest_ticker": latest.get("ticker"),
        "latest_feature_date": latest.get("latest_feature_date"),
        "latest_data_freshness_status": latest.get("data_freshness_status"),
        "latest_stale_at_run_time": bool(latest.get("stale_at_run_time")),
        "latest_notes": latest.get("notes"),
    }


def _record_summary(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "total_records": 0,
            "official_records": 0,
            "latest_records": 0,
            "binary_records": 0,
            "multiclass_records": 0,
            "source_counts": {},
            "model_type_counts": {},
        }
    frame = records.copy()
    return {
        "total_records": int(len(frame)),
        "official_records": _truthy_count(frame, "is_official"),
        "latest_records": _truthy_count(frame, "is_latest"),
        "binary_records": int(frame.get("model_type", pd.Series(dtype=str)).eq("binary").sum()),
        "multiclass_records": int(
            frame.get("model_type", pd.Series(dtype=str)).eq("multiclass").sum()
        ),
        "source_counts": _value_counts(frame, "source"),
        "model_type_counts": _value_counts(frame, "model_type"),
    }


def _feature_date_summary(records: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "feature_date",
        "run_count",
        "run_ids",
        "record_count",
        "official_records",
        "latest_records",
        "binary_records",
        "multiclass_records",
        "run_types",
        "manual_run_ids",
        "scheduled_run_ids",
        "official_run_ids",
        "sources",
        "latest_run_timestamp",
        "latest_run_id",
    ]
    if records.empty:
        return pd.DataFrame(columns=columns)

    frame = records.copy()
    run_lookup = _run_lookup(runs)
    frame["run_type"] = frame["run_id"].map(lambda value: run_lookup.get(value, {}).get("run_type"))
    frame["run_timestamp"] = frame["run_id"].map(
        lambda value: run_lookup.get(value, {}).get("run_timestamp")
    )
    grouped_rows = []
    for feature_date, group in frame.groupby("feature_date", dropna=False):
        ordered = group.copy()
        ordered["_run_timestamp"] = pd.to_datetime(ordered["run_timestamp"], errors="coerce")
        ordered = ordered.sort_values(
            ["_run_timestamp", "created_at", "record_id"],
            ascending=[False, False, False],
        )
        grouped_rows.append(
            {
                "feature_date": feature_date,
                "run_count": int(group["run_id"].nunique()),
                "run_ids": _joined_unique(group["run_id"]),
                "record_count": int(len(group)),
                "official_records": _truthy_count(group, "is_official"),
                "latest_records": _truthy_count(group, "is_latest"),
                "binary_records": int(group["model_type"].eq("binary").sum()),
                "multiclass_records": int(group["model_type"].eq("multiclass").sum()),
                "run_types": _joined_unique(group["run_type"]),
                "manual_run_ids": _joined_unique(
                    group.loc[group["run_type"].astype(str).str.contains("manual"), "run_id"]
                ),
                "scheduled_run_ids": _joined_unique(
                    group.loc[group["run_type"].astype(str).str.contains("scheduled"), "run_id"]
                ),
                "official_run_ids": _joined_unique(
                    group.loc[_truthy_mask(group, "is_official"), "run_id"]
                ),
                "sources": _joined_unique(group["source"]),
                "latest_run_timestamp": ordered.iloc[0].get("run_timestamp"),
                "latest_run_id": ordered.iloc[0].get("run_id"),
            }
        )
    result = pd.DataFrame(grouped_rows, columns=columns)
    result["_sort_date"] = pd.to_datetime(result["feature_date"], errors="coerce")
    result = result.sort_values("_sort_date", ascending=False).drop(columns="_sort_date")
    return result.reset_index(drop=True)


def _evaluation_summary(records: pd.DataFrame) -> dict[str, Any]:
    if records.empty:
        return {
            "pending_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "unevaluated_count": 0,
            "latest_pending_target_close_date": None,
            "latest_evaluated_at": None,
        }
    frame = records.copy()
    result = frame.get("result", pd.Series(index=frame.index, dtype=str)).fillna("pending").astype(str)
    evaluated_at = pd.to_datetime(frame.get("evaluated_at"), errors="coerce")
    pending_mask = result.eq("pending")
    hit_mask = result.eq("hit")
    miss_mask = result.eq("miss")
    evaluated_mask = hit_mask | miss_mask
    target_dates = pd.to_datetime(frame.get("target_close_date"), errors="coerce")
    pending_target_dates = target_dates[pending_mask].dropna()
    latest_evaluated_at = evaluated_at[evaluated_mask].dropna()
    return {
        "pending_count": int(pending_mask.sum()),
        "hit_count": int(hit_mask.sum()),
        "miss_count": int(miss_mask.sum()),
        "unevaluated_count": int((~evaluated_mask).sum()),
        "latest_pending_target_close_date": (
            pending_target_dates.max().strftime("%Y-%m-%d")
            if not pending_target_dates.empty
            else None
        ),
        "latest_evaluated_at": (
            latest_evaluated_at.max().isoformat()
            if not latest_evaluated_at.empty
            else None
        ),
    }


def _run_lookup(runs: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if runs.empty:
        return {}
    return {
        str(row.get("run_id")): row.to_dict()
        for _, row in runs.iterrows()
    }


def _empty_run_detail(runs: pd.DataFrame, records: pd.DataFrame) -> dict[str, Any]:
    return {
        "run": {},
        "records": pd.DataFrame(),
        "run_summary": build_append_only_run_summary(pd.DataFrame()),
        "record_table": build_append_only_run_record_table(pd.DataFrame()),
        "snapshot_compare": {
            "run_id": None,
            "snapshot_official_count": 0,
            "run_record_count": 0,
            "matched_count": 0,
            "mismatch_count": 0,
            "missing_in_append_only_count": 0,
            "missing_in_snapshot_count": 0,
            "value_mismatch_count": 0,
            "matches": False,
            "rows": [],
        },
        "feature_date_run_summary": build_append_only_feature_date_run_summary(
            records,
            runs,
        ),
        "available_runs": runs,
    }


def _compare_run_records_to_snapshot(
    run_records: pd.DataFrame,
    official_snapshot: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    snapshot_lookup = {
        _compare_key(row): row
        for _, row in official_snapshot.iterrows()
    }
    record_lookup = {
        _compare_key(row, feature_date_column="feature_date"): row
        for _, row in run_records.iterrows()
    }

    for key, snapshot_row in snapshot_lookup.items():
        record_row = record_lookup.get(key)
        if record_row is None:
            rows.append(_missing_append_only_compare_row(snapshot_row))
            continue
        rows.append(_value_compare_row(snapshot_row, record_row))

    for key, record_row in record_lookup.items():
        if key in snapshot_lookup:
            continue
        rows.append(_missing_snapshot_compare_row(record_row))
    return rows


def _compare_key(row: pd.Series, *, feature_date_column: str = "date") -> tuple[str, str, str]:
    return (
        _date_text(row.get(feature_date_column)),
        str(row.get("target") or ""),
        str(row.get("model_name") or ""),
    )


def _missing_append_only_compare_row(snapshot_row: pd.Series) -> dict[str, Any]:
    return {
        "feature_date": _date_text(snapshot_row.get("date")),
        "target": snapshot_row.get("target"),
        "model_name": snapshot_row.get("model_name"),
        "snapshot_predicted_label": str(snapshot_row.get("predicted_class")),
        "append_predicted_label": None,
        "snapshot_confidence": snapshot_row.get("confidence"),
        "append_confidence": None,
        "snapshot_direction": _direction_from_row(snapshot_row, "predicted_class"),
        "append_direction": None,
        "confidence_delta": None,
        "label_match": False,
        "direction_match": False,
        "confidence_match": False,
        "match": False,
        "reason": "missing_in_append_only",
    }


def _missing_snapshot_compare_row(record_row: pd.Series) -> dict[str, Any]:
    return {
        "feature_date": _date_text(record_row.get("feature_date")),
        "target": record_row.get("target"),
        "model_name": record_row.get("model_name"),
        "snapshot_predicted_label": None,
        "append_predicted_label": record_row.get("predicted_label"),
        "snapshot_confidence": None,
        "append_confidence": record_row.get("confidence"),
        "snapshot_direction": None,
        "append_direction": record_row.get("predicted_direction"),
        "confidence_delta": None,
        "label_match": False,
        "direction_match": False,
        "confidence_match": False,
        "match": False,
        "reason": "missing_in_snapshot",
    }


def _value_compare_row(snapshot_row: pd.Series, record_row: pd.Series) -> dict[str, Any]:
    snapshot_label = str(snapshot_row.get("predicted_class"))
    record_label = str(record_row.get("predicted_label"))
    snapshot_confidence = _float_or_none(snapshot_row.get("confidence"))
    record_confidence = _float_or_none(record_row.get("confidence"))
    confidence_delta = (
        None
        if snapshot_confidence is None or record_confidence is None
        else record_confidence - snapshot_confidence
    )
    snapshot_direction = _direction_from_row(snapshot_row, "predicted_class")
    record_direction = record_row.get("predicted_direction")
    label_match = snapshot_label == record_label
    direction_match = str(snapshot_direction) == str(record_direction)
    confidence_match = confidence_delta is not None and abs(confidence_delta) < 1e-12
    match = bool(label_match and direction_match and confidence_match)
    return {
        "feature_date": _date_text(snapshot_row.get("date")),
        "target": snapshot_row.get("target"),
        "model_name": snapshot_row.get("model_name"),
        "snapshot_predicted_label": snapshot_label,
        "append_predicted_label": record_label,
        "snapshot_confidence": snapshot_confidence,
        "append_confidence": record_confidence,
        "snapshot_direction": snapshot_direction,
        "append_direction": record_direction,
        "confidence_delta": confidence_delta,
        "label_match": label_match,
        "direction_match": direction_match,
        "confidence_match": confidence_match,
        "match": match,
        "reason": "match" if match else "value_mismatch",
    }


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%Y-%m-%d")


def _direction_from_row(row: pd.Series, label_column: str) -> str | None:
    label = row.get(label_column)
    target = str(row.get("target") or "")
    if label is None or pd.isna(label):
        return None
    try:
        label_int = int(label)
    except (TypeError, ValueError):
        return None
    if "binary" in target:
        return "up" if label_int == 1 else "down"
    if "multiclass" in target:
        return {
            0: "crash",
            1: "down",
            2: "flat",
            3: "up",
            4: "surge",
        }.get(label_int)
    return str(label)


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _truthy_count(frame: pd.DataFrame, column: str) -> int:
    if column not in frame.columns:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(values.astype(int).eq(1).sum())


def _truthy_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return values.astype(int).eq(1)


def _value_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame.columns:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("unknown").astype(str).value_counts().items()
    }


def _joined_unique(values: pd.Series) -> str:
    unique = sorted({str(value) for value in values.dropna().tolist() if str(value)})
    return ", ".join(unique)
