from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any


DEFAULT_BINARY_SUMMARY = {
    "role": "primary direction reference",
    "historical_accuracy": "about 50.5%",
    "historical_balanced_accuracy": "about 50.5%",
    "historical_up_recall": "about 37%",
    "decision": "production change not ready",
}

DEFAULT_MULTICLASS_SUMMARY = {
    "role": "experimental/reference only",
    "current_issue": "structurally flat-biased",
    "predicted_flat_ratio": "about 95.5%",
    "decision": "do not use as primary direction judgement",
}


def build_model_decision_notes_summary(
    *,
    health_report: dict[str, Any] | None = None,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Build a compact, human-readable summary of the current model decision state."""
    warning_summary = {}
    known_missed_dates: list[dict[str, Any]] = []
    if isinstance(health_report, dict):
        warning_summary = health_report.get("warning_summary") or {}
        known_missed_dates = known_missed_live_date_details(warning_summary)

    return {
        "report_date": report_date or date.today().isoformat(),
        "operational_health": {
            "overall_status": (health_report or {}).get("overall_status", "unknown"),
            "action_required_count": warning_summary.get("action_required_count", 0),
            "known_missed_date_count": warning_summary.get(
                "known_missed_date_count",
                warning_summary.get("known_missed_count", len(known_missed_dates)),
            ),
            "known_ok_count": warning_summary.get("known_ok_count", 0),
        },
        "known_missed_live_dates": known_missed_dates,
        "scheduler_observation": {
            "current_daily_run_time": "10:30",
            "status": "observing",
            "note": "10:30 is being monitored before deciding whether 18:00 is necessary.",
        },
        "binary_evaluation": DEFAULT_BINARY_SUMMARY.copy(),
        "binary_offline_experiments": {
            "summary": (
                "Class weight and threshold variants showed small offline improvements, "
                "but the edge versus simple baselines remains weak."
            ),
            "decision": "proposal only; not applied to production",
        },
        "multiclass_evaluation": DEFAULT_MULTICLASS_SUMMARY.copy(),
        "multiclass_offline_boundary_experiments": {
            "summary": (
                "3-class and boundary variants remain future research candidates. "
                "No production multiclass change has been applied."
            ),
            "decision": "future experiment candidate",
        },
        "production_decision": {
            "binary_model_changed": False,
            "multiclass_model_changed": False,
            "daily_predict_changed": False,
            "sqlite_history_changed": False,
            "decision": "no production model change has been applied",
        },
        "next_monitoring_items": [
            "Confirm 10:30 daily runs continue with data_freshness_status OK.",
            "Keep Action required at 0 and unknown_source_count at 0.",
            "Monitor additional live binary outcomes before production threshold changes.",
            "Review multiclass as reference only until flat bias is reduced offline.",
        ],
    }


def known_missed_live_date_details(warning_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in warning_summary.get("known_missed", []) or []:
        if not isinstance(item, dict):
            continue
        details = item.get("details") or {}
        if not isinstance(details, dict):
            continue
        for missed in details.get("known_missed_live_dates", []) or []:
            if not isinstance(missed, dict):
                continue
            rows.append(
                {
                    "feature_date": missed.get("missing_date") or missed.get("date"),
                    "reason_code": missed.get("known_reason_code") or missed.get("reason_code"),
                    "reason": missed.get("known_reason") or missed.get("reason"),
                    "action": missed.get("known_action") or missed.get("action"),
                    "allow_backfill": missed.get("known_allow_backfill")
                    if "known_allow_backfill" in missed
                    else missed.get("allow_backfill"),
                    "note": missed.get("note") or missed.get("recommended_action"),
                }
            )
    return rows


def model_decision_notes_markdown(summary: dict[str, Any]) -> str:
    health = summary.get("operational_health") or {}
    known_missed = summary.get("known_missed_live_dates") or []
    scheduler = summary.get("scheduler_observation") or {}
    binary = summary.get("binary_evaluation") or {}
    binary_offline = summary.get("binary_offline_experiments") or {}
    multiclass = summary.get("multiclass_evaluation") or {}
    boundary = summary.get("multiclass_offline_boundary_experiments") or {}
    production = summary.get("production_decision") or {}
    next_items = summary.get("next_monitoring_items") or []

    lines = [
        f"# Model Decision Notes ({summary.get('report_date')})",
        "",
        "## Current operational health",
        f"- Overall status: {health.get('overall_status')}",
        f"- Action required: {health.get('action_required_count')}",
        f"- Known missed dates: {health.get('known_missed_date_count')}",
        f"- Known OK: {health.get('known_ok_count')}",
        "",
        "## Known missed live dates",
    ]
    if known_missed:
        for row in known_missed:
            lines.append(
                "- "
                f"{row.get('feature_date')}: {row.get('reason_code')} / "
                f"action={row.get('action')} / allow_backfill={row.get('allow_backfill')}"
            )
            if row.get("reason"):
                lines.append(f"  - Reason: {row.get('reason')}")
    else:
        lines.append("- None")

    lines.extend(
        [
            "",
            "## 10:30 scheduler observation",
            f"- Current daily run time: {scheduler.get('current_daily_run_time')}",
            f"- Status: {scheduler.get('status')}",
            f"- Note: {scheduler.get('note')}",
            "",
            "## Binary evaluation summary",
            f"- Role: {binary.get('role')}",
            f"- Historical accuracy: {binary.get('historical_accuracy')}",
            f"- Historical balanced accuracy: {binary.get('historical_balanced_accuracy')}",
            f"- Historical up recall: {binary.get('historical_up_recall')}",
            f"- Decision: {binary.get('decision')}",
            "",
            "## Binary offline experiment summary",
            f"- Summary: {binary_offline.get('summary')}",
            f"- Decision: {binary_offline.get('decision')}",
            "",
            "## Multiclass flat bias summary",
            f"- Role: {multiclass.get('role')}",
            f"- Current issue: {multiclass.get('current_issue')}",
            f"- Predicted flat ratio: {multiclass.get('predicted_flat_ratio')}",
            f"- Decision: {multiclass.get('decision')}",
            "",
            "## Multiclass offline boundary experiment summary",
            f"- Summary: {boundary.get('summary')}",
            f"- Decision: {boundary.get('decision')}",
            "",
            "## Current production decision",
            f"- Binary model changed: {production.get('binary_model_changed')}",
            f"- Multiclass model changed: {production.get('multiclass_model_changed')}",
            f"- daily_predict changed: {production.get('daily_predict_changed')}",
            f"- SQLite history changed: {production.get('sqlite_history_changed')}",
            f"- Decision: {production.get('decision')}",
            "",
            "## Next monitoring items",
        ]
    )
    lines.extend(f"- {item}" for item in next_items)
    lines.append("")
    return "\n".join(lines)


def write_model_decision_notes_report(
    summary: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_date = str(summary.get("report_date") or date.today().isoformat())
    path = output / f"model_decision_notes_{report_date}.md"
    path.write_text(model_decision_notes_markdown(summary), encoding="utf-8")
    return path
