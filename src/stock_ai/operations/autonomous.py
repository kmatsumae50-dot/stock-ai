from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd

from stock_ai.backtest.append_only_history import build_append_only_health_visibility
from stock_ai.backtest.append_only_policy import (
    build_official_selection_diagnostics_from_db,
)
from stock_ai.config import PROJECT_ROOT, Settings, load_settings
from stock_ai.evaluation.accuracy_diagnostics import (
    baseline_accuracy_frame,
    build_binary_diagnostic_frame,
    build_multiclass_diagnostic_frame,
    source_comparison_frame,
)
from stock_ai.evaluation.binary_shadow import (
    classify_shadow_conclusions,
    compare_shadow_with_live,
    summarize_shadow_predictions,
)
from stock_ai.health.checks import run_health_check
from stock_ai.news.quality import check_news_for_ticker
from stock_ai.news.sources import load_news_source_profiles
from stock_ai.operations.freshness import summarize_daily_data_freshness
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.storage.sqlite_store import (
    compare_append_only_official_to_snapshot,
    read_prediction_history,
)
from stock_ai.utils.safe_cast import safe_float, safe_int


SCHEMA_VERSION = "autonomous_operations/v1"
DEFAULT_STATE_PATH = PROJECT_ROOT / "reports" / "autonomous" / "operations_state.json"
DEFAULT_TASK_QUEUE_PATH = (
    PROJECT_ROOT / "reports" / "autonomous" / "operations_task_queue.json"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "autonomous"
SHADOW_PATH = PROJECT_ROOT / "reports" / "evaluation" / "binary_shadow_predictions.csv"
DAILY_LOG_PATH = PROJECT_ROOT / "logs" / "daily_predict.log"
CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.65)


class OperationsStateError(RuntimeError):
    pass


class OperationsLockError(RuntimeError):
    pass


class OperationsTimeoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class RuntimeBudget:
    started_monotonic: float
    max_runtime_minutes: float

    @classmethod
    def start(cls, max_runtime_minutes: float) -> RuntimeBudget:
        if max_runtime_minutes <= 0:
            raise ValueError("max_runtime_minutes must be positive")
        return cls(time.monotonic(), float(max_runtime_minutes))

    def check(self, stage: str) -> None:
        elapsed = time.monotonic() - self.started_monotonic
        if elapsed > self.max_runtime_minutes * 60:
            raise OperationsTimeoutError(
                f"Autonomous Operations exceeded its runtime budget at {stage}."
            )


class OperationsFileLock:
    def __init__(self, path: str | Path, *, force: bool = False) -> None:
        self.path = Path(path)
        self.force = force
        self._acquired = False

    def __enter__(self) -> OperationsFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.path.exists():
            if _lock_owner_is_active(self.path):
                raise OperationsLockError(
                    f"Operations lock belongs to an active process: {self.path}"
                )
            self.path.unlink()
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise OperationsLockError(f"Operations lock already exists: {self.path}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {"pid": os.getpid(), "created_at": _now_iso()},
                    ensure_ascii=False,
                )
            )
        self._acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._acquired:
            self.path.unlink(missing_ok=True)
            self._acquired = False


def default_operations_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "last_cycle_at": None,
        "last_successful_cycle_at": None,
        "last_health_status": None,
        "last_action_required_count": None,
        "last_unknown_source_count": None,
        "last_known_missed_dates": [],
        "last_post_start_missing_dates": [],
        "last_expected_trading_date": None,
        "last_raw_date": None,
        "last_feature_date": None,
        "last_live_binary_confirmed_count": None,
        "last_live_binary_metrics": {},
        "last_confidence_metrics": {},
        "last_shadow_confirmed_count": None,
        "last_append_only_run_count": None,
        "last_append_only_record_count": None,
        "last_snapshot_compare_status": None,
        "open_findings": [],
        "consecutive_failures": 0,
        "last_report_path": None,
    }


def load_operations_state(path: str | Path) -> tuple[dict[str, Any], bool]:
    state_path = Path(path)
    if not state_path.exists():
        return default_operations_state(), True
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationsStateError(f"Operations state is unreadable: {state_path}") from error
    if not isinstance(loaded, dict):
        raise OperationsStateError(f"Operations state must be a JSON object: {state_path}")
    state = default_operations_state()
    state.update(loaded)
    return state, False


def save_operations_state(path: str | Path, state: dict[str, Any]) -> None:
    _write_json(Path(path), state)


def load_task_queue(path: str | Path) -> dict[str, Any]:
    queue_path = Path(path)
    if not queue_path.exists():
        return {"schema_version": SCHEMA_VERSION, "updated_at": None, "tasks": []}
    try:
        loaded = json.loads(queue_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OperationsStateError(f"Operations task queue is unreadable: {queue_path}") from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("tasks", []), list):
        raise OperationsStateError(f"Operations task queue has an invalid schema: {queue_path}")
    return {
        "schema_version": str(loaded.get("schema_version") or SCHEMA_VERSION),
        "updated_at": loaded.get("updated_at"),
        "tasks": [task for task in loaded.get("tasks", []) if isinstance(task, dict)],
    }


def reconcile_task_queue(
    queue: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    cycle_at: str,
    source_report: str,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    existing = {
        str(task.get("fingerprint")): dict(task)
        for task in queue.get("tasks", [])
        if task.get("fingerprint")
    }
    active_fingerprints: set[str] = set()
    changes = {"created": [], "updated": [], "resolved": []}

    for finding in findings:
        fingerprint = str(finding["fingerprint"])
        active_fingerprints.add(fingerprint)
        task = existing.get(fingerprint)
        status = _finding_task_status(finding)
        if task is None:
            task = {
                "task_id": _task_id(fingerprint),
                "fingerprint": fingerprint,
                "created_at": cycle_at,
            }
            changes["created"].append(fingerprint)
        else:
            changes["updated"].append(fingerprint)
        task.update(
            {
                "last_seen_at": cycle_at,
                "category": finding["category"],
                "risk_level": finding["risk_level"],
                "priority": finding["priority"],
                "title": finding["title"],
                "description": finding["description"],
                "evidence": _json_ready(finding.get("evidence", {})),
                "proposed_action": finding["proposed_action"],
                "status": status,
                "requires_human_approval": bool(finding.get("requires_human_approval")),
                "blocked_reason": finding.get("blocked_reason"),
                "source_report": source_report,
            }
        )
        existing[fingerprint] = task

    for fingerprint, task in existing.items():
        if fingerprint in active_fingerprints:
            continue
        if task.get("status") in {"open", "monitoring", "approval_required"}:
            task["status"] = "resolved"
            task["last_seen_at"] = cycle_at
            task["source_report"] = source_report
            changes["resolved"].append(fingerprint)

    tasks = sorted(
        existing.values(),
        key=lambda task: (
            _priority_rank(str(task.get("priority"))),
            str(task.get("created_at") or ""),
            str(task.get("fingerprint") or ""),
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": cycle_at,
        "tasks": tasks,
    }, changes


def run_autonomous_operations(
    *,
    ticker: str = "7203.T",
    state_path: str | Path = DEFAULT_STATE_PATH,
    task_queue_path: str | Path = DEFAULT_TASK_QUEUE_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    max_runtime_minutes: float = 10,
    force: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    state_file = Path(state_path)
    queue_file = Path(task_queue_path)
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    budget = RuntimeBudget.start(max_runtime_minutes)
    cycle_started_at = _now_iso()
    report_paths = _report_paths(output_dir, cycle_started_at, force=force)
    lock_path = output_dir / ".operations.cli.lock"

    with OperationsFileLock(lock_path, force=force):
        try:
            previous_state, first_cycle = load_operations_state(state_file)
            queue = load_task_queue(queue_file)
        except OperationsStateError as error:
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(error),
                stop_reason="state_or_task_queue_corrupt",
                write_queue=False,
            )

        production_before = _production_guard(settings)
        scheduler_before = _read_scheduler_status()
        try:
            budget.check("health-check")
        except OperationsTimeoutError as error:
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(error),
                stop_reason="runtime_budget_exceeded",
                previous_state=previous_state,
                state_path=state_file,
            )

        try:
            health = run_health_check(ticker=ticker)
        except Exception as error:  # health-check failure is a required stop condition.
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(error),
                stop_reason="health_check_unavailable",
                previous_state=previous_state,
                state_path=state_file,
            )

        try:
            budget.check("production database validation")
        except OperationsTimeoutError as error:
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(error),
                stop_reason="runtime_budget_exceeded",
                previous_state=previous_state,
                state_path=state_file,
            )
        database_validation = _validate_database(settings.paths.database_path)
        if not database_validation["readable"]:
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(database_validation.get("error") or "predictions table unavailable"),
                stop_reason="production_database_unreadable",
                previous_state=previous_state,
                state_path=state_file,
            )

        predictions = read_prediction_history(settings.paths.database_path, ticker=ticker)
        feature_data = _safe_read_parquet(settings.paths.feature_data_dir / f"{ticker}.parquet")
        raw_data = _safe_read_parquet(settings.paths.raw_data_dir / f"{ticker}.parquet")
        freshness = summarize_daily_data_freshness(
            ticker=ticker,
            raw_data_dir=settings.paths.raw_data_dir,
            feature_data_dir=settings.paths.feature_data_dir,
        )
        health_summary = _summarize_health(health)
        scheduler = _merge_scheduler_status(scheduler_before, health, DAILY_LOG_PATH)
        daily_prediction = _summarize_daily_prediction(health, scheduler)
        live_coverage = _summarize_live_coverage(predictions, feature_data)
        binary_metrics = _summarize_live_binary_metrics(
            predictions,
            feature_data,
            previous_state,
        )
        confidence = _summarize_confidence(binary_metrics.pop("_frame"), previous_state)
        shadow = _summarize_shadow(predictions)
        append_only = _summarize_append_only(settings.paths.database_path, ticker)
        unknown_sources = _diagnose_unknown_sources(settings, ticker)
        missing_live = _diagnose_missing_live(
            health_summary,
            raw_data=raw_data,
            feature_data=feature_data,
        )
        try:
            budget.check("finding classification")
        except OperationsTimeoutError as error:
            return _write_stopped_cycle(
                ticker=ticker,
                cycle_started_at=cycle_started_at,
                report_paths=report_paths,
                queue_path=queue_file,
                error=str(error),
                stop_reason="runtime_budget_exceeded",
                previous_state=previous_state,
                state_path=state_file,
            )

        findings = _build_findings(
            ticker=ticker,
            health_summary=health_summary,
            freshness=freshness,
            scheduler=scheduler,
            unknown_sources=unknown_sources,
            missing_live=missing_live,
            binary_metrics=binary_metrics,
            confidence=confidence,
            shadow=shadow,
            append_only=append_only,
        )
        classifications = classify_operations_cycle(
            health_summary=health_summary,
            freshness=freshness,
            scheduler=scheduler,
            unknown_sources=unknown_sources,
            missing_live=missing_live,
            binary_metrics=binary_metrics,
            shadow=shadow,
            append_only=append_only,
        )
        stop_reason = _stop_reason(
            health_summary=health_summary,
            database_validation=database_validation,
            previous_state=previous_state,
        )

        queue, task_changes = reconcile_task_queue(
            queue,
            findings,
            cycle_at=cycle_started_at,
            source_report=str(report_paths["markdown"]),
        )
        production_after = _production_guard(settings)
        scheduler_after = _read_scheduler_status()
        production_changes = _production_change_summary(
            production_before,
            production_after,
            scheduler_before,
            scheduler_after,
        )
        if any(production_changes.values()):
            stop_reason = stop_reason or "unexpected_production_change_detected"
            classifications = list(
                dict.fromkeys([*classifications, "operations_check_failed"])
            )

        cycle_completed_at = _now_iso()
        successful = _cycle_completed_successfully(stop_reason)
        state = _next_state(
            previous_state,
            cycle_at=cycle_completed_at,
            successful=successful,
            health_summary=health_summary,
            freshness=freshness,
            binary_metrics=binary_metrics,
            confidence=confidence,
            shadow=shadow,
            append_only=append_only,
            findings=findings,
            report_path=str(report_paths["markdown"].resolve()),
        )
        state_changes = _state_changes(previous_state, state)
        files_modified = [
            str(state_file.resolve()),
            str(queue_file.resolve()),
            str(report_paths["markdown"].resolve()),
            str(report_paths["json"].resolve()),
        ]
        result = {
            "schema_version": SCHEMA_VERSION,
            "cycle_started_at": cycle_started_at,
            "cycle_completed_at": cycle_completed_at,
            "ticker": ticker,
            "dry_run": True,
            "risk_level": "Level 0: read-only",
            "overall_classification": classifications,
            "health_summary": health_summary,
            "daily_prediction": daily_prediction,
            "task_scheduler": scheduler,
            "data_freshness": _json_ready(freshness),
            "live_prediction_coverage": live_coverage,
            "unknown_source_findings": unknown_sources,
            "missing_live_findings": missing_live,
            "live_binary_metrics": binary_metrics,
            "baseline_comparison": binary_metrics.get("baselines", []),
            "confidence_summary": confidence,
            "shadow_summary": shadow,
            "append_only_summary": append_only,
            "changes_since_previous_cycle": state_changes,
            "state_was_initialized": first_cycle,
            "tasks_created": task_changes["created"],
            "tasks_updated": task_changes["updated"],
            "tasks_resolved": task_changes["resolved"],
            "human_approval_required": [
                task["fingerprint"]
                for task in queue["tasks"]
                if task.get("status") == "approval_required"
            ],
            "recommended_next_action": _recommended_next_action(queue),
            "files_modified": files_modified,
            "production_changes": {
                "Production model changed": "No"
                if not production_changes["models"]
                else "Unexpected external change detected",
                "Production settings changed": "No"
                if not production_changes["settings"]
                else "Unexpected external change detected",
                "News source config changed": "No"
                if not production_changes["news_sources"]
                else "Unexpected external change detected",
                "Production SQLite changed": "No"
                if not production_changes["database"]
                else "Unexpected external change detected",
                "daily_predict changed": "No"
                if not production_changes["daily_predict"]
                else "Unexpected external change detected",
                "Task Scheduler changed": "No"
                if not production_changes["task_scheduler"]
                else "Unexpected external change detected",
                "Live/backfill prediction created": "No"
                if not production_changes["database"]
                else "Unable to prove",
            },
            "database_validation": database_validation,
            "stop_reason": stop_reason or "completed_read_only_cycle",
            "report_paths": {key: str(value.resolve()) for key, value in report_paths.items()},
        }

        save_operations_state(state_file, state)
        _write_json(queue_file, queue)
        _write_json(report_paths["json"], result)
        report_paths["markdown"].write_text(
            render_operations_markdown(result),
            encoding="utf-8",
        )
        return result


def classify_operations_cycle(
    *,
    health_summary: dict[str, Any],
    freshness: dict[str, Any],
    scheduler: dict[str, Any],
    unknown_sources: list[dict[str, Any]],
    missing_live: list[dict[str, Any]],
    binary_metrics: dict[str, Any],
    shadow: dict[str, Any],
    append_only: dict[str, Any],
) -> list[str]:
    classifications: list[str] = []
    actionable: list[str] = []
    if unknown_sources:
        actionable.append("unknown_source_review_needed")
    if missing_live:
        actionable.append("missing_live_review_needed")
    if bool(freshness.get("stale_at_run_time")):
        actionable.append("stale_data_review_needed")
    if scheduler.get("needs_review"):
        actionable.append("scheduler_review_needed")
    if _live_model_degraded(binary_metrics):
        actionable.append("live_model_degradation")
    if shadow.get("needs_review"):
        actionable.append("shadow_review_needed")
    classifications.extend(actionable)
    if append_only.get("mismatch_count", 0) > 0:
        classifications.append("append_only_info_mismatch")
    if len(actionable) > 1:
        classifications.append("multiple_issues_require_review")
    if health_summary.get("action_required_count", 0) > 0 and not actionable:
        classifications.append("operations_check_failed")
    if not actionable and health_summary.get("known_ok_count", 0) + health_summary.get(
        "known_missed_date_count", 0
    ) > 0:
        classifications.insert(0, "healthy_known_warn")
    elif not actionable:
        classifications.insert(0, "healthy_no_action")
    return list(dict.fromkeys(classifications))


def render_operations_markdown(result: dict[str, Any]) -> str:
    health = result["health_summary"]
    daily_prediction = result.get("daily_prediction", {})
    scheduler = result["task_scheduler"]
    freshness = result["data_freshness"]
    coverage = result["live_prediction_coverage"]
    metrics = result["live_binary_metrics"]
    confidence = result["confidence_summary"]
    shadow = result["shadow_summary"]
    append_only = result["append_only_summary"]
    production = result["production_changes"]
    lines = [
        "# Autonomous Operations Report",
        "",
        f"- Cycle start: {result['cycle_started_at']}",
        f"- Cycle end: {result['cycle_completed_at']}",
        f"- Ticker: {result['ticker']}",
        f"- Overall classification: {', '.join(result['overall_classification'])}",
        f"- Risk level: {result['risk_level']}",
        "",
        "## Health Summary",
        "",
        f"- Overall status: {health.get('overall_status')}",
        f"- Action required: {health.get('action_required_count')}",
        f"- Known missed sections: {health.get('known_missed_section_count')}",
        f"- Known missed dates: {health.get('known_missed_date_count')} ({_join(health.get('known_missed_dates'))})",
        f"- Known OK: {health.get('known_ok_count')}",
        f"- Unknown sources: {health.get('unknown_source_count')}",
        f"- check-news: {health.get('check_news_status')}",
        "",
        "## Latest Daily Prediction Run",
        "",
        f"- Result: {daily_prediction.get('result')}",
        f"- Latest event: {daily_prediction.get('latest_event')}",
        f"- Latest success: {daily_prediction.get('latest_success')}",
        f"- Data freshness at run: {daily_prediction.get('data_freshness_status')}",
        f"- Expected / raw / feature date at run: {daily_prediction.get('expected_latest_trading_date')} / {daily_prediction.get('latest_raw_date_at_run')} / {daily_prediction.get('latest_feature_date_at_run')}",
        f"- Prediction feature / expected saved date: {daily_prediction.get('prediction_feature_date_at_run')} / {daily_prediction.get('saved_prediction_date_expected')}",
        f"- Stale at run time: {daily_prediction.get('stale_at_run_time')}",
        "",
        "## Task Scheduler Summary",
        "",
        f"- Task: {scheduler.get('task_name')}",
        f"- Registered: {scheduler.get('registered')}",
        f"- Trigger time: {scheduler.get('trigger_time')}",
        f"- Last run: {scheduler.get('last_run_time')}",
        f"- Last result: {scheduler.get('last_result')}",
        f"- Next run: {scheduler.get('next_run_time')}",
        f"- Command: {scheduler.get('command')}",
        f"- Working directory: {scheduler.get('working_directory')}",
        f"- Duplicate task count: {scheduler.get('duplicate_task_count')}",
        f"- 10:30 maintained: {scheduler.get('scheduled_time_1030')}",
        "",
        "## Data Freshness Summary",
        "",
        f"- Expected trading date: {freshness.get('expected_latest_trading_date')}",
        f"- Latest raw date: {freshness.get('latest_raw_date')}",
        f"- Latest feature date: {freshness.get('latest_feature_date')}",
        f"- Lag trading days: {freshness.get('freshness_lag_trading_days')}",
        f"- Status: {freshness.get('freshness_status')}",
        f"- Stale at run time: {freshness.get('stale_at_run_time')}",
        "",
        "## Live Prediction Coverage",
        "",
        f"- Latest binary feature date: {coverage.get('latest_binary_feature_date')}",
        f"- Latest multiclass feature date: {coverage.get('latest_multiclass_feature_date')}",
        f"- Both saved for latest date: {coverage.get('both_saved_for_latest_date')}",
        f"- Binary confirmed / pending: {coverage.get('binary_confirmed_count')} / {coverage.get('binary_pending_count')}",
        f"- Multiclass confirmed / pending: {coverage.get('multiclass_confirmed_count')} / {coverage.get('multiclass_pending_count')}",
        "",
        "## Unknown Source Findings",
        "",
        *_finding_lines(result["unknown_source_findings"], "No unknown sources."),
        "",
        "## Missing Live Findings",
        "",
        *_finding_lines(result["missing_live_findings"], "No new unknown post-start missing dates."),
        "",
        "## Live Binary Metrics",
        "",
        f"- Rows / confirmed / pending: {metrics.get('live_rows')} / {metrics.get('confirmed')} / {metrics.get('pending')}",
        f"- Evaluated period: {metrics.get('evaluated_start')} to {metrics.get('evaluated_end')}",
        f"- Accuracy: {_percent(metrics.get('accuracy'))}",
        f"- Balanced accuracy: {_percent(metrics.get('balanced_accuracy'))}",
        f"- Up / down recall: {_percent(metrics.get('up_recall'))} / {_percent(metrics.get('down_recall'))}",
        f"- Up / down precision: {_percent(metrics.get('up_precision'))} / {_percent(metrics.get('down_precision'))}",
        "",
        "## Baseline Comparison",
        "",
        *_baseline_lines(result["baseline_comparison"]),
        "",
        "## Confidence Summary",
        "",
        f"- Classification: {confidence.get('classification')}",
        *_confidence_lines(confidence.get("thresholds", [])),
        f"- Latest high-confidence prediction: {confidence.get('latest_high_confidence_prediction')}",
        "",
        "## Shadow Summary",
        "",
        f"- File status: {shadow.get('file_status')}",
        f"- Total / confirmed / pending: {shadow.get('total_rows')} / {shadow.get('confirmed_count')} / {shadow.get('pending_count')}",
        f"- Conclusions: {_join(shadow.get('conclusions'))}",
        "",
        "## Append-Only Summary",
        "",
        f"- Runs / records / snapshot rows: {append_only.get('prediction_runs_count')} / {append_only.get('prediction_records_count')} / {append_only.get('existing_predictions_count')}",
        f"- Latest run: {append_only.get('latest_run_id')}",
        f"- Latest append-only / snapshot feature date: {append_only.get('latest_feature_date')} / {append_only.get('latest_snapshot_feature_date')}",
        f"- Snapshot compare: {append_only.get('compare_status')}",
        f"- Mismatch count: {append_only.get('mismatch_count')}",
        f"- Selected official candidate: {append_only.get('selected_official_candidate_count')}",
        "",
        "## Changes Since Previous Cycle",
        "",
        *[f"- {key}: {value}" for key, value in result["changes_since_previous_cycle"].items()],
        "",
        "## Task Queue Changes",
        "",
        f"- Created: {_join(result['tasks_created'])}",
        f"- Updated: {_join(result['tasks_updated'])}",
        f"- Resolved: {_join(result['tasks_resolved'])}",
        f"- Human approval required: {_join(result['human_approval_required'])}",
        f"- Recommended next action: {result['recommended_next_action']}",
        "",
        "## Safety Record",
        "",
        f"- Production model changed: {production['Production model changed']}",
        f"- Production settings changed: {production['Production settings changed']}",
        f"- News source config changed: {production['News source config changed']}",
        f"- Production SQLite changed: {production['Production SQLite changed']}",
        f"- daily_predict changed: {production['daily_predict changed']}",
        f"- Task Scheduler changed: {production['Task Scheduler changed']}",
        f"- Live/backfill prediction created: {production['Live/backfill prediction created']}",
        f"- Files modified: {_join(result['files_modified'])}",
        f"- Stop reason: {result['stop_reason']}",
    ]
    return "\n".join(lines) + "\n"


def _summarize_health(health: dict[str, Any]) -> dict[str, Any]:
    summary = health.get("warning_summary") or {}
    missing_item = _health_item(health, "SQLite", "Missing live dates")
    news_item = _health_item(health, "News", "News CSV quality")
    known_rows = (missing_item.get("details") or {}).get("known_missed_live_dates") or []
    diagnostics = (missing_item.get("details") or {}).get("missing_live_date_diagnostics") or []
    known_dates = [str(row.get("missing_date")) for row in known_rows if row.get("missing_date")]
    post_start_dates = [
        str(row.get("missing_date")) for row in diagnostics if row.get("missing_date")
    ]
    known_missed_section_count = safe_int(summary.get("known_missed_section_count"))
    known_missed_date_count = safe_int(summary.get("known_missed_date_count"))
    return {
        "overall_status": health.get("overall_status"),
        "action_required_count": int(summary.get("action_required_count", 0)),
        "known_missed_section_count": (
            max(known_missed_section_count or 0, int(bool(known_dates)))
        ),
        "known_missed_date_count": (
            max(known_missed_date_count or 0, len(known_dates))
        ),
        "known_ok_count": int(summary.get("known_ok_count", 0)),
        "known_missed_dates": known_dates,
        "post_start_missing_dates": post_start_dates,
        "pre_start_excluded_date_count": safe_int(
            (missing_item.get("details") or {}).get("pre_start_excluded_date_count")
        )
        or 0,
        "unknown_missing_dates": [
            row
            for row in diagnostics
            if isinstance(row, dict) and not bool(row.get("known_missed"))
        ],
        "unknown_source_count": safe_int(
            (news_item.get("details") or {}).get("unknown_source_count")
        )
        or 0,
        "check_news_status": (news_item.get("details") or {}).get("status")
        or news_item.get("status"),
        "topix_fallback": any(
            item.get("name") == "TOPIX parquet"
            and item.get("classification") == "known_ok"
            for item in summary.get("known_ok", [])
        ),
        "action_required": summary.get("action_required", []),
        "known_ok": summary.get("known_ok", []),
        "known_missed": summary.get("known_missed", []),
    }


def _summarize_daily_prediction(
    health: dict[str, Any],
    scheduler: dict[str, Any],
) -> dict[str, Any]:
    item = _health_item(health, "daily_predict", "Latest run")
    details = dict(item.get("details") or {})
    latest_event = details.get("latest_event") or details.get("latest_success")
    result = "success" if item.get("status") == "OK" and details.get("latest_success") else "review"
    if details.get("latest_error"):
        result = "failed"
    return {
        "result": result,
        "health_status": item.get("status"),
        "latest_event": latest_event,
        "latest_success": details.get("latest_success"),
        "latest_error": details.get("latest_error"),
        "task_last_run_time": scheduler.get("last_run_time"),
        "task_last_result": scheduler.get("last_result"),
        "data_freshness_status": details.get("data_freshness_status"),
        "expected_latest_trading_date": details.get("expected_latest_trading_date"),
        "latest_raw_date_at_run": details.get("latest_raw_date_at_run"),
        "latest_feature_date_at_run": details.get("latest_feature_date_at_run"),
        "prediction_feature_date_at_run": details.get("prediction_feature_date_at_run"),
        "saved_prediction_date_expected": details.get("saved_prediction_date_expected"),
        "freshness_lag_trading_days": details.get("freshness_lag_trading_days"),
        "stale_at_run_time": details.get("stale_at_run_time"),
        "binary_and_multiclass_steps_completed": bool(
            details.get("last_done_step") == "predict-multiclass"
            or details.get("latest_success")
        ),
    }


def _summarize_live_coverage(
    predictions: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> dict[str, Any]:
    binary = build_binary_diagnostic_frame(predictions, feature_data, prediction_type="live")
    multiclass = build_multiclass_diagnostic_frame(
        predictions,
        feature_data,
        prediction_type="live",
    )
    binary_latest = _latest_date(binary.get("date"))
    multiclass_latest = _latest_date(multiclass.get("date"))
    return {
        "latest_binary_feature_date": binary_latest,
        "latest_multiclass_feature_date": multiclass_latest,
        "both_saved_for_latest_date": bool(
            binary_latest and binary_latest == multiclass_latest
        ),
        "binary_live_rows": int(len(binary)),
        "multiclass_live_rows": int(len(multiclass)),
        "binary_confirmed_count": _status_count(binary, "confirmed"),
        "binary_pending_count": _status_count(binary, "pending"),
        "multiclass_confirmed_count": _status_count(multiclass, "confirmed"),
        "multiclass_pending_count": _status_count(multiclass, "pending"),
    }


def _summarize_live_binary_metrics(
    predictions: pd.DataFrame,
    feature_data: pd.DataFrame,
    previous_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    frame = build_binary_diagnostic_frame(predictions, feature_data, prediction_type="live")
    comparison = source_comparison_frame({"live": frame}, min_sample_warning=30)
    row = comparison.iloc[0].to_dict() if not comparison.empty else {}
    confirmed = frame[frame.get("status", pd.Series(dtype=str)).eq("confirmed")].copy()
    pending = frame[frame.get("status", pd.Series(dtype=str)).eq("pending")].copy()
    actual = pd.to_numeric(confirmed.get("actual_class"), errors="coerce")
    predicted = pd.to_numeric(confirmed.get("predicted_class"), errors="coerce")
    baselines = baseline_accuracy_frame(frame)
    result = {
        "live_rows": int(len(frame)),
        "confirmed": int(len(confirmed)),
        "pending": int(len(pending)),
        "evaluated_start": row.get("evaluated_start"),
        "evaluated_end": row.get("evaluated_end"),
        "latest_confirmed_feature_date": _latest_date(confirmed.get("date")),
        "latest_pending_feature_date": _latest_date(pending.get("date")),
        "accuracy": _float(row.get("model_accuracy")),
        "balanced_accuracy": _float(row.get("balanced_accuracy")),
        "up_recall": _float(row.get("up_recall")),
        "down_recall": _float(row.get("down_recall")),
        "up_precision": _float(row.get("predicted_up_accuracy")),
        "down_precision": _float(row.get("predicted_down_accuracy")),
        "predicted_up_accuracy": _float(row.get("predicted_up_accuracy")),
        "predicted_down_accuracy": _float(row.get("predicted_down_accuracy")),
        "predicted_up_count": int((predicted == 1).sum()),
        "predicted_down_count": int((predicted == 0).sum()),
        "actual_up_count": int((actual == 1).sum()),
        "actual_down_count": int((actual == 0).sum()),
        "baselines": _records(baselines),
        "_frame": frame,
    }
    tracked = {
        key: result.get(key)
        for key in (
            "confirmed",
            "accuracy",
            "balanced_accuracy",
            "up_recall",
            "down_recall",
            "up_precision",
            "down_precision",
        )
    }
    result["change_from_previous"] = _metric_delta_map(
        tracked,
        (previous_state or {}).get("last_live_binary_metrics", {}),
    )
    return result


def _summarize_confidence(
    frame: pd.DataFrame,
    previous_state: dict[str, Any],
) -> dict[str, Any]:
    confirmed = frame[frame.get("status", pd.Series(dtype=str)).eq("confirmed")].copy()
    confidence = pd.to_numeric(confirmed.get("confidence"), errors="coerce")
    thresholds: list[dict[str, Any]] = []
    for threshold in CONFIDENCE_THRESHOLDS:
        rows = confirmed[confidence >= threshold]
        hit = int(rows.get("is_correct", pd.Series(dtype=bool)).eq(True).sum())
        miss = int(rows.get("is_correct", pd.Series(dtype=bool)).eq(False).sum())
        thresholds.append(
            {
                "threshold": threshold,
                "count": int(len(rows)),
                "hit": hit,
                "miss": miss,
                "hit_rate": _safe_divide(hit, hit + miss),
            }
        )
    high = frame[pd.to_numeric(frame.get("confidence"), errors="coerce") >= 0.60].copy()
    high = high.sort_values("date", ascending=False) if not high.empty else high
    latest = high.iloc[0].to_dict() if not high.empty else None
    classification = _confidence_classification(thresholds)
    previous = previous_state.get("last_confidence_metrics") or {}
    return {
        "thresholds": thresholds,
        "latest_high_confidence_prediction": _json_ready(latest),
        "classification": classification,
        "change_from_previous": _metric_delta_map(
            {str(row["threshold"]): row.get("hit_rate") for row in thresholds},
            previous.get("hit_rates", {}),
        ),
        "hit_rates": {str(row["threshold"]): row.get("hit_rate") for row in thresholds},
    }


def _summarize_shadow(live_predictions: pd.DataFrame) -> dict[str, Any]:
    if not SHADOW_PATH.exists():
        return {
            "file_status": "missing",
            "path": str(SHADOW_PATH),
            "total_rows": 0,
            "confirmed_count": 0,
            "pending_count": 0,
            "candidate_summary": [],
            "direction_agreement_count": 0,
            "direction_disagreement_count": 0,
            "class_weight_balanced_won_dates": [],
            "production_current_won_dates": [],
            "conclusions": ["shadow_file_missing", "continue_shadow_monitoring"],
            "needs_review": True,
        }
    try:
        rows = pd.read_csv(SHADOW_PATH, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return {
            "file_status": "unreadable",
            "path": str(SHADOW_PATH),
            "total_rows": 0,
            "confirmed_count": 0,
            "pending_count": 0,
            "candidate_summary": [],
            "conclusions": ["shadow_file_unreadable", "continue_shadow_monitoring"],
            "needs_review": True,
        }
    summary = summarize_shadow_predictions(rows)
    comparison = compare_shadow_with_live(rows, live_predictions)
    result_text = rows.get("result", pd.Series(dtype=str)).astype(str)
    balanced_wins, current_wins = _shadow_win_dates(rows)
    confirmed_count = int(result_text.isin(["hit", "miss"]).sum())
    modified = datetime.fromtimestamp(SHADOW_PATH.stat().st_mtime, tz=UTC)
    age_days = max(0, (datetime.now(UTC) - modified).days)
    conclusions = classify_shadow_conclusions(summary)
    return {
        "file_status": "stale" if age_days > 7 else "available",
        "path": str(SHADOW_PATH),
        "modified_at": modified.isoformat(),
        "age_days": age_days,
        "total_rows": int(len(rows)),
        "confirmed_count": confirmed_count,
        "pending_count": int(result_text.eq("pending").sum()),
        "candidate_summary": _records(summary),
        "direction_agreement_count": int(
            comparison.get("direction_matches_live", pd.Series(dtype=bool)).eq(True).sum()
        ),
        "direction_disagreement_count": int(
            comparison.get("direction_matches_live", pd.Series(dtype=bool)).eq(False).sum()
        ),
        "class_weight_balanced_won_dates": balanced_wins,
        "production_current_won_dates": current_wins,
        "conclusions": conclusions,
        "needs_review": bool(age_days > 7 or confirmed_count < 10),
    }


def _summarize_append_only(database_path: Path, ticker: str) -> dict[str, Any]:
    visibility = build_append_only_health_visibility(database_path, ticker=ticker)
    comparison = compare_append_only_official_to_snapshot(database_path, ticker=ticker)
    selection = build_official_selection_diagnostics_from_db(database_path, ticker=ticker)
    return {
        "prediction_runs_count": int(visibility.get("prediction_runs_count", 0)),
        "prediction_records_count": int(visibility.get("prediction_records_count", 0)),
        "existing_predictions_count": int(visibility.get("existing_predictions_count", 0)),
        "latest_run_id": visibility.get("latest_run_id"),
        "latest_run_timestamp": visibility.get("latest_run_timestamp"),
        "latest_run_type": visibility.get("latest_run_type"),
        "latest_feature_date": visibility.get("latest_feature_date"),
        "latest_snapshot_feature_date": _latest_compare_feature_date(comparison),
        "compare_status": visibility.get("latest_compare_status"),
        "snapshot_matches": bool(comparison.get("matches")),
        "mismatch_count": int(visibility.get("mismatch_count", 0)),
        "scheduled_run_count": int(selection.get("scheduled_run_count", 0)),
        "manual_run_count": int(selection.get("manual_run_count", 0)),
        "selected_official_candidate_count": int(
            selection.get("selected_official_candidate_count", 0)
        ),
        "visibility_classifications": visibility.get("visibility_classifications", []),
        "info_only": True,
    }


def _diagnose_unknown_sources(settings: Settings, ticker: str) -> list[dict[str, Any]]:
    quality = check_news_for_ticker(settings.paths.raw_data_dir.parent / "news", ticker)
    if int(quality.get("unknown_source_count", 0)) <= 0:
        return []
    path = Path(str(quality["path"]))
    try:
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError):
        return []
    source_types = rows.get(
        "source_type",
        pd.Series("", index=rows.index, dtype=str),
    )
    unknown = rows[source_types.astype(str).str.lower().isin(["", "unknown"])]
    if "source" not in unknown.columns:
        return []
    profiles = load_news_source_profiles()
    findings: list[dict[str, Any]] = []
    for source, group in unknown.groupby("source", dropna=False):
        source_name = str(source or "").strip() or "(empty)"
        latest = group.sort_values("date", ascending=False).iloc[0]
        alias_candidate = _known_alias_candidate(source_name, profiles) or _closest_source_profile(
            source_name,
            profiles,
        )
        title_example = str(latest.get("title") or "")
        source_quality = (
            "mojibake_or_placeholder_suspect"
            if _looks_like_mojibake_or_placeholder(f"{source_name} {title_example}")
            else "readable"
        )
        proposal = None
        if source_quality == "readable":
            proposal = (
                _alias_addition_proposal(source_name, alias_candidate, profiles)
                if alias_candidate and alias_candidate.get("match_method") == "known_translation"
                else _source_profile_proposal(
                    source_name,
                    str(latest.get("url") or ""),
                )
            )
        findings.append(
            {
                "source": source_name,
                "count": int(len(group)),
                "title_example": title_example,
                "url_example": str(latest.get("url") or ""),
                "published_date": str(latest.get("date") or ""),
                "query_profile": str(latest.get("query_profile") or ""),
                "category": str(latest.get("category") or ""),
                "source_variants": sorted(set(group["source"].astype(str))),
                "source_quality": source_quality,
                "existing_alias_candidate": alias_candidate,
                "classification_candidate": (
                    proposal["source_type"] if proposal else "unknown"
                ),
                "profile_proposal": proposal,
                "proposal_note": (
                    "No profile proposal was generated because the source/title may be corrupted or a placeholder."
                    if proposal is None
                    else proposal["reason"]
                ),
            }
        )
    return findings


def _diagnose_missing_live(
    health_summary: dict[str, Any],
    *,
    raw_data: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> list[dict[str, Any]]:
    findings = []
    for row in health_summary.get("unknown_missing_dates", []):
        missing_date = str(row.get("missing_date") or "")
        raw_exists = _frame_has_date(raw_data, missing_date)
        feature_exists = _frame_has_date(feature_data, missing_date)
        log_evidence = _log_lines_for_date(DAILY_LOG_PATH, missing_date)
        reason = _missing_reason(row, raw_exists, feature_exists, log_evidence)
        findings.append(
            {
                "missing_date": missing_date,
                "expected_trading_date": True,
                "raw_data_exists": raw_exists,
                "feature_data_exists": feature_exists,
                "binary_live_exists": bool(row.get("binary_live_exists")),
                "multiclass_live_exists": bool(row.get("multiclass_live_exists")),
                "backfill_exists": bool(row.get("backfill_exists")),
                "previous_live_prediction_date": row.get("previous_live_prediction_date"),
                "next_live_prediction_date": row.get("next_live_prediction_date"),
                "task_scheduler_history": _task_scheduler_events(missing_date),
                "daily_predict_log_evidence": log_evidence,
                "pc_power_evidence": _pc_power_events(missing_date),
                "reason_code": reason,
                "recommended_action": (
                    "Review the evidence; do not create a live row or backfill automatically."
                ),
            }
        )
    return findings


def _build_findings(
    *,
    ticker: str,
    health_summary: dict[str, Any],
    freshness: dict[str, Any],
    scheduler: dict[str, Any],
    unknown_sources: list[dict[str, Any]],
    missing_live: list[dict[str, Any]],
    binary_metrics: dict[str, Any],
    confidence: dict[str, Any],
    shadow: dict[str, Any],
    append_only: dict[str, Any],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for item in unknown_sources:
        findings.append(
            _finding(
                fingerprint=f"unknown_source:{item['source']}",
                category="unknown_source",
                risk_level="Level 2",
                priority="high",
                title=f"Review unknown news source: {item['source']}",
                description="Source profile proposal requires human review; no config was changed.",
                evidence=item,
                proposed_action="Review the proposed conservative source profile and approve or reject a later config change.",
                requires_human_approval=True,
            )
        )
    for item in missing_live:
        findings.append(
            _finding(
                fingerprint=f"missing_live:{ticker}:{item['missing_date']}",
                category="missing_live",
                risk_level="Level 2",
                priority="high",
                title=f"Review missing live feature date {item['missing_date']}",
                description="The live gap is not registered as known missed and requires human review.",
                evidence=item,
                proposed_action="Confirm the root cause; do not fabricate live/backfill rows or edit known-missed config automatically.",
                requires_human_approval=True,
            )
        )
    if bool(freshness.get("stale_at_run_time")):
        findings.append(
            _finding(
                fingerprint=f"stale_data:{ticker}:{freshness.get('expected_latest_trading_date')}",
                category="stale_data",
                risk_level="Level 2",
                priority="high",
                title="Review stale data at the latest daily run",
                description=str(freshness.get("message") or "Feature data was stale."),
                evidence=_json_ready(freshness),
                proposed_action="Investigate data availability and scheduler timing; do not change the schedule automatically.",
                requires_human_approval=True,
            )
        )
    if scheduler.get("needs_review"):
        findings.append(
            _finding(
                fingerprint=f"scheduler_failure:{scheduler.get('task_name')}:{scheduler.get('last_run_time')}",
                category="scheduler",
                risk_level="Level 2",
                priority="critical" if scheduler.get("last_result") not in (0, None) else "high",
                title="Review the daily prediction Task Scheduler configuration",
                description="The scheduled task is missing, changed, duplicated, or returned a failure.",
                evidence=scheduler,
                proposed_action="Inspect the task manually; Autonomous Operations must not update Task Scheduler.",
                requires_human_approval=True,
            )
        )
    if _live_model_degraded(binary_metrics):
        findings.append(
            _finding(
                fingerprint=f"model_up_recall_low:{ticker}",
                category="live_model",
                risk_level="Level 2",
                priority="medium",
                title="Continue review of weak live binary direction performance",
                description="Live binary metrics remain weak or materially below simple baselines.",
                evidence={
                    key: value
                    for key, value in binary_metrics.items()
                    if key not in {"baselines"}
                },
                proposed_action="Continue monitoring; any model or threshold change requires a separate approved task.",
                requires_human_approval=True,
            )
        )
    if confidence.get("classification") in {"confidence_signal_negative", "sample_too_small"}:
        findings.append(
            _finding(
                fingerprint=f"confidence_monitoring:{ticker}",
                category="confidence",
                risk_level="Level 0",
                priority="low",
                title="Continue confidence monitoring",
                description=f"Confidence classification: {confidence.get('classification')}",
                evidence=confidence,
                proposed_action="Collect more confirmed live rows before drawing conclusions.",
                requires_human_approval=False,
            )
        )
    if shadow.get("needs_review"):
        findings.append(
            _finding(
                fingerprint=f"shadow_review:{ticker}",
                category="shadow",
                risk_level="Level 0",
                priority="low",
                title="Continue binary shadow monitoring",
                description="Shadow data is missing, stale, or has too few confirmed rows.",
                evidence=shadow,
                proposed_action="Review existing shadow observations only; do not generate new shadow predictions automatically.",
                requires_human_approval=False,
            )
        )
    if append_only.get("mismatch_count", 0) > 0:
        findings.append(
            _finding(
                fingerprint=f"append_only_mismatch:{ticker}",
                category="append_only",
                risk_level="Level 0",
                priority="low",
                title="Append-only snapshot mismatch remains informational",
                description="Automatic daily append-only saving is disabled, so an older audit run may differ from the latest snapshot.",
                evidence=append_only,
                proposed_action="Monitor only; do not change append-only rows or official/latest flags.",
                requires_human_approval=False,
            )
        )
    covered = {"unknown_source", "missing_live", "stale_data", "scheduler"}
    for item in health_summary.get("action_required", []):
        details = item.get("details") or {}
        if item.get("name") == "News CSV quality" and details.get("unknown_source_count"):
            continue
        if item.get("name") == "Missing live dates":
            continue
        category = "health_check"
        if category in covered:
            continue
        fingerprint = f"health:{item.get('section')}:{item.get('name')}"
        findings.append(
            _finding(
                fingerprint=fingerprint,
                category=category,
                risk_level="Level 1",
                priority="high",
                title=f"Review health-check item: {item.get('name')}",
                description=str(item.get("message") or "Health-check requires review."),
                evidence=item,
                proposed_action="Investigate manually; Autonomous Operations must not repair code or production state.",
                requires_human_approval=False,
            )
        )
    return _deduplicate_findings(findings)


def _next_state(
    previous: dict[str, Any],
    *,
    cycle_at: str,
    successful: bool,
    health_summary: dict[str, Any],
    freshness: dict[str, Any],
    binary_metrics: dict[str, Any],
    confidence: dict[str, Any],
    shadow: dict[str, Any],
    append_only: dict[str, Any],
    findings: list[dict[str, Any]],
    report_path: str,
) -> dict[str, Any]:
    state = default_operations_state()
    state.update(previous)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "last_cycle_at": cycle_at,
            "last_successful_cycle_at": (
                cycle_at if successful else previous.get("last_successful_cycle_at")
            ),
            "last_health_status": health_summary.get("overall_status"),
            "last_action_required_count": health_summary.get("action_required_count"),
            "last_unknown_source_count": health_summary.get("unknown_source_count"),
            "last_known_missed_dates": health_summary.get("known_missed_dates", []),
            "last_post_start_missing_dates": health_summary.get(
                "post_start_missing_dates", []
            ),
            "last_expected_trading_date": freshness.get("expected_latest_trading_date"),
            "last_raw_date": freshness.get("latest_raw_date"),
            "last_feature_date": freshness.get("latest_feature_date"),
            "last_live_binary_confirmed_count": binary_metrics.get("confirmed"),
            "last_live_binary_metrics": {
                key: value
                for key, value in binary_metrics.items()
                if key not in {"baselines", "change_from_previous"}
            },
            "last_confidence_metrics": {
                "thresholds": confidence.get("thresholds", []),
                "classification": confidence.get("classification"),
                "hit_rates": confidence.get("hit_rates", {}),
            },
            "last_shadow_confirmed_count": shadow.get("confirmed_count"),
            "last_append_only_run_count": append_only.get("prediction_runs_count"),
            "last_append_only_record_count": append_only.get("prediction_records_count"),
            "last_snapshot_compare_status": append_only.get("compare_status"),
            "open_findings": [finding["fingerprint"] for finding in findings],
            "consecutive_failures": 0
            if successful
            else int(previous.get("consecutive_failures") or 0) + 1,
            "last_report_path": report_path,
        }
    )
    return _json_ready(state)


def _write_stopped_cycle(
    *,
    ticker: str,
    cycle_started_at: str,
    report_paths: dict[str, Path],
    queue_path: Path,
    error: str,
    stop_reason: str,
    previous_state: dict[str, Any] | None = None,
    state_path: Path | None = None,
    write_queue: bool = True,
) -> dict[str, Any]:
    cycle_completed_at = _now_iso()
    finding = _finding(
        fingerprint=f"operations_stop:{stop_reason}",
        category="operations",
        risk_level="Level 1",
        priority="critical",
        title="Autonomous Operations stopped safely",
        description=error,
        evidence={"stop_reason": stop_reason},
        proposed_action="Review the stop reason before running another cycle.",
        requires_human_approval=False,
    )
    changes = {"created": [], "updated": [], "resolved": []}
    queue = {"schema_version": SCHEMA_VERSION, "updated_at": None, "tasks": []}
    if write_queue:
        try:
            queue = load_task_queue(queue_path)
        except OperationsStateError:
            write_queue = False
        if write_queue:
            queue, changes = reconcile_task_queue(
                queue,
                [finding],
                cycle_at=cycle_started_at,
                source_report=str(report_paths["markdown"]),
            )
    files_modified = [
        str(report_paths["markdown"].resolve()),
        str(report_paths["json"].resolve()),
    ]
    if write_queue:
        files_modified.insert(0, str(queue_path.resolve()))
    result = {
        "schema_version": SCHEMA_VERSION,
        "cycle_started_at": cycle_started_at,
        "cycle_completed_at": cycle_completed_at,
        "ticker": ticker,
        "dry_run": True,
        "risk_level": "Level 0: read-only",
        "overall_classification": ["operations_check_failed"],
        "health_summary": {},
        "daily_prediction": {},
        "task_scheduler": {},
        "data_freshness": {},
        "live_prediction_coverage": {},
        "unknown_source_findings": [],
        "missing_live_findings": [],
        "live_binary_metrics": {},
        "baseline_comparison": [],
        "confidence_summary": {},
        "shadow_summary": {},
        "append_only_summary": {},
        "changes_since_previous_cycle": {},
        "tasks_created": changes["created"],
        "tasks_updated": changes["updated"],
        "tasks_resolved": changes["resolved"],
        "human_approval_required": [],
        "recommended_next_action": "Review the stop reason before retrying.",
        "files_modified": files_modified,
        "production_changes": {
            "Production model changed": "No",
            "Production settings changed": "No",
            "News source config changed": "No",
            "Production SQLite changed": "No",
            "daily_predict changed": "No",
            "Task Scheduler changed": "No",
            "Live/backfill prediction created": "No",
        },
        "stop_reason": stop_reason,
        "error": error,
        "report_paths": {key: str(value.resolve()) for key, value in report_paths.items()},
    }
    if previous_state is not None:
        previous_state = dict(previous_state)
        previous_state["last_cycle_at"] = cycle_completed_at
        previous_state["consecutive_failures"] = int(
            previous_state.get("consecutive_failures") or 0
        ) + 1
        previous_state["last_report_path"] = str(report_paths["markdown"])
        if state_path is not None:
            save_operations_state(state_path, previous_state)
            result["files_modified"].insert(0, str(state_path.resolve()))
    if write_queue:
        _write_json(queue_path, queue)
    _write_json(report_paths["json"], result)
    report_paths["markdown"].write_text(
        "# Autonomous Operations Report\n\n"
        f"- Cycle start: {cycle_started_at}\n"
        f"- Cycle end: {cycle_completed_at}\n"
        f"- Ticker: {ticker}\n"
        "- Overall classification: operations_check_failed\n"
        f"- Stop reason: {stop_reason}\n"
        f"- Error: {error}\n\n"
        "Production model changed: No\n\n"
        "Production settings changed: No\n\n"
        "News source config changed: No\n\n"
        "Production SQLite changed: No\n\n"
        "daily_predict changed: No\n\n"
        "Task Scheduler changed: No\n\n"
        "Live/backfill prediction created: No\n",
        encoding="utf-8",
    )
    return result


def _read_scheduler_status() -> dict[str, Any]:
    script = PROJECT_ROOT / "scripts" / "show_daily_predict_task.ps1"
    if not script.exists():
        return {"registered": False, "error": "show task script missing"}
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-AsJson",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0:
        return {"registered": None, "error": result.stderr.strip()}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return {"registered": None, "error": str(error)}


def _merge_scheduler_status(
    status: dict[str, Any],
    health: dict[str, Any],
    log_path: Path,
) -> dict[str, Any]:
    health_task = _health_item(health, "daily_predict", "Windows scheduled task")
    details = dict(health_task.get("details") or {})
    details.update(status)
    trigger_time = _scheduler_trigger_time(details.get("next_run_time"))
    command = str(details.get("command") or "")
    working = str(details.get("working_directory") or "")
    duplicate_count = _daily_predict_task_inventory_count() - (1 if details.get("registered") else 0)
    extra_runs = _unusual_daily_runs(log_path)
    last_run_date = str(details.get("last_run_time") or "")[:10]
    latest_extra_run = next(
        (item for item in reversed(extra_runs) if item.get("date") == last_run_date),
        None,
    )
    last_result = safe_int(details.get("last_result"))
    needs_review = bool(
        not details.get("registered")
        or last_result not in (0, None)
        or trigger_time != "10:30"
        or duplicate_count > 0
        or latest_extra_run is not None
        or "daily_predict.ps1" not in command
        or not _paths_equal(working, PROJECT_ROOT)
    )
    return {
        "task_name": details.get("task_name") or "stock-ai daily predict",
        "registered": details.get("registered"),
        "state": details.get("state"),
        "trigger_time": trigger_time,
        "last_run_time": details.get("last_run_time"),
        "last_result": last_result,
        "next_run_time": details.get("next_run_time"),
        "command": command,
        "working_directory": working,
        "duplicate_task_count": max(0, duplicate_count),
        "scheduled_time_1030": trigger_time == "10:30",
        "unusual_additional_runs": extra_runs,
        "latest_run_has_additional_execution": latest_extra_run is not None,
        "needs_review": needs_review,
        "health_status": health_task.get("status"),
    }


def _daily_predict_task_inventory_count() -> int:
    command = (
        "$tasks = Get-ScheduledTask -ErrorAction SilentlyContinue | "
        "Where-Object { (@($_.Actions) | Where-Object { "
        "$_.Arguments -like '*daily_predict.ps1*' }).Count -gt 0 }; "
        "@($tasks).Count"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return safe_int(result.stdout.strip()) or 0
    except (OSError, subprocess.SubprocessError):
        return 0


def _unusual_daily_runs(log_path: Path) -> list[dict[str, Any]]:
    if not log_path.exists():
        return []
    pattern = re.compile(r"^\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2})\].*daily_predict started")
    by_date: dict[str, list[str]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            by_date.setdefault(match.group(1), []).append(match.group(2))
    return [
        {"date": date_value, "run_times": times, "run_count": len(times)}
        for date_value, times in sorted(by_date.items())[-10:]
        if len(times) > 1
    ]


def _production_guard(settings: Settings) -> dict[str, Any]:
    paths = {
        "database": settings.paths.database_path,
        "settings": PROJECT_ROOT / "configs" / "settings.yml",
        "news_sources": PROJECT_ROOT / "configs" / "news_sources.yaml",
        "daily_predict": PROJECT_ROOT / "scripts" / "daily_predict.ps1",
        "binary_model": settings.paths.model_dir / "lightgbm_binary_7203.T.joblib",
        "multiclass_model": settings.paths.model_dir / "lightgbm_multiclass_7203.T.joblib",
    }
    return {name: _file_fingerprint(path) for name, path in paths.items()}


def _production_change_summary(
    before: dict[str, Any],
    after: dict[str, Any],
    scheduler_before: dict[str, Any],
    scheduler_after: dict[str, Any],
) -> dict[str, bool]:
    return {
        "database": before.get("database") != after.get("database"),
        "settings": before.get("settings") != after.get("settings"),
        "news_sources": before.get("news_sources") != after.get("news_sources"),
        "daily_predict": before.get("daily_predict") != after.get("daily_predict"),
        "models": any(
            before.get(name) != after.get(name)
            for name in ("binary_model", "multiclass_model")
        ),
        "task_scheduler": (
            _scheduler_configuration_fingerprint(scheduler_before)
            != _scheduler_configuration_fingerprint(scheduler_after)
        ),
    }


def _scheduler_configuration_fingerprint(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "registered": status.get("registered"),
        "task_name": status.get("task_name"),
        "command": status.get("command"),
        "working_directory": status.get("working_directory"),
        "trigger_time": _scheduler_trigger_time(status.get("next_run_time")),
    }


def _validate_database(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"readable": False, "error": "database file missing", "path": str(path)}
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if "predictions" not in tables:
                return {
                    "readable": False,
                    "error": "predictions table missing",
                    "path": str(path),
                }
            counts = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("predictions", "prediction_runs", "prediction_records")
                if table in tables
            }
        return {"readable": True, "path": str(path), "counts": counts}
    except sqlite3.Error as error:
        return {"readable": False, "error": str(error), "path": str(path)}


def _stop_reason(
    *,
    health_summary: dict[str, Any],
    database_validation: dict[str, Any],
    previous_state: dict[str, Any],
) -> str | None:
    if not database_validation.get("readable"):
        return "production_database_unreadable"
    if int(previous_state.get("consecutive_failures") or 0) >= 3:
        return "three_consecutive_failures"
    unknown_actions = health_summary.get("action_required", [])
    if len(unknown_actions) >= 2:
        return "multiple_action_required_items_need_human_review"
    return None


def _cycle_completed_successfully(stop_reason: str | None) -> bool:
    return stop_reason in {None, "multiple_action_required_items_need_human_review"}


def _finding(
    *,
    fingerprint: str,
    category: str,
    risk_level: str,
    priority: str,
    title: str,
    description: str,
    evidence: Any,
    proposed_action: str,
    requires_human_approval: bool,
) -> dict[str, Any]:
    return {
        "fingerprint": fingerprint,
        "category": category,
        "risk_level": risk_level,
        "priority": priority,
        "title": title,
        "description": description,
        "evidence": _json_ready(evidence),
        "proposed_action": proposed_action,
        "requires_human_approval": requires_human_approval,
        "blocked_reason": None,
    }


def _finding_task_status(finding: dict[str, Any]) -> str:
    if finding.get("blocked_reason"):
        return "blocked"
    if finding.get("requires_human_approval"):
        return "approval_required"
    if finding.get("risk_level") == "Level 0":
        return "monitoring"
    return "open"


def _deduplicate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({str(item["fingerprint"]): item for item in findings}.values())


def _task_id(fingerprint: str) -> str:
    digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:16]
    return f"operations_{digest}"


def _priority_rank(priority: str) -> int:
    return {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(priority, 9)


def _recommended_next_action(queue: dict[str, Any]) -> str:
    active = [
        task
        for task in queue.get("tasks", [])
        if task.get("status") in {"open", "monitoring", "approval_required", "blocked"}
    ]
    if not active:
        return "No immediate action. Continue the read-only monitoring cycle."
    active.sort(key=lambda task: _priority_rank(str(task.get("priority"))))
    return str(active[0].get("proposed_action") or active[0].get("title"))


def _source_profile_proposal(source: str, url: str) -> dict[str, Any]:
    source_text = source.lower()
    if any(token in source_text for token in ("yahoo", "google news")):
        source_type, credibility, impact = "aggregator", 0.45, 0.25
    elif any(
        token in source_text
        for token in ("finance", "market", "trading", "stock", "capital", "invest")
    ):
        source_type, credibility, impact = "financial_media", 0.50, 0.40
    else:
        source_type, credibility, impact = "general_news", 0.45, 0.25
    aliases = [source]
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host and host not in {"news.google.com"}:
        aliases.append(host)
    return {
        "name": source,
        "aliases": list(dict.fromkeys(aliases)),
        "source_type": source_type,
        "credibility_score": credibility,
        "market_impact_weight": impact,
        "reason": "Conservative read-only proposal; verify the original publisher before editing config.",
    }


def _looks_like_mojibake_or_placeholder(value: str) -> bool:
    text = str(value or "").strip()
    if not text or text in {"?", "??", "???", "????", "unknown", "(empty)"}:
        return True
    if "\ufffd" in text:
        return True
    mojibake_tokens = ("繝", "繧", "縺", "譁", "蜊", "隕", "逧")
    return sum(text.count(token) for token in mojibake_tokens) >= 3


def _closest_source_profile(source: str, profiles) -> dict[str, Any] | None:
    normalized = source.strip().lower()
    best: tuple[float, str, str] | None = None
    for profile in profiles:
        for alias in {profile.name.lower(), *(item.lower() for item in profile.aliases)}:
            if not alias:
                continue
            ratio = SequenceMatcher(None, normalized, alias).ratio()
            if best is None or ratio > best[0]:
                best = (ratio, profile.name, alias)
    if best is None or best[0] < 0.72:
        return None
    return {"profile": best[1], "alias": best[2], "similarity": round(best[0], 3)}


def _known_alias_candidate(source: str, profiles) -> dict[str, Any] | None:
    translations = {
        "ロイター": "Reuters",
        "ブルームバーグ": "Bloomberg",
        "日本経済新聞": "Nikkei",
    }
    target = translations.get(source.strip())
    if not target:
        return None
    profile = next((item for item in profiles if item.name.lower() == target.lower()), None)
    if profile is None:
        return None
    return {
        "profile": profile.name,
        "alias": source.strip(),
        "similarity": 1.0,
        "match_method": "known_translation",
    }


def _alias_addition_proposal(
    source: str,
    alias_candidate: dict[str, Any],
    profiles,
) -> dict[str, Any]:
    profile_name = str(alias_candidate["profile"])
    profile = next(item for item in profiles if item.name == profile_name)
    return {
        "name": profile.name,
        "aliases": list(dict.fromkeys([*profile.aliases, source.strip().lower()])),
        "source_type": profile.source_type,
        "credibility_score": profile.credibility_score,
        "market_impact_weight": profile.market_impact_weight,
        "proposal_type": "add_alias_to_existing_profile",
        "reason": (
            f"'{source}' appears to be a translated label for existing profile "
            f"'{profile.name}'. Review an alias-only config change."
        ),
    }


def _missing_reason(
    row: dict[str, Any],
    raw_exists: bool,
    feature_exists: bool,
    log_evidence: list[str],
) -> str:
    if not raw_exists:
        return "raw_data_missing"
    if not feature_exists:
        return "feature_generation_failed"
    if row.get("binary_live_exists") and not row.get("multiclass_live_exists"):
        return "prediction_save_failed"
    if row.get("multiclass_live_exists") and not row.get("binary_live_exists"):
        return "prediction_save_failed"
    if not log_evidence:
        return "task_scheduler_not_run"
    inferred = str(row.get("inferred_reason") or "")
    return inferred if inferred else "unknown_requires_manual_review"


def _pc_power_events(date_text: str) -> list[str]:
    if os.name != "nt" or not date_text:
        return []
    command = (
        f"$start=[datetime]'{date_text}'; $end=$start.AddDays(2); "
        "Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=$start; EndTime=$end; "
        "Id=12,13,1074,6005,6006} -ErrorAction SilentlyContinue | "
        "Select-Object -First 20 TimeCreated,Id,ProviderName | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        return [result.stdout.strip()] if result.stdout.strip() else []
    except (OSError, subprocess.SubprocessError):
        return []


def _task_scheduler_events(date_text: str) -> list[dict[str, Any]]:
    if os.name != "nt" or not date_text:
        return []
    command = (
        f"$start=[datetime]'{date_text}'; $end=$start.AddDays(2); "
        "$events=Get-WinEvent -FilterHashtable "
        "@{LogName='Microsoft-Windows-TaskScheduler/Operational'; "
        "StartTime=$start; EndTime=$end} -ErrorAction SilentlyContinue | "
        "Where-Object { $_.Message -like '*stock-ai daily predict*' } | "
        "Select-Object -First 30 TimeCreated,Id,LevelDisplayName,Message; "
        "@($events) | ConvertTo-Json -Depth 3 -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        if not result.stdout.strip():
            return []
        parsed = json.loads(result.stdout)
        rows = parsed if isinstance(parsed, list) else [parsed]
        return [row for row in rows if isinstance(row, dict)]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []


def _shadow_win_dates(rows: pd.DataFrame) -> tuple[list[str], list[str]]:
    if rows.empty or not {"feature_date", "candidate_name", "result"} <= set(rows.columns):
        return [], []
    pivot = rows.pivot_table(
        index="feature_date",
        columns="candidate_name",
        values="result",
        aggfunc="last",
    )
    if not {"production_current", "class_weight_balanced"} <= set(pivot.columns):
        return [], []
    balanced = pivot[
        pivot["class_weight_balanced"].eq("hit")
        & pivot["production_current"].eq("miss")
    ].index.astype(str).tolist()
    current = pivot[
        pivot["production_current"].eq("hit")
        & pivot["class_weight_balanced"].eq("miss")
    ].index.astype(str).tolist()
    return balanced, current


def _confidence_classification(thresholds: list[dict[str, Any]]) -> str:
    ge_60 = next((row for row in thresholds if row["threshold"] == 0.60), None)
    if ge_60 is None or ge_60["count"] < 5:
        return "sample_too_small"
    hit_rate = ge_60.get("hit_rate")
    if hit_rate is None:
        return "confidence_signal_unclear"
    if hit_rate >= 0.60:
        return "confidence_signal_positive"
    if hit_rate < 0.45:
        return "confidence_signal_negative"
    return "confidence_signal_unclear"


def _live_model_degraded(metrics: dict[str, Any]) -> bool:
    confirmed = int(metrics.get("confirmed") or 0)
    if confirmed < 15:
        return False
    balanced = _float(metrics.get("balanced_accuracy"))
    up_recall = _float(metrics.get("up_recall"))
    if up_recall is not None and up_recall <= 0.10:
        return True
    baselines = metrics.get("baselines", [])
    best_baseline = max(
        (_float(row.get("baseline_accuracy")) or -1 for row in baselines),
        default=-1,
    )
    accuracy = _float(metrics.get("accuracy"))
    return bool(
        balanced is not None
        and balanced < 0.48
        and accuracy is not None
        and accuracy < best_baseline - 0.02
    )


def _state_changes(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "last_health_status",
        "last_action_required_count",
        "last_unknown_source_count",
        "last_known_missed_dates",
        "last_expected_trading_date",
        "last_raw_date",
        "last_feature_date",
        "last_live_binary_confirmed_count",
        "last_shadow_confirmed_count",
        "last_append_only_run_count",
        "last_append_only_record_count",
        "last_snapshot_compare_status",
    )
    return {
        key: {"before": previous.get(key), "after": current.get(key)}
        for key in keys
        if previous.get(key) != current.get(key)
    }


def _metric_delta_map(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for key, value in current.items():
        now = _float(value)
        old = _float(previous.get(key))
        result[key] = None if now is None or old is None else now - old
    return result


def _report_paths(report_dir: Path, cycle_at: str, *, force: bool) -> dict[str, Path]:
    stamp = pd.Timestamp(cycle_at).strftime("%Y-%m-%d_%H%M")
    base = report_dir / f"operations_{stamp}"
    if not base.with_suffix(".md").exists() and not base.with_suffix(".json").exists():
        return {"markdown": base.with_suffix(".md"), "json": base.with_suffix(".json")}
    index = 2
    while True:
        candidate = report_dir / f"operations_{stamp}_{index}"
        if not candidate.with_suffix(".md").exists() and not candidate.with_suffix(".json").exists():
            return {
                "markdown": candidate.with_suffix(".md"),
                "json": candidate.with_suffix(".json"),
            }
        index += 1


def _health_item(health: dict[str, Any], section: str, name: str) -> dict[str, Any]:
    return next(
        (
            item
            for item in health.get("items", [])
            if item.get("section") == section and item.get("name") == name
        ),
        {},
    )


def _frame_has_date(frame: pd.DataFrame, date_text: str) -> bool:
    if frame.empty or "date" not in frame:
        return False
    values = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return bool(values.eq(date_text).any())


def _safe_read_parquet(path: Path) -> pd.DataFrame:
    try:
        return read_parquet(path)
    except (OSError, ValueError, FileNotFoundError):
        return pd.DataFrame()


def _log_lines_for_date(path: Path, date_text: str) -> list[str]:
    if not path.exists() or not date_text:
        return []
    return [
        line
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if date_text in line
    ][-40:]


def _status_count(frame: pd.DataFrame, status: str) -> int:
    return int(frame.get("status", pd.Series(dtype=str)).eq(status).sum())


def _latest_date(values: Any) -> str | None:
    if values is None:
        return None
    parsed = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    return pd.Timestamp(parsed.max()).strftime("%Y-%m-%d") if not parsed.empty else None


def _latest_compare_feature_date(comparison: dict[str, Any]) -> str | None:
    return _latest_date([row.get("feature_date") for row in comparison.get("rows", [])])


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_ready(frame.to_dict(orient="records")) if not frame.empty else []


def _float(value: Any) -> float | None:
    numeric = safe_float(value)
    return None if numeric is None else float(numeric)


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path.resolve()),
        "size": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _lock_owner_is_active(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = int(payload.get("pid"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    if pid == os.getpid():
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _scheduler_trigger_time(value: Any) -> str | None:
    text = str(value or "")
    match = re.search(r"\b(\d{2}:\d{2})(?::\d{2})?\b", text)
    return match.group(1) if match else None


def _paths_equal(left: str | Path | None, right: str | Path | None) -> bool:
    if not left or not right:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return False


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and pd.isna(value):
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    return value


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _join(values: Iterable[Any] | None) -> str:
    rows = list(values or [])
    return ", ".join(str(value) for value in rows) if rows else "none"


def _percent(value: Any) -> str:
    numeric = _float(value)
    return "not available" if numeric is None else f"{numeric:.1%}"


def _finding_lines(findings: list[dict[str, Any]], empty_text: str) -> list[str]:
    if not findings:
        return [f"- {empty_text}"]
    return [f"- {json.dumps(_json_ready(item), ensure_ascii=False)}" for item in findings]


def _baseline_lines(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No confirmed baseline comparison is available."]
    return [
        f"- {row.get('baseline')}: {_percent(row.get('baseline_accuracy'))} "
        f"(model delta {_percent(row.get('model_minus_baseline'))})"
        for row in rows
    ]


def _confidence_lines(rows: list[dict[str, Any]]) -> list[str]:
    return [
        f"- Confidence >= {row['threshold']:.0%}: count {row['count']}, "
        f"hit {row['hit']}, miss {row['miss']}, hit rate {_percent(row['hit_rate'])}"
        for row in rows
    ]
