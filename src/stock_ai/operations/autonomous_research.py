from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any, Callable

import pandas as pd

from stock_ai.anomaly.rule_based import add_anomaly_scores
from stock_ai.config import PROJECT_ROOT, Settings, load_settings
from stock_ai.evaluation import accuracy_diagnostics
from stock_ai.evaluation.binary_shadow import (
    _evaluate_prediction_rows,
    compare_shadow_with_live,
    summarize_shadow_predictions,
)
from stock_ai.evaluation.binary_up_detection_experiments import (
    BinaryUpCandidate,
    BinaryUpExperimentConfig,
    run_binary_up_detection_experiments,
)
from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.regime.rule_based import add_regime_labels
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.storage.sqlite_store import read_prediction_history
from stock_ai.utils.safe_cast import safe_float, safe_int


SCHEMA_VERSION = "autonomous_research/v1"
EVALUATION_VERSION = "binary_research_windows/v1"
TARGET_DEFINITION = "target_next_close_binary:next_close_return_gt_0/v1"
DEFAULT_CANDIDATES = (
    "production_current",
    "class_weight_balanced",
    "scale_pos_weight_medium",
)
MAX_CANDIDATES = 6
SHADOW_PATH = PROJECT_ROOT / "reports" / "evaluation" / "binary_shadow_predictions.csv"
OPERATIONS_REPORT_DIR = PROJECT_ROOT / "reports" / "autonomous"
TASK_QUEUE_PATH = OPERATIONS_REPORT_DIR / "operations_task_queue.json"
LOCK_PATH = OPERATIONS_REPORT_DIR / ".research.lock"


class ResearchError(RuntimeError):
    pass


class ResearchStateError(ResearchError):
    pass


class ResearchLockError(ResearchError):
    pass


class ResearchTimeoutError(ResearchError):
    pass


@dataclass(frozen=True)
class ResearchPolicy:
    live_count_trigger: int = 5
    shadow_count_trigger: int = 5
    heavy_due_days: int = 7
    balanced_accuracy_delta_trigger: float = 0.05
    up_recall_delta_trigger: float = 0.10
    shadow_disagreement_trigger: int = 3
    min_offline_sample: int = 100
    min_live_like_sample: int = 30
    min_shadow_sample: int = 10
    material_balanced_delta: float = 0.01
    material_up_recall_delta: float = 0.03
    max_down_recall_drop: float = 0.05
    confidence_sample_warning: int = 20


@dataclass
class RuntimeBudget:
    started: float
    max_minutes: float

    @classmethod
    def start(cls, max_minutes: float) -> RuntimeBudget:
        if max_minutes <= 0 or max_minutes > 120:
            raise ValueError("max_runtime_minutes must be in (0, 120].")
        return cls(time.monotonic(), float(max_minutes))

    def check(self, stage: str) -> None:
        if time.monotonic() - self.started > self.max_minutes * 60:
            raise ResearchTimeoutError(f"Research runtime limit reached at {stage}.")


class ResearchFileLock:
    def __init__(self, path: str | Path, *, force: bool = False) -> None:
        self.path = Path(path)
        self.force = force
        self.acquired = False

    def __enter__(self) -> ResearchFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.force and self.path.exists() and not _lock_owner_is_active(self.path):
            self.path.unlink()
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise ResearchLockError(f"Research lock already exists: {self.path}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump({"pid": os.getpid(), "created_at": _now_iso()}, file)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def default_research_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "last_cycle_at": None,
        "last_successful_cycle_at": None,
        "last_heavy_experiment_at": None,
        "last_operations_report": None,
        "last_operations_status": None,
        "last_live_confirmed_count": None,
        "last_shadow_confirmed_count": None,
        "last_live_metrics": {},
        "last_shadow_metrics": {},
        "last_candidate_ranking": [],
        "last_recommendation": None,
        "last_report_path": None,
        "consecutive_failures": 0,
        "next_due_reason": "first_run",
        "experiment_fingerprint": None,
        "last_trigger_reasons": [],
        "last_skip_reason": None,
        "production_guard": {},
        "shadow_direction_disagreement_count": 0,
    }


def load_research_state(path: str | Path) -> tuple[dict[str, Any], bool]:
    state_path = Path(path)
    if not state_path.exists():
        return default_research_state(), True
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchStateError(
            f"Research state is corrupt or unreadable and was not deleted: {state_path}"
        ) from error
    if not isinstance(loaded, dict):
        raise ResearchStateError(f"Research state must be a JSON object: {state_path}")
    state = default_research_state()
    state.update(loaded)
    return state, False


def build_evaluation_windows(
    predictions: pd.DataFrame,
    *,
    since_date: str | None = None,
) -> list[dict[str, Any]]:
    dates = pd.to_datetime(predictions.get("date", pd.Series(dtype=object)), errors="coerce")
    dates = dates.dropna()
    if since_date:
        parsed_since = pd.to_datetime(since_date, errors="coerce")
        if pd.isna(parsed_since):
            raise ValueError(f"Invalid --since-date: {since_date}")
        dates = dates[dates >= parsed_since]
    if dates.empty:
        return []
    start = pd.Timestamp(dates.min()).normalize()
    end = pd.Timestamp(dates.max()).normalize()
    definitions: list[tuple[str, pd.Timestamp, pd.Timestamp, str]] = [
        ("historical_all", start, end, "offline"),
        ("since_2023", max(start, pd.Timestamp("2023-01-01")), end, "offline"),
        ("since_2024", max(start, pd.Timestamp("2024-01-01")), end, "offline"),
        ("since_2025", max(start, pd.Timestamp("2025-01-01")), end, "offline"),
        ("since_2026", max(start, pd.Timestamp("2026-01-01")), end, "offline"),
        ("recent_6_months", max(start, end - pd.DateOffset(months=6)), end, "offline"),
        ("recent_3_months", max(start, end - pd.DateOffset(months=3)), end, "offline"),
    ]
    unique_dates = sorted(pd.Timestamp(value).normalize() for value in dates.unique())
    holdout_size = max(30, math.ceil(len(unique_dates) * 0.2))
    holdout_start = unique_dates[max(0, len(unique_dates) - holdout_size)]
    definitions.append(("live_like_holdout", holdout_start, end, "offline"))
    return [
        {
            "name": name,
            "start_date": left.strftime("%Y-%m-%d"),
            "end_date": right.strftime("%Y-%m-%d"),
            "source": source,
        }
        for name, left, right, source in definitions
    ]


def compute_experiment_fingerprint(
    *,
    ticker: str,
    latest_feature_date: str | None,
    live_confirmed_count: int,
    shadow_confirmed_count: int,
    candidates: list[str],
    windows: list[dict[str, Any]],
    model_config_hash: str,
    feature_config_hash: str,
) -> str:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "evaluation_version": EVALUATION_VERSION,
        "ticker": ticker,
        "latest_feature_date": latest_feature_date,
        "live_confirmed_count": int(live_confirmed_count),
        "shadow_confirmed_count": int(shadow_confirmed_count),
        "candidates": sorted(candidates),
        "windows": windows,
        "model_config_hash": model_config_hash,
        "feature_config_hash": feature_config_hash,
        "target_definition": TARGET_DEFINITION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def determine_trigger_reasons(
    state: dict[str, Any],
    *,
    first_run: bool,
    force: bool,
    live_confirmed_count: int,
    shadow_confirmed_count: int,
    live_metrics: dict[str, Any],
    shadow_disagreement_count: int,
    now: pd.Timestamp,
    policy: ResearchPolicy,
) -> list[str]:
    reasons: list[str] = []
    if first_run:
        reasons.append("first_run")
    if force:
        reasons.append("force")
    old_live = safe_int(state.get("last_live_confirmed_count"))
    if old_live is not None and live_confirmed_count - old_live >= policy.live_count_trigger:
        reasons.append("live_confirmed_increased")
    old_shadow = safe_int(state.get("last_shadow_confirmed_count"))
    if old_shadow is not None and shadow_confirmed_count - old_shadow >= policy.shadow_count_trigger:
        reasons.append("shadow_confirmed_increased")
    last_heavy = pd.to_datetime(state.get("last_heavy_experiment_at"), errors="coerce")
    if not pd.isna(last_heavy) and now - last_heavy >= pd.Timedelta(days=policy.heavy_due_days):
        reasons.append("heavy_research_due")
    previous = state.get("last_live_metrics") or {}
    for metric, threshold, reason in (
        ("balanced_accuracy", policy.balanced_accuracy_delta_trigger, "live_balanced_accuracy_changed"),
        ("up_recall", policy.up_recall_delta_trigger, "live_up_recall_changed"),
    ):
        current_value = safe_float(live_metrics.get(metric))
        old_value = safe_float(previous.get(metric))
        if current_value is not None and old_value is not None and abs(current_value - old_value) >= threshold:
            reasons.append(reason)
    old_disagreement = safe_int(state.get("shadow_direction_disagreement_count")) or 0
    if shadow_disagreement_count - old_disagreement >= policy.shadow_disagreement_trigger:
        reasons.append("shadow_direction_disagreement_increased")
    return list(dict.fromkeys(reasons))


def evaluate_candidate_windows(
    predictions: pd.DataFrame,
    windows: list[dict[str, Any]],
    *,
    production_name: str = "production_current",
    min_sample: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metrics: list[dict[str, Any]] = []
    baselines: list[dict[str, Any]] = []
    confidence: list[dict[str, Any]] = []
    for candidate, candidate_rows in predictions.groupby("candidate_name", dropna=False):
        for window in windows:
            rows = _window_rows(candidate_rows, window)
            row = _metric_row(rows, str(candidate), window["name"], min_sample=min_sample)
            metrics.append(row)
            baseline_rows = _baseline_rows(rows, str(candidate), window["name"])
            baselines.extend(baseline_rows)
            confidence.extend(_confidence_rows(rows, str(candidate), window["name"]))

    index = {(row["candidate_name"], row["window"]): row for row in metrics}
    for row in metrics:
        production = index.get((production_name, row["window"]))
        row["delta_vs_production_balanced_accuracy"] = _delta(
            row.get("balanced_accuracy"),
            None if production is None else production.get("balanced_accuracy"),
        )
        row["delta_vs_production_up_recall"] = _delta(
            row.get("up_recall"),
            None if production is None else production.get("up_recall"),
        )
        row["delta_vs_production_down_recall"] = _delta(
            row.get("down_recall"),
            None if production is None else production.get("down_recall"),
        )
        relevant = [
            baseline
            for baseline in baselines
            if baseline["candidate_name"] == row["candidate_name"]
            and baseline["window"] == row["window"]
        ]
        available = [safe_float(item.get("candidate_minus_baseline_balanced_accuracy")) for item in relevant]
        available = [value for value in available if value is not None]
        row["delta_vs_best_baseline_balanced_accuracy"] = min(available) if available else None
    return metrics, baselines, confidence


def rank_candidates(
    metrics: list[dict[str, Any]],
    *,
    production_name: str,
    shadow_summary: dict[str, Any],
    policy: ResearchPolicy,
) -> list[dict[str, Any]]:
    by_candidate: dict[str, dict[str, dict[str, Any]]] = {}
    for row in metrics:
        by_candidate.setdefault(str(row["candidate_name"]), {})[str(row["window"])] = row
    production = by_candidate.get(production_name, {})
    weights = {
        "historical_all": 0.35,
        "since_2023": 0.15,
        "since_2024": 0.15,
        "since_2025": 0.12,
        "since_2026": 0.08,
        "recent_6_months": 0.08,
        "live_like_holdout": 0.07,
    }
    ranked: list[dict[str, Any]] = []
    for candidate, windows in by_candidate.items():
        components = []
        improved_windows = 0
        baseline_wins = 0
        for name, weight in weights.items():
            row = windows.get(name)
            reference = production.get(name)
            if not row or row.get("sample_too_small") or safe_float(row.get("balanced_accuracy")) is None:
                continue
            balanced = float(row["balanced_accuracy"])
            up = safe_float(row.get("up_recall")) or 0.0
            down = safe_float(row.get("down_recall")) or 0.0
            baseline_delta = safe_float(row.get("delta_vs_best_baseline_balanced_accuracy")) or -0.25
            production_delta = _delta(balanced, None if reference is None else reference.get("balanced_accuracy")) or 0.0
            component = 0.45 * balanced + 0.20 * up + 0.15 * down + 0.10 * max(-0.25, baseline_delta) + 0.10 * production_delta
            components.append((weight, component))
            if production_delta > policy.material_balanced_delta:
                improved_windows += 1
            if baseline_delta > 0:
                baseline_wins += 1
        score = (
            sum(weight * value for weight, value in components) / sum(weight for weight, _ in components)
            if components
            else None
        )
        historical = windows.get("historical_all", {})
        live_like = windows.get("live_like_holdout", {})
        decision = _candidate_decision(
            candidate,
            historical=historical,
            live_like=live_like,
            production_historical=production.get("historical_all", {}),
            production_live_like=production.get("live_like_holdout", {}),
            improved_windows=improved_windows,
            baseline_wins=baseline_wins,
            shadow_confirmed=safe_int(shadow_summary.get("confirmed_count")) or 0,
            policy=policy,
        )
        ranked.append(
            {
                "candidate_name": candidate,
                "score": None if score is None else round(score, 4),
                "evaluated_weight": round(sum(weight for weight, _ in components), 3),
                "improved_window_count": improved_windows,
                "baseline_win_window_count": baseline_wins,
                "decision": decision,
                "historical_balanced_accuracy": historical.get("balanced_accuracy"),
                "historical_up_recall": historical.get("up_recall"),
                "historical_down_recall": historical.get("down_recall"),
                "live_like_sample_count": live_like.get("sample_count", 0),
                "live_like_balanced_accuracy": live_like.get("balanced_accuracy"),
            }
        )
    ranked.sort(key=lambda row: (-1 if row["score"] is None else -float(row["score"]), row["candidate_name"]))
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    return ranked


def run_autonomous_research(
    *,
    ticker: str,
    state_path: str | Path,
    history_path: str | Path,
    report_dir: str | Path,
    max_runtime_minutes: float = 120.0,
    since_date: str | None = None,
    candidate_names: list[str] | None = None,
    skip_shadow: bool = False,
    skip_offline: bool = False,
    dry_run: bool = False,
    force: bool = False,
    task_queue_path: str | Path = TASK_QUEUE_PATH,
    settings: Settings | None = None,
    experiment_runner: Callable[..., Any] = run_binary_up_detection_experiments,
    now: datetime | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    budget = RuntimeBudget.start(max_runtime_minutes)
    started = pd.Timestamp(now or datetime.now().astimezone())
    state_file = _resolve(state_path)
    history_file = _resolve(history_path)
    report_folder = _resolve(report_dir)
    queue_file = _resolve(task_queue_path)
    candidates, unavailable = _resolve_candidates(candidate_names)
    if len(candidates) > MAX_CANDIDATES:
        raise ResearchError(f"Candidate limit exceeded: {len(candidates)} > {MAX_CANDIDATES}")

    lock_held_by_wrapper = os.environ.get("STOCK_AI_RESEARCH_WRAPPER_LOCK_HELD") == "1"
    lock_context = _NullLock() if lock_held_by_wrapper else ResearchFileLock(LOCK_PATH, force=force)
    with lock_context:
        budget.check("initialization")
        state, first_run = load_research_state(state_file)
        operations = load_latest_operations_report(report_folder)
        operations_gate = assess_operations_health(operations)
        production_before = _production_guard(settings)
        database_before = _database_snapshot(settings.paths.database_path)

        feature_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
        dataset = read_parquet(feature_path)
        if "target_next_close_binary" not in dataset.columns:
            dataset = add_next_close_binary_target(dataset)
        context = add_anomaly_scores(add_regime_labels(dataset))
        target_validation = validate_target_definition(context)
        if not target_validation["valid"]:
            operations_gate["healthy"] = False
            operations_gate["reasons"].append("target_definition_inconsistent")

        live_raw = read_prediction_history(
            settings.paths.database_path,
            ticker=ticker,
            target="next_close_binary",
            model_name="lightgbm_binary",
            prediction_type="live",
        )
        live_frame = accuracy_diagnostics.build_binary_diagnostic_frame(
            live_raw,
            context,
            prediction_type="live",
        )
        live_metrics = _metric_row(
            live_frame[live_frame.get("status", pd.Series(dtype=str)).eq("confirmed")],
            "production_live",
            "live_confirmed",
            min_sample=policy_or_default(None).min_live_like_sample,
        )
        shadow = summarize_shadow_read_only(
            context,
            live_raw,
            skip=skip_shadow,
        )
        windows = build_evaluation_windows(context, since_date=since_date)
        evaluation_periods = describe_evaluation_periods(
            context,
            windows,
            live_frame=live_frame,
            shadow=shadow,
            policy=ResearchPolicy(),
        )
        model_hash = _file_sha256(settings.paths.model_dir / f"lightgbm_binary_{ticker}.joblib")
        feature_config_hash = _combined_hash(
            [PROJECT_ROOT / "configs" / "settings.yml", feature_path]
        )
        fingerprint = compute_experiment_fingerprint(
            ticker=ticker,
            latest_feature_date=_latest_date(context.get("date")),
            live_confirmed_count=int(live_metrics.get("sample_count") or 0),
            shadow_confirmed_count=int(shadow.get("confirmed_count") or 0),
            candidates=[candidate.name for candidate in candidates],
            windows=windows,
            model_config_hash=model_hash,
            feature_config_hash=feature_config_hash,
        )
        policy = ResearchPolicy()
        trigger_reasons = determine_trigger_reasons(
            state,
            first_run=first_run,
            force=force,
            live_confirmed_count=int(live_metrics.get("sample_count") or 0),
            shadow_confirmed_count=int(shadow.get("confirmed_count") or 0),
            live_metrics=live_metrics,
            shadow_disagreement_count=int(shadow.get("direction_disagreement_count") or 0),
            now=started,
            policy=policy,
        )
        guard_changed = bool(state.get("production_guard")) and state.get("production_guard") != production_before
        if guard_changed:
            operations_gate["healthy"] = False
            operations_gate["reasons"].append("unexpected_production_guard_change")
        if int(state.get("consecutive_failures") or 0) >= 3:
            operations_gate["healthy"] = False
            operations_gate["reasons"].append("three_consecutive_research_failures")

        duplicate = state.get("experiment_fingerprint") == fingerprint and not force
        executed = False
        skip_reason = None
        classification = "research_completed_no_material_change"
        metrics: list[dict[str, Any]] = []
        baselines: list[dict[str, Any]] = []
        confidence: list[dict[str, Any]] = []
        ranking: list[dict[str, Any]] = []
        experiment_notes: list[str] = []
        if not operations_gate["healthy"]:
            classification = "research_skipped_operations_unhealthy"
            skip_reason = "; ".join(operations_gate["reasons"])
        elif duplicate:
            classification = "research_skipped_duplicate_experiment"
            skip_reason = "experiment fingerprint matches the previous heavy Research"
        elif not trigger_reasons:
            classification = "research_skipped_not_due"
            skip_reason = "no Research trigger condition was met"
        elif skip_offline:
            classification = "research_completed_no_material_change"
            skip_reason = "offline evaluation explicitly skipped"
            experiment_notes.append("Force/CLI orchestration verified without running heavy offline evaluation.")
        else:
            budget.check("before offline evaluation")
            result = experiment_runner(
                context,
                config=BinaryUpExperimentConfig(
                    ticker=ticker,
                    start_date=since_date,
                    min_train_rows=252,
                ),
                candidates=candidates,
                live_frame=live_frame,
            )
            budget.check("after offline evaluation")
            executed = True
            metrics, baselines, confidence = evaluate_candidate_windows(
                result.predictions,
                windows,
                min_sample=policy.min_offline_sample,
            )
            ranking = rank_candidates(
                metrics,
                production_name="production_current",
                shadow_summary=shadow,
                policy=policy,
            )
            classification = _research_classification(ranking)

        recommendation = _recommendation(classification, ranking)
        completed = pd.Timestamp(datetime.now().astimezone())
        report_paths = _report_paths(report_folder, completed)
        findings = (
            _research_findings(
                ticker=ticker,
                fingerprint=fingerprint,
                ranking=ranking,
                live_metrics=live_metrics,
                shadow=shadow,
                report_path=str(report_paths["markdown"]),
                existing_live_model_review=_has_active_live_model_review(
                    queue_file,
                    ticker,
                ),
            )
            if executed
            else []
        )
        queue, task_changes = update_research_tasks(
            queue_file,
            findings,
            cycle_at=completed.isoformat(),
            source_report=str(report_paths["markdown"]),
            dry_run=dry_run,
            resolve_inactive=executed,
        )
        production_after = _production_guard(settings)
        database_after = _database_snapshot(settings.paths.database_path)
        invariants = _invariant_summary(
            production_before,
            production_after,
            database_before,
            database_after,
        )
        if not all(invariants.values()):
            classification = "research_failed"
            recommendation = "production_change_not_ready"
            skip_reason = "forbidden production invariant changed during Research"

        result_payload = {
            "schema_version": SCHEMA_VERSION,
            "cycle_started_at": started.isoformat(),
            "cycle_completed_at": completed.isoformat(),
            "cycle_id": _cycle_id(ticker, started, fingerprint, classification),
            "ticker": ticker,
            "dry_run": dry_run,
            "force": force,
            "trigger_reasons": trigger_reasons,
            "experiment_fingerprint": fingerprint,
            "data_fingerprint": feature_config_hash,
            "operations_health": operations_gate,
            "operations_report": operations.get("report_paths", {}).get("json"),
            "experiment_executed": executed,
            "classification": classification,
            "skip_reason": skip_reason,
            "candidates": [candidate.name for candidate in candidates],
            "unavailable_candidates": unavailable,
            "evaluation_periods": evaluation_periods,
            "metrics": metrics,
            "baselines": baselines,
            "confidence_analysis": confidence,
            "live_metrics": live_metrics,
            "shadow_metrics": shadow,
            "candidate_ranking": ranking,
            "ranking_formula": "Weighted multi-window score: 45% balanced accuracy, 20% up recall, 15% down recall, 10% worst baseline edge, 10% production edge; recent 3 months is report-only.",
            "recommendation": recommendation,
            "previous_research_delta": _previous_delta(state, live_metrics, shadow),
            "approval_required": bool(findings),
            "tasks_created": task_changes["created"],
            "tasks_updated": task_changes["updated"],
            "tasks_resolved": task_changes["resolved"],
            "task_queue_preview": queue if dry_run else None,
            "experiment_notes": experiment_notes,
            "target_validation": target_validation,
            "production_change_applied": False,
            "invariants": invariants,
            "production_changes": {
                "Production model changed": "No",
                "Production artifact changed": "No",
                "Production config changed": "No",
                "Production SQLite changed": "No",
                "Existing predictions changed": "No",
                "Live/backfill prediction created": "No",
                "daily_predict changed": "No",
                "Task Scheduler changed": "No",
                "Autonomous Operations changed": "No",
                "Automatic merge performed": "No",
            },
            "next_due_condition": _next_due_reason(classification),
            "stop_reason": classification,
            "files_modified": [],
            "report_paths": {key: str(value) for key, value in report_paths.items()},
        }
        result_payload["exit_code"] = _exit_code(result_payload)
        report_paths["json"].parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_paths["json"], result_payload)
        report_paths["markdown"].write_text(render_research_markdown(result_payload), encoding="utf-8")
        result_payload["files_modified"].extend(
            [str(report_paths["markdown"]), str(report_paths["json"])]
        )

        if not dry_run:
            updated_state = _updated_state(
                state,
                result_payload,
                production_guard=production_after,
            )
            _write_json(state_file, updated_state)
            append_research_history(history_file, result_payload)
            result_payload["files_modified"].extend([str(state_file), str(history_file)])
            if task_changes["changed"]:
                result_payload["files_modified"].append(str(queue_file))
            _write_json(report_paths["json"], result_payload)
            report_paths["markdown"].write_text(
                render_research_markdown(result_payload), encoding="utf-8"
            )
        return result_payload


def load_latest_operations_report(report_dir: str | Path) -> dict[str, Any]:
    folder = _resolve(report_dir)
    files = sorted(
        (
            path
            for path in folder.glob("operations_*.json")
            if re.fullmatch(r"operations_\d{4}-\d{2}-\d{2}_\d{4}(?:_\d+)?", path.stem)
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return {"stop_reason": "operations_report_missing", "report_paths": {}}
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"stop_reason": "operations_report_unreadable", "report_paths": {"json": str(files[0])}}
    payload.setdefault("report_paths", {})["json"] = str(files[0])
    return payload


def assess_operations_health(report: dict[str, Any]) -> dict[str, Any]:
    health = report.get("health_summary") or {}
    freshness = report.get("data_freshness") or {}
    database = report.get("database_validation") or {}
    reasons = []
    if report.get("stop_reason") != "completed_read_only_cycle":
        reasons.append("latest_operations_cycle_not_completed")
    if int(health.get("action_required_count") or 0) > 0:
        reasons.append("operations_action_required")
    if int(health.get("unknown_source_count") or 0) > 0 or report.get("unknown_source_findings"):
        reasons.append("unknown_news_source")
    if report.get("missing_live_findings"):
        reasons.append("unresolved_post_start_missing_live")
    if freshness.get("freshness_status") != "OK" or bool(freshness.get("stale_at_run_time")):
        reasons.append("stale_data")
    if not database.get("readable"):
        reasons.append("production_database_unreadable")
    return {
        "healthy": not reasons,
        "reasons": reasons,
        "stop_reason": report.get("stop_reason"),
        "action_required_count": int(health.get("action_required_count") or 0),
        "unknown_source_count": int(health.get("unknown_source_count") or 0),
        "unresolved_missing_live_count": len(report.get("missing_live_findings") or []),
        "freshness_status": freshness.get("freshness_status"),
        "stale_at_run_time": bool(freshness.get("stale_at_run_time")),
        "database_readable": bool(database.get("readable")),
    }


def describe_evaluation_periods(
    feature_data: pd.DataFrame,
    windows: list[dict[str, Any]],
    *,
    live_frame: pd.DataFrame,
    shadow: dict[str, Any],
    policy: ResearchPolicy,
) -> list[dict[str, Any]]:
    periods = []
    for window in windows:
        rows = _window_rows(feature_data, window)
        target = pd.to_numeric(rows.get("target_next_close_binary"), errors="coerce")
        target = target.dropna().astype(int)
        periods.append(
            {
                **window,
                "sample_count": int(len(target)),
                "positive_count": int((target == 1).sum()),
                "negative_count": int((target == 0).sum()),
                "data_available": bool(len(target)),
                "sample_too_small": int(len(target)) < policy.min_offline_sample,
            }
        )
    live = live_frame[live_frame.get("status", pd.Series(dtype=str)).eq("confirmed")].copy()
    live_actual = live.get("actual_class", pd.Series(dtype=object)).map(safe_int).dropna().astype(int)
    live_dates = pd.to_datetime(live.get("date"), errors="coerce").dropna()
    periods.append(
        {
            "name": "live_confirmed",
            "start_date": pd.Timestamp(live_dates.min()).strftime("%Y-%m-%d") if not live_dates.empty else None,
            "end_date": pd.Timestamp(live_dates.max()).strftime("%Y-%m-%d") if not live_dates.empty else None,
            "source": "live",
            "sample_count": int(len(live_actual)),
            "positive_count": int((live_actual == 1).sum()),
            "negative_count": int((live_actual == 0).sum()),
            "data_available": bool(len(live_actual)),
            "sample_too_small": int(len(live_actual)) < policy.min_live_like_sample,
        }
    )
    periods.append(
        {
            "name": "shadow_confirmed",
            "start_date": shadow.get("confirmed_start_date"),
            "end_date": shadow.get("confirmed_end_date"),
            "source": "shadow",
            "sample_count": int(shadow.get("confirmed_count") or 0),
            "positive_count": shadow.get("actual_up_count"),
            "negative_count": shadow.get("actual_down_count"),
            "data_available": bool(shadow.get("confirmed_count")),
            "sample_too_small": int(shadow.get("confirmed_count") or 0) < policy.min_shadow_sample,
        }
    )
    return periods


def summarize_shadow_read_only(
    feature_data: pd.DataFrame,
    live_predictions: pd.DataFrame,
    *,
    skip: bool = False,
    shadow_path: str | Path = SHADOW_PATH,
) -> dict[str, Any]:
    if skip:
        return _empty_shadow("skipped")
    path = Path(shadow_path)
    if not path.exists():
        return _empty_shadow("missing")
    try:
        rows = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return _empty_shadow("unreadable")
    for column in ("actual_direction", "result", "target_close_date", "evaluated_at"):
        if column not in rows.columns:
            rows[column] = ""
        rows[column] = rows[column].fillna("").astype(object)
    rows = _evaluate_prediction_rows(rows, feature_data)
    summary = summarize_shadow_predictions(rows)
    comparison = compare_shadow_with_live(rows, live_predictions)
    result_text = rows.get("result", pd.Series(dtype=str)).astype(str)
    confirmed = rows[result_text.isin(["hit", "miss"])].copy()
    confirmed_dates = pd.to_datetime(confirmed.get("feature_date"), errors="coerce").dropna()
    actual_direction = confirmed.get("actual_direction", pd.Series(dtype=str)).astype(str)
    pairs = _shadow_pair_counts(rows)
    return {
        "status": "available",
        "path": str(path),
        "total_rows": int(len(rows)),
        "confirmed_count": int(result_text.isin(["hit", "miss"]).sum()),
        "pending_count": int(result_text.eq("pending").sum()),
        "confirmed_start_date": pd.Timestamp(confirmed_dates.min()).strftime("%Y-%m-%d") if not confirmed_dates.empty else None,
        "confirmed_end_date": pd.Timestamp(confirmed_dates.max()).strftime("%Y-%m-%d") if not confirmed_dates.empty else None,
        "actual_up_count": int(actual_direction.eq("up").sum()),
        "actual_down_count": int(actual_direction.eq("down").sum()),
        "candidate_summary": _records(summary),
        "direction_agreement_count": int(comparison.get("direction_matches_live", pd.Series(dtype=bool)).eq(True).sum()),
        "direction_disagreement_count": int(comparison.get("direction_matches_live", pd.Series(dtype=bool)).eq(False).sum()),
        **pairs,
    }


def validate_target_definition(frame: pd.DataFrame) -> dict[str, Any]:
    if not {"target_next_close_binary", "next_close_return"}.issubset(frame.columns):
        return {"valid": False, "reason": "required target columns missing", "mismatch_count": None}
    actual = pd.to_numeric(frame["next_close_return"], errors="coerce")
    target = pd.to_numeric(frame["target_next_close_binary"], errors="coerce")
    valid = actual.notna() & target.notna()
    expected = (actual[valid] > 0).astype(int)
    mismatch = int((expected != target[valid].astype(int)).sum())
    return {
        "valid": mismatch == 0,
        "reason": "target matches next-close return sign" if mismatch == 0 else "target mismatch detected",
        "checked_rows": int(valid.sum()),
        "mismatch_count": mismatch,
        "target_definition": TARGET_DEFINITION,
    }


def update_research_tasks(
    path: str | Path,
    findings: list[dict[str, Any]],
    *,
    cycle_at: str,
    source_report: str,
    dry_run: bool,
    resolve_inactive: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_path = Path(path)
    try:
        queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.exists() else {"tasks": []}
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchStateError(f"Task queue is unreadable: {queue_path}") from error
    tasks = [dict(task) for task in queue.get("tasks", []) if isinstance(task, dict)]
    by_fingerprint = {str(task.get("fingerprint")): task for task in tasks if task.get("fingerprint")}
    changes = {"created": [], "updated": [], "resolved": [], "changed": False}
    active = set()
    for finding in findings:
        fingerprint = str(finding["fingerprint"])
        active.add(fingerprint)
        task = by_fingerprint.get(fingerprint)
        if task is None:
            task = {
                "task_id": f"research_{hashlib.sha256(fingerprint.encode()).hexdigest()[:16]}",
                "fingerprint": fingerprint,
                "created_at": cycle_at,
            }
            tasks.append(task)
            by_fingerprint[fingerprint] = task
            changes["created"].append(fingerprint)
        else:
            changes["updated"].append(fingerprint)
        task.update(
            {
                "last_seen_at": cycle_at,
                "category": "model_research",
                "risk_level": "Level 2",
                "priority": finding.get("priority", "medium"),
                "title": finding["title"],
                "description": finding["description"],
                "candidate": finding.get("candidate"),
                "evidence": finding.get("evidence", {}),
                "production_difference": finding.get("production_difference", {}),
                "baseline_difference": finding.get("baseline_difference", {}),
                "risks": finding.get("risks", []),
                "proposed_action": finding["proposed_action"],
                "status": "approval_required",
                "requires_human_approval": True,
                "blocked_reason": finding.get("blocked_reason"),
                "source_report": source_report,
            }
        )
    for task in tasks:
        fingerprint = str(task.get("fingerprint") or "")
        if not resolve_inactive or task.get("category") != "model_research" or fingerprint in active:
            continue
        if task.get("status") == "approval_required" and fingerprint.startswith(
            ("research_candidate:", "model_degradation:")
        ):
            task["status"] = "monitoring"
            task["last_seen_at"] = cycle_at
            task["resolution_reason"] = "not active in the latest Research result"
            changes["resolved"].append(fingerprint)
    changes["changed"] = bool(changes["created"] or changes["updated"] or changes["resolved"])
    result = {
        "schema_version": queue.get("schema_version", "autonomous_operations/v1"),
        "updated_at": cycle_at,
        "tasks": tasks,
    }
    if not dry_run and changes["changed"]:
        _write_json(queue_path, result)
    return result, changes


def append_research_history(path: str | Path, result: dict[str, Any]) -> bool:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    cycle_id = str(result["cycle_id"])
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("cycle_id") == cycle_id:
                    return False
            except json.JSONDecodeError:
                continue
    history_row = {
        "schema_version": SCHEMA_VERSION,
        "cycle_id": cycle_id,
        "timestamp": result["cycle_completed_at"],
        "ticker": result["ticker"],
        "trigger_reasons": result["trigger_reasons"],
        "skipped": not result["experiment_executed"],
        "executed": result["experiment_executed"],
        "operations_status": result["operations_health"],
        "data_fingerprint": result["data_fingerprint"],
        "experiment_fingerprint": result["experiment_fingerprint"],
        "candidates": result["candidates"],
        "evaluation_periods": result["evaluation_periods"],
        "metrics": result["metrics"],
        "baselines": result["baselines"],
        "ranking": result["candidate_ranking"],
        "recommendation": result["recommendation"],
        "production_change_applied": False,
        "report_path": result["report_paths"]["markdown"],
        "stop_reason": result["stop_reason"],
    }
    with history_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_json_ready(history_row), ensure_ascii=False) + "\n")
    return True


def render_research_markdown(result: dict[str, Any]) -> str:
    ranking = pd.DataFrame(result.get("candidate_ranking") or [])
    metrics = pd.DataFrame(result.get("metrics") or [])
    historical = metrics[metrics.get("window", pd.Series(dtype=str)).eq("historical_all")] if not metrics.empty else metrics
    lines = [
        "# Autonomous Research v1",
        "",
        f"- Cycle: {result['cycle_started_at']} -> {result['cycle_completed_at']}",
        f"- Ticker: {result['ticker']}",
        f"- Classification: {result['classification']}",
        f"- Trigger reasons: {_join(result.get('trigger_reasons'))}",
        f"- Experiment fingerprint: `{result['experiment_fingerprint']}`",
        f"- Experiment executed: {'Yes' if result['experiment_executed'] else 'No'}",
        f"- Skip reason: {result.get('skip_reason') or '-'}",
        f"- Recommendation: {result['recommendation']}",
        "",
        "## Operations health",
        f"- Healthy: {result['operations_health'].get('healthy')}",
        f"- Reasons: {_join(result['operations_health'].get('reasons'))}",
        f"- Action required: {result['operations_health'].get('action_required_count')}",
        f"- Unknown sources: {result['operations_health'].get('unknown_source_count')}",
        f"- Freshness: {result['operations_health'].get('freshness_status')}",
        "",
        "## Candidates",
        f"- Evaluated: {_join(result.get('candidates'))}",
        f"- Unavailable: {_join(result.get('unavailable_candidates'))}",
        "",
        "## Evaluation periods",
        _markdown_table(pd.DataFrame(result.get("evaluation_periods") or [])),
        "",
        "## Historical candidate comparison",
        _markdown_table(historical),
        "",
        "## Candidate ranking",
        _markdown_table(ranking),
        "",
        "## Live metrics",
        "```json",
        json.dumps(_json_ready(result.get("live_metrics")), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Shadow metrics",
        "```json",
        json.dumps(_json_ready(result.get("shadow_metrics")), indent=2, ensure_ascii=False),
        "```",
        "",
        "## Ranking formula",
        result["ranking_formula"],
        "",
        "## Task queue",
        f"- Approval required: {result.get('approval_required')}",
        f"- Created: {_join(result.get('tasks_created'))}",
        f"- Updated: {_join(result.get('tasks_updated'))}",
        f"- Resolved/monitoring: {_join(result.get('tasks_resolved'))}",
        "",
        "## Safety invariants",
        *[f"- {key}: {'OK' if value else 'FAILED'}" for key, value in result.get("invariants", {}).items()],
        "",
        "## Explicit non-changes",
        *[f"- {key}: {value}" for key, value in result.get("production_changes", {}).items()],
        "",
        "## Next due",
        f"- {result.get('next_due_condition')}",
        f"- Stop reason: {result.get('stop_reason')}",
        "",
        "## Files modified",
        *[f"- `{path}`" for path in result.get("files_modified", [])],
    ]
    return "\n".join(lines) + "\n"


def _resolve_candidates(names: list[str] | None) -> tuple[list[BinaryUpCandidate], list[str]]:
    requested = names or list(DEFAULT_CANDIDATES)
    if len(dict.fromkeys(requested)) > MAX_CANDIDATES:
        raise ResearchError(
            f"Candidate limit exceeded: {len(dict.fromkeys(requested))} > {MAX_CANDIDATES}"
        )
    aliases = {"current_model": "production_current"}
    supported = {
        "production_current": BinaryUpCandidate(
            name="production_current",
            note="Production-style binary LightGBM settings; offline reference only.",
        ),
        "class_weight_balanced": BinaryUpCandidate(
            name="class_weight_balanced",
            weight_mode="class_weight_balanced",
            note="Balanced class weights selected inside each train fold.",
        ),
        "scale_pos_weight_medium": BinaryUpCandidate(
            name="scale_pos_weight_medium",
            weight_mode="scale_pos_weight_medium",
            note="Medium train-fold positive-class weighting; offline only.",
        ),
    }
    candidates = []
    unavailable = []
    for raw_name in dict.fromkeys(requested):
        name = aliases.get(raw_name, raw_name)
        if name in supported:
            candidates.append(supported[name])
        else:
            unavailable.append(str(raw_name))
    if not candidates:
        raise ResearchError("No supported Research candidates were selected.")
    return candidates, unavailable


def _metric_row(
    frame: pd.DataFrame,
    candidate: str,
    window: str,
    *,
    min_sample: int,
) -> dict[str, Any]:
    rows = frame.copy()
    if "status" in rows:
        rows = rows[rows["status"].astype(str).eq("confirmed")]
    actual = rows.get("actual_class", pd.Series(dtype=object)).map(safe_int)
    predicted = rows.get("predicted_class", pd.Series(dtype=object)).map(safe_int)
    valid = actual.notna() & predicted.notna()
    rows = rows.loc[valid].copy()
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    up_recall = _divide(tp, tp + fn)
    down_recall = _divide(tn, tn + fp)
    probability = pd.to_numeric(
        rows.get("probability_up", pd.Series(index=rows.index, dtype=float)),
        errors="coerce",
    )
    auc = _roc_auc(actual, probability)
    brier = float(((probability - actual) ** 2).mean()) if probability.notna().all() and len(probability) else None
    dates = pd.to_datetime(rows.get("date"), errors="coerce").dropna()
    return {
        "candidate_name": candidate,
        "window": window,
        "start_date": pd.Timestamp(dates.min()).strftime("%Y-%m-%d") if not dates.empty else None,
        "end_date": pd.Timestamp(dates.max()).strftime("%Y-%m-%d") if not dates.empty else None,
        "sample_count": int(len(rows)),
        "positive_count": int((actual == 1).sum()),
        "negative_count": int((actual == 0).sum()),
        "data_available": bool(len(rows)),
        "sample_too_small": int(len(rows)) < min_sample,
        "accuracy": _divide(tp + tn, len(rows)),
        "balanced_accuracy": (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None,
        "up_recall": up_recall,
        "down_recall": down_recall,
        "up_precision": _divide(tp, tp + fp),
        "down_precision": _divide(tn, tn + fn),
        "f1_up": _f1(tp, fp, fn),
        "f1_down": _f1(tn, fn, fp),
        "roc_auc": auc,
        "roc_auc_status": "available" if auc is not None else "not_available",
        "brier_score": brier,
        "predicted_up_count": int((predicted == 1).sum()),
        "predicted_down_count": int((predicted == 0).sum()),
        "actual_up_count": int((actual == 1).sum()),
        "actual_down_count": int((actual == 0).sum()),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def _baseline_rows(frame: pd.DataFrame, candidate: str, window: str) -> list[dict[str, Any]]:
    rows = frame.copy()
    actual = rows.get("actual_class", pd.Series(dtype=object)).map(safe_int)
    model = rows.get("predicted_class", pd.Series(dtype=object)).map(safe_int)
    majority = int(actual.dropna().mean() >= 0.5) if actual.notna().any() else 0
    definitions = {
        "always_up": pd.Series(1, index=rows.index),
        "always_down": pd.Series(0, index=rows.index),
        "majority_class": pd.Series(majority, index=rows.index),
        "previous_day_direction": rows.get("baseline_previous_day_direction", pd.Series(index=rows.index)),
        "moving_average_trend_only": rows.get("baseline_moving_average_trend", pd.Series(index=rows.index)),
    }
    model_metrics = _simple_balanced(actual, model)
    output = []
    for name, prediction in definitions.items():
        baseline = _simple_balanced(actual, pd.Series(prediction, index=rows.index).map(safe_int))
        output.append(
            {
                "candidate_name": candidate,
                "window": window,
                "baseline": name,
                "sample_count": baseline["sample_count"],
                "baseline_accuracy": baseline["accuracy"],
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "candidate_accuracy": model_metrics["accuracy"],
                "candidate_balanced_accuracy": model_metrics["balanced_accuracy"],
                "candidate_minus_baseline_accuracy": _delta(model_metrics["accuracy"], baseline["accuracy"]),
                "candidate_minus_baseline_balanced_accuracy": _delta(model_metrics["balanced_accuracy"], baseline["balanced_accuracy"]),
            }
        )
    return output


def _confidence_rows(frame: pd.DataFrame, candidate: str, window: str) -> list[dict[str, Any]]:
    confidence = pd.to_numeric(
        frame.get("confidence", pd.Series(index=frame.index, dtype=float)),
        errors="coerce",
    )
    result = []
    for threshold in (0.50, 0.55, 0.60, 0.65):
        subset = frame[confidence >= threshold]
        actual = subset.get("actual_class", pd.Series(dtype=object)).map(safe_int)
        predicted = subset.get("predicted_class", pd.Series(dtype=object)).map(safe_int)
        valid = actual.notna() & predicted.notna()
        hit = int((actual[valid].astype(int) == predicted[valid].astype(int)).sum())
        count = int(valid.sum())
        result.append(
            {
                "candidate_name": candidate,
                "window": window,
                "threshold": threshold,
                "count": count,
                "hit": hit,
                "miss": count - hit,
                "accuracy": _divide(hit, count),
                "sample_too_small": count < 20,
            }
        )
    return result


def _candidate_decision(
    candidate: str,
    *,
    historical: dict[str, Any],
    live_like: dict[str, Any],
    production_historical: dict[str, Any],
    production_live_like: dict[str, Any],
    improved_windows: int,
    baseline_wins: int,
    shadow_confirmed: int,
    policy: ResearchPolicy,
) -> str:
    if candidate == "production_current":
        return "production_still_stronger"
    if int(historical.get("sample_count") or 0) < policy.min_offline_sample:
        return "insufficient_live_sample"
    balanced_delta = _delta(historical.get("balanced_accuracy"), production_historical.get("balanced_accuracy"))
    up_delta = _delta(historical.get("up_recall"), production_historical.get("up_recall"))
    down_delta = _delta(historical.get("down_recall"), production_historical.get("down_recall"))
    if balanced_delta is None or balanced_delta <= 0:
        return "production_still_stronger"
    if baseline_wins == 0:
        return "baseline_still_stronger"
    if abs(balanced_delta) < policy.material_balanced_delta:
        return "no_material_change"
    if int(live_like.get("sample_count") or 0) < policy.min_live_like_sample:
        return "promising_historical_only"
    live_like_delta = _delta(live_like.get("balanced_accuracy"), production_live_like.get("balanced_accuracy"))
    guards_pass = bool(
        balanced_delta >= policy.material_balanced_delta
        and (up_delta or 0) >= policy.material_up_recall_delta
        and (down_delta or 0) >= -policy.max_down_recall_drop
        and improved_windows >= 2
        and baseline_wins >= 2
        and (live_like_delta or 0) > 0
    )
    if not guards_pass:
        return "promising_historical_only"
    if shadow_confirmed < policy.min_shadow_sample:
        return "continue_shadow_monitoring"
    return "production_change_proposal_ready"


def _research_classification(ranking: list[dict[str, Any]]) -> str:
    if any(row.get("decision") == "production_change_proposal_ready" for row in ranking):
        return "research_candidate_ready_for_human_review"
    if any(str(row.get("decision", "")).startswith("promising") or row.get("decision") == "continue_shadow_monitoring" for row in ranking):
        return "research_candidate_requires_more_evidence"
    return "research_completed_no_material_change"


def _recommendation(classification: str, ranking: list[dict[str, Any]]) -> str:
    if classification == "research_candidate_ready_for_human_review":
        return "production_change_proposal_ready"
    if classification == "research_candidate_requires_more_evidence":
        return "production_change_not_ready; continue evidence collection"
    if classification.startswith("research_skipped"):
        return "production_change_not_ready; Research was safely skipped"
    return "production_change_not_ready"


def _research_findings(
    *,
    ticker: str,
    fingerprint: str,
    ranking: list[dict[str, Any]],
    live_metrics: dict[str, Any],
    shadow: dict[str, Any],
    report_path: str,
    existing_live_model_review: bool = False,
) -> list[dict[str, Any]]:
    findings = []
    for row in ranking:
        if row.get("decision") != "production_change_proposal_ready":
            continue
        candidate = str(row["candidate_name"])
        findings.append(
            {
                "fingerprint": f"research_candidate:{ticker}:{candidate}:{fingerprint}",
                "priority": "medium",
                "title": f"Review Research candidate: {candidate}",
                "description": "Offline, live-like, baseline, and shadow evidence met the proposal gate. No production change was applied.",
                "candidate": candidate,
                "evidence": row,
                "production_difference": {"balanced_accuracy": row.get("historical_balanced_accuracy")},
                "baseline_difference": {"baseline_win_window_count": row.get("baseline_win_window_count")},
                "risks": ["offline evidence is not a future performance guarantee", "human review required"],
                "proposed_action": "Review the evidence and approve or reject a separate production-change task.",
                "source_report": report_path,
            }
        )
    confirmed = int(live_metrics.get("sample_count") or 0)
    balanced = safe_float(live_metrics.get("balanced_accuracy"))
    up_recall = safe_float(live_metrics.get("up_recall"))
    if (
        not existing_live_model_review
        and confirmed >= 15
        and (
            (balanced is not None and balanced < 0.45)
            or (up_recall is not None and up_recall <= 0.10)
        )
    ):
        findings.append(
            {
                "fingerprint": f"model_degradation:{ticker}:{fingerprint}",
                "priority": "high",
                "title": "Review weak confirmed live binary performance",
                "description": "Confirmed live performance remains weak. Research did not modify production.",
                "candidate": "production_current",
                "evidence": {"live_metrics": live_metrics, "shadow_confirmed": shadow.get("confirmed_count")},
                "production_difference": {},
                "baseline_difference": {},
                "risks": ["small live sample", "weak up recall", "no automatic model change"],
                "proposed_action": "Review offline and shadow evidence; keep production unchanged until a separate approval.",
                "source_report": report_path,
            }
        )
    return findings


def _has_active_live_model_review(path: Path, ticker: str) -> bool:
    if not path.exists():
        return False
    try:
        queue = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    active_statuses = {"open", "monitoring", "approval_required", "blocked"}
    expected = f"model_up_recall_low:{ticker}"
    return any(
        str(task.get("fingerprint")) == expected
        and str(task.get("status")) in active_statuses
        for task in queue.get("tasks", [])
        if isinstance(task, dict)
    )


def _updated_state(
    previous: dict[str, Any],
    result: dict[str, Any],
    *,
    production_guard: dict[str, Any],
) -> dict[str, Any]:
    success = result["classification"] != "research_failed"
    state = default_research_state()
    state.update(previous)
    state.update(
        {
            "schema_version": SCHEMA_VERSION,
            "last_cycle_at": result["cycle_completed_at"],
            "last_successful_cycle_at": result["cycle_completed_at"] if success else state.get("last_successful_cycle_at"),
            "last_heavy_experiment_at": result["cycle_completed_at"] if result["experiment_executed"] else state.get("last_heavy_experiment_at"),
            "last_operations_report": result.get("operations_report"),
            "last_operations_status": result["operations_health"],
            "last_live_confirmed_count": result["live_metrics"].get("sample_count", 0),
            "last_shadow_confirmed_count": result["shadow_metrics"].get("confirmed_count", 0),
            "last_live_metrics": result["live_metrics"],
            "last_shadow_metrics": result["shadow_metrics"],
            "last_candidate_ranking": result["candidate_ranking"] or state.get("last_candidate_ranking", []),
            "last_recommendation": result["recommendation"],
            "last_report_path": result["report_paths"]["markdown"],
            "consecutive_failures": 0 if success else int(state.get("consecutive_failures") or 0) + 1,
            "next_due_reason": result["next_due_condition"],
            "experiment_fingerprint": result["experiment_fingerprint"] if result["experiment_executed"] else state.get("experiment_fingerprint") or result["experiment_fingerprint"],
            "last_trigger_reasons": result["trigger_reasons"],
            "last_skip_reason": result.get("skip_reason"),
            "production_guard": production_guard,
            "shadow_direction_disagreement_count": result["shadow_metrics"].get("direction_disagreement_count", 0),
        }
    )
    return state


def _previous_delta(state: dict[str, Any], live: dict[str, Any], shadow: dict[str, Any]) -> dict[str, Any]:
    previous_live = state.get("last_live_metrics") or {}
    return {
        "live_confirmed": _delta(live.get("sample_count"), state.get("last_live_confirmed_count")),
        "live_balanced_accuracy": _delta(live.get("balanced_accuracy"), previous_live.get("balanced_accuracy")),
        "live_up_recall": _delta(live.get("up_recall"), previous_live.get("up_recall")),
        "shadow_confirmed": _delta(shadow.get("confirmed_count"), state.get("last_shadow_confirmed_count")),
    }


def _next_due_reason(classification: str) -> str:
    if classification == "research_skipped_operations_unhealthy":
        return "after_operations_returns_healthy"
    return "five_new_live_or_shadow_confirmations_or_seven_days_or_material_metric_change"


def _exit_code(result: dict[str, Any]) -> int:
    if result["classification"] == "research_skipped_operations_unhealthy":
        return 3
    if result["classification"] == "research_failed":
        return 1
    if result.get("approval_required"):
        return 2
    return 0


def _report_paths(folder: Path, at: pd.Timestamp) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / f"research_{at.strftime('%Y-%m-%d_%H%M')}"
    index = 1
    candidate = base
    while candidate.with_suffix(".json").exists() or candidate.with_suffix(".md").exists():
        index += 1
        candidate = folder / f"{base.name}_{index}"
    return {"markdown": candidate.with_suffix(".md"), "json": candidate.with_suffix(".json")}


def _window_rows(frame: pd.DataFrame, window: dict[str, Any]) -> pd.DataFrame:
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    start = pd.Timestamp(window["start_date"])
    end = pd.Timestamp(window["end_date"])
    return frame[(dates >= start) & (dates <= end)].copy()


def _simple_balanced(actual: pd.Series, predicted: pd.Series) -> dict[str, Any]:
    valid = actual.notna() & predicted.notna()
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    up = actual == 1
    down = actual == 0
    up_recall = _divide(int((predicted[up] == 1).sum()), int(up.sum()))
    down_recall = _divide(int((predicted[down] == 0).sum()), int(down.sum()))
    return {
        "sample_count": int(valid.sum()),
        "accuracy": _divide(int((actual == predicted).sum()), int(valid.sum())),
        "balanced_accuracy": (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None,
    }


def _roc_auc(actual: pd.Series, probability: pd.Series) -> float | None:
    valid = actual.notna() & probability.notna()
    actual = actual[valid].astype(int)
    probability = probability[valid].astype(float)
    positives = int((actual == 1).sum())
    negatives = int((actual == 0).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = probability.rank(method="average")
    return float((ranks[actual == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _f1(tp: int, fp: int, fn: int) -> float | None:
    return _divide(2 * tp, 2 * tp + fp + fn)


def _shadow_pair_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts = {
        "production_current_only_correct": 0,
        "class_weight_balanced_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    if rows.empty or "candidate_name" not in rows:
        return counts
    for _date, group in rows.groupby("feature_date"):
        status = {str(row["candidate_name"]): str(row.get("result")) for row in group.to_dict(orient="records")}
        production = status.get("production_current")
        balanced = status.get("class_weight_balanced")
        if production not in {"hit", "miss"} or balanced not in {"hit", "miss"}:
            continue
        if production == "hit" and balanced == "hit":
            counts["both_correct"] += 1
        elif production == "miss" and balanced == "miss":
            counts["both_wrong"] += 1
        elif production == "hit":
            counts["production_current_only_correct"] += 1
        else:
            counts["class_weight_balanced_only_correct"] += 1
    return counts


def _empty_shadow(status: str) -> dict[str, Any]:
    return {
        "status": status,
        "total_rows": 0,
        "confirmed_count": 0,
        "pending_count": 0,
        "candidate_summary": [],
        "direction_agreement_count": 0,
        "direction_disagreement_count": 0,
        "production_current_only_correct": 0,
        "class_weight_balanced_only_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }


def _production_guard(settings: Settings) -> dict[str, Any]:
    return {
        "binary_model": _file_sha256(settings.paths.model_dir / "lightgbm_binary_7203.T.joblib"),
        "multiclass_model": _file_sha256(settings.paths.model_dir / "lightgbm_multiclass_7203.T.joblib"),
        "settings": _file_sha256(PROJECT_ROOT / "configs" / "settings.yml"),
        "daily_predict": _file_sha256(PROJECT_ROOT / "scripts" / "daily_predict.ps1"),
        "autonomous_operations": _file_sha256(PROJECT_ROOT / "src" / "stock_ai" / "operations" / "autonomous.py"),
        "autonomous_operations_wrapper": _file_sha256(PROJECT_ROOT / "scripts" / "run_autonomous_operations.ps1"),
    }


def _database_snapshot(path: Path) -> dict[str, Any]:
    snapshot = {"sha256": _file_sha256(path), "counts": {}, "integrity": "unavailable"}
    try:
        with sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True) as connection:
            snapshot["integrity"] = connection.execute("PRAGMA integrity_check").fetchone()[0]
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            snapshot["counts"] = {
                table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("predictions", "prediction_runs", "prediction_records")
                if table in tables
            }
    except (OSError, sqlite3.Error) as error:
        snapshot["error"] = str(error)
    return snapshot


def _invariant_summary(before: dict[str, Any], after: dict[str, Any], db_before: dict[str, Any], db_after: dict[str, Any]) -> dict[str, bool]:
    return {
        "production_model_unchanged": before.get("binary_model") == after.get("binary_model") and before.get("multiclass_model") == after.get("multiclass_model"),
        "production_config_unchanged": before.get("settings") == after.get("settings"),
        "daily_predict_unchanged": before.get("daily_predict") == after.get("daily_predict"),
        "autonomous_operations_unchanged": before.get("autonomous_operations") == after.get("autonomous_operations") and before.get("autonomous_operations_wrapper") == after.get("autonomous_operations_wrapper"),
        "production_sqlite_unchanged": db_before == db_after,
        "existing_predictions_unchanged": db_before.get("counts", {}).get("predictions") == db_after.get("counts", {}).get("predictions"),
    }


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _combined_hash(paths: list[Path]) -> str:
    payload = [(str(path), _file_sha256(path)) for path in paths]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _cycle_id(ticker: str, started: pd.Timestamp, fingerprint: str, classification: str) -> str:
    value = f"{ticker}|{started.isoformat()}|{fingerprint}|{classification}"
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _lock_owner_is_active(path: Path) -> bool:
    try:
        pid = int(json.loads(path.read_text(encoding="utf-8")).get("pid"))
        if pid == os.getpid():
            return True
        os.kill(pid, 0)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return _json_ready(frame.to_dict(orient="records")) if not frame.empty else []


def _latest_date(values: Any) -> str | None:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    return pd.Timestamp(parsed.max()).strftime("%Y-%m-%d") if not parsed.empty else None


def _delta(left: Any, right: Any) -> float | None:
    left_value = safe_float(left)
    right_value = safe_float(right)
    return None if left_value is None or right_value is None else float(left_value - right_value)


def _divide(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _join(values: Any) -> str:
    values = list(values or [])
    return ", ".join(str(value) for value in values) if values else "-"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No rows."
    rows = frame.where(pd.notna(frame), "")
    lines = [
        "| " + " | ".join(str(column) for column in rows.columns) + " |",
        "| " + " | ".join("---" for _ in rows.columns) + " |",
    ]
    for row in rows.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(value), indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def policy_or_default(policy: ResearchPolicy | None) -> ResearchPolicy:
    return policy or ResearchPolicy()


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None
