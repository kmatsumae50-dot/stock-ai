from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import time
from typing import Any, Sequence

from stock_ai.config import PROJECT_ROOT, Settings, load_settings
from stock_ai.operations.autonomous import load_task_queue
from stock_ai.operations.autonomous_research import (
    assess_operations_health,
    load_latest_operations_report,
)


SCHEMA_VERSION = "autonomous_codex_research_agent/v1.1"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "autonomous"
DEFAULT_STATE_PATH = DEFAULT_REPORT_DIR / "codex_research_agent_state.json"
DEFAULT_HYPOTHESES_PATH = DEFAULT_REPORT_DIR / "codex_research_hypotheses.json"
DEFAULT_TASK_QUEUE_PATH = DEFAULT_REPORT_DIR / "operations_task_queue.json"
DEFAULT_PROMPT_TEMPLATE = PROJECT_ROOT / "prompts" / "autonomous_codex_research_agent.md"
DEFAULT_WORKTREE_ROOT = PROJECT_ROOT.parent / "stock-ai-worktrees"
DEFAULT_LOCK_PATH = DEFAULT_REPORT_DIR / ".codex_research_agent.lock"
DEFAULT_CONTEXT_DIR = DEFAULT_REPORT_DIR / "context"
HEAVY_CYCLE_INTERVAL_DAYS = 7
CONFIRMED_COUNT_TRIGGER = 5
HYPOTHESIS_STATUSES = {
    "proposed",
    "queued",
    "running",
    "completed",
    "rejected",
    "blocked",
    "monitoring",
    "approval_required",
    "cleanup_ready",
    "archived",
}
TERMINAL_HYPOTHESIS_STATUSES = {"completed", "rejected", "archived", "cleanup_ready"}
NORMAL_SKIP_CLASSIFICATIONS = {
    "agent_skipped_not_due",
    "agent_skipped_no_safe_hypothesis",
    "agent_skipped_duplicate_hypothesis",
    "agent_skipped_duplicate_research_state",
    "agent_skipped_unhealthy_gate",
    "agent_skipped_worktree_review_required",
}
ALLOWED_CATEGORIES = {
    "binary_objective_weighting",
    "existing_feature_research",
    "ensemble_decision_rule",
    "evaluation_quality",
}
ALLOWED_CLASSIFICATIONS = {
    "hypothesis_rejected",
    "no_material_change",
    "promising_offline_only",
    "promising_recent_period",
    "promising_live_like",
    "continue_shadow_monitoring",
    "additional_experiment_needed",
    "human_review_recommended",
    "unsafe_or_leakage_risk",
    "experiment_failed",
}
REQUIRED_PERIODS = {
    "historical_all",
    "since_2024",
    "since_2025",
    "since_2026",
    "recent_6_months",
    "recent_3_months",
    "live_like_holdout",
}
REQUIRED_BASELINES = {
    "always_up",
    "always_down",
    "majority_class",
    "previous_day_direction",
    "moving_average_trend_only",
}


class CodexAgentError(RuntimeError):
    pass


class CodexAgentLockError(CodexAgentError):
    pass


class CodexAgentTimeoutError(CodexAgentError):
    pass


@dataclass(frozen=True)
class AgentPolicy:
    max_runtime_minutes: float = 90.0
    hard_max_runtime_minutes: float = 180.0
    max_hypotheses_per_cycle: int = 1
    max_worktrees_per_cycle: int = 1
    max_codex_tasks_per_cycle: int = 1
    max_implementation_attempts: int = 1
    max_retuning_rounds_after_result: int = 0
    max_experiment_variants: int = 6
    max_changed_files: int = 8
    max_patch_lines: int = 1500
    report_freshness_minutes: float = 240.0

    def validate(self) -> None:
        if self.hard_max_runtime_minutes != 180:
            raise ValueError("v1.1 hard_max_runtime_minutes must remain 180.")
        if self.max_runtime_minutes <= 0 or self.max_runtime_minutes > self.hard_max_runtime_minutes:
            raise ValueError("max_runtime_minutes must be in (0, hard_max_runtime_minutes].")
        if self.max_hypotheses_per_cycle != 1:
            raise ValueError("v1.1 requires exactly one hypothesis maximum per cycle.")
        if self.max_worktrees_per_cycle != 1 or self.max_codex_tasks_per_cycle != 1:
            raise ValueError("v1.1 permits at most one worktree and one Codex task per cycle.")
        if self.max_implementation_attempts != 1:
            raise ValueError("v1.1 permits exactly one implementation attempt per cycle.")
        if self.max_retuning_rounds_after_result != 0:
            raise ValueError("v1.1 does not permit same-cycle retuning after a result.")
        if self.max_experiment_variants <= 0 or self.max_experiment_variants > 6:
            raise ValueError("max_experiment_variants must be in [1, 6].")
        if self.max_changed_files <= 0 or self.max_changed_files > 8:
            raise ValueError("max_changed_files must be in [1, 8].")
        if self.max_patch_lines <= 0 or self.max_patch_lines > 1500:
            raise ValueError("max_patch_lines must be in [1, 1500].")

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_hypotheses_per_cycle": self.max_hypotheses_per_cycle,
            "max_codex_tasks_per_cycle": self.max_codex_tasks_per_cycle,
            "max_implementation_attempts": self.max_implementation_attempts,
            "max_retuning_rounds_after_result": self.max_retuning_rounds_after_result,
            "max_experiment_variants": self.max_experiment_variants,
            "max_changed_files": self.max_changed_files,
            "max_patch_lines": self.max_patch_lines,
            "max_runtime_minutes": self.max_runtime_minutes,
            "hard_max_runtime_minutes": self.hard_max_runtime_minutes,
        }


@dataclass
class RuntimeBudget:
    started: float
    max_minutes: float

    @classmethod
    def start(cls, max_minutes: float) -> RuntimeBudget:
        AgentPolicy(max_runtime_minutes=max_minutes).validate()
        return cls(time.monotonic(), float(max_minutes))

    def check(self, stage: str) -> None:
        if time.monotonic() - self.started > self.max_minutes * 60:
            raise CodexAgentTimeoutError(f"Codex Research Agent timed out at {stage}.")


class AgentFileLock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.acquired = False

    def __enter__(self) -> AgentFileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise CodexAgentLockError(f"Codex Research Agent lock exists: {self.path}") from error
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump({"pid": os.getpid(), "created_at": _now_iso()}, file)
        self.acquired = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self.acquired:
            self.path.unlink(missing_ok=True)


def default_agent_state() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "last_cycle_at": None,
        "last_successful_cycle_at": None,
        "last_hypothesis_id": None,
        "last_worktree": None,
        "last_report": None,
        "last_result": None,
        "last_operations_report": None,
        "last_research_report": None,
        "open_hypotheses": 0,
        "completed_hypotheses": 0,
        "rejected_hypotheses": 0,
        "active_worktree": None,
        "consecutive_failures": 0,
        "next_due_reason": "first_cycle",
        "last_preflight_at": None,
        "last_heavy_cycle_at": None,
        "last_heavy_cycle_token_usage": None,
        "last_heavy_cycle_runtime_minutes": None,
        "last_research_fingerprint": None,
        "last_live_confirmed_count": 0,
        "last_shadow_confirmed_count": 0,
        "last_selected_hypothesis_fingerprint": None,
        "completed_hypothesis_fingerprints": [],
        "rejected_hypothesis_fingerprints": [],
        "worktree_status": None,
        "next_due_at": None,
        "next_due_reasons": ["first_cycle"],
        "last_context_bundle": None,
        "last_skip_classification": None,
    }


def load_agent_state(path: str | Path) -> tuple[dict[str, Any], bool]:
    state_path = Path(path)
    if not state_path.exists():
        return default_agent_state(), True
    try:
        loaded = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CodexAgentError(f"Agent state is unreadable and was not deleted: {state_path}") from error
    if not isinstance(loaded, dict):
        raise CodexAgentError(f"Agent state must be a JSON object: {state_path}")
    state = default_agent_state()
    state.update(loaded)
    state["schema_version"] = SCHEMA_VERSION
    return state, False


def hypothesis_fingerprint(hypothesis: dict[str, Any]) -> str:
    stable = {
        "category": str(hypothesis.get("category") or "").strip().lower(),
        "title": _normalize_text(hypothesis.get("title")),
        "rationale": _normalize_text(hypothesis.get("rationale")),
        "implementation_scope": sorted(
            _normalize_text(value) for value in hypothesis.get("implementation_scope", [])
        ),
        "primary_metric": _normalize_text(hypothesis.get("primary_metric")),
    }
    return hashlib.sha256(
        json.dumps(stable, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def seed_hypothesis_backlog(now: str | None = None) -> dict[str, Any]:
    timestamp = now or _now_iso()
    seeds = [
        {
            "hypothesis_id": "balanced_threshold_guardrail",
            "title": "Use a train-only threshold to reduce class-weight down-recall loss",
            "category": "binary_objective_weighting",
            "rationale": "class_weight_balanced improves historical balance but weakens down recall; a validation-only threshold may restore the guardrail.",
            "evidence": ["class_weight_balanced is promising_historical_only", "production up recall remains weak"],
            "expected_benefit": "Improve up recall and balanced accuracy without a material down-recall loss.",
            "primary_metric": "balanced_accuracy",
            "guardrail_metrics": ["down_recall", "baseline_edge", "up_precision"],
            "implementation_scope": ["research-only threshold selection", "walk-forward validation folds"],
            "risk_level": "low",
            "estimated_runtime_minutes": 90,
            "status": "proposed",
        },
        {
            "hypothesis_id": "scale_pos_weight_stability",
            "title": "Test whether the medium positive-class weight is stable across periods",
            "category": "binary_objective_weighting",
            "rationale": "scale_pos_weight_medium ranked first but was classified no_material_change.",
            "evidence": ["ranked first in the latest bounded Research cycle"],
            "expected_benefit": "Determine whether the small improvement survives recent and live-like windows.",
            "primary_metric": "up_recall",
            "guardrail_metrics": ["balanced_accuracy", "down_recall", "baseline_edge"],
            "implementation_scope": ["existing scale_pos_weight candidate", "period robustness diagnostics"],
            "risk_level": "low",
            "estimated_runtime_minutes": 75,
            "status": "proposed",
        },
        {
            "hypothesis_id": "production_balanced_disagreement_rule",
            "title": "Evaluate a simple production and balanced-model disagreement hold rule",
            "category": "ensemble_decision_rule",
            "rationale": "A bounded disagreement rule may avoid low-confidence directional errors without replacing production.",
            "evidence": ["production edge is weak", "balanced candidate changes class behavior"],
            "expected_benefit": "Improve precision on acted predictions while reporting coverage loss.",
            "primary_metric": "balanced_accuracy",
            "guardrail_metrics": ["coverage", "up_recall", "down_recall"],
            "implementation_scope": ["simple two-model rule", "no production persistence"],
            "risk_level": "medium",
            "estimated_runtime_minutes": 100,
            "status": "proposed",
        },
        {
            "hypothesis_id": "confidence_calibration_diagnosis",
            "title": "Diagnose confidence calibration with train-only calibration folds",
            "category": "evaluation_quality",
            "rationale": "Low and unreliable confidence may obscure whether threshold changes are meaningful.",
            "evidence": ["confidence is generally low", "production baseline edge is weak"],
            "expected_benefit": "Measure reliability and identify whether calibration is useful before model changes.",
            "primary_metric": "brier_score",
            "guardrail_metrics": ["balanced_accuracy", "up_recall", "calibration_sample_count"],
            "implementation_scope": ["calibration diagnostics", "train-validation only fitting"],
            "risk_level": "low",
            "estimated_runtime_minutes": 70,
            "status": "proposed",
        },
        {
            "hypothesis_id": "up_prediction_decomposition",
            "title": "Decompose low up-prediction frequency into feature and threshold effects",
            "category": "evaluation_quality",
            "rationale": "Historical and live evaluations show low up recall and a down-side prediction bias.",
            "evidence": ["historical up recall is below down recall", "live up recall has been weak"],
            "expected_benefit": "Identify the smallest safe intervention before changing production.",
            "primary_metric": "predicted_up_ratio",
            "guardrail_metrics": ["balanced_accuracy", "up_precision", "down_recall"],
            "implementation_scope": ["feature contribution diagnostics", "threshold sensitivity"],
            "risk_level": "low",
            "estimated_runtime_minutes": 80,
            "status": "proposed",
        },
        {
            "hypothesis_id": "recent_regime_robustness",
            "title": "Compare current binary behavior across existing recent regimes",
            "category": "evaluation_quality",
            "rationale": "Recent behavior may differ from broad history, but small regimes must not drive production changes.",
            "evidence": ["historical edge is weak", "recent live sample is limited"],
            "expected_benefit": "Clarify whether candidate gains are concentrated in one regime.",
            "primary_metric": "balanced_accuracy",
            "guardrail_metrics": ["regime_sample_count", "up_recall", "down_recall"],
            "implementation_scope": ["existing regime labels", "minimum sample guard"],
            "risk_level": "low",
            "estimated_runtime_minutes": 75,
            "status": "proposed",
        },
        {
            "hypothesis_id": "feature_group_overfit_ablation",
            "title": "Identify overfit feature groups through bounded ablation",
            "category": "existing_feature_research",
            "rationale": "A weak baseline edge can indicate noisy feature groups rather than insufficient complexity.",
            "evidence": ["production barely exceeds simple baselines"],
            "expected_benefit": "Find removable feature groups without changing feature generation.",
            "primary_metric": "balanced_accuracy",
            "guardrail_metrics": ["up_recall", "down_recall", "window_consistency"],
            "implementation_scope": ["existing feature groups only", "bounded ablation candidates"],
            "risk_level": "medium",
            "estimated_runtime_minutes": 120,
            "status": "proposed",
        },
    ]
    hypotheses = []
    for item in seeds:
        hypothesis = {
            **item,
            "created_at": timestamp,
            "updated_at": timestamp,
            "selected_at": None,
            "worktree": None,
            "report_path": None,
            "result": None,
            "rejection_reason": None,
            "next_action": "await_selection",
        }
        hypothesis["fingerprint"] = hypothesis_fingerprint(hypothesis)
        hypotheses.append(hypothesis)
    return {"schema_version": SCHEMA_VERSION, "updated_at": timestamp, "hypotheses": hypotheses}


def load_hypothesis_backlog(path: str | Path) -> tuple[dict[str, Any], bool]:
    backlog_path = Path(path)
    if not backlog_path.exists():
        return seed_hypothesis_backlog(), True
    try:
        loaded = json.loads(backlog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CodexAgentError(f"Hypothesis backlog is unreadable: {backlog_path}") from error
    if not isinstance(loaded, dict) or not isinstance(loaded.get("hypotheses"), list):
        raise CodexAgentError(f"Hypothesis backlog has an invalid schema: {backlog_path}")
    loaded["schema_version"] = SCHEMA_VERSION
    loaded["hypotheses"] = deduplicate_hypotheses(loaded["hypotheses"])
    return loaded, False


def deduplicate_hypotheses(hypotheses: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for raw in hypotheses:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        fingerprint = str(item.get("fingerprint") or hypothesis_fingerprint(item))
        item["fingerprint"] = fingerprint
        unique.setdefault(fingerprint, item)
    return list(unique.values())


def summarize_hypothesis_lifecycle(
    backlog: dict[str, Any],
    *,
    state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = deduplicate_hypotheses(backlog.get("hypotheses", []))
    completed = set((state or {}).get("completed_hypothesis_fingerprints") or [])
    rejected = set((state or {}).get("rejected_hypothesis_fingerprints") or [])
    invalid_status = []
    duplicate_ids = []
    seen_ids: set[str] = set()
    for item in items:
        hypothesis_id = str(item.get("hypothesis_id") or "")
        status = str(item.get("status") or "")
        fingerprint = str(item.get("fingerprint") or hypothesis_fingerprint(item))
        if hypothesis_id in seen_ids:
            duplicate_ids.append(hypothesis_id)
        seen_ids.add(hypothesis_id)
        if status not in HYPOTHESIS_STATUSES:
            invalid_status.append(hypothesis_id)
        if status == "completed":
            completed.add(fingerprint)
        elif status == "rejected":
            rejected.add(fingerprint)
    selectable = [
        item
        for item in items
        if item.get("status") in {"proposed", "queued"}
        and item.get("fingerprint") not in completed | rejected
    ]
    return {
        "hypothesis_count": len(items),
        "status_counts": {
            status: sum(str(item.get("status")) == status for item in items)
            for status in sorted(HYPOTHESIS_STATUSES)
        },
        "selectable_count": len(selectable),
        "selectable_fingerprints": [item.get("fingerprint") for item in selectable],
        "completed_fingerprints": sorted(completed),
        "rejected_fingerprints": sorted(rejected),
        "duplicate_hypothesis_ids": duplicate_ids,
        "invalid_status_hypothesis_ids": invalid_status,
        "consistent": not duplicate_ids and not invalid_status,
    }


def select_hypothesis(
    backlog: dict[str, Any],
    *,
    research_report: dict[str, Any] | None = None,
    excluded_fingerprints: Sequence[str] | None = None,
) -> dict[str, Any]:
    excluded = set(excluded_fingerprints or [])
    candidates = [
        item
        for item in deduplicate_hypotheses(backlog.get("hypotheses", []))
        if item.get("status") in {"proposed", "queued"}
        and item.get("fingerprint") not in excluded
        and item.get("category") in ALLOWED_CATEGORIES
        and item.get("risk_level") in {"low", "medium"}
    ]
    if not candidates:
        return {"selected": None, "rejected_candidates": [], "reason": "no_safe_hypothesis_available"}

    classification = str((research_report or {}).get("classification") or "")
    ranking = (research_report or {}).get("candidate_ranking") or []
    ranked_names = [str(row.get("candidate_name") or "") for row in ranking if isinstance(row, dict)]
    priority_ids = []
    if "class_weight_balanced" in ranked_names or "promising_historical_only" in classification:
        priority_ids.append("balanced_threshold_guardrail")
    if "scale_pos_weight_medium" in ranked_names:
        priority_ids.append("scale_pos_weight_stability")
    priority_ids.extend(
        [
            "up_prediction_decomposition",
            "confidence_calibration_diagnosis",
            "recent_regime_robustness",
            "feature_group_overfit_ablation",
            "production_balanced_disagreement_rule",
        ]
    )
    order = {hypothesis_id: index for index, hypothesis_id in enumerate(priority_ids)}
    candidates.sort(
        key=lambda item: (
            order.get(str(item.get("hypothesis_id")), len(order)),
            0 if item.get("risk_level") == "low" else 1,
            int(item.get("estimated_runtime_minutes") or 999),
        )
    )
    selected = candidates[0]
    rejected = [
        {
            "hypothesis_id": item.get("hypothesis_id"),
            "reason": "lower_priority_for_this_single-hypothesis_cycle",
        }
        for item in candidates[1:]
    ]
    return {
        "selected": selected,
        "rejected_candidates": rejected,
        "reason": "highest_safe_information_value_under_current_research_evidence",
        "success_conditions": [
            "balanced_accuracy improves over production",
            "primary metric improves",
            "down recall guardrail has no material degradation",
            "candidate remains above simple baselines across multiple periods",
            "no leakage is detected",
        ],
        "stop_conditions": [
            "target leakage",
            "candidate scope exceeds one hypothesis",
            "external data or production change is required",
            "tests cannot be made green safely",
            "runtime limit is reached",
        ],
    }


def inspect_git_state(project_root: str | Path = PROJECT_ROOT) -> dict[str, Any]:
    root = Path(project_root)
    top = _git(root, "rev-parse", "--show-toplevel")
    head = _git(root, "rev-parse", "--verify", "HEAD")
    status = _git(root, "status", "--porcelain=v1", "--untracked-files=all")
    branch = _git(root, "branch", "--show-current")
    status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
    return {
        "is_repository": top["returncode"] == 0,
        "repository_root": top["stdout"].strip() or None,
        "has_head": head["returncode"] == 0,
        "head": head["stdout"].strip() or None,
        "branch": branch["stdout"].strip() or None,
        "clean": head["returncode"] == 0 and status["returncode"] == 0 and not status_lines,
        "status_lines": status_lines,
        "status_count": len(status_lines),
        "error": top["stderr"].strip() or head["stderr"].strip() or None,
    }


def probe_codex_cli(executable: str = "codex") -> dict[str, Any]:
    resolved = shutil.which(executable)
    if not resolved:
        return {"available": False, "executable": executable, "reason": "codex_cli_not_found"}
    help_result = _subprocess([resolved, "--help"], timeout=30)
    if help_result["returncode"] != 0:
        return {
            "available": False,
            "executable": resolved,
            "reason": "codex_help_failed",
            "returncode": help_result["returncode"],
            "stderr": help_result["stderr"],
        }
    exec_help = _subprocess([resolved, "exec", "--help"], timeout=30)
    if exec_help["returncode"] != 0:
        return {
            "available": False,
            "executable": resolved,
            "reason": "codex_exec_help_failed",
            "returncode": exec_help["returncode"],
            "stderr": exec_help["stderr"],
        }
    help_text = exec_help["stdout"]
    required_flags = ["--cd", "--sandbox", "--ask-for-approval", "--output-last-message"]
    missing_flags = [flag for flag in required_flags if flag not in help_text]
    return {
        "available": not missing_flags,
        "executable": resolved,
        "reason": "available" if not missing_flags else "required_safe_flags_unavailable",
        "missing_safe_flags": missing_flags,
        "supports_model": "--model" in help_text,
        "supports_reasoning_config": "--config" in help_text,
        "supports_working_directory": "--cd" in help_text,
        "supports_sandbox": "--sandbox" in help_text,
        "supports_approval": "--ask-for-approval" in help_text,
        "supports_timeout": False,
        "help_verified": True,
    }


def assess_agent_gate(
    *,
    operations_report: dict[str, Any],
    research_report: dict[str, Any],
    git_state: dict[str, Any],
    db_readable: bool,
    production_config_matches: bool,
    report_freshness: dict[str, Any] | None = None,
    lock_active: bool = False,
) -> dict[str, Any]:
    operations_gate = assess_operations_health(operations_report)
    reasons = list(operations_gate.get("reasons") or [])
    research_classification = str(research_report.get("classification") or "")
    if not research_report or research_classification in {
        "research_skipped_operations_unhealthy",
        "experiment_failed",
    }:
        reasons.append("latest_research_unhealthy")
    if not research_report.get("experiment_fingerprint"):
        reasons.append("latest_research_fingerprint_missing")
    freshness = report_freshness or {}
    if freshness and not freshness.get("operations_fresh"):
        reasons.append("operations_report_stale")
    if freshness and not freshness.get("research_fresh"):
        reasons.append("research_report_stale")
    if not git_state.get("is_repository"):
        reasons.append("main_not_git_repository")
    elif not git_state.get("has_head"):
        reasons.append("main_git_head_missing")
    if not git_state.get("clean"):
        reasons.append("main_worktree_dirty")
    if not db_readable:
        reasons.append("production_database_unreadable")
    if not production_config_matches:
        reasons.append("production_config_unexpected_change")
    if lock_active:
        reasons.append("another_agent_cycle_running")
    reasons = list(dict.fromkeys(reasons))
    return {
        "healthy": not reasons,
        "reasons": reasons,
        "operations": operations_gate,
        "research_classification": research_classification or None,
        "research_fingerprint": research_report.get("experiment_fingerprint"),
        "main_git": git_state,
        "production_database_readable": db_readable,
        "production_config_matches": production_config_matches,
        "report_freshness": freshness,
    }


def research_observation_counts(research_report: dict[str, Any]) -> dict[str, int]:
    live = research_report.get("live_metrics") or {}
    shadow = research_report.get("shadow_metrics") or {}
    live_count = int(live.get("confirmed_count") or live.get("sample_count") or 0)
    shadow_count = int(shadow.get("confirmed_count") or 0)
    return {"live_confirmed_count": live_count, "shadow_confirmed_count": shadow_count}


def inspect_research_worktree(
    hypothesis: dict[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    worktree_root: str | Path = DEFAULT_WORKTREE_ROOT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    check_processes: bool = True,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    worktrees = Path(worktree_root).resolve()
    reports = Path(report_dir).resolve()
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    configured_path = hypothesis.get("worktree")
    path = Path(configured_path).resolve() if configured_path else worktrees / hypothesis_id
    registered = _registered_worktrees(root)
    registration = registered.get(str(path).casefold())
    path_exists = path.exists()
    report_path = _existing_artifact_path(hypothesis.get("report_path"))
    patch_path = _existing_artifact_path(hypothesis.get("patch_path"))
    if report_path is None:
        report_path = _latest_matching_path(reports, f"codex_agent_{hypothesis_id}_*.json")
        report_path = report_path or _latest_matching_path(reports, f"codex_agent_{hypothesis_id}_*.md")
    if patch_path is None:
        patch_path = _latest_matching_path(reports / "patches", f"{hypothesis_id}_*.patch")

    main_git = inspect_git_state(root)
    worktree_git = {
        "clean": False,
        "status_lines": [],
        "head": None,
        "commit_count_ahead": None,
        "branch": (registration or {}).get("branch"),
    }
    current_patch_sha256 = None
    patch_matches_worktree = False
    if path_exists and registration:
        status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
        head = _git(path, "rev-parse", "HEAD")
        diff = _git(path, "diff", "--binary")
        status_lines = [line for line in status["stdout"].splitlines() if line.strip()]
        current_patch_sha256 = hashlib.sha256(diff["stdout"].encode("utf-8")).hexdigest()
        main_head = str(main_git.get("head") or "")
        ahead = _git(path, "rev-list", "--count", f"{main_head}..HEAD") if main_head else {}
        worktree_git = {
            "clean": not status_lines,
            "status_lines": status_lines,
            "head": head.get("stdout", "").strip() or None,
            "commit_count_ahead": _safe_int(ahead.get("stdout")) if ahead else None,
            "branch": (registration or {}).get("branch"),
        }
        if patch_path:
            try:
                recorded_patch_text = patch_path.read_text(encoding="utf-8")
            except OSError:
                recorded_patch_text = ""
            patch_matches_worktree = _normalized_newlines(recorded_patch_text) == _normalized_newlines(
                diff["stdout"]
            )

    recorded_patch_hash = str(hypothesis.get("patch_sha256") or "").lower() or None
    actual_patch_hash = (_file_sha256(patch_path) or "").lower() if patch_path else None
    patch_hash_recorded_and_valid = bool(
        recorded_patch_hash and actual_patch_hash and recorded_patch_hash == actual_patch_hash
    )
    process_matches = _worktree_processes(path) if check_processes and path_exists else []
    status = str(hypothesis.get("status") or "")
    reasons: list[str] = []
    if not path_exists and not registration:
        lifecycle_status = "orphaned" if configured_path else "inactive"
    elif path_exists != bool(registration):
        lifecycle_status = "orphaned"
        reasons.append("path_and_git_registration_disagree")
    elif status == "running":
        lifecycle_status = "active"
    elif status not in {"rejected", "completed", "cleanup_ready", "archived"}:
        lifecycle_status = "review_required"
        reasons.append("hypothesis_not_terminal")
    else:
        cleanup_checks = {
            "report_exists": report_path is not None,
            "patch_exists": patch_path is not None,
            "patch_hash_recorded_and_valid": patch_hash_recorded_and_valid,
            "patch_matches_worktree": patch_matches_worktree,
            "no_worktree_commits": worktree_git.get("commit_count_ahead") == 0,
            "no_running_process": not process_matches,
            "main_clean": bool(main_git.get("clean")),
        }
        failed = [name for name, passed in cleanup_checks.items() if not passed]
        reasons.extend(failed)
        lifecycle_status = "cleanup_ready" if not failed else "unsafe_to_remove"
    return {
        "hypothesis_id": hypothesis_id,
        "hypothesis_status": status,
        "path": str(path),
        "path_exists": path_exists,
        "registered": bool(registration),
        "branch": (registration or {}).get("branch"),
        "worktree_git": worktree_git,
        "report_path": str(report_path) if report_path else None,
        "report_exists": report_path is not None,
        "patch_path": str(patch_path) if patch_path else None,
        "patch_exists": patch_path is not None,
        "recorded_patch_sha256": recorded_patch_hash,
        "actual_patch_sha256": actual_patch_hash,
        "current_worktree_patch_sha256": current_patch_sha256,
        "patch_hash_recorded_and_valid": patch_hash_recorded_and_valid,
        "patch_matches_worktree": patch_matches_worktree,
        "process_matches": process_matches,
        "main_clean": bool(main_git.get("clean")),
        "lifecycle_status": lifecycle_status,
        "reasons": reasons,
        "safe_to_remove": lifecycle_status == "cleanup_ready",
    }


def assess_agent_due(
    *,
    state: dict[str, Any],
    backlog: dict[str, Any],
    research_report: dict[str, Any],
    task_queue: dict[str, Any],
    gate: dict[str, Any],
    worktree_audits: Sequence[dict[str, Any]] = (),
    now: datetime,
    force: bool = False,
) -> dict[str, Any]:
    lifecycle = summarize_hypothesis_lifecycle(backlog, state=state)
    terminal = set(lifecycle["completed_fingerprints"]) | set(lifecycle["rejected_fingerprints"])
    unrestricted_selection = select_hypothesis(backlog, research_report=research_report)
    selection = select_hypothesis(
        backlog,
        research_report=research_report,
        excluded_fingerprints=terminal,
    )
    counts = research_observation_counts(research_report)
    latest_fingerprint = str(research_report.get("experiment_fingerprint") or "") or None
    previous_fingerprint = str(state.get("last_research_fingerprint") or "") or None
    last_heavy = _parse_datetime(state.get("last_heavy_cycle_at") or state.get("last_cycle_at"))
    due_reasons: list[str] = []
    if force:
        due_reasons.append("explicit_force")
    if previous_fingerprint and latest_fingerprint and latest_fingerprint != previous_fingerprint:
        due_reasons.append("research_fingerprint_changed")
    if last_heavy is None and not state.get("last_cycle_at"):
        due_reasons.append("first_heavy_cycle")
    live_delta = counts["live_confirmed_count"] - int(state.get("last_live_confirmed_count") or 0)
    shadow_delta = counts["shadow_confirmed_count"] - int(
        state.get("last_shadow_confirmed_count") or 0
    )
    if live_delta >= CONFIRMED_COUNT_TRIGGER:
        due_reasons.append("live_confirmed_increased_by_5")
    if shadow_delta >= CONFIRMED_COUNT_TRIGGER:
        due_reasons.append("shadow_confirmed_increased_by_5")
    approval_tasks = _new_approval_model_research_tasks(task_queue, after=last_heavy)
    if approval_tasks:
        due_reasons.append("new_approval_required_model_research_task")
    safe_hypothesis = selection.get("selected") is not None
    if last_heavy and now.astimezone(UTC) >= last_heavy.astimezone(UTC) + timedelta(
        days=HEAVY_CYCLE_INTERVAL_DAYS
    ) and safe_hypothesis:
        due_reasons.append("seven_days_elapsed_with_safe_hypothesis")

    unresolved_worktrees = [
        audit
        for audit in worktree_audits
        if audit.get("lifecycle_status")
        in {"active", "review_required", "cleanup_ready", "orphaned", "unsafe_to_remove"}
    ]
    duplicate_candidate = unrestricted_selection.get("selected") and not safe_hypothesis
    research_duplicate = bool(
        latest_fingerprint
        and previous_fingerprint == latest_fingerprint
        and research_report.get("classification") == "research_skipped_duplicate_experiment"
    )
    due = bool(due_reasons)
    if not gate.get("healthy"):
        classification = "agent_skipped_unhealthy_gate"
    elif unresolved_worktrees:
        classification = "agent_skipped_worktree_review_required"
    elif duplicate_candidate:
        classification = "agent_skipped_duplicate_hypothesis"
    elif not safe_hypothesis:
        classification = "agent_skipped_no_safe_hypothesis"
    elif not due and research_duplicate:
        classification = "agent_skipped_duplicate_research_state"
    elif not due:
        classification = "agent_skipped_not_due"
    else:
        classification = "agent_due_ready"
    invocation_required = classification == "agent_due_ready"
    next_due = last_heavy + timedelta(days=HEAVY_CYCLE_INTERVAL_DAYS) if last_heavy else now
    reason_codes = list(due_reasons)
    if classification != "agent_due_ready":
        reason_codes.insert(0, classification)
    return {
        "due": invocation_required,
        "trigger_due": due,
        "classification": classification,
        "reason_codes": reason_codes,
        "trigger_reasons": due_reasons,
        "latest_research_fingerprint": latest_fingerprint,
        "last_agent_research_fingerprint": previous_fingerprint,
        **counts,
        "last_live_confirmed_count": int(state.get("last_live_confirmed_count") or 0),
        "last_shadow_confirmed_count": int(state.get("last_shadow_confirmed_count") or 0),
        "live_confirmed_delta": live_delta,
        "shadow_confirmed_delta": shadow_delta,
        "new_approval_required_tasks": approval_tasks,
        "selected_next_hypothesis": selection.get("selected"),
        "selection": selection,
        "duplicate_candidate": bool(duplicate_candidate),
        "duplicate_research_state": research_duplicate,
        "unresolved_worktrees": unresolved_worktrees,
        "codex_invocation_required": invocation_required,
        "worktree_creation_required": invocation_required,
        "next_due_at": next_due.isoformat(),
        "lifecycle": lifecycle,
    }


def autonomous_codex_research_status(
    *,
    ticker: str = "7203.T",
    state_path: str | Path = DEFAULT_STATE_PATH,
    hypotheses_path: str | Path = DEFAULT_HYPOTHESES_PATH,
    task_queue_path: str | Path = DEFAULT_TASK_QUEUE_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    worktree_root: str | Path = DEFAULT_WORKTREE_ROOT,
    now: datetime | None = None,
    force: bool = False,
    settings: Settings | None = None,
    ignore_lock: bool = False,
) -> dict[str, Any]:
    current = now or datetime.now().astimezone()
    report_folder = _resolve(report_dir)
    state, state_initialized = load_agent_state(_resolve(state_path))
    backlog, backlog_initialized = load_hypothesis_backlog(_resolve(hypotheses_path))
    operations_report = load_latest_operations_report(report_folder)
    research_report = _load_latest_json(report_folder, "research")
    queue = load_task_queue(_resolve(task_queue_path))
    git_state = inspect_git_state(PROJECT_ROOT)
    production = capture_production_guard(settings=settings)
    configured_guard = _load_configured_production_guard(report_folder)
    config_matches = not configured_guard or configured_guard.get("settings") == production.get(
        "settings"
    )
    freshness = assess_report_freshness(
        operations_report,
        research_report,
        now=current,
        max_age_minutes=AgentPolicy().report_freshness_minutes,
    )
    lock_active = DEFAULT_LOCK_PATH.exists() and not ignore_lock
    gate = assess_agent_gate(
        operations_report=operations_report,
        research_report=research_report,
        git_state=git_state,
        db_readable=bool(production.get("database_readable")),
        production_config_matches=config_matches,
        report_freshness=freshness,
        lock_active=lock_active,
    )
    audits = [
        inspect_research_worktree(
            item,
            project_root=PROJECT_ROOT,
            worktree_root=worktree_root,
            report_dir=report_folder,
        )
        for item in backlog.get("hypotheses", [])
        if item.get("worktree") or (Path(worktree_root) / str(item.get("hypothesis_id"))).exists()
    ]
    due = assess_agent_due(
        state=state,
        backlog=backlog,
        research_report=research_report,
        task_queue=queue,
        gate=gate,
        worktree_audits=audits,
        now=current,
        force=force,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "read_only_preflight",
        "ticker": ticker,
        "checked_at": current.isoformat(),
        "state_was_initialized": state_initialized,
        "backlog_was_initialized": backlog_initialized,
        "lock_active": lock_active,
        "operations_gate": gate.get("operations"),
        "research_gate": {
            "classification": gate.get("research_classification"),
            "fingerprint": gate.get("research_fingerprint"),
            "fresh": freshness.get("research_fresh"),
        },
        "main_git": git_state,
        "gate": gate,
        "due": due,
        "worktrees": audits,
        "budgets": AgentPolicy().as_dict(),
        "side_effects": {
            "codex_invoked": False,
            "worktree_created": False,
            "code_changed": False,
            "database_changed": False,
            "hypothesis_marked_running": False,
        },
    }


def build_compact_context_bundle(
    *,
    ticker: str,
    operations_report: dict[str, Any],
    research_report: dict[str, Any],
    backlog: dict[str, Any],
    selected_hypothesis: dict[str, Any],
    policy: AgentPolicy,
) -> dict[str, Any]:
    lifecycle = summarize_hypothesis_lifecycle(backlog)
    open_hypotheses = [
        {
            "hypothesis_id": item.get("hypothesis_id"),
            "fingerprint": item.get("fingerprint"),
            "status": item.get("status"),
            "category": item.get("category"),
            "primary_metric": item.get("primary_metric"),
        }
        for item in backlog.get("hypotheses", [])
        if item.get("status") in {"proposed", "queued", "monitoring", "approval_required"}
    ]
    live = research_report.get("live_metrics") or {}
    return {
        "schema_version": "autonomous_codex_context/v1.1",
        "project": "stock-ai",
        "ticker": ticker,
        "target": "next close direction / target_next_close_binary",
        "production_model_slot": "binary production model (read-only)",
        "operations_summary": {
            "cycle_completed_at": operations_report.get("cycle_completed_at"),
            "stop_reason": operations_report.get("stop_reason"),
            "health_summary": operations_report.get("health_summary") or {},
            "data_freshness": operations_report.get("data_freshness") or {},
        },
        "research_summary": {
            "cycle_completed_at": research_report.get("cycle_completed_at"),
            "classification": research_report.get("classification"),
            "experiment_fingerprint": research_report.get("experiment_fingerprint"),
            "recommendation": research_report.get("recommendation"),
        },
        "current_production_metrics": {
            key: live.get(key)
            for key in (
                "sample_count",
                "accuracy",
                "balanced_accuracy",
                "up_recall",
                "down_recall",
                "up_precision",
                "down_precision",
            )
        },
        "baseline_summary": research_report.get("baseline_summary")
        or research_report.get("baseline_comparison")
        or [],
        "shadow_summary": {
            key: (research_report.get("shadow_metrics") or {}).get(key)
            for key in ("status", "confirmed_count", "pending_count", "conclusions")
        },
        "rejected_hypothesis_fingerprints": lifecycle["rejected_fingerprints"],
        "open_hypotheses": open_hypotheses,
        "selected_hypothesis": selected_hypothesis,
        "success_criteria": [
            "improve the primary metric on train/validation-selected settings",
            "preserve stated guardrail metrics",
            "remain competitive with required simple baselines",
        ],
        "guardrail_metrics": selected_hypothesis.get("guardrail_metrics") or [],
        "cycle_budgets": policy.as_dict(),
        "permitted_scope": [
            "one selected offline research hypothesis",
            "existing features and offline reports",
            "worktree-only code and focused tests",
        ],
        "forbidden_scope": [
            "production model or config changes",
            "production SQLite or live/backfill writes",
            "daily_predict or Task Scheduler changes",
            "commit, merge, push, or same-cycle retuning",
        ],
        "relevant_paths": {
            "project_root": str(PROJECT_ROOT),
            "operations_module": str(PROJECT_ROOT / "src/stock_ai/operations/autonomous.py"),
            "research_module": str(PROJECT_ROOT / "src/stock_ai/operations/autonomous_research.py"),
            "agent_module": str(Path(__file__).resolve()),
            "prompt_template": str(DEFAULT_PROMPT_TEMPLATE),
        },
    }


def write_compact_context_bundle(
    bundle: dict[str, Any],
    *,
    context_dir: str | Path = DEFAULT_CONTEXT_DIR,
    now: datetime | None = None,
) -> dict[str, str]:
    current = now or datetime.now().astimezone()
    folder = _resolve(context_dir)
    folder.mkdir(parents=True, exist_ok=True)
    stem = f"codex_agent_context_{current:%Y-%m-%d_%H%M}"
    json_path = folder / f"{stem}.json"
    markdown_path = folder / f"{stem}.md"
    json_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_path.write_text(
        "# Codex Agent Compact Context\n\n```json\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    return {"json": str(json_path), "markdown": str(markdown_path)}


def cleanup_research_worktree(
    hypothesis: dict[str, Any],
    *,
    project_root: str | Path = PROJECT_ROOT,
    worktree_root: str | Path = DEFAULT_WORKTREE_ROOT,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    confirm: bool = False,
) -> dict[str, Any]:
    audit = inspect_research_worktree(
        hypothesis,
        project_root=project_root,
        worktree_root=worktree_root,
        report_dir=report_dir,
    )
    if audit.get("safe_to_remove"):
        initial_reason = "confirmation_required" if not confirm else "ready_for_confirmed_removal"
    else:
        initial_reason = "unsafe_to_remove"
    result = {
        "mode": "confirmed" if confirm else "dry_run",
        "audit": audit,
        "would_remove": bool(audit.get("safe_to_remove")),
        "removed": False,
        "reason": initial_reason,
    }
    if not confirm or not audit.get("safe_to_remove"):
        return result
    root = Path(project_root).resolve()
    target = Path(str(audit["path"])).resolve()
    worktree_root_path = Path(worktree_root).resolve()
    if target == root or worktree_root_path not in target.parents:
        result["reason"] = "target_outside_configured_worktree_root"
        return result
    removed = _git(root, "worktree", "remove", "--force", str(target))
    if removed["returncode"] != 0:
        result["reason"] = "git_worktree_remove_failed"
        result["error"] = removed["stderr"]
        return result
    branch = audit.get("branch")
    if branch:
        _git(root, "branch", "-D", str(branch))
    result.update({"removed": True, "reason": "removed_after_confirmed_safe_audit"})
    return result


def build_agent_prompt(
    *,
    template_path: str | Path,
    ticker: str,
    hypothesis: dict[str, Any],
    operations_report_path: str,
    research_report_path: str,
    worktree_path: str,
    max_runtime_minutes: float,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    context = {
        "ticker": ticker,
        "target": "next close direction / target_next_close_binary",
        "operations_report": operations_report_path,
        "research_report": research_report_path,
        "selected_hypothesis": hypothesis,
        "worktree_path": worktree_path,
        "max_runtime_minutes": max_runtime_minutes,
        "max_hypothesis_count": 1,
    }
    return template.rstrip() + "\n\n## Cycle Context\n\n```json\n" + json.dumps(
        context, ensure_ascii=False, indent=2
    ) + "\n```\n"


def build_codex_invocation(
    probe: dict[str, Any],
    *,
    worktree_path: str | Path,
    output_path: str | Path,
) -> list[str]:
    if not probe.get("available"):
        raise CodexAgentError(f"Codex CLI is unavailable: {probe.get('reason')}")
    return [
        str(probe["executable"]),
        "exec",
        "--cd",
        str(worktree_path),
        "--sandbox",
        "workspace-write",
        "--ask-for-approval",
        "never",
        "--output-last-message",
        str(output_path),
        "-",
    ]


def create_research_worktree(
    *,
    project_root: str | Path,
    worktree_root: str | Path,
    hypothesis_id: str,
    cycle_date: str,
) -> dict[str, Any]:
    root = Path(project_root)
    target = Path(worktree_root) / hypothesis_id
    branch = f"autonomous-research/{cycle_date}/{hypothesis_id}"
    tracked_database = _git(root, "ls-files", "--error-unmatch", "database/market.db")
    if tracked_database["returncode"] == 0:
        raise CodexAgentError(
            "Production database is Git-tracked and would be copied into the worktree."
        )
    if target.exists():
        raise CodexAgentError(f"Worktree path already exists: {target}")
    branch_check = _git(root, "show-ref", "--verify", f"refs/heads/{branch}")
    if branch_check["returncode"] == 0:
        raise CodexAgentError(f"Research branch already exists: {branch}")
    target.parent.mkdir(parents=True, exist_ok=True)
    result = _git(root, "worktree", "add", "-b", branch, str(target), "HEAD")
    if result["returncode"] != 0:
        raise CodexAgentError(f"Failed to create research worktree: {result['stderr']}")
    if (target / "database" / "market.db").exists():
        _git(root, "worktree", "remove", "--force", str(target))
        _git(root, "branch", "-D", branch)
        raise CodexAgentError("Production database appeared in the research worktree.")
    return {"path": str(target), "branch": branch, "created": True}


def validate_experiment_result(result: dict[str, Any]) -> dict[str, Any]:
    errors = []
    classification = str(result.get("classification") or "")
    if classification not in ALLOWED_CLASSIFICATIONS:
        errors.append("classification_invalid")
    periods = {
        str(row.get("name"))
        for row in result.get("evaluation_periods", [])
        if isinstance(row, dict)
    }
    missing_periods = sorted(REQUIRED_PERIODS - periods)
    if missing_periods:
        errors.append("required_evaluation_periods_missing")
    baselines = set((result.get("baseline_comparison") or {}).keys())
    missing_baselines = sorted(REQUIRED_BASELINES - baselines)
    if missing_baselines:
        errors.append("required_baselines_missing")
    tests = result.get("tests") or {}
    if not tests.get("passed"):
        errors.append("tests_not_passed")
    leakage = result.get("leakage_checks") or {}
    if leakage.get("status") != "passed":
        errors.append("leakage_check_not_passed")
    safety = result.get("safety") or {}
    safety_fields = {
        "main_working_tree_changed": False,
        "production_model_changed": False,
        "production_artifact_changed": False,
        "production_config_changed": False,
        "production_sqlite_changed": False,
        "existing_predictions_changed": False,
        "live_backfill_prediction_created": False,
        "automatic_commit_performed": False,
        "automatic_merge_performed": False,
        "task_scheduler_changed": False,
    }
    for field, expected in safety_fields.items():
        if safety.get(field) is not expected:
            errors.append(f"unsafe_{field}")
    human_review_ready = _human_review_gate(result) if not errors else False
    if classification == "human_review_recommended" and not human_review_ready:
        errors.append("human_review_gate_not_met")
    return {
        "valid": not errors,
        "errors": errors,
        "missing_periods": missing_periods,
        "missing_baselines": missing_baselines,
        "human_review_ready": human_review_ready,
    }


def run_autonomous_codex_research_agent(
    *,
    ticker: str = "7203.T",
    state_path: str | Path = DEFAULT_STATE_PATH,
    hypotheses_path: str | Path = DEFAULT_HYPOTHESES_PATH,
    task_queue_path: str | Path = DEFAULT_TASK_QUEUE_PATH,
    report_dir: str | Path = DEFAULT_REPORT_DIR,
    prompt_template_path: str | Path = DEFAULT_PROMPT_TEMPLATE,
    worktree_root: str | Path = DEFAULT_WORKTREE_ROOT,
    max_runtime_minutes: float = 90,
    codex_executable: str = "codex",
    dry_run: bool = False,
    force: bool = False,
    now: datetime | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    policy = AgentPolicy(max_runtime_minutes=max_runtime_minutes)
    policy.validate()
    budget = RuntimeBudget.start(max_runtime_minutes)
    cycle_started = now or datetime.now().astimezone()
    state_file = _resolve(state_path)
    backlog_file = _resolve(hypotheses_path)
    queue_file = _resolve(task_queue_path)
    report_folder = _resolve(report_dir)
    prompt_file = _resolve(prompt_template_path)
    worktree_folder = Path(worktree_root)
    if not worktree_folder.is_absolute():
        worktree_folder = PROJECT_ROOT / worktree_folder
    report_folder.mkdir(parents=True, exist_ok=True)

    with AgentFileLock(DEFAULT_LOCK_PATH):
        budget.check("load_state")
        state, state_initialized = load_agent_state(state_file)
        backlog, backlog_initialized = load_hypothesis_backlog(backlog_file)
        operations_report = load_latest_operations_report(report_folder)
        research_report = _load_latest_json(report_folder, "research")
        operations_path = str((operations_report.get("report_paths") or {}).get("json") or "")
        research_path = str((research_report.get("report_paths") or {}).get("json") or "")
        git_before = inspect_git_state(PROJECT_ROOT)
        production_before = capture_production_guard(settings=settings)
        configured_guard = _load_configured_production_guard(report_folder)
        config_matches = not configured_guard or configured_guard.get("settings") == production_before.get(
            "settings"
        )
        freshness = assess_report_freshness(
            operations_report,
            research_report,
            now=cycle_started,
            max_age_minutes=policy.report_freshness_minutes,
        )
        gate = assess_agent_gate(
            operations_report=operations_report,
            research_report=research_report,
            git_state=git_before,
            db_readable=bool(production_before.get("database_readable")),
            production_config_matches=config_matches,
            report_freshness=freshness,
        )
        queue = load_task_queue(queue_file)
        gate["latest_candidate_ranking"] = research_report.get("candidate_ranking") or []
        gate["open_model_research_tasks"] = _open_model_research_tasks(queue)
        lifecycle = summarize_hypothesis_lifecycle(backlog, state=state)
        worktree_audits = [
            inspect_research_worktree(
                item,
                project_root=PROJECT_ROOT,
                worktree_root=worktree_folder,
                report_dir=report_folder,
            )
            for item in backlog.get("hypotheses", [])
            if item.get("worktree") or (worktree_folder / str(item.get("hypothesis_id"))).exists()
        ]
        due = assess_agent_due(
            state=state,
            backlog=backlog,
            research_report=research_report,
            task_queue=queue,
            gate=gate,
            worktree_audits=worktree_audits,
            now=cycle_started,
            force=force,
        )
        codex_probe = {
            "available": False,
            "executable": codex_executable,
            "reason": "not_probed_before_due_gate",
        }
        selected = None
        selection = {"selected": None, "rejected_candidates": [], "reason": "gate_not_passed"}
        worktree = None
        invocation: list[str] = []
        invocation_result: dict[str, Any] = {}
        experiment_result: dict[str, Any] | None = None
        validation: dict[str, Any] | None = None
        changed_files: list[str] = []
        patch_path = None
        context_bundle_paths: dict[str, str] | None = None

        if due["classification"] != "agent_due_ready":
            classification = str(due["classification"])
            stop_reason = (
                _primary_gate_reason(gate["reasons"])
                if classification == "agent_skipped_unhealthy_gate"
                else classification
            )
            exit_code = 0
            task_change = {"changed": False, "task_id": None}
        else:
            selection = due["selection"]
            selected = selection["selected"]
            if selected is None:
                classification = "agent_skipped_no_safe_hypothesis"
                stop_reason = classification
                exit_code = 0
                task_change = {"changed": False, "task_id": None}
            elif dry_run:
                classification = "agent_dry_run_ready"
                stop_reason = "dry_run_before_worktree_creation"
                exit_code = 0
                task_change = {"changed": False, "task_id": None}
            else:
                codex_probe = probe_codex_cli(codex_executable)
                if not codex_probe.get("available"):
                    classification = "hypothesis_blocked"
                    stop_reason = str(codex_probe.get("reason") or "codex_cli_unavailable")
                    exit_code = 5
                    task_change = {"changed": False, "task_id": None}
                    _update_hypothesis(
                        backlog,
                        str(selected["hypothesis_id"]),
                        status="blocked",
                        result=classification,
                        next_action="run manually from a supported Codex environment",
                    )
                else:
                    context_bundle = build_compact_context_bundle(
                        ticker=ticker,
                        operations_report=operations_report,
                        research_report=research_report,
                        backlog=backlog,
                        selected_hypothesis=selected,
                        policy=policy,
                    )
                    context_bundle_paths = write_compact_context_bundle(
                        context_bundle,
                        context_dir=report_folder / "context",
                        now=cycle_started,
                    )
                    changed_files.extend(context_bundle_paths.values())
                    budget.check("create_worktree")
                    worktree = create_research_worktree(
                        project_root=PROJECT_ROOT,
                        worktree_root=worktree_folder,
                        hypothesis_id=str(selected["hypothesis_id"]),
                        cycle_date=cycle_started.strftime("%Y-%m-%d"),
                    )
                    _update_hypothesis(
                        backlog,
                        str(selected["hypothesis_id"]),
                        status="running",
                        selected_at=cycle_started.isoformat(),
                        worktree=worktree["path"],
                    )
                    prompt = build_agent_prompt(
                        template_path=prompt_file,
                        ticker=ticker,
                        hypothesis=selected,
                        operations_report_path=context_bundle_paths["json"],
                        research_report_path=context_bundle_paths["json"],
                        worktree_path=worktree["path"],
                        max_runtime_minutes=max_runtime_minutes,
                    )
                    prompt_output = report_folder / f"codex_agent_prompt_{cycle_started:%Y-%m-%d_%H%M}.md"
                    prompt_output.write_text(prompt, encoding="utf-8")
                    changed_files.append(str(prompt_output))
                    final_output = (
                        Path(worktree["path"])
                        / "reports"
                        / "autonomous"
                        / "codex_agent_final.md"
                    )
                    final_output.parent.mkdir(parents=True, exist_ok=True)
                    invocation = build_codex_invocation(
                        codex_probe,
                        worktree_path=worktree["path"],
                        output_path=final_output,
                    )
                    budget.check("invoke_codex")
                    invocation_result = _invoke_codex(
                        invocation,
                        prompt,
                        timeout_seconds=max(1, int(max_runtime_minutes * 60 - 30)),
                    )
                    result_path = (
                        Path(worktree["path"])
                        / "reports"
                        / "autonomous"
                        / "codex_agent_result.json"
                    )
                    if invocation_result["returncode"] == 0 and result_path.exists():
                        experiment_result = json.loads(result_path.read_text(encoding="utf-8"))
                        validation = validate_experiment_result(experiment_result)
                    else:
                        validation = {"valid": False, "errors": ["codex_result_missing_or_failed"]}
                    worktree_head = _git(Path(worktree["path"]), "rev-parse", "HEAD")
                    if worktree_head["stdout"].strip() != git_before.get("head"):
                        validation.setdefault("errors", []).append("automatic_commit_detected")
                        validation["valid"] = False
                    diff = _git(Path(worktree["path"]), "diff", "--binary")
                    worktree_status = _git(
                        Path(worktree["path"]),
                        "status",
                        "--porcelain=v1",
                        "--untracked-files=all",
                    )
                    changed_count = len(
                        [line for line in worktree_status["stdout"].splitlines() if line.strip()]
                    )
                    patch_line_count = len(diff["stdout"].splitlines())
                    variants = len((experiment_result or {}).get("experiment_variants") or [])
                    budget_errors = []
                    if changed_count > policy.max_changed_files:
                        budget_errors.append("max_changed_files_exceeded")
                    if patch_line_count > policy.max_patch_lines:
                        budget_errors.append("max_patch_lines_exceeded")
                    if variants > policy.max_experiment_variants:
                        budget_errors.append("max_experiment_variants_exceeded")
                    if budget_errors:
                        validation.setdefault("errors", []).extend(budget_errors)
                        validation["valid"] = False
                    validation["cycle_budget"] = {
                        "changed_files": changed_count,
                        "patch_lines": patch_line_count,
                        "experiment_variants": variants,
                        "implementation_attempts": 1,
                        "retuning_rounds_after_result": 0,
                    }
                    patch_folder = report_folder / "patches"
                    patch_folder.mkdir(parents=True, exist_ok=True)
                    patch_path = (
                        patch_folder
                        / f"{selected['hypothesis_id']}_{cycle_started:%Y-%m-%d_%H%M}.patch"
                    )
                    patch_path.write_text(diff["stdout"], encoding="utf-8")
                    changed_files.append(str(patch_path))
                    if validation.get("valid"):
                        classification = str(experiment_result["classification"])
                        stop_reason = classification
                        exit_code = 2 if validation.get("human_review_ready") else 0
                    else:
                        classification = "experiment_failed"
                        stop_reason = ";".join(validation.get("errors") or ["experiment_failed"])
                        exit_code = 1
                    status = _hypothesis_status(classification, validation or {})
                    _update_hypothesis(
                        backlog,
                        str(selected["hypothesis_id"]),
                        status=status,
                        report_path=str(result_path) if result_path.exists() else None,
                        patch_path=str(patch_path),
                        patch_sha256=_file_sha256(patch_path),
                        result=classification,
                        rejection_reason=stop_reason if status == "rejected" else None,
                        next_action="human review" if exit_code == 2 else "review report and patch",
                    )
                    if exit_code == 2:
                        queue, task_change = upsert_agent_task(
                            queue,
                            fingerprint=f"codex_agent_result:{selected['fingerprint']}",
                            title=f"Review Codex Research result: {selected['title']}",
                            description=classification,
                            status="approval_required",
                            cycle_at=cycle_started.isoformat(),
                        )
                    else:
                        task_change = {"changed": False, "task_id": None}

        budget.check("final_guards")
        production_after = capture_production_guard(settings=settings)
        git_after = inspect_git_state(PROJECT_ROOT)
        safety = compare_safety_guards(
            production_before,
            production_after,
            git_before=git_before,
            git_after=git_after,
        )
        if not safety["safe"]:
            classification = "unsafe_or_leakage_risk"
            stop_reason = "production_or_main_guard_changed"
            exit_code = 1

        cycle_completed = datetime.now().astimezone()
        report_paths = _next_report_paths(report_folder, cycle_started)
        result = {
            "schema_version": SCHEMA_VERSION,
            "cycle_started_at": cycle_started.isoformat(),
            "cycle_completed_at": cycle_completed.isoformat(),
            "duration_seconds": round((cycle_completed - cycle_started).total_seconds(), 3),
            "ticker": ticker,
            "target": "next_close_binary",
            "dry_run": dry_run,
            "force": force,
            "gate": gate,
            "due_gate": due,
            "hypothesis_lifecycle": lifecycle,
            "worktree_audits": worktree_audits,
            "cycle_budgets": policy.as_dict(),
            "codex_cli": codex_probe,
            "selected_hypothesis": selected,
            "selection": selection,
            "worktree": worktree,
            "codex_invocation": invocation,
            "codex_result": invocation_result,
            "experiment_result": experiment_result,
            "experiment_validation": validation,
            "classification": classification,
            "recommendation": _recommendation(classification),
            "human_approval_required": exit_code == 2,
            "patch_path": str(patch_path) if patch_path else None,
            "context_bundle": context_bundle_paths,
            "changed_files": changed_files,
            "task_queue_change": task_change,
            "safety": safety,
            "state_was_initialized": state_initialized,
            "backlog_was_initialized": backlog_initialized,
            "main_changes": False,
            "production_changes": False,
            "database_changes": False,
            "live_backfill_created": False,
            "automatic_commit_performed": False,
            "automatic_merge_performed": False,
            "task_scheduler_changed": False,
            "stop_reason": stop_reason,
            "exit_code": exit_code,
            "report_paths": {key: str(value) for key, value in report_paths.items()},
        }
        report_paths["json"].write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        report_paths["markdown"].write_text(render_agent_report(result), encoding="utf-8")
        result["changed_files"].extend([str(report_paths["markdown"]), str(report_paths["json"])])

        if not dry_run:
            _write_json(backlog_file, {**backlog, "updated_at": cycle_completed.isoformat()})
            if task_change.get("changed"):
                _write_json(queue_file, queue)
            updated_state = _updated_state(
                state,
                result,
                backlog,
                report_paths["markdown"],
                operations_path,
                research_path,
            )
            _write_json(state_file, updated_state)
            result["changed_files"].extend([str(backlog_file), str(state_file)])
            if task_change.get("changed"):
                result["changed_files"].append(str(queue_file))
            report_paths["json"].write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            report_paths["markdown"].write_text(render_agent_report(result), encoding="utf-8")
        return result


def assess_report_freshness(
    operations_report: dict[str, Any],
    research_report: dict[str, Any],
    *,
    now: datetime,
    max_age_minutes: float,
) -> dict[str, Any]:
    operation_time = _parse_datetime(operations_report.get("cycle_completed_at"))
    research_time = _parse_datetime(
        research_report.get("cycle_completed_at") or research_report.get("cycle_started_at")
    )
    return {
        "operations_cycle_at": operation_time.isoformat() if operation_time else None,
        "research_cycle_at": research_time.isoformat() if research_time else None,
        "operations_age_minutes": _age_minutes(now, operation_time),
        "research_age_minutes": _age_minutes(now, research_time),
        "operations_fresh": _is_fresh(now, operation_time, max_age_minutes),
        "research_fresh": _is_fresh(now, research_time, max_age_minutes),
        "max_age_minutes": max_age_minutes,
    }


def capture_production_guard(*, settings: Settings | None = None) -> dict[str, Any]:
    config = settings or load_settings()
    database_path = config.paths.database_path
    counts = {}
    database_readable = False
    database_integrity = None
    if database_path.exists():
        try:
            connection = sqlite3.connect(f"file:{database_path}?mode=ro", uri=True)
            try:
                database_integrity = connection.execute("PRAGMA quick_check").fetchone()[0]
                for table in ("predictions", "prediction_runs", "prediction_records"):
                    exists = connection.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
                    ).fetchone()
                    counts[table] = (
                        int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                        if exists
                        else None
                    )
                database_readable = database_integrity == "ok"
            finally:
                connection.close()
        except sqlite3.Error:
            database_readable = False
    return {
        "database": _file_sha256(database_path),
        "database_readable": database_readable,
        "database_integrity": database_integrity,
        "database_counts": counts,
        "binary_artifact": _file_sha256(
            PROJECT_ROOT / "models" / "artifacts" / "lightgbm_binary_7203.T.joblib"
        ),
        "multiclass_artifact": _file_sha256(
            PROJECT_ROOT / "models" / "artifacts" / "lightgbm_multiclass_7203.T.joblib"
        ),
        "settings": _file_sha256(PROJECT_ROOT / "configs" / "settings.yml"),
        "daily_predict": _file_sha256(PROJECT_ROOT / "scripts" / "daily_predict.ps1"),
    }


def compare_safety_guards(
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    git_before: dict[str, Any],
    git_after: dict[str, Any],
) -> dict[str, Any]:
    changed = [key for key in before if before.get(key) != after.get(key)]
    git_status_changed = git_before.get("status_lines") != git_after.get("status_lines")
    git_head_changed = git_before.get("head") != git_after.get("head")
    return {
        "safe": not changed and not git_status_changed and not git_head_changed,
        "production_guard_changes": changed,
        "main_git_status_changed": git_status_changed,
        "main_git_head_changed": git_head_changed,
        "before": before,
        "after": after,
    }


def upsert_agent_task(
    queue: dict[str, Any],
    *,
    fingerprint: str,
    title: str,
    description: str,
    status: str,
    cycle_at: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = [dict(task) for task in queue.get("tasks", []) if isinstance(task, dict)]
    existing = next((task for task in tasks if task.get("fingerprint") == fingerprint), None)
    created = existing is None
    if existing is None:
        existing = {
            "task_id": "codex_agent_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16],
            "fingerprint": fingerprint,
            "created_at": cycle_at,
        }
        tasks.append(existing)
    existing.update(
        {
            "last_seen_at": cycle_at,
            "category": "codex_research_agent",
            "risk_level": "Level 2" if status == "approval_required" else "Level 1",
            "priority": "medium",
            "title": title,
            "description": description,
            "status": status,
            "human_approval_required": status == "approval_required",
            "source": "autonomous_codex_research_agent",
        }
    )
    return (
        {"schema_version": queue.get("schema_version") or SCHEMA_VERSION, "updated_at": cycle_at, "tasks": tasks},
        {"changed": True, "created": created, "task_id": existing["task_id"]},
    )


def render_agent_report(result: dict[str, Any]) -> str:
    gate = result.get("gate") or {}
    selected = result.get("selected_hypothesis") or {}
    safety = result.get("safety") or {}
    lines = [
        "# Autonomous Codex Research Agent v1",
        "",
        f"- Cycle start: {result.get('cycle_started_at')}",
        f"- Cycle end: {result.get('cycle_completed_at')}",
        f"- Ticker / target: {result.get('ticker')} / {result.get('target')}",
        f"- Gate healthy: {gate.get('healthy')}",
        f"- Gate reasons: {', '.join(gate.get('reasons') or []) or 'none'}",
        f"- Due classification: {(result.get('due_gate') or {}).get('classification')}",
        f"- Due reasons: {json.dumps((result.get('due_gate') or {}).get('reason_codes') or [], ensure_ascii=False)}",
        f"- Selected hypothesis: {selected.get('hypothesis_id') or 'none'}",
        f"- Classification: {result.get('classification')}",
        f"- Recommendation: {result.get('recommendation')}",
        f"- Stop reason: {result.get('stop_reason')}",
        f"- Exit code: {result.get('exit_code')}",
        "",
        "## Isolation And Output",
        "",
        f"- Worktree: {(result.get('worktree') or {}).get('path') or 'not created'}",
        f"- Changed files: {json.dumps(result.get('changed_files') or [], ensure_ascii=False)}",
        f"- Patch / diff: {result.get('patch_path') or 'not created'}",
        f"- Compact context: {json.dumps(result.get('context_bundle'), ensure_ascii=False)}",
        f"- Cycle budgets: {json.dumps(result.get('cycle_budgets') or {}, ensure_ascii=False)}",
        f"- Codex CLI: {json.dumps(result.get('codex_cli') or {}, ensure_ascii=False)}",
        "",
        "## Safety Contract",
        "",
        f"- Main working tree changed: {'Yes' if safety.get('main_git_status_changed') or safety.get('main_git_head_changed') else 'No'}",
        f"- Production model changed: {'Yes' if 'binary_artifact' in (safety.get('production_guard_changes') or []) else 'No'}",
        f"- Production artifact changed: {'Yes' if any(key.endswith('artifact') for key in safety.get('production_guard_changes') or []) else 'No'}",
        f"- Production config changed: {'Yes' if 'settings' in (safety.get('production_guard_changes') or []) else 'No'}",
        f"- Production SQLite changed: {'Yes' if 'database' in (safety.get('production_guard_changes') or []) else 'No'}",
        f"- Existing predictions changed: {'Yes' if 'database_counts' in (safety.get('production_guard_changes') or []) else 'No'}",
        "- Live/backfill prediction created: No",
        "- Automatic commit performed: No",
        "- Automatic merge performed: No",
        "- Task Scheduler changed: No",
        "",
        "## Selection And Evaluation",
        "",
        "```json",
        json.dumps(result.get("selection") or {}, ensure_ascii=False, indent=2),
        "```",
        "",
        "```json",
        json.dumps(result.get("experiment_result") or {}, ensure_ascii=False, indent=2),
        "```",
    ]
    return "\n".join(lines) + "\n"


def _load_latest_json(folder: Path, prefix: str) -> dict[str, Any]:
    pattern = re.compile(rf"{re.escape(prefix)}_\d{{4}}-\d{{2}}-\d{{2}}_\d{{4}}(?:_\d+)?\.json$")
    files = sorted(
        (path for path in folder.glob(f"{prefix}_*.json") if pattern.fullmatch(path.name)),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return {}
    try:
        payload = json.loads(files[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    payload.setdefault("report_paths", {})["json"] = str(files[0])
    return payload


def _open_model_research_tasks(queue: dict[str, Any]) -> list[dict[str, Any]]:
    closed = {"resolved", "closed", "completed", "rejected"}
    return [
        {
            "task_id": task.get("task_id"),
            "status": task.get("status"),
            "fingerprint": task.get("fingerprint"),
            "title": task.get("title"),
        }
        for task in queue.get("tasks", [])
        if isinstance(task, dict)
        and task.get("category") == "model_research"
        and task.get("status") not in closed
    ]


def _new_approval_model_research_tasks(
    queue: dict[str, Any],
    *,
    after: datetime | None,
) -> list[dict[str, Any]]:
    closed = {"resolved", "closed", "completed", "rejected", "archived"}
    tasks = []
    for task in queue.get("tasks", []):
        if not isinstance(task, dict) or task.get("category") != "model_research":
            continue
        if str(task.get("status") or "") in closed or not task.get("requires_human_approval"):
            continue
        created = _parse_datetime(task.get("created_at"))
        if after and created and created.astimezone(UTC) <= after.astimezone(UTC):
            continue
        tasks.append(
            {
                "task_id": task.get("task_id"),
                "fingerprint": task.get("fingerprint"),
                "status": task.get("status"),
                "created_at": task.get("created_at"),
                "title": task.get("title"),
            }
        )
    return tasks


def _load_configured_production_guard(report_folder: Path) -> dict[str, Any]:
    state_path = report_folder / "research_state.json"
    if not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return state.get("production_guard") or {}


def _update_hypothesis(backlog: dict[str, Any], hypothesis_id: str, **updates: Any) -> None:
    for item in backlog.get("hypotheses", []):
        if item.get("hypothesis_id") == hypothesis_id:
            item.update({key: value for key, value in updates.items() if value is not None})
            item["updated_at"] = _now_iso()
            return


def _registered_worktrees(project_root: Path) -> dict[str, dict[str, Any]]:
    result = _git(project_root, "worktree", "list", "--porcelain")
    if result["returncode"] != 0:
        return {}
    entries: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] = {}
    for line in [*result["stdout"].splitlines(), ""]:
        if not line.strip():
            if current.get("path"):
                entries[str(Path(current["path"]).resolve()).casefold()] = current
            current = {}
            continue
        key, _, value = line.partition(" ")
        if key == "worktree":
            current["path"] = value
        elif key == "HEAD":
            current["head"] = value
        elif key == "branch":
            current["branch"] = value.removeprefix("refs/heads/")
        else:
            current[key] = value or True
    return entries


def _existing_artifact_path(value: Any) -> Path | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve() if path.exists() else None


def _latest_matching_path(folder: Path, pattern: str) -> Path | None:
    if not folder.exists():
        return None
    paths = sorted(folder.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return paths[0].resolve() if paths else None


def _worktree_processes(path: Path) -> list[dict[str, Any]]:
    if os.name != "nt":
        return []
    command = [
        "powershell.exe",
        "-NoProfile",
        "-Command",
        "Get-CimInstance Win32_Process | Select-Object ProcessId,ParentProcessId,CommandLine "
        "| ConvertTo-Json -Compress",
    ]
    result = _subprocess(command, timeout=20)
    if result["returncode"] != 0 or not result["stdout"].strip():
        return []
    try:
        parsed = json.loads(result["stdout"])
    except json.JSONDecodeError:
        return []
    processes = parsed if isinstance(parsed, list) else [parsed]
    by_pid = {
        int(item.get("ProcessId")): item
        for item in processes
        if isinstance(item, dict) and item.get("ProcessId") is not None
    }
    excluded = {os.getpid()}
    parent = os.getppid()
    while parent and parent not in excluded:
        excluded.add(parent)
        parent = int((by_pid.get(parent) or {}).get("ParentProcessId") or 0)
    needle = str(path).casefold()
    return [
        {
            "pid": int(item.get("ProcessId")),
            "command_line": item.get("CommandLine"),
        }
        for item in processes
        if isinstance(item, dict)
        and item.get("ProcessId") not in excluded
        and needle in str(item.get("CommandLine") or "").casefold()
    ]


def _safe_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _normalized_newlines(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _updated_state(
    state: dict[str, Any],
    result: dict[str, Any],
    backlog: dict[str, Any],
    report_path: Path,
    operations_path: str,
    research_path: str,
) -> dict[str, Any]:
    statuses = [str(item.get("status")) for item in backlog.get("hypotheses", [])]
    exit_code = int(result.get("exit_code") or 0)
    technical_failure = exit_code not in {0, 2, 3, 4, 5}
    successful_cycle = (
        exit_code in {0, 2} and result.get("classification") not in NORMAL_SKIP_CLASSIFICATIONS
    )
    previous_success = state.get("last_successful_cycle_at")
    if (
        state.get("last_result") == "agent_skipped_gate"
        and previous_success == state.get("last_cycle_at")
    ):
        previous_success = None
    lifecycle = summarize_hypothesis_lifecycle(backlog, state=state)
    due = result.get("due_gate") or {}
    selected = result.get("selected_hypothesis") or {}
    heavy_executed = bool(result.get("codex_invocation"))
    last_heavy_at = result["cycle_completed_at"] if heavy_executed else state.get(
        "last_heavy_cycle_at"
    )
    return {
        **state,
        "schema_version": SCHEMA_VERSION,
        "last_cycle_at": result["cycle_completed_at"],
        "last_successful_cycle_at": (
            result["cycle_completed_at"] if successful_cycle else previous_success
        ),
        "last_hypothesis_id": (result.get("selected_hypothesis") or {}).get("hypothesis_id"),
        "last_worktree": (result.get("worktree") or {}).get("path"),
        "last_report": str(report_path),
        "last_result": result.get("classification"),
        "last_operations_report": operations_path or None,
        "last_research_report": research_path or None,
        "open_hypotheses": sum(status in {"proposed", "queued", "running", "monitoring"} for status in statuses),
        "completed_hypotheses": statuses.count("completed"),
        "rejected_hypotheses": statuses.count("rejected"),
        "active_worktree": (result.get("worktree") or {}).get("path") if statuses.count("running") else None,
        "consecutive_failures": (
            int(state.get("consecutive_failures") or 0) + 1 if technical_failure else 0
        ),
        "next_due_reason": _next_due_reason(result),
        "last_preflight_at": result["cycle_completed_at"],
        "last_heavy_cycle_at": last_heavy_at,
        "last_heavy_cycle_token_usage": (
            (result.get("codex_result") or {}).get("token_usage")
            if heavy_executed
            else state.get("last_heavy_cycle_token_usage")
        ),
        "last_heavy_cycle_runtime_minutes": (
            round(float(result.get("duration_seconds") or 0) / 60, 3)
            if heavy_executed
            else state.get("last_heavy_cycle_runtime_minutes")
        ),
        "last_research_fingerprint": (
            due.get("latest_research_fingerprint")
            if heavy_executed
            else state.get("last_research_fingerprint")
        ),
        "last_live_confirmed_count": (
            due.get("live_confirmed_count")
            if heavy_executed
            else state.get("last_live_confirmed_count", 0)
        ),
        "last_shadow_confirmed_count": (
            due.get("shadow_confirmed_count")
            if heavy_executed
            else state.get("last_shadow_confirmed_count", 0)
        ),
        "last_selected_hypothesis_fingerprint": (
            selected.get("fingerprint")
            or state.get("last_selected_hypothesis_fingerprint")
        ),
        "completed_hypothesis_fingerprints": lifecycle["completed_fingerprints"],
        "rejected_hypothesis_fingerprints": lifecycle["rejected_fingerprints"],
        "worktree_status": next(
            (
                audit.get("lifecycle_status")
                for audit in result.get("worktree_audits", [])
                if audit.get("path") == state.get("active_worktree")
                or audit.get("hypothesis_id") == state.get("last_hypothesis_id")
            ),
            state.get("worktree_status"),
        ),
        "next_due_at": due.get("next_due_at") or state.get("next_due_at"),
        "next_due_reasons": due.get("reason_codes") or [],
        "last_context_bundle": result.get("context_bundle") or state.get("last_context_bundle"),
        "last_skip_classification": (
            result.get("classification")
            if result.get("classification") in NORMAL_SKIP_CLASSIFICATIONS
            else None
        ),
    }


def _human_review_gate(result: dict[str, Any]) -> bool:
    comparison = result.get("production_comparison") or {}
    consistency = result.get("window_consistency") or {}
    leakage = result.get("leakage_checks") or {}
    sample = int(result.get("sample_count") or 0)
    return bool(
        float(comparison.get("balanced_accuracy_delta") or 0) > 0
        and float(comparison.get("primary_metric_delta") or 0) > 0
        and float(comparison.get("down_recall_delta") or 0) >= -0.05
        and bool(comparison.get("beats_all_required_baselines"))
        and int(consistency.get("improving_period_count") or 0) >= 2
        and not bool(consistency.get("recent_only_gain"))
        and leakage.get("status") == "passed"
        and sample >= 100
    )


def _hypothesis_status(classification: str, validation: dict[str, Any]) -> str:
    if not validation.get("valid") or classification in {"experiment_failed", "unsafe_or_leakage_risk"}:
        return "rejected"
    if classification == "human_review_recommended":
        return "approval_required"
    if classification in {"continue_shadow_monitoring", "promising_offline_only", "promising_recent_period", "promising_live_like"}:
        return "monitoring"
    if classification in {"hypothesis_rejected", "no_material_change"}:
        return "rejected"
    return "completed"


def _primary_gate_reason(reasons: Sequence[str]) -> str:
    priority = [
        "main_git_head_missing",
        "main_worktree_dirty",
        "main_not_git_repository",
        "operations_report_stale",
        "research_report_stale",
        "production_config_unexpected_change",
        "production_database_unreadable",
    ]
    return next((reason for reason in priority if reason in reasons), reasons[0] if reasons else "gate_unhealthy")


def _recommendation(classification: str) -> str:
    if classification == "agent_skipped_unhealthy_gate":
        return "Resolve the gate finding before creating a research worktree."
    if classification == "agent_skipped_worktree_review_required":
        return "Review or clean the existing research worktree before another heavy cycle."
    if classification in NORMAL_SKIP_CLASSIFICATIONS:
        return "No heavy Codex cycle is required; keep the current production system unchanged."
    if classification == "hypothesis_blocked":
        return "Run the generated workflow manually from a supported Codex environment."
    if classification == "agent_dry_run_ready":
        return "Review the selected hypothesis; no worktree or Codex task was created."
    if classification == "human_review_recommended":
        return "Human review is required; no production promotion was performed."
    return "Review the report and patch; production remains unchanged."


def _next_due_reason(result: dict[str, Any]) -> str:
    if result.get("classification") == "agent_skipped_unhealthy_gate":
        return "after_gate_recovery"
    if result.get("classification") == "hypothesis_blocked":
        return "after_codex_cli_available"
    return "human_review_or_next_safe_hypothesis"


def _next_report_paths(folder: Path, now: datetime) -> dict[str, Path]:
    stem = f"codex_agent_{now:%Y-%m-%d_%H%M}"
    suffix = 1
    while (folder / f"{stem}.json").exists() or (folder / f"{stem}.md").exists():
        suffix += 1
        stem = f"codex_agent_{now:%Y-%m-%d_%H%M}_{suffix}"
    return {"json": folder / f"{stem}.json", "markdown": folder / f"{stem}.md"}


def _invoke_codex(command: list[str], prompt: str, *, timeout_seconds: int) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise CodexAgentTimeoutError("Codex CLI invocation timed out.") from error
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _subprocess(command: list[str], *, timeout: int) -> dict[str, Any]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "stdout": "", "stderr": str(error)}
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _git(root: Path, *args: str) -> dict[str, Any]:
    return _subprocess(["git", "-C", str(root), *args], timeout=30)


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.astimezone()
    return parsed


def _age_minutes(now: datetime, then: datetime | None) -> float | None:
    if then is None:
        return None
    comparable_now = now if now.tzinfo else now.astimezone()
    comparable_then = then if then.tzinfo else then.astimezone()
    return round((comparable_now - comparable_then).total_seconds() / 60, 3)


def _is_fresh(now: datetime, then: datetime | None, max_age_minutes: float) -> bool:
    age = _age_minutes(now, then)
    return bool(then and then.date() == now.date() and age is not None and -5 <= age <= max_age_minutes)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _file_sha256(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(UTC).astimezone().isoformat()
