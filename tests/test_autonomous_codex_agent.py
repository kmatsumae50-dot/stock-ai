from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time

import pytest

from stock_ai.operations import autonomous_codex_agent as agent


def _healthy_operations(cycle_at: str = "2026-07-26T11:20:00+09:00") -> dict:
    return {
        "cycle_completed_at": cycle_at,
        "stop_reason": "completed_read_only_cycle",
        "health_summary": {"action_required_count": 0, "unknown_source_count": 0},
        "unknown_source_findings": [],
        "missing_live_findings": [],
        "data_freshness": {"freshness_status": "OK", "stale_at_run_time": False},
        "database_validation": {"readable": True},
        "report_paths": {"json": "operations.json"},
    }


def _healthy_research(cycle_at: str = "2026-07-26T12:00:00+09:00") -> dict:
    return {
        "cycle_started_at": cycle_at,
        "cycle_completed_at": cycle_at,
        "classification": "research_skipped_duplicate_experiment",
        "experiment_fingerprint": "abc123",
        "candidate_ranking": [
            {"candidate_name": "scale_pos_weight_medium", "rank": 1},
            {"candidate_name": "class_weight_balanced", "rank": 2},
        ],
        "report_paths": {"json": "research.json"},
    }


def _valid_experiment_result() -> dict:
    return {
        "classification": "promising_offline_only",
        "sample_count": 845,
        "evaluation_periods": [{"name": name} for name in sorted(agent.REQUIRED_PERIODS)],
        "baseline_comparison": {name: {"delta": 0.01} for name in agent.REQUIRED_BASELINES},
        "production_comparison": {
            "balanced_accuracy_delta": 0.02,
            "primary_metric_delta": 0.04,
            "down_recall_delta": -0.01,
            "beats_all_required_baselines": True,
        },
        "window_consistency": {"improving_period_count": 3, "recent_only_gain": False},
        "tests": {"passed": True},
        "leakage_checks": {"status": "passed", "test_fold_threshold_selection": False},
        "safety": {
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
        },
    }


def _write_cycle_reports(report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "operations_2026-07-26_1120.json").write_text(
        json_dumps(_healthy_operations()), encoding="utf-8"
    )
    (report_dir / "research_2026-07-26_1200.json").write_text(
        json_dumps(_healthy_research()), encoding="utf-8"
    )


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def test_hypothesis_backlog_has_required_schema_and_deterministic_fingerprints() -> None:
    first = agent.seed_hypothesis_backlog("2026-07-26T00:00:00+09:00")
    second = agent.seed_hypothesis_backlog("2026-07-26T01:00:00+09:00")

    assert len(first["hypotheses"]) == 7
    assert [item["fingerprint"] for item in first["hypotheses"]] == [
        item["fingerprint"] for item in second["hypotheses"]
    ]
    required = {
        "hypothesis_id",
        "fingerprint",
        "created_at",
        "updated_at",
        "title",
        "category",
        "rationale",
        "evidence",
        "expected_benefit",
        "primary_metric",
        "guardrail_metrics",
        "implementation_scope",
        "risk_level",
        "estimated_runtime_minutes",
        "status",
        "selected_at",
        "worktree",
        "report_path",
        "result",
        "rejection_reason",
        "next_action",
    }
    assert required <= first["hypotheses"][0].keys()


def test_hypothesis_deduplication_uses_fingerprint() -> None:
    item = agent.seed_hypothesis_backlog()["hypotheses"][0]
    duplicate = {**item, "hypothesis_id": "renamed", "created_at": "later"}

    deduplicated = agent.deduplicate_hypotheses([item, duplicate])

    assert len(deduplicated) == 1
    assert deduplicated[0]["hypothesis_id"] == item["hypothesis_id"]


def test_selects_only_one_safe_hypothesis_and_records_rejections() -> None:
    backlog = agent.seed_hypothesis_backlog()

    result = agent.select_hypothesis(backlog, research_report=_healthy_research())

    assert result["selected"]["hypothesis_id"] == "balanced_threshold_guardrail"
    assert len(result["rejected_candidates"]) == len(backlog["hypotheses"]) - 1
    assert result["success_conditions"]
    assert result["stop_conditions"]


def test_no_safe_hypothesis_is_normal() -> None:
    backlog = agent.seed_hypothesis_backlog()
    for item in backlog["hypotheses"]:
        item["status"] = "rejected"

    result = agent.select_hypothesis(backlog, research_report=_healthy_research())

    assert result["selected"] is None
    assert result["reason"] == "no_safe_hypothesis_available"


def test_unhealthy_operations_gate_stops_before_worktree() -> None:
    operations = _healthy_operations()
    operations["health_summary"]["action_required_count"] = 1

    gate = agent.assess_agent_gate(
        operations_report=operations,
        research_report=_healthy_research(),
        git_state={"is_repository": True, "has_head": True, "clean": True},
        db_readable=True,
        production_config_matches=True,
    )

    assert gate["healthy"] is False
    assert "operations_action_required" in gate["reasons"]


@pytest.mark.parametrize("reason_field", ["has_head", "clean"])
def test_unborn_or_dirty_main_stops_gate(reason_field: str) -> None:
    git_state = {"is_repository": True, "has_head": True, "clean": True}
    git_state[reason_field] = False

    gate = agent.assess_agent_gate(
        operations_report=_healthy_operations(),
        research_report=_healthy_research(),
        git_state=git_state,
        db_readable=True,
        production_config_matches=True,
    )

    assert gate["healthy"] is False
    expected = "main_git_head_missing" if reason_field == "has_head" else "main_worktree_dirty"
    assert expected in gate["reasons"]


def test_stale_report_is_not_healthy() -> None:
    freshness = agent.assess_report_freshness(
        _healthy_operations("2026-07-25T11:20:00+09:00"),
        _healthy_research("2026-07-25T12:00:00+09:00"),
        now=datetime.fromisoformat("2026-07-26T12:10:00+09:00"),
        max_age_minutes=240,
    )

    assert freshness["operations_fresh"] is False
    assert freshness["research_fresh"] is False


def test_codex_cli_unavailable_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent.shutil, "which", lambda value: None)

    result = agent.probe_codex_cli("codex")

    assert result == {"available": False, "executable": "codex", "reason": "codex_cli_not_found"}


def test_codex_invocation_requires_verified_safe_flags() -> None:
    with pytest.raises(agent.CodexAgentError, match="unavailable"):
        agent.build_codex_invocation(
            {"available": False, "reason": "help failed"},
            worktree_path="worktree",
            output_path="result.md",
        )


def test_prompt_contains_safety_and_single_hypothesis_contract(tmp_path: Path) -> None:
    template = tmp_path / "prompt.md"
    template.write_text(
        "production DB禁止 merge禁止 max hypothesis count = 1 required periods",
        encoding="utf-8",
    )
    hypothesis = agent.seed_hypothesis_backlog()["hypotheses"][0]

    prompt = agent.build_agent_prompt(
        template_path=template,
        ticker="7203.T",
        hypothesis=hypothesis,
        operations_report_path="ops.json",
        research_report_path="research.json",
        worktree_path="C:/worktree",
        max_runtime_minutes=180,
    )

    assert "7203.T" in prompt
    assert "next close direction" in prompt
    assert hypothesis["hypothesis_id"] in prompt
    assert '"max_hypothesis_count": 1' in prompt
    assert '"max_runtime_minutes": 180' in prompt


def test_experiment_validation_requires_periods_baselines_tests_and_leakage() -> None:
    result = _valid_experiment_result()

    validation = agent.validate_experiment_result(result)

    assert validation["valid"] is True
    assert validation["missing_periods"] == []
    assert validation["missing_baselines"] == []


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda value: value.update(evaluation_periods=[]), "required_evaluation_periods_missing"),
        (lambda value: value.update(baseline_comparison={}), "required_baselines_missing"),
        (lambda value: value.update(tests={"passed": False}), "tests_not_passed"),
        (lambda value: value.update(leakage_checks={"status": "failed"}), "leakage_check_not_passed"),
    ],
)
def test_invalid_experiment_evidence_stops_result(mutation, expected: str) -> None:
    result = _valid_experiment_result()
    mutation(result)

    validation = agent.validate_experiment_result(result)

    assert validation["valid"] is False
    assert expected in validation["errors"]


def test_unsafe_production_or_prediction_change_rejects_result() -> None:
    result = _valid_experiment_result()
    result["safety"]["production_sqlite_changed"] = True
    result["safety"]["live_backfill_prediction_created"] = True

    validation = agent.validate_experiment_result(result)

    assert validation["valid"] is False
    assert "unsafe_production_sqlite_changed" in validation["errors"]
    assert "unsafe_live_backfill_prediction_created" in validation["errors"]


def test_human_review_requires_full_evidence_gate() -> None:
    result = _valid_experiment_result()
    result["classification"] = "human_review_recommended"

    assert agent.validate_experiment_result(result)["human_review_ready"] is True

    result["production_comparison"]["down_recall_delta"] = -0.2
    validation = agent.validate_experiment_result(result)
    assert validation["valid"] is False
    assert "human_review_gate_not_met" in validation["errors"]


def test_task_queue_upsert_does_not_duplicate_fingerprint() -> None:
    queue = {"schema_version": "x", "tasks": []}
    queue, first = agent.upsert_agent_task(
        queue,
        fingerprint="same",
        title="gate",
        description="first",
        status="review_needed",
        cycle_at="2026-07-26T00:00:00+09:00",
    )
    queue, second = agent.upsert_agent_task(
        queue,
        fingerprint="same",
        title="gate",
        description="second",
        status="review_needed",
        cycle_at="2026-07-26T01:00:00+09:00",
    )

    assert len(queue["tasks"]) == 1
    assert first["created"] is True
    assert second["created"] is False
    assert queue["tasks"][0]["description"] == "second"


def test_open_model_research_tasks_excludes_resolved_items() -> None:
    queue = {
        "tasks": [
            {"task_id": "open", "category": "model_research", "status": "review_needed"},
            {"task_id": "done", "category": "model_research", "status": "resolved"},
            {"task_id": "other", "category": "news", "status": "review_needed"},
        ]
    }

    tasks = agent._open_model_research_tasks(queue)

    assert [task["task_id"] for task in tasks] == ["open"]


def test_runtime_budget_times_out() -> None:
    budget = agent.RuntimeBudget(started=time.monotonic() - 120, max_minutes=1)

    with pytest.raises(agent.CodexAgentTimeoutError):
        budget.check("test")


def test_lock_prevents_second_cycle(tmp_path: Path) -> None:
    lock = tmp_path / "agent.lock"

    with agent.AgentFileLock(lock):
        with pytest.raises(agent.CodexAgentLockError):
            with agent.AgentFileLock(lock):
                pass
    assert not lock.exists()


def test_worktree_creation_isolated_from_clean_main(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    worktrees = tmp_path / "worktrees"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "README.md").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True, capture_output=True)
    head_before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()

    result = agent.create_research_worktree(
        project_root=repository,
        worktree_root=worktrees,
        hypothesis_id="test-hypothesis",
        cycle_date="2026-07-26",
    )

    assert Path(result["path"]).exists()
    assert result["branch"] == "autonomous-research/2026-07-26/test-hypothesis"
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip() == head_before
    assert subprocess.check_output(["git", "status", "--porcelain"], cwd=repository, text=True) == ""


def test_worktree_refuses_git_tracked_production_database(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "database").mkdir()
    (repository / "database" / "market.db").write_bytes(b"not production")
    subprocess.run(["git", "add", "."], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True, capture_output=True)

    with pytest.raises(agent.CodexAgentError, match="would be copied"):
        agent.create_research_worktree(
            project_root=repository,
            worktree_root=tmp_path / "worktrees",
            hypothesis_id="unsafe-db",
            cycle_date="2026-07-26",
        )


def test_safety_guard_detects_db_artifact_and_main_changes() -> None:
    before = {"database": "a", "database_counts": {"predictions": 1}, "binary_artifact": "b"}
    after = {"database": "x", "database_counts": {"predictions": 2}, "binary_artifact": "b"}

    result = agent.compare_safety_guards(
        before,
        after,
        git_before={"head": "1", "status_lines": []},
        git_after={"head": "1", "status_lines": [" M file"]},
    )

    assert result["safe"] is False
    assert "database" in result["production_guard_changes"]
    assert "database_counts" in result["production_guard_changes"]
    assert result["main_git_status_changed"] is True


def test_full_cycle_dirty_main_writes_skipped_state_without_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir = tmp_path / "reports"
    _write_cycle_reports(report_dir)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("safe template", encoding="utf-8")
    guard = {
        "database": "db",
        "database_readable": True,
        "database_counts": {"predictions": 1},
        "binary_artifact": "binary",
        "multiclass_artifact": "multi",
        "settings": "settings",
        "daily_predict": "daily",
    }
    monkeypatch.setattr(agent, "DEFAULT_LOCK_PATH", tmp_path / "agent.lock")
    monkeypatch.setattr(agent, "capture_production_guard", lambda **kwargs: dict(guard))
    monkeypatch.setattr(
        agent,
        "inspect_git_state",
        lambda root: {
            "is_repository": True,
            "has_head": True,
            "clean": False,
            "head": "abc",
            "status_lines": [" M file"],
        },
    )
    monkeypatch.setattr(
        agent,
        "probe_codex_cli",
        lambda executable: {"available": False, "reason": "codex_help_failed"},
    )

    result = agent.run_autonomous_codex_research_agent(
        state_path=tmp_path / "state.json",
        hypotheses_path=tmp_path / "hypotheses.json",
        task_queue_path=tmp_path / "queue.json",
        report_dir=report_dir,
        prompt_template_path=prompt,
        worktree_root=tmp_path / "worktrees",
        now=datetime.fromisoformat("2026-07-26T12:10:00+09:00"),
    )

    assert result["exit_code"] == 0
    assert result["classification"] == "agent_skipped_unhealthy_gate"
    assert result["stop_reason"] == "main_worktree_dirty"
    assert result["worktree"] is None
    assert not (tmp_path / "worktrees").exists()
    assert (tmp_path / "state.json").exists()
    assert (tmp_path / "hypotheses.json").exists()
    state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert state["last_successful_cycle_at"] is None
    assert not (tmp_path / "queue.json").exists()


def test_clean_gate_with_unavailable_codex_blocks_selected_hypothesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir = tmp_path / "reports"
    _write_cycle_reports(report_dir)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("safe template", encoding="utf-8")
    guard = {
        "database": "db",
        "database_readable": True,
        "database_counts": {"predictions": 1},
        "binary_artifact": "binary",
        "multiclass_artifact": "multi",
        "settings": "settings",
        "daily_predict": "daily",
    }
    monkeypatch.setattr(agent, "DEFAULT_LOCK_PATH", tmp_path / "agent.lock")
    monkeypatch.setattr(agent, "capture_production_guard", lambda **kwargs: dict(guard))
    monkeypatch.setattr(
        agent,
        "inspect_git_state",
        lambda root: {
            "is_repository": True,
            "has_head": True,
            "clean": True,
            "head": "abc",
            "status_lines": [],
        },
    )
    monkeypatch.setattr(
        agent,
        "probe_codex_cli",
        lambda executable: {"available": False, "reason": "codex_help_failed"},
    )

    result = agent.run_autonomous_codex_research_agent(
        state_path=tmp_path / "state.json",
        hypotheses_path=tmp_path / "hypotheses.json",
        task_queue_path=tmp_path / "queue.json",
        report_dir=report_dir,
        prompt_template_path=prompt,
        worktree_root=tmp_path / "worktrees",
        now=datetime.fromisoformat("2026-07-26T12:10:00+09:00"),
    )

    assert result["exit_code"] == 5
    assert result["selected_hypothesis"]["hypothesis_id"] == "balanced_threshold_guardrail"
    assert result["worktree"] is None
    backlog = json.loads((tmp_path / "hypotheses.json").read_text(encoding="utf-8"))
    selected = next(
        item for item in backlog["hypotheses"] if item["hypothesis_id"] == "balanced_threshold_guardrail"
    )
    assert selected["status"] == "blocked"
    assert selected["next_action"] == "run manually from a supported Codex environment"


def test_wrapper_has_no_scheduler_registration_and_enforces_formal_root() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_autonomous_codex_research_agent.ps1").read_text(
        encoding="utf-8-sig"
    )

    assert '"C:\\Users\\kmats\\Documents\\ai-stock\\stock-ai"' in text
    assert ".venv\\Scripts\\python.exe" in text
    assert "MaxRuntimeMinutes = 90" in text
    assert "MaxRuntimeMinutes -gt 180" in text
    assert ".codex_research_agent_wrapper.lock" in text
    assert "$process.WaitForExit()" in text
    assert '"exit_code: $exitCode"' in text
    assert "exit 125" in text
    assert "exit 124" in text
    assert "Register-ScheduledTask" not in text


def test_agent_policy_enforces_v11_cycle_budgets() -> None:
    policy = agent.AgentPolicy()

    policy.validate()
    assert policy.as_dict() == {
        "max_hypotheses_per_cycle": 1,
        "max_codex_tasks_per_cycle": 1,
        "max_implementation_attempts": 1,
        "max_retuning_rounds_after_result": 0,
        "max_experiment_variants": 6,
        "max_changed_files": 8,
        "max_patch_lines": 1500,
        "max_runtime_minutes": 90.0,
        "hard_max_runtime_minutes": 180.0,
    }
    with pytest.raises(ValueError, match="retuning"):
        agent.AgentPolicy(max_retuning_rounds_after_result=1).validate()
    with pytest.raises(ValueError, match="changed_files"):
        agent.AgentPolicy(max_changed_files=9).validate()


def test_old_agent_state_loads_with_v11_defaults(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps({"schema_version": "autonomous_codex_research_agent/v1", "last_cycle_at": "x"}),
        encoding="utf-8",
    )

    state, initialized = agent.load_agent_state(path)

    assert initialized is False
    assert state["schema_version"] == agent.SCHEMA_VERSION
    assert state["last_cycle_at"] == "x"
    assert state["last_preflight_at"] is None
    assert state["completed_hypothesis_fingerprints"] == []
    assert state["rejected_hypothesis_fingerprints"] == []


def test_rejected_fingerprint_is_not_selected_even_if_status_is_requeued() -> None:
    backlog = agent.seed_hypothesis_backlog()
    rejected = backlog["hypotheses"][0]
    rejected["status"] = "proposed"
    state = agent.default_agent_state()
    state["rejected_hypothesis_fingerprints"] = [rejected["fingerprint"]]
    lifecycle = agent.summarize_hypothesis_lifecycle(backlog, state=state)

    selected = agent.select_hypothesis(
        backlog,
        research_report=_healthy_research(),
        excluded_fingerprints=lifecycle["rejected_fingerprints"],
    )["selected"]

    assert selected["fingerprint"] != rejected["fingerprint"]


def test_due_gate_skips_duplicate_research_without_new_evidence() -> None:
    backlog = agent.seed_hypothesis_backlog()
    state = agent.default_agent_state()
    state.update(
        {
            "last_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_heavy_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_research_fingerprint": "abc123",
            "last_live_confirmed_count": 4,
            "last_shadow_confirmed_count": 2,
        }
    )
    research = _healthy_research()
    research["live_metrics"] = {"sample_count": 4}
    research["shadow_metrics"] = {"confirmed_count": 2}

    due = agent.assess_agent_due(
        state=state,
        backlog=backlog,
        research_report=research,
        task_queue={"tasks": []},
        gate={"healthy": True},
        now=datetime.fromisoformat("2026-07-26T12:00:00+09:00"),
    )

    assert due["due"] is False
    assert due["classification"] == "agent_skipped_duplicate_research_state"
    assert due["codex_invocation_required"] is False
    assert due["worktree_creation_required"] is False


def test_due_gate_triggers_after_five_new_confirmed_live_rows() -> None:
    state = agent.default_agent_state()
    state.update(
        {
            "last_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_heavy_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_research_fingerprint": "abc123",
            "last_live_confirmed_count": 4,
        }
    )
    research = _healthy_research()
    research["live_metrics"] = {"sample_count": 9}

    due = agent.assess_agent_due(
        state=state,
        backlog=agent.seed_hypothesis_backlog(),
        research_report=research,
        task_queue={"tasks": []},
        gate={"healthy": True},
        now=datetime.fromisoformat("2026-07-26T12:00:00+09:00"),
    )

    assert due["due"] is True
    assert "live_confirmed_increased_by_5" in due["reason_codes"]


def test_no_due_full_cycle_never_probes_codex_or_creates_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir = tmp_path / "reports"
    _write_cycle_reports(report_dir)
    state = agent.default_agent_state()
    state.update(
        {
            "last_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_heavy_cycle_at": "2026-07-26T11:00:00+09:00",
            "last_research_fingerprint": "abc123",
        }
    )
    (tmp_path / "state.json").write_text(json.dumps(state), encoding="utf-8")
    guard = {
        "database": "db",
        "database_readable": True,
        "database_counts": {"predictions": 1},
        "binary_artifact": "binary",
        "multiclass_artifact": "multi",
        "settings": "settings",
        "daily_predict": "daily",
    }
    monkeypatch.setattr(agent, "DEFAULT_LOCK_PATH", tmp_path / "agent.lock")
    monkeypatch.setattr(agent, "capture_production_guard", lambda **kwargs: dict(guard))
    monkeypatch.setattr(
        agent,
        "inspect_git_state",
        lambda root: {
            "is_repository": True,
            "has_head": True,
            "clean": True,
            "head": "abc",
            "status_lines": [],
        },
    )
    monkeypatch.setattr(
        agent, "probe_codex_cli", lambda value: pytest.fail("Codex probe must not run when not due")
    )
    monkeypatch.setattr(
        agent,
        "create_research_worktree",
        lambda **kwargs: pytest.fail("worktree must not be created when not due"),
    )

    result = agent.run_autonomous_codex_research_agent(
        state_path=tmp_path / "state.json",
        hypotheses_path=tmp_path / "hypotheses.json",
        task_queue_path=tmp_path / "queue.json",
        report_dir=report_dir,
        prompt_template_path=tmp_path / "unused.md",
        worktree_root=tmp_path / "worktrees",
        now=datetime.fromisoformat("2026-07-26T12:10:00+09:00"),
    )

    assert result["exit_code"] == 0
    assert result["classification"] == "agent_skipped_duplicate_research_state"
    assert result["worktree"] is None
    saved_state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved_state["last_research_fingerprint"] == "abc123"
    assert saved_state["last_heavy_cycle_at"] == "2026-07-26T11:00:00+09:00"


def test_read_only_status_does_not_create_or_modify_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_dir = tmp_path / "reports"
    _write_cycle_reports(report_dir)
    before = {
        path: path.read_bytes()
        for path in report_dir.rglob("*")
        if path.is_file()
    }
    monkeypatch.setattr(agent, "DEFAULT_LOCK_PATH", tmp_path / "agent.lock")
    monkeypatch.setattr(
        agent,
        "inspect_git_state",
        lambda root: {
            "is_repository": True,
            "has_head": True,
            "clean": True,
            "head": "abc",
            "status_lines": [],
        },
    )
    monkeypatch.setattr(
        agent,
        "capture_production_guard",
        lambda **kwargs: {"database_readable": True, "settings": "settings"},
    )

    status = agent.autonomous_codex_research_status(
        state_path=tmp_path / "state.json",
        hypotheses_path=tmp_path / "hypotheses.json",
        task_queue_path=tmp_path / "queue.json",
        report_dir=report_dir,
        worktree_root=tmp_path / "worktrees",
        now=datetime.fromisoformat("2026-07-26T12:10:00+09:00"),
    )

    assert status["mode"] == "read_only_preflight"
    assert status["side_effects"] == {
        "codex_invoked": False,
        "worktree_created": False,
        "code_changed": False,
        "database_changed": False,
        "hypothesis_marked_running": False,
    }
    assert not (tmp_path / "state.json").exists()
    assert not (tmp_path / "hypotheses.json").exists()
    assert not (tmp_path / "queue.json").exists()
    after = {path: path.read_bytes() for path in report_dir.rglob("*") if path.is_file()}
    assert after == before


def test_compact_context_excludes_full_logs_and_task_queue() -> None:
    operations = _healthy_operations()
    operations["very_large_log"] = "SHOULD_NOT_APPEAR"
    research = _healthy_research()
    research["full_task_queue"] = "SHOULD_NOT_APPEAR"
    selected = agent.seed_hypothesis_backlog()["hypotheses"][0]

    bundle = agent.build_compact_context_bundle(
        ticker="7203.T",
        operations_report=operations,
        research_report=research,
        backlog=agent.seed_hypothesis_backlog(),
        selected_hypothesis=selected,
        policy=agent.AgentPolicy(),
    )
    serialized = json.dumps(bundle, ensure_ascii=False)

    assert bundle["ticker"] == "7203.T"
    assert bundle["selected_hypothesis"]["hypothesis_id"] == selected["hypothesis_id"]
    assert "SHOULD_NOT_APPEAR" not in serialized
    assert "full_task_queue" not in serialized


def test_cleanup_ready_requires_report_patch_hash_and_matching_worktree(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    worktree_root = tmp_path / "worktrees"
    worktree = worktree_root / "candidate"
    reports = tmp_path / "reports" / "autonomous"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True, capture_output=True)
    worktree_root.mkdir()
    subprocess.run(
        ["git", "worktree", "add", "-b", "research/candidate", str(worktree), "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
    )
    (worktree / "tracked.txt").write_text("changed\n", encoding="utf-8")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=worktree, text=True)
    reports.mkdir(parents=True)
    report_path = reports / "codex_agent_candidate_2026-07-26_1200.json"
    patch_path = reports / "patches" / "candidate_2026-07-26_1200.patch"
    patch_path.parent.mkdir()
    report_path.write_text("{}", encoding="utf-8")
    patch_path.write_text(diff, encoding="utf-8")
    patch_hash = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    hypothesis = {
        "hypothesis_id": "candidate",
        "status": "rejected",
        "worktree": str(worktree),
        "report_path": str(report_path),
        "patch_path": str(patch_path),
        "patch_sha256": patch_hash,
    }

    audit = agent.inspect_research_worktree(
        hypothesis,
        project_root=repository,
        worktree_root=worktree_root,
        report_dir=reports,
        check_processes=False,
    )

    assert audit["lifecycle_status"] == "cleanup_ready"
    assert audit["safe_to_remove"] is True
    assert audit["patch_hash_recorded_and_valid"] is True
    assert audit["patch_matches_worktree"] is True

    hypothesis["patch_sha256"] = None
    unsafe = agent.inspect_research_worktree(
        hypothesis,
        project_root=repository,
        worktree_root=worktree_root,
        report_dir=reports,
        check_processes=False,
    )
    assert unsafe["lifecycle_status"] == "unsafe_to_remove"
    assert unsafe["safe_to_remove"] is False


def test_cleanup_is_dry_run_by_default_and_unsafe_confirm_does_not_remove(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe = {"safe_to_remove": True, "lifecycle_status": "cleanup_ready", "path": "C:/safe"}
    monkeypatch.setattr(agent, "inspect_research_worktree", lambda *args, **kwargs: safe)
    dry_run = agent.cleanup_research_worktree({"hypothesis_id": "x"})
    assert dry_run["removed"] is False
    assert dry_run["would_remove"] is True

    unsafe = {"safe_to_remove": False, "lifecycle_status": "unsafe_to_remove", "path": "C:/bad"}
    monkeypatch.setattr(agent, "inspect_research_worktree", lambda *args, **kwargs: unsafe)
    monkeypatch.setattr(agent, "_git", lambda *args, **kwargs: pytest.fail("must not remove"))
    confirmed = agent.cleanup_research_worktree({"hypothesis_id": "x"}, confirm=True)
    assert confirmed["removed"] is False


def test_removed_terminal_worktree_is_cleanup_completed_not_orphaned(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    reports = tmp_path / "reports"
    repository.mkdir()
    subprocess.run(["git", "init"], cwd=repository, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repository, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repository, check=True)
    (repository / "README.md").write_text("base", encoding="utf-8")
    subprocess.run(["git", "add", "README.md"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repository, check=True, capture_output=True)
    reports.mkdir()
    report_path = reports / "result.json"
    patch_path = reports / "result.patch"
    report_path.write_text("{}", encoding="utf-8")
    patch_path.write_text("preserved patch", encoding="utf-8")
    hypothesis = {
        "hypothesis_id": "removed",
        "status": "rejected",
        "worktree": str(tmp_path / "worktrees" / "removed"),
        "report_path": str(report_path),
        "patch_path": str(patch_path),
        "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest(),
    }

    audit = agent.inspect_research_worktree(
        hypothesis,
        project_root=repository,
        worktree_root=tmp_path / "worktrees",
        report_dir=reports,
        check_processes=False,
    )

    assert audit["lifecycle_status"] == "cleanup_completed"
    assert audit["safe_to_remove"] is False


def test_prompt_template_contains_all_nonnegotiable_guards() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "prompts" / "autonomous_codex_research_agent.md").read_text(
        encoding="utf-8"
    )

    for phrase in (
        "production_sqlite_changed",
        "live_backfill_prediction_created",
        "automatic_commit_performed",
        "automatic_merge_performed",
        "task_scheduler_changed",
        "historical all",
        "recent 3 months",
        "moving-average trend only",
    ):
        assert phrase in text
