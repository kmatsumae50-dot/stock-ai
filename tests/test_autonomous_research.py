from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from stock_ai.config import (
    FeatureSettings,
    LabelSettings,
    OperationsSettings,
    PathSettings,
    Settings,
)
from stock_ai.operations import autonomous_research as research
from stock_ai.storage.sqlite_store import init_database


def test_first_run_trigger_and_force_trigger() -> None:
    policy = research.ResearchPolicy()
    state = research.default_research_state()

    first = research.determine_trigger_reasons(
        state,
        first_run=True,
        force=False,
        live_confirmed_count=0,
        shadow_confirmed_count=0,
        live_metrics={},
        shadow_disagreement_count=0,
        now=pd.Timestamp("2026-07-25"),
        policy=policy,
    )
    forced = research.determine_trigger_reasons(
        state,
        first_run=False,
        force=True,
        live_confirmed_count=0,
        shadow_confirmed_count=0,
        live_metrics={},
        shadow_disagreement_count=0,
        now=pd.Timestamp("2026-07-25"),
        policy=policy,
    )

    assert first == ["first_run"]
    assert forced == ["force"]


def test_trigger_thresholds_are_conservative() -> None:
    state = research.default_research_state()
    state.update(
        {
            "last_live_confirmed_count": 10,
            "last_shadow_confirmed_count": 2,
            "last_live_metrics": {"balanced_accuracy": 0.50, "up_recall": 0.20},
            "shadow_direction_disagreement_count": 1,
        }
    )
    reasons = research.determine_trigger_reasons(
        state,
        first_run=False,
        force=False,
        live_confirmed_count=15,
        shadow_confirmed_count=7,
        live_metrics={"balanced_accuracy": 0.56, "up_recall": 0.31},
        shadow_disagreement_count=4,
        now=pd.Timestamp("2026-07-25"),
        policy=research.ResearchPolicy(),
    )

    assert "live_confirmed_increased" in reasons
    assert "shadow_confirmed_increased" in reasons
    assert "live_balanced_accuracy_changed" in reasons
    assert "live_up_recall_changed" in reasons
    assert "shadow_direction_disagreement_increased" in reasons


def test_deterministic_experiment_fingerprint_ignores_runtime() -> None:
    kwargs = {
        "ticker": "7203.T",
        "latest_feature_date": "2026-07-24",
        "live_confirmed_count": 37,
        "shadow_confirmed_count": 2,
        "candidates": ["class_weight_balanced", "production_current"],
        "windows": [{"name": "historical_all", "start_date": "2023-01-01", "end_date": "2026-07-24"}],
        "model_config_hash": "model",
        "feature_config_hash": "feature",
    }

    first = research.compute_experiment_fingerprint(**kwargs)
    second = research.compute_experiment_fingerprint(**kwargs)

    assert first == second
    assert len(first) == 64


def test_evaluation_windows_include_required_periods_and_live_like() -> None:
    frame = pd.DataFrame({"date": pd.bdate_range("2022-01-03", periods=1000)})

    windows = research.build_evaluation_windows(frame)

    names = {row["name"] for row in windows}
    assert {
        "historical_all",
        "since_2023",
        "since_2024",
        "since_2025",
        "since_2026",
        "recent_6_months",
        "recent_3_months",
        "live_like_holdout",
    }.issubset(names)


def test_metrics_handle_single_class_auc_and_small_sample() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2026-01-01", periods=3),
            "actual_class": [1, 1, 1],
            "predicted_class": [1, 0, 1],
            "probability_up": [0.8, 0.4, 0.7],
        }
    )

    row = research._metric_row(frame, "candidate", "small", min_sample=100)

    assert row["sample_too_small"] is True
    assert row["roc_auc"] is None
    assert row["roc_auc_status"] == "not_available"
    assert row["f1_up"] is not None


def test_baseline_comparison_includes_majority_and_balanced_accuracy() -> None:
    frame = _prediction_frame(20, ["production_current"])

    rows = research._baseline_rows(frame, "production_current", "historical_all")

    assert {row["baseline"] for row in rows} == {
        "always_up",
        "always_down",
        "majority_class",
        "previous_day_direction",
        "moving_average_trend_only",
    }
    assert all("baseline_balanced_accuracy" in row for row in rows)


def test_candidate_ranking_does_not_promote_without_shadow_evidence() -> None:
    frame = _prediction_frame(160, ["production_current", "class_weight_balanced"])
    frame.loc[frame["candidate_name"].eq("production_current"), "predicted_class"] = 0
    windows = research.build_evaluation_windows(frame)
    metrics, _baselines, _confidence = research.evaluate_candidate_windows(
        frame,
        windows,
        min_sample=30,
    )

    ranking = research.rank_candidates(
        metrics,
        production_name="production_current",
        shadow_summary={"confirmed_count": 0},
        policy=research.ResearchPolicy(min_offline_sample=30, min_live_like_sample=20),
    )

    balanced = next(row for row in ranking if row["candidate_name"] == "class_weight_balanced")
    assert balanced["decision"] != "production_change_proposal_ready"


def test_candidate_proposal_gate_requires_all_evidence() -> None:
    decision = research._candidate_decision(
        "class_weight_balanced",
        historical={
            "sample_count": 500,
            "balanced_accuracy": 0.55,
            "up_recall": 0.50,
            "down_recall": 0.60,
        },
        live_like={"sample_count": 60, "balanced_accuracy": 0.54},
        production_historical={
            "balanced_accuracy": 0.50,
            "up_recall": 0.40,
            "down_recall": 0.62,
        },
        production_live_like={"balanced_accuracy": 0.50},
        improved_windows=3,
        baseline_wins=3,
        shadow_confirmed=12,
        policy=research.ResearchPolicy(),
    )

    assert decision == "production_change_proposal_ready"


def test_candidate_selection_reports_unavailable_and_enforces_limit() -> None:
    candidates, unavailable = research._resolve_candidates(
        ["production_current", "not_available"]
    )

    assert [candidate.name for candidate in candidates] == ["production_current"]
    assert unavailable == ["not_available"]
    try:
        research._resolve_candidates([f"candidate_{index}" for index in range(7)])
    except research.ResearchError as error:
        assert "Candidate limit exceeded" in str(error)
    else:
        raise AssertionError("candidate limit should stop safely")


def test_operations_unhealthy_gate_detects_action_and_stale() -> None:
    report = _operations_report()
    report["health_summary"]["action_required_count"] = 1
    report["data_freshness"]["freshness_status"] = "WARN"
    report["data_freshness"]["stale_at_run_time"] = True

    gate = research.assess_operations_health(report)

    assert gate["healthy"] is False
    assert "operations_action_required" in gate["reasons"]
    assert "stale_data" in gate["reasons"]


def test_latest_operations_report_ignores_state_and_task_queue(tmp_path: Path) -> None:
    expected = _operations_report()
    (tmp_path / "operations_2026-07-25_1200.json").write_text(
        json.dumps(expected), encoding="utf-8"
    )
    (tmp_path / "operations_task_queue.json").write_text(
        json.dumps({"tasks": []}), encoding="utf-8"
    )
    (tmp_path / "operations_state.json").write_text(
        json.dumps({"last_cycle_at": "later"}), encoding="utf-8"
    )

    loaded = research.load_latest_operations_report(tmp_path)

    assert loaded["stop_reason"] == "completed_read_only_cycle"


def test_task_queue_deduplicates_research_fingerprint(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    finding = _finding("research_candidate:7203.T:test:fingerprint")

    first, first_changes = research.update_research_tasks(
        path,
        [finding],
        cycle_at="2026-07-25T12:00:00+09:00",
        source_report="first.md",
        dry_run=False,
    )
    second, second_changes = research.update_research_tasks(
        path,
        [finding],
        cycle_at="2026-07-26T12:00:00+09:00",
        source_report="second.md",
        dry_run=False,
    )

    assert len(first["tasks"]) == 1
    assert len(second["tasks"]) == 1
    assert first_changes["created"] == [finding["fingerprint"]]
    assert second_changes["created"] == []
    assert second_changes["updated"] == [finding["fingerprint"]]


def test_task_queue_preserves_non_research_tasks(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps({"tasks": [{"fingerprint": "unknown_source:X", "category": "unknown_source", "status": "open"}]}),
        encoding="utf-8",
    )

    queue, _changes = research.update_research_tasks(
        path,
        [],
        cycle_at="2026-07-25T12:00:00+09:00",
        source_report="report.md",
        dry_run=False,
    )

    assert queue["tasks"][0]["status"] == "open"


def test_existing_live_model_review_prevents_duplicate_degradation_task(tmp_path: Path) -> None:
    path = tmp_path / "queue.json"
    path.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "fingerprint": "model_up_recall_low:7203.T",
                        "status": "approval_required",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert research._has_active_live_model_review(path, "7203.T") is True
    findings = research._research_findings(
        ticker="7203.T",
        fingerprint="experiment",
        ranking=[],
        live_metrics={"sample_count": 30, "balanced_accuracy": 0.35, "up_recall": 0.05},
        shadow={},
        report_path="report.md",
        existing_live_model_review=True,
    )

    assert findings == []


def test_history_is_append_only_and_cycle_id_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "history.jsonl"
    result = _history_result()

    assert research.append_research_history(path, result) is True
    assert research.append_research_history(path, result) is False
    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


def test_runtime_budget_raises_timeout() -> None:
    budget = research.RuntimeBudget(started=0.0, max_minutes=0.00001)

    try:
        budget.check("test")
    except research.ResearchTimeoutError:
        pass
    else:
        raise AssertionError("expired runtime budget should raise")


def test_evaluation_period_summary_distinguishes_live_and_shadow_samples() -> None:
    feature = _feature_frame(120)
    windows = research.build_evaluation_windows(feature)
    live = _prediction_frame(8, ["production_live"])
    live["status"] = "confirmed"

    periods = research.describe_evaluation_periods(
        feature,
        windows,
        live_frame=live,
        shadow={
            "confirmed_count": 2,
            "confirmed_start_date": "2026-07-08",
            "confirmed_end_date": "2026-07-08",
            "actual_up_count": 1,
            "actual_down_count": 1,
        },
        policy=research.ResearchPolicy(),
    )

    by_name = {row["name"]: row for row in periods}
    assert by_name["live_confirmed"]["sample_count"] == 8
    assert by_name["shadow_confirmed"]["sample_count"] == 2
    assert by_name["shadow_confirmed"]["sample_too_small"] is True


def test_corrupt_state_is_not_deleted(tmp_path: Path) -> None:
    path = tmp_path / "research_state.json"
    path.write_text("{broken", encoding="utf-8")

    try:
        research.load_research_state(path)
    except research.ResearchStateError:
        pass
    else:
        raise AssertionError("corrupt state should fail safely")

    assert path.read_text(encoding="utf-8") == "{broken"


def test_lock_blocks_concurrent_run(tmp_path: Path) -> None:
    path = tmp_path / ".research.lock"

    with research.ResearchFileLock(path):
        try:
            with research.ResearchFileLock(path):
                pass
        except research.ResearchLockError:
            pass
        else:
            raise AssertionError("concurrent Research lock should fail")

    assert not path.exists()


def test_full_first_run_then_duplicate_skip_without_production_write(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    feature = _feature_frame(360)
    feature.to_parquet(settings.paths.feature_data_dir / "7203.T.parquet", index=False)
    init_database(settings.paths.database_path)
    report_dir = tmp_path / "reports" / "autonomous"
    report_dir.mkdir(parents=True)
    (report_dir / "operations_2026-07-25_1200.json").write_text(
        json.dumps(_operations_report()), encoding="utf-8"
    )
    predictions = _prediction_frame(120, list(research.DEFAULT_CANDIDATES))
    runner_calls = []

    def runner(*_args, **_kwargs):
        runner_calls.append(1)
        return SimpleNamespace(predictions=predictions)

    kwargs = {
        "ticker": "7203.T",
        "state_path": report_dir / "research_state.json",
        "history_path": report_dir / "research_history.jsonl",
        "report_dir": report_dir,
        "task_queue_path": report_dir / "queue.json",
        "settings": settings,
        "experiment_runner": runner,
        "now": datetime.fromisoformat("2026-07-25T12:00:00+09:00"),
    }
    before = research._database_snapshot(settings.paths.database_path)
    first = research.run_autonomous_research(**kwargs)
    second = research.run_autonomous_research(**kwargs)
    after = research._database_snapshot(settings.paths.database_path)

    assert first["experiment_executed"] is True
    assert second["classification"] == "research_skipped_duplicate_experiment"
    assert runner_calls == [1]
    assert before == after
    assert first["production_change_applied"] is False


def test_unhealthy_operations_skips_runner(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _feature_frame(40).to_parquet(settings.paths.feature_data_dir / "7203.T.parquet", index=False)
    init_database(settings.paths.database_path)
    report_dir = tmp_path / "reports" / "autonomous"
    report_dir.mkdir(parents=True)
    operations = _operations_report()
    operations["health_summary"]["action_required_count"] = 1
    (report_dir / "operations_2026-07-25_1200.json").write_text(json.dumps(operations), encoding="utf-8")

    def runner(*_args, **_kwargs):
        raise AssertionError("offline runner must not execute")

    result = research.run_autonomous_research(
        ticker="7203.T",
        state_path=report_dir / "state.json",
        history_path=report_dir / "history.jsonl",
        report_dir=report_dir,
        task_queue_path=report_dir / "queue.json",
        settings=settings,
        experiment_runner=runner,
        now=datetime.fromisoformat("2026-07-25T12:00:00+09:00"),
    )

    assert result["classification"] == "research_skipped_operations_unhealthy"
    assert result["exit_code"] == 3


def test_wrapper_is_not_a_scheduler_registration_script() -> None:
    text = (research.PROJECT_ROOT / "scripts" / "run_autonomous_research.ps1").read_text(encoding="utf-8")

    assert "Register-ScheduledTask" not in text
    assert "daily_predict.ps1" not in text
    assert "STOCK_AI_RESEARCH_WRAPPER_LOCK_HELD" in text
    assert "Stop-Process" in text
    assert "exit 124" in text


def _prediction_frame(rows: int, candidates: list[str]) -> pd.DataFrame:
    frames = []
    dates = pd.bdate_range("2023-01-02", periods=rows)
    actual = pd.Series([index % 2 for index in range(rows)])
    for index, candidate in enumerate(candidates):
        predicted = actual.copy() if index else pd.Series([0 if value else 1 for value in actual])
        probability = predicted.map({1: 0.65, 0: 0.35})
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "candidate_name": candidate,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "probability_up": probability,
                    "confidence": 0.65,
                    "baseline_previous_day_direction": [index % 2 for index in range(rows)],
                    "baseline_moving_average_trend": [index % 2 for index in range(rows)],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _feature_frame(rows: int) -> pd.DataFrame:
    close = pd.Series([100 + index * 0.1 + (0.3 if index % 2 else -0.2) for index in range(rows)])
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=rows),
            "ticker": "7203.T",
            "close": close,
            "return_1d": close.pct_change(),
            "ma_5": close.rolling(5, min_periods=1).mean(),
            "ma_25": close.rolling(25, min_periods=1).mean(),
            "rsi_14": 50.0,
        }
    )
    return add_next_close_binary_target_for_test(frame)


def add_next_close_binary_target_for_test(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["next_close_return"] = result["close"].shift(-1) / result["close"] - 1
    result["target_next_close_binary"] = (result["next_close_return"] > 0).astype("Int64")
    result.loc[result["next_close_return"].isna(), "target_next_close_binary"] = pd.NA
    return result


def _settings(tmp_path: Path) -> Settings:
    raw = tmp_path / "data" / "raw"
    features = tmp_path / "data" / "features"
    models = tmp_path / "models"
    for path in (raw, features, models, tmp_path / "data" / "processed", tmp_path / "data" / "predictions"):
        path.mkdir(parents=True, exist_ok=True)
    (models / "lightgbm_binary_7203.T.joblib").write_bytes(b"binary")
    (models / "lightgbm_multiclass_7203.T.joblib").write_bytes(b"multi")
    return Settings(
        paths=PathSettings(
            raw_data_dir=raw,
            processed_data_dir=tmp_path / "data" / "processed",
            feature_data_dir=features,
            prediction_data_dir=tmp_path / "data" / "predictions",
            model_dir=models,
            database_path=tmp_path / "database" / "market.db",
        ),
        labels=LabelSettings(0.05, 0.01, -0.01, -0.05),
        features=FeatureSettings((5, 25, 75), 14, 20, 20),
        operations=OperationsSettings(),
        data={},
    )


def _operations_report() -> dict:
    return {
        "stop_reason": "completed_read_only_cycle",
        "health_summary": {"action_required_count": 0, "unknown_source_count": 0},
        "unknown_source_findings": [],
        "missing_live_findings": [],
        "data_freshness": {"freshness_status": "OK", "stale_at_run_time": False},
        "database_validation": {"readable": True},
        "report_paths": {},
    }


def _finding(fingerprint: str) -> dict:
    return {
        "fingerprint": fingerprint,
        "title": "Review candidate",
        "description": "Evidence",
        "candidate": "test",
        "evidence": {},
        "production_difference": {},
        "baseline_difference": {},
        "risks": [],
        "proposed_action": "Review only",
    }


def _history_result() -> dict:
    return {
        "cycle_id": "cycle-1",
        "cycle_completed_at": "2026-07-25T12:00:00+09:00",
        "ticker": "7203.T",
        "trigger_reasons": ["first_run"],
        "experiment_executed": True,
        "operations_health": {"healthy": True},
        "data_fingerprint": "data",
        "experiment_fingerprint": "experiment",
        "candidates": list(research.DEFAULT_CANDIDATES),
        "evaluation_periods": [],
        "metrics": [],
        "baselines": [],
        "candidate_ranking": [],
        "recommendation": "production_change_not_ready",
        "report_paths": {"markdown": "report.md"},
        "stop_reason": "research_completed_no_material_change",
    }
