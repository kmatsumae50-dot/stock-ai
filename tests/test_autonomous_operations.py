from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_ai.operations import autonomous


def test_default_state_can_be_saved_and_reloaded(tmp_path: Path) -> None:
    state_path = tmp_path / "operations_state.json"

    initial, created = autonomous.load_operations_state(state_path)

    assert created is True
    assert initial["schema_version"] == autonomous.SCHEMA_VERSION
    initial["last_health_status"] = "WARN"
    autonomous.save_operations_state(state_path, initial)
    loaded, created = autonomous.load_operations_state(state_path)
    assert created is False
    assert loaded["last_health_status"] == "WARN"


def test_corrupt_state_is_not_silently_recovered(tmp_path: Path) -> None:
    state_path = tmp_path / "operations_state.json"
    state_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(autonomous.OperationsStateError):
        autonomous.load_operations_state(state_path)


def test_task_queue_deduplicates_and_resolves_findings() -> None:
    finding = autonomous._finding(
        fingerprint="unknown_source:Example",
        category="unknown_source",
        risk_level="Level 2",
        priority="high",
        title="Review Example",
        description="Needs review",
        evidence={"source": "Example"},
        proposed_action="Review manually",
        requires_human_approval=True,
    )
    empty = {"schema_version": autonomous.SCHEMA_VERSION, "tasks": []}

    first, first_changes = autonomous.reconcile_task_queue(
        empty,
        [finding],
        cycle_at="2026-07-24T10:00:00+09:00",
        source_report="first.md",
    )
    second, second_changes = autonomous.reconcile_task_queue(
        first,
        [finding],
        cycle_at="2026-07-25T10:00:00+09:00",
        source_report="second.md",
    )
    resolved, resolved_changes = autonomous.reconcile_task_queue(
        second,
        [],
        cycle_at="2026-07-26T10:00:00+09:00",
        source_report="third.md",
    )

    assert first_changes["created"] == ["unknown_source:Example"]
    assert len(second["tasks"]) == 1
    assert second_changes["created"] == []
    assert second_changes["updated"] == ["unknown_source:Example"]
    assert resolved_changes["resolved"] == ["unknown_source:Example"]
    assert resolved["tasks"][0]["status"] == "resolved"


def test_operations_file_lock_blocks_concurrent_and_active_force(tmp_path: Path) -> None:
    lock_path = tmp_path / ".operations.lock"

    with autonomous.OperationsFileLock(lock_path):
        with pytest.raises(autonomous.OperationsLockError):
            with autonomous.OperationsFileLock(lock_path):
                pass
        with pytest.raises(autonomous.OperationsLockError):
            with autonomous.OperationsFileLock(lock_path, force=True):
                pass

    assert not lock_path.exists()


def test_runtime_budget_detects_timeout() -> None:
    budget = autonomous.RuntimeBudget(
        started_monotonic=0.0,
        max_runtime_minutes=0.00001,
    )

    with pytest.raises(autonomous.OperationsTimeoutError):
        budget.check("test")


def test_review_stop_is_not_counted_as_cycle_execution_failure() -> None:
    assert autonomous._cycle_completed_successfully(None) is True
    assert (
        autonomous._cycle_completed_successfully(
            "multiple_action_required_items_need_human_review"
        )
        is True
    )
    assert autonomous._cycle_completed_successfully("health_check_unavailable") is False


def test_health_summary_counts_known_missed_dates_even_when_summary_is_legacy() -> None:
    health = {
        "overall_status": "WARN",
        "warning_summary": {
            "action_required_count": 0,
            "known_missed_section_count": 0,
            "known_missed_date_count": 0,
            "known_ok_count": 1,
            "known_ok": [],
        },
        "items": [
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "details": {
                    "known_missed_live_dates": [
                        {"missing_date": "2026-06-15"},
                        {"missing_date": "2026-06-23"},
                    ],
                    "missing_live_date_diagnostics": [],
                },
            },
            {
                "section": "News",
                "name": "News CSV quality",
                "status": "WARN",
                "details": {"status": "OK", "unknown_source_count": 0},
            },
        ],
    }

    summary = autonomous._summarize_health(health)

    assert summary["known_missed_section_count"] == 1
    assert summary["known_missed_date_count"] == 2
    assert summary["check_news_status"] == "OK"


def test_shadow_missing_is_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(autonomous, "SHADOW_PATH", tmp_path / "missing.csv")

    summary = autonomous._summarize_shadow(pd.DataFrame())

    assert summary["file_status"] == "missing"
    assert summary["confirmed_count"] == 0
    assert "continue_shadow_monitoring" in summary["conclusions"]


def test_append_only_empty_is_safe(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        autonomous,
        "build_append_only_health_visibility",
        lambda *_args, **_kwargs: {
            "prediction_runs_count": 0,
            "prediction_records_count": 0,
            "existing_predictions_count": 65,
            "latest_compare_status": "append_only_empty",
            "mismatch_count": 0,
            "visibility_classifications": ["append_only_empty"],
        },
    )
    monkeypatch.setattr(
        autonomous,
        "compare_append_only_official_to_snapshot",
        lambda *_args, **_kwargs: {"matches": False, "rows": []},
    )
    monkeypatch.setattr(
        autonomous,
        "build_official_selection_diagnostics_from_db",
        lambda *_args, **_kwargs: {
            "scheduled_run_count": 0,
            "manual_run_count": 0,
            "selected_official_candidate_count": 0,
        },
    )

    summary = autonomous._summarize_append_only(tmp_path / "market.db", "7203.T")

    assert summary["prediction_runs_count"] == 0
    assert summary["existing_predictions_count"] == 65
    assert summary["info_only"] is True


def test_cycle_classifies_append_only_mismatch_as_info() -> None:
    classifications = autonomous.classify_operations_cycle(
        health_summary={
            "action_required_count": 0,
            "known_ok_count": 1,
            "known_missed_date_count": 0,
        },
        freshness={"stale_at_run_time": False},
        scheduler={"needs_review": False},
        unknown_sources=[],
        missing_live=[],
        binary_metrics={"confirmed": 0},
        shadow={"needs_review": False},
        append_only={"mismatch_count": 1},
    )

    assert classifications == ["healthy_known_warn", "append_only_info_mismatch"]


def test_unknown_source_and_missing_live_become_level_two_review_tasks() -> None:
    findings = autonomous._build_findings(
        ticker="7203.T",
        health_summary={"action_required": []},
        freshness={"stale_at_run_time": False},
        scheduler={"needs_review": False},
        unknown_sources=[{"source": "Example News"}],
        missing_live=[{"missing_date": "2026-07-21"}],
        binary_metrics={"confirmed": 0},
        confidence={"classification": "confidence_signal_unclear"},
        shadow={"needs_review": False},
        append_only={"mismatch_count": 0},
    )

    by_fingerprint = {row["fingerprint"]: row for row in findings}
    assert by_fingerprint["unknown_source:Example News"]["risk_level"] == "Level 2"
    assert by_fingerprint["unknown_source:Example News"]["requires_human_approval"] is True
    assert by_fingerprint["missing_live:7203.T:2026-07-21"]["risk_level"] == "Level 2"
    assert by_fingerprint["missing_live:7203.T:2026-07-21"]["requires_human_approval"] is True


def test_source_proposal_uses_original_publisher_not_google_news_transport() -> None:
    proposal = autonomous._source_profile_proposal(
        "Top Speed",
        "https://news.google.com/rss/articles/example",
    )

    assert proposal["source_type"] == "general_news"
    assert "news.google.com" not in proposal["aliases"]


def test_mojibake_or_placeholder_source_is_not_safe_for_profile_proposal() -> None:
    assert autonomous._looks_like_mojibake_or_placeholder("????") is True
    assert autonomous._looks_like_mojibake_or_placeholder("繝ｭ繧､繧ｿ繝ｼ") is True
    assert autonomous._looks_like_mojibake_or_placeholder("Reuters") is False


def test_known_translated_source_proposes_alias_only() -> None:
    profiles = [
        SimpleNamespace(
            name="Reuters",
            aliases=("reuters",),
            source_type="wire",
            credibility_score=0.9,
            market_impact_weight=0.85,
        )
    ]

    candidate = autonomous._known_alias_candidate("ロイター", profiles)
    proposal = autonomous._alias_addition_proposal("ロイター", candidate, profiles)

    assert candidate["profile"] == "Reuters"
    assert proposal["proposal_type"] == "add_alias_to_existing_profile"
    assert proposal["source_type"] == "wire"


def test_confidence_summary_uses_only_confirmed_rows() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-07-01", "status": "confirmed", "confidence": 0.61, "is_correct": True},
            {"date": "2026-07-02", "status": "confirmed", "confidence": 0.66, "is_correct": False},
            {"date": "2026-07-03", "status": "pending", "confidence": 0.99, "is_correct": False},
        ]
    )

    summary = autonomous._summarize_confidence(frame, autonomous.default_operations_state())

    by_threshold = {row["threshold"]: row for row in summary["thresholds"]}
    assert by_threshold[0.60]["count"] == 2
    assert by_threshold[0.65]["count"] == 1
    assert summary["classification"] == "sample_too_small"


def test_read_only_cycle_writes_only_state_queue_and_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    database = project_root / "database" / "market.db"
    raw_dir = project_root / "data" / "raw"
    feature_dir = project_root / "data" / "features"
    model_dir = project_root / "models" / "artifacts"
    for directory in (
        database.parent,
        raw_dir,
        feature_dir,
        model_dir,
        project_root / "configs",
        project_root / "scripts",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    production_files = {
        database: b"database-before",
        project_root / "configs" / "settings.yml": b"settings-before",
        project_root / "configs" / "news_sources.yaml": b"sources-before",
        project_root / "scripts" / "daily_predict.ps1": b"daily-before",
        model_dir / "lightgbm_binary_7203.T.joblib": b"binary-before",
        model_dir / "lightgbm_multiclass_7203.T.joblib": b"multiclass-before",
    }
    for path, payload in production_files.items():
        path.write_bytes(payload)
    settings = SimpleNamespace(
        paths=SimpleNamespace(
            database_path=database,
            raw_data_dir=raw_dir,
            feature_data_dir=feature_dir,
            model_dir=model_dir,
        )
    )
    health = {
        "overall_status": "OK",
        "warning_summary": {
            "action_required_count": 0,
            "known_missed_section_count": 0,
            "known_missed_date_count": 0,
            "known_ok_count": 0,
            "action_required": [],
            "known_ok": [],
            "known_missed": [],
        },
        "items": [
            {
                "section": "News",
                "name": "News CSV quality",
                "status": "OK",
                "details": {"status": "OK", "unknown_source_count": 0},
            },
            {
                "section": "SQLite",
                "name": "Missing live dates",
                "status": "OK",
                "details": {
                    "known_missed_live_dates": [],
                    "missing_live_date_diagnostics": [],
                },
            },
            {
                "section": "daily_predict",
                "name": "Windows scheduled task",
                "status": "OK",
                "details": {},
            },
            {
                "section": "daily_predict",
                "name": "Latest run",
                "status": "OK",
                "details": {"latest_success": "[2026-07-24 10:30:30] success"},
            },
        ],
    }
    scheduler = {
        "registered": True,
        "task_name": "stock-ai daily predict",
        "next_run_time": "2026-07-25 10:30:00",
        "last_run_time": "2026-07-24 10:30:00",
        "last_result": 0,
        "command": f'powershell -File "{project_root / "scripts" / "daily_predict.ps1"}"',
        "working_directory": str(project_root),
    }
    freshness = {
        "expected_latest_trading_date": "2026-07-24",
        "latest_raw_date": "2026-07-24",
        "latest_feature_date": "2026-07-24",
        "freshness_lag_trading_days": 0,
        "freshness_status": "OK",
        "stale_at_run_time": False,
    }
    monkeypatch.setattr(autonomous, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(autonomous, "load_settings", lambda: settings)
    monkeypatch.setattr(autonomous, "run_health_check", lambda **_kwargs: health)
    monkeypatch.setattr(autonomous, "_read_scheduler_status", lambda: dict(scheduler))
    monkeypatch.setattr(autonomous, "_daily_predict_task_inventory_count", lambda: 1)
    monkeypatch.setattr(autonomous, "_validate_database", lambda _path: {"readable": True})
    monkeypatch.setattr(autonomous, "read_prediction_history", lambda *_args, **_kwargs: pd.DataFrame())
    monkeypatch.setattr(autonomous, "_safe_read_parquet", lambda _path: pd.DataFrame())
    monkeypatch.setattr(autonomous, "summarize_daily_data_freshness", lambda **_kwargs: freshness)
    monkeypatch.setattr(
        autonomous,
        "_summarize_live_coverage",
        lambda *_args: {
            "latest_binary_feature_date": "2026-07-24",
            "latest_multiclass_feature_date": "2026-07-24",
            "both_saved_for_latest_date": True,
            "binary_confirmed_count": 1,
            "binary_pending_count": 0,
            "multiclass_confirmed_count": 1,
            "multiclass_pending_count": 0,
        },
    )
    monkeypatch.setattr(
        autonomous,
        "_summarize_live_binary_metrics",
        lambda *_args: {
            "live_rows": 1,
            "confirmed": 1,
            "pending": 0,
            "baselines": [],
            "_frame": pd.DataFrame(columns=["status", "confidence", "date"]),
        },
    )
    monkeypatch.setattr(
        autonomous,
        "_summarize_shadow",
        lambda _rows: {
            "file_status": "missing",
            "total_rows": 0,
            "confirmed_count": 0,
            "pending_count": 0,
            "conclusions": ["continue_shadow_monitoring"],
            "needs_review": False,
        },
    )
    monkeypatch.setattr(
        autonomous,
        "_summarize_append_only",
        lambda *_args: {
            "prediction_runs_count": 0,
            "prediction_records_count": 0,
            "existing_predictions_count": 1,
            "compare_status": "append_only_empty",
            "mismatch_count": 0,
            "selected_official_candidate_count": 0,
        },
    )
    monkeypatch.setattr(autonomous, "_diagnose_unknown_sources", lambda *_args: [])
    monkeypatch.setattr(autonomous, "_diagnose_missing_live", lambda *_args, **_kwargs: [])

    state_path = tmp_path / "out" / "state.json"
    queue_path = tmp_path / "out" / "queue.json"
    report_dir = tmp_path / "out" / "reports"
    before = {path: path.read_bytes() for path in production_files}

    result = autonomous.run_autonomous_operations(
        ticker="7203.T",
        state_path=state_path,
        task_queue_path=queue_path,
        report_dir=report_dir,
    )

    assert result["stop_reason"] == "completed_read_only_cycle"
    assert state_path.exists()
    assert queue_path.exists()
    assert Path(result["report_paths"]["markdown"]).exists()
    assert Path(result["report_paths"]["json"]).exists()
    assert {path: path.read_bytes() for path in production_files} == before
    markdown = Path(result["report_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "Production model changed: No" in markdown
    assert "Production SQLite changed: No" in markdown
    assert "Live/backfill prediction created: No" in markdown


def test_report_paths_never_overwrite_same_minute(tmp_path: Path) -> None:
    cycle_at = datetime(2026, 7, 24, 12, 34, tzinfo=UTC).isoformat()
    first = autonomous._report_paths(tmp_path, cycle_at, force=False)
    first["markdown"].write_text("first", encoding="utf-8")
    first["json"].write_text("{}", encoding="utf-8")

    second = autonomous._report_paths(tmp_path, cycle_at, force=True)

    assert second["markdown"] != first["markdown"]


def test_state_json_does_not_nest_previous_confidence_deltas(tmp_path: Path) -> None:
    state = autonomous.default_operations_state()
    confidence = {
        "thresholds": [{"threshold": 0.6, "count": 1}],
        "classification": "sample_too_small",
        "hit_rates": {"0.6": 1.0},
        "change_from_previous": {"0.6": 0.1},
    }
    updated = autonomous._next_state(
        state,
        cycle_at="2026-07-24T10:00:00+09:00",
        successful=True,
        health_summary={},
        freshness={},
        binary_metrics={},
        confidence=confidence,
        shadow={},
        append_only={},
        findings=[],
        report_path=str(tmp_path / "report.md"),
    )

    assert "change_from_previous" not in updated["last_confidence_metrics"]
    json.dumps(updated)
