from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _script_text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8-sig")


def _healthy_report(path: Path, *, completed_at: str = "2026-07-25T11:20:00") -> None:
    payload = {
        "cycle_completed_at": completed_at,
        "stop_reason": "completed_read_only_cycle",
        "health_summary": {"action_required_count": 0, "unknown_source_count": 0},
        "unknown_source_findings": [],
        "missing_live_findings": [],
        "data_freshness": {"freshness_status": "OK", "stale_at_run_time": False},
        "database_validation": {"readable": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_dependency_helper(body: str) -> dict[str, object]:
    helper = SCRIPTS / "autonomous_research_wrapper_helpers.ps1"
    command = f". '{helper}'; {body} | ConvertTo-Json -Depth 6 -Compress"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_running_operations_waits_then_healthy_report_starts_research(tmp_path: Path) -> None:
    _healthy_report(tmp_path / "operations_2026-07-25_1120.json")
    result = _run_dependency_helper(
        f"$script:reads=0; Wait-AutonomousResearchDependency -TaskName 'ops' "
        f"-ReportDir '{tmp_path}' -WaitMinutes 30 -PollSeconds 1 -FreshnessMinutes 180 "
        "-TaskReader { param($name) $script:reads++; if ($script:reads -eq 1) "
        "{ [pscustomobject]@{State='Running'} } else { [pscustomobject]@{State='Ready'} } } "
        "-TaskInfoReader { param($name) [pscustomobject]@{LastRunTime=[datetime]'2026-07-25T11:15:00'; LastTaskResult=0} } "
        "-SleepAction { param($seconds) } -NowProvider { [datetime]'2026-07-25T12:00:00' }"
    )

    assert result["ShouldProceed"] is True
    assert result["Status"] == "ready_to_research"
    assert result["ExitCode"] == 0
    assert any("dependency_wait:" in line for line in result["LogLines"])


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
@pytest.mark.parametrize(
    ("last_result", "status", "exit_code"),
    [
        (2, "operations_human_review_required", 3),
        (1, "operations_technical_failure", 1),
    ],
)
def test_nonzero_operations_result_prevents_research(
    tmp_path: Path,
    last_result: int,
    status: str,
    exit_code: int,
) -> None:
    _healthy_report(tmp_path / "operations_2026-07-25_1120.json")
    result = _run_dependency_helper(
        f"Wait-AutonomousResearchDependency -TaskName 'ops' -ReportDir '{tmp_path}' "
        "-WaitMinutes 30 -PollSeconds 60 -FreshnessMinutes 180 "
        "-TaskReader { param($name) [pscustomobject]@{State='Ready'} } "
        f"-TaskInfoReader {{ param($name) [pscustomobject]@{{LastRunTime=[datetime]'2026-07-25T11:15:00'; LastTaskResult={last_result}}} }} "
        "-NowProvider { [datetime]'2026-07-25T12:00:00' }"
    )

    assert result["ShouldProceed"] is False
    assert result["Status"] == status
    assert result["ExitCode"] == exit_code


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_stale_operations_report_is_not_treated_as_healthy(tmp_path: Path) -> None:
    _healthy_report(
        tmp_path / "operations_2026-07-24_1120.json",
        completed_at="2026-07-24T11:20:00",
    )
    result = _run_dependency_helper(
        f"Wait-AutonomousResearchDependency -TaskName 'ops' -ReportDir '{tmp_path}' "
        "-WaitMinutes 30 -PollSeconds 60 -FreshnessMinutes 180 "
        "-TaskReader { param($name) [pscustomobject]@{State='Ready'} } "
        "-TaskInfoReader { param($name) [pscustomobject]@{LastRunTime=[datetime]'2026-07-25T11:15:00'; LastTaskResult=0} } "
        "-NowProvider { [datetime]'2026-07-25T12:00:00' }"
    )

    assert result["ShouldProceed"] is False
    assert result["Status"] == "operations_report_unhealthy"
    assert result["ExitCode"] == 3
    assert any("operations_report_stale" in line for line in result["LogLines"])


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_operations_dependency_wait_timeout_uses_exit_125(tmp_path: Path) -> None:
    result = _run_dependency_helper(
        f"Wait-AutonomousResearchDependency -TaskName 'ops' -ReportDir '{tmp_path}' "
        "-WaitMinutes 0 -PollSeconds 1 -FreshnessMinutes 180 "
        "-TaskReader { param($name) [pscustomobject]@{State='Running'} }"
    )

    assert result["ShouldProceed"] is False
    assert result["Status"] == "wait_timeout"
    assert result["ExitCode"] == 125


def test_research_wrapper_keeps_lock_timeout_and_dependency_gate() -> None:
    text = _script_text("run_autonomous_research.ps1")

    assert '"C:\\Users\\kmats\\Documents\\ai-stock\\stock-ai"' in text
    assert ".venv\\Scripts\\python.exe" in text
    assert ".research.lock" in text
    assert "Wait-AutonomousResearchDependency" in text
    assert "OperationsFreshnessMinutes" in text
    assert "exit ([int]$dependency.ExitCode)" in text
    assert "Research exceeded the wrapper runtime limit" in text
    assert "safe_operations_unhealthy_skip" in text
    assert "StaleLockMinutes" in text
    assert "Move-AutonomousStaleLock" in text
    assert "exit $exitCode" in text
    assert "finally {" in text


def test_research_task_registration_is_idempotent_and_has_no_retry() -> None:
    text = _script_text("register_autonomous_research_task.ps1")

    assert "SupportsShouldProcess = $true" in text
    assert '"stock-ai autonomous research"' in text
    assert '"12:00"' in text
    assert "New-ScheduledTaskTrigger -Daily" in text
    assert "-StartWhenAvailable" in text
    assert "-MultipleInstances IgnoreNew" in text
    assert "-AllowStartIfOnBatteries" in text
    assert "-DontStopIfGoingOnBatteries" in text
    assert "Existing task already matches" in text
    assert "Use -Force to update" in text
    assert "Automatic retry: disabled" in text
    assert "Settings.RestartCount -ne 0" in text
    assert 'ExecutionTimeLimit -ne "PT2H10M"' in text


def test_research_unregister_refuses_other_task_names() -> None:
    text = _script_text("unregister_autonomous_research_task.ps1")

    assert "$TaskName -ne $ExpectedTaskName" in text
    assert "Refusing to unregister any task except" in text
    assert "Unregister-ScheduledTask -TaskName $TaskName" in text
    assert "daily prediction and Autonomous Operations tasks were not changed" in text


def test_research_show_script_exposes_required_scheduler_fields() -> None:
    text = _script_text("show_autonomous_research_task.ps1")

    for field in (
        "state",
        "enabled",
        "trigger",
        "next_run_time",
        "last_run_time",
        "last_result",
        "command",
        "arguments",
        "working_directory",
        "start_when_available",
        "multiple_instances",
        "allow_start_if_on_batteries",
        "execution_time_limit",
    ):
        assert field in text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Task Scheduler integration")
def test_research_registration_whatif_does_not_create_task() -> None:
    task_name = "stock-ai autonomous research pytest dry run"
    script = SCRIPTS / "register_autonomous_research_task.ps1"
    subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue | "
            "Unregister-ScheduledTask -Confirm:$false",
        ],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TaskName",
            task_name,
            "-Time",
            "12:01",
            "-WhatIf",
        ],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        errors="replace",
    )

    assert result.returncode == 0
    assert "What if" in result.stdout
    check = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            f"if (Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue) "
            "{ exit 1 } else { exit 0 }",
        ],
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    assert check.returncode == 0
