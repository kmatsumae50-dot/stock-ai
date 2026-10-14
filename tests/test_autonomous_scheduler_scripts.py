from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"


def _script_text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8-sig")


def test_autonomous_wrapper_keeps_safety_controls_and_dependency_wait() -> None:
    text = _script_text("run_autonomous_operations.ps1")

    assert '"C:\\Users\\kmats\\Documents\\ai-stock\\stock-ai"' in text
    assert '.venv\\Scripts\\python.exe' in text
    assert ".operations.lock" in text
    assert "finally {" in text
    assert "Remove-Item -LiteralPath $LockPath -Force" in text
    assert "RedirectStandardOutput = $true" in text
    assert "RedirectStandardError = $true" in text
    assert "DependencyTaskName" in text
    assert "DependencyWaitMinutes" in text
    assert "DependencyPollSeconds" in text
    assert "Wait-AutonomousDependencyTask" in text
    assert "$ExitCode = 125" in text
    assert "$ExitCode = [int]$Process.ExitCode" in text
    assert "exit $ExitCode" in text
    assert "StaleLockMinutes" in text
    assert "Move-AutonomousStaleLock" in text


def _run_dependency_helper(body: str) -> dict[str, object]:
    helper = SCRIPTS / "autonomous_operations_wrapper_helpers.ps1"
    command = f". '{helper}'; {body} | ConvertTo-Json -Depth 5 -Compress"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_dependency_task_not_registered_proceeds_without_waiting() -> None:
    result = _run_dependency_helper(
        "Wait-AutonomousDependencyTask -TaskName 'missing' -WaitMinutes 30 "
        "-PollSeconds 60 -TaskReader { param($name) $null }"
    )

    assert result["ShouldProceed"] is True
    assert result["Status"] == "not_registered_proceeding"


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_dependency_task_running_then_ready_proceeds() -> None:
    result = _run_dependency_helper(
        "$script:reads=0; "
        "Wait-AutonomousDependencyTask -TaskName 'daily' -WaitMinutes 30 -PollSeconds 1 "
        "-TaskReader { param($name) $script:reads++; if ($script:reads -eq 1) "
        "{ [pscustomobject]@{State='Running'} } else { [pscustomobject]@{State='Ready'} } } "
        "-TaskInfoReader { param($name) [pscustomobject]@{LastRunTime=[datetime]'2026-07-25T10:30:00'; LastTaskResult=0} } "
        "-SleepAction { param($seconds) }"
    )

    assert result["ShouldProceed"] is True
    assert result["Status"] == "ready_to_proceed"
    assert result["LastResult"] == 0
    assert any("dependency_wait:" in line for line in result["LogLines"])


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_dependency_wait_timeout_stops_before_operations() -> None:
    result = _run_dependency_helper(
        "Wait-AutonomousDependencyTask -TaskName 'daily' -WaitMinutes 0 -PollSeconds 1 "
        "-TaskReader { param($name) [pscustomobject]@{State='Running'} }"
    )

    assert result["ShouldProceed"] is False
    assert result["Status"] == "wait_timeout"


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell dependency helper test")
def test_failed_dependency_result_still_allows_diagnostics() -> None:
    result = _run_dependency_helper(
        "Wait-AutonomousDependencyTask -TaskName 'daily' -WaitMinutes 30 -PollSeconds 60 "
        "-TaskReader { param($name) [pscustomobject]@{State='Ready'} } "
        "-TaskInfoReader { param($name) [pscustomobject]@{LastRunTime=[datetime]'2026-07-25T10:30:00'; LastTaskResult=1} }"
    )

    assert result["ShouldProceed"] is True
    assert result["LastResult"] == 1
    assert any("without retrying daily_predict" in line for line in result["LogLines"])


@pytest.mark.skipif(sys.platform != "win32", reason="Windows lock integration")
def test_active_lock_prevents_concurrent_wrapper_run(tmp_path: Path) -> None:
    report_dir = tmp_path / "autonomous"
    report_dir.mkdir()
    lock_path = report_dir / ".operations.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "created_at": "2026-07-25T00:00:00+09:00"}),
        encoding="utf-8",
    )
    wrapper = SCRIPTS / "run_autonomous_operations.ps1"

    result = subprocess.run(
        [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(wrapper),
            "-ReportDir",
            str(report_dir),
        ],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Autonomous Operations lock stop" in (result.stdout + result.stderr)
    assert lock_path.exists()


def test_autonomous_registration_is_daily_idempotent_and_has_no_retry() -> None:
    text = _script_text("register_autonomous_operations_task.ps1")

    assert "SupportsShouldProcess = $true" in text
    assert '"stock-ai autonomous operations"' in text
    assert '"11:15"' in text
    assert "New-ScheduledTaskTrigger -Daily" in text
    assert "-StartWhenAvailable" in text
    assert "-MultipleInstances IgnoreNew" in text
    assert "-AllowStartIfOnBatteries" in text
    assert "-DontStopIfGoingOnBatteries" in text
    assert "Existing task already matches" in text
    assert "Use -Force to update" in text
    assert "Automatic retry: disabled" in text
    assert "New-ScheduledTaskSettingsSet" in text
    assert "Settings.RestartCount -eq 0" in text
    assert "Settings.ExecutionTimeLimit -eq \"PT1H\"" in text
    assert "Settings.DisallowStartIfOnBatteries" in text


def test_autonomous_unregister_only_targets_named_task() -> None:
    text = _script_text("unregister_autonomous_operations_task.ps1")

    assert "SupportsShouldProcess = $true" in text
    assert '"stock-ai autonomous operations"' in text
    assert "Unregister-ScheduledTask -TaskName $TaskName" in text
    assert "stock-ai daily predict task was not changed" in text


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Task Scheduler integration")
def test_autonomous_registration_whatif_does_not_create_task() -> None:
    task_name = "stock-ai autonomous operations pytest dry run"
    powershell = "powershell.exe"
    script = SCRIPTS / "register_autonomous_operations_task.ps1"
    unregister = [
        powershell,
        "-NoProfile",
        "-Command",
        f"Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue | "
        "Unregister-ScheduledTask -Confirm:$false",
    ]
    subprocess.run(unregister, check=False, capture_output=True, text=True)

    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-TaskName",
            task_name,
            "-Time",
            "11:16",
            "-WhatIf",
        ],
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert "What if" in result.stdout
    check = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-Command",
            f"if (Get-ScheduledTask -TaskName '{task_name}' -ErrorAction SilentlyContinue) "
            "{ exit 1 } else { exit 0 }",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0
