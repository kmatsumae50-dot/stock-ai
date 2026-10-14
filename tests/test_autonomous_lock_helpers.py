from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HELPER = PROJECT_ROOT / "scripts" / "autonomous_lock_helpers.ps1"


def _run_helper(body: str) -> dict[str, object]:
    command = f". '{HELPER}'; {body} | ConvertTo-Json -Depth 5 -Compress"
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip())


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell lock helper test")
def test_live_process_lock_is_never_recoverable(tmp_path: Path) -> None:
    lock = tmp_path / ".research.lock"
    lock.write_text(json.dumps({"pid": 123, "created_at": "2026-08-19T08:00:00+09:00"}), encoding="utf-8")

    result = _run_helper(
        f"Get-AutonomousLockState -LockPath '{lock}' -StaleAfterMinutes 150 "
        "-NowProvider {[datetime]'2026-08-19T12:00:00'} "
        "-ProcessReader {param($id) [pscustomobject]@{Id=$id}}"
    )

    assert result["Classification"] == "lock_active"
    assert result["CanRecover"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell lock helper test")
def test_recent_orphan_lock_does_not_meet_stale_threshold(tmp_path: Path) -> None:
    lock = tmp_path / ".research.lock"
    lock.write_text(json.dumps({"pid": 123, "created_at": "2026-08-19T11:00:00+09:00"}), encoding="utf-8")

    result = _run_helper(
        f"Get-AutonomousLockState -LockPath '{lock}' -StaleAfterMinutes 150 "
        "-NowProvider {[datetime]'2026-08-19T12:00:00'} -ProcessReader {param($id) $null}"
    )

    assert result["Classification"] == "lock_orphaned_recent"
    assert result["CanRecover"] is False


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell lock helper test")
def test_verified_stale_lock_is_quarantined_with_audit_result(tmp_path: Path) -> None:
    lock = tmp_path / ".research.lock"
    quarantine = tmp_path / "quarantine"
    lock.write_text(json.dumps({"pid": 123, "created_at": "2026-08-19T08:00:00+09:00"}), encoding="utf-8")

    result = _run_helper(
        f"$s=Get-AutonomousLockState -LockPath '{lock}' -StaleAfterMinutes 150 "
        "-NowProvider {[datetime]'2026-08-19T12:00:00'} -ProcessReader {param($id) $null}; "
        f"Move-AutonomousStaleLock -LockPath '{lock}' -QuarantineDirectory '{quarantine}' -LockState $s"
    )

    assert result["Status"] == "stale_lock_quarantined"
    assert not lock.exists()
    assert Path(result["QuarantinePath"]).exists()


@pytest.mark.skipif(sys.platform != "win32", reason="PowerShell lock helper test")
def test_unreadable_lock_is_not_recoverable(tmp_path: Path) -> None:
    lock = tmp_path / ".research.lock"
    lock.write_text("{broken", encoding="utf-8")

    result = _run_helper(f"Get-AutonomousLockState -LockPath '{lock}' -StaleAfterMinutes 150")

    assert result["Classification"] == "lock_unreadable"
    assert result["CanRecover"] is False
    assert lock.exists()
