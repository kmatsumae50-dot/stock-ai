# Environment Handoff And Recovery

Last reviewed: 2026-09-04

This document describes how to recover `stock-ai` on a replacement PC or after local storage failure. It does not authorize production prediction writes, database migration, model retraining, or Task Scheduler changes on the current machine.

Official project root:

```powershell
C:\Users\kmats\Documents\ai-stock\stock-ai
```

## Recovery Boundaries

Git can restore:

- source code under `src/`
- tracked configuration under `configs/`
- scripts, tests, and documentation
- tracked design and decision records

Git alone cannot restore:

- `data/` market, feature, news, and event files
- `database/` SQLite history
- `models/` production artifacts
- required autonomous runtime state, history, and saved patches that are not tracked
- `.venv/`, caches, logs, or Windows Task Scheduler registrations

A private Git remote is useful for source-code continuity, but it is not a complete project backup. Do not commit secrets, the production database, large model artifacts, or private raw data merely to make them remote-accessible.

## Backup Priority

Critical:

- `database/market.db` and any SQLite sidecar files captured consistently
- `models/artifacts/`
- `configs/`
- non-regenerable parts of `data/raw/`, `data/features/`, `data/news/`, and `data/events/`

High:

- `reports/autonomous/operations_state.json`
- `reports/autonomous/operations_task_queue.json`
- `reports/autonomous/research_state.json`
- `reports/autonomous/research_history.jsonl`
- `reports/autonomous/codex_research_agent_state.json`
- `reports/autonomous/codex_research_hypotheses.json`
- `reports/autonomous/patches/` and selected context/report files
- decision and evaluation reports needed to interpret production choices

Regenerable:

- `.venv/`
- Python, pytest, Ruff, mypy, and Streamlit caches
- `__pycache__/` and compile outputs
- logs and reports that can be recreated from preserved inputs

Stop scheduled writers before copying a live SQLite database. A copied database and its `-wal`/`-shm` files must represent one consistent point in time. Prefer an SQLite backup operation or a copy made while all writers are stopped.

## Backup Manifest

Before copying files, create a manifest outside the source tree. This is read-only with respect to project files.

```powershell
cd "C:\Users\kmats\Documents\ai-stock\stock-ai"

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$manifest = Join-Path $env:USERPROFILE "stock-ai-backup-manifest-$stamp.json"
$roots = @(
    "database",
    "models",
    "configs",
    "reports\autonomous",
    "reports\evaluation"
)

$items = foreach ($root in $roots) {
    Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue | ForEach-Object {
        [PSCustomObject]@{
            relative_path = [IO.Path]::GetRelativePath((Get-Location).Path, $_.FullName)
            size_bytes = $_.Length
            modified_at = $_.LastWriteTimeUtc.ToString("o")
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
}

$items | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Host "Manifest: $manifest"
```

Store the manifest with the backup. After restoration, generate a new manifest and compare path, size, and SHA-256 values.

## Python Environment Reconstruction

The current reviewed environment uses Python 3.14.5. The project metadata supports Python 3.11 or newer. On a new PC, prefer the same Python minor version as the backed-up environment when practical.

```powershell
cd "C:\Users\kmats\Documents\ai-stock\stock-ai"
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-freeze.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

If the frozen set is incompatible with the replacement Python, use `requirements.txt`, record the resulting versions, and run the full verification suite before trusting the environment.

## Windows Task Scheduler Reconstruction

Current tasks:

| Task | Schedule | Wrapper / command |
| --- | --- | --- |
| `stock-ai daily predict` | 10:30 daily | `daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3` |
| `stock-ai autonomous operations` | 11:15 daily | `run_autonomous_operations.ps1` with the daily task dependency |
| `stock-ai autonomous research` | 12:00 daily | `run_autonomous_research.ps1` with the Operations dependency |

All tasks currently use the official working directory, `StartWhenAvailable`, `MultipleInstances: IgnoreNew`, and allow battery execution without stopping on battery transition.

Review commands before registering on a replacement PC:

```powershell
.\scripts\register_daily_predict_task.ps1 -Time "10:30"
.\scripts\register_autonomous_operations_task.ps1 -Time "11:15"
.\scripts\register_autonomous_research_task.ps1 -Time "12:00"
```

Use `-Force` only after inspecting an existing task and intentionally replacing it. Registration is not part of a read-only restoration check.

Export the current definitions before migration:

```powershell
$out = Join-Path $env:USERPROFILE "stock-ai-scheduler-xml"
New-Item -ItemType Directory -Force -Path $out | Out-Null
foreach ($name in @(
    "stock-ai daily predict",
    "stock-ai autonomous operations",
    "stock-ai autonomous research"
)) {
    $safe = $name -replace '[^A-Za-z0-9._-]', '_'
    Export-ScheduledTask -TaskName $name | Set-Content -LiteralPath (Join-Path $out "$safe.xml") -Encoding Unicode
}
```

Record both `scheduled_for` and `actual_started_at`. Because `StartWhenAvailable` is enabled, a task started after resume is a catch-up run and must not be described as a prediction produced at its scheduled time.

## Path Migration

The intended root is:

```text
C:\Users\kmats\Documents\ai-stock\stock-ai
```

Search tracked files for embedded paths after moving:

```powershell
git grep -n -I -E "C:\\\\Users\\\\kmats|ai-stock|ai 株"
```

Also inspect Task Scheduler actions and runtime JSON/Markdown records. Historical report paths may legitimately point to the old machine; executable scripts and active task actions must point to the restored root.

## Restoration Verification

Do not save a real prediction merely to test restoration. Use read-only checks and dry-run commands where available.

```powershell
cd "C:\Users\kmats\Documents\ai-stock\stock-ai"
.\.venv\Scripts\python.exe main.py check-news --ticker 7203.T
.\.venv\Scripts\python.exe main.py health-check --ticker 7203.T
.\.venv\Scripts\python.exe main.py data-freshness --ticker 7203.T
.\.venv\Scripts\python.exe main.py prediction-append-only-check --ticker 7203.T
.\.venv\Scripts\python.exe main.py binary-shadow-status --ticker 7203.T
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m compileall main.py src tests
```

Then verify:

- Streamlit returns HTTP 200 and displays the restored project root.
- SQLite table counts match the backup manifest or recorded handoff report.
- production model and configuration hashes match the handoff report.
- all three scheduled task commands, working directories, schedules, and settings match the reviewed definitions.
- no unexpected post-start missing live date or unknown news source exists.
- no active lock or research worktree was carried across unintentionally.

Any prediction test must default to dry-run. Do not use `source=live` for restoration validation, and do not create backfill merely to make counts match.
