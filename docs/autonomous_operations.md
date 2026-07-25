# Autonomous Operations v1.1

Autonomous Operations v1 is a read-only monitoring cycle for the formal
`stock-ai` workspace. It replaces repeated manual inspection with one command,
but it does not repair production state.

## Run

```powershell
.\.venv\Scripts\python.exe main.py autonomous-operations --ticker 7203.T
.\scripts\run_autonomous_operations.ps1
```

The PowerShell wrapper verifies the formal project root, uses `.venv`, prevents
concurrent execution with a lock file, applies a timeout, and writes a timestamped
log. Before starting the diagnostic cycle, it checks the `stock-ai daily predict`
task. If daily prediction is still running, the wrapper polls until it finishes or
the dependency wait timeout is reached. A missing or previously failed daily task
does not trigger a retry; Autonomous Operations proceeds and reports the condition.

Dependency timing can be adjusted without editing the script:

```powershell
.\scripts\run_autonomous_operations.ps1 `
  -DependencyTaskName "stock-ai daily predict" `
  -DependencyWaitMinutes 30 `
  -DependencyPollSeconds 60
```

## Scheduled Deployment

The recommended Windows Task Scheduler deployment runs daily at 11:15, after the
10:30 daily prediction task:

```powershell
# Preview only; Task Scheduler is not changed.
.\scripts\register_autonomous_operations_task.ps1 -Time "11:15" -WhatIf

# Register or safely update after reviewing the preview.
.\scripts\register_autonomous_operations_task.ps1 -Time "11:15" -Force

# Read back the registered configuration and latest result.
.\scripts\show_autonomous_operations_task.ps1

# Remove only the Autonomous Operations task.
.\scripts\unregister_autonomous_operations_task.ps1 -Force
```

Task name: `stock-ai autonomous operations`

Command:

```text
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kmats\Documents\ai-stock\stock-ai\scripts\run_autonomous_operations.ps1" -DependencyTaskName "stock-ai daily predict" -DependencyWaitMinutes 30 -DependencyPollSeconds 60
```

The working directory is the formal project root. The task uses
`StartWhenAvailable`, permits battery operation, and uses `IgnoreNew` so a second
instance is not started while one is running. Automatic retries are disabled.
If the scheduled start is missed while the PC is off, Windows starts it when the
PC next becomes available. The wrapper then checks the current daily task state.

The task runs as the current interactive user with limited run level. Registration
normally does not require elevation for that user, but local Windows policy may
require an Administrator PowerShell session.

Exit codes:

- `0`: cycle completed; only known/informational warnings may remain.
- `2`: human review is required. This is not retried automatically.
- `124`: Autonomous Operations exceeded its runtime limit.
- `125`: the daily prediction dependency remained running past the wait limit.
- Other nonzero values: technical failure; inspect the wrapper log and report.

## Outputs

- State: `reports/autonomous/operations_state.json`
- Task queue: `reports/autonomous/operations_task_queue.json`
- Reports: `reports/autonomous/operations_YYYY-MM-DD_HHMM.md` and `.json`
- Wrapper logs: `logs/autonomous/operations_YYYY-MM-DD_HHMMSS.log`

State and task queue files are monitoring artifacts. They are separate from
production data and configuration.

## Scope

Every cycle reads the latest daily run, data freshness, Task Scheduler state,
health-check, unknown news sources, missing live dates, live binary metrics,
simple baselines, confidence thresholds, existing shadow results, and append-only
audit status.

Unknown sources and missing live dates may create review tasks with evidence and
conservative proposals. They never edit source profiles, known-missed settings,
SQLite prediction history, or live/backfill rows.

## Risk Levels

- Level 0: read-only inspection, reporting, state updates, and task queue updates.
- Level 1: a later code, test, UI, or documentation change. v1 only queues it.
- Level 2: production model, threshold, feature, config, SQLite, scheduler, or
  prediction-history change. Human approval is required; v1 never executes it.

## Safety

The cycle stops safely when core health diagnostics or the production prediction
database cannot be read, state cannot be trusted, multiple unexplained operational
problems require review, a runtime limit is reached, or a protected production
artifact changes while the cycle is running.

The report always records whether the production model, production SQLite,
`daily_predict.ps1`, Task Scheduler, or live/backfill predictions changed. Under
normal operation each value must remain `No`.

Autonomous Research, Safe Maintenance, automatic code changes, commits, merges,
model experiments, source-profile edits, and production deployment remain outside
v1.1. Scheduled deployment runs the existing read-only Operations cycle only.

## Pause And Resume

```powershell
Disable-ScheduledTask -TaskName "stock-ai autonomous operations"
Enable-ScheduledTask -TaskName "stock-ai autonomous operations"
```

Disabling or unregistering this task does not alter `stock-ai daily predict`.
