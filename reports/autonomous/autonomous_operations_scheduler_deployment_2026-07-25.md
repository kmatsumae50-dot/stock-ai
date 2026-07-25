# Autonomous Operations v1.1 Scheduled Deployment

## Deployment

- Task name: `stock-ai autonomous operations`
- Trigger: daily at `11:15`
- Command: `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\Users\kmats\Documents\ai-stock\stock-ai\scripts\run_autonomous_operations.ps1" -DependencyTaskName "stock-ai daily predict" -DependencyWaitMinutes 30 -DependencyPollSeconds 60`
- Working directory: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- User/logon: `kmats`, interactive token, limited run level
- Start when available: enabled
- Battery start: allowed
- Stop on battery: disabled
- Multiple instances: `IgnoreNew`
- Execution time limit: 1 hour
- Automatic retry: disabled (`RestartCount: 0`)
- Duplicate task count: 1 task with the requested name

The existing `stock-ai daily predict` task remains scheduled at 10:30 with its
original command, working directory, battery settings, and `Last Result: 0`.

## Dependency Wait

The wrapper checks `stock-ai daily predict` before launching the read-only cycle.
If it is running, the wrapper polls every 60 seconds for up to 30 minutes. When it
finishes, Autonomous Operations starts. If the daily task is absent or previously
failed, Operations still starts and diagnoses that state; it does not rerun
`daily_predict.ps1`. A dependency timeout exits with code 125 and leaves a wrapper
log. The lock file prevents overlapping wrapper cycles and is removed on exit.

## Exit Codes

- `0`: completed cycle or known warnings only
- `2`: human review required; no automatic retry
- `124`: Autonomous Operations runtime timeout
- `125`: daily prediction dependency wait timeout
- Other nonzero values: technical failure

## Registration Verification

The registration script was first run with `-WhatIf`; no task was created. It was
then registered with `-Force` and read back from Task Scheduler. Re-running the
registration without `-Force` detected an exact match and made no change.

Manual trigger result:

- Last run time after wrapper-helper verification: `2026-07-25 09:38:06`
- Last result: `0`
- Next run time: `2026-07-25 11:15:00`
- Wrapper log: `logs/autonomous/operations_2026-07-25_093806.log`
- Operations report: `reports/autonomous/operations_2026-07-25_0938.md`
- JSON report: `reports/autonomous/operations_2026-07-25_0938.json`
- Stop reason: `completed_read_only_cycle`
- State and task queue were updated
- New queue tasks: none
- Queue fingerprints remained unique
- Lock file after completion: absent

## Production Safety

- Production database SHA-256 before: `318DF6831D6C5BDA6499BC2EE9C84F7B53B82250560E5B1A90214AACC47E7B52`
- Production database SHA-256 after: `318DF6831D6C5BDA6499BC2EE9C84F7B53B82250560E5B1A90214AACC47E7B52`
- `predictions`: 81 before and after
- `prediction_runs`: 1 before and after
- `prediction_records`: 2 before and after
- Live rows: 80 before and after
- Backfill rows: 1 before and after
- `daily_predict.ps1` SHA-256 remained `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2`
- No production model, threshold, feature-generation, SQLite, append-only,
  prediction-history, known-missed, news-source, or daily-task change was made
- Autonomous Research and Safe Maintenance were not implemented

## Verification

- Autonomous Operations scheduled manual run: exit 0
- Autonomous Operations CLI: exit 0
- Dependency behavior tests: not registered proceeds; Running then Ready proceeds;
  nonzero dependency result proceeds without retry; wait timeout stops safely
- Concurrent wrapper lock test: active lock blocks a second run
- Health check: `Action required 0`, `Known missed dates 4`, `Known OK 1`
- Ruff: OK
- Pytest: 349 passed
- Compileall: OK
- Streamlit: HTTP 200

## Rollback And Operations

Inspect the task:

```powershell
.\scripts\show_autonomous_operations_task.ps1
```

Temporarily pause and resume:

```powershell
Disable-ScheduledTask -TaskName "stock-ai autonomous operations"
Enable-ScheduledTask -TaskName "stock-ai autonomous operations"
```

Remove only this task:

```powershell
.\scripts\unregister_autonomous_operations_task.ps1 -Force
```

After the first scheduled 11:15 cycle, inspect Task Scheduler `Last Result`, the
latest `logs/autonomous/operations_*.log`, the generated Markdown/JSON report,
`operations_state.json`, and unresolved or approval-required queue entries. Exit
code 2 should be reviewed by a person and must not trigger an automatic retry.
