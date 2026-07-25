# Autonomous Research v1.1 Scheduled Deployment

## Deployment Summary

- Deployment date: 2026-07-25
- Project root: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- Task name: `stock-ai autonomous research`
- Status: registered, enabled, and manually verified
- Trigger: daily at 12:00
- Next run at verification: 2026-07-26 12:00:00
- Working directory: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- Command: `C:\WINDOWS\System32\WindowsPowerShell\v1.0\powershell.exe`
- Arguments: `-NoProfile -ExecutionPolicy Bypass -File "C:\Users\kmats\Documents\ai-stock\stock-ai\scripts\run_autonomous_research.ps1" -OperationsTaskName "stock-ai autonomous operations" -DependencyWaitMinutes 30 -DependencyPollSeconds 60 -OperationsFreshnessMinutes 180`
- StartWhenAvailable: enabled
- MultipleInstances: IgnoreNew
- Battery execution: allowed; the task is not stopped when switching to battery
- Automatic retry: disabled (`RestartCount=0`)
- Task execution time limit: 130 minutes
- Wrapper runtime limit: 120 minutes

## Dependency Contract

Research is third in the daily sequence: daily prediction at 10:30, Autonomous Operations at 11:15, and Autonomous Research at 12:00. The Research wrapper reads but never starts or modifies the Operations task.

If Operations is Running, Research polls every 60 seconds for up to 30 minutes. A timeout exits 125. Research proceeds only when the latest Operations Task Scheduler result is zero and its latest date-stamped JSON report is healthy. The report must be from the current day, no more than 180 minutes old, and correspond to the latest task run. It must report `completed_read_only_cycle`, zero Action required items, zero unknown sources, no unresolved missing-live findings, fresh data, and a readable production database.

Operations result 2 or a stale/unhealthy cycle produces Research exit 3 without a heavy experiment. Other nonzero Operations results are technical failures and prevent Research. Operations is never retried by Research.

## Scheduler Policy

- `0`: completed, no material change, duplicate-fingerprint skip, or not-due skip
- `2`: human-review candidate; no production change and no automatic retry
- `3`: deliberate unhealthy-Operations skip; no automatic retry
- `124`: Research runtime timeout; no automatic retry
- `125`: dependency wait or lock safety stop; no automatic retry
- Other nonzero: technical failure

The scheduled command does not use `--force`. Trigger policy and experiment fingerprinting keep routine daily launches lightweight.

## Dry Run And Registration

`register_autonomous_research_task.ps1 -WhatIf` completed without creating a task. The actual registration then created exactly one task named `stock-ai autonomous research`. A second registration detected a complete match and made no change, confirming idempotency. The read-back showed `Ready`, enabled, daily 12:00, `StartWhenAvailable=True`, `IgnoreNew`, battery execution enabled, `RestartCount=0`, and `PT2H10M` execution limit.

The existing tasks remained configured as before:

- `stock-ai daily predict`: daily 10:30, formal project root, `-FetchNews -DailyNewsProfiles -NewsLimit 3`, Last Result 0
- `stock-ai autonomous operations`: daily 11:15, formal project root, `StartWhenAvailable=True`, `IgnoreNew`, Last Result 0

## Manual Trigger Result

- Manual Task Scheduler run: 2026-07-25 13:27:05 JST
- Task state after run: Ready
- Last Result: 0
- Operations task Last Result observed by wrapper: 0
- Operations report: `reports/autonomous/operations_2026-07-25_1222.json`
- Operations cycle completed: 2026-07-25 12:22:44 JST
- Operations report age at gate: about 64 minutes
- Dependency result: `ready_to_research`
- Research classification: `research_skipped_duplicate_experiment`
- Heavy experiment executed: false
- Duplicate fingerprint skip: true
- Fingerprint: `38a44da802307be5a4b14998848e428974954ec77027e7906b4dd86f0f91119a`
- Recommendation: `production_change_not_ready; Research was safely skipped`
- Wrapper log: `logs/autonomous/research_2026-07-25_1327.log`
- Report: `reports/autonomous/research_2026-07-25_1327.md` and `.json`
- State updated: yes, with the normal lightweight cycle result
- History updated: yes, append-only history increased from 8 to 9 lines for the scheduled manual run
- Review task duplication: none; the one model-research task remained resolved and no active duplicate was created
- Research lock remaining: no
- Operations lock remaining: no

A subsequent required direct CLI check also skipped the identical fingerprint with exit 0 and wrote `research_2026-07-25_1328.*`. History then contained 10 lines.

## Production Guards

The following production values were unchanged before and after Research deployment and manual execution:

| Guard | Before | After |
|---|---|---|
| `database/market.db` SHA-256 | `7CFA733FD5EFF9FEAF234C8E2E5AE4E67AFB5F097466E9BA0D61ED3ACC1F25AA` | same |
| predictions rows | 81 | 81 |
| prediction_runs rows | 1 | 1 |
| prediction_records rows | 2 | 2 |
| binary artifact SHA-256 | `EA947A34A711F560236F89411EA25B58E1F58337A6F0BBF497E9ED2B10D626FB` | same |
| multiclass artifact SHA-256 | `CBA07F0A648884876EDF65CBA691EE54987B22A48E66AC138B91306AB02F300D` | same |
| `configs/settings.yml` SHA-256 | `F9818647DC268C134227427E6376CAD235872B6A217E504BC29868950C583431` | same |
| `scripts/daily_predict.ps1` SHA-256 | `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2` | same |
| Operations wrapper SHA-256 | `669BB3027E669DB767BCA45CEF55648B5E91E89DD94606765F426C2F5BECF0DD` | same |
| Operations implementation SHA-256 | `CA424ECC234364399A0B7D4DAD3C56CD02F8B4EA6A108CE8E58745AAC3AB7775` | same |

No production model, artifact, config, SQLite row, existing prediction, live prediction, backfill prediction, daily wrapper, Operations implementation, or Operations wrapper changed. The manual Research task run did not change `operations_state.json` or `operations_task_queue.json`; their hashes still matched the pre-deployment baseline immediately afterward. The separately requested final Autonomous Operations validation cycle later updated its own normal state, queue, and report outputs. That was an expected read-only monitoring-cycle output, not a Research mutation or code change.

Safe Maintenance, Git auto-commit, and Git merge were not implemented.

## Validation

- Registration dry-run: passed
- Registration/read-back/idempotency: passed
- Manual scheduled trigger: passed, Last Result 0
- Normal Research CLI: passed, duplicate-fingerprint skip
- Autonomous Operations read-only CLI: passed, `completed_read_only_cycle`
- Targeted scheduler/research tests: passed
- Ruff: all checks passed
- Pytest: 381 passed
- Compileall: passed
- Health-check: Overall WARN, Action required 0, Known missed dates 4, Known OK 1; WARNs are known operational items
- Streamlit: formal `.venv` process on port 8501 returned HTTP 200

## Rollback

Preview removal:

```powershell
.\scripts\unregister_autonomous_research_task.ps1 -WhatIf
```

Remove only the Research task:

```powershell
.\scripts\unregister_autonomous_research_task.ps1 -Force
```

The unregister helper refuses any task name other than `stock-ai autonomous research`, so it cannot remove the daily prediction or Autonomous Operations tasks.

## First Scheduled Run Checklist

After the first regular 12:00 run, check:

1. Task Last Result and Last Run Time via `show_autonomous_research_task.ps1`.
2. Latest `logs/autonomous/research_*.log` dependency and exit lines.
3. Latest Research Markdown and JSON report.
4. `research_state.json` and the newest line of `research_history.jsonl`.
5. `model_research` items in `operations_task_queue.json` for duplicates.
6. Absence of `.research.lock` and `.operations.lock`.
7. `experiment_executed`, trigger reasons, fingerprint, and recommendation.

The expected common outcome is Last Result 0 with a not-due or duplicate-fingerprint skip. A heavy experiment should run only when the established trigger policy says it is due.
