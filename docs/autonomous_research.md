# Autonomous Research v1

Autonomous Research is a gated, read-only research workflow for the binary direction model. It reuses the existing walk-forward and shadow evaluation code, compares only a bounded candidate set, and writes reports and review tasks. It never applies a candidate to production.

## Command

```powershell
.\.venv\Scripts\python.exe main.py autonomous-research --ticker 7203.T
```

Options include `--dry-run`, `--force`, `--state-path`, `--history-path`, `--report-dir`, `--max-runtime-minutes`, `--since-date`, repeatable `--candidate`, `--skip-shadow`, and `--skip-offline`.

The default candidate set is limited to:

- `production_current`
- `class_weight_balanced`
- `scale_pos_weight_medium`

Unsupported candidate names are reported as unavailable. Candidate count is capped at six. No grid search, new model family, feature generation change, model promotion, or artifact overwrite occurs.

## Trigger And Gate

A heavy offline cycle is due on the first run, with `--force`, after five new confirmed live or shadow observations, seven days after the prior heavy cycle, after a material live metric change, or after three new shadow disagreements. Material changes use fixed policy thresholds: 0.05 balanced accuracy and 0.10 up recall.

Before a heavy cycle, Research reads the latest Autonomous Operations JSON report. It requires a completed read-only Operations cycle, zero Action required items, zero unknown sources, no unresolved post-start missing live date, fresh raw/features, and a readable production database. Known-missed dates do not block Research. An unhealthy gate produces a report and exit code 3 without running the offline models.

Research also validates the next-close binary target and compares production guard hashes. A corrupt state file is never deleted silently.

## Fingerprint And Duplicate Prevention

The deterministic experiment fingerprint includes ticker, latest feature date, confirmed live and shadow counts, candidate set, evaluation windows, binary artifact hash, feature/config hash, target definition, and evaluation schema version. It excludes runtime and report names. A normal repeated fingerprint skips the heavy experiment and references the current state. `--force` is the only override.

## Evaluation

Research reports historical all, years 2023 through 2026, recent six and three months, and a recent live-like holdout. Small samples are marked and are not promoted. Metrics include accuracy, balanced accuracy, class recall and precision, class F1, ROC AUC, Brier score, counts, confusion matrix, confidence thresholds, production deltas, and baseline deltas.

Baselines are always-up, always-down, majority class, previous-day direction, and moving-average trend. A candidate below simple baselines cannot become a proposal.

The ranking score is deliberately multi-factor and multi-window: 45% balanced accuracy, 20% up recall, 15% down recall, 10% worst baseline edge, and 10% production edge. Window weights favor broad history. The recent three-month window is reported but receives no ranking weight.

Proposal readiness additionally requires broad historical improvement, improved up recall, guarded down recall, multiple improving windows, baseline superiority, a live-like sample of at least 30, and non-contradictory shadow evidence. Fewer than ten confirmed shadow rows yields continued shadow monitoring, not promotion.

## State, History, And Tasks

- State: `reports/autonomous/research_state.json`
- Append-only history: `reports/autonomous/research_history.jsonl`
- Reports: `reports/autonomous/research_YYYY-MM-DD_HHMM.md` and `.json`
- Shared queue: `reports/autonomous/operations_task_queue.json`

Only true candidate proposals or material live model degradation create Level 2 human-review tasks. Fingerprints prevent duplicate tasks. Research updates only `model_research` tasks and preserves unrelated Operations tasks.

## Wrapper

```powershell
.\scripts\run_autonomous_research.ps1
```

The wrapper fixes the working directory and virtual environment, holds `reports/autonomous/.research.lock`, enforces a maximum runtime of 120 minutes, logs stdout and stderr to `logs/autonomous/research_YYYY-MM-DD_HHMM.log`, and passes through the CLI exit code.

Before starting Research, the wrapper checks the `stock-ai autonomous operations` task read-only. If Operations is still running, Research waits up to 30 minutes and polls every 60 seconds. Research starts only when the latest Operations task result is zero and the latest date-stamped Operations report:

- completed on the current day and is no more than 180 minutes old
- corresponds to the latest Operations task run
- has `completed_read_only_cycle`
- has no Action required item, unknown source, or unresolved missing-live finding
- reports fresh raw and feature data
- reports a readable production database

Research never starts or retries Operations. Operations result `2`, an unhealthy or stale report, and an Operations cycle that did not run today produce a Research exit `3`. A dependency wait timeout produces `125`. Other Operations technical failures prevent Research and remain technical failures.

Exit codes:

- `0`: completed, no material change, or normal skip
- `2`: human review is required
- `3`: skipped because Operations is unhealthy
- `124`: runtime timeout
- `125`: lock/dependency safe stop
- other nonzero: technical failure

Task Scheduler does not retry any exit code automatically. In particular, result `2` means a human-review candidate exists, result `3` is a deliberate unhealthy-dependency skip rather than a crash, and duplicate-fingerprint or not-due skips are normal result `0` outcomes.

## Scheduled Deployment

The daily sequence is:

1. `stock-ai daily predict` at 10:30
2. `stock-ai autonomous operations` at 11:15
3. `stock-ai autonomous research` at 12:00

The Research task uses `StartWhenAvailable`, allows battery execution, and uses `IgnoreNew` to prevent overlap. Its Task Scheduler execution limit is 130 minutes, leaving the wrapper's 120-minute timeout authoritative. It has no automatic restart policy. Research is safe to launch every day because trigger policy and the experiment fingerprint keep most cycles lightweight; a heavy experiment does not run every day.

Preview and register:

```powershell
.\scripts\register_autonomous_research_task.ps1 -WhatIf
.\scripts\register_autonomous_research_task.ps1
```

If an existing task differs, review the displayed fields and use `-Force` intentionally:

```powershell
.\scripts\register_autonomous_research_task.ps1 -Force
```

Inspect, disable, enable, manually run, and unregister:

```powershell
.\scripts\show_autonomous_research_task.ps1
Disable-ScheduledTask -TaskName "stock-ai autonomous research"
Enable-ScheduledTask -TaskName "stock-ai autonomous research"
Start-ScheduledTask -TaskName "stock-ai autonomous research"
.\scripts\unregister_autonomous_research_task.ps1 -WhatIf
.\scripts\unregister_autonomous_research_task.ps1 -Force
```

The unregister helper refuses to target any name other than `stock-ai autonomous research`. It does not alter the daily prediction or Autonomous Operations tasks. Do not add `--force` to the scheduled command; forced experiments are manual diagnostics only.

## Scheduled Run Review

After the first scheduled 12:00 run, check only these operational items first:

- Research task Last Result and Last Run Time
- latest `logs/autonomous/research_*.log`
- latest Research Markdown and JSON reports
- `reports/autonomous/research_state.json`
- `reports/autonomous/research_history.jsonl`
- the `model_research` entries in `reports/autonomous/operations_task_queue.json`
- absence of `.research.lock` and `.operations.lock`
- `experiment_executed`, trigger reasons, fingerprint, and recommendation

The state records the most recent trigger and fingerprint. History is append-only. Reports explain whether the heavy experiment executed or was skipped. A duplicate fingerprint should be read as a healthy lightweight cycle, not a failure.

## Safety Contract

Research does not change production models, artifacts, model settings, SQLite, existing predictions, live/backfill rows, feature generation, target definitions, `daily_predict.ps1`, the existing daily/Operations tasks, Autonomous Operations, NewsLimit, daily profiles, known-missed dates, source profiles, earnings data, append-only policy, or shadow persistence. It does not perform Safe Maintenance, commit, or merge code automatically.

The first normal run may execute the three-candidate offline experiment. A second identical normal run should skip by fingerprint. A forced orchestration check can use `--force --skip-offline` to avoid an unnecessary duplicate heavy experiment.
