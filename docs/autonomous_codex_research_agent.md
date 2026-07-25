# Autonomous Codex Research Agent v1

## Purpose

The Agent extends the existing read-only operational chain without changing production:

1. Daily Prediction creates the operational prediction.
2. Autonomous Operations diagnoses health and data quality.
3. Autonomous Research evaluates the fixed production/current candidates.
4. Autonomous Codex Research Agent selects at most one new bounded hypothesis and, only after all gates pass, asks Codex to implement and evaluate it in an isolated Git worktree.

The Agent never promotes a model, merges a branch, commits, pushes, writes a production prediction, changes Task Scheduler, or changes production artifacts/config/SQLite.

## Current Deployment Status

The Agent is implemented for manual use only. It is not registered in Task Scheduler.

At implementation time, the formal project folder was an unborn Git repository: `HEAD` did not exist and all project files were untracked. A Git worktree cannot be created safely from that state. The Desktop-bundled `codex.exe` was discoverable, but PowerShell returned Access denied for `codex --help`, so its non-interactive flags could not be verified. The Agent therefore stops before worktree creation and records the exact blocker. It does not invent CLI flags or fall back to editing main.

An actual research cycle becomes eligible only after a human establishes a reviewed initial Git commit, leaves main clean, and provides a Codex CLI environment whose `--help` and `exec --help` are executable. The Agent itself does not create that commit.

## Commands

Inspect the CLI:

```powershell
.\.venv\Scripts\python.exe main.py autonomous-codex-agent --help
```

Dry-run the current gates without creating a worktree or Codex task:

```powershell
.\.venv\Scripts\python.exe main.py autonomous-codex-agent --ticker 7203.T --dry-run
```

Run one manual cycle after all prerequisites are satisfied:

```powershell
.\scripts\run_autonomous_codex_research_agent.ps1 -Ticker 7203.T -MaxRuntimeMinutes 180
```

The wrapper fixes the formal root and `.venv`, holds a wrapper lock, captures stdout/stderr under `logs/autonomous`, passes through exit codes, and enforces a finite timeout. It contains no Task Scheduler registration.

## Gate

Before selecting or creating a worktree, the Agent checks:

- latest Operations cycle completed read-only
- Operations report is current and fresh
- Action required, unknown source, and unresolved post-start missing-live counts are zero
- raw/features are fresh
- production SQLite is readable and passes `PRAGMA quick_check`
- latest Autonomous Research result and fingerprint are available and healthy
- production settings hash matches the Research production guard
- the formal folder is a Git repository with a valid `HEAD`
- main worktree is clean, including untracked files
- no Agent cycle lock exists

If a gate fails, no worktree is created. The Agent writes only its report and, on a normal non-dry run, deduplicated state/backlog/task-queue diagnostics.

## Hypothesis Backlog

- Backlog: `reports/autonomous/codex_research_hypotheses.json`
- State: `reports/autonomous/codex_research_agent_state.json`
- Reports: `reports/autonomous/codex_agent_YYYY-MM-DD_HHMM.md` and `.json`
- Patch outputs: `reports/autonomous/patches/`
- Shared review queue: `reports/autonomous/operations_task_queue.json`

Hypotheses use deterministic fingerprints. Rewording an existing item without changing its actual category, rationale, scope, and primary metric does not create a duplicate. One cycle selects at most one low/medium-risk hypothesis. Completed, rejected, blocked, monitoring, and approval-required outcomes remain in the backlog to prevent repeated failed research.

The initial bounded backlog covers:

- train-only threshold guardrails for `class_weight_balanced`
- stability of `scale_pos_weight_medium`
- a simple disagreement/hold rule
- confidence calibration diagnostics
- low up-prediction decomposition
- existing-regime robustness
- bounded existing-feature-group ablation

## Permitted And Forbidden Research

Permitted categories are binary weighting/thresholds, existing-data feature research, simple ensemble/decision rules, and evaluation-quality improvements. Candidate scope is deliberately small.

Forbidden work includes production changes, new target definitions, open-price prediction, new external data contracts, new tickers, deep learning, reinforcement learning, LLM trading decisions, automatic trading, broad optimization, live/backfill creation, commit, merge, push, and Task Scheduler changes.

## Worktree Isolation

Eligible worktrees use:

- branch: `autonomous-research/YYYY-MM-DD/<hypothesis-id>`
- folder: `C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\<hypothesis-id>`

The Agent refuses an existing path or branch. It uses `git worktree add` from the reviewed main `HEAD`. Production SQLite is not copied or written. Source changes remain uncommitted in the worktree. The Agent saves a reviewable binary-capable patch after the cycle.

Review a worktree:

```powershell
git -C C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\<hypothesis-id> status --short
git -C C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\<hypothesis-id> diff
```

Apply a reviewed patch manually only in a separate approved task. The Agent never applies it to main.

Remove a discarded worktree manually:

```powershell
git -C C:\Users\kmats\Documents\ai-stock\stock-ai worktree remove C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\<hypothesis-id>
git -C C:\Users\kmats\Documents\ai-stock\stock-ai branch -D autonomous-research/YYYY-MM-DD/<hypothesis-id>
```

## Codex Invocation Contract

The Agent probes `codex --help` and `codex exec --help` read-only. It constructs an invocation only when the executable works and the help text confirms all required safe flags for working directory, workspace-write sandboxing, approval policy, and final-output capture. Otherwise it records `codex_cli_not_found`, `codex_help_failed`, `codex_exec_help_failed`, or `required_safe_flags_unavailable` and exits safely.

No unverified invocation is documented here because the current installed executable could not be run from PowerShell. When available, the exact verified command array is recorded in the cycle JSON report.

## Evaluation And Decision

The worktree agent must evaluate historical all, since 2024/2025/2026, recent six/three months, and a live-like holdout. It compares production and always-up, always-down, majority, previous-day-direction, and moving-average baselines. Live/shadow evidence is included when available.

Human review is allowed only when balanced accuracy and the primary metric improve, down recall remains within its guardrail, simple baselines are beaten, multiple periods agree, gains are not recent-only, leakage checks pass, and sample size is sufficient. Missing periods, missing baselines, failed tests, leakage, or any production safety flag invalidates the result.

## Exit Codes

- `0`: completed, rejected, no safe hypothesis, or dry-run ready
- `2`: human review required
- `3`: Operations or Research gate unhealthy/stale
- `4`: main Git repository is unborn or dirty
- `5`: selected hypothesis blocked, including unavailable verified Codex CLI
- `124`: runtime timeout
- `125`: lock/dependency safety stop
- other nonzero: technical failure

## Review Checklist

For a completed worktree cycle, review:

1. gate and selected-hypothesis evidence
2. worktree status and patch
3. focused and existing test results
4. every required evaluation period
5. production and baseline deltas
6. leakage and sample-size checks
7. safety booleans and production hashes/counts
8. classification and whether the human-review gate was actually met
9. absence of commits, merge, push, live/backfill rows, and Scheduler changes

The Agent remains manual until several clean, bounded cycles demonstrate safe behavior and a separate human-approved deployment task is completed.
