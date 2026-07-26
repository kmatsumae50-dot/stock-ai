# Autonomous Codex Research Agent v1.1 Cost and Lifecycle Guardrails

## Scope

This change adds orchestration guardrails only. It does not run a new research
hypothesis, re-evaluate `balanced_threshold_guardrail`, register a scheduler,
change production models/configuration, write production predictions, or delete a
research worktree.

## First-cycle audit

- Baseline commit: `fb144b9` (`Initial reviewed baseline`).
- Main was clean before the v1.1 implementation began.
- Git remote/push: no remote and no push.
- Hypothesis: `balanced_threshold_guardrail`.
- Final hypothesis status: `rejected`.
- Fingerprint: `b340541ce751c70387e5f9e3533a571f1b7ea544859e679eab924a2799c30493`.
- Rejection reason, result metrics, report path, patch path, and worktree path are retained.
- Report JSON SHA-256: `a52b112547e8aaf77ca1a9924853d6898e45a0b7d851f5d8ec0b12e0449924c8`.
- Report Markdown SHA-256: `cdcf96f1f544afcdebe3773b0e4c7a90e775ecdf170729ac436de256b6af8206`.
- Patch SHA-256: `ebeaa89fdb6a57923958eebfde1761d92aec6f7b9c07a7b7bda3037f49bbd2a7`.
- Worktree: `C:\Users\kmats\Documents\ai-stock\stock-ai-worktrees\balanced_threshold_guardrail`.
- Worktree branch has zero commits ahead of the baseline and was not merged.
- Current worktree diff matches the preserved patch after newline normalization.
- No process using the worktree was found during the audit.
- The rejected fingerprint is terminal and cannot return to `proposed` or `queued` selection.
- Equivalent hypothesis content is deduplicated by stable fingerprint.

The backlog was minimally supplemented with artifact paths and hashes. Research
metrics and the rejection decision were not changed.

## Due gate

A heavy Codex cycle is eligible only when at least one of these triggers is present:

1. Latest Autonomous Research fingerprint changed since the last heavy Agent cycle.
2. Confirmed live count increased by at least 5.
3. Confirmed shadow count increased by at least 5.
4. A new approval-required `model_research` task exists.
5. Seven days elapsed and a safe untested hypothesis exists.
6. A human explicitly supplied `--force`.

The trigger is still blocked before Codex probing or worktree creation when Operations
or Research is unhealthy/stale, main is dirty, the Agent lock is active, a duplicate
hypothesis/research state is detected, no safe hypothesis exists, or an existing
worktree needs review. Normal skips exit with code 0:

- `agent_skipped_not_due`
- `agent_skipped_no_safe_hypothesis`
- `agent_skipped_duplicate_hypothesis`
- `agent_skipped_duplicate_research_state`
- `agent_skipped_unhealthy_gate`
- `agent_skipped_worktree_review_required`

The preflight command is read-only:

```powershell
.\.venv\Scripts\python.exe main.py autonomous-codex-research-status --ticker 7203.T
```

It does not probe/invoke Codex, create a worktree, edit code or the database, or mark a
hypothesis as running.

## Cycle budget

- Maximum hypotheses per cycle: 1
- Maximum Codex tasks per cycle: 1
- Maximum implementation attempts: 1
- Maximum retuning rounds after a result: 0
- Maximum experiment variants: 6
- Maximum changed files: 8
- Maximum patch lines: 1500
- Default runtime: 90 minutes
- Hard runtime maximum: 180 minutes

Same-cycle result-driven retuning is prohibited. A performance rejection ends that
hypothesis. Only a separately reviewed technical failure can justify a later retry.

## Compact context

The heavy path now generates compact JSON and Markdown under:

`reports/autonomous/context/codex_agent_context_YYYY-MM-DD_HHMM.*`

It contains project/ticker/target, compact Operations and Research summaries,
production/baseline/shadow summaries, terminal fingerprints, open and selected
hypotheses, success/guardrail criteria, budgets, allowed/forbidden scope, and relevant
paths. It excludes full historical reports/logs/task queues/diffs, production DB
contents, and raw/news datasets. No context bundle was emitted by the read-only
preflight in this maintenance task.

## Hypothesis lifecycle

Supported states are `proposed`, `queued`, `running`, `completed`, `rejected`,
`blocked`, `monitoring`, `approval_required`, `cleanup_ready`, and `archived`.
Completed/rejected fingerprints are terminal. A genuinely distinct follow-up must
have changed substance and should record `parent_hypothesis_id`; cosmetic renaming or
small parameter retuning does not create a new hypothesis. Performance rejection is
not automatically retryable.

## Worktree lifecycle and cleanup dry-run

Worktree visibility states are `active`, `review_required`, `cleanup_ready`,
`orphaned`, and `unsafe_to_remove`. Cleanup readiness requires a terminal hypothesis,
existing report and patch, a recorded valid patch hash, patch/worktree equivalence,
zero worktree commits, no running process, and a clean main worktree.

```powershell
.\.venv\Scripts\python.exe main.py autonomous-codex-research-cleanup `
  --hypothesis-id balanced_threshold_guardrail
```

The command defaults to dry-run. Actual removal also requires `--confirm` and a fresh
successful safety audit. No worktree was deleted in this task. Before implementation,
the audited worktree met cleanup conditions. During final verification it reports
`unsafe_to_remove` solely because the reviewed v1.1 changes make main dirty until the
human accepts a new baseline.

## State compatibility

State schema is now `autonomous_codex_research_agent/v1.1`. Old state remains readable
and receives defaults for preflight/heavy-cycle timestamps, token/runtime observation,
Research/live/shadow baselines, terminal fingerprints, worktree status, next due
information, compact context path, and skip classification.

The first cycle's `931,574` tokens are recorded as legacy architecture/development
cost, not as an expected normal-cycle budget. The next approved heavy cycle should
record its own runtime and token usage so ordinary cost can be measured independently.

## Scheduler recommendation

No Task Scheduler entry was created or changed. A future scheduler should run only the
light preflight periodically and dispatch the heavy Agent only when `due=true`.
Initially, permit at most one heavy cycle per week. Run and review a second heavy cycle
manually before automating that dispatch. Do not add an empty scheduler workaround if
the available Codex CLI/Goal environment cannot enforce these gates safely.

## Current production decision

- Production model/config/artifacts: unchanged.
- Production SQLite and prediction history: unchanged.
- `daily_predict.ps1` and its scheduler: unchanged.
- Autonomous Operations/Research behavior: unchanged.
- News, known-missed dates, earnings data, live/backfill/shadow behavior: unchanged.
- Commit/merge/push: none.

## Next manual heavy-cycle conditions

1. Human reviews and commits the v1.1 guardrails so main is clean.
2. Existing rejected worktree is reviewed; cleanup remains an explicit human action.
3. Read-only preflight reports healthy gates, no active lock, no unresolved worktree,
   a safe nonterminal fingerprint, and at least one real due trigger.
4. The selected hypothesis is not equivalent to a completed/rejected hypothesis.
5. One-hypothesis/one-attempt/no-retuning budgets are accepted.
6. Normal-cycle runtime and token usage are captured separately from the legacy cost.
