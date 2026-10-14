# Post-Unattended Audit Recovery and Level 1 Maintenance

Date: 2026-08-19

## Outcome

- News quality recovered: `unknown_source_count=0`, `Action required=0`.
- The completed `model_disagreement_selective_up_signal` worktree was removed after a clean dry-run audit. Its report and patch remain preserved.
- Operations and Research wrappers now recover only verified old orphan locks. Active, recent, and unreadable locks still stop safely.
- The Research Operations gate still uses exit code `3` for a safe unhealthy skip and now labels that exit separately from technical stops in wrapper logs.
- The two existing binary shadow rows were evaluated in place. Both are confirmed hits; no new historical shadow or live predictions were generated.
- Production models, production SQLite prediction history, `daily_predict.ps1`, and Task Scheduler configuration were unchanged.

## TechStock2 Source Profile

The CSV contained one row with source `TechStock²`:

- Date: 2026-08-18
- Query profile: `toyota_stock`
- Category: `recall_safety`
- Title: `Toyota Stock: 508,000 Camry Recall Challenges Hybrid Expansion - TechStock²`
- Delivery: Google News RSS
- Duplicate title or URL: none
- Original/reprint status: the feed attributes the item to TechStock2, but the Google News URL did not expose enough canonical metadata to prove whether it was original or syndicated; no matching duplicate exists in the local CSV.

The article is directly about a Toyota Camry recall, but the publisher describes itself as an independent financial news/data platform that uses automated systems with editorial review. It is not treated as Toyota, Reuters, or another primary/high-impact source.

Configured profile:

- Name: `TechStock2`
- Aliases: `TechStock²`, `techstock2`, `ts2.tech`
- Source type: `general_news`
- Credibility: `0.48`
- Market impact: `0.34`

Publisher reference: <https://ts2.tech/en/about/>

## Worktree Cleanup

Dry-run and confirmed cleanup both reported:

- Hypothesis status: `completed`
- Lifecycle: `cleanup_ready`
- Commit count ahead: `0`
- Matching process: none
- Saved report: present
- Saved patch: present
- Recorded and actual patch SHA-256: `a955b9753bf238e9372862024ab0b86f3fc8abd93a9f1fdea16d57bff545cdbd`
- Patch matches worktree: true
- Safe to remove: true

The confirmed cleanup removed only the registered worktree. The patch and report remain in `reports/autonomous`.

## Stale Lock Root Cause and Recovery

Normal wrapper exits already remove locks in PowerShell `finally` blocks. A PC shutdown or process termination can bypass `finally`, leaving the PID-bearing lock behind. The previous Research wrapper returned `125` until a human used `-Force`; the previous force path checked process existence but did not require a minimum lock age. `StartWhenAvailable` can therefore catch up a scheduled task while the abandoned lock is still present and stop on the lock.

The shared `autonomous_lock_helpers.ps1` now classifies locks as:

- `lock_active`: PID exists; never remove.
- `lock_unreadable`: PID or `created_at` cannot be trusted; never remove.
- `lock_orphaned_recent`: PID is absent but the stale threshold is not met; never remove.
- `lock_stale_candidate`: PID is absent and the age threshold is met; quarantine and log before continuing.

Defaults are 150 minutes for Research, above its 120-minute maximum runtime, and 30 minutes for Operations, above its 10-minute maximum runtime. `-Force` does not bypass these checks. Recovered locks are moved under ignored `logs/autonomous/stale_locks` with PID, age, and destination logged.

## Research Guard

The health criteria remain unchanged: report freshness, scheduler result, Action required, unknown sources, unresolved missing live dates, data freshness, and database readability are still checked. Operations-unhealthy skips retain exit code `3`. Wrapper logs now add `dependency_exit_classification=safe_operations_unhealthy_skip`; technical or concurrency stops are labeled separately.

After the source change, the first Research rerun safely stopped on `unexpected_production_guard_change`. The second rerun returned `research_skipped_not_due`, with `experiment_executed=false` and no production change.

## Shadow Lifecycle

Root cause:

- `binary-shadow-predict` was manual-only.
- `binary-shadow-evaluate` could persist evaluation, but it was not called by Daily, Operations, or Research scheduling.
- Research could join actual feature data in memory, so its report saw confirmed outcomes while the CSV remained pending.

Maintenance changes:

- `binary-shadow-status` is read-only and reports totals, pending/confirmed, candidate set, target availability, stale pending rows, duplicates, and next-generation eligibility.
- `binary-shadow-evaluate --dry-run` evaluates without writing.
- Normal evaluation persists only shadow CSV/JSON/summary artifacts, preserves other tickers, and keeps `evaluated_at` stable on idempotent reruns.
- `binary-shadow-predict --dry-run` builds candidates without saving.
- Candidate set remains `production_current` and `class_weight_balanced`.

Existing row result:

- Total: 2
- Confirmed: 2
- Pending: 0
- Hits: 2
- Misses: 0
- Feature date: 2026-07-08
- Actual direction: Down

No shadow prediction was back-created. No SQLite row was written.

## Future Shadow Generation Contract

No schedule was registered in this maintenance. A later approved task can add a separate post-Daily opt-in wrapper that:

1. Requires successful Daily and current feature data.
2. Runs `binary-shadow-status`.
3. Runs `binary-shadow-predict` only when the latest feature date lacks the fixed candidate set.
4. Runs `binary-shadow-evaluate` to settle older pending rows.
5. Relies on `(ticker, feature_date, candidate_name)` duplicate prevention.
6. Never writes production SQLite and never creates historical live rows.

This should be a separate Task Scheduler decision; `daily_predict.ps1` remains unchanged.

## Safety Verification

Unchanged SHA-256 values:

- Production DB: `4BC338CB2DEF2998CEEEED6ACFF5AD8A1ACA104D2BCE0627B6BBA234400A2D34`
- Binary model: `A4C265DDA88E68D749D16FE14ACD9DF73B9322D64906D92FB2468AE55CE4896C`
- Multiclass model: `1F0C9F74557B329CC024AEB4152BCFF52DD31BCD583426E7F84C15E62AD5ACAE`
- Settings: `A3EE5F158C88DEA3F76E71C1973865CE6B32AA64E579C522702C373EF82ACCBE`
- Daily script: `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2`

Unchanged database counts:

- `predictions`: 109
- `prediction_runs`: 1
- `prediction_records`: 2

Unchanged Scheduler XML hashes:

- Daily: `9DD52F60042481DCDEE84A489C74C73B23406D739946A22CB80CCBD02D49D59B`
- Operations: `2F7F2F64C06FB00485A6B52F40FDBC6576BE698F7503B4CA11550F46F27906D8`
- Research: `469B592505E92C1BFAA73DA8B40F396D3D5C3913D36EE3031A5721E971F36536`

Validation completed:

- Ruff: passed
- Pytest: 433 passed
- Compileall: passed
- Health: Overall WARN, Action required 0, Known missed dates 7, Known OK 1
- Operations: `completed_read_only_cycle`
- Research: `research_skipped_not_due`; no experiment executed
- Locks: none
- Research worktrees: none

The Agent preflight correctly remains closed while this reviewed maintenance diff is uncommitted: `due=false`, Codex invocation false, worktree creation false. No new heavy research cycle was started.
