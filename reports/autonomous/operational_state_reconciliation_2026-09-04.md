# Operational State Reconciliation And Migration Hardening

Date: 2026-09-04

Scope: one-time operational reconciliation and migration documentation. No production model change, retraining, threshold change, prediction write, database migration, Task Scheduler change, remote operation, commit, merge, push, research worktree, or heavy research was performed.

## Current Health Source Of Truth

The latest successful daily run before this review completed on 2026-09-03. At that run:

- expected latest trading date: `2026-09-03`
- latest raw date: `2026-09-03`
- latest feature date: `2026-09-03`
- freshness lag: `0` trading days
- `stale_at_run_time`: `false`
- data freshness status: `OK`
- `check-news`: `OK`
- unknown source count: `0`
- health Action required: `0`
- known missed dates: `8`
- Known OK: TOPIX proxy / fallback

At 01:06 JST on 2026-09-04, the standalone `data-freshness` command expected the new trading day while the 10:30 daily run had not yet occurred. It correctly returned lag `1` and `WARN`. Running Autonomous Operations at that point would create a false reconciliation target for the new pre-run day. Operations and Research refresh are therefore deferred until the 2026-09-04 daily cycle has completed fresh.

## Dirty Main Audit

The reviewed pre-existing dirty set matched the previously reviewed maintenance work. No unexpected file was found.

### A. Source configuration

- `configs/news_sources.yaml`
- `tests/test_news_sources.py`

Purpose: reviewed conservative news-source profiles and alias coverage.

### B. Stale-lock and Research guard maintenance

- `scripts/run_autonomous_operations.ps1`
- `scripts/run_autonomous_research.ps1`
- `scripts/autonomous_lock_helpers.ps1`
- `tests/test_autonomous_research_scheduler.py`
- `tests/test_autonomous_scheduler_scripts.py`
- `tests/test_autonomous_lock_helpers.py`
- `reports/autonomous/post_unattended_audit_recovery_2026-08-19.md`

Purpose: quarantine-only stale lock recovery, live-PID and unreadable-lock protection, and safe Research skip behavior.

### C. Shadow lifecycle

- `src/stock_ai/cli.py`
- `src/stock_ai/evaluation/binary_shadow.py`
- `tests/test_binary_shadow.py`

Purpose: read-only status, dry-run support, idempotent evaluation persistence, and per-ticker preservation.

### D. Known missed settings

- `configs/settings.yml`

Purpose: document 2026-08-24 as a reasoned known missed date. No live or backfill row was created.

### E. Documentation

- `README.md`
- `PROJECT_STATE.md`
- `docs/current_user_manual.md`
- `docs/environment_handoff.md`
- `reports/autonomous/production_prediction_integrity_audit_2026-09-02.md`
- this report

### F. Runtime/generated

- `reports/evaluation/binary_shadow_summary_2026-08-19.md`

This generated summary should not be mixed into the reviewed source changes automatically.

### G. Unexpected

- none

## Runtime State Reconciliation

Before refresh, the persisted Operations state ended at 2026-09-03 11:40, before the successful 11:42 catch-up Daily run. It still reported:

- Action required `1`
- feature date `2026-09-02`
- unresolved known-missed, stale-data, and old scheduler findings

The persisted Research state ended on 2026-09-02 with an unhealthy Operations gate. The Codex Agent state retained a historical `active_worktree` path, but read-only preflight reconciled actual Git state correctly:

- registered worktrees: main only
- active lock: false
- all four completed/rejected research worktree paths absent
- lifecycle status: `cleanup_completed`
- completed/rejected fingerprints remain blocked from reselection
- selected next candidate: `recent_regime_robustness`
- main clean: false, so Agent gate remains safely closed

No state JSON was hand-edited. Final Operations and Research refresh results are recorded below after the fresh daily gate is available.

## Prediction And Model Semantics

### Dates

- `predictions.date` is effectively the Feature date.
- Target close date is the next trading day's close after the Feature date.
- `created_at` is the SQLite write time.
- `scheduled_for` and `actual_started_at` are distinct scheduler facts.

### Binary and multiclass

- Binary remains the primary production direction reference.
- Latest reviewed live diagnostics materially underperform simple baselines; production change is not ready.
- Multiclass remains flat-biased and reference/experimental only.

### Shadow

- generation mode: manual/opt-in
- confirmed rows: `2`
- pending rows: `0`
- automatic scheduled generation: none
- SQLite production history writes: none

## Append-Only Semantics

Current inventory:

- `prediction_runs`: `1`
- `prediction_records`: `2`
- existing `predictions`: `129`
- run type: `manual_live_dual_write`
- latest snapshot comparison: mismatch, because the current 2026-09-03 snapshot has no automatic scheduled dual-write records

The manual audit rows contain database `is_official=1` / `is_latest=1` flags from validation, but policy excludes `manual_live_dual_write` from official selection and live metrics. Policy-derived official eligibility is authoritative. Append-only remains informational and is not the health-check or Prediction History metric source. No flags or rows were changed.

## Old Experimental Multiclass Live Rows

Five retained rows use `prediction_type=live` but do not use the production model name:

| Feature date | Model name |
| --- | --- |
| 2026-06-01 | `lightgbm_multiclass_practical_3_0_5` |
| 2026-06-01 | `lightgbm_multiclass_quantile` |
| 2026-06-02 | `lightgbm_multiclass_practical_3_0_5_balanced` |
| 2026-06-02 | `lightgbm_multiclass_practical_3_0_5_tailboost` |
| 2026-06-02 | `lightgbm_multiclass_quantile_balanced` |

They originated from early multiclass profile/weight experiments. Production History scope helpers and official snapshot selection distinguish the production model slot. However, `build_multiclass_diagnostic_frame` currently filters target and prediction type, not `model_name`; the Evaluation Prediction History diagnostics and Autonomous Operations live-coverage summary pass the unfiltered history and can include these five rows.

Level 1 maintenance candidate: add explicit production model-slot filtering at those aggregation call sites, with tests. Do not delete or rewrite historical rows.

## 2026-09-03 Daily Timing

- scheduled for: `10:30`
- actual Task Scheduler start: `11:42:03`
- result: `0`
- successful daily completion: `11:51:20`
- `StartWhenAvailable`: true

Windows System events show boot at 10:43, sleep at 10:45, and wake at 11:39. The timing is consistent with a `StartWhenAvailable` catch-up after resume. Task Scheduler Operational history did not provide enough matching events to prove the exact trigger path, so manual trigger cannot be ruled out absolutely. It must be reported as an 11:42 actual start, not a 10:30 prediction.

## Earnings Data Gap

Both candidate files are present but header-only:

- `data/events/earnings_7203.T.csv`: `0` data rows
- `data/events/toyota_earnings.csv`: `0` data rows
- columns: `date,event_type,period,note`

The feature pipeline reads these candidates. With no valid events it sets `days_to_earnings` and `days_after_earnings` to missing, keeps event flags/state at their neutral values, and keeps derived event metrics at zero. Health currently checks file presence and feature-column presence, so it does not identify a header-only population gap.

Future phase candidate: `earnings_data_pipeline`, with sourced event dates and explicit coverage health. It was not implemented in this cycle.

## Migration And Backup Readiness

`docs/environment_handoff.md` now documents:

- Git-restorable and non-Git state
- Critical, High, and regenerable backup classes
- Python 3.14 environment reconstruction
- all three Task Scheduler registrations and XML export
- path migration and hard-coded path search
- read-only restoration verification
- a SHA-256 backup manifest procedure for database, models, configuration, and important reports
- why a private Git remote is useful but insufficient as a full backup

No backup was created and no remote was added. Migration readiness is therefore `documented_but_not_backed_up`.

## Proposed Commit Groups

No files were staged or committed.

1. `Add reviewed news source profiles`
   - files: `configs/news_sources.yaml`, `tests/test_news_sources.py`
   - dependency: none
2. `Harden autonomous lock recovery and research guard`
   - files: `scripts/run_autonomous_operations.ps1`, `scripts/run_autonomous_research.ps1`, `scripts/autonomous_lock_helpers.ps1`, `tests/test_autonomous_research_scheduler.py`, `tests/test_autonomous_scheduler_scripts.py`, `tests/test_autonomous_lock_helpers.py`, `reports/autonomous/post_unattended_audit_recovery_2026-08-19.md`
   - dependency: group 1 not required
3. `Add binary shadow lifecycle diagnostics`
   - files: `src/stock_ai/cli.py`, `src/stock_ai/evaluation/binary_shadow.py`, `tests/test_binary_shadow.py`
   - dependency: none
4. `Record reviewed missing date and prediction integrity audit`
   - files: `configs/settings.yml`, `reports/autonomous/production_prediction_integrity_audit_2026-09-02.md`
   - dependency: keep known-missed reason and audit together
5. `Reconcile operations and document recovery`
   - files: `README.md`, `PROJECT_STATE.md`, `docs/current_user_manual.md`, `docs/environment_handoff.md`, `reports/autonomous/operational_state_reconciliation_2026-09-04.md`
   - dependency: group 4 supplies the eighth known-missed setting documented here

Exclude the generated `reports/evaluation/binary_shadow_summary_2026-08-19.md` unless a human explicitly decides it is a durable artifact.

## Production Safety Baseline

Captured before documentation work:

- Git HEAD: `b351f7af9a73ac9445403c1e6b04a77cbcc8b81c`
- production DB SHA-256: `515B6E90DC3CEF5B309755081CAE0837587724C19604C402A046204547F8C069`
- predictions / runs / records: `129 / 1 / 2`
- binary artifact SHA-256: `3326303B15A869B3F0AACB562590264347AB2154FFEDDABFFD6C1E200830067C`
- multiclass artifact SHA-256: `C70A1A017FE09DB6A919181418989977AD33DBA9D7ACA35885A250BD02E8569D`
- settings SHA-256: `6EDFD8BD9ED7A66949CE255ACFB7FCD1736D7FBAE7F37DE961D53835FC6FEF0E`
- news source config SHA-256: `F4CB112AD0DA25535A8B9D40085AEF9D6F8C9EC00F3D674BF8425F7767A90772`
- `daily_predict.ps1` SHA-256: `83F25744BC1409033EDF74E37C506BAF84BD6E2BD0315CF463407B74B659CBA2`
- Daily / Operations / Research task XML SHA-256: `9DD52F60042481DCDEE84A489C74C73B23406D739946A22CB80CCBD02D49D59B` / `2F7F2F64C06FB00485A6B52F40FDBC6576BE698F7503B4CA11550F46F27906D8` / `469B592505E92C1BFAA73DA8B40F396D3D5C3913D36EE3031A5721E971F36536`

Post-verification hashes and counts must match these values except for the intentionally edited documentation and reviewed configuration working-tree changes.

## Final Gate And Verification

Pending the 2026-09-04 10:30 daily cycle:

- Autonomous Operations: deferred to avoid a pre-run false stale finding
- Autonomous Research: not run; heavy research is prohibited in this reconciliation
- Codex Agent: read-only preflight safely closed because main is dirty and the Research gate is stale
- new worktree: not created
- `recent_regime_robustness`: not started

Final CLI, test, Streamlit, and readiness results will be appended after the fresh daily gate is available.
