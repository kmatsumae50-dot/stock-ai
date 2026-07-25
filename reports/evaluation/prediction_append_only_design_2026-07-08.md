# Prediction Append-Only Design Report

Generated: 2026-07-08

## Purpose

Investigate the current prediction-history storage design and propose a safe append-only design for live predictions.

This report does not change the SQLite schema, production models, daily_predict, Task Scheduler, prediction history rows, feature generation, NewsLimit 3, known_missed_live_dates, source profiles, earnings CSV, offline experiment outputs, or shadow prediction storage.

## Current State

The current SQLite table is `predictions`.

Important implementation points:

- `src/stock_ai/storage/sqlite_store.py:27` defines the primary key as `(date, ticker, target, model_name)`.
- `src/stock_ai/storage/sqlite_store.py:136` uses `ON CONFLICT(date, ticker, target, model_name) DO UPDATE`.
- `src/stock_ai/storage/sqlite_store.py:146` updates `created_at = CURRENT_TIMESTAMP` on conflict.
- `src/stock_ai/cli.py:1129` saves `predict-binary` rows through `upsert_predictions`.
- `src/stock_ai/cli.py:1252` saves `predict-multiclass` rows through `upsert_predictions`.
- `src/stock_ai/cli.py:1287` saves `backfill-predictions` rows through the same `upsert_predictions`.
- `scripts/daily_predict.ps1:185` already logs that identical ticker/date/target/model rows are overwritten.
- `scripts/daily_predict.ps1:243` and `scripts/daily_predict.ps1:245` run binary and multiclass prediction saves.
- `src/stock_ai/evaluation/binary_shadow.py:134` saves shadow predictions to report files, not SQLite.

Current SQLite row counts for `7203.T`:

| Scope | Rows | Date range |
|---|---:|---|
| binary production live | 29 | 2026-05-22 to 2026-07-08 |
| binary backfill | 1 | 2026-05-01 |
| multiclass production live | 28 | 2026-05-26 to 2026-07-08 |
| multiclass experiment live variants | 5 | 2026-06-01 to 2026-06-02 |

Recent evidence of repeated saves for the same feature date:

- Feature date `2026-07-03` was saved by daily_predict on `2026-07-03`, then again on `2026-07-04`, then again on `2026-07-05`.
- The final SQLite row only keeps the latest version because the primary key does not include a run identifier.
- Similar delayed or repeated-save patterns exist for `2026-06-12`, `2026-06-19`, `2026-06-26`, and `2026-07-03`.

## Current Problems

### 1. Live rows are not a true append-only live log

If the same feature date is predicted more than once with the same ticker, target, and model, the previous row is overwritten.

Lost information:

- earlier predicted class
- earlier confidence
- earlier probabilities
- earlier top features
- original saved time
- whether the first run was scheduled or manual
- whether the rerun was a weekend/stale-data rerun or an intentional rerun
- whether the prediction changed between runs

### 2. Manual and scheduled runs are indistinguishable

The current table has `prediction_type`, but it does not distinguish:

- scheduled daily task
- manual daily_predict run
- direct CLI predict run
- rerun after data correction
- weekend rerun using the same feature date

### 3. Backfill can overwrite live rows if dates overlap

`backfill-predictions` currently writes into the same `predictions` table with the same primary key. Because `prediction_type` is not part of the primary key, a backfill row for an existing live feature date can overwrite the live row, changing `prediction_type` and probabilities.

Current data has not suffered from this for the recent live window, but the storage design allows it.

### 4. Health-check currently asks only whether a feature date has a live row

This is fine for current snapshot storage, but append-only storage should answer:

- Did any live run exist for this feature date?
- Was there an official run?
- Was the official run scheduled or manual?
- Was the run stale at execution time?

### 5. Evaluation metrics need a single canonical row per feature date

If append-only rows are added, metrics must avoid double-counting multiple live runs for the same feature date. The metric policy needs to be explicit.

Possible policies:

- official run only
- first scheduled successful run
- latest successful run
- latest run with freshness OK

The current table implicitly uses latest overwritten row. That is convenient but not auditable.

## Design Options

### Option A: Make existing `predictions` append-only

Add `run_id`, `run_timestamp`, `run_context`, and `is_latest` to the existing `predictions` table, and change the primary key.

Pros:

- Single table for all prediction records.
- No parallel storage path.

Cons:

- High migration risk because existing code assumes uniqueness by `(date, ticker, target, model_name)`.
- `read_latest_predictions`, Prediction History, health-check, and evaluation logic all need careful changes.
- Rollback is harder once the primary key changes.
- Backfill/live/shadow separation still needs careful filtering.

Assessment: not recommended as the first step.

### Option B: Add `prediction_runs` and make `predictions` one-to-many

Create a run table and link prediction records to runs. This is conceptually clean, but still requires changing the current `predictions` table shape.

Pros:

- Strong run metadata model.
- Clean support for scheduled/manual/freshness/run status.

Cons:

- Similar migration risk to Option A if the existing `predictions` table is modified directly.
- Many current readers need updates at the same time.

Assessment: good end-state, but too risky as the first migration if it modifies the existing table.

### Option C: Add append-only history tables and keep existing `predictions` as latest snapshot

Create new append-only tables for run records and prediction records. Keep the existing `predictions` table as the current/latest snapshot for backward compatibility.

Suggested tables:

```sql
prediction_runs (
  run_id TEXT PRIMARY KEY,
  run_started_at TEXT NOT NULL,
  run_finished_at TEXT,
  run_trigger TEXT NOT NULL,        -- scheduled, manual, cli, backfill, legacy_import
  run_status TEXT NOT NULL,         -- success, partial, failed
  project_root TEXT,
  command TEXT,
  ticker TEXT,
  data_freshness_status TEXT,
  expected_latest_trading_date TEXT,
  latest_raw_date_at_run TEXT,
  latest_feature_date_at_run TEXT,
  stale_at_run_time INTEGER,
  notes TEXT
)
```

```sql
prediction_records (
  record_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  feature_date TEXT NOT NULL,
  ticker TEXT NOT NULL,
  target TEXT NOT NULL,
  model_name TEXT NOT NULL,
  source TEXT NOT NULL,             -- live, backfill
  run_trigger TEXT NOT NULL,         -- scheduled, manual, cli, backfill, legacy_import
  official_for_metrics INTEGER DEFAULT 0,
  is_latest_snapshot INTEGER DEFAULT 0,
  predicted_class INTEGER NOT NULL,
  confidence REAL,
  probability_surge REAL,
  probability_up REAL,
  probability_flat REAL,
  probability_down REAL,
  probability_crash REAL,
  top_features TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(run_id) REFERENCES prediction_runs(run_id)
)
```

The existing `predictions` table remains the latest snapshot used by current code until each reader is migrated.

Pros:

- Lowest migration risk.
- Existing UI and health-check can keep working during rollout.
- Append-only audit trail can start immediately after migration.
- Rollback is easy: ignore/drop the new tables and keep using `predictions`.
- Backfill can be kept separate through `source` and `run_trigger`.
- Shadow can remain outside SQLite.

Cons:

- Temporarily stores latest rows in two places.
- Requires consistency checks between latest snapshot and append-only records.

Assessment: recommended.

### Option D: Keep SQLite unchanged and write append-only reports only

Write live predictions to CSV/JSON report files while keeping SQLite as-is.

Pros:

- Very low risk.
- No SQLite migration.

Cons:

- Harder to query for health-check and UI.
- More fragile than database-backed audit history.
- Eventually creates two sources of truth.

Assessment: useful as a temporary audit log, but not enough for the long-term design.

## Recommended Design

Recommend Option C: add append-only history tables while preserving the existing `predictions` table as a latest snapshot.

This gives the project an audit-quality live history without forcing an immediate rewrite of Prediction History, health-check, and evaluation logic.

## Recommended Rules

### Storage

- Live predictions are append-only in `prediction_records`.
- Same feature date with a different `run_id` is stored as a separate row.
- Existing `predictions` remains a latest snapshot during migration.
- `prediction_records.source` can be `live` or `backfill`.
- Shadow predictions remain outside SQLite in `reports/evaluation`.

### Run identity

Each run gets a `run_id`, ideally:

```text
{run_started_at_utc}_{ticker}_{trigger}_{short_hash}
```

Examples:

- `20260708T014437Z_7203T_scheduled_ab12cd`
- `20260708T091500Z_7203T_manual_ef34gh`

### Scheduled vs manual

Add an explicit run trigger:

- `scheduled`
- `manual`
- `cli`
- `backfill`
- `legacy_import`

Later, `daily_predict.ps1` can pass `--run-trigger scheduled` when invoked by Task Scheduler.

Manual CLI runs should default to `manual` or `cli`.

### Latest prediction

For UI latest snapshot:

- Use existing `predictions` initially.
- Later, derive latest from `prediction_records` using `is_latest_snapshot=1` or latest `created_at`.

### Official evaluation row

Metrics should not count every run. Default policy should be:

1. Use rows with `official_for_metrics=1`.
2. Initially mark scheduled successful production live rows as official.
3. If multiple scheduled successful rows exist for the same feature date, choose the earliest scheduled successful row for that feature date.
4. Manual reruns should not become official unless explicitly marked.

Reason:

- This preserves what the system would have shown in daily operation.
- It avoids artificially improving or degrading history through later reruns.
- It keeps manual experimentation out of live metrics.

### Health-check missing logic

Coverage check should consider a feature date covered if at least one production live append-only record exists for that feature date.

Recommended detail levels:

- `has_any_live_run`
- `has_official_live_run`
- `has_scheduled_live_run`
- `latest_live_run_status`
- `stale_at_run_time`

Action required should apply to unknown post-start dates with no production live run, unless registered in `known_missed_live_dates`.

### Prediction History UI

Default view:

- Production official live rows only.

Additional expandable views:

- Latest snapshot rows
- Same feature-date run history
- Manual runs
- Scheduled runs
- Backfill rows
- Experimental model rows

For each feature date, show:

- official prediction
- latest prediction
- run count
- whether prediction changed across runs
- scheduled/manual breakdown

### Backfill separation

Backfill should not share the same unique key as live rows.

Recommended:

- Store backfill in append-only `prediction_records` with `source='backfill'` and `run_trigger='backfill'`.
- Do not mark backfill as official live.
- Existing `predictions` table should not be overwritten by backfill for dates that already have live rows.

### Shadow separation

Keep binary shadow predictions out of SQLite prediction history.

Current shadow behavior is appropriate:

- saved in `reports/evaluation/binary_shadow_predictions.csv`
- saved in `reports/evaluation/binary_shadow_predictions.json`
- `sqlite_saved: false`

If shadow ever moves into SQLite, use a separate `shadow_prediction_records` table or `source='shadow'` with UI hidden by default. For now, do not mix it.

## Migration Plan

### Migration schema

Add tables:

- `prediction_runs`
- `prediction_records`

Do not modify the existing `predictions` primary key in the first migration.

### Existing data handling

Seed existing rows into append-only history as legacy records:

- `run_trigger='legacy_import'`
- `source=prediction_type`
- `official_for_metrics=0` or `NULL` initially
- `is_latest_snapshot=1`
- `created_at` copied from existing row

Do not invent missing runs.

Known missed dates remain known missed and should not receive fabricated live rows.

### Rollback

Rollback is simple in Option C:

- Stop writing new tables.
- Continue using existing `predictions`.
- Drop or ignore `prediction_runs` and `prediction_records`.

No production prediction rows need to be rewritten.

## Phased Implementation Plan

### Phase A: Design report only

Current phase. No schema or production logic changes.

### Phase B: Add schema migration only

Add `prediction_runs` and `prediction_records`.

Do not yet switch daily_predict to append-only.

Add read-only helpers and tests.

### Phase C: Dual-write append-only records

When `predict-binary` and `predict-multiclass` save to `predictions`, also write append-only records.

Keep existing `predictions` as latest snapshot.

### Phase D: Prediction History append-only UI

Add:

- official live rows
- latest snapshot rows
- run history per feature date
- changed prediction detection

Default remains clean and non-duplicated.

### Phase E: health-check and metrics append-only support

Update coverage checks and metrics to read official append-only records.

Keep fallback to current `predictions` for older databases.

### Phase F: Backfill safety

Stop backfill from overwriting live latest snapshot.

Store backfill only in append-only records or use a separate backfill snapshot path.

### Phase G: Retire old assumptions

After stable operation, decide whether `predictions` remains a compatibility snapshot or becomes a view derived from append-only records.

## Test Plan

Required tests:

- Saving the same feature date multiple times creates multiple append-only records.
- Latest snapshot still returns one row per ticker/target/model.
- Latest live prediction is selected correctly.
- Scheduled and manual runs are distinguishable.
- Same feature date with scheduled + manual runs does not double-count metrics.
- `official_for_metrics` selects one row per feature date.
- Health-check does not mark a feature date missing if any production live run exists.
- Health-check can separately report missing official runs.
- Backfill rows never satisfy live coverage.
- Backfill rows do not overwrite live rows.
- Shadow predictions remain outside SQLite.
- Migration imports existing rows without changing current `predictions`.
- Rollback can ignore new append-only tables.
- Prediction History default UI does not show duplicate run rows unless expanded.
- Same feature-date run history shows prediction changes.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Existing UI double-counts runs | Default to official production rows only |
| Metrics change unexpectedly | Keep old metrics until official selection is tested |
| Backfill overwrites live snapshot | Phase F prevents overlapping backfill upsert |
| Migration creates false live history | Import as `legacy_import`, not scheduled live |
| Manual reruns distort live accuracy | Manual runs are not official by default |
| Shadow contaminates production history | Keep shadow in reports only |

## Recommendation

Adopt Option C in phases.

Do not change the existing `predictions` primary key immediately. Add append-only tables beside it, dual-write for a while, then migrate readers gradually.

The first implementation task should be Phase B: add append-only schema and read helpers, with no daily_predict behavior change.

## Explicit Non-Changes In This Report

This report did not change:

- SQLite schema
- SQLite prediction history rows
- production model config
- binary production model
- multiclass production model
- daily_predict.ps1
- Task Scheduler time
- actual live prediction saving logic
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- known_missed_live_dates
- news source profiles
- earnings CSV
- offline experiment results
- binary shadow prediction storage behavior

