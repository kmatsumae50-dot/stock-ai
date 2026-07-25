# Prediction Append-Only Phase B Report

Generated: 2026-07-09

## Purpose

Add the schema and read helpers needed for future append-only prediction history without changing production prediction saving behavior.

This phase intentionally keeps the existing `predictions` table as the current latest-snapshot source. Daily live predictions, backfill predictions, Prediction History UI, health-check missing-live logic, Task Scheduler, and daily_predict are not switched to append-only storage in this phase.

## Added Schema

### `prediction_runs`

Represents one prediction execution unit.

Columns:

- `run_id`
- `run_timestamp`
- `ticker`
- `run_type`
- `command_name`
- `source`
- `working_directory`
- `data_freshness_status`
- `latest_raw_date`
- `latest_feature_date`
- `expected_latest_trading_date`
- `stale_at_run_time`
- `created_at`
- `notes`

Indexes:

- `idx_prediction_runs_ticker_timestamp`
- `idx_prediction_runs_type`

### `prediction_records`

Represents individual prediction records within a run.

Columns:

- `record_id`
- `run_id`
- `ticker`
- `feature_date`
- `target_close_date`
- `target`
- `model_name`
- `model_type`
- `predicted_label`
- `predicted_direction`
- `probability_up`
- `probability_down`
- `confidence`
- `source`
- `is_official`
- `is_latest`
- `actual_return`
- `actual_label`
- `actual_direction`
- `result`
- `created_at`
- `evaluated_at`
- `notes`

Indexes:

- `idx_prediction_records_run_id`
- `idx_prediction_records_feature`
- `idx_prediction_records_source`
- `idx_prediction_records_official`
- `idx_prediction_records_latest`

## Added Helpers

Added to `src/stock_ai/storage/sqlite_store.py`:

- `list_prediction_runs`
- `get_prediction_run`
- `list_prediction_records`
- `get_latest_official_prediction_records`
- `get_records_for_feature_date`
- `get_run_history_for_feature_date`
- `get_latest_live_record_by_feature_date`
- `summarize_append_only_records`

The helpers are read-oriented. Tests insert synthetic rows directly to validate read behavior, but production prediction save paths do not write to the append-only tables yet.

## Added CLI

Added:

```powershell
python main.py prediction-append-only-check --ticker 7203.T
```

This command:

- initializes the append-only tables if needed
- checks whether the append-only tables exist
- reports append-only row counts
- reports current `predictions` row count
- treats empty append-only tables as OK in Phase B
- does not change health-check behavior
- does not change Prediction History behavior

Latest CLI check:

```text
append-only tables exist: true
prediction_runs rows: 0
prediction_records rows: 0
existing predictions rows: 65
official records rows: 0
latest records rows: 0
source counts: {}
run type counts: {}
```

## Existing `predictions` Relationship

The existing `predictions` table is still the production latest snapshot table.

The existing primary key and upsert behavior are unchanged:

```text
PRIMARY KEY (date, ticker, target, model_name)
ON CONFLICT(date, ticker, target, model_name) DO UPDATE
```

Current Prediction History and health-check still read the existing snapshot/history behavior. The new append-only helpers are available for Phase C and later, but are not yet used as the production source of truth.

## Migration Safety

The new tables are created with `CREATE TABLE IF NOT EXISTS` and indexes use `CREATE INDEX IF NOT EXISTS`.

The initialization is idempotent:

- running schema initialization multiple times is safe
- existing `predictions` rows are not updated or deleted
- no current live prediction row is copied into append-only storage in Phase B
- rollback can ignore or drop the new append-only tables without affecting the existing snapshot table

## Current Row Counts

For ticker `7203.T` after Phase B initialization:

| Table | Rows |
|---|---:|
| `prediction_runs` | 0 |
| `prediction_records` | 0 |
| existing `predictions` | 65 |

Append-only tables being empty is expected in Phase B.

## Tests Added

Added tests cover:

- append-only tables can be created
- schema initialization is idempotent
- empty `prediction_runs` and `prediction_records` can be read
- synthetic run and record rows can be listed and fetched
- feature-date run history can be retrieved
- latest official record selection works
- latest live record by feature date works
- live/backfill source separation works
- existing `predictions` upsert behavior remains unchanged
- append-only schema does not populate records through current snapshot upsert

## Non-Changes

This phase did not change:

- production model config
- binary production model
- multiclass production model
- `daily_predict.ps1`
- Task Scheduler time
- actual live prediction save behavior
- `upsert_predictions` behavior
- existing SQLite prediction history rows
- Prediction History production display logic
- health-check missing-live logic
- evaluation metric row selection logic
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- `known_missed_live_dates`
- news source profiles
- earnings CSV
- offline experiment outputs
- shadow prediction storage behavior

## Phase C Recommendations

Next phase should add dual-write behavior behind a conservative switch or explicit function.

Recommended Phase C work:

- create an append-only write helper
- generate stable `run_id` values per prediction execution
- write scheduled/manual/backfill run metadata into `prediction_runs`
- write individual prediction outputs into `prediction_records`
- keep writing the existing `predictions` latest snapshot during the transition
- do not let backfill satisfy live coverage
- keep shadow predictions out of SQLite unless a separate explicit design is approved
- add comparison checks between `predictions` latest snapshot and append-only latest official records
- keep Prediction History default view non-duplicated

## Caveat

The old overwrite risk is not removed yet. Phase B only adds the storage foundation. Until Phase C switches or dual-writes production saves, repeated same-feature-date predictions can still overwrite rows in the existing `predictions` snapshot table.
