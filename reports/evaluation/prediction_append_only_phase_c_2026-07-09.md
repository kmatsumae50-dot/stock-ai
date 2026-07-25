# Prediction Append-Only Phase C Report

Generated: 2026-07-09

## Purpose

Add append-only write helpers and a manual verification CLI without switching production prediction saves to append-only storage.

The existing `predictions` table remains the production latest-snapshot source. `predict-binary`, `predict-multiclass`, `backfill-predictions`, `daily_predict.ps1`, Prediction History UI, health-check missing-live logic, and model evaluation metrics are not switched in this phase.

## Added Write Helpers

Added to `src/stock_ai/storage/sqlite_store.py`:

- `create_prediction_run`
- `get_or_create_prediction_run`
- `append_prediction_record`
- `append_prediction_records`
- `mark_latest_official_records`
- `update_prediction_record_evaluation`
- `evaluate_prediction_records`
- `write_append_only_prediction_run`

These helpers write only to:

- `prediction_runs`
- `prediction_records`

They do not update existing rows in `predictions`.

## Duplicate Policy

Append-only records are scoped by run.

Rules:

- Different `run_id` values can store separate records for the same `feature_date`, `target`, and `model_name`.
- Within the same `run_id`, duplicate records are ignored by the unique index on:
  - `run_id`
  - `ticker`
  - `feature_date`
  - `target`
  - `model_name`
  - `source`
- This keeps append-only behavior while preventing accidental double-inserts from repeating the same manual command with the same `run_id`.

## Evaluation Update Policy

Prediction records start as `pending`.

Evaluation helpers may update:

- `actual_return`
- `actual_label`
- `actual_direction`
- `result`
- `evaluated_at`

This is the only intentional update path added in Phase C. It is meant for later hit/miss evaluation after actual target-date data exists.

No production metrics reader has been switched to these records yet.

## Added CLI

Added:

```powershell
python main.py prediction-append-only-write-check --ticker 7203.T --dry-run
```

Behavior:

- default is dry-run unless `--write` is explicitly supplied
- reads the current latest snapshot from `predictions`
- shows append-only record candidates
- shows append-only row count before/after
- does not write to `predictions`
- does not write to shadow CSV/JSON

Manual write mode:

```powershell
python main.py prediction-append-only-write-check --ticker 7203.T --write
```

This writes only to append-only tables. It is a manual verification command, not a production save switch.

## Dry-Run Result

Production DB dry-run:

```text
latest snapshot candidates: 7
append-only runs before: 0
append-only records before: 0
existing predictions rows: 65
status: OK - dry-run only; append-only tables were not written.
```

The 7 candidates include current production binary/multiclass latest rows plus older multiclass experiment model latest rows that still exist in the `predictions` snapshot table. Phase D should consider a production-only filter for UI and official run selection.

## Safe Write Verification

`--write` was tested against a copied database:

```text
C:\Users\kmats\AppData\Local\Temp\stock_ai_append_only_phase_c_check.db
```

First write:

```text
append-only runs before: 0
append-only records before: 0
existing predictions rows: 65
record candidates: 7
inserted records: 7
duplicate records ignored: 0
append-only runs after: 1
append-only records after: 7
existing predictions rows after: 65
```

Same `run_id` repeated:

```text
append-only runs before: 1
append-only records before: 7
record candidates: 7
inserted records: 0
duplicate records ignored: 7
append-only runs after: 1
append-only records after: 7
existing predictions rows after: 65
```

Production DB was not populated by the write test.

Production DB after verification:

```text
prediction_runs rows: 0
prediction_records rows: 0
existing predictions rows: 65
```

## Tests Added

Added tests cover:

- creating a prediction run
- idempotent get-or-create behavior for run IDs
- appending one record
- appending multiple records
- preventing duplicate records within the same run
- keeping same-feature-date records across different runs
- separating live / backfill / shadow / experiment records
- safe latest/official marking scope
- updating pending records to hit/miss after actuals exist
- writing append-only records from existing latest snapshot rows
- preserving existing `predictions` rows and `upsert_predictions` behavior

## Non-Changes

This phase did not change:

- production model config
- binary production model
- multiclass production model
- `daily_predict.ps1`
- Task Scheduler time
- `predict-binary` production save behavior
- `predict-multiclass` production save behavior
- `backfill-predictions` save behavior
- `upsert_predictions` behavior
- existing SQLite `predictions` rows
- Prediction History production display logic
- health-check live missing-date logic
- evaluation metric row selection logic
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- `known_missed_live_dates`
- news source profiles
- earnings CSV
- offline experiment outputs
- shadow prediction storage behavior

## Phase D Recommendations

Next phase should wire append-only writes carefully, without changing default UI semantics.

Recommended Phase D work:

- add an explicit production-only latest snapshot filter
- add a controlled dual-write path for `predict-binary` and `predict-multiclass`
- generate run metadata from daily_predict logs or CLI context
- keep writing the current `predictions` latest snapshot during transition
- show append-only run history only in an expanded detail view
- keep health-check live coverage based on current `predictions` until append-only coverage is proven
- add a comparison report between `predictions` latest snapshot and append-only latest official rows

## Caveat

The existing overwrite risk in `predictions` is still present until production save paths dual-write or switch. Phase C only adds manual append-only write capability and helper-level safety.
