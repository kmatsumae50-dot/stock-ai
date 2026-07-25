# Prediction Append-Only Official Filter Report

Generated: 2026-07-09

## Purpose

Add a production-only snapshot filter before append-only dual-write is introduced.

The goal is to prevent old experiment models, backfill rows, shadow rows, stale feature dates, and non-official model rows from becoming append-only official records.

This phase does not enable production dual-write.

## Official Filter Rule

A latest snapshot row is included only when all conditions are true:

- `ticker` matches the requested ticker
- `prediction_type` / source is `live`
- feature date equals the latest live feature date for the requested ticker
- `(target, model_name)` is one of the current production pairs:
  - `next_close_binary` / `lightgbm_binary`
  - `next_close_multiclass` / `lightgbm_multiclass`

Rows are excluded with a reason:

- `ticker_mismatch`
- `source_not_live`
- `non_live_source_backfill`
- `non_live_source_shadow`
- `non_live_source_experiment`
- `stale_feature_date`
- `non_official_target_model`

Binary remains the primary direction model. The current production multiclass row is included because daily live currently saves both binary and multiclass. Older multiclass experiment rows are excluded.

## Added Helpers

Added to `src/stock_ai/storage/sqlite_store.py`:

- `filter_official_prediction_snapshot_records`
- `list_official_latest_prediction_snapshot_records`
- `is_official_prediction_snapshot_record`
- `get_append_only_candidate_records_from_snapshot`

## CLI Behavior

`prediction-append-only-write-check` now shows:

- raw latest snapshot candidate count
- included official candidate count
- excluded candidate count
- included records
- excluded records
- excluded reason
- append-only row counts before/after
- existing `predictions` row count

`--write` now writes only included official candidates to append-only tables.

## Raw Candidate Breakdown

Dry-run against the production DB:

```text
raw latest snapshot candidates: 7
included official candidates: 2
excluded candidates: 5
append-only runs before: 0
append-only records before: 0
existing predictions rows: 65
```

Included:

| feature_date | target | model_name | model_type | source | reason |
|---|---|---|---|---|---|
| 2026-07-09 | next_close_binary | lightgbm_binary | binary | live | official_production_current |
| 2026-07-09 | next_close_multiclass | lightgbm_multiclass | multiclass | live | official_production_current |

Excluded:

| feature_date | target | model_name | source | reason |
|---|---|---|---|---|
| 2026-06-02 | next_close_multiclass | lightgbm_multiclass_quantile_balanced | live | stale_feature_date |
| 2026-06-02 | next_close_multiclass | lightgbm_multiclass_practical_3_0_5_tailboost | live | stale_feature_date |
| 2026-06-02 | next_close_multiclass | lightgbm_multiclass_practical_3_0_5_balanced | live | stale_feature_date |
| 2026-06-01 | next_close_multiclass | lightgbm_multiclass_quantile | live | stale_feature_date |
| 2026-06-01 | next_close_multiclass | lightgbm_multiclass_practical_3_0_5 | live | stale_feature_date |

The excluded rows are old multiclass experiment/reference model rows and should not become official append-only production records.

## Safe Write Verification

`--write` was tested against a copied database only:

```text
C:\Users\kmats\AppData\Local\Temp\stock_ai_official_filter_write_check.db
```

Result:

```text
raw latest snapshot candidates: 7
included official candidates: 2
excluded candidates: 5
append-only runs before: 0
append-only records before: 0
existing predictions rows: 65
record candidates: 2
inserted records: 2
duplicate records ignored: 0
append-only runs after: 1
append-only records after: 2
existing predictions rows after: 65
```

Production DB was not written by this `--write` verification.

Production DB after verification:

```text
prediction_runs rows: 0
prediction_records rows: 0
existing predictions rows: 65
```

## Tests Added

Tests cover:

- official production rows are included
- old experiment model rows are excluded
- backfill rows are excluded
- shadow rows are excluded
- experiment rows are excluded
- stale feature dates are excluded
- excluded reasons are returned
- append-only write check CLI dry-run does not write records
- append-only write check CLI writes only included records when pointed at a test DB
- existing `predictions` rows remain unchanged

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
- production append-only auto-save behavior
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

## Phase D Readiness

The official production filter is ready for Phase D.

Recommended next step:

- add a controlled dual-write path for current production `predict-binary` and `predict-multiclass`
- keep writing the existing `predictions` latest snapshot
- write append-only records only for rows passing this official filter
- keep append-only run history hidden from the default Prediction History view until validated
