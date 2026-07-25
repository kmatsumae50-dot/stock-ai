# Prediction Append-Only Phase D: Controlled Dual-Write

Date: 2026-07-09
Ticker: 7203.T

## Purpose

Phase D adds a manual, opt-in controlled dual-write path for append-only prediction history.
The production snapshot table remains the source used by the current app and daily operation.

This phase does not switch daily_predict, Prediction History, health-check missing-live logic, or evaluation metrics to append-only reads.

## Implemented

- Added `prediction-append-only-dual-write`.
- Default mode is dry-run unless `--write` is explicitly passed.
- Added optional `--idempotency-key` to make repeated manual writes safe.
- Added `prediction-append-only-compare`.
- Added comparison helper to verify append-only latest official records against the current `predictions` snapshot.
- Added tests for dry-run, idempotent write, official filtering, and snapshot comparison.

## Official Candidate Filter

Current production snapshot candidates:

- Raw latest snapshot candidates: 7
- Included official candidates: 2
- Excluded candidates: 5

Included:

- 2026-07-09 / next_close_binary / lightgbm_binary / source=live
- 2026-07-09 / next_close_multiclass / lightgbm_multiclass / source=live

Excluded:

- 2026-06-02 / lightgbm_multiclass_quantile_balanced / stale_feature_date
- 2026-06-02 / lightgbm_multiclass_practical_3_0_5_tailboost / stale_feature_date
- 2026-06-02 / lightgbm_multiclass_practical_3_0_5_balanced / stale_feature_date
- 2026-06-01 / lightgbm_multiclass_quantile / stale_feature_date
- 2026-06-01 / lightgbm_multiclass_practical_3_0_5 / stale_feature_date

## Production Manual Dual-Write Result

Command:

```powershell
.\.venv\Scripts\python.exe main.py prediction-append-only-dual-write --ticker 7203.T --write --idempotency-key phase-d-2026-07-09
```

Result:

- run_id: `manual_live_dual_write_7203_T_phase_d_2026_07_09`
- record candidates: 2
- inserted records: 2
- duplicate records ignored: 0
- append-only runs: 0 -> 1
- append-only records: 0 -> 2
- existing predictions rows: 65 -> 65

Idempotency check with the same key:

- inserted records: 0
- duplicate records ignored: 2
- append-only runs remained 1
- append-only records remained 2

## Snapshot Compare

Command:

```powershell
.\.venv\Scripts\python.exe main.py prediction-append-only-compare --ticker 7203.T
```

Result:

- snapshot official count: 2
- snapshot excluded count: 5
- append official count: 2
- matches: true

Matched rows:

- 2026-07-09 / next_close_binary / lightgbm_binary / label 0 / confidence 0.5299918287295814
- 2026-07-09 / next_close_multiclass / lightgbm_multiclass / label 2 / confidence 0.4784916036382058

## Non-Changes

The following were intentionally not changed:

- production model config
- binary production model
- multiclass production model
- daily_predict.ps1
- Task Scheduler time
- default predict-binary / predict-multiclass save behavior
- upsert_predictions behavior
- existing SQLite `predictions` rows
- Prediction History UI
- health-check missing-live logic
- evaluation metrics row selection
- feature generation
- NewsLimit 3
- daily 4 profile configuration
- known_missed_live_dates
- news source profiles
- earnings CSV
- shadow prediction behavior

## Current Status

Append-only controlled dual-write is available for manual use.
The existing `predictions` table remains the production latest snapshot source.

Next recommended phase:

- Phase E: add read-side UI/details for append-only run history while keeping default Prediction History stable.
