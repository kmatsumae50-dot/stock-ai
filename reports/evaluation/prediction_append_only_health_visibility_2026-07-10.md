# Prediction Append-Only Health Visibility

Date: 2026-07-10
Ticker: 7203.T

## Purpose

This phase adds lightweight visibility for append-only prediction tables.
It is informational only.

Append-only records are still an audit/history view and are not the source of truth for health-check, Prediction History metrics, or production evaluation.

## Added Display Locations

- Operations
  - Append-only Health Visibility
- CLI
  - `prediction-append-only-check`

## Displayed Items

- append-only table exists
- prediction_runs count
- prediction_records count
- latest run_id
- latest run_timestamp
- latest run_type
- latest feature_date
- latest official records count
- append-only vs snapshot matches
- mismatch count
- pending count
- hit count
- miss count
- unevaluated count
- latest compare status
- visibility classifications

## Visibility Classifications

Possible classification labels:

- `append_only_not_initialized`
- `append_only_empty`
- `append_only_available`
- `append_only_latest_matches_snapshot`
- `append_only_mismatch_detected_info`
- `append_only_manual_only`
- `append_only_not_health_source`

These labels are informational and do not create Action required warnings.

## Current Counts

- append-only tables exist: true
- prediction_runs: 1
- prediction_records: 2
- existing predictions: 65
- latest official records: 2
- pending: 2
- hit: 0
- miss: 0
- unevaluated: 2

## Current Snapshot Compare

- append-only vs snapshot matches: true
- mismatch count: 0
- latest compare status: matches
- latest run_id: `manual_live_dual_write_7203_T_phase_d_2026_07_09`
- latest run_type: `manual_live_dual_write`
- latest feature_date: 2026-07-09

## Non-Changes

The following were intentionally not changed:

- health-check Action required classification
- health-check missing-live source
- Prediction History metrics source
- evaluation metrics row selection
- daily_predict.ps1
- Task Scheduler time
- production model config
- binary production model
- multiclass production model
- predict-binary save behavior
- predict-multiclass save behavior
- backfill-predictions save behavior
- upsert_predictions behavior
- existing SQLite predictions rows
- automatic daily append-only write behavior
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- known_missed_live_dates
- news source profiles
- earnings CSV
- offline experiment results
- shadow prediction behavior

## Phase F Candidates

- Add a non-blocking append-only section to health-check output if desired.
- Decide whether scheduled/manual append-only runs should be displayed separately in Operations.
- Define a future opt-in `daily_predict` append-only dual-write flag.
- Define an official append-only metric row selection policy before using append-only for evaluation.
