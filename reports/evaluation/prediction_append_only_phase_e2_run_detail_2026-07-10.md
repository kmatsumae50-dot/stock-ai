# Prediction Append-Only Phase E2: Run ID Drilldown

Date: 2026-07-10
Ticker: 7203.T

## Purpose

Phase E2 adds a read-only run_id drilldown for append-only prediction history.
It is for audit and verification only.

Prediction History metrics, health-check missing-live logic, production model behavior, and daily_predict remain unchanged.

## Added UI Location

- Prediction History
  - Append-only Run History
    - Run ID drilldown
    - Selected run detail

## Run ID Detail Fields

The selected run displays:

- run_id
- run_timestamp
- ticker
- run_type
- source
- command_name
- working_directory
- latest_feature_date
- latest_raw_date
- expected_latest_trading_date
- data_freshness_status
- stale_at_run_time
- created_at
- notes

## Selected Run Record Detail

The selected run record table displays:

- record_id
- feature_date
- target_close_date
- target
- model_name
- model_type
- predicted_label
- predicted_direction
- probability_up
- probability_down
- confidence
- source
- is_official
- is_latest
- actual_return
- actual_label
- actual_direction
- result
- evaluated_at
- notes

## Selected Run Summary

The selected run summary displays:

- record_count
- official_record_count
- latest_record_count
- binary_record_count
- multiclass_record_count
- pending_count
- hit_count
- miss_count
- unevaluated_count
- targets included
- model_names included

## Selected Run vs Snapshot Compare

The selected run is compared with the current official predictions snapshot.

Displayed comparison fields:

- matched count
- mismatch count
- missing in append-only
- missing in snapshot
- confidence delta
- predicted label match
- predicted direction match
- confidence match
- mismatch reason

This comparison is audit-only. It is not used by Prediction History metrics or health-check.

## Feature-Date Run History

Feature-date run history now includes:

- feature_date
- run_count
- run_ids
- latest run
- official run ids
- manual run ids
- scheduled run ids
- binary record count
- multiclass record count

## Current Counts

At the time of this phase:

- append-only runs: 1
- append-only records: 2
- existing predictions rows: 65
- append-only vs snapshot: matches true

## Not Changed

The following remain unchanged:

- production model config
- binary production model
- multiclass production model
- daily_predict.ps1
- Task Scheduler time
- predict-binary default save behavior
- predict-multiclass default save behavior
- backfill-predictions save behavior
- upsert_predictions behavior
- existing SQLite predictions rows
- automatic daily append-only writes
- Prediction History metrics source
- health-check missing-live logic
- evaluation metrics row selection
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- known_missed_live_dates
- news source profiles
- earnings CSV
- offline experiment results
- shadow prediction behavior

## Tests Added

- run_id detail can be built for one selected run
- empty run history does not break detail helpers
- one run / two records creates run detail
- binary and multiclass records are counted separately
- pending / hit / miss / unevaluated counts are aggregated
- selected run vs snapshot compare reports matches
- selected run vs snapshot compare reports value mismatches
- same feature_date with multiple runs creates a feature-date run summary

## Phase F Candidates

- Add optional append-only health visibility without changing existing health-check decisions.
- Add run_id detail links from future scheduled/manual run rows.
- Consider an opt-in daily_predict append-only flag after more manual validation.
- Define an official append-only metric row selection policy before using append-only for evaluation.
