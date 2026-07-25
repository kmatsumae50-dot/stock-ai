# Prediction Append-Only Phase E: UI History View

Date: 2026-07-09
Ticker: 7203.T

## Purpose

Phase E adds a read-only UI view for append-only prediction runs and records.
This is an audit/history view only.

The existing `predictions` snapshot remains the source of truth for Prediction History metrics, health-check missing-live logic, and production displays.

## Added UI Location

- Prediction History tab
  - Expander: `Append-only Run History`

## Displayed Run Summary

- total runs
- total records
- official records
- latest records
- existing snapshot rows
- latest run id
- latest run timestamp
- latest run type
- latest source
- latest feature date
- data_freshness_status
- stale_at_run_time
- notes

## Displayed Record Summary

- feature_date
- target_close_date
- target
- model_name
- model_type
- predicted_label
- predicted_direction
- confidence
- source
- is_official
- is_latest
- result
- actual_return
- evaluated_at

## Feature-Date Run History

The UI shows a feature-date summary so repeated runs for the same feature date can be inspected.

Displayed fields:

- feature_date
- run_count
- record_count
- official_records
- latest_records
- binary_records
- multiclass_records
- run_types
- sources
- latest_run_timestamp
- latest_run_id

## Append-Only vs Snapshot Compare

The UI displays the same comparison concept as:

```powershell
python main.py prediction-append-only-compare --ticker 7203.T
```

Current result:

- snapshot official count: 2
- snapshot excluded count: 5
- append official count: 2
- matches: true
- mismatch count: 0
- matched records: 2

## Current Counts

- append-only runs: 1
- append-only records: 2
- existing predictions rows: 65
- source counts: live=2
- run type counts: manual_live_dual_write=1

## Evaluation Status Display

The UI summarizes current append-only record result status:

- pending count
- hit count
- miss count
- unevaluated count
- latest pending target_close_date
- latest evaluated_at

No evaluation update is performed by this UI.

## Not Switched In This Phase

The following are intentionally unchanged:

- Prediction History metrics still use existing `predictions` snapshot rows.
- health-check missing-live logic still uses existing prediction history logic.
- evaluation metrics row selection is unchanged.
- daily_predict is not writing append-only automatically.
- Task Scheduler is unchanged.
- predict-binary / predict-multiclass default save behavior is unchanged.
- upsert_predictions behavior is unchanged.
- existing SQLite `predictions` rows are unchanged.
- production models are unchanged.
- NewsLimit 3 and daily profile configuration are unchanged.

## Test Coverage Added

- empty append-only tables can produce UI summaries
- one manual dual-write run / two records produce summaries
- multiple runs for the same feature_date are counted
- append-only vs snapshot comparison can be displayed
- pending / hit / miss / unevaluated counts are aggregated

## Next Phase F Candidates

- Add a richer run-details drilldown by run_id.
- Add optional append-only health-check visibility without changing missing-live logic.
- Add an explicit scheduled/manual run distinction in the UI.
- Consider an opt-in daily_predict append-only flag only after more manual validation.
- Define official metric row selection before any append-only-backed evaluation.
