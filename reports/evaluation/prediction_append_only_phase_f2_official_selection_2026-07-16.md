# Prediction Append-Only Phase F2 Official Selection

Date: 2026-07-16  
Ticker: 7203.T

## Purpose

Phase F2 translates the approved Phase F policy into read-only code. It defines run type
policies, production model slots, complete-run diagnostics, and deterministic future official
candidate selection.

This phase does not write append-only rows and does not update `is_official` or `is_latest`.
The existing `predictions` snapshot remains the source for production history, health checks,
and evaluation metrics.

## Run Type Policy

| run_type | official eligible | live metrics eligible | audit only | complete target set required | allowed source | priority |
|---|---:|---:|---:|---:|---|---:|
| `scheduled_live_dual_write` | yes | yes | no | yes | `live` | 10 |
| `manual_live_dual_write` | no | no | yes | yes | `live` | 20 |
| `manual_live_check` | no | no | yes | no | `live` | 30 |
| `dry_run` | no | no | yes | no | none | 90 |
| `backfill` | no | no | yes | no | `backfill` | 60 |
| `shadow` | no | no | yes | no | `shadow` | 70 |
| `experiment` | no | no | yes | no | `experiment` | 80 |

An unknown run type is classified as audit-only and is never official eligible.

## Production Model Slots

| slot | target | model_name | model_type | allowed source |
|---|---|---|---|---|
| binary | `next_close_binary` | `lightgbm_binary` | binary | `live` |
| multiclass | `next_close_multiclass` | `lightgbm_multiclass` | multiclass | `live` |

Older multiclass experiment/reference models are not production slots.

## Complete Run Rule

A run is complete for one `ticker + feature_date + run_id` only when:

- exactly one binary production record exists;
- exactly one multiclass production record exists;
- both records use `source=live`;
- both records match the configured production target and model name;
- neither production slot is duplicated.

Non-production, non-live, backfill, shadow, and experiment records are reported as ineligible.
Extra ineligible records do not hide an otherwise complete production pair, but they remain
visible in diagnostics. A duplicated production slot makes the run incomplete.

Returned diagnostics include:

- `is_complete`;
- present, missing, and duplicate slots;
- selected binary and multiclass record IDs;
- ineligible records and reason codes.

## Official Candidate Selection

Selection is read-only and is calculated independently from current database flags.

1. Group append-only history by `ticker + feature_date`.
2. Consider only `scheduled_live_dual_write` runs.
3. Require an allowed run source and a complete production target set.
4. Sort by `run_timestamp` ascending.
5. If timestamps are equal, sort by `run_id` ascending for deterministic selection.
6. Select the first complete scheduled run.
7. Return binary and multiclass as separate selected records.
8. Keep the latest run as a separate diagnostic concept.

Manual, backfill, shadow, experiment, and dry-run rows are never selected. Current database
`is_official` and `is_latest` values are shown for comparison but are not changed.

## Retry And Partial Diagnostics

The diagnostics identify:

- multiple scheduled runs for one feature date;
- partial scheduled runs;
- a later complete retry after an earlier partial run;
- multiple complete scheduled runs;
- the selected official candidate and the latest run;
- manual runs interleaved with scheduled runs;
- missing binary or multiclass slots;
- duplicate production slots;
- non-production records mixed into a run;
- current DB flag state and snapshot comparison status.

The first complete scheduled run remains the candidate even if a later complete or manual run
exists. This prevents a later rerun from replacing the prediction first available to users.

## Read-Only CLI

Added command:

```powershell
python main.py prediction-append-only-official-selection-check --ticker 7203.T
```

The command reports feature dates, scheduled/manual counts, complete/partial counts, selected
run and record IDs, latest run, missing/duplicate slots, reason codes, current DB flags, and
snapshot comparison status.

It does not initialize new append-only data, add records, or update any flag.

## Current Database Diagnostics

Current result on 2026-07-16:

- feature dates: 1;
- scheduled runs: 0;
- manual runs: 1;
- complete scheduled runs: 0;
- partial scheduled runs: 0;
- selected official candidate: none;
- latest run: `manual_live_dual_write_7203_T_phase_d_2026_07_09`;
- latest run type: `manual_live_dual_write`;
- classifications: `manual_only`, `no_scheduled_official_candidate`;
- current DB official records for 2026-07-09: 2;
- selected candidate DB flag comparison: not applicable.

The existing manual run is deliberately not selected despite its historical DB
`is_official=1` and `is_latest=1` flags. The policy-derived future official candidate remains
none.

The current snapshot has advanced to feature date 2026-07-16 while the only append-only manual
run is for 2026-07-09. Therefore the existing append-only snapshot comparison is currently a
mismatch with two missing current snapshot records. This remains informational and does not
change health-check severity in Phase F2.

Current row counts:

- `prediction_runs`: 1;
- `prediction_records`: 2;
- existing `predictions`: 73.

## Tests

Phase F2 tests cover:

- explicit run type policy and production slots;
- complete scheduled selection;
- manual/backfill/shadow/experiment exclusion;
- partial scheduled exclusion;
- partial-first and complete-retry selection;
- first complete selection when several complete runs exist;
- latest run differing from the official candidate;
- separate binary and multiclass records;
- non-production record exclusion;
- duplicate slot detection;
- deterministic equal-timestamp selection;
- snapshot comparison without DB flag mutation;
- CLI read-only behavior across runs, records, flags, and snapshot rows.

## Non-Changes

Phase F2 does not change:

- `daily_predict.ps1` or Task Scheduler;
- production model config or model artifacts;
- binary/multiclass/backfill save behavior;
- `upsert_predictions`;
- existing SQLite `predictions`, `prediction_runs`, or `prediction_records` rows;
- `is_official` or `is_latest` values;
- automatic daily append-only saving;
- health-check missing-live or Action required sources;
- Prediction History metrics source;
- evaluation metrics source;
- feature generation, NewsLimit 3, daily profiles, known missed dates, news profiles, earnings
  data, offline experiments, or shadow behavior.

## Recommended Phase F3

Phase F3 should use these read-only helpers to observe controlled manual and synthetic retry
cases, confirm policy diagnostics over several feature dates, and finalize the opt-in contract
for a future scheduled dual-write command. Automatic daily append-only saving should remain off
until that observation is complete and an explicit Phase G decision is made.
