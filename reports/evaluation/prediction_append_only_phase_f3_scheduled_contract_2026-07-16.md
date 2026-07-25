# Prediction Append-Only Phase F3 Scheduled Contract

Date: 2026-07-16  
Ticker: 7203.T

## Scope

Phase F3 adds an opt-in scheduled dual-write contract, deterministic idempotency, official
eligibility preflight, and a dry-run/test-database CLI. It does not connect append-only writes
to `daily_predict.ps1` or Task Scheduler, and it does not write scheduled rows to the production
database.

The existing `predictions` table remains the production snapshot and the source for Prediction
History metrics, health-check missing-live detection, and evaluation metrics.

## Scheduled Dual-Write Contract

The `ScheduledDualWriteContract` captures:

- ticker, feature date, and expected feature date;
- scheduled, execution-start, and execution-complete timestamps;
- run type, execution mode, command name, and source;
- binary and multiclass success flags;
- data-freshness status, stale-at-run flag, latest raw/feature dates, and expected trading date;
- task name and last task result;
- notes, target-set version, and idempotency key.

No schema migration is introduced in Phase F3. A test append-only write serializes the full
contract as versioned JSON in the existing `prediction_runs.notes` field. Existing run-table
columns continue to receive their matching freshness and execution metadata.

## Execution Modes

Supported modes are:

- `on_schedule`;
- `delayed_start`;
- `catch_up_after_missed_schedule`;
- `manual`;
- `retry`;
- `unknown`.

`on_schedule`, `delayed_start`, `catch_up_after_missed_schedule`, and `retry` may pass official
eligibility. Delayed and catch-up runs are marked explicitly. A catch-up may record the actual
current feature date when a normal current prediction exists. It must not manufacture a live
prediction for an earlier missing date such as 2026-07-13.

## Official Eligibility Preflight

A contract is eligible only when:

- `run_type` is `scheduled_live_dual_write`;
- source is `live`;
- execution mode is scheduled, delayed, catch-up, or retry;
- feature date equals expected feature date;
- freshness status is `OK` and `stale_at_run_time` is false;
- binary and multiclass snapshot saves both succeeded;
- exactly one current production binary slot exists;
- exactly one current production multiclass slot exists;
- no production slot is duplicated;
- all append-only record candidates were written when write mode is used.

Manual, backfill, shadow, experiment, non-live, stale, partial, duplicate-slot, and feature-date
mismatch cases are not eligible. Older experiment models remain visible as ignored records but
do not replace or duplicate a production slot.

The helper returns eligibility, complete-target status, present/missing/duplicate slots,
reason codes, execution mode, delayed/catch-up status, attempt status, and selected-candidate
status. It does not update `is_official` or `is_latest`.

## Idempotency

The scheduled key is:

```text
scheduled_live_dual_write:{ticker}:{feature_date}:binary_multiclass
```

`scheduled_for`, execution timestamps, retry timing, and catch-up timing do not change the key.
A retry for the same ticker and feature date reuses the same logical run ID. Manual runs receive
a distinct generated key unless an explicit key is provided.

For a partial scheduled write, the existing logical run is retained. A retry appends only the
missing production slot because the append-only unique index ignores an already-recorded slot.
The original successful slot is not overwritten. Phase F3 does not add a separate attempt table,
so per-attempt audit rows remain a possible later schema decision rather than part of this phase.

## Partial, Retry, And Catch-Up Policy

- one successful slot is `partial` and cannot be official;
- zero successful slots is `failed` and cannot be official;
- retry uses the same deterministic key and can complete the existing logical run;
- the first complete scheduled logical run remains the F2 policy candidate;
- a delayed or catch-up run must still use fresh data for its actual feature date;
- if the missed feature date has no contemporaneous snapshot prediction, no live row is created;
- backfill, if ever requested separately, remains `source=backfill`;
- an append-only write failure does not invalidate or remove the already-saved snapshot prediction.

## Opt-In CLI

Added command:

```powershell
python main.py prediction-append-only-scheduled-contract-check --ticker 7203.T
```

The default is dry-run. Options include feature and expected dates, scheduled/execution times,
execution mode, binary/multiclass status, freshness status, stale flag, task metadata,
idempotency key, database path, and `--write`.

`--write` requires an explicit `--database-path`. The command refuses a write when that resolved
path is the configured production database. This keeps scheduled production writes opt-in and
off during Phase F3.

## Dry-Run Result

Production database dry-run on 2026-07-16:

- resolved feature date: 2026-07-16;
- expected feature date: 2026-07-16;
- raw snapshot candidates: 2;
- official filtered candidates: 2;
- present slots: binary and multiclass;
- complete target set: true;
- contract eligibility: true;
- generated key: `scheduled_live_dual_write:7203.T:2026-07-16:binary_multiclass`;
- append-only counts before/after: 1 run / 2 records;
- existing predictions: 73;
- production database modified: false;
- selected official run in the database: none.

## Temporary Copy Write Result

A copied database in the Windows temporary directory was used for opt-in write verification.
The temporary copy changed from 1 to 2 append-only runs and from 2 to 4 append-only records.
The new scheduled run contained one binary and one multiclass record, both with
`is_official=0` and `is_latest=0`. Existing predictions stayed at 73 rows. The temporary copy was
removed after verification.

The production database remained at:

- `prediction_runs`: 1;
- `prediction_records`: 2;
- `predictions`: 73.

Its only append-only run is still the manual 2026-07-09 run, which is not an F2 official
candidate.

## Future Daily Predict Call Contract

The future Phase G call order should be:

1. complete the current binary and multiclass snapshot saves;
2. confirm both save outcomes and freshness metadata;
3. invoke the dedicated scheduled contract service with the actual feature date;
4. log append-only success, partial status, or warning;
5. preserve the existing daily success result if only the append-only audit write fails.

The future caller should pass all contract fields, especially the expected feature date,
execution mode, stale status, binary/multiclass save outcomes, task metadata, and deterministic
key. Append-only failure should produce a non-fatal warning and retain the same key for retry.
Task Scheduler catch-up must describe the actual execution mode and must never fabricate the
missed historical live row.

Rollback remains simple while the integration is opt-in: stop invoking the service. The
snapshot path is untouched. Any test-only scheduled rows are confined to copied databases.

## Exit-Code Policy For Phase G

- invalid contract or attempted production write from the Phase F3 CLI: non-zero;
- snapshot save failure: existing daily behavior remains authoritative;
- append-only audit write failure after successful snapshot saves: log warning and keep the
  existing daily success decision;
- partial append-only result: warning plus deterministic retry key;
- complete audit write: zero.

This policy is documented only. `daily_predict.ps1` is unchanged in Phase F3.

## Phase G Entry Conditions

Phase G can be considered after:

- repeated dry-run and copied-DB checks remain deterministic;
- partial and retry behavior is accepted operationally;
- the official-selector integration point for contract freshness metadata is approved;
- logging and warning wording for append-only-only failures is approved;
- a decision is made on whether separate retry-attempt rows are required;
- an explicit opt-in decision is made before editing `daily_predict.ps1`.

## Non-Changes

Phase F3 does not change production models, prediction commands, `daily_predict.ps1`, Task
Scheduler, snapshot upserts, production database rows, append-only production rows,
`is_official`/`is_latest`, health-check sources or severity, Prediction History metrics,
evaluation metrics, feature generation, news settings, known missed dates, earnings data,
offline experiments, or shadow behavior.
