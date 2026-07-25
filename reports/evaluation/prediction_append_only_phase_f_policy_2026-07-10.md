# Prediction Append-Only Phase F Policy

Date: 2026-07-10
Ticker: 7203.T

## Purpose

Phase F defines policy for future append-only production use.

This is a policy/report-only phase. It does not enable daily append-only writes and does not change the current production prediction flow.

Current append-only status:

- `prediction_runs`: 1
- `prediction_records`: 2
- latest run: `manual_live_dual_write_7203_T_phase_d_2026_07_09`
- latest feature date: 2026-07-09
- append-only vs snapshot: matches true
- mismatch count: 0

## Non-Changes In This Phase

The following remain unchanged:

- `daily_predict.ps1`
- Task Scheduler
- production model config
- binary production model
- multiclass production model
- `predict-binary` production save behavior
- `predict-multiclass` production save behavior
- `backfill-predictions` save behavior
- `upsert_predictions`
- existing SQLite `predictions` rows
- automatic daily append-only saving
- health-check missing-live source
- health-check Action required logic
- Prediction History metrics source
- evaluation metrics source
- feature generation logic
- NewsLimit 3
- daily 4 profile configuration
- `known_missed_live_dates`
- news source profiles
- earnings CSV
- offline experiment results
- shadow prediction save behavior

## Run Type Classification

Recommended run type policy:

| run_type | Meaning | Created by | is_official | is_latest | Metrics source | Health-check source | Normal Prediction History | Audit detail |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `scheduled_live_dual_write` | Future scheduled append-only copy of the production live snapshot after daily prediction succeeds. | Future opt-in daily flow after binary and multiclass live saves are confirmed. | Yes, after opt-in validation. | Yes, if it is the latest production snapshot mirror. | Yes, after metrics migration. | Candidate after Phase H/I only. | Yes, after migration. | Yes |
| `manual_live_dual_write` | Manual controlled append-only copy of the current production live snapshot. | `prediction-append-only-dual-write --write`. | No by default for future metrics. Current Phase D manual row is validation-only. | Can be latest for audit/snapshot comparison. | No by default. | No. | Summary only; details in append-only run history. | Yes |
| `manual_live_check` | Manual inspection or verification around the live snapshot. | Future manual check command, or dry-run style workflow. | No. | No. | No. | No. | No. | Yes |
| `dry_run` | No durable prediction output; used to inspect candidates. | CLI dry-run. | No. | No. | No. | No. | No. | Optional command log only |
| `backfill` | Historical reconstruction for research/evaluation. | `backfill-predictions` or a future append-only backfill writer. | No for live metrics. | No for live latest. | Only in separate backfill evaluation. | No. | Separate backfill view only. | Yes |
| `shadow` | Alternative model/reference prediction not used for production. | Shadow or challenger model workflow. | No. | No. | Only shadow diagnostics. | No. | No. | Yes, if stored at all |
| `experiment` | Offline experiment output. | Offline experiment scripts/reports. | No. | No. | Only experiment reports. | No. | No. | Optional, preferably report files |

Key rule: `source=live` is not enough to make a record official. Official status must depend on `run_type`, timing, target completeness, and policy.

## Official Row Selection Policy

Recommendation: select one official append-only record per:

```text
ticker + feature_date + target + production model slot
```

Binary and multiclass should have separate official records because they are different targets and can fail independently. For daily run health, the target set should still be considered complete only when both the binary and multiclass official records exist for the same feature date.

Recommended selection order:

1. Prefer the first complete successful `scheduled_live_dual_write` for the feature date.
2. Require both production targets to be present before considering the daily run complete:
   - `next_close_binary`
   - `next_close_multiclass`
3. Use `manual_live_dual_write` as audit by default, not as official metrics input.
4. Allow manual official override only with an explicit future override flag and reason, and only for same-day operational recovery. Do not use historical manual rows to fabricate live history.
5. Keep `official` and `latest` separate:
   - `is_official`: row used for official live coverage and future metrics.
   - `is_latest`: latest snapshot mirror for display/comparison.
6. Exclude `backfill`, `shadow`, `experiment`, and `dry_run` from live official row selection.

Earliest scheduled run vs latest scheduled run:

- Recommended official choice: first complete successful scheduled run.
- Reason: it best represents the first production prediction available to the user and avoids choosing a later, better-looking rerun after seeing more information.
- Latest scheduled run can still be marked latest for display, but should not replace the official metrics row unless a future explicit correction policy is introduced.

## Daily Dual-Write Opt-In Options

| Option | Description | Difficulty | Existing flow impact | Failure safety | Rollback | Duplicate risk | Idempotency | Manual/scheduled separation | Metrics migration |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | Add an opt-in flag directly to `daily_predict.ps1`. | Medium | Medium | Good if called after all saves. | Easy if flag defaults off. | Medium | Needs deterministic run id. | Good | Good |
| B | Add `--append-only-dual-write` to `predict-binary` and `predict-multiclass`. | High | High | Weaker; partial binary/multiclass writes are easier. | Medium | Medium | Harder across two commands. | Medium | Medium |
| C | After successful `daily_predict`, call a dedicated append-only dual-write command. | Medium | Low | Strong; writes only after snapshot saves are confirmed. | Easy; remove/disable one call. | Low to medium | Good with deterministic scheduled run id. | Good | Good |
| D | Create a separate `daily_append_only.ps1`. | Low | Low | Good, but can drift from daily run timing. | Easy | Medium | Needs coordination. | Good | Medium |
| E | Keep manual append-only only. | Low | None | Safest. | N/A | Low | Manual only. | Good | Poor for automated history |

Recommended option: Option C.

Use a dedicated append-only dual-write command after `daily_predict` has confirmed binary and multiclass live saves. Keep the opt-in disabled by default at first. This keeps the current snapshot save path unchanged, reduces partial-write risk, and gives a clean rollback path.

Implementation shape for the future:

```text
daily_predict successful
  -> binary live saved
  -> multiclass live saved
  -> confirmation passes
  -> if append-only opt-in enabled:
       prediction-append-only-dual-write --write --run-type scheduled_live_dual_write --idempotency-key ...
```

## Idempotency Policy

Recommended scheduled idempotency key:

```text
scheduled_live_dual_write:{ticker}:{feature_date}:binary_multiclass
```

Recommended behavior:

- Scheduled runs should be idempotent by default.
- Re-running the same scheduled append-only step for the same `ticker + feature_date + target set` should not create duplicate official records.
- If a rerun must be preserved as a separate audit run, it should use a new explicit run id and remain non-official unless an override is supplied.
- Manual runs should create a new run by default.
- Manual runs may reuse a run only when an explicit `--idempotency-key` or `--run-id` is provided.
- Retry after a transient failure should reuse the scheduled idempotency key.
- If binary succeeds but multiclass fails before append-only write, do not create an official append-only run.
- If a future writer stores partial records, those records should be `is_official=false`, `is_latest=false`, and marked in notes as partial.

Current schema note:

- There is no dedicated `idempotency_key` column yet.
- The safe near-term approach is to use deterministic `run_id` values for scheduled runs.
- A future schema extension can add `idempotency_key` and `run_status` if partial-run tracking becomes important.

## Partial Run And Retry Policy

Recommended policy:

- Write official append-only records only after the production snapshot contains both expected targets.
- Treat missing binary or missing multiclass as an append-only dual-write failure, not as a partial official run.
- Keep partial attempts out of official metrics.
- For manual troubleshooting, partial attempts can be logged as non-official audit records only if a future command supports it.
- Retries should be deterministic for scheduled runs and append-only for manual audit runs.

## Append-Only Health Visibility Future Policy

Current status:

- Visibility is Info only.
- Empty append-only tables are not Action required.
- Mismatches are not Action required.
- health-check missing-live logic still reads the existing `predictions` snapshot/history.

Future Action required candidates after opt-in is enabled and validated:

| Condition | Future severity candidate | Notes |
|---|---|---|
| Expected `scheduled_live_dual_write` missing for a date where daily prediction succeeded | Action required | Only after scheduled append-only is explicitly enabled. |
| Binary official exists but multiclass official is missing, or vice versa | Action required | Indicates partial append-only capture. |
| Snapshot vs latest official append-only mismatch after daily run | Action required or WARN | Start as WARN until false positives are understood. |
| Multiple official records for the same ticker/feature_date/target/model slot | Action required | Breaks metrics row selection. |
| Latest official append-only record is stale while snapshot is current | WARN or Action required | Depends on migration phase. |
| Append-only tables inaccessible after opt-in | Action required | Only after append-only becomes required. |

Do not promote any of these to Action required in the current phase.

## Prediction History Metrics Migration Policy

Current policy:

- Keep Prediction History metrics on existing `predictions` snapshot/history.
- Append-only records remain audit/reference only.

Migration prerequisites:

- At least 10 consecutive scheduled append-only days with snapshot comparison match.
- Prefer 20 consecutive trading days before using append-only for default live metrics.
- Zero missing scheduled official target sets during the observation period.
- Zero duplicate official rows for the same ticker/feature_date/target/model slot.
- Clear exclusion rules for `manual_live_dual_write`, `backfill`, `shadow`, `experiment`, and `dry_run`.
- Backfill and shadow records must remain separate from live metrics.
- UI must show which row source is used for metrics.
- Snapshot fallback should remain available for one migration phase.

Recommended metrics row source after migration:

```text
append-only records
  where source = 'live'
    and run_type = 'scheduled_live_dual_write'
    and is_official = true
    and target in production target set
```

Duplicate run prevention:

- Group by `ticker + feature_date + target + production model slot`.
- Require exactly one official record per group.
- If more than one official row exists, exclude that feature date from metrics and raise a future health visibility warning.

## Recommended Phase Plan

| Phase | Scope | Production impact |
|---|---|---|
| Phase F1 | Policy report only. | None |
| Phase F2 | Add run type policy constants and official selection helper. | None if read-only. |
| Phase F3 | Continue manual dual-write for several days and observe snapshot matches. | None |
| Phase G | Add opt-in daily append-only dual-write, default off. Test manually before scheduler use. | Low, opt-in only |
| Phase H | Promote selected append-only visibility issues from Info to WARN candidates. | Limited to visibility |
| Phase I | Evaluate migration of Prediction History metrics to append-only official records. | Requires explicit decision |

## Phase F2 Helper Candidates

If code is added in the next phase, keep it read-only first:

- `RUN_TYPE_POLICIES`
- `classify_append_only_run_type(run_type)`
- `is_append_only_run_official_candidate(run)`
- `select_official_append_only_records(records, runs)`
- `build_scheduled_append_only_run_id(ticker, feature_date, target_set)`
- `summarize_append_only_official_selection(records, runs)`

These helpers should not write to SQLite and should not change current health-check or metrics sources.

## Recommended Next Step

Proceed with Phase F2 as a read-only helper phase:

1. Add run type policy constants.
2. Add official selection helper.
3. Add tests for scheduled vs manual selection.
4. Keep `daily_predict`, Task Scheduler, health-check source, and metrics source unchanged.

Do not enable automatic daily append-only writes until the helper behavior is tested and the manual dual-write observation period remains clean.
