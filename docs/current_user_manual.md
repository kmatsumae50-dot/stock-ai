# stock-ai Current Operation Manual

Last updated: 2026-09-04

Official working folder:

```powershell
C:\Users\kmats\Documents\ai-stock\stock-ai
```

This manual records the current production operation policy, model decision policy, health-check interpretation, and known missed live-date policy. It is meant to prevent future work from accidentally changing production behavior before the evidence is strong enough.

Related decision record:

- [Model Decision Notes 2026-06-24](../reports/evaluation/model_decision_notes_2026-06-24.md)

The old folder under `C:\Users\kmats\Documents\ai 株` plus `\stock-ai` must not be used as the working folder. It is referenced here only as a caution.

This project is an analysis and validation tool. It is not investment advice.

## Current Production Decision

Current production direction judgement mainly reads the binary model.

Important restrictions:

- The binary model is the main production reference for direction, but its advantage over baseline is weak.
- Do not use the binary model alone for investment decisions.
- Do not change the production binary model yet.
- Offline binary experiments, including class-weight and threshold variants, are somewhat promising but have not earned production adoption.
- The current multiclass model has strong flat bias and is not used for production direction judgement.
- Multiclass output is reference/experiment only.
- 3-class and boundary experiments remain future candidates, not production settings.

Current production state:

- Binary model changed: no.
- Multiclass model changed: no.
- `daily_predict.ps1` changed: no.
- SQLite prediction history changed: no.
- Production model config changed: no.

## Daily Operation

Use the official folder and official `.venv` for all app and CLI work:

```powershell
cd "C:\Users\kmats\Documents\ai-stock\stock-ai"
```

Streamlit app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src\stock_ai\app\main.py
```

Current Windows Task Scheduler settings:

- Task name: `stock-ai daily predict`
- Run time: `10:30`
- Working directory: `C:\Users\kmats\Documents\ai-stock\stock-ai`
- Command: `.\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3`

Current 10:30 operation state:

- latest daily_predict success: yes.
- `data_freshness_status`: `OK`.
- `stale_at_run_time`: `False`.
- `scheduled_for` and `actual_started_at` must be read separately. `StartWhenAvailable` can start a catch-up run after the scheduled time.
- A catch-up run must not be described as a prediction produced at 10:30.

Manual daily run, only when needed:

```powershell
.\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3
```

Health-check:

```powershell
.\.venv\Scripts\python.exe main.py health-check --ticker 7203.T
```

Do not operate from the old `Documents\ai 株` project location. All CLI commands should use the official `.venv` under `C:\Users\kmats\Documents\ai-stock\stock-ai`.

## Health Check Interpretation

Current expected health-check state:

- Overall: `WARN`
- Action required: `0`
- Known missed sections: `1`
- Known missed dates: `8`
- Known OK: `1`
- `unknown_source_count`: `0`
- `check-news`: `OK`

Interpretation:

- `Overall WARN` is acceptable when `Action required` is `0`.
- In that case, the system is in a known-WARN operating state rather than an unknown failure state.
- Known OK: TOPIX proxy / fallback. When the primary TOPIX symbol is unavailable, the stored series can come from fallback `1306.T`.
- Known missed dates are known live-prediction gaps, not current automation failures.
- `unknown_source_count: 0` and `check-news: OK` are the normal news-health state.
- If `Action required` appears, restore operational health before reading or trusting prediction output.

Known OK item:

- TOPIX proxy / fallback: acceptable as a known fallback state, with the actual fallback symbol kept visible.

Known missed summary:

- Known missed section count: `1`.
- Known missed date count: `8`.

## Known Missed Live Dates Policy

Live predictions must not be fabricated after the fact.

Policy:

- A `live` prediction means the prediction was actually produced by daily operation at the time.
- If the run happened before the required market data was available, the missing date should be recorded as `data_not_available_at_run_time`.
- Do not create a fake live prediction for that date.
- If necessary, register the date in `known_missed_live_dates`.
- Backfill is allowed only as backfill/research data and must remain separate from live history.
- Known missed dates must be shown explicitly in Evaluation, Data, Prediction History, and health-check context.

Current known missed live dates:

| Feature date | Reason code | Action | Backfill |
| --- | --- | --- | --- |
| 2026-06-15 | `data_not_available_at_run_time` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-06-23 | `data_not_available_at_run_time` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-07-13 | `task_scheduler_not_run` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-07-21 | `prediction_generation_failed` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-08-07 | `task_scheduler_not_run` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-08-10 | `task_scheduler_not_run` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-08-12 | `task_scheduler_not_run` | `do_not_create_live_prediction` | `allow_backfill: true` |
| 2026-08-24 | `task_scheduler_not_run` | `do_not_create_live_prediction` | `allow_backfill: true` |

The current `known_missed_live_dates` contents are intentional and should not be changed casually.

## Prediction Date Semantics

Read prediction dates this way:

- Feature date: the data date used by the model.
- Target close date: the next trading day's close after the feature date.
- Saved at: the timestamp when the prediction row was written into SQLite.
- Source: `live` or `backfill`.
- `predictions.date`: in the app, read this as the Feature date.

Operational meaning:

- A live row for `predictions.date = 2026-06-24` means the model used features through 2026-06-24.
- The actual target is the next available trading day's close.
- `saved_at` or `created_at` is not the market date. It is only the SQLite write time.
- Backfill rows can help evaluation, but they are not live operational history.

## Append-Only Audit Semantics

The append-only tables are an audit aid and are not yet the production health or metric source.

- Current audit inventory: one `prediction_runs` row and two `prediction_records` rows.
- The existing run is `manual_live_dual_write` and was created for validation.
- Its database rows contain legacy `is_official=1` and `is_latest=1` flags.
- The policy layer excludes manual dual-write runs from official selection and live metrics.
- Policy-derived official eligibility is authoritative. A raw database `is_official` flag alone does not make a manual run an official scheduled prediction.
- The latest production snapshot remains the `predictions` table until a separately reviewed migration changes that contract.
- A snapshot mismatch is currently informational because automatic scheduled dual-write is not enabled.

Do not rewrite append-only flags or migrate the database merely to make the current audit run look official.

## Binary Shadow Semantics

Binary shadow output is manual/opt-in and is not generated by a Windows scheduled task.

- Current reviewed status: two confirmed rows and zero pending rows.
- Shadow rows are not stored as production live predictions.
- Shadow results are supporting evidence only and do not change the production model.

## Binary vs Multiclass Usage

Binary:

- Primary production direction reference.
- Materially underperforming simple baselines in the latest reviewed live diagnostics.
- Do not use alone for investment decisions.
- Continue monitoring live binary accuracy, balanced accuracy, and up recall.
- Do not change the production binary model yet.

Multiclass:

- Reference and experiment only.
- Current 5-class model is structurally flat-biased.
- Do not use it as the main production direction signal.
- Do not replace binary direction judgement with multiclass output.
- 3-class and boundary experiments are future candidates only.

Practical reading:

- Use binary as the primary direction lens.
- Use multiclass as context, especially for risk/flat-bias awareness.
- If binary and multiclass disagree, do not force a stronger production decision from multiclass.

## Offline Experiments Status

Binary experiment status:

- Historical walk-forward accuracy is around the 50% range.
- The edge versus simple baselines is weak.
- Up recall is relatively low.
- `class_weight_balanced` and balanced threshold ideas are candidates.
- These candidates are not production changes yet.

Multiclass experiment status:

- The current 5-class model predicts flat about 95% of the time.
- This flat bias appears structural in the current setup.
- It is hard to claim multiclass is better than binary for direction judgement right now.
- The current 5-class model should not be the production main signal.
- `3class_down_flat_up` and boundary experiments are future candidates, not adopted production settings.

Reference files:

- [Model Decision Notes 2026-06-24](../reports/evaluation/model_decision_notes_2026-06-24.md)
- `reports/evaluation/multiclass_offline_boundary_experiments_2026-06-24.csv`
- `reports/evaluation/multiclass_offline_boundary_experiments_2026-06-24.json`
- `reports/autonomous/production_prediction_integrity_audit_2026-09-02.md`

## What Not To Change Yet

Do not casually change the following while the 10:30 operation and offline experiments are still under observation:

- production model config
- binary production model
- multiclass production model
- `daily_predict.ps1`
- Task Scheduler time
- SQLite schema
- prediction history
- feature generation
- `NewsLimit 3`
- daily 4 profile configuration
- `known_missed_live_dates`
- news source profile
- earnings CSV
- offline experiment result files

Current daily 4 profile configuration should remain unchanged. The current operating assumption is that the daily profile set is stable enough for observation, not that it is permanently optimal.

## Next Monitoring Items

Near-term monitoring:

- Confirm 10:30 operation keeps `stale_at_run_time: False`.
- Confirm `data_freshness_status: OK` continues for several more runs.
- Continue monitoring live binary accuracy.
- Continue monitoring live binary balanced accuracy.
- Continue monitoring live binary up recall.
- If `Action required` appears, fix the health-check cause before interpreting model output.
- If unknown sources appear, review source profiles conservatively.
- Keep `unknown_source_count` at `0`.
- Keep `check-news` at `OK`.
- Do not change production model settings until offline experiments show sufficient advantage.
- Keep append-only audit status separate from the snapshot health source.
- Keep manual shadow generation separate from scheduled production operation.

Production-change threshold:

- More evidence is needed before changing production binary behavior.
- Multiclass must reduce flat bias offline before it can become a production direction candidate.
- Any production model change should be documented in a new decision note before implementation.

## README vs Manual Role

README:

- Short project entry point.
- Current operation snapshot.
- Basic app/CLI commands.
- Links to this manual and the model decision report.

This manual:

- Full current operation policy.
- Health-check interpretation.
- Known missed live-date policy.
- Prediction date semantics.
- Model usage rules.
- Guardrails on what not to change yet.

Environment recovery and migration procedures are maintained in [environment_handoff.md](environment_handoff.md).
