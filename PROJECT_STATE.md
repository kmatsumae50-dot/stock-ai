# PROJECT_STATE.md

Last updated: 2026-09-04

## Current Scope

This repository is an operational stock-analysis and daily-prediction system centered on Toyota Motor (`7203.T`). It includes price and external-market ingestion, feature generation, LightGBM predictions, SQLite history, Streamlit views, news quality controls, offline evaluation, and Windows scheduled operation.

The system is not an automated trading system and is not ready for broad Nikkei 225 production use. Production behavior remains intentionally conservative.

## Operational Posture

- Daily prediction is scheduled for 10:30 with `FetchNews`, `DailyNewsProfiles`, and `NewsLimit 3`.
- Autonomous Operations and Autonomous Research are separately scheduled behind health and freshness gates.
- Health is acceptable only when `Action required` is zero. The current known WARN classes are documented missing live dates and the TOPIX proxy / fallback.
- Eight known missed live dates are recorded in `configs/settings.yml`. They must not be recreated as live predictions.
- The binary model is the primary production direction reference, but current reviewed evidence materially underperforms simple baselines. Production change is not ready.
- The multiclass model is strongly flat-biased and remains reference/experimental only.
- Binary shadow generation is manual/opt-in. Two rows are confirmed and no automatic generation task is registered.
- Append-only tables are audit support only. The `predictions` snapshot remains the health and evaluation source.

## Data And History Semantics

- `predictions.date` is effectively the Feature date used by the model.
- Target close date is the next trading day's close after the Feature date.
- `created_at` is the SQLite save timestamp.
- `live`, `backfill`, shadow, and append-only audit rows must remain distinct.
- Scheduled time and actual task start time must be reported separately because `StartWhenAvailable` can produce catch-up runs.

## Known Gaps

- Earnings event CSV files are present but header-only. The pipeline therefore uses the documented empty-event fallback. An `earnings_data_pipeline` is a future phase, not part of current production.
- Five old experimental multiclass rows are labeled `live`. Production history views filter them, but some aggregate diagnostics can include them unless they filter the production model slot. Do not delete the rows; harden aggregation in a later Level 1 maintenance task.
- Append-only audit rows do not cover current scheduled production snapshots. The mismatch is informational until an explicit migration is reviewed.
- Runtime state can lag actual scheduler and filesystem state. Existing reconciliation CLIs should be used instead of editing state JSON by hand.
- Source code alone is not a complete backup. Database, models, data, and selected runtime lifecycle records require a separate backup.

## Next Safe Work

1. Review and commit the current dirty main changes in small groups. Do not auto-commit runtime/generated files.
2. Create and verify an offline backup using `docs/environment_handoff.md`.
3. Re-run Operations after a fresh daily cycle and let Research follow its normal gate without `--force`.
4. Run the Codex Agent preflight read-only. Only after main is clean and all gates are healthy may `recent_regime_robustness` be considered.
5. Treat `earnings_data_pipeline` and production-model changes as separate reviewed phases.

## Reference Documents

- `README.md`
- `docs/current_user_manual.md`
- `docs/environment_handoff.md`
- `reports/autonomous/production_prediction_integrity_audit_2026-09-02.md`
- `reports/autonomous/operational_state_reconciliation_2026-09-04.md`

For any future task, preserve data-leakage controls, production history integrity, and the distinction between analysis evidence and production decisions.
