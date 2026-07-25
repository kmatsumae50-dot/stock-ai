# Model Decision Notes (2026-06-24)

## Current operational health
- Overall status: WARN
- Action required: 0
- Known missed dates: 2
- Known OK: 1

## Known missed live dates
- 2026-06-15: data_not_available_at_run_time / action=do_not_create_live_prediction / allow_backfill=True
  - Reason: daily_predict ran at 2026-06-15 09:12, but latest available feature data was 2026-06-12 at run time.
- 2026-06-23: data_not_available_at_run_time / action=do_not_create_live_prediction / allow_backfill=True
  - Reason: daily_predict ran at 2026-06-23 08:30, but latest available raw/feature data was 2026-06-22 at run time.

## 10:30 scheduler observation
- Current daily run time: 10:30
- Status: observing
- Note: 10:30 is being monitored before deciding whether 18:00 is necessary.

## Binary evaluation summary
- Role: primary direction reference
- Historical accuracy: about 50.5%
- Historical balanced accuracy: about 50.5%
- Historical up recall: about 37%
- Decision: production change not ready

## Binary offline experiment summary
- Summary: Class weight and threshold variants showed small offline improvements, but the edge versus simple baselines remains weak.
- Decision: proposal only; not applied to production

## Multiclass flat bias summary
- Role: experimental/reference only
- Current issue: structurally flat-biased
- Predicted flat ratio: about 95.5%
- Decision: do not use as primary direction judgement

## Multiclass offline boundary experiment summary
- Summary: 3-class and boundary variants remain future research candidates. No production multiclass change has been applied.
- Decision: future experiment candidate

## Current production decision
- Binary model changed: False
- Multiclass model changed: False
- daily_predict changed: False
- SQLite history changed: False
- Decision: no production model change has been applied

## Next monitoring items
- Confirm 10:30 daily runs continue with data_freshness_status OK.
- Keep Action required at 0 and unknown_source_count at 0.
- Monitor additional live binary outcomes before production threshold changes.
- Review multiclass as reference only until flat bias is reduced offline.
