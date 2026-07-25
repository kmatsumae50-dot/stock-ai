# Binary Shadow Evaluation

This report tracks shadow binary candidates outside production. It does not write to SQLite and does not change daily_predict or production model artifacts.

## Conclusions
- shadow_sample_too_small
- production_change_not_ready
- continue_shadow_monitoring

## Candidate metrics
| candidate_name | sample_count | confirmed_count | pending_count | accuracy | balanced_accuracy | up_recall | down_recall | up_precision | down_precision | predicted_up_count | predicted_down_count | confidence_ge_55_count | confidence_ge_55_hit_count | confidence_ge_55_miss_count | confidence_ge_55_hit_rate | latest_feature_date | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| class_weight_balanced | 1 | 0 | 1 |  |  |  |  |  |  | 0 | 1 | 0 | 0 | 0 |  | 2026-07-08 | low confirmed sample count |
| production_current | 1 | 0 | 1 |  |  |  |  |  |  | 0 | 1 | 0 | 0 | 0 |  | 2026-07-08 | low confirmed sample count |

## Latest shadow predictions
| run_timestamp | ticker | feature_date | target_close_date | candidate_name | predicted_direction | probability_up | probability_down | confidence | source | model_note | actual_return | actual_direction | result | evaluated_at | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-08T16:48:55 | 7203.T | 2026-07-08 | 2026-07-09 | production_current | down | 0.4528 | 0.5472 | 0.5472 | shadow | Current production binary model loaded from artifact; not saved to SQLite. |  |  | pending |  | shadow only; not saved to SQLite; production_change_not_ready |
| 2026-07-08T16:48:55 | 7203.T | 2026-07-08 | 2026-07-09 | class_weight_balanced | down | 0.4992 | 0.5008 | 0.5008 | shadow | Temporary production-style binary model with class_weight='balanced'; not saved as production. |  |  | pending |  | shadow only; not saved to SQLite; production_change_not_ready |

## Pending predictions
| run_timestamp | ticker | feature_date | target_close_date | candidate_name | predicted_direction | probability_up | probability_down | confidence | source | model_note | actual_return | actual_direction | result | evaluated_at | caveat |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-08T16:48:55 | 7203.T | 2026-07-08 | 2026-07-09 | production_current | down | 0.4528 | 0.5472 | 0.5472 | shadow | Current production binary model loaded from artifact; not saved to SQLite. |  |  | pending |  | shadow only; not saved to SQLite; production_change_not_ready |
| 2026-07-08T16:48:55 | 7203.T | 2026-07-08 | 2026-07-09 | class_weight_balanced | down | 0.4992 | 0.5008 | 0.5008 | shadow | Temporary production-style binary model with class_weight='balanced'; not saved as production. |  |  | pending |  | shadow only; not saved to SQLite; production_change_not_ready |

## Confirmed predictions
No confirmed rows.

## Production live comparison
| feature_date | candidate_name | shadow_direction | shadow_confidence | production_live_direction | production_live_confidence | direction_matches_live | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-07-08 | production_current | down | 0.5472 | down | 0.5472 | True | pending |
| 2026-07-08 | class_weight_balanced | down | 0.5008 | down | 0.5472 | True | pending |

## Caveats
- Shadow rows are research records only.
- production_current is included for comparison, but no SQLite live rows are created here.
- class_weight_balanced is not a production model change.
- Keep production_change_not_ready until enough confirmed shadow rows accumulate.
