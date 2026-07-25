# Binary Up Detection Experiments

This is an offline historical walk-forward report. It does not change production models, daily_predict, SQLite prediction history, or feature generation.

## Conclusions
- mild_up_weight_promising
- balanced_threshold_promising
- down_recall_too_damaged
- confidence_unreliable
- no_candidate_ready_for_production
- production_change_not_ready
- next_experiment_needed

## Historical all summary
| candidate_name | sample_count | accuracy | balanced_accuracy | up_recall | down_recall | up_precision | constraint_passed | caveats |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_model | 3276 | 0.5098 | 0.5073 | 0.3505 | 0.6641 | 0.5027 | False |  |
| class_weight_balanced | 3276 | 0.5208 | 0.5200 | 0.4721 | 0.5679 | 0.5142 | True |  |
| class_weight_up_boost_mild | 3276 | 0.5079 | 0.5109 | 0.6973 | 0.3245 | 0.5000 | False | down recall below 45% guard |
| class_weight_up_boost_medium | 3276 | 0.5067 | 0.5116 | 0.8189 | 0.2043 | 0.4992 | False | down recall below 45% guard |
| scale_pos_weight_mild | 3276 | 0.5092 | 0.5074 | 0.3958 | 0.6190 | 0.5016 | True |  |
| scale_pos_weight_medium | 3276 | 0.5125 | 0.5114 | 0.4411 | 0.5817 | 0.5053 | True |  |
| balanced_accuracy_optimized_threshold | 3276 | 0.5092 | 0.5072 | 0.3834 | 0.6310 | 0.5016 | False |  |
| up_recall_mild_threshold | 3276 | 0.5076 | 0.5126 | 0.8282 | 0.1971 | 0.4998 | False | down recall below 45% guard |
| up_recall_medium_threshold | 3276 | 0.5003 | 0.5057 | 0.8455 | 0.1659 | 0.4955 | False | down recall below 45% guard |
| precision_guarded_up_threshold | 3276 | 0.5095 | 0.5123 | 0.6880 | 0.3365 | 0.5011 | False | down recall below 45% guard |
| previous_day_direction_hybrid | 3276 | 0.4948 | 0.4942 | 0.4560 | 0.5325 | 0.4858 | False |  |
| moving_average_trend_hybrid | 3276 | 0.4811 | 0.4811 | 0.4857 | 0.4766 | 0.4734 | False |  |
| model_ma_agreement_hybrid | 3276 | 0.4905 | 0.4849 | 0.1278 | 0.8419 | 0.4392 | False |  |
| regime_aware_threshold | 3276 | 0.5140 | 0.5118 | 0.3691 | 0.6544 | 0.5085 | True |  |
| bear_reversal_candidate_threshold | 3276 | 0.5125 | 0.5110 | 0.4132 | 0.6088 | 0.5057 | True |  |
| external_features_excluded | 3276 | 0.4893 | 0.4864 | 0.3021 | 0.6707 | 0.4705 | False |  |
| technical_only | 3276 | 0.4893 | 0.4864 | 0.3021 | 0.6707 | 0.4705 | False |  |
| market_external_only | 3322 | 0.5144 | 0.5116 | 0.3435 | 0.6797 | 0.5091 | False |  |
| news_excluded | 3276 | 0.5098 | 0.5073 | 0.3505 | 0.6641 | 0.5027 | False |  |

## Live failure pattern summary
| failure_type | sample_count | average_confidence | average_actual_return | regime_top | anomaly_level_top | previous_day_up_rate | ma_trend_up_rate | top_features_sample |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| actual_down_predicted_down | 8 | 0.5166 | -0.0106 | High Volatility | High | 0.0000 | 0.0000 | tesla_return_1d, honda_return_1d, volume_change_1d |
| actual_up_predicted_down | 13 | 0.5282 | 0.0154 | Bear Trend | High | 0.0000 | 0.0000 | close_vs_ma_5, usdjpy_return_1d, return_1d |
| predicted_up_actual_down | 7 | 0.5342 | -0.0174 | High Volatility | High | 0.0000 | 0.0000 | topix_ma_5, tesla_return_1d, byd_macd_hist |

## Caveat
Candidates that look promising here still require a separate production-change review. Thresholds and weights are selected from train/validation rows only, then applied to test rows.
