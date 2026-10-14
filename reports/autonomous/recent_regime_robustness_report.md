# Research Report: recent_regime_robustness

- **Executed at**: 2026-09-17T23:28:49.636094
- **Ticker**: 7203.T
- **Common Test Window**: 2023-01-01 to 2026-09-17 (906 trading days)
- **Condition A (Fixed 70%) Fit Window**: 2010-01-04 to 2021-09-02
- **Model Hyperparameters**: Production-identical LightGBM (`n_estimators=300, lr=0.03, metric='auc', early_stopping=30`)

## 1. Overall Performance Comparison (2023-2026)

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy | vs Prev-Day BA |
|---|---|---|---|---|---|---|---|
| **A.production_current_fixed_70pct** | 0.5375 | 0.556 | 0.3046 | 0.7704 | 0.2671 | 0.5375 | +0.043 |
| **B.expanding_walk_forward** | 0.5143 | 0.518 | 0.2406 | 0.7881 | 0.2263 | 0.5143 | +0.0198 |
| **C.rolling_recent_3y** | 0.4945 | 0.497 | 0.3377 | 0.6512 | 0.3433 | 0.4945 | 0.0 |
| **D.rolling_recent_2y** | 0.5 | 0.5003 | 0.3466 | 0.6534 | 0.3466 | 0.5 | +0.0055 |
| **E.rolling_recent_1y** | 0.4945 | 0.5049 | 0.404 | 0.585 | 0.4095 | 0.4945 | 0.0 |
| *(Baseline) Previous-Day* | 0.4945 | - | 0.4923 | 0.4967 | 0.4978 | 0.4945 | 0.0000 |

## 2. Annual Sub-period Breakdown

### Sub-period: 2023

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| A.production_current_fixed_70pct | 0.5661 | 0.5612 | 0.2946 | 0.8376 | 0.2317 | 0.5528 |
| B.expanding_walk_forward | 0.5006 | 0.5008 | 0.155 | 0.8462 | 0.1545 | 0.4837 |
| C.rolling_recent_3y | 0.5219 | 0.5137 | 0.2403 | 0.8034 | 0.2195 | 0.5081 |
| D.rolling_recent_2y | 0.5046 | 0.4869 | 0.4109 | 0.5983 | 0.4065 | 0.5 |
| E.rolling_recent_1y | 0.5408 | 0.5825 | 0.2868 | 0.7949 | 0.248 | 0.5285 |
| *(Baseline) Previous-Day* | 0.5153 | - | 0.5349 | 0.4957 | 0.5203 | 0.5163 |

### Sub-period: 2024

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| A.production_current_fixed_70pct | 0.5321 | 0.5818 | 0.3162 | 0.748 | 0.2828 | 0.541 |
| B.expanding_walk_forward | 0.5134 | 0.5342 | 0.2393 | 0.7874 | 0.2254 | 0.5246 |
| C.rolling_recent_3y | 0.4556 | 0.4807 | 0.5726 | 0.3386 | 0.6189 | 0.4508 |
| D.rolling_recent_2y | 0.4979 | 0.502 | 0.547 | 0.4488 | 0.5492 | 0.4959 |
| E.rolling_recent_1y | 0.4711 | 0.4528 | 0.6667 | 0.2756 | 0.6967 | 0.4631 |
| *(Baseline) Previous-Day* | 0.4745 | - | 0.453 | 0.4961 | 0.4795 | 0.4754 |

### Sub-period: 2025

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| A.production_current_fixed_70pct | 0.538 | 0.5375 | 0.2927 | 0.7833 | 0.2551 | 0.535 |
| B.expanding_walk_forward | 0.5217 | 0.5496 | 0.2602 | 0.7833 | 0.2387 | 0.5185 |
| C.rolling_recent_3y | 0.5216 | 0.5428 | 0.2683 | 0.775 | 0.2469 | 0.5185 |
| D.rolling_recent_2y | 0.5273 | 0.5541 | 0.1463 | 0.9083 | 0.1193 | 0.5226 |
| E.rolling_recent_1y | 0.4945 | 0.5287 | 0.1057 | 0.8833 | 0.1111 | 0.4897 |
| *(Baseline) Previous-Day* | 0.4609 | - | 0.4634 | 0.4583 | 0.5021 | 0.4609 |

### Sub-period: 2026 (Recent Regime)

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| A.production_current_fixed_70pct | 0.509 | 0.5384 | 0.3214 | 0.6966 | 0.3121 | 0.5145 |
| B.expanding_walk_forward | 0.5322 | 0.4891 | 0.3452 | 0.7191 | 0.3121 | 0.5376 |
| C.rolling_recent_3y | 0.4961 | 0.5029 | 0.2619 | 0.7303 | 0.2659 | 0.5029 |
| D.rolling_recent_2y | 0.468 | 0.501 | 0.2619 | 0.6742 | 0.2948 | 0.474 |
| E.rolling_recent_1y | 0.5015 | 0.5017 | 0.6548 | 0.3483 | 0.6532 | 0.4971 |
| *(Baseline) Previous-Day* | 0.5372 | - | 0.5238 | 0.5506 | 0.4855 | 0.5376 |

## 3. Findings and Conclusion

### Training Staleness vs. Recent Regime Performance
- **Condition A (Fixed 70%) in 2026**: BA = 0.509, AUC = 0.5384, Up Recall = 0.3214, Down Recall = 0.6966.
- **Condition B (Expanding) in 2026**: BA = 0.5322, AUC = 0.4891, Up Recall = 0.3452, Down Recall = 0.7191.
- **Condition C (Rolling 3y) in 2026**: BA = 0.4961, AUC = 0.5029, Up Recall = 0.2619, Down Recall = 0.7303.
- **Condition D (Rolling 2y) in 2026**: BA = 0.468, AUC = 0.501, Up Recall = 0.2619, Down Recall = 0.6742.
- **Condition E (Rolling 1y) in 2026**: BA = 0.5015, AUC = 0.5017, Up Recall = 0.6548, Down Recall = 0.3483.

### Key Insights
1. **Recency Impact**: Comparing fixed 70% against expanding and rolling windows demonstrates whether updating the fit boundary resolves the extreme down-bias observed in recent live predictions.
2. **Optimal Window Length**: Evaluation across rolling windows (3y, 2y, 1y) shows whether shorter regimes or larger historical sample sizes provide superior out-of-sample stability.

---
*Report generated autonomously under the single-hypothesis research protocol. No production models, database tables, or configurations were modified.*