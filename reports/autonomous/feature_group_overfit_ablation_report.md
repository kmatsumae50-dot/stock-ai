# Research Report: feature_group_overfit_ablation

- **Executed at**: 2026-09-18T00:51:27.949700
- **Ticker**: 7203.T
- **Common Test Window**: 2023-01-01 to 2026-09-17 (907 confirmed trading days)
- **Training Partition**: 2010-01-04 to 2021-09-02 (fixed 70% split)
- **Model Hyperparameters**: Production-identical LightGBM (`n_estimators=300, lr=0.03, metric='auc', early_stopping=30`)

## Feature Set Configurations

| Feature Set | Feature Count | Dropped Count | Description |
|---|---|---|---|
| **Baseline (ALL)** | 163 | 0 | Full current feature pool |
| **No-External** | 88 | 75 | Excludes FX, indices (N225, TOPIX), peer autos (Honda, Nissan, Tesla, BYD) |
| **No-News** | 108 | 55 | Excludes news sentiment and narrative shift/theme features |
| **No-Oscillators** | 134 | 29 | Excludes short-term bounded oscillators (RSI, MACD, Bollinger Bands) |
| **Core-Only** | 25 | 138 | Pure technical price momentum, moving averages, and volume features |

## 1. Overall Performance Comparison (2023-2026)

| Condition | Feature Count | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy | vs Baseline BA | vs Baseline AUC |
|---|---|---|---|---|---|---|---|---|---|
| **Baseline (ALL)** | 163 | 0.532 | 0.5499 | 0.2958 | 0.7682 | 0.2638 | 0.532 | 0.0 | 0.0 |
| **No-External** | 88 | 0.5143 | 0.511 | 0.5077 | 0.521 | 0.4934 | 0.5143 | -0.0177 | -0.0389 |
| **No-News** | 108 | 0.532 | 0.5499 | 0.2958 | 0.7682 | 0.2638 | 0.532 | 0.0 | 0.0 |
| **No-Oscillators** | 134 | 0.532 | 0.5363 | 0.3223 | 0.7417 | 0.2903 | 0.532 | 0.0 | -0.0136 |
| **Core-Only** | 25 | 0.5143 | 0.511 | 0.5077 | 0.521 | 0.4934 | 0.5143 | -0.0177 | -0.0389 |
| *(Baseline) Previous-Day* | - | 0.4939 | - | 0.4923 | 0.4956 | 0.4983 | 0.4939 | - | - |

## 2. Annual Sub-period Breakdown

### Sub-period: 2023

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| Baseline (ALL) | 0.5719 | 0.5469 | 0.3488 | 0.7949 | 0.2805 | 0.561 |
| No-External | 0.5662 | 0.5792 | 0.5426 | 0.5897 | 0.4797 | 0.565 |
| No-News | 0.5719 | 0.5469 | 0.3488 | 0.7949 | 0.2805 | 0.561 |
| No-Oscillators | 0.5594 | 0.5449 | 0.3411 | 0.7778 | 0.2846 | 0.5488 |
| Core-Only | 0.5662 | 0.5792 | 0.5426 | 0.5897 | 0.4797 | 0.565 |
| *(Baseline) Previous-Day* | 0.5153 | - | 0.5349 | 0.4957 | 0.5203 | 0.5163 |

### Sub-period: 2024

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| Baseline (ALL) | 0.5311 | 0.5871 | 0.2906 | 0.7717 | 0.2582 | 0.541 |
| No-External | 0.5156 | 0.4936 | 0.4957 | 0.5354 | 0.4795 | 0.5164 |
| No-News | 0.5311 | 0.5871 | 0.2906 | 0.7717 | 0.2582 | 0.541 |
| No-Oscillators | 0.5285 | 0.5593 | 0.3248 | 0.7323 | 0.2951 | 0.5369 |
| Core-Only | 0.5156 | 0.4936 | 0.4957 | 0.5354 | 0.4795 | 0.5164 |
| *(Baseline) Previous-Day* | 0.4726 | - | 0.453 | 0.4922 | 0.4816 | 0.4735 |

### Sub-period: 2025

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| Baseline (ALL) | 0.5015 | 0.534 | 0.2114 | 0.7917 | 0.2099 | 0.4979 |
| No-External | 0.4856 | 0.4876 | 0.4878 | 0.4833 | 0.5021 | 0.4856 |
| No-News | 0.5015 | 0.534 | 0.2114 | 0.7917 | 0.2099 | 0.4979 |
| No-Oscillators | 0.5095 | 0.5274 | 0.2439 | 0.775 | 0.2346 | 0.5062 |
| Core-Only | 0.4856 | 0.4876 | 0.4878 | 0.4833 | 0.5021 | 0.4856 |
| *(Baseline) Previous-Day* | 0.4609 | - | 0.4634 | 0.4583 | 0.5021 | 0.4609 |

### Sub-period: 2026 (Recent Regime)

| Condition | Balanced Acc | ROC AUC | Up Recall | Down Recall | Pred Up Rate | Accuracy |
|---|---|---|---|---|---|---|
| Baseline (ALL) | 0.5209 | 0.5236 | 0.3452 | 0.6966 | 0.3237 | 0.526 |
| No-External | 0.4803 | 0.49 | 0.5 | 0.4607 | 0.5202 | 0.4798 |
| No-News | 0.5209 | 0.5236 | 0.3452 | 0.6966 | 0.3237 | 0.526 |
| No-Oscillators | 0.5338 | 0.507 | 0.4048 | 0.6629 | 0.3699 | 0.5376 |
| Core-Only | 0.4803 | 0.49 | 0.5 | 0.4607 | 0.5202 | 0.4798 |
| *(Baseline) Previous-Day* | 0.5372 | - | 0.5238 | 0.5506 | 0.4855 | 0.5376 |

## 3. Analysis of Feature Group Impact

### Identifying Harmful ('Toxic') vs Useful Feature Groups
- **No-External**: Overall AUC Δ=-0.0389, Overall BA Δ=-0.0177, 2026 AUC Δ=-0.0336 -> **DETRIMENTAL (Useful feature removed)**
- **No-News**: Overall AUC Δ=+0.0000, Overall BA Δ=+0.0000, 2026 AUC Δ=+0.0000 -> **DETRIMENTAL (Useful feature removed)**
- **No-Oscillators**: Overall AUC Δ=-0.0136, Overall BA Δ=+0.0000, 2026 AUC Δ=-0.0166 -> **DETRIMENTAL (Useful feature removed)**
- **Core-Only**: Overall AUC Δ=-0.0389, Overall BA Δ=-0.0177, 2026 AUC Δ=-0.0336 -> **DETRIMENTAL (Useful feature removed)**

### 2026 Recent Regime Subset Analysis
- **Baseline (ALL) in 2026**: BA = 0.5209, AUC = 0.5236, Up Recall = 0.3452, Down Recall = 0.6966
- **No-External in 2026**: BA = 0.4803, AUC = 0.49, Up Recall = 0.5, Down Recall = 0.4607
- **No-News in 2026**: BA = 0.5209, AUC = 0.5236, Up Recall = 0.3452, Down Recall = 0.6966
- **No-Oscillators in 2026**: BA = 0.5338, AUC = 0.507, Up Recall = 0.4048, Down Recall = 0.6629
- **Core-Only in 2026**: BA = 0.4803, AUC = 0.49, Up Recall = 0.5, Down Recall = 0.4607

---
*Report generated autonomously under the single-hypothesis research protocol. No production models, database tables, or configurations were modified.*