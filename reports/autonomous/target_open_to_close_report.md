# Research Report: target_open_to_close_validation

- **Executed at**: 2026-10-14T13:43:25.676057
- **Ticker**: 7203.T
- **Feature Set**: Production 163 features (fixed as of previous day's close)
- **Model**: LightGBM Binary Classifier (n_estimators=300, lr=0.03, metric='auc', early_stopping=30)
- **Train Window**: 2010-08-16 to 2021-09-02 (2678 trading days)
- **Validation Window**: 2021-09-03 to 2024-03-11 (615 trading days)
- **Primary Clean Test Window**: 2024-04-01 to 2026-09-17 (604 trading days)
- **Secondary Recent Regime (2026)**: 2026-01-01 to 2026-09-17 (174 trading days)

---

## 1. Executive Summary & Diagnostic Answers

- **Hypothesis**: The Close-to-Close target suffers from overnight gap noise that induces overconfidence and Q5 saturation (drop in win rate at highest predicted probability). Re-targeting the binary label to Open-to-Close (intraday candlestick direction: `close(t+1) > open(t+1)`) will eliminate overnight noise, improve ROC AUC, and restore monotonicity across probability quintiles.
- **Experimental Outcome**: **`HYPOTHESIS_REJECTED`**

### Key Comparison Summary (Clean Test Window: 2024-04-01 to 2026-09-17)
- **ROC AUC**: Baseline Close-to-Close is **0.5636**, while Candidate Open-to-Close is **0.5079** (-0.0557 change).
- **Balanced Accuracy**: Baseline Close-to-Close is **0.5397**, while Candidate Open-to-Close is **0.5107** (-0.0290 change).
- **5-Quantile Lift (Q5 - Q1)**:
  - Baseline C2C: **+0.0909** (1.2245x)
  - Candidate O2C: **-0.0413** (0.9091x)
- **Q5 Overconfidence / Saturation Resolution**:
  - Baseline C2C: Q4 = **57.02%** vs Q5 = **49.59%** (Q5 dropped -7.43% pt, Saturation: True)
  - Candidate O2C: Q4 = **44.63%** vs Q5 = **41.32%** (Q5 dropped -3.31% pt, Saturation: True)

---

## 2. Comprehensive Metrics Comparison Table

### Primary: Clean Test Window (2024-04-01 to 2026-09-17, N=604)

| Metric | Baseline: Close-to-Close (C2C) | Candidate: Open-to-Close (O2C) | Difference (O2C - C2C) |
|---|:---:|:---:|:---:|
| **Market Base Up Rate** | 47.52% | 41.72% (陽線率) | -5.80% pt |
| **Model Mean p_up** | 0.4578 | 0.4746 | - |
| **ROC AUC** | **0.5636** | **0.5079** | **-0.0557** |
| **Average Precision** | 0.5225 | 0.4413 | -0.0812 |
| **Accuracy (at 0.50)** | 54.97% | 57.62% | +2.65% pt |
| **Balanced Accuracy** | 0.5397 | 0.5107 | -0.0290 |
| **Up Precision (at 0.50)** | 54.19% | 46.77% | -7.42% pt |
| **Down Precision (at 0.50)** | 55.29% | 58.86% | +3.57% pt |
| **Top vs Bottom Lift (Q5 - Q1)** | **+0.0909** | **-0.0413** | **-0.1322** |
| **Lift Ratio (Q5 / Q1)** | 1.2245x | 0.9091x | - |
| **Monotonic Inversions** | 1 | 3 | - |
| **Brier Score** | 0.2481 | 0.2463 | - |
| **Log Loss** | 0.6894 | 0.6858 | - |

---

## 3. Open-to-Close 5-Quantile (Q1〜Q5) Detail Tables

### Clean Test Window (2024-04-01 to 2026-09-17, N=604)

| Quantile | Sample Count | p_up Range | Mean p_up | Actual 陽線数 (Up) | Actual 陰線数 (Down) | **実績 陽線率** | キャリブレーションギャップ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 121 | [0.4316, 0.4573] | 0.4494 | 55 | 66 | **45.45%** | -0.0052 |
| **Q2** | 121 | [0.4573, 0.4678] | 0.4630 | 48 | 73 | **39.67%** | +0.0664 |
| **Q3** | 120 | [0.4678, 0.4778] | 0.4734 | 45 | 75 | **37.50%** | +0.0984 |
| **Q4** | 121 | [0.4779, 0.4907] | 0.4841 | 54 | 67 | **44.63%** | +0.0378 |
| **Q5** | 121 | [0.4907, 0.5352] | 0.5031 | 50 | 71 | **41.32%** | +0.0898 |

- **Top vs Bottom Lift (Q5 - Q1)**: **-0.0413** (0.9091x)
- **Monotonicity**: Strictly Monotonic = `False`, Inversions = `3`
- **Quantile Spearman Correlation**: `-0.2`
- **Q5 vs Q4 Saturation**: Q4 Rate = **44.63%**, Q5 Rate = **41.32%** (Diff: **-3.31% pt**, Saturates: `True`)

---

### Secondary: Recent Regime 2026 Alone (2026-01-01 to 2026-09-17, N=174)

| Quantile | Sample Count | p_up Range | Mean p_up | Actual 陽線数 (Up) | Actual 陰線数 (Down) | **実績 陽線率** | キャリブレーションギャップ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 35 | [0.4350, 0.4599] | 0.4526 | 22 | 13 | **62.86%** | -0.1759 |
| **Q2** | 35 | [0.4601, 0.4663] | 0.4628 | 12 | 23 | **34.29%** | +0.1199 |
| **Q3** | 34 | [0.4666, 0.4765] | 0.4721 | 11 | 23 | **32.35%** | +0.1486 |
| **Q4** | 35 | [0.4774, 0.4906] | 0.4826 | 12 | 23 | **34.29%** | +0.1397 |
| **Q5** | 35 | [0.4906, 0.5298] | 0.5059 | 14 | 21 | **40.00%** | +0.1059 |

- **Top vs Bottom Lift (Q5 - Q1)**: **-0.2286** (0.6364x)
- **Monotonicity**: Strictly Monotonic = `False`, Inversions = `2`

---

## 4. Multi-Period Performance Comparison

| Period | Target | ROC AUC | Balanced Acc | Accuracy | Top vs Bottom Lift | Inversions |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Clean Test Window (2024-04-01 to 2026-09-17) | C2C (Baseline) | 0.5636 | 0.5397 | 54.97% | +0.0909 | 1 |
| Clean Test Window (2024-04-01 to 2026-09-17) | O2C (Candidate) | 0.5079 | 0.5107 | 57.62% | -0.0413 | 3 |
| Recent Regime (2026-01-01 to 2026-09-17) | C2C (Baseline) | 0.5396 | 0.5107 | 51.72% | +0.0571 | 2 |
| Recent Regime (2026-01-01 to 2026-09-17) | O2C (Candidate) | 0.4376 | 0.5192 | 58.62% | -0.2286 | 2 |
| Clean 2024 Sub-period (2024-04-01 to 2024-12-31) | C2C (Baseline) | 0.6236 | 0.577 | 59.89% | +0.2368 | 1 |
| Clean 2024 Sub-period (2024-04-01 to 2024-12-31) | O2C (Candidate) | 0.5561 | 0.5343 | 59.36% | +0.2105 | 1 |
| Clean 2025 Sub-period (2025-01-01 to 2025-12-31) | C2C (Baseline) | 0.5375 | 0.538 | 53.50% | +0.0816 | 1 |
| Clean 2025 Sub-period (2025-01-01 to 2025-12-31) | O2C (Candidate) | 0.5212 | 0.4869 | 55.56% | -0.0204 | 3 |
| Contaminated Window Reference (2023-01-01 to 2026-09-17) | C2C (Baseline) | 0.5561 | 0.5378 | 53.80% | +0.1264 | 1 |
| Contaminated Window Reference (2023-01-01 to 2026-09-17) | O2C (Candidate) | 0.5071 | 0.5047 | 55.24% | -0.0055 | 2 |

---

## 5. In-Depth Analysis: Q5 Saturation and Intraday Market Dynamics

1. **Intraday Structural Drift (陽線率 vs C2C 上昇率)**:
   - In Toyota (7203.T), the market base rate of Close-to-Close Up is **47.52%**, but the actual intraday Open-to-Close 陽線率 is only **41.72%**.
   - This reflects a well-known market microstructure characteristic: large-cap Japanese equities frequently experience gap-up opens followed by intraday mean-reversion or selling into the open.
2. **Q5 Overconfidence Saturation Comparison**:
   - In Close-to-Close, Q4 actual Up rate was 57.02%, and Q5 collapsed to 49.59%.
   - In Open-to-Close, Q4 actual 陽線率 is **44.63%**, while Q5 actual 陽線率 is **41.32%**.
   - This diagnostic provides direct evidence on whether removing the overnight gap restores monotonicity in high-probability predictions.
3. **Predictability Comparison**:
   - The ROC AUC comparison across periods demonstrates whether yesterday's close-time features carry higher information content for overnight-to-next-close return or purely intraday open-to-close return.

---

## 6. Strategic Implications & Recommendations

- All findings were produced strictly under the 1 hypothesis / 1 attempt / 0 retuning rule on the clean holdout window without data contamination.
- Production models, databases, and live configurations remain completely unchanged.
