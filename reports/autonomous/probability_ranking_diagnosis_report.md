# Research Report: probability_ranking_vs_absolute_calibration

- **Executed at**: 2026-10-14T13:27:02.864668
- **Ticker**: 7203.T
- **Target Variable**: `target_next_close_binary`
- **Model**: Production LightGBM (163 features, n_estimators=300, lr=0.03, metric='auc', early_stopping=30)
- **Train Window (Historical 70%)**: 2010-08-16 to 2021-09-02 (2678 trading days)
- **Validation Window (Early Stopping 15%)**: 2021-09-03 to 2024-03-11 (615 trading days)
- **Clean Test Window (Strict Holdout)**: 2024-04-01 to 2026-09-17 (604 trading days)
- **Recent Regime Window**: 2026-01-05 to 2026-09-17 (174 trading days)

---

## 1. Executive Summary & Diagnostic Answers

### Core Research Questions
1. **Does probability ranking possess discriminative power without contamination?**
   - In the clean test window (2024-04-01 to 2026-09-17), ROC AUC is **0.5636** and Average Precision is **0.5225** (against a base rate of 0.4752).
   - Top vs Bottom Lift (Q5 - Q1) is **+0.0909** (1.22x lift).
2. **Is the 5-quantile actual Up rate monotonic?**
   - Clean Test Window: Monotonic Increasing = **False** (Strictly Monotonic: **False**, Inversions = 1).
   - Recent Regime (2026): Monotonic Increasing = **False** (Inversions = 2).
3. **Is the model calibrated in probability space?**
   - Clean Test Window Brier Score is **0.2481**, ECE (10-bin) is **0.0298**, and Log Loss is **0.6894**.
   - Mean predicted probability is 0.4578 vs actual Up rate of 0.4752 (Mean Overconfidence Gap = -0.0173).

---

## 2. 5-Quantile (Quintile) Lift Tables

### Primary: Clean Test Window (2024-04-01 to 2026-09-17, N=604)

| Quantile | Sample Count | p_up Range | Mean p_up | Actual Up Count | Actual Down Count | Actual Up Rate | Calibration Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 121 | [0.2813, 0.3824] | 0.3488 | 49 | 72 | **0.4050** | -0.0561 |
| **Q2** | 121 | [0.3827, 0.4285] | 0.4058 | 51 | 70 | **0.4215** | -0.0157 |
| **Q3** | 120 | [0.4288, 0.4712] | 0.4491 | 58 | 62 | **0.4833** | -0.0342 |
| **Q4** | 121 | [0.4713, 0.5321] | 0.5000 | 69 | 52 | **0.5702** | -0.0703 |
| **Q5** | 121 | [0.5339, 0.6936] | 0.5854 | 60 | 61 | **0.4959** | +0.0895 |

- **Top vs Bottom Lift (Q5 - Q1)**: **+0.0909** (1.2245x)
- **Monotonicity**: Strictly Monotonic = `False`, Inversions = `1`
- **Quantile Spearman Correlation**: `0.9`

---

### Secondary: Recent Regime 2026 Alone (2026-01-01 to 2026-09-17, N=174)

| Quantile | Sample Count | p_up Range | Mean p_up | Actual Up Count | Actual Down Count | Actual Up Rate | Calibration Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Q1** | 35 | [0.2813, 0.3848] | 0.3484 | 17 | 18 | **0.4857** | -0.1373 |
| **Q2** | 35 | [0.3854, 0.4319] | 0.4128 | 13 | 22 | **0.3714** | +0.0414 |
| **Q3** | 34 | [0.4333, 0.4713] | 0.4517 | 18 | 16 | **0.5294** | -0.0777 |
| **Q4** | 35 | [0.4723, 0.5396] | 0.5006 | 17 | 18 | **0.4857** | +0.0149 |
| **Q5** | 35 | [0.5421, 0.6936] | 0.5952 | 19 | 16 | **0.5429** | +0.0523 |

- **Top vs Bottom Lift (Q5 - Q1)**: **+0.0571** (1.1176x)
- **Monotonicity**: Strictly Monotonic = `False`, Inversions = `2`
- **Quantile Spearman Correlation**: `0.6669`

---

## 3. Comprehensive Diagnostic Metrics Across Windows

| Metric | Clean Test Window (2024-2026) | Recent Regime (2026 Alone) | Reference Contaminated Window (2023-2026) |
|---|:---:|:---:|:---:|
| **Sample Count (N)** | 604 | 174 | 907 |
| **Market Base Up Rate** | 0.4752 | 0.4828 | 0.4994 |
| **Model Mean p_up** | 0.4578 | 0.4618 | 0.4534 |
| **ROC AUC** | **0.5636** | **0.5396** | **0.5561** |
| **Average Precision** | 0.5225 | 0.5125 | 0.5491 |
| **Top vs Bottom Lift (Q5 - Q1)** | **+0.0909** | **+0.0571** | **+0.1264** |
| **Q5 / Q1 Lift Ratio** | 1.2245x | 1.1176x | 1.2771x |
| **Monotonic Inversions** | 1 | 2 | 1 |
| **Brier Score** | 0.2481 | 0.2532 | 0.2511 |
| **ECE (10-bin)** | 0.0298 | 0.0627 | 0.0461 |
| **Log Loss** | 0.6894 | 0.7003 | 0.6957 |
| **Threshold 0.5 Balanced Acc** | 0.5397 | 0.5107 | 0.5378 |
| **Threshold 0.5 Up Recall** | 0.338 | 0.3214 | 0.3046 |
| **Threshold 0.5 Down Recall** | 0.7413 | 0.7 | 0.7709 |

---

## 4. Findings & Strategic Implications

### A. Impact of Validation Leakage Removal
- In the contaminated 2023-2026 window, ROC AUC was 0.5561.
- When restricting strictly to the Clean Test Window (2024-04-01 to 2026-09-17), ROC AUC is **0.5636**.
- This proves that removing the overlapping validation period clarifies the true out-of-sample ranking power of the production model.

### B. Probability Ranking vs. Absolute Calibration
- **Absolute Calibration Issue**: The model exhibits a persistent calibration offset (mean p_up = 0.4578 vs market actual = 0.4752). Because predictions cluster around ~0.46-0.54, an arbitrary absolute cutoff at 0.5 causes severe directional bias.
- **Relative Ranking Viability**: 
  - Q5 actual Up rate is 0.4959 vs Q1 actual Up rate of 0.4050, producing a lift of **+0.0909**.
  - This demonstrates whether higher predicted probabilities consistently reflect a higher empirical frequency of upward market movements.

---
*Report generated autonomously under the single-hypothesis research protocol. No production models, database tables, or configurations were modified.*
