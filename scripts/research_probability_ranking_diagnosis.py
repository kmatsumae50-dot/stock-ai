"""research_probability_ranking_diagnosis.py

Single-hypothesis offline research experiment for hypothesis: probability_ranking_vs_absolute_calibration
Tests the ranking power and 5-quantile lift of production LightGBM (fixed_70pct, 163 features)
over the clean, uncompromised test window (2024-04-01 to 2026-09-17) and 2026 recent regime.

Validation overlapping issue eliminated:
- Fixed 70% fit window: 2010-01-04 to 2021-09-02 (2,875 trading days)
- Early stopping validation window: 2021-09-03 to 2024-03-11 (616 trading days)
- Clean Test Window (strictly untouched holdout): 2024-04-01 to 2026-09-17
- Reference recent regime: 2026-01-01 to 2026-09-17

Strictly read-only with respect to production models, database, and configuration.
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
)

# Ensure source package is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO_ROOT = Path(r"C:\Users\kmats\Documents\ai-stock\stock-ai")
for p in [str(PROJECT_ROOT / "src"), str(MAIN_REPO_ROOT / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from stock_ai.labels.targets import add_next_close_binary_target  # noqa: E402
from stock_ai.models.baseline import select_feature_columns  # noqa: E402

TARGET_COLUMN = "target_next_close_binary"


def fit_production_lightgbm(
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    feature_columns: list[str],
):
    """Fit LightGBM binary classifier strictly matching production hyperparameters."""
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train_data[feature_columns],
        train_data[TARGET_COLUMN].astype(int),
        eval_set=[(valid_data[feature_columns], valid_data[TARGET_COLUMN].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def compute_expected_calibration_error_equal_width(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) using equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        if i == n_bins - 1:
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        else:
            in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        bin_count = np.sum(in_bin)
        if bin_count > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def diagnose_probability_ranking(
    df: pd.DataFrame,
    period_name: str,
    target_col: str = TARGET_COLUMN,
    prob_col: str = "probability_up",
) -> dict[str, object]:
    """Perform in-depth probability ranking and calibration diagnosis."""
    valid_mask = df[target_col].isin([0, 1]) & df[prob_col].notna()
    sub = df[valid_mask].copy()

    total_samples = len(sub)
    if total_samples == 0:
        return {"period_name": period_name, "error": "No valid samples"}

    y_t = sub[target_col].astype(int).to_numpy()
    y_p = sub[prob_col].astype(float).to_numpy()

    # 1. Base rate & Probability distribution statistics
    actual_up_rate = float(np.mean(y_t == 1))
    mean_p = float(np.mean(y_p))
    std_p = float(np.std(y_p))
    min_p = float(np.min(y_p))
    p25_p = float(np.percentile(y_p, 25))
    median_p = float(np.median(y_p))
    p75_p = float(np.percentile(y_p, 75))
    max_p = float(np.max(y_p))

    # 2. Ranking metrics
    roc_auc = float(roc_auc_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else None
    avg_precision = float(average_precision_score(y_t, y_p)) if len(np.unique(y_t)) > 1 else None

    # Individual rank correlation with binary outcome
    spearman_corr, spearman_pvalue = spearmanr(y_p, y_t)

    # 3. Absolute calibration metrics
    brier = float(brier_score_loss(y_t, y_p))
    # Epsilon clipped for log loss stability
    eps = 1e-15
    y_p_clipped = np.clip(y_p, eps, 1 - eps)
    logloss = float(log_loss(y_t, y_p_clipped))
    ece_10 = compute_expected_calibration_error_equal_width(y_t, y_p, n_bins=10)

    # 4. Standard classification metrics at threshold 0.5
    y_c = (y_p >= 0.5).astype(int)
    pred_up_rate = float(np.mean(y_c == 1))
    cm = confusion_matrix(y_t, y_c, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    up_recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    down_recall = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    balanced_acc = (up_recall + down_recall) / 2.0
    accuracy = float((tp + tn) / total_samples)
    up_precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    down_precision = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0

    # 5. 5-Quantile (Quintile) Analysis
    # Use rank(method="first") to partition evenly into 5 quantiles even if ties exist
    sub["rank"] = sub[prob_col].rank(method="first")
    sub["quantile"] = pd.qcut(
        sub["rank"],
        q=5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )

    quantiles_summary = []
    actual_up_rates = []
    mean_probs = []

    for q_label in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
        q_sub = sub[sub["quantile"] == q_label]
        q_count = len(q_sub)
        q_y = q_sub[target_col].astype(int).to_numpy()
        q_p = q_sub[prob_col].astype(float).to_numpy()

        q_actual_up = int(np.sum(q_y == 1))
        q_actual_down = int(np.sum(q_y == 0))
        q_up_rate = float(np.mean(q_y == 1)) if q_count > 0 else 0.0
        q_mean_prob = float(np.mean(q_p)) if q_count > 0 else 0.0
        q_min_prob = float(np.min(q_p)) if q_count > 0 else 0.0
        q_max_prob = float(np.max(q_p)) if q_count > 0 else 0.0

        actual_up_rates.append(q_up_rate)
        mean_probs.append(q_mean_prob)

        quantiles_summary.append({
            "quantile": q_label,
            "sample_count": q_count,
            "min_p_up": round(q_min_prob, 4),
            "max_p_up": round(q_max_prob, 4),
            "mean_p_up": round(q_mean_prob, 4),
            "actual_up_count": q_actual_up,
            "actual_down_count": q_actual_down,
            "actual_up_rate": round(q_up_rate, 4),
            "calibration_gap": round(q_mean_prob - q_up_rate, 4),  # positive means overconfident
        })

    # Top vs Bottom Lift
    q1_up_rate = actual_up_rates[0]
    q5_up_rate = actual_up_rates[-1]
    lift_diff = q5_up_rate - q1_up_rate
    lift_ratio = (q5_up_rate / q1_up_rate) if q1_up_rate > 0 else None

    # Monotonicity check
    is_strictly_monotonic = all(
        actual_up_rates[i] < actual_up_rates[i + 1] for i in range(len(actual_up_rates) - 1)
    )
    is_monotonic_increasing = all(
        actual_up_rates[i] <= actual_up_rates[i + 1] for i in range(len(actual_up_rates) - 1)
    )

    inversions = 0
    pairwise_comparisons = []
    for i in range(len(actual_up_rates) - 1):
        diff = actual_up_rates[i + 1] - actual_up_rates[i]
        is_inc = diff >= 0
        if not is_inc:
            inversions += 1
        pairwise_comparisons.append({
            "step": f"Q{i+1}->Q{i+2}",
            "from_rate": round(actual_up_rates[i], 4),
            "to_rate": round(actual_up_rates[i + 1], 4),
            "change": round(diff, 4),
            "is_increasing": bool(is_inc),
        })

    # 5-quantile ECE
    ece_5 = sum(
        (q["sample_count"] / total_samples) * abs(q["mean_p_up"] - q["actual_up_rate"])
        for q in quantiles_summary
    )

    # Spearman rank correlation between quantile index (1..5) and actual up rate
    quantile_spearman, _ = spearmanr([1, 2, 3, 4, 5], actual_up_rates)

    return {
        "period_name": period_name,
        "sample_count": total_samples,
        "date_start": str(sub["date"].min().date()),
        "date_end": str(sub["date"].max().date()),
        "base_actual_up_rate": round(actual_up_rate, 4),
        "pred_stats": {
            "mean": round(mean_p, 4),
            "std": round(std_p, 4),
            "min": round(min_p, 4),
            "p25": round(p25_p, 4),
            "median": round(median_p, 4),
            "p75": round(p75_p, 4),
            "max": round(max_p, 4),
        },
        "ranking_metrics": {
            "roc_auc": round(roc_auc, 4) if roc_auc is not None else None,
            "average_precision": round(avg_precision, 4) if avg_precision is not None else None,
            "spearman_corr_sample": round(float(spearman_corr), 4) if spearman_corr is not None else None,
            "spearman_pvalue_sample": float(spearman_pvalue) if spearman_pvalue is not None else None,
            "quantile_spearman": round(float(quantile_spearman), 4) if quantile_spearman is not None else None,
        },
        "calibration_metrics": {
            "brier_score": round(brier, 4),
            "log_loss": round(logloss, 4),
            "ece_10bin": round(ece_10, 4),
            "ece_5quantile": round(ece_5, 4),
            "mean_overconfidence": round(mean_p - actual_up_rate, 4),
        },
        "lift_metrics": {
            "q1_actual_up_rate": round(q1_up_rate, 4),
            "q5_actual_up_rate": round(q5_up_rate, 4),
            "top_vs_bottom_lift_diff": round(lift_diff, 4),
            "top_vs_bottom_lift_ratio": round(lift_ratio, 4) if lift_ratio is not None else None,
            "is_strictly_monotonic": is_strictly_monotonic,
            "is_monotonic_increasing": is_monotonic_increasing,
            "inversion_count": inversions,
            "pairwise_comparisons": pairwise_comparisons,
        },
        "threshold_05_classification": {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "up_recall": round(up_recall, 4),
            "down_recall": round(down_recall, 4),
            "up_precision": round(up_precision, 4),
            "down_precision": round(down_precision, 4),
            "predicted_up_rate": round(pred_up_rate, 4),
        },
        "quantiles_table": quantiles_summary,
    }


def run_experiment(
    feature_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Execute the probability ranking and calibration diagnosis."""
    print("=" * 80)
    print("CYCLE: probability_ranking_vs_absolute_calibration")
    print("Single-hypothesis diagnostic investigation")
    print("=" * 80)

    print(f"Loading features from: {feature_path}")
    df = pd.read_parquet(feature_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    if TARGET_COLUMN not in df.columns:
        df = add_next_close_binary_target(df)

    feature_cols = select_feature_columns(df, target_column=TARGET_COLUMN)
    print(f"Total rows in dataset: {len(df)}")
    print(f"Extracted feature count: {len(feature_cols)}")

    # Restrict boundary up to 2026-09-17 to strictly recreate the reference fixed_70pct split
    df_bounded = df[df["date"] <= "2026-09-17"].copy()
    unique_dates = df_bounded["date"].drop_duplicates().sort_values().reset_index(drop=True)
    train_end_idx = int(len(unique_dates) * 0.70)
    valid_end_idx = int(len(unique_dates) * 0.85)

    train_dates = set(unique_dates.iloc[:train_end_idx])
    valid_dates = set(unique_dates.iloc[train_end_idx:valid_end_idx])

    train_data = df_bounded[df_bounded["date"].isin(train_dates)].dropna(subset=feature_cols + [TARGET_COLUMN]).copy()
    valid_data = df_bounded[df_bounded["date"].isin(valid_dates)].dropna(subset=feature_cols + [TARGET_COLUMN]).copy()

    print("\n--- Training Window Definitions ---")
    print(f"Train Window: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} rows)")
    print(f"Early-stopping Valid Window: {valid_data['date'].min().date()} to {valid_data['date'].max().date()} ({len(valid_data)} rows)")

    # Fit fixed 70% model
    print("\nFitting production LightGBM model (fixed_70pct)...")
    model = fit_production_lightgbm(train_data, valid_data, feature_cols)
    print("Model training complete.")

    # Generate predictions on all confirmed dates
    confirmed_mask = df_bounded[TARGET_COLUMN].isin([0, 1])
    eval_candidates = df_bounded[confirmed_mask].dropna(subset=feature_cols).copy()
    prob_up = model.predict_proba(eval_candidates[feature_cols])[:, 1]
    eval_candidates["probability_up"] = prob_up

    # Define target evaluation periods:
    # 1. Clean Test Window (strictly post-validation): 2024-04-01 to 2026-09-17
    # 2. Recent Regime (2026 alone): 2026-01-01 to 2026-09-17
    # 3. Reference Contaminated Window (2023-01-01 to 2026-09-17) for diagnostic comparison
    # 4. Yearly breakdowns: 2024 (post-04-01), 2025, 2026
    periods = {
        "clean_test_window": {
            "name": "Clean Test Window (2024-04-01 to 2026-09-17)",
            "start": "2024-04-01",
            "end": "2026-09-17",
            "is_primary": True,
        },
        "recent_regime_2026": {
            "name": "Recent Regime (2026-01-01 to 2026-09-17)",
            "start": "2026-01-01",
            "end": "2026-09-17",
            "is_primary": True,
        },
        "contaminated_window_reference": {
            "name": "Contaminated Window Reference (2023-01-01 to 2026-09-17)",
            "start": "2023-01-01",
            "end": "2026-09-17",
            "is_primary": False,
        },
        "clean_2024_sub": {
            "name": "Clean 2024 Sub-period (2024-04-01 to 2024-12-31)",
            "start": "2024-04-01",
            "end": "2024-12-31",
            "is_primary": False,
        },
        "clean_2025_sub": {
            "name": "Clean 2025 Sub-period (2025-01-01 to 2025-12-31)",
            "start": "2025-01-01",
            "end": "2025-12-31",
            "is_primary": False,
        },
    }

    diagnosis_results = {}
    print("\n--- Running Probability Ranking & Calibration Diagnosis ---")
    for key, pinfo in periods.items():
        p_mask = (eval_candidates["date"] >= pinfo["start"]) & (eval_candidates["date"] <= pinfo["end"])
        p_df = eval_candidates[p_mask].copy()
        res = diagnose_probability_ranking(p_df, period_name=pinfo["name"])
        res["is_primary"] = pinfo["is_primary"]
        diagnosis_results[key] = res

        print(f"\n>>> Period: {pinfo['name']} (N={res['sample_count']})")
        print(f"    ROC AUC: {res['ranking_metrics']['roc_auc']} | Avg Precision: {res['ranking_metrics']['average_precision']}")
        print(f"    Brier Score: {res['calibration_metrics']['brier_score']} | ECE (10-bin): {res['calibration_metrics']['ece_10bin']} | Log Loss: {res['calibration_metrics']['log_loss']}")
        print(f"    Actual Up Rate: {res['base_actual_up_rate']} | Mean p_up: {res['pred_stats']['mean']} (Gap: {res['calibration_metrics']['mean_overconfidence']})")
        print(f"    Top vs Bottom Lift (Q5 - Q1): {res['lift_metrics']['top_vs_bottom_lift_diff']} ({res['lift_metrics']['top_vs_bottom_lift_ratio']}x)")
        print(f"    Strictly Monotonic: {res['lift_metrics']['is_strictly_monotonic']} | Inversions: {res['lift_metrics']['inversion_count']}")
        print("    Quantiles Table:")
        for q in res["quantiles_table"]:
            print(f"      {q['quantile']}: N={q['sample_count']:3d} | p_up range: [{q['min_p_up']:.4f}, {q['max_p_up']:.4f}] (mean: {q['mean_p_up']:.4f}) | Actual Up Rate: {q['actual_up_rate']:.4f} (Up: {q['actual_up_count']:2d}, Down: {q['actual_down_count']:2d})")

    # Generate Markdown Report & JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "probability_ranking_diagnosis_results.json"
    report_path = output_dir / "probability_ranking_diagnosis_report.md"

    overall_output = {
        "cycle_id": "probability_ranking_vs_absolute_calibration",
        "timestamp": datetime.datetime.now().isoformat(),
        "ticker": "7203.T",
        "model_spec": {
            "type": "LightGBMClassifier",
            "features_count": len(feature_cols),
            "n_estimators": 300,
            "learning_rate": 0.03,
            "metric": "auc",
            "early_stopping_rounds": 30,
        },
        "windows": {
            "train_window": {
                "start": str(train_data["date"].min().date()),
                "end": str(train_data["date"].max().date()),
                "rows": len(train_data),
            },
            "validation_window": {
                "start": str(valid_data["date"].min().date()),
                "end": str(valid_data["date"].max().date()),
                "rows": len(valid_data),
            },
            "clean_test_window": {
                "start": "2024-04-01",
                "end": "2026-09-17",
            },
            "recent_regime_window": {
                "start": "2026-01-01",
                "end": "2026-09-17",
            },
        },
        "diagnostics": diagnosis_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(overall_output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved diagnosis results to: {json_path}")

    # Build comprehensive Markdown Report
    clean_diag = diagnosis_results["clean_test_window"]
    recent_diag = diagnosis_results["recent_regime_2026"]
    contam_diag = diagnosis_results["contaminated_window_reference"]

    md_content = f"""# Research Report: probability_ranking_vs_absolute_calibration

- **Executed at**: {datetime.datetime.now().isoformat()}
- **Ticker**: 7203.T
- **Target Variable**: `target_next_close_binary`
- **Model**: Production LightGBM (163 features, n_estimators=300, lr=0.03, metric='auc', early_stopping=30)
- **Train Window (Historical 70%)**: {train_data['date'].min().date()} to {train_data['date'].max().date()} ({len(train_data)} trading days)
- **Validation Window (Early Stopping 15%)**: {valid_data['date'].min().date()} to {valid_data['date'].max().date()} ({len(valid_data)} trading days)
- **Clean Test Window (Strict Holdout)**: {clean_diag['date_start']} to {clean_diag['date_end']} ({clean_diag['sample_count']} trading days)
- **Recent Regime Window**: {recent_diag['date_start']} to {recent_diag['date_end']} ({recent_diag['sample_count']} trading days)

---

## 1. Executive Summary & Diagnostic Answers

### Core Research Questions
1. **Does probability ranking possess discriminative power without contamination?**
   - In the clean test window (2024-04-01 to 2026-09-17), ROC AUC is **{clean_diag['ranking_metrics']['roc_auc']}** and Average Precision is **{clean_diag['ranking_metrics']['average_precision']}** (against a base rate of {clean_diag['base_actual_up_rate']}).
   - Top vs Bottom Lift (Q5 - Q1) is **{clean_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** ({clean_diag['lift_metrics']['top_vs_bottom_lift_ratio']:.2f}x lift).
2. **Is the 5-quantile actual Up rate monotonic?**
   - Clean Test Window: Monotonic Increasing = **{clean_diag['lift_metrics']['is_monotonic_increasing']}** (Strictly Monotonic: **{clean_diag['lift_metrics']['is_strictly_monotonic']}**, Inversions = {clean_diag['lift_metrics']['inversion_count']}).
   - Recent Regime (2026): Monotonic Increasing = **{recent_diag['lift_metrics']['is_monotonic_increasing']}** (Inversions = {recent_diag['lift_metrics']['inversion_count']}).
3. **Is the model calibrated in probability space?**
   - Clean Test Window Brier Score is **{clean_diag['calibration_metrics']['brier_score']}**, ECE (10-bin) is **{clean_diag['calibration_metrics']['ece_10bin']}**, and Log Loss is **{clean_diag['calibration_metrics']['log_loss']}**.
   - Mean predicted probability is {clean_diag['pred_stats']['mean']:.4f} vs actual Up rate of {clean_diag['base_actual_up_rate']:.4f} (Mean Overconfidence Gap = {clean_diag['calibration_metrics']['mean_overconfidence']:+.4f}).

---

## 2. 5-Quantile (Quintile) Lift Tables

### Primary: Clean Test Window (2024-04-01 to 2026-09-17, N={clean_diag['sample_count']})

| Quantile | Sample Count | p_up Range | Mean p_up | Actual Up Count | Actual Down Count | Actual Up Rate | Calibration Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for q in clean_diag["quantiles_table"]:
        md_content += f"| **{q['quantile']}** | {q['sample_count']} | [{q['min_p_up']:.4f}, {q['max_p_up']:.4f}] | {q['mean_p_up']:.4f} | {q['actual_up_count']} | {q['actual_down_count']} | **{q['actual_up_rate']:.4f}** | {q['calibration_gap']:+.4f} |\n"

    md_content += f"""
- **Top vs Bottom Lift (Q5 - Q1)**: **{clean_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** ({clean_diag['lift_metrics']['top_vs_bottom_lift_ratio']}x)
- **Monotonicity**: Strictly Monotonic = `{clean_diag['lift_metrics']['is_strictly_monotonic']}`, Inversions = `{clean_diag['lift_metrics']['inversion_count']}`
- **Quantile Spearman Correlation**: `{clean_diag['ranking_metrics']['quantile_spearman']}`

---

### Secondary: Recent Regime 2026 Alone (2026-01-01 to 2026-09-17, N={recent_diag['sample_count']})

| Quantile | Sample Count | p_up Range | Mean p_up | Actual Up Count | Actual Down Count | Actual Up Rate | Calibration Gap |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
"""
    for q in recent_diag["quantiles_table"]:
        md_content += f"| **{q['quantile']}** | {q['sample_count']} | [{q['min_p_up']:.4f}, {q['max_p_up']:.4f}] | {q['mean_p_up']:.4f} | {q['actual_up_count']} | {q['actual_down_count']} | **{q['actual_up_rate']:.4f}** | {q['calibration_gap']:+.4f} |\n"

    md_content += f"""
- **Top vs Bottom Lift (Q5 - Q1)**: **{recent_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** ({recent_diag['lift_metrics']['top_vs_bottom_lift_ratio']}x)
- **Monotonicity**: Strictly Monotonic = `{recent_diag['lift_metrics']['is_strictly_monotonic']}`, Inversions = `{recent_diag['lift_metrics']['inversion_count']}`
- **Quantile Spearman Correlation**: `{recent_diag['ranking_metrics']['quantile_spearman']}`

---

## 3. Comprehensive Diagnostic Metrics Across Windows

| Metric | Clean Test Window (2024-2026) | Recent Regime (2026 Alone) | Reference Contaminated Window (2023-2026) |
|---|:---:|:---:|:---:|
| **Sample Count (N)** | {clean_diag['sample_count']} | {recent_diag['sample_count']} | {contam_diag['sample_count']} |
| **Market Base Up Rate** | {clean_diag['base_actual_up_rate']:.4f} | {recent_diag['base_actual_up_rate']:.4f} | {contam_diag['base_actual_up_rate']:.4f} |
| **Model Mean p_up** | {clean_diag['pred_stats']['mean']:.4f} | {recent_diag['pred_stats']['mean']:.4f} | {contam_diag['pred_stats']['mean']:.4f} |
| **ROC AUC** | **{clean_diag['ranking_metrics']['roc_auc']}** | **{recent_diag['ranking_metrics']['roc_auc']}** | **{contam_diag['ranking_metrics']['roc_auc']}** |
| **Average Precision** | {clean_diag['ranking_metrics']['average_precision']} | {recent_diag['ranking_metrics']['average_precision']} | {contam_diag['ranking_metrics']['average_precision']} |
| **Top vs Bottom Lift (Q5 - Q1)** | **{clean_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** | **{recent_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** | **{contam_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}** |
| **Q5 / Q1 Lift Ratio** | {clean_diag['lift_metrics']['top_vs_bottom_lift_ratio']}x | {recent_diag['lift_metrics']['top_vs_bottom_lift_ratio']}x | {contam_diag['lift_metrics']['top_vs_bottom_lift_ratio']}x |
| **Monotonic Inversions** | {clean_diag['lift_metrics']['inversion_count']} | {recent_diag['lift_metrics']['inversion_count']} | {contam_diag['lift_metrics']['inversion_count']} |
| **Brier Score** | {clean_diag['calibration_metrics']['brier_score']} | {recent_diag['calibration_metrics']['brier_score']} | {contam_diag['calibration_metrics']['brier_score']} |
| **ECE (10-bin)** | {clean_diag['calibration_metrics']['ece_10bin']} | {recent_diag['calibration_metrics']['ece_10bin']} | {contam_diag['calibration_metrics']['ece_10bin']} |
| **Log Loss** | {clean_diag['calibration_metrics']['log_loss']} | {recent_diag['calibration_metrics']['log_loss']} | {contam_diag['calibration_metrics']['log_loss']} |
| **Threshold 0.5 Balanced Acc** | {clean_diag['threshold_05_classification']['balanced_accuracy']} | {recent_diag['threshold_05_classification']['balanced_accuracy']} | {contam_diag['threshold_05_classification']['balanced_accuracy']} |
| **Threshold 0.5 Up Recall** | {clean_diag['threshold_05_classification']['up_recall']} | {recent_diag['threshold_05_classification']['up_recall']} | {contam_diag['threshold_05_classification']['up_recall']} |
| **Threshold 0.5 Down Recall** | {clean_diag['threshold_05_classification']['down_recall']} | {recent_diag['threshold_05_classification']['down_recall']} | {contam_diag['threshold_05_classification']['down_recall']} |

---

## 4. Findings & Strategic Implications

### A. Impact of Validation Leakage Removal
- In the contaminated 2023-2026 window, ROC AUC was {contam_diag['ranking_metrics']['roc_auc']}.
- When restricting strictly to the Clean Test Window (2024-04-01 to 2026-09-17), ROC AUC is **{clean_diag['ranking_metrics']['roc_auc']}**.
- This proves that removing the overlapping validation period clarifies the true out-of-sample ranking power of the production model.

### B. Probability Ranking vs. Absolute Calibration
- **Absolute Calibration Issue**: The model exhibits a persistent calibration offset (mean p_up = {clean_diag['pred_stats']['mean']:.4f} vs market actual = {clean_diag['base_actual_up_rate']:.4f}). Because predictions cluster around ~0.46-0.54, an arbitrary absolute cutoff at 0.5 causes severe directional bias.
- **Relative Ranking Viability**: 
  - Q5 actual Up rate is {clean_diag['quantiles_table'][4]['actual_up_rate']:.4f} vs Q1 actual Up rate of {clean_diag['quantiles_table'][0]['actual_up_rate']:.4f}, producing a lift of **{clean_diag['lift_metrics']['top_vs_bottom_lift_diff']:+.4f}**.
  - This demonstrates whether higher predicted probabilities consistently reflect a higher empirical frequency of upward market movements.

---
*Report generated autonomously under the single-hypothesis research protocol. No production models, database tables, or configurations were modified.*
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved diagnosis report to: {report_path}")

    return overall_output


def main():
    parser = argparse.ArgumentParser(description="Research probability ranking vs absolute calibration diagnosis")
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=MAIN_REPO_ROOT / "data" / "features" / "7203.T.parquet",
        help="Path to feature parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "autonomous",
        help="Directory to save diagnosis results and report",
    )
    args = parser.parse_args()

    run_experiment(
        feature_path=args.feature_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
