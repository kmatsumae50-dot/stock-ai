"""overnight.py

Close-to-Open (Overnight Gap) production model pillar.
Complements Close-to-Close by predicting the overnight direction from previous close to next 9:00 AM open.
Provides Clean Test calibrated Quantile bands (Q1-Q5) and historical win rates.
"""

from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.labels.targets import TARGET_NEXT_OPEN_BINARY, add_next_open_binary_target
from stock_ai.models.baseline import select_feature_columns
from stock_ai.models.predict import get_binary_shap_contributions

OVERNIGHT_MODEL_NAME = "lightgbm_overnight_binary"
OVERNIGHT_TARGET = "next_open_binary"
OVERNIGHT_ARTIFACT_FILENAME = "lightgbm_overnight_binary_{ticker}.joblib"
OVERNIGHT_METRICS_FILENAME = "lightgbm_overnight_binary_{ticker}.json"

POOLED_OVERNIGHT_MODEL_NAME = "lightgbm_overnight_pooled_cs"
POOLED_OVERNIGHT_TARGET = "target_cs_outperform"
POOLED_OVERNIGHT_ARTIFACT_FILENAME = "lightgbm_overnight_pooled_cs.joblib"
POOLED_OVERNIGHT_METRICS_FILENAME = "lightgbm_overnight_pooled_cs.json"

# Clean Test Window (2024-04-01 to 2026-09-17) empirical quantile cutoffs
OVERNIGHT_QUANTILE_CUTOFFS = [
    {
        "quantile": "Q1",
        "min_p": 0.0,
        "max_p": 0.3846,
        "label": "ギャップダウン警戒帯",
        "direction": "down",
        "historical_win_rate": 0.4132,
        "description": "実績ギャップアップ率 41.3%（陰線・下落警戒）",
    },
    {
        "quantile": "Q2",
        "min_p": 0.3846,
        "max_p": 0.4638,
        "label": "弱気帯",
        "direction": "down",
        "historical_win_rate": 0.4463,
        "description": "実績ギャップアップ率 44.6%（弱気）",
    },
    {
        "quantile": "Q3",
        "min_p": 0.4638,
        "max_p": 0.5332,
        "label": "中立帯",
        "direction": "neutral",
        "historical_win_rate": 0.6250,
        "description": "実績ギャップアップ率 62.5%（中立・推移注視）",
    },
    {
        "quantile": "Q4",
        "min_p": 0.5332,
        "max_p": 0.6059,
        "label": "強気先行帯",
        "direction": "up",
        "historical_win_rate": 0.5785,
        "description": "実績ギャップアップ率 57.9%（堅調）",
    },
    {
        "quantile": "Q5",
        "min_p": 0.6059,
        "max_p": 1.0,
        "label": "強気ギャップアップ帯",
        "direction": "up",
        "historical_win_rate": 0.6446,
        "description": "実績ギャップアップ率 64.5%（2026年は68.6%・最高勝率群）",
    },
]


def compute_empirical_quantile_cutoffs(
    y_true: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray,
) -> list[dict[str, Any]]:
    """Compute empirical 5-quantile cutoffs and historical win rates from holdout predictions."""
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) < 5 or len(np.unique(y_t)) < 2:
        return OVERNIGHT_QUANTILE_CUTOFFS

    sub = pd.DataFrame({"y": y_t, "p": y_p})
    sub["rank"] = sub["p"].rank(method="first")
    sub["quantile"] = pd.qcut(sub["rank"], q=5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])

    labels_map = {
        "Q1": ("ギャップダウン警戒帯", "down", "陰線・下落警戒"),
        "Q2": ("弱気帯", "down", "弱気"),
        "Q3": ("中立帯", "neutral", "中立・推移注視"),
        "Q4": ("強気先行帯", "up", "堅調"),
        "Q5": ("強気ギャップアップ帯", "up", "最高勝率群"),
    }

    cutoffs = []
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    for i, q_name in enumerate(q_labels):
        q_rows = sub[sub["quantile"] == q_name]
        min_p = float(q_rows["p"].min()) if i > 0 else 0.0
        max_p = float(q_rows["p"].max()) if i < 4 else 1.0
        win_rate = float(q_rows["y"].mean()) if len(q_rows) > 0 else 0.5
        label, direction, desc_suffix = labels_map[q_name]

        cutoffs.append(
            {
                "quantile": q_name,
                "min_p": round(min_p, 4),
                "max_p": round(max_p, 4),
                "label": label,
                "direction": direction,
                "historical_win_rate": round(win_rate, 4),
                "description": f"実績ギャップアップ率 {win_rate * 100:.1f}%（{desc_suffix}）",
            }
        )
    return cutoffs


def load_overnight_quantile_cutoffs(
    ticker: str | None = None,
    metrics_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load ticker-specific quantile cutoffs from metrics JSON, falling back to 7203.T defaults."""
    if ticker:
        m_dir = Path(metrics_dir) if metrics_dir else Path("reports") / "metrics"
        m_path = m_dir / OVERNIGHT_METRICS_FILENAME.format(ticker=ticker)
        if m_path.exists():
            try:
                data = json.loads(m_path.read_text(encoding="utf-8"))
                if "quantile_cutoffs" in data and len(data["quantile_cutoffs"]) == 5:
                    return data["quantile_cutoffs"]
            except Exception:
                pass
    return OVERNIGHT_QUANTILE_CUTOFFS


def classify_overnight_quantile(
    p_up: float,
    ticker: str | None = None,
    metrics_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Determine the empirical 5-quantile band and expected edge for an overnight p_up."""
    cutoffs = load_overnight_quantile_cutoffs(ticker=ticker, metrics_dir=metrics_dir)
    q_info = cutoffs[0]
    for info in reversed(cutoffs):
        if p_up >= info.get("min_p", 0.0):
            q_info = info
            break

    return {
        "quantile": q_info["quantile"],
        "label": q_info["label"],
        "direction": q_info["direction"],
        "historical_win_rate": q_info.get("historical_win_rate"),
        "description": q_info.get("description", ""),
        "p_up": round(p_up, 4),
    }


def train_lightgbm_overnight_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str] | None = None,
):
    """Train LightGBM binary classifier on Close-to-Open target matching production spec."""
    import lightgbm as lgb

    train_data = train.copy()
    valid_data = valid.copy()

    if TARGET_NEXT_OPEN_BINARY not in train_data.columns:
        train_data = add_next_open_binary_target(train_data)
    if TARGET_NEXT_OPEN_BINARY not in valid_data.columns:
        valid_data = add_next_open_binary_target(valid_data)

    if feature_columns is None:
        from stock_ai.features.overnight import select_overnight_curated_features

        feature_columns = select_overnight_curated_features(train_data)

    train_subset = train_data.dropna(subset=feature_columns + [TARGET_NEXT_OPEN_BINARY])
    valid_subset = valid_data.dropna(subset=feature_columns + [TARGET_NEXT_OPEN_BINARY])

    train_subset = train_subset[train_subset[TARGET_NEXT_OPEN_BINARY] >= 0]
    valid_subset = valid_subset[valid_subset[TARGET_NEXT_OPEN_BINARY] >= 0]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train_subset[feature_columns],
        train_subset[TARGET_NEXT_OPEN_BINARY],
        eval_set=[(valid_subset[feature_columns], valid_subset[TARGET_NEXT_OPEN_BINARY])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, feature_columns


class HeterogeneousEnsembleClassifier:
    """Stefan Jansen ML4AT Chapter 12: Combines LightGBM non-linear decision trees with L2-regularized linear model."""

    def __init__(self, lgb_model: Any, linear_model: Any, w_lgb: float = 0.80, w_lin: float = 0.20):
        self.lgb_model = lgb_model
        self.linear_model = linear_model
        self.w_lgb = w_lgb
        self.w_lin = w_lin
        self.feature_importances_ = getattr(lgb_model, "feature_importances_", None)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        p_lgb = self.lgb_model.predict_proba(X_df)
        p_lin = self.linear_model.predict_proba(X_df.fillna(0.0))
        return self.w_lgb * p_lgb + self.w_lin * p_lin

    def predict(self, X: Any) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)


def predict_latest_overnight_rows(
    model,
    feature_columns: list[str],
    data: pd.DataFrame,
    top_features: list[dict[str, float]] | None = None,
    target: str = OVERNIGHT_TARGET,
    model_name: str = OVERNIGHT_MODEL_NAME,
    ticker: str | None = None,
    metrics_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Generate latest overnight prediction row with quantile metadata."""
    latest_rows = data.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest_rows = latest_rows.dropna(subset=feature_columns)

    if latest_rows.empty:
        return pd.DataFrame()

    probabilities = model.predict_proba(latest_rows[feature_columns])
    probability_down = probabilities[:, 0]
    probability_up = probabilities[:, 1]
    predicted_class = (probability_up >= 0.5).astype(int)

    shap_contributions = get_binary_shap_contributions(
        model,
        feature_columns,
        latest_rows[feature_columns],
        top_n=10,
    )

    result = latest_rows[["date", "ticker"]].copy()
    result["target"] = target
    result["model_name"] = model_name
    result["predicted_class"] = predicted_class
    result["confidence"] = probabilities.max(axis=1)
    result["probability_up"] = probability_up
    result["probability_down"] = probability_down

    row_tickers = (
        latest_rows["ticker"].tolist()
        if "ticker" in latest_rows.columns
        else [ticker] * len(probability_up)
    )
    quantile_infos = [
        classify_overnight_quantile(float(p), ticker=t or ticker, metrics_dir=metrics_dir)
        for p, t in zip(probability_up, row_tickers)
    ]
    result["notes"] = [json.dumps(qinfo, ensure_ascii=False) for qinfo in quantile_infos]

    result["top_features"] = [
        json.dumps(features or top_features or [], ensure_ascii=True)
        for features in shap_contributions
    ]
    return result.reset_index(drop=True)


def train_pooled_cross_sectional_overnight_model(
    universe: list[Any] | None = None,
    features_dir: Path | str = "data/features",
    models_dir: Path | str = "models/artifacts",
    metrics_dir: Path | str = "reports/metrics",
    train_end_date: str = "2024-04-01",
    sync_features_to_disk: bool = True,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Train a universe-wide pooled panel LightGBM model on cross-sectional outperformance.

    Strictly purges lookahead bias by using pre-2024 data and cross-sectional rank targets.
    """
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, roc_auc_score

    from stock_ai.data.universe import load_universe
    from stock_ai.features.overnight import (
        add_cross_sectional_overnight_features,
        select_overnight_curated_features,
    )
    from stock_ai.models.baseline import save_model
    from stock_ai.storage.parquet_store import read_parquet, write_parquet

    features_path = Path(features_dir)
    models_path = Path(models_dir)
    metrics_path = Path(metrics_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    uni = universe or load_universe()
    records: list[pd.DataFrame] = []
    ticker_dfs: dict[str, pd.DataFrame] = {}

    for t in uni:
        code = getattr(t, "code", str(t))
        name = getattr(t, "name", code)
        sec = getattr(t, "sector_ja", "") or getattr(t, "sector", "") or ""
        f_file = features_path / f"{code}.parquet"
        if f_file.exists():
            df_t = read_parquet(f_file)
            if not df_t.empty and "date" in df_t.columns:
                df_t["ticker"] = code
                df_t["name"] = name
                df_t["sector"] = sec
                records.append(df_t)
                ticker_dfs[code] = df_t

    if not records:
        raise ValueError("No feature files found in universe.")

    all_df = pd.concat(records, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Attach cross-sectional features and outperformance target across universe
    all_df = add_cross_sectional_overnight_features(all_df)

    # Persist the cross-sectional features back to individual parquet files
    if sync_features_to_disk:
        for code in ticker_dfs:
            t_sub = all_df[all_df["ticker"] == code].sort_values("date").reset_index(drop=True)
            write_parquet(t_sub, features_path / f"{code}.parquet")

    feature_columns = select_overnight_curated_features(all_df)

    # Purged time split: Train < train_end_date
    split_dt = pd.to_datetime(train_end_date)
    train_mask = all_df["date"] < split_dt
    feature_cols_clean = [c for c in feature_columns if c in all_df.columns]
    feature_columns = feature_cols_clean
    train_subset = all_df[train_mask].dropna(subset=feature_columns + [POOLED_OVERNIGHT_TARGET])

    if train_subset.empty:
        raise ValueError("Training subset is empty.")

    # Validation window: last 15% of train dates
    train_dates = sorted(train_subset["date"].unique())
    n_train_dates = len(train_dates)
    valid_start_idx = max(1, int(n_train_dates * 0.85))
    valid_start_dt = train_dates[valid_start_idx]

    train_data = train_subset[train_subset["date"] < valid_start_dt]
    valid_data = train_subset[train_subset["date"] >= valid_start_dt]

    from stock_ai.features.afml import compute_afml_sample_weights, prune_features_with_mda

    # AFML Chapter 4: Sample Weighting
    ret_col = "next_open_return" if "next_open_return" in train_subset.columns else (
        "return_1d" if "return_1d" in train_subset.columns else None
    )
    if ret_col is not None:
        train_rets = train_data[ret_col].fillna(0.0)
        subset_rets = train_subset[ret_col].fillna(0.0)
    else:
        train_rets = np.zeros(len(train_data))
        subset_rets = np.zeros(len(train_subset))

    w_train = compute_afml_sample_weights(train_rets, train_data["date"], time_decay=0.50, cost_bps=20.0)
    w_subset = compute_afml_sample_weights(subset_rets, train_subset["date"], time_decay=0.50, cost_bps=20.0)

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=75,
        learning_rate=0.05,
        num_leaves=12,
        min_child_samples=60,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train_data[feature_columns],
        train_data[POOLED_OVERNIGHT_TARGET].astype(int),
        sample_weight=w_train,
        eval_set=[(valid_data[feature_columns], valid_data[POOLED_OVERNIGHT_TARGET].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # AFML Chapter 8: OOS Permutation Importance (MDA) Feature Pruning
    pruned_features, mda_scores = prune_features_with_mda(
        model,
        valid_data,
        valid_data[POOLED_OVERNIGHT_TARGET],
        feature_columns,
        min_importance=0.0,
    )
    feature_columns = pruned_features

    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    # ML4AT Chapter 7 & 12: Train L2-regularized Linear Pipeline
    linear_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42)),
    ])
    linear_pipe.fit(
        train_data[feature_columns].fillna(0.0),
        train_data[POOLED_OVERNIGHT_TARGET].astype(int),
        clf__sample_weight=w_train,
    )

    # Fit LightGBM on selected features with AFML sample weights
    model.fit(
        train_data[feature_columns],
        train_data[POOLED_OVERNIGHT_TARGET].astype(int),
        sample_weight=w_train,
    )

    val_lgb_probs = model.predict_proba(valid_data[feature_columns])[:, 1]
    val_lin_probs = linear_pipe.predict_proba(valid_data[feature_columns].fillna(0.0))[:, 1]
    y_val_arr = valid_data[POOLED_OVERNIGHT_TARGET].astype(int).values

    best_w = 0.80
    best_auc = -1.0
    for candidate_w in [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]:
        blend_p = candidate_w * val_lgb_probs + (1.0 - candidate_w) * val_lin_probs
        try:
            score = float(roc_auc_score(y_val_arr, blend_p))
            if score > best_auc:
                best_auc = score
                best_w = candidate_w
        except Exception:
            pass

    blend_weights = {"lgb": best_w, "linear": round(1.0 - best_w, 2)}

    # Refit both models on full training set with sample weights
    model.fit(
        train_subset[feature_columns],
        train_subset[POOLED_OVERNIGHT_TARGET].astype(int),
        sample_weight=w_subset,
    )
    linear_pipe.fit(
        train_subset[feature_columns].fillna(0.0),
        train_subset[POOLED_OVERNIGHT_TARGET].astype(int),
        clf__sample_weight=w_subset,
    )

    ensemble = HeterogeneousEnsembleClassifier(
        lgb_model=model,
        linear_model=linear_pipe,
        w_lgb=best_w,
        w_lin=round(1.0 - best_w, 2),
    )

    # Validation metrics with heterogeneous ensemble
    val_probs = ensemble.predict_proba(valid_data[feature_columns])[:, 1]
    val_auc = float(roc_auc_score(valid_data[POOLED_OVERNIGHT_TARGET].astype(int), val_probs))
    val_acc = float(accuracy_score(valid_data[POOLED_OVERNIGHT_TARGET].astype(int), (val_probs >= 0.5).astype(int)))

    # Empirical quantile cutoffs on validation set
    quantile_cutoffs = compute_empirical_quantile_cutoffs(
        valid_data[POOLED_OVERNIGHT_TARGET].astype(int),
        val_probs,
    )

    metrics_info = {
        "model_name": POOLED_OVERNIGHT_MODEL_NAME,
        "target": POOLED_OVERNIGHT_TARGET,
        "trained_samples": len(train_data),
        "validation_samples": len(valid_data),
        "validation_auc": val_auc,
        "validation_accuracy": val_acc,
        "feature_columns_count": len(feature_columns),
        "top_features": [
            {"feature": f, "importance": int(imp)}
            for f, imp in sorted(
                zip(feature_columns, model.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
        ],
        "quantile_cutoffs": quantile_cutoffs,
    }

    # Save artifacts (Heterogeneous Ensemble: LightGBM + L2 Linear)
    save_model(
        model=ensemble,
        feature_columns=feature_columns,
        path=models_path / POOLED_OVERNIGHT_ARTIFACT_FILENAME,
        metadata=metrics_info,
    )

    (metrics_path / POOLED_OVERNIGHT_METRICS_FILENAME).write_text(
        json.dumps(metrics_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return ensemble, feature_columns, metrics_info
