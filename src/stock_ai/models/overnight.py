"""overnight.py

Close-to-Open (Overnight Gap) production model pillar.
Complements Close-to-Close by predicting the overnight direction from previous close to next 9:00 AM open.
Provides Clean Test calibrated Quantile bands (Q1-Q5) and historical win rates.
"""

from __future__ import annotations

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
        feature_columns = select_feature_columns(train_data, target_column=TARGET_NEXT_OPEN_BINARY)

    train_subset = train_data.dropna(subset=feature_columns + [TARGET_NEXT_OPEN_BINARY])
    valid_subset = valid_data.dropna(subset=feature_columns + [TARGET_NEXT_OPEN_BINARY])

    train_subset = train_subset[train_subset[TARGET_NEXT_OPEN_BINARY] >= 0]
    valid_subset = valid_subset[valid_subset[TARGET_NEXT_OPEN_BINARY] >= 0]

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
        train_subset[feature_columns],
        train_subset[TARGET_NEXT_OPEN_BINARY],
        eval_set=[(valid_subset[feature_columns], valid_subset[TARGET_NEXT_OPEN_BINARY])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
    )
    return model, feature_columns


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
