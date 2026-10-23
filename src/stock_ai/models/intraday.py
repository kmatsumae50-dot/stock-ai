"""src/stock_ai/models/intraday.py

Intraday (Open-to-Close: 09:00 -> 15:30) LightGBM Pooled Cross-Sectional Model.
Predicts daytime return and cross-sectional outperformance for morning 07:30 JST pre-market deployment.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from stock_ai.config import PROJECT_ROOT, Settings, load_settings
from stock_ai.data.universe import TickerInfo, load_universe
from stock_ai.events.earnings_guard import check_earnings_risk
from stock_ai.features.intraday import (
    ALL_INTRADAY_FEATURES,
    add_cross_sectional_intraday_features,
    add_intraday_premarket_features,
)
from stock_ai.features.overnight import check_jumping_catch_risk
from stock_ai.storage.parquet_store import read_parquet, write_parquet

logger = logging.getLogger(__name__)

INTRADAY_MODEL_NAME = "lightgbm_intraday_pooled_cs"
INTRADAY_TARGET = "target_intraday_cs_outperform"
CLEAN_WIN_TARGET = "target_clean_intraday_win"
INTRADAY_ARTIFACT_FILENAME = "lightgbm_intraday_pooled_cs.joblib"
INTRADAY_METRICS_FILENAME = "lightgbm_intraday_pooled_cs.json"

INTRADAY_HIGH_CONVICTION_RETURN_THRESH = 0.0030  # +30 bps expected return hurdle
INTRADAY_HIGH_CONVICTION_PROB_THRESH = 0.50

INTRADAY_QUANTILE_CUTOFFS = [
    {
        "quantile": "Q1",
        "min_p": 0.0,
        "max_p": 0.44,
        "label": "寄り天・陰線警戒帯",
        "direction": "down",
        "description": "寄り付き後の利確売り・陰線反落リスクが高い警戒ゾーン",
    },
    {
        "quantile": "Q2",
        "min_p": 0.44,
        "max_p": 0.48,
        "label": "軟調推移帯",
        "direction": "down",
        "description": "日中上値が重くアンダーパフォームしやすい軟調ゾーン",
    },
    {
        "quantile": "Q3",
        "min_p": 0.48,
        "max_p": 0.52,
        "label": "中立推移帯",
        "direction": "neutral",
        "description": "市場平均並みの値動き（キャッシュ待機推奨）",
    },
    {
        "quantile": "Q4",
        "min_p": 0.52,
        "max_p": 0.56,
        "label": "日中堅調帯",
        "direction": "up",
        "description": "寄り付きからの底堅い買い支えが期待できる堅調ゾーン",
    },
    {
        "quantile": "Q5",
        "min_p": 0.56,
        "max_p": 1.0,
        "label": "日中陽線急伸期待帯",
        "direction": "up",
        "description": "最も高い確率で日中大引けにかけて陽線上昇が期待できる特選ゾーン",
    },
]


def classify_intraday_quantile(prob: float, pred_ret: float | None = None) -> dict[str, Any]:
    """Classify an intraday predicted probability and return into human-interpretable bands."""
    clamped_p = max(0.0, min(1.0, float(prob)))
    selected = INTRADAY_QUANTILE_CUTOFFS[2]  # Default Q3
    for b in INTRADAY_QUANTILE_CUTOFFS:
        if b["min_p"] <= clamped_p < b["max_p"] or (b["quantile"] == "Q5" and clamped_p >= b["min_p"]):
            selected = b
            break

    is_conviction = (
        clamped_p >= INTRADAY_HIGH_CONVICTION_PROB_THRESH
        and (pred_ret is None or pred_ret >= INTRADAY_HIGH_CONVICTION_RETURN_THRESH)
    )

    return {
        "quantile": selected["quantile"],
        "label": selected["label"],
        "direction": selected["direction"],
        "description": selected["description"],
        "probability": clamped_p,
        "expected_return": pred_ret,
        "is_high_conviction": is_conviction,
    }


def train_pooled_cross_sectional_intraday_model(
    universe: list[TickerInfo] | None = None,
    settings: Settings | None = None,
    features_dir: Path | str | None = None,
    train_end_date: str = "2024-04-01",
    sync_features_to_disk: bool = True,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Train a leak-free Pooled Cross-Sectional Intraday model (Classifier + Clean Win + Regressor bundle).

    Trained strictly on dates prior to `train_end_date` (2024-04-01) across all universe tickers.
    """
    import lightgbm as lgb

    if settings is None:
        settings = load_settings()
    if universe is None:
        universe = load_universe()

    features_path = Path(features_dir or settings.paths.feature_data_dir)
    models_path = settings.paths.model_dir
    metrics_path = Path("reports") / "metrics"
    models_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    records: list[pd.DataFrame] = []
    ticker_dfs: dict[str, pd.DataFrame] = {}

    for t in universe:
        code = t.code
        f_file = features_path / f"{code}.parquet"
        if not f_file.exists():
            continue
        df = read_parquet(f_file)
        if df.empty or len(df) < 50:
            continue
        df["ticker"] = code
        df["sector"] = getattr(t, "sector_ja", "") or getattr(t, "sector", "") or "General"
        df = add_intraday_premarket_features(df)
        records.append(df)
        ticker_dfs[code] = df

    if not records:
        raise ValueError("No feature files found for intraday training.")

    all_df = pd.concat(records, ignore_index=True)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # Attach Cross-Sectional relative features
    all_df = add_cross_sectional_intraday_features(all_df)

    if sync_features_to_disk:
        for code in ticker_dfs:
            t_sub = all_df[all_df["ticker"] == code].sort_values("date").reset_index(drop=True)
            write_parquet(t_sub, features_path / f"{code}.parquet")

    from stock_ai.features.afml import (
        calculate_bet_size,
        compute_afml_sample_weights,
        compute_dynamic_triple_barrier_events,
        generate_meta_labels,
        prune_features_with_mda,
    )

    feature_columns = [c for c in ALL_INTRADAY_FEATURES if c in all_df.columns]

    split_dt = pd.to_datetime(train_end_date)
    train_mask = all_df["date"] < split_dt
    train_subset = all_df[train_mask].dropna(subset=feature_columns + [INTRADAY_TARGET, "intraday_return"]).copy()

    if train_subset.empty:
        raise ValueError("Training subset for Intraday model is empty.")

    # Validation window: last 15% of train dates
    train_dates = sorted(train_subset["date"].unique())
    valid_start_idx = max(1, int(len(train_dates) * 0.85))
    valid_start_dt = train_dates[valid_start_idx]

    train_data = train_subset[train_subset["date"] < valid_start_dt].copy()
    valid_data = train_subset[train_subset["date"] >= valid_start_dt].copy()

    # AFML Chapter 4: Sample Weighting (Informativeness x Time-Decay)
    w_train = compute_afml_sample_weights(
        train_data["intraday_return"],
        train_data["date"],
        time_decay=0.50,
        cost_bps=20.0,
    )
    w_subset = compute_afml_sample_weights(
        train_subset["intraday_return"],
        train_subset["date"],
        time_decay=0.50,
        cost_bps=20.0,
    )

    # 1. Train Classifier for Cross-Sectional Outperform (with AFML Sample Weighting)
    clf = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=12,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    clf.fit(
        train_data[feature_columns],
        train_data[INTRADAY_TARGET].astype(int),
        sample_weight=w_train,
        eval_set=[(valid_data[feature_columns], valid_data[INTRADAY_TARGET].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    # AFML Chapter 8: OOS Permutation Importance (MDA) Feature Pruning on validation set
    pruned_features, mda_scores = prune_features_with_mda(
        clf,
        valid_data,
        valid_data[INTRADAY_TARGET],
        feature_columns,
        min_importance=0.0,
    )
    feature_columns = pruned_features

    # Refit Primary classifier on selected features with sample weights
    clf.fit(
        train_subset[feature_columns],
        train_subset[INTRADAY_TARGET].astype(int),
        sample_weight=w_subset,
    )
    val_probs = clf.predict_proba(valid_data[feature_columns])[:, 1]
    val_auc = float(roc_auc_score(valid_data[INTRADAY_TARGET].astype(int), val_probs))
    val_acc = float(accuracy_score(valid_data[INTRADAY_TARGET].astype(int), (val_probs >= 0.5).astype(int)))

    # 2. Train Classifier for Clean Intraday Win
    if CLEAN_WIN_TARGET not in train_subset.columns:
        train_subset[CLEAN_WIN_TARGET] = (train_subset["intraday_return"] >= 0.0020).astype(int)
        train_data[CLEAN_WIN_TARGET] = (train_data["intraday_return"] >= 0.0020).astype(int)
        valid_data[CLEAN_WIN_TARGET] = (valid_data["intraday_return"] >= 0.0020).astype(int)

    clf_clean = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=12,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    clf_clean.fit(
        train_data[feature_columns],
        train_data[CLEAN_WIN_TARGET].astype(int),
        sample_weight=w_train,
        eval_set=[(valid_data[feature_columns], valid_data[CLEAN_WIN_TARGET].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_clean_probs = clf_clean.predict_proba(valid_data[feature_columns])[:, 1]
    val_clean_auc = float(roc_auc_score(valid_data[CLEAN_WIN_TARGET].astype(int), val_clean_probs))
    val_clean_acc = float(accuracy_score(valid_data[CLEAN_WIN_TARGET].astype(int), (val_clean_probs >= 0.5).astype(int)))
    clf_clean.fit(
        train_subset[feature_columns],
        train_subset[CLEAN_WIN_TARGET].astype(int),
        sample_weight=w_subset,
    )

    # 3. AFML Chapter 3: Dynamic Volatility Triple-Barrier & Meta-Labeling Model
    tb_subset = compute_dynamic_triple_barrier_events(train_subset, k_pt=1.0, k_sl=0.8, cost_bps=20.0)
    primary_subset_probs = clf.predict_proba(train_subset[feature_columns])[:, 1]
    train_subset["target_afml_meta"] = generate_meta_labels(
        primary_preds=primary_subset_probs,
        actual_returns=train_subset["intraday_return"],
        lower_barrier_hit=tb_subset["tb_hit_lower"],
        cost_bps=20.0,
        primary_buy_threshold=0.50,
    )

    tb_train = compute_dynamic_triple_barrier_events(train_data, k_pt=1.0, k_sl=0.8, cost_bps=20.0)
    primary_train_probs = clf.predict_proba(train_data[feature_columns])[:, 1]
    train_data["target_afml_meta"] = generate_meta_labels(
        primary_preds=primary_train_probs,
        actual_returns=train_data["intraday_return"],
        lower_barrier_hit=tb_train["tb_hit_lower"],
        cost_bps=20.0,
        primary_buy_threshold=0.50,
    )

    tb_valid = compute_dynamic_triple_barrier_events(valid_data, k_pt=1.0, k_sl=0.8, cost_bps=20.0)
    primary_valid_probs = clf.predict_proba(valid_data[feature_columns])[:, 1]
    valid_data["target_afml_meta"] = generate_meta_labels(
        primary_preds=primary_valid_probs,
        actual_returns=valid_data["intraday_return"],
        lower_barrier_hit=tb_valid["tb_hit_lower"],
        cost_bps=20.0,
        primary_buy_threshold=0.50,
    )

    clf_meta = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.03,
        num_leaves=10,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    clf_meta.fit(
        train_data[feature_columns],
        train_data["target_afml_meta"].astype(int),
        sample_weight=w_train,
        eval_set=[(valid_data[feature_columns], valid_data["target_afml_meta"].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(40, verbose=False)],
    )

    val_meta_probs = clf_meta.predict_proba(valid_data[feature_columns])[:, 1]
    val_meta_auc = float(roc_auc_score(valid_data["target_afml_meta"].astype(int), val_meta_probs))
    val_meta_acc = float(accuracy_score(valid_data["target_afml_meta"].astype(int), (val_meta_probs >= 0.5).astype(int)))
    clf_meta.fit(
        train_subset[feature_columns],
        train_subset["target_afml_meta"].astype(int),
        sample_weight=w_subset,
    )

    # 4. Train Regressor for expected return hurdle (with sample weights)
    reg = lgb.LGBMRegressor(
        n_estimators=180,
        learning_rate=0.02,
        num_leaves=10,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    reg.fit(train_subset[feature_columns], train_subset["intraday_return"], sample_weight=w_subset)

    metrics_info = {
        "model_name": INTRADAY_MODEL_NAME,
        "target": INTRADAY_TARGET,
        "clean_target": CLEAN_WIN_TARGET,
        "meta_target": "target_afml_meta",
        "trained_samples": len(train_data),
        "validation_samples": len(valid_data),
        "validation_auc": val_auc,
        "validation_accuracy": val_acc,
        "validation_clean_auc": val_clean_auc,
        "validation_clean_accuracy": val_clean_acc,
        "validation_meta_auc": val_meta_auc,
        "validation_meta_accuracy": val_meta_acc,
        "feature_columns_count": len(feature_columns),
        "mda_pruned_count": len(ALL_INTRADAY_FEATURES) - len(feature_columns),
        "top_features": [
            {"feature": f, "importance": int(imp)}
            for f, imp in sorted(
                zip(feature_columns, clf.feature_importances_),
                key=lambda x: x[1],
                reverse=True,
            )[:15]
        ],
    }

    bundle = {
        "model": clf,
        "model_clean": clf_clean,
        "model_meta": clf_meta,
        "regressor": reg,
        "feature_columns": feature_columns,
        "mda_scores": mda_scores,
        "metadata": metrics_info,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }

    artifact_path = models_path / INTRADAY_ARTIFACT_FILENAME
    joblib.dump(bundle, artifact_path)
    (metrics_path / INTRADAY_METRICS_FILENAME).write_text(
        json.dumps(metrics_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved Intraday model bundle to %s", artifact_path)

    return bundle, feature_columns, metrics_info


def predict_intraday_for_ticker(
    ticker: str,
    settings: Settings | None = None,
    features_df: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Predict intraday open-to-close probability and expected return for a ticker.

    Ensembles:
    1. Cross-Sectional Outperform probability (`p_intraday_cs`)
    2. Clean Intraday Win probability (`p_clean_win`: no severe drawdown <= -1.2% & cost-adjusted profit)
    """
    if settings is None:
        settings = load_settings()

    artifact_path = settings.paths.model_dir / INTRADAY_ARTIFACT_FILENAME
    if not artifact_path.exists():
        return None

    try:
        bundle = joblib.load(artifact_path)
    except Exception as exc:
        logger.warning("Failed loading intraday model bundle: %s", exc)
        return None

    if features_df is None:
        f_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
        if not f_path.exists():
            return None
        features_df = read_parquet(f_path)

    if features_df.empty:
        return None

    # Ensure premarket features exist
    if "prior_return_1d" not in features_df.columns:
        features_df = add_intraday_premarket_features(features_df)

    feat_cols = [c for c in bundle["feature_columns"] if c in features_df.columns]
    if len(feat_cols) < len(bundle["feature_columns"]) * 0.70:
        return None

    # Fill any missing feature columns with default 0.0
    row_df = features_df.tail(1).copy()
    for col in bundle["feature_columns"]:
        if col not in row_df.columns:
            row_df[col] = 0.0

    X_row = row_df[bundle["feature_columns"]].fillna(0.0)

    from stock_ai.features.afml import calculate_bet_size

    clf = bundle["model"]
    clf_clean = bundle.get("model_clean")
    clf_meta = bundle.get("model_meta")
    reg = bundle.get("regressor")

    p_cs = float(clf.predict_proba(X_row)[:, 1][0])
    p_clean = float(clf_clean.predict_proba(X_row)[:, 1][0]) if clf_clean is not None else p_cs
    p_meta = float(clf_meta.predict_proba(X_row)[:, 1][0]) if clf_meta is not None else p_clean
    bet_size = calculate_bet_size(p_meta)

    # AFML Enhanced Tri-Model Ensemble: 40% CS Outperform + 35% Clean Win + 25% Meta Net-Profit
    prob = 0.40 * p_cs + 0.35 * p_clean + 0.25 * p_meta

    pred_ret = float(reg.predict(X_row)[0]) if reg is not None else 0.0

    info = classify_intraday_quantile(prob, pred_ret=pred_ret)
    info["ticker"] = ticker
    info["date"] = str(row_df["date"].iloc[0]) if "date" in row_df.columns else "-"
    info["p_intraday_cs"] = p_cs
    info["p_clean_win"] = p_clean
    info["p_afml_meta"] = p_meta
    info["bet_size"] = bet_size
    info["p_intraday_ensemble"] = prob
    info["pred_intraday_return"] = pred_ret
    info["expected_return_bps"] = pred_ret * 10000.0

    return info


# ---------------------------------------------------------------------------
# 1-Day Daytrade Sniper Engine (100% Intraday Focused)
# ---------------------------------------------------------------------------

def determine_market_phase(now_dt: datetime | None = None) -> tuple[str, str]:
    """Determine market trading phase in JST:
    - "premarket" (00:00 - 08:59 JST)
    - "intraday_live" (09:00 - 15:29 JST)
    - "post_market" (15:30 - 23:59 JST)
    """
    if now_dt is None:
        now_dt = datetime.now()
    hour_min = now_dt.hour * 60 + now_dt.minute
    if hour_min < 9 * 60:
        return "premarket", "☀️ 寄前 09:00〜15:30 当日陽線・上昇予測"
    elif hour_min < 15 * 60 + 30:
        return "intraday_live", "⚡ 場中リアルタイム 15:30大引け着地予測"
    else:
        return "post_market", "🌙 翌営業日 1-Day (翌朝寄付き＋日中トレンド) 総合予測"


def fetch_realtime_intraday_quote(ticker: str) -> dict[str, Any] | None:
    """Fetch real-time fast quote from yfinance, returning current, open, high, low, prev_close."""
    try:
        import yfinance as yf
        fi = yf.Ticker(ticker).fast_info
        last_price = getattr(fi, "last_price", None)
        if last_price is not None and not np.isnan(last_price) and last_price > 0:
            open_p = getattr(fi, "open", None) or last_price
            high_p = getattr(fi, "day_high", None) or max(last_price, open_p)
            low_p = getattr(fi, "day_low", None) or min(last_price, open_p)
            prev_close = getattr(fi, "previous_close", None) or open_p
            return {
                "last_price": float(last_price),
                "open": float(open_p),
                "high": float(high_p),
                "low": float(low_p),
                "previous_close": float(prev_close),
                "source": "yfinance_realtime",
            }
    except Exception as exc:
        logger.debug("yfinance real-time quote fetch failed for %s: %s", ticker, exc)
    return None


def analyze_intraday_5m_microstructure(
    ticker: str,
    now_dt: datetime | None = None,
) -> dict[str, Any] | None:
    """Analyze real-time 5-minute microstructure from yfinance.

    Computes:
    1. Intraday VWAP (Volume Weighted Average Price) & VWAP deviation
    2. Opening Range (09:00 - 09:30 JST) High/Low and Breakout status
    3. Relative Volume Pace (RVOL) compared to prior 4 trading days at the same time
    """
    try:
        import yfinance as yf

        df = yf.download(ticker, period="5d", interval="5m", progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        latest_date = df.index[-1].date()
        today_df = df[df.index.date == latest_date]
        if today_df.empty:
            return None

        # 1. Intraday VWAP & Deviation
        typical_p = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3.0
        cum_vol = float(today_df["Volume"].sum())
        if cum_vol > 0:
            vwap = float((typical_p * today_df["Volume"]).sum() / cum_vol)
        else:
            vwap = float(today_df["Close"].iloc[-1])

        latest_price = float(today_df["Close"].iloc[-1])
        open_price = float(today_df["Open"].iloc[0])
        day_high = float(today_df["High"].max())
        day_low = float(today_df["Low"].min())
        vwap_dev_pct = ((latest_price - vwap) / vwap) * 100.0 if vwap > 0 else 0.0
        is_above_vwap = latest_price >= vwap

        # 2. Opening Range (09:00 - 09:30) High / Low & Breakout
        orb_bars = today_df[
            (today_df.index.time >= datetime.strptime("09:00", "%H:%M").time())
            & (today_df.index.time <= datetime.strptime("09:30", "%H:%M").time())
        ]
        if orb_bars.empty:
            orb_bars = today_df.head(6)

        orb_high = float(orb_bars["High"].max()) if not orb_bars.empty else day_high
        orb_low = float(orb_bars["Low"].min()) if not orb_bars.empty else day_low

        if latest_price > orb_high:
            orb_status = "朝イチ高値ブレイク（順張り加速）"
            orb_code = "BREAK_HIGH"
        elif latest_price < orb_low:
            orb_status = "朝イチ安値割れ（寄り天崩壊）"
            orb_code = "BREAK_LOW"
        else:
            orb_status = "朝イチレンジ内（揉み合い推移）"
            orb_code = "RANGE"

        # 3. Relative Volume Pace (RVOL)
        cur_time = today_df.index[-1].time()
        past_df = df[df.index.date < latest_date]
        past_dates = np.unique(past_df.index.date)
        past_cum_vols: list[float] = []
        for p_date in past_dates:
            day_slice = past_df[(past_df.index.date == p_date) & (past_df.index.time <= cur_time)]
            if not day_slice.empty:
                past_cum_vols.append(float(day_slice["Volume"].sum()))

        if past_cum_vols:
            mean_past_vol = sum(past_cum_vols) / len(past_cum_vols)
            rvol = cum_vol / mean_past_vol if mean_past_vol > 0 else 1.0
        else:
            mean_past_vol = 0.0
            rvol = 1.0

        return {
            "vwap": round(vwap, 1),
            "latest_price": round(latest_price, 1),
            "open_price": round(open_price, 1),
            "day_high": round(day_high, 1),
            "day_low": round(day_low, 1),
            "vwap_deviation_pct": round(vwap_dev_pct, 2),
            "is_above_vwap": is_above_vwap,
            "vwap_status": "VWAPより上（機関投資家の買い支え圏）" if is_above_vwap else "VWAP割れ（戻り売り・失速圏）",
            "orb_high": round(orb_high, 1),
            "orb_low": round(orb_low, 1),
            "orb_status": orb_status,
            "orb_code": orb_code,
            "rvol_intraday": round(rvol, 2),
            "cum_volume": int(cum_vol),
            "as_of_time": str(cur_time),
            "bars_count": len(today_df),
        }
    except Exception as exc:
        logger.debug("5m microstructure analysis failed for %s: %s", ticker, exc)
        return None

def format_price_yen(val: float) -> str:
    """Format yen price with commas, using integer if >= 1000 or rounded to 1 decimal."""
    if val >= 1000.0:
        return f"{round(val):,}円"
    return f"{val:,.1f}円"


WAVEFORM_PATTERNS: dict[str, dict[str, Any]] = {
    "gu_rebound_high": {
        "title": "🚀【GU朝押し ➔ 後場高値引け型】",
        "description": "高く寄り付き、朝イチの利確こなして1日じり高",
        "direction": "UP",
    },
    "gd_v_shape": {
        "title": "📈【GDスタート ➔ 窓埋めV字反発型】",
        "description": "安く寄り付き、寄り底から前場・後場にかけて陽線浮上",
        "direction": "UP",
    },
    "trend_breakout": {
        "title": "🔥【寄り付き突破 ➔ 終日トレンド独歩高型】",
        "description": "押し目を作らず大口買いで上値を追う展開",
        "direction": "UP",
    },
    "yoriten_fade": {
        "title": "⚠️【寄り天（朝イチ高値）➔ 後場ダラ下げ失速型】",
        "description": "朝9時台が天井で、飛びつくと終日含み損になる警戒波形",
        "direction": "DOWN",
    },
    "gd_downtrend": {
        "title": "📉【GD寄り付き ➔ 戻り売り一段安型】",
        "description": "反発が鈍く、後場にかけて安値を掘り下げる陰線波形",
        "direction": "DOWN",
    },
    "box_range": {
        "title": "⚖️【上下往来 ➔ ボックス揉み合い型】",
        "description": "方向感に欠け、高値追い厳禁のレンジ波形",
        "direction": "NEUTRAL",
    },
}


def simulate_intraday_waveform(
    cur_p: float,
    open_p: float,
    high_p: float,
    low_p: float,
    prev_c: float,
    prob_up: float,
    expected_bps: float,
    vol_20: float,
    phase: str,
    clv: float,
    pullback_pct: float,
    risk_flags: list[str],
    now_dt: datetime | None = None,
    micro: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Simulate which of the 6 intraday waveform patterns the stock will draw today,

    and project realistic 09:00 - 15:30 time-slot price bounds and actionable trading zones.
    """
    hl_range = max(1e-6, high_p - low_p)
    intra_ret_pct = ((cur_p - open_p) / open_p) * 100.0 if open_p > 0 else 0.0
    day_ret_pct = ((cur_p - prev_c) / prev_c) * 100.0 if prev_c > 0 else 0.0
    gap_pct = ((open_p - prev_c) / prev_c) * 100.0 if prev_c > 0 else 0.0
    vol_pct = max(0.015, vol_20)

    # 1. Determine Waveform Pattern Type
    if phase == "intraday_live":
        if micro is not None and micro.get("orb_code") == "BREAK_HIGH" and micro.get("is_above_vwap"):
            waveform_key = "trend_breakout"
        elif micro is not None and micro.get("orb_code") == "BREAK_LOW" and not micro.get("is_above_vwap"):
            waveform_key = "yoriten_fade"
        elif any("寄り天" in f for f in risk_flags) or (high_p >= open_p * 1.006 and pullback_pct >= 0.40) or (intra_ret_pct > 0.8 and pullback_pct >= 0.40):
            waveform_key = "yoriten_fade"
        elif open_p < prev_c and cur_p >= open_p and clv >= 0.55:
            waveform_key = "gd_v_shape"
        elif clv <= 0.30 and intra_ret_pct < -0.6:
            waveform_key = "gd_downtrend"
        elif open_p >= prev_c and clv >= 0.80 and intra_ret_pct >= 0.8:
            waveform_key = "trend_breakout"
        elif open_p >= prev_c and clv >= 0.55:
            waveform_key = "gu_rebound_high"
        elif prob_up >= 55.0 and clv >= 0.50:
            waveform_key = "gu_rebound_high"
        elif prob_up <= 44.0:
            waveform_key = "gd_downtrend"
        else:
            waveform_key = "box_range"
    else:
        # Pre-market or post-market quant forecast
        has_risk = any("警戒" in f for f in risk_flags)
        if has_risk or (prob_up < 48.0 and expected_bps <= -15.0 and gap_pct >= 0.0):
            waveform_key = "yoriten_fade"
        elif prob_up >= 58.0 and expected_bps >= 35.0:
            waveform_key = "trend_breakout"
        elif prob_up >= 52.0 and gap_pct >= -0.2:
            waveform_key = "gu_rebound_high"
        elif prob_up >= 48.0 and gap_pct < -0.2:
            waveform_key = "gd_v_shape"
        elif prob_up <= 45.0:
            waveform_key = "gd_downtrend"
        else:
            waveform_key = "box_range"

    wf_meta = WAVEFORM_PATTERNS[waveform_key]
    wf_title = wf_meta["title"]
    wf_desc = wf_meta["description"]

    # 2. Simulate 09:00 - 15:30 Concrete Price Trajectories
    base_ref = prev_c if phase != "intraday_live" else cur_p

    if waveform_key == "gu_rebound_high":
        # 高く寄り付き、朝イチの利確こなして1日じり高
        open_lo = round(prev_c * (1.0 + max(0.003, gap_pct / 100.0)), 1)
        open_hi = round(prev_c * (1.0 + max(0.008, gap_pct / 100.0 + 0.005)), 1)
        dip_lo = round(prev_c * (1.0 - vol_pct * 0.15), 1)
        dip_hi = round(prev_c * (1.0 + vol_pct * 0.20), 1)
        morn_lo = round(prev_c * (1.0 + vol_pct * 0.35), 1)
        morn_hi = round(prev_c * (1.0 + vol_pct * 0.70), 1)
        aft_lo = round(prev_c * (1.0 + vol_pct * 0.55), 1)
        aft_hi = round(prev_c * (1.0 + vol_pct * 0.95), 1)
        close_lo = round(prev_c * (1.0 + vol_pct * 0.80), 1)
        close_hi = round(prev_c * (1.0 + vol_pct * 1.30), 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 +{((open_lo/prev_c)-1)*100:.1f}% GU予想）"
        t_early = f"利確売りを吸収し {format_price_yen(dip_lo)}〜{format_price_yen(dip_hi)} で下ヒゲ押し目形成"
        t_morning = f"押し目完了から反発、{format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} 突破へじり高"
        t_afternoon = f"大口の買い支えが入り {format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} の高値圏維持"
        t_close = f"日中最高値圏 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} での高値引け着地予想"

        if micro is not None and micro.get("is_above_vwap"):
            buy_zone = f"{format_price_yen(micro['vwap'])}（VWAP押し目）〜 {format_price_yen(dip_hi)}"
            stop_loss = round(min(cur_p * (1.0 - max(0.012, vol_pct * 0.45)), micro["vwap"] * 0.995, micro.get("orb_low", dip_lo) * 0.998), 1)
        else:
            buy_zone = f"{format_price_yen(dip_lo)} 〜 {format_price_yen(dip_hi)}"
            stop_loss = round(min(cur_p * (1.0 - max(0.012, vol_pct * 0.50)), prev_c * (1.0 - vol_pct * 0.40)), 1)

        target_1 = morn_hi
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = close_hi
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    elif waveform_key == "gd_v_shape":
        # 安く寄り付き、寄り底から前場・後場にかけて陽線浮上
        open_lo = round(prev_c * (1.0 - max(0.008, abs(gap_pct) / 100.0)), 1)
        open_hi = round(prev_c * (1.0 - max(0.003, abs(gap_pct) / 100.0 * 0.5)), 1)
        dip_lo = round(prev_c * (1.0 - vol_pct * 0.65), 1)
        dip_hi = round(prev_c * (1.0 - vol_pct * 0.25), 1)
        morn_lo = round(prev_c * (1.0 - vol_pct * 0.10), 1)
        morn_hi = round(prev_c * (1.0 + vol_pct * 0.30), 1)
        aft_lo = round(prev_c * (1.0 + vol_pct * 0.20), 1)
        aft_hi = round(prev_c * (1.0 + vol_pct * 0.65), 1)
        close_lo = round(prev_c * (1.0 + vol_pct * 0.40), 1)
        close_hi = round(prev_c * (1.0 + vol_pct * 0.90), 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 -{abs((open_hi/prev_c)-1)*100:.1f}% GDスタート）"
        t_early = f"{format_price_yen(dip_lo)}〜{format_price_yen(dip_hi)} で寄り底を確認し、急速な買い戻し発動"
        t_morning = f"前日終値（{format_price_yen(prev_c)}）の窓埋めを完了し、{format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} へ浮上"
        t_afternoon = f"買い優勢が継続、{format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} でプラス圏定着"
        t_close = f"後場高値水準 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} で陽線着地予想"

        if micro is not None and micro.get("is_above_vwap"):
            buy_zone = f"{format_price_yen(micro['vwap'])}（VWAP浮上）〜 {format_price_yen(dip_hi)}"
            stop_loss = round(min(cur_p * (1.0 - max(0.012, vol_pct * 0.45)), micro["vwap"] * 0.995, dip_lo * 0.995), 1)
        else:
            buy_zone = f"{format_price_yen(dip_lo)} 〜 {format_price_yen(dip_hi)}"
            stop_loss = round(min(cur_p * (1.0 - max(0.012, vol_pct * 0.50)), dip_lo * 0.995), 1)

        target_1 = morn_hi
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = close_hi
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    elif waveform_key == "trend_breakout":
        # 押し目を作らず大口買いで上値を追う展開
        open_lo = round(prev_c * (1.0 + max(0.005, gap_pct / 100.0)), 1)
        open_hi = round(prev_c * (1.0 + max(0.012, gap_pct / 100.0 + 0.006)), 1)
        dip_lo = round(prev_c * (1.0 + vol_pct * 0.30), 1)
        dip_hi = round(prev_c * (1.0 + vol_pct * 0.65), 1)
        morn_lo = round(prev_c * (1.0 + vol_pct * 0.70), 1)
        morn_hi = round(prev_c * (1.0 + vol_pct * 1.20), 1)
        aft_lo = round(prev_c * (1.0 + vol_pct * 1.10), 1)
        aft_hi = round(prev_c * (1.0 + vol_pct * 1.65), 1)
        close_lo = round(prev_c * (1.0 + vol_pct * 1.40), 1)
        close_hi = round(prev_c * (1.0 + vol_pct * 2.00), 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 +{((open_lo/prev_c)-1)*100:.1f}% 強気スタート）"
        t_early = f"押し目を待たずに {format_price_yen(dip_lo)}〜{format_price_yen(dip_hi)} を一気にブレイク突破"
        t_morning = f"大口の買い上がりが加速、{format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} へ一段高"
        t_afternoon = f"売り方の踏み上げを巻き込み {format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} へ独歩高"
        t_close = f"当日最高値 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} での大陽線引け予想"

        if micro is not None and micro.get("orb_high"):
            buy_zone = f"{format_price_yen(micro.get('vwap', open_lo))}（VWAP押し目）〜 {format_price_yen(micro['orb_high'])}（ORBブレイク）"
            stop_loss = round(min(cur_p * (1.0 - max(0.010, vol_pct * 0.40)), micro.get("vwap", open_lo) * 0.995, micro.get("orb_low", open_lo)), 1)
        else:
            buy_zone = f"{format_price_yen(open_lo)} 〜 {format_price_yen(dip_lo)}"
            stop_loss = round(min(cur_p * (1.0 - max(0.010, vol_pct * 0.40)), open_lo * 0.995), 1)

        target_1 = morn_hi
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = close_hi
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    elif waveform_key == "yoriten_fade":
        # 朝9時台が天井で、飛びつくと終日含み損になる警戒波形
        open_lo = round(prev_c * (1.0 + max(0.004, gap_pct / 100.0)), 1)
        open_hi = round(prev_c * (1.0 + max(0.012, gap_pct / 100.0 + 0.005)), 1)
        peak_lo = round(prev_c * (1.0 + vol_pct * 0.30), 1)
        peak_hi = round(prev_c * (1.0 + vol_pct * 0.65), 1)
        morn_lo = round(prev_c * (1.0 - vol_pct * 0.25), 1)
        morn_hi = round(prev_c * (1.0 + vol_pct * 0.05), 1)
        aft_lo = round(prev_c * (1.0 - vol_pct * 0.65), 1)
        aft_hi = round(prev_c * (1.0 - vol_pct * 0.20), 1)
        close_lo = round(prev_c * (1.0 - vol_pct * 0.95), 1)
        close_hi = round(prev_c * (1.0 - vol_pct * 0.50), 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 +{((open_lo/prev_c)-1)*100:.1f}% 高寄り見かけ）"
        t_early = f"{format_price_yen(peak_lo)}〜{format_price_yen(peak_hi)} で朝イチ高値をつけ、大量利確売りに押され失速"
        t_morning = f"始値割れから前日終値割れへ沈下、{format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} へ失速"
        t_afternoon = f"買い支えが入らず戻り売り優勢、{format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} へ下値切り下げ"
        t_close = f"安値圏 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} での陰線着地警戒"

        if micro is not None:
            buy_zone = f"⚠️ 買い厳禁（戻り売りゾーン: VWAP {format_price_yen(micro['vwap'])} 〜 朝イチ高値 {format_price_yen(micro['orb_high'])}）"
            stop_loss = round(max(cur_p * 1.012, micro["vwap"] * 1.005, micro["orb_high"] * 1.004), 1)
        else:
            buy_zone = f"⚠️ 買い厳禁（戻り売りゾーン: {format_price_yen(peak_lo)} 〜 {format_price_yen(peak_hi)}）"
            stop_loss = round(max(cur_p * 1.012, peak_hi * 1.004), 1)

        target_1 = morn_lo
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = close_lo
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    elif waveform_key == "gd_downtrend":
        # 反発が鈍く、後場にかけて安値を掘り下げる陰線波形
        open_lo = round(prev_c * (1.0 - max(0.012, abs(gap_pct) / 100.0)), 1)
        open_hi = round(prev_c * (1.0 - max(0.004, abs(gap_pct) / 100.0 * 0.5)), 1)
        dip_lo = round(prev_c * (1.0 - vol_pct * 0.65), 1)
        dip_hi = round(prev_c * (1.0 - vol_pct * 0.25), 1)
        morn_lo = round(prev_c * (1.0 - vol_pct * 0.95), 1)
        morn_hi = round(prev_c * (1.0 - vol_pct * 0.50), 1)
        aft_lo = round(prev_c * (1.0 - vol_pct * 1.30), 1)
        aft_hi = round(prev_c * (1.0 - vol_pct * 0.80), 1)
        close_lo = round(prev_c * (1.0 - vol_pct * 1.60), 1)
        close_hi = round(prev_c * (1.0 - vol_pct * 1.10), 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 -{abs((open_hi/prev_c)-1)*100:.1f}% GD寄り付き）"
        t_early = f"自律反発が極めて弱く {format_price_yen(dip_lo)}〜{format_price_yen(dip_hi)} で下値模索"
        t_morning = f"前場安値を下回り {format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} へ一段安"
        t_afternoon = f"戻り待ちの投げ売りが重なり {format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} へ下落加速"
        t_close = f"最安値圏 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} での大陰線警戒"

        if micro is not None:
            buy_zone = f"⚠️ 買いエントリー完全見送り（VWAP {format_price_yen(micro['vwap'])} 割れ逆張り厳禁）"
            stop_loss = round(max(cur_p * 1.015, micro["vwap"] * 1.005), 1)
        else:
            buy_zone = "⚠️ 買いエントリー完全見送り（逆張り厳禁）"
            stop_loss = round(max(cur_p * 1.015, open_hi), 1)

        target_1 = morn_lo
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = close_lo
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    else:  # box_range
        # 方向感に欠け、高値追い厳禁のレンジ波形
        open_lo = round(prev_c * 0.998, 1)
        open_hi = round(prev_c * 1.004, 1)
        box_lo = round(prev_c * (1.0 - vol_pct * 0.40), 1)
        box_hi = round(prev_c * (1.0 + vol_pct * 0.40), 1)
        morn_lo = round(prev_c * (1.0 - vol_pct * 0.25), 1)
        morn_hi = round(prev_c * (1.0 + vol_pct * 0.25), 1)
        aft_lo = round(prev_c * (1.0 - vol_pct * 0.30), 1)
        aft_hi = round(prev_c * (1.0 + vol_pct * 0.30), 1)
        close_lo = round(prev_c * 0.996, 1)
        close_hi = round(prev_c * 1.004, 1)

        t_open = f"{format_price_yen(open_lo)} 〜 {format_price_yen(open_hi)}（前日比 ±0.2% 前日並み）"
        t_early = f"{format_price_yen(box_lo)}〜{format_price_yen(box_hi)} のレンジ上限・下限を確認"
        t_morning = f"{format_price_yen(morn_lo)}〜{format_price_yen(morn_hi)} のボックス中央値で揉み合い"
        t_afternoon = f"出来高細り {format_price_yen(aft_lo)}〜{format_price_yen(aft_hi)} の膠着状態"
        t_close = f"前日終値近辺 {format_price_yen(close_lo)}〜{format_price_yen(close_hi)} で着地予想"

        if micro is not None:
            buy_zone = f"{format_price_yen(micro.get('vwap', box_lo))} 付近（VWAP押し目・レンジ下限打診）"
            stop_loss = round(min(cur_p * 0.992, micro.get("orb_low", box_lo) * 0.995), 1)
        else:
            buy_zone = f"{format_price_yen(box_lo)} 付近（レンジ下限打診）"
            stop_loss = round(min(cur_p * 0.992, box_lo * 0.994), 1)

        target_1 = max(round(cur_p * 1.008, 1), box_hi)
        target_1_pct = round(((target_1 - cur_p) / cur_p) * 100.0, 1)
        target_final = max(round(cur_p * 1.015, 1), round(box_hi * 1.002, 1))
        target_final_pct = round(((target_final - cur_p) / cur_p) * 100.0, 1)
        stop_loss_pct = round(((stop_loss - cur_p) / cur_p) * 100.0, 1)

    # 3. Format Intraday Live Dynamic Trajectory when market is actively trading
    if phase == "intraday_live":
        if micro is not None:
            t_actual = (
                f"始値 {format_price_yen(open_p)} ➔ 現在 {format_price_yen(cur_p)}（始値比 {intra_ret_pct:+.2f}%）"
                f" | VWAP {format_price_yen(micro['vwap'])}（乖離 {micro['vwap_deviation_pct']:+.1f}%）"
                f" | 30分ORB {format_price_yen(micro['orb_low'])}〜{format_price_yen(micro['orb_high'])}（{micro['orb_status']}）"
            )
        else:
            t_actual = f"始値 {format_price_yen(open_p)} ➔ 安値 {format_price_yen(low_p)} / 高値 {format_price_yen(high_p)} ➔ 現在値 {format_price_yen(cur_p)}（始値比 {intra_ret_pct:+.2f}%）"

        timeline_items = [
            ("🕒【09:00〜現在 実績】", t_actual),
            ("🕚【10:00〜11:30 前場】", t_morning),
            ("🕜【12:30〜14:30 後場】", t_afternoon),
            ("🏁【15:00〜15:30 大引け】", t_close),
        ]
    else:
        timeline_items = [
            ("🌅【09:00 寄付き予想】", t_open),
            ("⚡【09:00〜09:45 初動】", t_early),
            ("🕚【10:00〜11:30 前場】", t_morning),
            ("🕜【12:30〜14:30 後場】", t_afternoon),
            ("🏁【15:00〜15:30 大引け】", t_close),
        ]

    return {
        "waveform_key": waveform_key,
        "waveform_title": wf_title,
        "waveform_description": wf_desc,
        "direction": wf_meta["direction"],
        "timeline_items": timeline_items,
        "best_entry_zone": buy_zone,
        "target_1_price": target_1,
        "target_1_pct": target_1_pct,
        "target_final_price": target_final,
        "target_final_pct": target_final_pct,
        "stop_loss_price": stop_loss,
        "stop_loss_pct": stop_loss_pct,
    }


def predict_daytrade_sniper(
    ticker: str,
    settings: Settings | None = None,
    now_dt: datetime | None = None,
    name: str = "",
    fetch_5m_micro: bool = True,
) -> dict[str, Any] | None:
    """Predict strictly 'How will the price trajectory and waveform behave TODAY (09:00 - 15:30)'.

    Integrates:
    1. 07:30 JST pre-market pooled intraday model (lightgbm_intraday_pooled_cs)
    2. Real-time intraday quote & 5-minute microstructure (VWAP, 30m ORB Break, RVOL)
    3. 6-pattern waveform classification and 09:00-15:30 concrete price trajectory simulation
    """
    if settings is None:
        settings = load_settings()

    phase, phase_title = determine_market_phase(now_dt)
    base_pred = predict_intraday_for_ticker(ticker, settings=settings)
    if not base_pred:
        return None

    base_prob = float(base_pred.get("p_intraday_cs", 0.50))
    clean_prob = float(base_pred.get("p_clean_win", base_prob))
    meta_prob = float(base_pred.get("p_afml_meta", clean_prob))
    bet_size = float(base_pred.get("bet_size", 0.50))
    ensemble_prob = float(base_pred.get("p_intraday_ensemble", base_prob))
    base_bps = float(base_pred.get("expected_return_bps", 0.0))
    q_band = str(base_pred.get("quantile", "Q3"))

    if bet_size >= 0.70:
        lot_recommendation = "🔥 推奨ロット: 100%（AFML強気フルサイズ）"
    elif bet_size >= 0.25:
        lot_recommendation = f"⚡ 推奨ロット: {int(round(bet_size * 100))}%（AFML打診ポジション）"
    else:
        lot_recommendation = "⚠️ 推奨ロット: 0%（AFMLリスク回避・ノーポジ推奨）"

    # Load technical features for volatility and fallback
    f_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    vol_20 = 0.022
    features_df = None
    if f_path.exists():
        try:
            features_df = read_parquet(f_path)
            if not features_df.empty and "volatility_20" in features_df.columns:
                v_val = float(features_df["volatility_20"].iloc[-1])
                if not np.isnan(v_val) and v_val > 0.005:
                    vol_20 = v_val
        except Exception:
            pass

    # Real-time quote or fallback to features parquet
    quote = fetch_realtime_intraday_quote(ticker)
    if quote is None and features_df is not None and not features_df.empty:
        last_row = features_df.iloc[-1]
        c = float(last_row.get("close", 1000.0))
        o = float(last_row.get("open", c))
        h = float(last_row.get("high", max(c, o)))
        l_p = float(last_row.get("low", min(c, o)))
        pc = float(features_df["close"].iloc[-2]) if len(features_df) >= 2 else o
        quote = {
            "last_price": c,
            "open": o,
            "high": h,
            "low": l_p,
            "previous_close": pc,
            "source": "parquet_fallback",
        }
    elif quote is None:
        quote = {
            "last_price": 1000.0,
            "open": 1000.0,
            "high": 1000.0,
            "low": 1000.0,
            "previous_close": 1000.0,
            "source": "default_mock",
        }

    # Analyze 5-minute Microstructure (VWAP, 30m ORB, RVOL) if enabled
    micro = analyze_intraday_5m_microstructure(ticker, now_dt=now_dt) if fetch_5m_micro else None
    if micro is not None and phase == "intraday_live":
        # Override with exact latest bar data from 5m stream
        cur_p = micro["latest_price"]
        open_p = micro["open_price"]
        high_p = micro["day_high"]
        low_p = micro["day_low"]
    else:
        cur_p = quote["last_price"]
        open_p = quote["open"]
        high_p = quote["high"]
        low_p = quote["low"]

    prev_c = quote["previous_close"]

    day_ret_pct = ((cur_p - prev_c) / prev_c) * 100.0 if prev_c > 0 else 0.0
    intra_ret_pct = ((cur_p - open_p) / open_p) * 100.0 if open_p > 0 else 0.0
    hl_range = max(1e-6, high_p - low_p)
    clv = float(np.clip((cur_p - low_p) / hl_range, 0.0, 1.0))
    pullback_pct = float(np.clip((high_p - cur_p) / hl_range, 0.0, 1.0))

    # Phase-adaptive probability and momentum fusion
    mom_adj = 0.0
    risk_flags: list[str] = []

    # Check earnings risk
    e_risk, _ = check_earnings_risk(ticker)
    if e_risk:
        risk_flags.append("⚠️ 直近決算リスク")

    if phase == "intraday_live":
        if micro is not None:
            # 5-minute Microstructure Signals
            if micro["is_above_vwap"] and micro["orb_code"] == "BREAK_HIGH":
                mom_adj += 0.12
                risk_flags.append(f"⚡ VWAP上＋朝イチ30分高値ブレイク（{format_price_yen(micro['orb_high'])}突破）")
            elif not micro["is_above_vwap"] and micro["orb_code"] == "BREAK_LOW":
                mom_adj -= 0.15
                risk_flags.append(f"⚠️ VWAP割れ＋朝イチ30分安値割れ（{format_price_yen(micro['orb_low'])}割れ寄り天）")
            elif micro["is_above_vwap"]:
                mom_adj += 0.05
                risk_flags.append(f"🛡️ VWAP買い支え圏（乖離 {micro['vwap_deviation_pct']:+.1f}%）")
            else:
                mom_adj -= 0.05
                risk_flags.append(f"⚠️ VWAP下抜け警戒（乖離 {micro['vwap_deviation_pct']:+.1f}%）")

            if micro["rvol_intraday"] >= 1.4:
                mom_adj += 0.04
                risk_flags.append(f"🔥 大口資金流入（出来高ペース {micro['rvol_intraday']:.1f}倍）")
        else:
            if clv >= 0.70 and intra_ret_pct > 0:
                mom_adj = +0.10
            elif intra_ret_pct > 1.2 and pullback_pct >= 0.50:
                mom_adj = -0.15
                risk_flags.append("⚠️ 寄り天・上ヒゲ失速警戒")
            elif intra_ret_pct < -0.8 and clv >= 0.60:
                mom_adj = +0.07
            elif intra_ret_pct < -0.8 and clv <= 0.30:
                mom_adj = -0.12
                risk_flags.append("⚠️ 安値引け警戒")

        eff_prob = float(np.clip(ensemble_prob + mom_adj, 0.05, 0.95))
        eff_bps = base_bps + (mom_adj * 10000.0 * 0.4)
    elif phase == "premarket":
        eff_prob = ensemble_prob
        eff_bps = base_bps
        if q_band in ("Q1", "Q2") or base_bps <= -25.0:
            risk_flags.append("⚠️ 寄り天・陰線警戒")
    else:  # post_market
        eff_prob = ensemble_prob
        eff_bps = base_bps

    prob_up_pct = round(eff_prob * 100.0, 1)

    # Clear Black-or-White Judgment
    if prob_up_pct >= 55.0:
        if phase == "intraday_live":
            judgment = "🟢 今日上がる [後場買い・大引け高値期待]"
        elif phase == "premarket":
            judgment = "🟢 今日上がる [買い推奨 / 陽線期待]"
        else:
            judgment = "🟢 翌営業日上がる [買いエントリー検討]"
        judge_code = "BUY"
    elif prob_up_pct <= 45.0:
        if phase == "intraday_live":
            judgment = "🔴 今日下がる [売り・失速警戒・見送り]"
        elif phase == "premarket":
            judgment = "🔴 今日下がる [売り・見送り / 陰線警戒]"
        else:
            judgment = "🔴 翌営業日下がる [警戒・見送り]"
        judge_code = "SELL"
    else:
        judgment = "⚪ レンジ拮抗 [揉み合い・様子見]"
        judge_code = "NEUTRAL"

    # Simulate waveform pattern and 09:00 - 15:30 timeline
    waveform_data = simulate_intraday_waveform(
        cur_p=cur_p,
        open_p=open_p,
        high_p=high_p,
        low_p=low_p,
        prev_c=prev_c,
        prob_up=prob_up_pct,
        expected_bps=eff_bps,
        vol_20=vol_20,
        phase=phase,
        clv=clv,
        pullback_pct=pullback_pct,
        risk_flags=risk_flags,
        now_dt=now_dt,
        micro=micro,
    )

    return {
        "ticker": ticker,
        "name": name or ticker,
        "phase": phase,
        "phase_title": phase_title,
        "current_price": cur_p,
        "open_price": open_p,
        "high_price": high_p,
        "low_price": low_p,
        "previous_close": prev_c,
        "day_return_pct": round(day_ret_pct, 2),
        "intraday_return_pct": round(intra_ret_pct, 2),
        "clv": round(clv, 2),
        "pullback_pct": round(pullback_pct, 2),
        "probability_up": prob_up_pct,
        "p_intraday_cs": round(base_prob * 100.0, 1),
        "p_clean_win": round(clean_prob * 100.0, 1),
        "p_afml_meta": round(meta_prob * 100.0, 1),
        "bet_size": round(bet_size, 2),
        "lot_recommendation": lot_recommendation,
        "expected_return_bps": round(eff_bps, 1),
        "judgment": judgment,
        "judgment_code": judge_code,
        "q_band": q_band,
        "volatility_20": vol_20,
        "risk_flags": risk_flags,
        "quote_source": quote.get("source", "unknown"),
        # Microstructure Analysis Data
        "microstructure": micro,
        "vwap": micro.get("vwap") if micro else None,
        "vwap_deviation_pct": micro.get("vwap_deviation_pct") if micro else None,
        "is_above_vwap": micro.get("is_above_vwap") if micro else None,
        "orb_high": micro.get("orb_high") if micro else None,
        "orb_low": micro.get("orb_low") if micro else None,
        "orb_status": micro.get("orb_status") if micro else None,
        "rvol_intraday": micro.get("rvol_intraday") if micro else None,
        # Waveform & Timeline Engine Data
        "waveform_key": waveform_data["waveform_key"],
        "waveform_title": waveform_data["waveform_title"],
        "waveform_description": waveform_data["waveform_description"],
        "timeline_items": waveform_data["timeline_items"],
        "best_entry_zone": waveform_data["best_entry_zone"],
        "target_1_price": waveform_data["target_1_price"],
        "target_1_pct": waveform_data["target_1_pct"],
        "target_final_price": waveform_data["target_final_price"],
        "target_final_pct": waveform_data["target_final_pct"],
        "stop_loss_price": waveform_data["stop_loss_price"],
        "stop_loss_pct": waveform_data["stop_loss_pct"],
        # Backwards compatible alias keys
        "target_price": waveform_data["target_final_price"],
        "target_pct": waveform_data["target_final_pct"],
        "support_price": waveform_data["target_1_price"],
        "stop_loss_pct": waveform_data["stop_loss_pct"],
        "tactical_comment": waveform_data["waveform_description"],
    }


def get_daily_daytrade_rankings(
    universe: list[TickerInfo] | None = None,
    settings: Settings | None = None,
    top_n: int = 5,
    now_dt: datetime | None = None,
) -> dict[str, Any]:
    """Scan all 50 universe tickers and generate Top 5 Up (buy candidates) and Top 5 Down (risk candidates).

    Optimized: Performs fast scan first, then enriches top focus tickers with 5m microstructure.
    """
    if universe is None:
        universe = load_universe()
    if settings is None:
        settings = load_settings()

    # 1. Fast screening scan across all 50 tickers
    results: list[dict[str, Any]] = []
    for t in universe:
        res = predict_daytrade_sniper(t.code, settings=settings, now_dt=now_dt, name=t.name, fetch_5m_micro=False)
        if res:
            res["sector"] = getattr(t, "sector_ja", "") or getattr(t, "sector", "") or "-"
            results.append(res)

    results.sort(key=lambda x: (x["probability_up"], x["expected_return_bps"]), reverse=True)
    raw_top_up = results[:top_n]
    raw_top_down = sorted(results, key=lambda x: (x["probability_up"], x["expected_return_bps"]))[:top_n]

    # 2. Enrich selected Top Up & Top Down with real-time 5m microstructure
    top_up: list[dict[str, Any]] = []
    for r in raw_top_up:
        enriched = predict_daytrade_sniper(r["ticker"], settings=settings, now_dt=now_dt, name=r["name"], fetch_5m_micro=True)
        if enriched:
            enriched["sector"] = r.get("sector", "-")
            top_up.append(enriched)
        else:
            top_up.append(r)

    top_down: list[dict[str, Any]] = []
    for r in raw_top_down:
        enriched = predict_daytrade_sniper(r["ticker"], settings=settings, now_dt=now_dt, name=r["name"], fetch_5m_micro=True)
        if enriched:
            enriched["sector"] = r.get("sector", "-")
            top_down.append(enriched)
        else:
            top_down.append(r)

    phase, phase_title = determine_market_phase(now_dt)
    date_str = (now_dt or datetime.now()).strftime("%Y-%m-%d")

    return {
        "date": date_str,
        "phase": phase,
        "phase_title": phase_title,
        "top_up": top_up,
        "top_down": top_down,
        "total_scanned": len(results),
    }

