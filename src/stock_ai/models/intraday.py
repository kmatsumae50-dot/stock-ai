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
    """Train a leak-free Pooled Cross-Sectional Intraday model (Classifier + Regressor bundle).

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

    feature_columns = [c for c in ALL_INTRADAY_FEATURES if c in all_df.columns]

    split_dt = pd.to_datetime(train_end_date)
    train_mask = all_df["date"] < split_dt
    train_subset = all_df[train_mask].dropna(subset=feature_columns + [INTRADAY_TARGET, "intraday_return"])

    if train_subset.empty:
        raise ValueError("Training subset for Intraday model is empty.")

    # Validation window: last 15% of train dates
    train_dates = sorted(train_subset["date"].unique())
    valid_start_idx = max(1, int(len(train_dates) * 0.85))
    valid_start_dt = train_dates[valid_start_idx]

    train_data = train_subset[train_subset["date"] < valid_start_dt]
    valid_data = train_subset[train_subset["date"] >= valid_start_dt]

    # 1. Train Classifier for Cross-Sectional Outperform
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
        eval_set=[(valid_data[feature_columns], valid_data[INTRADAY_TARGET].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    val_probs = clf.predict_proba(valid_data[feature_columns])[:, 1]
    val_auc = float(roc_auc_score(valid_data[INTRADAY_TARGET].astype(int), val_probs))
    val_acc = float(accuracy_score(valid_data[INTRADAY_TARGET].astype(int), (val_probs >= 0.5).astype(int)))

    # Refit classifier on full training set
    clf.fit(train_subset[feature_columns], train_subset[INTRADAY_TARGET].astype(int))

    # 2. Train Regressor for expected return hurdle
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
    reg.fit(train_subset[feature_columns], train_subset["intraday_return"])

    metrics_info = {
        "model_name": INTRADAY_MODEL_NAME,
        "target": INTRADAY_TARGET,
        "trained_samples": len(train_data),
        "validation_samples": len(valid_data),
        "validation_auc": val_auc,
        "validation_accuracy": val_acc,
        "feature_columns_count": len(feature_columns),
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
        "regressor": reg,
        "feature_columns": feature_columns,
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
    """Predict intraday open-to-close probability and expected return for a ticker."""
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

    clf = bundle["model"]
    reg = bundle.get("regressor")

    prob = float(clf.predict_proba(X_row)[:, 1][0])
    pred_ret = float(reg.predict(X_row)[0]) if reg is not None else 0.0

    info = classify_intraday_quantile(prob, pred_ret=pred_ret)
    info["ticker"] = ticker
    info["date"] = str(row_df["date"].iloc[0]) if "date" in row_df.columns else "-"
    info["p_intraday_cs"] = prob
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


def predict_daytrade_sniper(
    ticker: str,
    settings: Settings | None = None,
    now_dt: datetime | None = None,
    name: str = "",
) -> dict[str, Any] | None:
    """Predict strictly 'Will it go Up or Down TODAY (or next 1-Day)' with targets and tactics.

    Integrates:
    1. 07:30 JST pre-market pooled intraday model (lightgbm_intraday_pooled_cs)
    2. Real-time intraday quote (yfinance Open, High, Low, Current CLV, Momentum)
    3. Concrete intraday volatility targets and stop-loss boundaries
    """
    if settings is None:
        settings = load_settings()

    phase, phase_title = determine_market_phase(now_dt)
    base_pred = predict_intraday_for_ticker(ticker, settings=settings)
    if not base_pred:
        return None

    base_prob = float(base_pred.get("p_intraday_cs", 0.50))
    base_bps = float(base_pred.get("expected_return_bps", 0.0))
    q_band = str(base_pred.get("quantile", "Q3"))

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
        # Live momentum fusion
        if clv >= 0.70 and intra_ret_pct > 0:
            mom_adj = +0.10
            tactic = (
                f"高値圏を維持（CLV {clv:.0%}、始値比 {intra_ret_pct:+.1f}%）。大引けにかけて踏み上げ高値引けが期待できる強いチャート形状。"
                "高値ブレイクに乗るかVWAP近辺の押し目で順張りエントリー推奨。"
            )
        elif intra_ret_pct > 1.2 and pullback_pct >= 0.50:
            mom_adj = -0.15
            risk_flags.append("⚠️ 寄り天・上ヒゲ失速警戒")
            tactic = (
                f"⚠️ 寄り天・上ヒゲ失速パターンを検知！寄付き急騰後に戻り売りに押され（上ヒゲ押し戻し {pullback_pct:.0%}）、安値方向への失速リスク大。"
                "買いエントリーは厳禁、見送り推奨。"
            )
        elif intra_ret_pct < -0.8 and clv >= 0.60:
            mom_adj = +0.07
            tactic = (
                f"前場安値（{low_p:,.0f}円）から底打ち反転中（安値から回復 CLV {clv:.0%}）。後場のリバウンド押し目買いが機能しており、"
                f"前場安値（{low_p:,.0f}円）を背にしたデイトレ逆張り買いが有効。"
            )
        elif intra_ret_pct < -0.8 and clv <= 0.30:
            mom_adj = -0.12
            risk_flags.append("⚠️ 安値引け警戒")
            tactic = (
                f"安値圏を這う下落トレンド（CLV {clv:.0%}、始値比 {intra_ret_pct:+.1f}%）。買い圧力が弱く後場一段安の恐れあり。"
                "安易なナンピン買いは危険、見送り推奨。"
            )
        else:
            mom_adj = 0.0
            tactic = (
                f"日中レンジ揉み合い（現在値 {cur_p:,.0f}円、始値比 {intra_ret_pct:+.1f}%）。上下の方向感に乏しいため、"
                "明確なブレイクアウトまたはレンジ下限での反発確認まで待機推奨。"
            )
        eff_prob = float(np.clip(base_prob + mom_adj, 0.05, 0.95))
        eff_bps = base_bps + (mom_adj * 10000.0 * 0.4)
    elif phase == "premarket":
        eff_prob = base_prob
        eff_bps = base_bps
        if q_band in ("Q4", "Q5") and base_bps >= 25.0:
            tactic = (
                f"寄前モデル強気ゾーン（{q_band} / 期待値 {base_bps:+.0f}bps）。寄付きから大引けにかけて陽線上昇が期待できる高確信ゾーン。"
                "前場初動の押し目でエントリー推奨。"
            )
        elif q_band in ("Q1", "Q2") or base_bps <= -25.0:
            risk_flags.append("⚠️ 寄り天・陰線警戒")
            tactic = (
                f"寄り天・陰線警戒ゾーン（{q_band} / 期待値 {base_bps:+.0f}bps）。寄付き後の利確売り・戻り売りに押されやすく、ロングエントリーは見送り推奨。"
            )
        else:
            tactic = (
                f"中立推移ゾーン（{q_band} / 期待値 {base_bps:+.0f}bps）。上下方向感が定まらないため無理なエントリーは避け、明確なトレンド発生を待つのが賢明。"
            )
    else:  # post_market
        eff_prob = base_prob
        eff_bps = base_bps
        tactic = (
            "大引け後の翌営業日スナイパー判定。翌朝寄付きから15:30大引けまでのトータル方向感を診断。"
            "今夜の米国市場確定値を反映し、明朝07:30寄前に最終確定します。"
        )

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

    # Dynamic target & stop-loss bounds from 20-day volatility
    expected_range_pct = max(0.012, vol_20 * 1.0)
    target_price = round(cur_p * (1.0 + expected_range_pct), 1)
    target_pct = round(expected_range_pct * 100.0, 2)
    support_price = round(cur_p * (1.0 - expected_range_pct * 0.70), 1)
    stop_loss_price = round(cur_p * (1.0 - expected_range_pct * 0.85), 1)
    stop_loss_pct = round(-expected_range_pct * 0.85 * 100.0, 2)

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
        "expected_return_bps": round(eff_bps, 1),
        "judgment": judgment,
        "judgment_code": judge_code,
        "target_price": target_price,
        "target_pct": target_pct,
        "support_price": support_price,
        "stop_loss_price": stop_loss_price,
        "stop_loss_pct": stop_loss_pct,
        "tactical_comment": tactic,
        "q_band": q_band,
        "volatility_20": vol_20,
        "risk_flags": risk_flags,
        "quote_source": quote.get("source", "unknown"),
    }


def get_daily_daytrade_rankings(
    universe: list[TickerInfo] | None = None,
    settings: Settings | None = None,
    top_n: int = 5,
    now_dt: datetime | None = None,
) -> dict[str, Any]:
    """Scan all 50 universe tickers and generate Top 5 Up (buy candidates) and Top 5 Down (risk candidates)."""
    if universe is None:
        universe = load_universe()
    if settings is None:
        settings = load_settings()

    results: list[dict[str, Any]] = []
    for t in universe:
        res = predict_daytrade_sniper(t.code, settings=settings, now_dt=now_dt, name=t.name)
        if res:
            res["sector"] = getattr(t, "sector_ja", "") or getattr(t, "sector", "") or "-"
            results.append(res)

    results.sort(key=lambda x: (x["probability_up"], x["expected_return_bps"]), reverse=True)
    top_up = results[:top_n]
    top_down = sorted(results, key=lambda x: (x["probability_up"], x["expected_return_bps"]))[:top_n]

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

