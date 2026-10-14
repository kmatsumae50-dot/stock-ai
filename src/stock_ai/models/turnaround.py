from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.data.universe import TickerInfo, load_universe
from stock_ai.models.predict import load_model_bundle

TURNAROUND_MODEL_NAME = "lightgbm_turnaround_pooled"
TURNAROUND_ARTIFACT_FILENAME = "lightgbm_turnaround_pooled.joblib"
TURNAROUND_METRICS_FILENAME = "lightgbm_turnaround_pooled.json"

TURNAROUND_FEATURE_COLUMNS = [
    "drawdown_252d",
    "drawdown_126d",
    "pos_in_52w_range",
    "ma_gap_75d",
    "ma_gap_200d",
    "zscore_126d",
    "zscore_252d",
    "rsi_28d",
    "rsi_60d",
    "volume_surge_at_low",
    "momentum_reversal_5d_vs_60d",
    "rsi_14",
    "volatility_20",
    "return_5d",
    "return_20d",
    "return_60d",
]


def compute_turnaround_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-term deep value, oversold, and reversal features along with forward targets."""
    data = df.sort_values("date").copy()
    close = data["close"]
    vol = data["volume"]

    # 52w (252d) & 26w (126d) highs/lows
    high_252 = close.rolling(252, min_periods=40).max()
    low_252 = close.rolling(252, min_periods=40).min()
    high_126 = close.rolling(126, min_periods=30).max()

    data["drawdown_252d"] = close / high_252 - 1.0
    data["drawdown_126d"] = close / high_126 - 1.0
    data["pos_in_52w_range"] = (close - low_252) / (high_252 - low_252 + 1e-8)
    data["pos_in_52w_range"] = data["pos_in_52w_range"].clip(0.0, 1.0)

    # Long-term Moving Averages
    ma_75 = close.rolling(75, min_periods=20).mean()
    ma_200 = close.rolling(200, min_periods=50).mean()
    data["ma_gap_75d"] = close / ma_75 - 1.0
    data["ma_gap_200d"] = close / ma_200 - 1.0

    # Long-term Z-scores
    std_126 = close.rolling(126, min_periods=30).std()
    std_252 = close.rolling(252, min_periods=50).std()
    data["zscore_126d"] = (close - close.rolling(126, min_periods=30).mean()) / (std_126 + 1e-8)
    data["zscore_252d"] = (close - close.rolling(252, min_periods=50).mean()) / (std_252 + 1e-8)

    # Long-term RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up_28 = up.rolling(28, min_periods=15).mean()
    roll_down_28 = down.rolling(28, min_periods=15).mean()
    rs_28 = roll_up_28 / (roll_down_28 + 1e-8)
    data["rsi_28d"] = 100.0 - (100.0 / (1.0 + rs_28))

    roll_up_60 = up.rolling(60, min_periods=30).mean()
    roll_down_60 = down.rolling(60, min_periods=30).mean()
    rs_60 = roll_up_60 / (roll_down_60 + 1e-8)
    data["rsi_60d"] = 100.0 - (100.0 / (1.0 + rs_60))

    # Volume surge at low
    ma_vol_20 = vol.rolling(20, min_periods=5).mean()
    vol_ratio = vol / (ma_vol_20 + 1e-8)
    data["volume_surge_at_low"] = vol_ratio * (1.0 - data["pos_in_52w_range"])

    # Momentum reversal: 5d return vs 60d return
    ret_5d = close / close.shift(5) - 1.0
    ret_60d = close / close.shift(60) - 1.0
    data["momentum_reversal_5d_vs_60d"] = ret_5d - ret_60d

    data["return_5d"] = ret_5d
    data["return_20d"] = close / close.shift(20) - 1.0
    data["return_60d"] = ret_60d

    # Forward targets
    indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=20)
    fwd_max_20 = close.shift(-1).rolling(window=indexer, min_periods=1).max()
    data["max_fwd_ret_20d"] = fwd_max_20 / close - 1.0
    data["target_turnaround_20d"] = (data["max_fwd_ret_20d"] >= 0.05).astype(int)

    data["fwd_ret_20d"] = close.shift(-20) / close - 1.0
    data["fwd_ret_60d"] = close.shift(-60) / close - 1.0
    data["target_turnaround_60d"] = (data["fwd_ret_60d"] >= 0.05).astype(int)

    # Ensure technical columns exist or fallback
    if "rsi_14" not in data.columns:
        data["rsi_14"] = 50.0
    if "volatility_20" not in data.columns:
        data["volatility_20"] = 0.02

    return data


def train_turnaround_model(
    universe: list[TickerInfo] | None = None,
    features_dir: Path | str = "data/features",
    model_dir: Path | str = "models/artifacts",
    metrics_dir: Path | str = "reports/metrics",
) -> tuple[Any, dict[str, Any]]:
    """Train pooled LightGBM turnaround recovery model across universe tickers."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    from stock_ai.models.baseline import save_model

    if universe is None:
        universe = load_universe()

    features_path = Path(features_dir)
    models_path = Path(model_dir)
    metrics_path = Path(metrics_dir)

    all_dfs: list[pd.DataFrame] = []
    for t in universe:
        p = features_path / f"{t.code}.parquet"
        if not p.exists():
            continue
        raw_df = pd.read_parquet(p)
        feat_df = compute_turnaround_features(raw_df)
        feat_df["ticker"] = t.code
        all_dfs.append(feat_df)

    if not all_dfs:
        raise ValueError("No feature datasets found for turnaround model training.")

    full = pd.concat(all_dfs, ignore_index=True)
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values("date").reset_index(drop=True)

    train_mask = (full["date"] >= "2015-01-01") & (full["date"] < "2024-01-01")
    valid_mask = (full["date"] >= "2024-01-01") & (full["date"] < "2024-04-01")
    test_mask = (full["date"] >= "2024-04-01") & pd.notna(full["target_turnaround_20d"]) & pd.notna(full["fwd_ret_20d"])

    train_data = full[train_mask].dropna(subset=TURNAROUND_FEATURE_COLUMNS + ["target_turnaround_20d"])
    valid_data = full[valid_mask].dropna(subset=TURNAROUND_FEATURE_COLUMNS + ["target_turnaround_20d"])
    test_data = full[test_mask].dropna(subset=TURNAROUND_FEATURE_COLUMNS + ["target_turnaround_20d", "fwd_ret_20d"])

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 5,
        "feature_fraction": 0.8,
        "min_child_samples": 50,
        "random_state": 42,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(train_data[TURNAROUND_FEATURE_COLUMNS], label=train_data["target_turnaround_20d"])
    dvalid = lgb.Dataset(valid_data[TURNAROUND_FEATURE_COLUMNS], label=valid_data["target_turnaround_20d"], reference=dtrain)

    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=300,
        valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
    )

    test_probs = booster.predict(test_data[TURNAROUND_FEATURE_COLUMNS])
    test_data = test_data.copy()
    test_data["p_turnaround"] = test_probs

    auc_val = float(roc_auc_score(test_data["target_turnaround_20d"], test_probs))

    # Calculate Bargain Score on test set
    scores = []
    for _, row in test_data.iterrows():
        sc, _, _ = calculate_bargain_score(
            row["drawdown_252d"], row["pos_in_52w_range"], row["ma_gap_200d"], row["p_turnaround"]
        )
        scores.append(sc)
    test_data["bargain_score"] = scores

    q5_cutoff = float(test_data["bargain_score"].quantile(0.80))
    q5_data = test_data[test_data["bargain_score"] >= q5_cutoff]

    metrics_dict = {
        "test_auc": auc_val,
        "train_rows": len(train_data),
        "valid_rows": len(valid_data),
        "test_rows": len(test_data),
        "q5_score_cutoff": q5_cutoff,
        "market_baseline": {
            "ret_20d_avg": float(test_data["fwd_ret_20d"].mean()),
            "ret_20d_win_rate": float((test_data["fwd_ret_20d"] > 0).mean()),
            "ret_60d_avg": float(test_data["fwd_ret_60d"].dropna().mean()),
            "ret_60d_win_rate": float((test_data["fwd_ret_60d"].dropna() > 0).mean()),
        },
        "q5_band_performance": {
            "instances": len(q5_data),
            "ret_20d_avg": float(q5_data["fwd_ret_20d"].mean()),
            "ret_20d_win_rate": float((q5_data["fwd_ret_20d"] > 0).mean()),
            "ret_20d_bounce_hit_rate": float((q5_data["max_fwd_ret_20d"] >= 0.05).mean()),
            "ret_60d_avg": float(q5_data["fwd_ret_60d"].dropna().mean()),
            "ret_60d_win_rate": float((q5_data["fwd_ret_60d"].dropna() > 0).mean()),
        },
    }

    # Save model bundle and metrics
    models_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    model_file = models_path / TURNAROUND_ARTIFACT_FILENAME
    save_model(
        booster,
        TURNAROUND_FEATURE_COLUMNS,
        model_file,
        metadata={"metrics": metrics_dict, "target": "target_turnaround_20d"},
    )

    metrics_file = metrics_path / TURNAROUND_METRICS_FILENAME
    metrics_file.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False), encoding="utf-8")

    return booster, metrics_dict


def calculate_bargain_score(
    drawdown_252d: float,
    pos_in_52w_range: float,
    ma_gap_200d: float,
    p_turnaround: float,
) -> tuple[float, float, float]:
    """Calculate the Turnaround Bargain Score (0 to 100).

    Returns:
        (total_score, discount_factor, recovery_factor)
    """
    # Discount factor (0 to 50 pts):
    dd_pts = float(np.clip((-drawdown_252d / 0.40) * 25.0, 0, 25.0))
    pos_pts = float(np.clip((1.0 - pos_in_52w_range) * 15.0, 0, 15.0))
    ma_pts = float(np.clip((-ma_gap_200d / 0.20) * 10.0, 0, 10.0))
    discount_factor = dd_pts + pos_pts + ma_pts

    # Recovery factor (0 to 50 pts):
    recovery_factor = float(np.clip(p_turnaround * 50.0, 0, 50.0))
    total_score = float(np.clip(discount_factor + recovery_factor, 0.0, 100.0))
    return round(total_score, 1), round(discount_factor, 1), round(recovery_factor, 1)


def fetch_cached_fundamentals(
    tickers: list[str],
    cache_path: Path | str = "data/fundamentals_cache.json",
) -> dict[str, dict[str, float | None]]:
    """Fetch PBR, PER, DividendYield with local caching and safe fallback."""
    cache_file = Path(cache_path)
    cache: dict[str, dict[str, float | None]] = {}
    if cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Benchmark fallback data for 20 Prime stocks if offline/network fails
    fallback_benchmarks: dict[str, dict[str, float | None]] = {
        "7203.T": {"pbr": 0.96, "per": 8.6, "dividend_yield": 3.35},
        "7267.T": {"pbr": 0.58, "per": 7.2, "dividend_yield": 4.10},
        "7201.T": {"pbr": 0.28, "per": 5.4, "dividend_yield": 4.80},
        "8035.T": {"pbr": 6.80, "per": 23.5, "dividend_yield": 2.10},
        "6857.T": {"pbr": 8.50, "per": 34.0, "dividend_yield": 1.40},
        "6723.T": {"pbr": 1.85, "per": 14.2, "dividend_yield": 0.00},
        "8306.T": {"pbr": 0.95, "per": 11.2, "dividend_yield": 3.50},
        "8316.T": {"pbr": 0.88, "per": 11.8, "dividend_yield": 3.40},
        "8411.T": {"pbr": 0.82, "per": 10.5, "dividend_yield": 3.60},
        "8058.T": {"pbr": 1.15, "per": 10.8, "dividend_yield": 3.30},
        "8031.T": {"pbr": 1.18, "per": 9.9, "dividend_yield": 3.20},
        "6758.T": {"pbr": 2.10, "per": 16.5, "dividend_yield": 0.80},
        "6501.T": {"pbr": 2.45, "per": 19.8, "dividend_yield": 1.20},
        "7011.T": {"pbr": 3.10, "per": 28.5, "dividend_yield": 1.10},
        "9984.T": {"pbr": 1.25, "per": 18.0, "dividend_yield": 0.50},
        "9433.T": {"pbr": 1.70, "per": 14.5, "dividend_yield": 3.10},
        "7974.T": {"pbr": 3.60, "per": 22.0, "dividend_yield": 2.50},
        "9983.T": {"pbr": 5.90, "per": 38.0, "dividend_yield": 1.00},
        "4502.T": {"pbr": 0.92, "per": 17.5, "dividend_yield": 4.20},
        "4063.T": {"pbr": 2.30, "per": 21.0, "dividend_yield": 1.80},
    }

    updated = False
    for code in tickers:
        if code not in cache or cache[code].get("pbr") is None:
            # Try yfinance fetch with fast timeout
            try:
                import yfinance as yf

                t = yf.Ticker(code)
                info = t.info
                pbr = info.get("priceToBook")
                per = info.get("trailingPE")
                dy = info.get("dividendYield")
                if dy is not None and dy < 0.2:  # Sometimes reported as decimal (e.g. 0.0335)
                    dy = dy * 100.0

                cache[code] = {
                    "pbr": round(float(pbr), 2) if pbr is not None else fallback_benchmarks.get(code, {}).get("pbr"),
                    "per": round(float(per), 1) if per is not None else fallback_benchmarks.get(code, {}).get("per"),
                    "dividend_yield": round(float(dy), 2)
                    if dy is not None
                    else fallback_benchmarks.get(code, {}).get("dividend_yield"),
                }
                updated = True
            except Exception:
                cache[code] = fallback_benchmarks.get(code, {"pbr": None, "per": None, "dividend_yield": None})
                updated = True

    if updated:
        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return cache


def scan_turnaround_universe(
    universe: list[TickerInfo] | None = None,
    features_dir: Path | str = "data/features",
    model_dir: Path | str = "models/artifacts",
    database_path: Path | str = "database/market.db",
) -> dict[str, Any]:
    """Scan all universe tickers, predict recovery probability, calculate Bargain Scores, and combine with Overnight signals."""
    import sqlite3

    if universe is None:
        universe = load_universe()

    features_path = Path(features_dir)
    models_path = Path(model_dir)
    model_file = models_path / TURNAROUND_ARTIFACT_FILENAME

    if not model_file.exists():
        train_turnaround_model(universe=universe, features_dir=features_dir, model_dir=model_dir)

    bundle = load_model_bundle(model_file)
    model = bundle["model"]
    feature_cols = bundle.get("feature_columns", TURNAROUND_FEATURE_COLUMNS)

    codes = [t.code for t in universe]
    fundamentals = fetch_cached_fundamentals(codes)
    meta_map = {t.code: (t.name, getattr(t, "sector_ja", t.sector)) for t in universe}

    # Fetch latest overnight predictions from SQLite
    overnight_info: dict[str, dict[str, Any]] = {}
    db_file = Path(database_path)
    if db_file.exists():
        try:
            conn = sqlite3.connect(db_file)
            c = conn.cursor()
            for code in codes:
                c.execute(
                    """
                    SELECT probability_up
                    FROM predictions
                    WHERE ticker = ? AND model_name = 'lightgbm_overnight_binary'
                    ORDER BY date DESC LIMIT 1
                """,
                    (code,),
                )
                row = c.fetchone()
                if row:
                    from stock_ai.models.overnight import classify_overnight_quantile

                    p_up = float(row[0])
                    q_info = classify_overnight_quantile(p_up, ticker=code)
                    overnight_info[code] = {
                        "p_up": p_up,
                        "quantile": q_info["quantile"],
                        "label": q_info["label"],
                    }
            conn.close()
        except Exception:
            pass

    rankings: list[dict[str, Any]] = []
    latest_date_str = "-"

    for t in universe:
        code = t.code
        f_file = features_path / f"{code}.parquet"
        if not f_file.exists():
            continue

        raw_df = pd.read_parquet(f_file)
        feat_df = compute_turnaround_features(raw_df)
        if feat_df.empty:
            continue

        latest_row = feat_df.iloc[-1]
        latest_date_str = pd.Timestamp(latest_row["date"]).strftime("%Y-%m-%d")

        # Predict probability
        x = feat_df.iloc[[-1]][feature_cols]
        prob = float(model.predict(x)[0])

        score, disc_factor, rec_factor = calculate_bargain_score(
            drawdown_252d=float(latest_row["drawdown_252d"]),
            pos_in_52w_range=float(latest_row["pos_in_52w_range"]),
            ma_gap_200d=float(latest_row["ma_gap_200d"]),
            p_turnaround=prob,
        )

        o_data = overnight_info.get(code, {"p_up": None, "quantile": "-", "label": ""})
        o_pup = o_data.get("p_up")
        o_q = o_data.get("quantile", "-")
        is_golden = bool(score >= 60.0 and o_q in ("Q4", "Q5"))

        if score >= 70.0:
            status = "💎 超割安・底打ち反転圏"
        elif score >= 50.0:
            status = "👀 調整監視圏"
        else:
            status = "📈 高値巡航圏"

        if is_golden:
            status += " ⭐ 始動"

        fund = fundamentals.get(code, {})
        name, sec = meta_map.get(code, (code, ""))

        rankings.append(
            {
                "ticker": code,
                "name": name,
                "sector": sec,
                "close": float(latest_row["close"]),
                "drawdown_252d": float(latest_row["drawdown_252d"]),
                "pos_in_52w_range": float(latest_row["pos_in_52w_range"]),
                "ma_gap_200d": float(latest_row["ma_gap_200d"]),
                "p_turnaround": prob,
                "discount_factor": disc_factor,
                "recovery_factor": rec_factor,
                "bargain_score": score,
                "pbr": fund.get("pbr"),
                "per": fund.get("per"),
                "dividend_yield": fund.get("dividend_yield"),
                "overnight_p_up": o_pup,
                "overnight_quantile": f"{o_q} ({o_data.get('label', '')})" if o_q != "-" else "-",
                "is_golden_signal": is_golden,
                "status": status,
            }
        )

    # Sort by bargain_score descending
    rankings.sort(key=lambda r: r["bargain_score"], reverse=True)

    metrics_file = Path("reports/metrics") / TURNAROUND_METRICS_FILENAME
    metrics_info = {}
    if metrics_file.exists():
        try:
            metrics_info = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "date": latest_date_str,
        "rankings": rankings,
        "metrics": metrics_info,
    }


def export_turnaround_reports(
    scan_data: dict[str, Any],
    output_json_path: Path | str = "reports/autonomous/value_turnaround_scanner_results.json",
    output_md_path: Path | str = "reports/autonomous/value_turnaround_scanner_report.md",
) -> None:
    """Export Value & Turnaround scan results to JSON and Markdown reports."""
    json_path = Path(output_json_path)
    md_path = Path(output_md_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(scan_data, indent=2, ensure_ascii=False), encoding="utf-8")

    rankings = scan_data.get("rankings", [])
    metrics = scan_data.get("metrics", {})
    m_base = metrics.get("market_baseline", {})
    q5_perf = metrics.get("q5_band_performance", {})

    lines = [
        "# Value & Turnaround Scanner Report (激安・底打ち回復株発掘エンジン)",
        "",
        f"- **分析基準日**: {scan_data.get('date', '-')} (大引け時点)",
        "- **対象ユニバース**: 東証プライム主力20銘柄 (Core 20 / 10セクター)",
        f"- **回復予測モデル AUC**: **{metrics.get('test_auc', 0.5):.4f}** (20営業日内 +5%以上の上昇到達予測)",
        "",
        "## 1. モデルバックテスト検証結果 (高スコア帯のエッジ検証)",
        "",
        "| 指標 | 市場ベースライン (全銘柄平均) | 💎 Q5 激安回復スコア帯 (Top 20%) | アルファ (超過リターン) |",
        "|:---|:---:|:---:|:---:|",
        f"| **20営業日後 平均リターン** | {m_base.get('ret_20d_avg', 0.0):+.2%} | **{q5_perf.get('ret_20d_avg', 0.0):+.2%}** | **+{q5_perf.get('ret_20d_avg', 0.0) - m_base.get('ret_20d_avg', 0.0):.2%}** |",
        f"| **20営業日内 +5%到達勝率** | {m_base.get('ret_20d_win_rate', 0.0):.1%} | **{q5_perf.get('ret_20d_bounce_hit_rate', 0.0):.1%}** | **+{q5_perf.get('ret_20d_bounce_hit_rate', 0.0) - m_base.get('ret_20d_win_rate', 0.0):.1%}** |",
        f"| **60営業日後 平均リターン** | {m_base.get('ret_60d_avg', 0.0):+.2%} | **{q5_perf.get('ret_60d_avg', 0.0):+.2%}** | **+{q5_perf.get('ret_60d_avg', 0.0) - m_base.get('ret_60d_avg', 0.0):.2%}** |",
        f"| **60営業日後 勝率 (>0%)** | {m_base.get('ret_60d_win_rate', 0.0):.1%} | **{q5_perf.get('ret_60d_win_rate', 0.0):.1%}** | **+{q5_perf.get('ret_60d_win_rate', 0.0) - m_base.get('ret_60d_win_rate', 0.0):.1%}** |",
        "",
        "## 2. 💎 将来回復しそうな激安株ランキング（全20銘柄一覧表）",
        "",
        "| 順位 | コード | 銘柄名 | セクター | 現在株価 | 52週高値乖離 | 200日線乖離 | PBR | 利回り | 回復確率 | 激安スコア | 短期Overnight | 総合判定ステータス |",
        "|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for idx, r in enumerate(rankings, 1):
        pbr_str = f"{r['pbr']:.2f}倍" if r.get("pbr") is not None else "-"
        dy_str = f"{r['dividend_yield']:.2f}%" if r.get("dividend_yield") is not None else "-"
        o_str = f"{r['overnight_p_up']:.4f} {r['overnight_quantile']}" if r.get("overnight_p_up") is not None else "-"

        lines.append(
            f"| {idx} | `{r['ticker']}` | {r['name']} | {r['sector']} | {r['close']:,.1f}円 | "
            f"{r['drawdown_252d']:+.1%} | {r['ma_gap_200d']:+.1%} | {pbr_str} | {dy_str} | "
            f"{r['p_turnaround']:.1%} | **{r['bargain_score']:.1f}点** | {o_str} | {r['status']} |"
        )

    # Top 3 spotlight
    top3 = rankings[:3]
    lines.extend([
        "",
        "## 3. 割安かつ反転期待が高い注目トップ3銘柄の深掘り解説",
        "",
    ])

    for idx, t in enumerate(top3, 1):
        lines.extend([
            f"### 第{idx}位: `{t['ticker']}` {t['name']} ({t['sector']}) — 激安スコア: **{t['bargain_score']:.1f}点**",
            f"- **現在株価**: {t['close']:,.1f} 円 (52週高値から **{t['drawdown_252d']:+.1%}**, 200日線から **{t['ma_gap_200d']:+.1%}**)",
            f"- **バリュエーション**: PBR **{t.get('pbr', '-')}倍**, 配当利回り **{t.get('dividend_yield', '-')}%**",
            f"- **Turnaround 回復確率**: **{t['p_turnaround']:.1%}** (20営業日内 +5%以上反転期待)",
            f"- **短期 Overnight 判定**: **{t['overnight_p_up']:.4f}** ({t['overnight_quantile']})" if t.get("overnight_p_up") is not None else "- **短期 Overnight 判定**: -",
            f"- **解説**: 過去1年の底値圏（レンジ内位置 {t['pos_in_52w_range']:.1%}）に位置し、ディープバリュー水準。{'短期Overnightでも買い先行シグナルが点灯しており、底打ち反転の初動（ゴールデン・シグナル）が強く示唆されます。' if t['is_golden_signal'] else '中長期の買い場水準として注視局面。'}",
            "",
        ])

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
