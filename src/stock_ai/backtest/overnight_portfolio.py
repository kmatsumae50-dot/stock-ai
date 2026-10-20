from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.data.universe import TickerInfo, load_universe
from stock_ai.events.earnings_guard import check_earnings_risk
from stock_ai.models.intraday import (
    INTRADAY_ARTIFACT_FILENAME,
    classify_intraday_quantile,
)
from stock_ai.models.overnight import (
    OVERNIGHT_ARTIFACT_FILENAME,
    POOLED_OVERNIGHT_ARTIFACT_FILENAME,
    classify_overnight_quantile,
)
from stock_ai.features.overnight import check_jumping_catch_risk
from stock_ai.models.predict import load_model_bundle


@dataclass
class StrategyMetrics:
    strategy_name: str
    cost_bps: float
    initial_capital: float
    final_equity: float
    cumulative_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: float
    total_days: int
    active_days: int
    win_rate: float
    profit_factor: float
    avg_trade_bps: float


def collect_overnight_universe_predictions(
    universe: list[TickerInfo] | None = None,
    features_dir: Path | str = "data/features",
    models_dir: Path | str = "models/artifacts",
    metrics_dir: Path | str = "reports/metrics",
    start_date: str = "2024-04-01",
) -> tuple[pd.DataFrame, list[pd.Timestamp]]:
    """Collect cross-sectional out-of-sample overnight predictions and actual next_open returns."""
    if universe is None:
        universe = load_universe()

    features_path = Path(features_dir)
    models_path = Path(models_dir)
    metrics_path = Path(metrics_dir)

    pooled_m_file = models_path / POOLED_OVERNIGHT_ARTIFACT_FILENAME
    pooled_bundle = None
    if pooled_m_file.exists():
        try:
            pooled_bundle = load_model_bundle(pooled_m_file)
        except Exception:
            pass

    intraday_m_file = models_path / INTRADAY_ARTIFACT_FILENAME
    intraday_bundle = None
    if intraday_m_file.exists():
        try:
            import joblib
            intraday_bundle = joblib.load(intraday_m_file)
        except Exception:
            pass

    records: list[pd.DataFrame] = []
    meta_map = {t.code: (t.name, getattr(t, "sector_ja", t.sector)) for t in universe}

    for t in universe:
        code = t.code
        f_file = features_path / f"{code}.parquet"
        m_file = models_path / OVERNIGHT_ARTIFACT_FILENAME.format(ticker=code)
        met_file = metrics_path / f"lightgbm_overnight_binary_{code}.json"

        if not (f_file.exists() and m_file.exists()):
            continue

        df = pd.read_parquet(f_file)
        if "next_open_return" not in df.columns or "date" not in df.columns:
            continue

        df["date"] = pd.to_datetime(df["date"])
        sub = df[df["date"] >= pd.to_datetime(start_date)].copy()
        if sub.empty:
            continue

        bundle = load_model_bundle(m_file)
        feature_cols = bundle["feature_columns"]

        auc_val = 0.5
        if met_file.exists():
            try:
                met = json.loads(met_file.read_text(encoding="utf-8"))
                auc_val = met.get("metrics", {}).get("auc", 0.5)
            except Exception:
                pass

        probs = bundle["model"].predict_proba(sub[feature_cols])[:, 1]
        sub["p_up"] = probs
        sub["auc"] = auc_val
        sub["ticker"] = code
        sub["name"] = meta_map.get(code, (code, ""))[0]
        sub["sector"] = meta_map.get(code, (code, ""))[1]

        # Calculate pooled cross-sectional outperform score if available
        if pooled_bundle is not None:
            p_cols = [c for c in pooled_bundle["feature_columns"] if c in sub.columns]
            if len(p_cols) == len(pooled_bundle["feature_columns"]):
                sub["p_cs"] = pooled_bundle["model"].predict_proba(sub[p_cols])[:, 1]
            else:
                sub["p_cs"] = probs
        else:
            sub["p_cs"] = probs

        # Calculate morning 07:30 intraday predictions if available
        if intraday_bundle is not None:
            i_cols = intraday_bundle["feature_columns"]
            missing_i_cols = [c for c in i_cols if c not in sub.columns]
            if missing_i_cols:
                from stock_ai.features.intraday import add_intraday_premarket_features
                sub = add_intraday_premarket_features(sub)
            for c in i_cols:
                if c not in sub.columns:
                    sub[c] = 0.0
            sub["p_intraday_cs"] = intraday_bundle["model"].predict_proba(sub[i_cols].fillna(0.0))[:, 1]
            if "regressor" in intraday_bundle and intraday_bundle["regressor"] is not None:
                sub["pred_intraday_ret"] = intraday_bundle["regressor"].predict(sub[i_cols].fillna(0.0))
            else:
                sub["pred_intraday_ret"] = 0.0
        else:
            sub["p_intraday_cs"] = 0.5
            sub["pred_intraday_ret"] = 0.0

        quantiles: list[str] = []
        win_rates: list[float] = []
        directions: list[str] = []
        for p in probs:
            q_info = classify_overnight_quantile(float(p), ticker=code)
            quantiles.append(q_info["quantile"])
            win_rates.append(q_info.get("historical_win_rate", 0.5) or 0.5)
            directions.append(q_info["direction"])

        sub["quantile"] = quantiles
        sub["win_rate"] = win_rates
        sub["direction"] = directions

        # Attach earnings risk flag (especially for live pending date)
        sub["earnings_risk"] = False
        latest_dt = sub["date"].max()
        e_risk, _ = check_earnings_risk(code, ref_date=latest_dt)
        sub.loc[sub["date"] == latest_dt, "earnings_risk"] = e_risk

        # Attach execution guard & risk features
        if "high" in sub.columns and "low" in sub.columns and "close" in sub.columns:
            hl = (sub["high"] - sub["low"]).replace(0.0, np.nan)
            sub["pullback_from_high"] = ((sub["high"] - sub["close"]) / (hl + 1e-8)).clip(0.0, 1.0).fillna(0.0)
        else:
            sub["pullback_from_high"] = 0.0

        if "intraday_return" not in sub.columns and "open" in sub.columns and "close" in sub.columns:
            sub["intraday_return"] = ((sub["close"] - sub["open"]) / (sub["open"] + 1e-8)).fillna(0.0)
        elif "intraday_return" not in sub.columns:
            sub["intraday_return"] = sub.get("return_1d", 0.0)

        if "overnight_gap" not in sub.columns and "open" in sub.columns and "close" in sub.columns:
            prev_c = sub["close"].shift(1)
            sub["overnight_gap"] = ((sub["open"] - prev_c) / (prev_c + 1e-8)).fillna(0.0)
        elif "overnight_gap" not in sub.columns:
            sub["overnight_gap"] = 0.0

        # Check jumping catch risk row-by-row
        jc_risks: list[bool] = []
        jc_reasons: list[str] = []
        for _, row_val in sub.iterrows():
            is_jc, jc_rsn = check_jumping_catch_risk(row_val)
            jc_risks.append(is_jc)
            jc_reasons.append(jc_rsn)

        sub["jumping_catch_risk"] = jc_risks
        sub["jumping_catch_reason"] = jc_reasons
        if "volatility_20" not in sub.columns:
            sub["volatility_20"] = 0.02
        if "return_1d" not in sub.columns:
            sub["return_1d"] = 0.0
        if "rsi_14" not in sub.columns:
            sub["rsi_14"] = 50.0
        if "vol_adjusted_momentum_1d" not in sub.columns:
            sub["vol_adjusted_momentum_1d"] = 0.0

        cols_to_keep = [
            "date",
            "ticker",
            "name",
            "sector",
            "close",
            "next_open_return",
            "intraday_return",
            "overnight_gap",
            "p_up",
            "p_cs",
            "p_intraday_cs",
            "pred_intraday_ret",
            "quantile",
            "win_rate",
            "direction",
            "auc",
            "earnings_risk",
            "jumping_catch_risk",
            "jumping_catch_reason",
            "volatility_20",
            "return_1d",
            "rsi_14",
            "pullback_from_high",
        ]
        records.append(sub[cols_to_keep])

    if not records:
        return pd.DataFrame(), []

    full_df = pd.concat(records, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    if "p_cs" in full_df.columns:
        full_df["cs_rank"] = full_df.groupby("date")["p_cs"].rank(pct=True).fillna(0.5)
    dates = sorted(full_df["date"].unique())
    return full_df, dates


def simulate_overnight_strategy(
    df: pd.DataFrame,
    dates: list[pd.Timestamp],
    strategy_name: str,
    cost: float = 0.0020,
    initial_capital: float = 1_000_000.0,
    max_position_weight: float | None = None,
) -> dict[str, Any]:
    """Simulate a given portfolio strategy over the date range with realistic costs and position limits."""
    equity = initial_capital
    trade_logs: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []

    for d in dates:
        day_slice = df[df["date"] == d].copy()
        is_latest = d == dates[-1]

        long_picks: list[str] = []
        short_picks: list[str] = []

        if strategy_name == "q5_equal_weight":
            # All Q5, up to 5, p_up desc
            q5s = day_slice[day_slice["quantile"] == "Q5"].sort_values("p_up", ascending=False)
            if not q5s.empty:
                long_picks = q5s.head(5)["ticker"].tolist()

        elif strategy_name == "elite_top2_long":
            # High Conviction Long: Dual confirmation + Execution Guard
            # 1. Individual Model AUC >= 0.50 (edge over coin flip)
            # 2. Individual Quantile == Q5 with validation win_rate >= 0.50
            # 3. Cross-Sectional Confirmation: p_cs >= daily median p_cs (when pooled model available)
            # 4. Earnings Guard: Filter out earnings risk tickers
            # 5. Anti-Jumping-Catch Execution Guard: Filter out overbought/exhaustion traps
            has_cs = "p_cs" in day_slice.columns
            med_cs = day_slice["p_cs"].median() if has_cs else 0.5
            eligible = day_slice[
                (day_slice["auc"] >= 0.50)
                & (day_slice["quantile"] == "Q5")
                & (day_slice["win_rate"] >= 0.50)
                & ((day_slice["p_cs"] >= med_cs) if has_cs else True)
            ]
            if "earnings_risk" in eligible.columns:
                eligible = eligible[~eligible["earnings_risk"].fillna(False).astype(bool)]
            if "jumping_catch_risk" in eligible.columns:
                eligible = eligible[~eligible["jumping_catch_risk"].fillna(False).astype(bool)]
            sort_col = "p_cs" if has_cs else "p_up"
            elites = eligible.sort_values(sort_col, ascending=False)
            if not elites.empty:
                long_picks = elites.head(2)["ticker"].tolist()

        elif strategy_name == "long_short_spread":
            # Long: Strategy B with Earnings Guard and Jumping Catch Guard
            has_cs = "p_cs" in day_slice.columns
            med_cs = day_slice["p_cs"].median() if has_cs else 0.5
            eligible = day_slice[
                (day_slice["auc"] >= 0.50)
                & (day_slice["quantile"] == "Q5")
                & (day_slice["win_rate"] >= 0.50)
                & ((day_slice["p_cs"] >= med_cs) if has_cs else True)
            ]
            if "earnings_risk" in eligible.columns:
                eligible = eligible[~eligible["earnings_risk"].fillna(False).astype(bool)]
            if "jumping_catch_risk" in eligible.columns:
                eligible = eligible[~eligible["jumping_catch_risk"].fillna(False).astype(bool)]
            sort_col = "p_cs" if has_cs else "p_up"
            elites = eligible.sort_values(sort_col, ascending=False)
            if not elites.empty:
                long_picks = elites.head(2)["ticker"].tolist()

            # Short: Q1 with auc >= 0.50, and p_cs <= median, up to 2, p_cs asc
            q1s = day_slice[
                (day_slice["auc"] >= 0.50)
                & (day_slice["quantile"] == "Q1")
                & ((day_slice["p_cs"] <= med_cs) if has_cs else True)
            ].sort_values(sort_col, ascending=True)
            if not q1s.empty:
                short_picks = q1s.head(2)["ticker"].tolist()

        if is_latest:
            trade_logs.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "long_picks": long_picks,
                    "short_picks": short_picks,
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "pnl": 0.0,
                    "equity": equity,
                    "is_pending": True,
                }
            )
            equity_curve.append({"date": d.strftime("%Y-%m-%d"), "equity": equity})
            continue

        long_rets = []
        for t in long_picks:
            match = day_slice.loc[day_slice["ticker"] == t, "next_open_return"]
            if not match.empty and pd.notna(match.values[0]):
                long_rets.append(float(match.values[0]))

        short_rets = []
        for t in short_picks:
            match = day_slice.loc[day_slice["ticker"] == t, "next_open_return"]
            if not match.empty and pd.notna(match.values[0]):
                short_rets.append(float(match.values[0]))

        gross_return = 0.0
        applied_cost = 0.0

        if long_rets and short_rets:
            # Long/Short spread: longs allocated max 50%, shorts allocated max 50%
            w_long_single = (
                min(0.50 / len(long_rets), max_position_weight)
                if max_position_weight is not None
                else (0.50 / len(long_rets))
            )
            w_short_single = (
                min(0.50 / len(short_rets), max_position_weight)
                if max_position_weight is not None
                else (0.50 / len(short_rets))
            )
            w_total_long = w_long_single * len(long_rets)
            w_total_short = w_short_single * len(short_rets)
            gross_return = (w_total_long * float(np.mean(long_rets))) - (w_total_short * float(np.mean(short_rets)))
            applied_cost = cost * (w_total_long + w_total_short)
        elif long_rets:
            if max_position_weight is not None:
                # Volatility-adaptive sizing when volatility_20 is available
                pick_vols = [
                    float(day_slice.loc[day_slice["ticker"] == t, "volatility_20"].values[0])
                    if "volatility_20" in day_slice.columns and not day_slice.loc[day_slice["ticker"] == t, "volatility_20"].empty
                    else 0.025
                    for t in long_picks
                ]
                if len(long_rets) > 1 and all(v > 0 for v in pick_vols):
                    v_arr = np.clip(np.array(pick_vols), 0.015, 0.055)
                    inv_v = 1.0 / v_arr
                    raw_w = (inv_v / inv_v.sum()) * 0.95
                    pos_weights = np.minimum(raw_w, max_position_weight)
                else:
                    pos_w = min(1.0 / len(long_rets), max_position_weight)
                    pos_weights = np.array([pos_w] * len(long_rets))

                total_w = float(np.sum(pos_weights))
                gross_return = float(np.sum(np.array(long_rets) * pos_weights))
                applied_cost = cost * total_w
            else:
                gross_return = float(np.mean(long_rets))
                applied_cost = cost
        elif short_rets:
            if max_position_weight is not None:
                pos_w = min(1.0 / len(short_rets), max_position_weight)
                total_w = pos_w * len(short_rets)
                gross_return = -float(np.sum(np.array(short_rets) * pos_w))
                applied_cost = cost * total_w
            else:
                gross_return = -float(np.mean(short_rets))
                applied_cost = cost

        net_return = gross_return - applied_cost if (long_rets or short_rets) else 0.0
        pnl = equity * net_return
        equity += pnl

        trade_logs.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "long_picks": long_picks,
                "short_picks": short_picks,
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl": pnl,
                "equity": equity,
                "is_pending": False,
            }
        )
        equity_curve.append({"date": d.strftime("%Y-%m-%d"), "equity": equity})

    settled_logs = [log for log in trade_logs if not log["is_pending"]]
    returns = np.array([log["net_return"] for log in settled_logs])
    active_returns = np.array([log["net_return"] for log in settled_logs if (log["long_picks"] or log["short_picks"])])

    cum_return = (equity / initial_capital) - 1.0
    total_days = len(settled_logs)
    cagr = ((1.0 + cum_return) ** (250.0 / total_days)) - 1.0 if total_days > 0 and (1.0 + cum_return) > 0 else 0.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(250.0)) if np.std(returns) > 1e-9 else 0.0

    eq_series = pd.Series([log["equity"] for log in settled_logs])
    peak = eq_series.cummax()
    dd = (peak - eq_series) / peak
    max_dd = float(dd.max()) if not dd.empty else 0.0

    win_rate = float(np.mean(active_returns > 0)) if len(active_returns) > 0 else 0.0

    gains = [log["pnl"] for log in settled_logs if log["pnl"] > 0]
    losses = [abs(log["pnl"]) for log in settled_logs if log["pnl"] < 0]
    profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(gains) > 0 else 1.0)
    avg_trade_bps = float(np.mean(active_returns) * 10000.0) if len(active_returns) > 0 else 0.0

    metrics = StrategyMetrics(
        strategy_name=strategy_name,
        cost_bps=cost * 10000.0,
        initial_capital=initial_capital,
        final_equity=equity,
        cumulative_return=cum_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_days=total_days,
        active_days=len(active_returns),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_bps=avg_trade_bps,
    )

    return {
        "metrics": asdict(metrics),
        "trade_logs": trade_logs,
        "equity_curve": equity_curve,
    }


def simulate_intraday_strategy(
    df: pd.DataFrame,
    dates: list[pd.Timestamp],
    strategy_name: str = "intraday_top1_long",
    cost: float = 0.0020,
    initial_capital: float = 1_000_000.0,
    hurdle_bps: float = 30.0,
) -> dict[str, Any]:
    """Simulate morning 07:30 pre-market intraday (Open[t] -> Close[t]) strategy with strict cost deduction."""
    equity = initial_capital
    trade_logs: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    hurdle = hurdle_bps / 10000.0

    for d in dates:
        day_slice = df[df["date"] == d].copy()
        is_latest = d == dates[-1]

        picks: list[str] = []
        if "pred_intraday_ret" in day_slice.columns:
            cands = day_slice[day_slice["pred_intraday_ret"] >= hurdle].sort_values("pred_intraday_ret", ascending=False)
            if not cands.empty:
                for _, c in cands.iterrows():
                    # Intraday earnings guard
                    e_risk, _ = check_earnings_risk(c["ticker"], ref_date=d)
                    if not e_risk:
                        picks = [c["ticker"]]
                        break

        if is_latest:
            trade_logs.append(
                {
                    "date": d.strftime("%Y-%m-%d"),
                    "long_picks": picks,
                    "short_picks": [],
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "pnl": 0.0,
                    "equity": equity,
                    "is_pending": True,
                }
            )
            equity_curve.append({"date": d.strftime("%Y-%m-%d"), "equity": equity})
            continue

        gross_return = 0.0
        applied_cost = 0.0
        if picks:
            match = day_slice.loc[day_slice["ticker"] == picks[0], "intraday_return"]
            if not match.empty and pd.notna(match.values[0]):
                gross_return = float(match.values[0])
                applied_cost = cost

        net_return = gross_return - applied_cost if picks else 0.0
        pnl = equity * net_return
        equity += pnl

        trade_logs.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "long_picks": picks,
                "short_picks": [],
                "gross_return": gross_return,
                "net_return": net_return,
                "pnl": pnl,
                "equity": equity,
                "is_pending": False,
            }
        )
        equity_curve.append({"date": d.strftime("%Y-%m-%d"), "equity": equity})

    settled_logs = [log for log in trade_logs if not log["is_pending"]]
    returns = np.array([log["net_return"] for log in settled_logs])
    active_returns = np.array([log["net_return"] for log in settled_logs if log["long_picks"]])

    cum_return = (equity / initial_capital) - 1.0
    total_days = len(settled_logs)
    cagr = ((1.0 + cum_return) ** (250.0 / total_days)) - 1.0 if total_days > 0 and (1.0 + cum_return) > 0 else 0.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(250.0)) if np.std(returns) > 1e-9 else 0.0

    eq_series = pd.Series([log["equity"] for log in settled_logs])
    peak = eq_series.cummax()
    dd = (peak - eq_series) / peak
    max_dd = float(dd.max()) if not dd.empty else 0.0

    win_rate = float(np.mean(active_returns > 0)) if len(active_returns) > 0 else 0.0
    gains = [log["pnl"] for log in settled_logs if log["pnl"] > 0]
    losses = [abs(log["pnl"]) for log in settled_logs if log["pnl"] < 0]
    profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(gains) > 0 else 1.0)
    avg_trade_bps = float(np.mean(active_returns) * 10000.0) if len(active_returns) > 0 else 0.0

    metrics = StrategyMetrics(
        strategy_name=strategy_name,
        cost_bps=cost * 10000.0,
        initial_capital=initial_capital,
        final_equity=equity,
        cumulative_return=cum_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_days=total_days,
        active_days=len(active_returns),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_bps=avg_trade_bps,
    )

    return {
        "metrics": asdict(metrics),
        "trade_logs": trade_logs,
        "equity_curve": equity_curve,
    }


def simulate_dual_shift_strategy(
    overnight_res: dict[str, Any],
    intraday_res: dict[str, Any],
    cost: float = 0.0020,
    initial_capital: float = 1_000_000.0,
) -> dict[str, Any]:
    """Simulate Dual-Shift 2毛作 Compound Fund: Overnight leg (15:30 -> 09:00) × Intraday leg (09:00 -> 15:30)."""
    equity = initial_capital
    trade_logs: list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []

    on_by_date = {t["date"]: t for t in overnight_res["trade_logs"]}
    intra_by_date = {t["date"]: t for t in intraday_res["trade_logs"]}
    all_dates = [t["date"] for t in overnight_res["trade_logs"]]

    for d_str in all_dates:
        on_log = on_by_date.get(d_str, {})
        intra_log = intra_by_date.get(d_str, {})
        is_latest = on_log.get("is_pending", False) or intra_log.get("is_pending", False)

        on_picks = on_log.get("long_picks", [])
        intra_picks = intra_log.get("long_picks", [])

        if is_latest:
            trade_logs.append(
                {
                    "date": d_str,
                    "long_picks": on_picks,
                    "intraday_picks": intra_picks,
                    "short_picks": [],
                    "gross_return": 0.0,
                    "net_return": 0.0,
                    "pnl": 0.0,
                    "equity": equity,
                    "is_pending": True,
                }
            )
            equity_curve.append({"date": d_str, "equity": equity})
            continue

        r_on = on_log.get("net_return", 0.0)
        r_intra = intra_log.get("net_return", 0.0)
        compound_ret = (1.0 + r_on) * (1.0 + r_intra) - 1.0
        pnl = equity * compound_ret
        equity += pnl

        trade_logs.append(
            {
                "date": d_str,
                "long_picks": on_picks,
                "intraday_picks": intra_picks,
                "short_picks": [],
                "overnight_return": r_on,
                "intraday_return": r_intra,
                "gross_return": compound_ret + (cost if (on_picks or intra_picks) else 0.0),
                "net_return": compound_ret,
                "pnl": pnl,
                "equity": equity,
                "is_pending": False,
            }
        )
        equity_curve.append({"date": d_str, "equity": equity})

    settled_logs = [log for log in trade_logs if not log["is_pending"]]
    returns = np.array([log["net_return"] for log in settled_logs])
    active_returns = np.array([log["net_return"] for log in settled_logs if (log["long_picks"] or log.get("intraday_picks"))])

    cum_return = (equity / initial_capital) - 1.0
    total_days = len(settled_logs)
    cagr = ((1.0 + cum_return) ** (250.0 / total_days)) - 1.0 if total_days > 0 and (1.0 + cum_return) > 0 else 0.0
    sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(250.0)) if np.std(returns) > 1e-9 else 0.0

    eq_series = pd.Series([log["equity"] for log in settled_logs])
    peak = eq_series.cummax()
    dd = (peak - eq_series) / peak
    max_dd = float(dd.max()) if not dd.empty else 0.0

    win_rate = float(np.mean(active_returns > 0)) if len(active_returns) > 0 else 0.0
    gains = [log["pnl"] for log in settled_logs if log["pnl"] > 0]
    losses = [abs(log["pnl"]) for log in settled_logs if log["pnl"] < 0]
    profit_factor = (sum(gains) / sum(losses)) if sum(losses) > 0 else (99.0 if sum(gains) > 0 else 1.0)
    avg_trade_bps = float(np.mean(active_returns) * 10000.0) if len(active_returns) > 0 else 0.0

    metrics = StrategyMetrics(
        strategy_name="dual_shift_compound",
        cost_bps=cost * 10000.0,
        initial_capital=initial_capital,
        final_equity=equity,
        cumulative_return=cum_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        total_days=total_days,
        active_days=len(active_returns),
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_trade_bps=avg_trade_bps,
    )

    return {
        "metrics": asdict(metrics),
        "trade_logs": trade_logs,
        "equity_curve": equity_curve,
    }


def run_full_overnight_portfolio_backtest(
    universe: list[TickerInfo] | None = None,
    features_dir: Path | str = "data/features",
    models_dir: Path | str = "models/artifacts",
    metrics_dir: Path | str = "reports/metrics",
    start_date: str = "2024-04-01",
    initial_capital: float = 1_000_000.0,
) -> dict[str, Any]:
    """Execute complete multi-strategy backtest across 0bps and 5bps costs with YTD sub-analysis."""
    df, dates = collect_overnight_universe_predictions(
        universe=universe,
        features_dir=features_dir,
        models_dir=models_dir,
        metrics_dir=metrics_dir,
        start_date=start_date,
    )

    if df.empty or not dates:
        return {"error": "No prediction data found."}

    strategies = [
        "q5_equal_weight",
        "elite_top2_long",
        "long_short_spread",
        "intraday_top1_long",
        "dual_shift_compound",
    ]
    cost_tiers = [
        ("0bps", 0.0),
        ("5bps", 0.0005),
        ("10bps", 0.0010),
        ("20bps", 0.0020),
        ("30bps", 0.0030),
    ]
    results_by_cost: dict[str, dict[str, Any]] = {k: {} for k, _ in cost_tiers}

    for cost_key, cost_val in cost_tiers:
        # 1. Overnight standard strategies
        for s in ["q5_equal_weight", "elite_top2_long", "long_short_spread"]:
            res = simulate_overnight_strategy(
                df,
                dates,
                strategy_name=s,
                cost=cost_val,
                initial_capital=initial_capital,
                max_position_weight=0.50,
            )
            results_by_cost[cost_key][s] = res

        # 2. Intraday standalone strategy (09:00 -> 15:30)
        intra_res = simulate_intraday_strategy(
            df,
            dates,
            strategy_name="intraday_top1_long",
            cost=cost_val,
            initial_capital=initial_capital,
            hurdle_bps=30.0,
        )
        results_by_cost[cost_key]["intraday_top1_long"] = intra_res

        # 3. Dual-Shift Compound (2毛作) strategy: Overnight B × Intraday Top 1
        dual_res = simulate_dual_shift_strategy(
            overnight_res=results_by_cost[cost_key]["elite_top2_long"],
            intraday_res=intra_res,
            cost=cost_val,
            initial_capital=initial_capital,
        )
        results_by_cost[cost_key]["dual_shift_compound"] = dual_res

    # 2026 YTD sub-analysis (Standard 20bps operational cost)
    dates_2026 = [d for d in dates if d >= pd.to_datetime("2026-01-01")]
    ytd_2026: dict[str, Any] = {}
    if dates_2026:
        for s in ["q5_equal_weight", "elite_top2_long", "long_short_spread"]:
            res_2026 = simulate_overnight_strategy(
                df,
                dates_2026,
                strategy_name=s,
                cost=0.0020,
                initial_capital=initial_capital,
                max_position_weight=0.50,
            )
            ytd_2026[s] = res_2026["metrics"]

        intra_2026 = simulate_intraday_strategy(
            df,
            dates_2026,
            strategy_name="intraday_top1_long",
            cost=0.0020,
            initial_capital=initial_capital,
            hurdle_bps=30.0,
        )
        ytd_2026["intraday_top1_long"] = intra_2026["metrics"]

        dual_2026 = simulate_dual_shift_strategy(
            overnight_res=simulate_overnight_strategy(df, dates_2026, "elite_top2_long", cost=0.0020, initial_capital=initial_capital, max_position_weight=0.50),
            intraday_res=intra_2026,
            cost=0.0020,
            initial_capital=initial_capital,
        )
        ytd_2026["dual_shift_compound"] = dual_2026["metrics"]

    # Ticker contribution analysis (Strategy B, 2026 YTD at standard 20bps cost)
    ticker_contrib: list[dict[str, Any]] = []
    strat_b_logs = results_by_cost["20bps"]["elite_top2_long"]["trade_logs"]
    pnl_by_ticker: dict[str, float] = {}
    trades_by_ticker: dict[str, int] = {}
    meta_map = {t.code: (t.name, getattr(t, "sector_ja", t.sector)) for t in (universe or load_universe())}

    for log in strat_b_logs:
        if log["is_pending"] or pd.to_datetime(log["date"]) < pd.to_datetime("2026-01-01"):
            continue
        picks = log["long_picks"]
        if not picks:
            continue
        d = pd.to_datetime(log["date"])
        day_slice = df[df["date"] == d]
        weight = min(1.0 / len(picks), 0.50)
        for t in picks:
            match = day_slice.loc[day_slice["ticker"] == t, "next_open_return"]
            if not match.empty and pd.notna(match.values[0]):
                r = float(match.values[0])
                net_r = r - 0.0020
                pnl = log["equity"] * (weight * net_r)
                pnl_by_ticker[t] = pnl_by_ticker.get(t, 0.0) + pnl
                trades_by_ticker[t] = trades_by_ticker.get(t, 0) + 1

    for t, pnl in sorted(pnl_by_ticker.items(), key=lambda x: x[1], reverse=True):
        ticker_contrib.append(
            {
                "ticker": t,
                "name": meta_map.get(t, (t, ""))[0],
                "sector": meta_map.get(t, (t, ""))[1],
                "profit_jpy": pnl,
                "trades": trades_by_ticker.get(t, 0),
            }
        )

    # Pending allocation for today (latest date)
    latest_date_str = dates[-1].strftime("%Y-%m-%d")
    latest_pending = {
        "date": latest_date_str,
        "strategies": {
            s: {
                "long_picks": results_by_cost["20bps"][s]["trade_logs"][-1].get("long_picks", []),
                "intraday_picks": results_by_cost["20bps"][s]["trade_logs"][-1].get("intraday_picks", []),
                "short_picks": results_by_cost["20bps"][s]["trade_logs"][-1].get("short_picks", []),
                "equity": results_by_cost["20bps"][s]["metrics"]["final_equity"],
            }
            for s in strategies
        },
    }

    return {
        "start_date": dates[0].strftime("%Y-%m-%d"),
        "end_date": latest_date_str,
        "total_days": len(dates),
        "results_by_cost": results_by_cost,
        "ytd_2026": ytd_2026,
        "ticker_contribution_2026": ticker_contrib,
        "latest_pending": latest_pending,
    }


def export_portfolio_simulation_reports(
    simulation_data: dict[str, Any],
    output_json_path: Path | str = "reports/autonomous/overnight_portfolio_simulation_results.json",
    output_md_path: Path | str = "reports/autonomous/overnight_portfolio_simulation_report.md",
) -> None:
    """Export JSON results and a rich Markdown simulation report."""
    json_path = Path(output_json_path)
    md_path = Path(output_md_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(simulation_data, indent=2, ensure_ascii=False), encoding="utf-8")

    res_20 = simulation_data["results_by_cost"].get("20bps") or simulation_data["results_by_cost"].get("5bps")
    res_0 = simulation_data["results_by_cost"]["0bps"]
    ytd = simulation_data.get("ytd_2026", {})
    contrib = simulation_data.get("ticker_contribution_2026", [])
    pending = simulation_data.get("latest_pending", {})

    lines = [
        "# Overnight Portfolio Simulation & Paper Trader Report",
        "",
        f"- **期間**: {simulation_data['start_date']} 〜 {simulation_data['end_date']} (全 {simulation_data['total_days']} 営業日)",
        "- **初期資金**: 1,000,000 円 (100万円)",
        "- **対象ユニバース**: 東証プライム主力20銘柄 (Core 20 / 10セクター)",
        "- **執行コスト基準**: 往復 20bps (0.20% スリッページ・手数料控除)",
        "- **資金管理ルール**: 1銘柄最大投資比率上限 50%（確信度不足時は100%キャッシュ待機）",
        "",
        "## 1. 3大戦略 パフォーマンス比較表 (2024.04 〜 2026.10)",
        "",
        "### 実運用標準コスト控除後 (往復コスト 20bps / 0.20%)",
        "",
        "| 戦略名 | 最終資産額 (円) | 累積リターン | 年率リターン (CAGR) | Sharpe | 最大DD | 日次勝率 | PF | 取引日数 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    strat_labels = {
        "q5_equal_weight": "Strategy A: Q5 均等配分 (Overnight 最大5銘柄)",
        "elite_top2_long": "Strategy B: Overnight 精鋭集中 (AUC>=0.50, Q5, Execution Guard)",
        "long_short_spread": "Strategy C: Long/Short スプレッド (B Long + Q1 Short)",
        "intraday_top1_long": "Strategy D: Intraday (Open-to-Close) 単体 (寄前確定・高確信度厳選)",
        "dual_shift_compound": "Strategy E: 🚀 Overnight × Intraday 2毛作 (Dual-Shift Compound) ファンド",
    }

    for s_key, s_label in strat_labels.items():
        if s_key in res_20:
            m = res_20[s_key]["metrics"]
            lines.append(
                f"| **{s_label}** | **{m['final_equity']:,.0f} 円** | **{m['cumulative_return']:+.1%}** | "
                f"{m['cagr']:+.1%} | **{m['sharpe_ratio']:.2f}** | {m['max_drawdown']:.1%} | "
                f"{m['win_rate']:.1%} | {m['profit_factor']:.2f} | {m['active_days']}日 |"
            )

    lines.extend([
        "",
        "### コスト感度分析 (往復コスト別 2毛作 Dual-Shift Compound vs Overnight 単体)",
        "",
        "| コスト水準 | 2毛作ファンド最終資産 | 2毛作累積 | 2毛作Sharpe | 2毛作PF | Overnight単体資産 | 判定 |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ])
    cost_display = [
        ("0bps (理論値)", "0bps"),
        ("5bps (旧レポート表示)", "5bps"),
        ("10bps (低コスト証券)", "10bps"),
        ("20bps (実運用標準)", "20bps"),
        ("30bps (厳格スリッページ)", "30bps"),
    ]
    for label, k in cost_display:
        if k in simulation_data["results_by_cost"]:
            m_dual = simulation_data["results_by_cost"][k].get("dual_shift_compound", {}).get("metrics", {})
            m_on = simulation_data["results_by_cost"][k].get("elite_top2_long", {}).get("metrics", {})
            status = "✅ プラス期待値" if m_dual.get("final_equity", 0) >= 1_000_000 else "⚠️ コスト負け"
            lines.append(
                f"| {label} | **{m_dual.get('final_equity', 0):,.0f} 円** | **{m_dual.get('cumulative_return', 0):+.1%}** | "
                f"{m_dual.get('sharpe_ratio', 0):.2f} | {m_dual.get('profit_factor', 0):.2f} | "
                f"{m_on.get('final_equity', 0):,.0f} 円 | {status} |"
            )

    lines.extend([
        "",
        "## 2. 2026年単独 (年初来) パフォーマンス (実運用コスト 20bps)",
        "",
        "| 戦略名 | 最終資産額 (円) | 累積リターン | Sharpe | 最大DD | 日次勝率 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s_key, s_label in strat_labels.items():
        m = ytd.get(s_key, {})
        if m:
            lines.append(
                f"| **{s_label}** | {m['final_equity']:,.0f} 円 | **{m['cumulative_return']:+.1%}** | "
                f"{m['sharpe_ratio']:.2f} | {m['max_drawdown']:.1%} | {m['win_rate']:.1%} |"
            )

    lines.extend([
        "",
        "## 3. 2026年 利益貢献銘柄トップ5 (Strategy B 精鋭集中)",
        "",
        "| 順位 | コード | 銘柄名 | セクター | 累積獲得損益 (円) | 取引回数 |",
        "|:---:|:---:|:---|:---|:---:|:---:|",
    ])
    for idx, c in enumerate(contrib[:5], 1):
        lines.append(
            f"| {idx} | `{c['ticker']}` | {c['name']} | {c['sector']} | **+{c['profit_jpy']:,.0f} 円** | {c['trades']}回 |"
        )

    b_pending = pending.get("strategies", {}).get("elite_top2_long", {})
    picks = b_pending.get("long_picks", [])
    equity_val = b_pending.get("equity", 1_000_000.0)
    lines.extend([
        "",
        f"## 4. 本日 ({pending.get('date', '-')}) 大引けの仮想エントリー予定",
        "",
        "- **対象戦略**: Strategy B (精鋭集中ロング)",
        f"- **ファンド残高**: {equity_val:,.0f} 円",
        f"- **選抜銘柄数**: {len(picks)} 銘柄",
    ])
    if picks:
        weight = 1.0 / len(picks)
        for p in picks:
            alloc = equity_val * weight
            lines.append(f"  - `{p}`: 配分比率 **{weight:.0%}** (約 **{alloc:,.0f} 円**)")
    else:
        lines.append("  - 本日の選抜基準該当なし（キャッシュ保有 100%）")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
