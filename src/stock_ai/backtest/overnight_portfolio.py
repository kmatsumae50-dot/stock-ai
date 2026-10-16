from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.data.universe import TickerInfo, load_universe
from stock_ai.events.earnings_guard import check_earnings_risk
from stock_ai.models.overnight import (
    OVERNIGHT_ARTIFACT_FILENAME,
    classify_overnight_quantile,
)
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

        cols_to_keep = [
            "date",
            "ticker",
            "name",
            "sector",
            "close",
            "next_open_return",
            "p_up",
            "quantile",
            "win_rate",
            "direction",
            "auc",
            "earnings_risk",
        ]
        records.append(sub[cols_to_keep])

    if not records:
        return pd.DataFrame(), []

    full_df = pd.concat(records, ignore_index=True)
    full_df["date"] = pd.to_datetime(full_df["date"])
    dates = sorted(full_df["date"].unique())
    return full_df, dates


def simulate_overnight_strategy(
    df: pd.DataFrame,
    dates: list[pd.Timestamp],
    strategy_name: str,
    cost: float = 0.0005,
    initial_capital: float = 1_000_000.0,
) -> dict[str, Any]:
    """Simulate a given portfolio strategy over the date range."""
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
            # AUC >= 0.58 and (Q5 or (Q4 and win_rate >= 0.60)), up to 2, p_up desc
            # Earnings Guard: Filter out earnings risk tickers
            eligible = day_slice[
                (day_slice["auc"] >= 0.58)
                & (
                    (day_slice["quantile"] == "Q5")
                    | ((day_slice["quantile"] == "Q4") & (day_slice["win_rate"] >= 0.60))
                )
            ]
            if "earnings_risk" in eligible.columns:
                eligible = eligible[~eligible["earnings_risk"].fillna(False).astype(bool)]
            elites = eligible.sort_values("p_up", ascending=False)
            if not elites.empty:
                long_picks = elites.head(2)["ticker"].tolist()

        elif strategy_name == "long_short_spread":
            # Long: Strategy B with Earnings Guard
            eligible = day_slice[
                (day_slice["auc"] >= 0.58)
                & (
                    (day_slice["quantile"] == "Q5")
                    | ((day_slice["quantile"] == "Q4") & (day_slice["win_rate"] >= 0.60))
                )
            ]
            if "earnings_risk" in eligible.columns:
                eligible = eligible[~eligible["earnings_risk"].fillna(False).astype(bool)]
            elites = eligible.sort_values("p_up", ascending=False)
            if not elites.empty:
                long_picks = elites.head(2)["ticker"].tolist()

            # Short: Q1, up to 2, p_up asc
            q1s = day_slice[day_slice["quantile"] == "Q1"].sort_values("p_up", ascending=True)
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
                long_rets.append(match.values[0])

        short_rets = []
        for t in short_picks:
            match = day_slice.loc[day_slice["ticker"] == t, "next_open_return"]
            if not match.empty and pd.notna(match.values[0]):
                short_rets.append(match.values[0])

        gross_return = 0.0
        applied_cost = 0.0

        if long_rets and short_rets:
            gross_return = 0.5 * float(np.mean(long_rets)) - 0.5 * float(np.mean(short_rets))
            applied_cost = cost
        elif long_rets:
            gross_return = float(np.mean(long_rets))
            applied_cost = cost
        elif short_rets:
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

    strategies = ["q5_equal_weight", "elite_top2_long", "long_short_spread"]
    results_by_cost: dict[str, dict[str, Any]] = {"0bps": {}, "5bps": {}}

    for cost_key, cost_val in [("0bps", 0.0), ("5bps", 0.0005)]:
        for s in strategies:
            res = simulate_overnight_strategy(
                df,
                dates,
                strategy_name=s,
                cost=cost_val,
                initial_capital=initial_capital,
            )
            results_by_cost[cost_key][s] = res

    # 2026 YTD sub-analysis
    dates_2026 = [d for d in dates if d >= pd.to_datetime("2026-01-01")]
    ytd_2026: dict[str, Any] = {}
    if dates_2026:
        for s in strategies:
            res_2026 = simulate_overnight_strategy(
                df,
                dates_2026,
                strategy_name=s,
                cost=0.0005,
                initial_capital=initial_capital,
            )
            ytd_2026[s] = res_2026["metrics"]

    # Ticker contribution analysis (Strategy B, 2026 YTD)
    ticker_contrib: list[dict[str, Any]] = []
    strat_b_logs = results_by_cost["5bps"]["elite_top2_long"]["trade_logs"]
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
        weight = 1.0 / len(picks)
        for t in picks:
            match = day_slice.loc[day_slice["ticker"] == t, "next_open_return"]
            if not match.empty and pd.notna(match.values[0]):
                r = match.values[0]
                net_r = r - 0.0005
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
                "long_picks": results_by_cost["5bps"][s]["trade_logs"][-1]["long_picks"],
                "short_picks": results_by_cost["5bps"][s]["trade_logs"][-1]["short_picks"],
                "equity": results_by_cost["5bps"][s]["metrics"]["final_equity"],
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

    res_0 = simulation_data["results_by_cost"]["0bps"]
    res_5 = simulation_data["results_by_cost"]["5bps"]
    ytd = simulation_data.get("ytd_2026", {})
    contrib = simulation_data.get("ticker_contribution_2026", [])
    pending = simulation_data.get("latest_pending", {})

    lines = [
        "# Overnight Portfolio Simulation & Paper Trader Report",
        "",
        f"- **期間**: {simulation_data['start_date']} 〜 {simulation_data['end_date']} (全 {simulation_data['total_days']} 営業日)",
        "- **初期資金**: 1,000,000 円 (100万円)",
        "- **対象ユニバース**: 東証プライム主力20銘柄 (Core 20 / 10セクター)",
        "",
        "## 1. 3大戦略 パフォーマンス比較表 (2024.04 〜 2026.10)",
        "",
        "### コスト控除後 (スリッページ・往復コスト 5bps / 0.05%)",
        "",
        "| 戦略名 | 最終資産額 (円) | 累積リターン | 年率リターン (CAGR) | Sharpe | 最大DD | 日次勝率 | PF | 取引日数 |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    strat_labels = {
        "q5_equal_weight": "Strategy A: Q5 均等配分 (最大5銘柄)",
        "elite_top2_long": "Strategy B: 精鋭集中 (AUC>=0.58 上位2銘柄)",
        "long_short_spread": "Strategy C: Long/Short スプレッド (B Long + Q1 Short)",
    }

    for s_key, s_label in strat_labels.items():
        m = res_5[s_key]["metrics"]
        lines.append(
            f"| **{s_label}** | **{m['final_equity']:,.0f} 円** | **{m['cumulative_return']:+.1%}** | "
            f"{m['cagr']:+.1%} | **{m['sharpe_ratio']:.2f}** | {m['max_drawdown']:.1%} | "
            f"{m['win_rate']:.1%} | {m['profit_factor']:.2f} | {m['active_days']}日 |"
        )

    lines.extend([
        "",
        "### 理論値 (コスト 0bps)",
        "",
        "| 戦略名 | 最終資産額 (円) | 累積リターン | Sharpe | 最大DD | 日次勝率 | PF |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])
    for s_key, s_label in strat_labels.items():
        m = res_0[s_key]["metrics"]
        lines.append(
            f"| {s_label} | {m['final_equity']:,.0f} 円 | {m['cumulative_return']:+.1%} | "
            f"{m['sharpe_ratio']:.2f} | {m['max_drawdown']:.1%} | {m['win_rate']:.1%} | {m['profit_factor']:.2f} |"
        )

    lines.extend([
        "",
        "## 2. 2026年単独 (年初来) パフォーマンス (コスト 5bps)",
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
