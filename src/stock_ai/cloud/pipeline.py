from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.cloud.notifications import (
    build_alert_report_payload,
    build_fund_report_payload,
    build_multi_agent_webhook_payloads,
    build_turnaround_report_payload,
    build_webhook_payload,
    get_configured_channel_webhooks,
    get_current_pipeline_summary,
    normalize_webhook_url,
    send_daily_summary_webhook,
)
from stock_ai.config import load_settings
from stock_ai.data.universe import TickerInfo, load_turnaround_universe, load_universe
from stock_ai.data.yfinance_provider import fetch_daily_ohlcv
from stock_ai.events.earnings_guard import check_earnings_risk
from stock_ai.external.market_data import download_external_market_data_report
from stock_ai.models.overnight import (
    OVERNIGHT_ARTIFACT_FILENAME,
    classify_overnight_quantile,
)
from stock_ai.models.turnaround import (
    TURNAROUND_ARTIFACT_FILENAME,
    export_turnaround_reports,
    scan_turnaround_universe,
    train_turnaround_model,
)
from stock_ai.storage.parquet_store import read_parquet, write_parquet

__all__ = [
    "build_alert_report_payload",
    "build_fund_report_payload",
    "build_multi_agent_webhook_payloads",
    "build_turnaround_report_payload",
    "build_webhook_payload",
    "get_configured_channel_webhooks",
    "get_current_pipeline_summary",
    "normalize_webhook_url",
    "run_cloud_daily_pipeline",
    "save_overnight_scanner_snapshot",
    "send_daily_summary_webhook",
]

logger = logging.getLogger(__name__)




def save_overnight_scanner_snapshot(
    rankings: list[dict[str, Any]],
    output_path: Path | str = "reports/autonomous/latest_overnight_scanner_snapshot.json",
    date_str: str | None = None,
) -> Path:
    """Save clean, lightweight snapshot of the multi-ticker overnight scanner results."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if not date_str and rankings:
        date_str = rankings[0].get("date")
    if not date_str:
        date_str = datetime.now().strftime("%Y-%m-%d")

    q5_count = sum(1 for r in rankings if "Q5" in str(r.get("quantile", "")))
    snapshot = {
        "date": date_str,
        "updated_at": datetime.now().isoformat(),
        "tickers_count": len(rankings),
        "q5_count": q5_count,
        "rankings": rankings,
    }
    out_file.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_file


def run_cloud_daily_pipeline(
    universe: list[TickerInfo] | None = None,
    start: str = "2010-01-01",
    force_download: bool = False,
    force_features: bool = False,
    force_train: bool = False,
    snapshot_path: Path | str = "reports/autonomous/latest_overnight_scanner_snapshot.json",
    webhook_url: str | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end cloud daily pipeline in an OS-independent, fault-tolerant manner.

    Steps:
      A. External market data & Core 20 tickers download & 163-features build.
      B. Self-healing model training for any missing overnight/turnaround models.
      C. 20-ticker Overnight prediction, SQLite save & JSON snapshot generation.
      D. Overnight Paper Trader portfolio simulation & ledger update.
      E. Value & Turnaround Scanner rankings & report update.
      F. Notification to Discord/Slack webhook (if configured).
    """
    from stock_ai.cli import (
        _build_features_for_ticker,
        predict_overnight_for_ticker,
        train_overnight_model_for_ticker,
    )
    from stock_ai.backtest.overnight_portfolio import (
        export_portfolio_simulation_reports,
        run_full_overnight_portfolio_backtest,
    )

    settings = load_settings()
    if universe is None:
        universe = load_universe()

    step_results: dict[str, Any] = {}
    failed_steps: list[str] = []

    print("=================================================================")
    print("☁️ [Stock-AI] Running Cloud Daily Autonomous Pipeline")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Universe: {len(universe)} tickers")
    print("=================================================================")

    # -------------------------------------------------------------
    # Step A: External Market Data & OHLCV Download & Features
    # -------------------------------------------------------------
    print("\n--- [Step A] Downloading External Data & Building Features ---")
    try:
        ext_report = download_external_market_data_report(
            settings.paths.raw_data_dir,
            start=start,
        )
        print(f"External market data status: {ext_report.status}")
        step_results["external_data"] = ext_report.status
    except Exception as exc:
        print(f"WARN: External data download encountered issue: {exc}")
        step_results["external_data"] = f"WARN: {exc}"

    ticker_features_ready = 0
    for idx, t in enumerate(universe, 1):
        code = t.code
        try:
            raw_path = settings.paths.raw_data_dir / f"{code}.parquet"
            if not raw_path.exists() or force_download:
                raw_df = fetch_daily_ohlcv(code, start=start)
                write_parquet(raw_df, raw_path)

            feat_path = settings.paths.feature_data_dir / f"{code}.parquet"
            needs_feat = (
                not feat_path.exists()
                or force_features
                or "target_next_open_binary" not in read_parquet(feat_path).columns
            )
            if needs_feat:
                _build_features_for_ticker(code, settings)
            ticker_features_ready += 1
        except Exception as exc:
            print(f"WARN: Failed feature prep for {code}: {exc}")

    step_results["features_ready_count"] = ticker_features_ready

    # -------------------------------------------------------------
    # Step B: Self-Healing Model Check & Training
    # -------------------------------------------------------------
    print("\n--- [Step B] Self-Healing Model Verification ---")
    trained_count = 0
    for t in universe:
        code = t.code
        m_file = settings.paths.model_dir / OVERNIGHT_ARTIFACT_FILENAME.format(ticker=code)
        if not m_file.exists() or force_train:
            print(f"Self-Healing: Training missing overnight model for {code}...")
            try:
                train_overnight_model_for_ticker(code, settings)
                trained_count += 1
            except Exception as exc:
                print(f"WARN: Failed self-healing training for {code}: {exc}")

    # Check Turnaround model self-healing
    turnaround_model_file = settings.paths.model_dir / TURNAROUND_ARTIFACT_FILENAME
    if not turnaround_model_file.exists() or force_train:
        print("Self-Healing: Training missing Turnaround LightGBM model...")
        try:
            train_turnaround_model(universe=universe)
            trained_count += 1
        except Exception as exc:
            print(f"WARN: Failed self-healing turnaround training: {exc}")

    step_results["models_self_healed"] = trained_count

    # -------------------------------------------------------------
    # Step C: Overnight Multi-Ticker Prediction & Snapshot
    # -------------------------------------------------------------
    print("\n--- [Step C] Predicting Overnight (Close-to-Open) Universe ---")
    scanner_rows: list[dict[str, Any]] = []
    latest_pred_date: str | None = None

    for t in universe:
        code = t.code
        name = t.name
        sector = getattr(t, "sector_ja", "") or getattr(t, "sector", "") or ""

        # Load metrics if available
        auc_val = None
        m_path = Path("reports") / "metrics" / f"lightgbm_overnight_binary_{code}.json"
        if m_path.exists():
            try:
                m_data = json.loads(m_path.read_text(encoding="utf-8"))
                auc_val = m_data.get("metrics", {}).get("auc")
            except Exception:
                pass

        try:
            count, preds = predict_overnight_for_ticker(code, settings)
            if not preds.empty:
                r = preds.iloc[0]
                p_up = float(r.get("probability_up", 0.5))
                raw_date = r.get("date")
                p_date = pd.Timestamp(raw_date).strftime("%Y-%m-%d") if pd.notna(raw_date) else "-"
                latest_pred_date = p_date

                notes_raw = r.get("notes")
                notes = json.loads(notes_raw) if notes_raw else {}
                if not notes:
                    notes = classify_overnight_quantile(p_up, ticker=code)

                q_band = notes.get("quantile", "-")
                q_label = notes.get("label", "")
                q_win = notes.get("historical_win_rate")

                raw_close = None
                feat_p = settings.paths.feature_data_dir / f"{code}.parquet"
                if feat_p.exists():
                    f_df = read_parquet(feat_p)
                    if not f_df.empty and "close" in f_df.columns:
                        raw_close = float(f_df["close"].iloc[-1])

                e_risk, e_date = check_earnings_risk(code, ref_date=p_date)
                scanner_rows.append({
                    "ticker": code,
                    "name": name,
                    "sector": sector,
                    "date": p_date,
                    "close": raw_close,
                    "probability_up": p_up,
                    "direction": "Up ⬆️" if p_up >= 0.5 else "Down ⬇️",
                    "q_band": q_band,
                    "quantile": f"{q_band} ({q_label})" if q_label else q_band,
                    "label": q_label,
                    "historical_win_rate": q_win,
                    "auc": auc_val,
                    "earnings_risk": e_risk,
                    "earnings_date": e_date,
                    "earnings_status": "⚠️ 決算警戒" if e_risk else "✅ 通常",
                })
        except Exception as exc:
            print(f"WARN: Overnight predict failed for {code}: {exc}")

    # Sort rankings by probability_up descending
    scanner_rows.sort(key=lambda x: x["probability_up"], reverse=True)
    for idx, row in enumerate(scanner_rows, 1):
        row["rank"] = idx
        row["is_top_pick"] = (idx == 1)

    # Save JSON snapshot
    saved_snapshot = save_overnight_scanner_snapshot(
        scanner_rows,
        output_path=snapshot_path,
        date_str=latest_pred_date,
    )
    print(f"Saved Overnight Scanner Snapshot: {saved_snapshot}")
    step_results["overnight_snapshot"] = str(saved_snapshot)
    step_results["overnight_count"] = len(scanner_rows)

    # -------------------------------------------------------------
    # Step D: Overnight Paper Trader Execution
    # -------------------------------------------------------------
    print("\n--- [Step D] Updating Overnight Paper Trader Portfolio ---")
    paper_metrics: dict[str, Any] = {}
    try:
        sim_data = run_full_overnight_portfolio_backtest(
            universe=universe,
            initial_capital=1_000_000.0,
        )
        export_portfolio_simulation_reports(
            sim_data,
            output_json_path="reports/autonomous/overnight_portfolio_simulation_results.json",
            output_md_path="reports/autonomous/overnight_portfolio_simulation_report.md",
        )
        if "results_by_cost" in sim_data and "5bps" in sim_data["results_by_cost"]:
            best_s = sim_data["results_by_cost"]["5bps"].get("elite_top2_long") or {}
            m = best_s.get("metrics", {})
            pending_b = sim_data.get("latest_pending", {}).get("strategies", {}).get("elite_top2_long", {})
            picks = pending_b.get("long_picks", [])
            paper_metrics = {
                "final_equity": m.get("final_equity", 1_000_000.0),
                "cumulative_return": m.get("cumulative_return", 0.0),
                "win_rate": m.get("win_rate", 0.0),
                "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                "max_drawdown": m.get("max_drawdown", 0.0),
                "next_positions": ", ".join(picks) if picks else "本日エントリーなし (キャッシュ待機判定)",
            }
        print("Paper trader portfolio simulation successfully updated.")
        step_results["paper_trader"] = "SUCCESS"
    except Exception as exc:
        print(f"WARN: Overnight paper trader update failed: {exc}")
        failed_steps.append("paper_trader")
        step_results["paper_trader"] = f"FAILED: {exc}"

    # -------------------------------------------------------------
    # Step E: Value & Turnaround Scanner Execution
    # -------------------------------------------------------------
    print("\n--- [Step E] Updating Value & Turnaround Scanner ---")
    turnaround_picks: list[dict[str, Any]] = []
    try:
        t_universe = load_turnaround_universe()
        t_data = scan_turnaround_universe(universe=t_universe)
        export_turnaround_reports(
            t_data,
            output_json_path="reports/autonomous/value_turnaround_scanner_results.json",
            output_md_path="reports/autonomous/value_turnaround_scanner_report.md",
        )
        turnaround_picks = t_data.get("rankings", [])
        print(f"Value & Turnaround scanner report successfully updated ({len(turnaround_picks)} tickers).")
        step_results["turnaround"] = "SUCCESS"
    except Exception as exc:
        print(f"WARN: Value & Turnaround scanner update failed: {exc}")
        failed_steps.append("turnaround")
        step_results["turnaround"] = f"FAILED: {exc}"

    # -------------------------------------------------------------
    # Step F: Webhook Notification
    # -------------------------------------------------------------
    print("\n--- [Step F] Webhook Notification ---")
    summary_for_webhook = {
        "date": latest_pred_date,
        "top_overnight": scanner_rows[:5],
        "paper_trader": paper_metrics,
        "top_turnaround": turnaround_picks[:3],
    }
    wh_result = send_daily_summary_webhook(summary_for_webhook, webhook_url=webhook_url)
    print(f"Webhook status: {wh_result.get('status')} ({wh_result.get('reason') or wh_result.get('error') or 'OK'})")
    step_results["webhook"] = wh_result

    status_str = "SUCCESS" if not failed_steps else "PARTIAL"
    print("\n=================================================================")
    print(f"✅ Cloud Daily Autonomous Pipeline Finished [{status_str}]")
    print("=================================================================")

    return {
        "status": status_str,
        "date": latest_pred_date,
        "steps": step_results,
        "top_overnight": scanner_rows[:5],
        "paper_trader": paper_metrics,
        "top_turnaround": turnaround_picks[:3],
    }
