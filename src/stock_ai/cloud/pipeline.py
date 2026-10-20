from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

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
from stock_ai.features.overnight import check_jumping_catch_risk
from stock_ai.models.intraday import (
    INTRADAY_ARTIFACT_FILENAME,
    predict_intraday_for_ticker,
    train_pooled_cross_sectional_intraday_model,
)
from stock_ai.models.overnight import (
    OVERNIGHT_ARTIFACT_FILENAME,
    POOLED_OVERNIGHT_ARTIFACT_FILENAME,
    classify_overnight_quantile,
    train_pooled_cross_sectional_overnight_model,
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


def _should_skip_webhook_delivery(pred_date: str, min_interval_minutes: float = 15.0) -> bool:
    """Check if an identical or recent webhook delivery occurred within min_interval_minutes."""
    state_file = Path("reports/autonomous/webhook_delivery_state.json")
    if not state_file.exists():
        return False
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        last_time_str = data.get("last_delivery_time")
        last_date = data.get("date")
        if last_time_str and last_date == pred_date:
            last_dt = datetime.fromisoformat(last_time_str)
            elapsed_minutes = (datetime.now(timezone.utc) - last_dt).total_seconds() / 60.0
            if 0 <= elapsed_minutes < min_interval_minutes:
                return True
    except Exception:
        pass
    return False


def _record_webhook_delivery(pred_date: str) -> None:
    """Record delivery timestamp to avoid duplicate alerts between cloud and local."""
    state_file = Path("reports/autonomous/webhook_delivery_state.json")
    state_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = {
            "date": pred_date,
            "last_delivery_time": datetime.now(timezone.utc).isoformat(),
        }
        state_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


def run_cloud_daily_pipeline(
    universe: list[TickerInfo] | None = None,
    start: str = "2010-01-01",
    force_download: bool = False,
    force_features: bool = False,
    force_train: bool = False,
    snapshot_path: Path | str = "reports/autonomous/latest_overnight_scanner_snapshot.json",
    webhook_url: str | None = None,
    force_webhook: bool = False,
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

    # Check Pooled Cross-Sectional Overnight model self-healing
    pooled_overnight_file = settings.paths.model_dir / POOLED_OVERNIGHT_ARTIFACT_FILENAME
    if not pooled_overnight_file.exists() or force_train:
        print("Self-Healing: Training missing Cross-Sectional Pooled Overnight LightGBM model...")
        try:
            train_pooled_cross_sectional_overnight_model(universe=universe)
            trained_count += 1
        except Exception as exc:
            print(f"WARN: Failed self-healing pooled overnight training: {exc}")

    # Check Pooled Cross-Sectional Intraday (07:30 JST) model self-healing
    intraday_model_file = settings.paths.model_dir / INTRADAY_ARTIFACT_FILENAME
    if not intraday_model_file.exists() or force_train:
        print("Self-Healing: Training missing 07:30 JST Pre-Market Intraday LightGBM model...")
        try:
            train_pooled_cross_sectional_intraday_model(universe=universe)
            trained_count += 1
        except Exception as exc:
            print(f"WARN: Failed self-healing intraday training: {exc}")

    step_results["models_self_healed"] = trained_count

    # -------------------------------------------------------------
    # Step C: Overnight Multi-Ticker Prediction & Snapshot
    # -------------------------------------------------------------
    print("\n--- [Step C] Predicting Overnight (Close-to-Open) Universe ---")
    scanner_rows: list[dict[str, Any]] = []
    latest_pred_date: str | None = None

    # Load Cross-Sectional Pooled Model if present
    pooled_bundle = None
    pooled_model_file = settings.paths.model_dir / POOLED_OVERNIGHT_ARTIFACT_FILENAME
    if pooled_model_file.exists():
        try:
            import joblib
            pooled_bundle = joblib.load(pooled_model_file)
        except Exception as exc:
            print(f"WARN: Could not load pooled overnight model: {exc}")

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
                p_cs = None
                is_jc_risk, jc_reason = False, ""
                feat_p = settings.paths.feature_data_dir / f"{code}.parquet"
                if feat_p.exists():
                    f_df = read_parquet(feat_p)
                    if not f_df.empty:
                        if "close" in f_df.columns:
                            raw_close = float(f_df["close"].iloc[-1])
                        is_jc_risk, jc_reason = check_jumping_catch_risk(f_df.iloc[-1])
                    if pooled_bundle is not None and not f_df.empty:
                        try:
                            p_cols = [c for c in pooled_bundle["feature_columns"] if c in f_df.columns]
                            if len(p_cols) == len(pooled_bundle["feature_columns"]):
                                p_cs = float(pooled_bundle["model"].predict_proba(f_df[p_cols].tail(1))[:, 1][0])
                        except Exception as e_cs:
                            logger.debug("Failed computing p_cs for %s: %s", code, e_cs)

                # Compute 07:30 JST Pre-Market Intraday prediction if available
                intraday_pred = predict_intraday_for_ticker(code, settings)
                p_intraday_cs = intraday_pred.get("p_intraday_cs") if intraday_pred else None
                pred_intraday_ret = intraday_pred.get("pred_intraday_return") if intraday_pred else None
                intraday_quantile = intraday_pred.get("quantile") if intraday_pred else None
                intraday_label = intraday_pred.get("label") if intraday_pred else None

                e_risk, e_date = check_earnings_risk(code, ref_date=p_date)
                scanner_rows.append({
                    "ticker": code,
                    "name": name,
                    "sector": sector,
                    "date": p_date,
                    "close": raw_close,
                    "probability_up": p_up,
                    "p_cs": p_cs,
                    "p_intraday_cs": p_intraday_cs,
                    "pred_intraday_return": pred_intraday_ret,
                    "intraday_quantile": intraday_quantile,
                    "intraday_label": intraday_label,
                    "direction": "Up ⬆️" if p_up >= 0.5 else "Down ⬇️",
                    "q_band": q_band,
                    "quantile": f"{q_band} ({q_label})" if q_label else q_band,
                    "label": q_label,
                    "historical_win_rate": q_win,
                    "auc": auc_val,
                    "earnings_risk": e_risk,
                    "earnings_date": e_date,
                    "earnings_status": "⚠️ 決算警戒" if e_risk else "✅ 通常",
                    "jumping_catch_risk": is_jc_risk,
                    "jumping_catch_reason": jc_reason,
                    "jumping_catch_status": "⚠️ 高値掴み警戒（過熱見送り）" if is_jc_risk else "✅ エントリー適格",
                })
        except Exception as exc:
            print(f"WARN: Overnight predict failed for {code}: {exc}")

    # Calculate Cross-Sectional rank & rank percentiles if available
    valid_cs = [r for r in scanner_rows if r.get("p_cs") is not None]
    if valid_cs:
        sorted_by_cs = sorted(valid_cs, key=lambda x: x["p_cs"], reverse=True)
        n = len(sorted_by_cs)
        for rank_idx, r in enumerate(sorted_by_cs, 1):
            r["cs_rank"] = rank_idx
            r["cs_rank_pct"] = round(1.0 - (rank_idx - 1) / max(1, n - 1), 2)

    # Sort rankings by Cross-Sectional outperform score if available, else probability_up
    has_cs = any(r.get("p_cs") is not None for r in scanner_rows)
    if has_cs:
        scanner_rows.sort(key=lambda x: (x.get("p_cs") is not None, x.get("p_cs") or 0.0), reverse=True)
    else:
        scanner_rows.sort(key=lambda x: x["probability_up"], reverse=True)

    # Set rank and select top pick (highest ranking candidate without earnings or jumping catch risk)
    top_pick_found = False
    for idx, row in enumerate(scanner_rows, 1):
        row["rank"] = idx
        if not top_pick_found and not row.get("earnings_risk") and not row.get("jumping_catch_risk"):
            row["is_top_pick"] = True
            top_pick_found = True
        else:
            row["is_top_pick"] = False

    # Fallback to top rank if all have warnings
    if not top_pick_found and scanner_rows:
        scanner_rows[0]["is_top_pick"] = True

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
        if "results_by_cost" in sim_data:
            res_20 = sim_data["results_by_cost"].get("20bps") or sim_data["results_by_cost"].get("5bps") or {}
            best_s = res_20.get("elite_top2_long", {})
            dual_s = res_20.get("dual_shift_compound", {})
            m = best_s.get("metrics", {})
            m_dual = dual_s.get("metrics", {})
            pending_b = sim_data.get("latest_pending", {}).get("strategies", {}).get("elite_top2_long", {})
            pending_dual = sim_data.get("latest_pending", {}).get("strategies", {}).get("dual_shift_compound", {})
            picks = pending_b.get("long_picks", [])
            intra_picks = pending_dual.get("intraday_picks", [])
            paper_metrics = {
                "final_equity": m_dual.get("final_equity", m.get("final_equity", 1_000_000.0)),
                "cumulative_return": m_dual.get("cumulative_return", m.get("cumulative_return", 0.0)),
                "overnight_equity": m.get("final_equity", 1_000_000.0),
                "overnight_return": m.get("cumulative_return", 0.0),
                "overnight_sharpe": m.get("sharpe_ratio", 0.0),
                "overnight_max_dd": m.get("max_drawdown", 0.0),
                "win_rate": m_dual.get("win_rate", m.get("win_rate", 0.0)),
                "sharpe_ratio": m_dual.get("sharpe_ratio", m.get("sharpe_ratio", 0.0)),
                "max_drawdown": m_dual.get("max_drawdown", m.get("max_drawdown", 0.0)),
                "profit_factor": m_dual.get("profit_factor", m.get("profit_factor", 1.0)),
                "next_positions": ", ".join(picks) if picks else "本日エントリーなし (キャッシュ待機判定)",
                "dual_shift_positions": (
                    f"🌙 夜間Overnight: {', '.join(picks) if picks else 'キャッシュ待機'} | "
                    f"☀️ 日中Intraday: {', '.join(intra_picks) if intra_picks else 'キャッシュ待機'}"
                ),
                "next_picks": picks,
                "next_intraday_picks": intra_picks,
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
    # Step F: Webhook Notification (with deduplication protection)
    # -------------------------------------------------------------
    print("\n--- [Step F] Webhook Notification ---")
    summary_for_webhook = {
        "date": latest_pred_date,
        "top_overnight": scanner_rows[:5],
        "paper_trader": paper_metrics,
        "top_turnaround": turnaround_picks[:3],
    }
    if not force_webhook and _should_skip_webhook_delivery(latest_pred_date, min_interval_minutes=15.0):
        print(f"INFO: Webhook notification skipped (already delivered within 15 minutes for {latest_pred_date}).")
        wh_result = {"status": "SKIPPED", "reason": f"Already delivered recently for {latest_pred_date}"}
    else:
        wh_result = send_daily_summary_webhook(summary_for_webhook, webhook_url=webhook_url)
        if wh_result.get("status") in ("SUCCESS", "OK", "PARTIAL") or wh_result.get("sent_count", 0) > 0:
            _record_webhook_delivery(latest_pred_date)
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


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Stock-AI Cloud Daily Autonomous Pipeline")
    parser.add_argument("--start", type=str, default="2010-01-01", help="Data start date")
    parser.add_argument(
        "--force-download",
        dest="force_download",
        action="store_true",
        default=True,
        help="Force re-download of raw market data (default: True)",
    )
    parser.add_argument(
        "--no-force-download",
        dest="force_download",
        action="store_false",
        help="Skip downloading raw market data if existing files are present",
    )
    parser.add_argument(
        "--force-features",
        dest="force_features",
        action="store_true",
        default=True,
        help="Force recalculation of features (default: True)",
    )
    parser.add_argument(
        "--no-force-features",
        dest="force_features",
        action="store_false",
        help="Skip recalculation of features if existing files are present",
    )
    parser.add_argument(
        "--force-train",
        dest="force_train",
        action="store_true",
        default=False,
        help="Force retraining models (default: False, self-healing retains trained models)",
    )
    parser.add_argument(
        "--webhook-url",
        type=str,
        default=None,
        help="Override webhook URL (default: uses configured webhooks in .env)",
    )
    parser.add_argument(
        "--force-webhook",
        dest="force_webhook",
        action="store_true",
        default=False,
        help="Force sending webhook even if delivered recently (default: False)",
    )
    args = parser.parse_args()

    pipeline_result = run_cloud_daily_pipeline(
        start=args.start,
        force_download=args.force_download,
        force_features=args.force_features,
        force_train=args.force_train,
        webhook_url=args.webhook_url,
        force_webhook=args.force_webhook,
    )
    if pipeline_result.get("status") not in ("SUCCESS", "PARTIAL"):
        sys.exit(1)

