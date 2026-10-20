from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from stock_ai.config import PROJECT_ROOT, ensure_project_dirs, load_settings
from stock_ai.data.universe import load_universe
from stock_ai.data.yfinance_provider import fetch_daily_ohlcv
from stock_ai.events.earnings import add_earnings_event_features, default_events_dir
from stock_ai.external.market_data import (
    add_external_market_features,
    download_external_market_data_report,
)
from stock_ai.features.technical import add_technical_features
from stock_ai.health.checks import add_health_check_parser, handle_health_check_command
from stock_ai.labels import distribution as label_distribution
from stock_ai.labels.targets import (
    add_next_close_binary_target,
    add_next_close_target,
    add_next_open_binary_target,
    add_open_to_close_target,
)
from stock_ai.narrative.rule_based import add_narrative_features
from stock_ai.news.manual_csv import add_news_features, default_news_dir
from stock_ai.storage.parquet_store import read_parquet, write_parquet
from stock_ai.storage.sqlite_store import upsert_predictions
from stock_ai.validation.time_split import chronological_train_valid_test_split


def main() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="replace")

    parser = argparse.ArgumentParser(description="stock-ai command line tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download daily OHLCV data")
    download.add_argument("--ticker", required=True)
    download.add_argument("--start", default=None)
    download.add_argument("--end", default=None)

    download_universe = subparsers.add_parser(
        "download-universe",
        help="Download daily OHLCV data for configured tickers",
    )
    download_universe.add_argument("--start", default=None)
    download_universe.add_argument("--end", default=None)
    download_universe.add_argument("--limit", type=int, default=None)

    download_external = subparsers.add_parser(
        "download-external",
        help="Download external market data such as USDJPY, Nikkei 225, and TOPIX",
    )
    download_external.add_argument("--start", default=None)
    download_external.add_argument("--end", default=None)
    download_external.add_argument("--symbols", nargs="*", default=None)

    fetch_news = subparsers.add_parser(
        "fetch-news",
        help="Fetch recent news into the manual news CSV format",
    )
    fetch_news.add_argument("--ticker", default="7203.T")
    fetch_news.add_argument("--query", default=None)
    fetch_news.add_argument("--query-profile", default=None)
    fetch_news.add_argument("--all-profiles", action="store_true")
    fetch_news.add_argument("--daily-profiles", action="store_true")
    fetch_news.add_argument("--limit", type=int, default=20)
    fetch_news.add_argument("--category", default="auto")
    fetch_news.add_argument("--language", default=None)
    fetch_news.add_argument("--region", default=None)

    check_news = subparsers.add_parser("check-news", help="Check news CSV quality")
    check_news.add_argument("--ticker", default="7203.T")

    refresh_news_scores = subparsers.add_parser(
        "refresh-news-scores",
        help="Re-score existing news CSV rows with the current scoring rules",
    )
    refresh_news_scores.add_argument("--ticker", default="7203.T")
    refresh_news_scores.add_argument("--dry-run", action="store_true")

    subparsers.add_parser("list-universe", help="Show configured tickers")

    build_features = subparsers.add_parser("build-features", help="Build feature dataset")
    build_features.add_argument("--ticker", required=True)

    data_freshness = subparsers.add_parser(
        "data-freshness",
        help="Show daily run data freshness for logs and operations checks",
    )
    data_freshness.add_argument("--ticker", default="7203.T")
    data_freshness.add_argument("--reference-date", default=None)

    build_features_universe = subparsers.add_parser(
        "build-features-universe",
        help="Build feature datasets for configured tickers",
    )
    build_features_universe.add_argument("--limit", type=int, default=None)

    train = subparsers.add_parser("train", help="Train first LightGBM model")
    train.add_argument("--ticker", required=True)

    train_binary = subparsers.add_parser("train-binary", help="Train LightGBM up/down model")
    train_binary.add_argument("--ticker", required=True)

    train_overnight = subparsers.add_parser(
        "train-overnight",
        help="Train LightGBM overnight (Close-to-Open gap) model",
    )
    train_overnight.add_argument("--ticker", required=True)

    train_multiclass = subparsers.add_parser(
        "train-multiclass",
        help="Train LightGBM 5-class next-close model",
    )
    train_multiclass.add_argument("--ticker", required=True)
    train_multiclass.add_argument(
        "--threshold-profile",
        default=label_distribution.DEFAULT_THRESHOLD_PROFILE,
        choices=label_distribution.threshold_profile_options(),
        help="5-class label threshold profile",
    )
    train_multiclass.add_argument(
        "--weight-mode",
        default=label_distribution.DEFAULT_WEIGHT_MODE,
        choices=label_distribution.weight_mode_options(),
        help="Multiclass rare-event class weighting mode",
    )

    backtest_binary = subparsers.add_parser(
        "backtest-binary",
        help="Run walk-forward LightGBM binary backtest",
    )
    backtest_binary.add_argument("--ticker", required=True)
    backtest_binary.add_argument("--start", default=None)
    backtest_binary.add_argument("--end", default=None)
    backtest_binary.add_argument("--train-years", type=int, default=3)
    backtest_binary.add_argument("--test-step-months", type=int, default=1)
    backtest_binary.add_argument("--threshold", type=float, default=0.55)

    offline_model_experiments = subparsers.add_parser(
        "offline-model-experiments",
        help="Run offline binary model improvement experiments without changing production models",
    )
    offline_model_experiments.add_argument("--ticker", required=True)
    offline_model_experiments.add_argument("--start", default=None)
    offline_model_experiments.add_argument("--end", default=None)
    offline_model_experiments.add_argument("--train-years", type=int, default=3)
    offline_model_experiments.add_argument("--test-step-months", type=int, default=1)
    offline_model_experiments.add_argument("--min-train-rows", type=int, default=252)
    offline_model_experiments.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    offline_model_experiments.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Defaults to the MVP candidate set.",
    )

    up_detection_experiments = subparsers.add_parser(
        "binary-up-detection-experiments",
        help="Run offline binary up-detection improvement experiments without production changes",
    )
    up_detection_experiments.add_argument("--ticker", required=True)
    up_detection_experiments.add_argument("--start", default=None)
    up_detection_experiments.add_argument("--end", default=None)
    up_detection_experiments.add_argument("--train-years", type=int, default=3)
    up_detection_experiments.add_argument("--test-step-months", type=int, default=1)
    up_detection_experiments.add_argument("--min-train-rows", type=int, default=252)
    up_detection_experiments.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    up_detection_experiments.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Defaults to the safe up-detection candidate set.",
    )
    up_detection_experiments.add_argument(
        "--skip-live-comparison",
        action="store_true",
        help="Do not read saved live binary rows for failure pattern comparison",
    )

    multiclass_walk_forward = subparsers.add_parser(
        "multiclass-walk-forward-diagnostics",
        help="Run historical 5-class walk-forward diagnostics without saving SQLite rows",
    )
    multiclass_walk_forward.add_argument("--ticker", required=True)
    multiclass_walk_forward.add_argument("--start", default=None)
    multiclass_walk_forward.add_argument("--end", default=None)
    multiclass_walk_forward.add_argument("--train-years", type=int, default=3)
    multiclass_walk_forward.add_argument("--test-step-months", type=int, default=1)
    multiclass_walk_forward.add_argument("--min-train-rows", type=int, default=252)
    multiclass_walk_forward.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    multiclass_walk_forward.add_argument(
        "--threshold-profile",
        default=label_distribution.DEFAULT_THRESHOLD_PROFILE,
        choices=label_distribution.threshold_profile_options(),
        help="5-class label threshold profile",
    )
    multiclass_walk_forward.add_argument(
        "--weight-mode",
        default=label_distribution.DEFAULT_WEIGHT_MODE,
        choices=label_distribution.weight_mode_options(),
        help="Multiclass rare-event class weighting mode",
    )
    multiclass_walk_forward.add_argument(
        "--skip-live-comparison",
        action="store_true",
        help="Do not read saved live multiclass rows for comparison",
    )
    multiclass_walk_forward.add_argument(
        "--skip-binary-comparison",
        action="store_true",
        help="Do not run the binary walk-forward reference comparison",
    )

    boundary_experiments = subparsers.add_parser(
        "multiclass-offline-boundary-experiments",
        help="Compare offline multiclass class-boundary and weighting candidates without production changes",
    )
    boundary_experiments.add_argument("--ticker", required=True)
    boundary_experiments.add_argument("--start", default=None)
    boundary_experiments.add_argument("--end", default=None)
    boundary_experiments.add_argument("--train-years", type=int, default=3)
    boundary_experiments.add_argument("--test-step-months", type=int, default=1)
    boundary_experiments.add_argument("--min-train-rows", type=int, default=252)
    boundary_experiments.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    boundary_experiments.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="Optional candidate names to run. Defaults to the safe offline boundary set.",
    )
    boundary_experiments.add_argument(
        "--skip-live-comparison",
        action="store_true",
        help="Do not read saved live multiclass rows for comparison",
    )
    boundary_experiments.add_argument(
        "--skip-binary-comparison",
        action="store_true",
        help="Do not run the binary walk-forward reference comparison",
    )

    predict = subparsers.add_parser("predict", help="Predict latest row and store it in SQLite")
    predict.add_argument("--ticker", required=True)
    predict.add_argument("--model-path", default=None)

    predict_binary = subparsers.add_parser("predict-binary", help="Predict latest up/down result")
    predict_binary.add_argument("--ticker", required=True)
    predict_binary.add_argument("--model-path", default=None)

    predict_overnight = subparsers.add_parser(
        "predict-overnight",
        help="Predict latest overnight (Close-to-Open gap) result with quantile bands and store in SQLite",
    )
    predict_overnight.add_argument("--ticker", default=None, help="Target ticker code (omit if using --all-universe)")
    predict_overnight.add_argument("--all-universe", action="store_true", help="Predict overnight for all tickers in universe")
    predict_overnight.add_argument("--model-path", default=None)

    sync_overnight = subparsers.add_parser(
        "sync-overnight-universe",
        help="Bulk sync (download, build features, train overnight model, calibrate quantiles, predict) for universe tickers",
    )
    sync_overnight.add_argument("--tickers", default=None, help="Comma-separated tickers (defaults to universe)")
    sync_overnight.add_argument("--start", default=None, help="Start date for downloads")
    sync_overnight.add_argument("--end", default=None, help="End date for downloads")
    sync_overnight.add_argument("--force-download", action="store_true", help="Force re-download raw data")
    sync_overnight.add_argument("--force-features", action="store_true", help="Force re-build 163 features")
    sync_overnight.add_argument("--force-train", action="store_true", help="Force re-train overnight model")
    sync_overnight.add_argument("--skip-train", action="store_true", help="Skip model training")
    sync_overnight.add_argument("--skip-predict", action="store_true", help="Skip prediction step")

    overnight_status = subparsers.add_parser(
        "overnight-status",
        help="Inspect overnight model artifact, metrics, and latest prediction status",
    )
    overnight_status.add_argument("--ticker", required=True)

    paper_trader = subparsers.add_parser(
        "run-overnight-paper-trader",
        help="Run 20-ticker overnight portfolio backtest simulation and update paper trader reports",
    )
    paper_trader.add_argument(
        "--output-json",
        default="reports/autonomous/overnight_portfolio_simulation_results.json",
        help="Output path for simulation JSON results",
    )
    paper_trader.add_argument(
        "--output-md",
        default="reports/autonomous/overnight_portfolio_simulation_report.md",
        help="Output path for simulation Markdown report",
    )
    paper_trader.add_argument(
        "--initial-capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital in JPY",
    )

    turnaround_scanner = subparsers.add_parser(
        "scan-turnaround-universe",
        help="Scan 50-ticker Prime universe for deep-value turnaround candidates and update reports",
    )
    turnaround_scanner.add_argument(
        "--output-json",
        default="reports/autonomous/value_turnaround_scanner_results.json",
        help="Output path for turnaround scanner JSON results",
    )
    turnaround_scanner.add_argument(
        "--output-md",
        default="reports/autonomous/value_turnaround_scanner_report.md",
        help="Output path for turnaround scanner Markdown report",
    )
    turnaround_scanner.add_argument(
        "--force-train",
        action="store_true",
        help="Force re-training of the turnaround LightGBM model",
    )

    cloud_pipeline = subparsers.add_parser(
        "run-cloud-daily-pipeline",
        help="Run end-to-end cloud daily pipeline (features, overnight predict, paper trader, turnaround, webhook)",
    )
    cloud_pipeline.add_argument("--start", default="2010-01-01", help="Start date for data downloads")
    cloud_pipeline.add_argument("--force-download", action="store_true", help="Force re-download raw data")
    cloud_pipeline.add_argument("--force-features", action="store_true", help="Force re-build features")
    cloud_pipeline.add_argument("--force-train", action="store_true", help="Force re-train models")
    cloud_pipeline.add_argument(
        "--snapshot-path",
        default="reports/autonomous/latest_overnight_scanner_snapshot.json",
        help="Output path for overnight scanner JSON snapshot",
    )
    cloud_pipeline.add_argument("--webhook-url", default=None, help="Discord/Slack webhook URL (or via STOCK_AI_WEBHOOK_URL)")

    shadow_predict = subparsers.add_parser(
        "binary-shadow-predict",
        help="Create binary shadow predictions in report files without writing SQLite",
    )
    shadow_predict.add_argument("--ticker", required=True)
    shadow_predict.add_argument("--model-path", default=None)
    shadow_predict.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    shadow_predict.add_argument("--dry-run", action="store_true")

    shadow_evaluate = subparsers.add_parser(
        "binary-shadow-evaluate",
        help="Evaluate binary shadow predictions and update pending results when actual data is available",
    )
    shadow_evaluate.add_argument("--ticker", required=True)
    shadow_evaluate.add_argument("--shadow-path", default=str(Path("reports") / "evaluation" / "binary_shadow_predictions.csv"))
    shadow_evaluate.add_argument("--output-dir", default=str(Path("reports") / "evaluation"))
    shadow_evaluate.add_argument("--dry-run", action="store_true")

    shadow_status = subparsers.add_parser(
        "binary-shadow-status",
        help="Inspect binary shadow lifecycle without generating predictions or writing files",
    )
    shadow_status.add_argument("--ticker", required=True)
    shadow_status.add_argument(
        "--shadow-path",
        default=str(Path("reports") / "evaluation" / "binary_shadow_predictions.csv"),
    )

    predict_multiclass = subparsers.add_parser(
        "predict-multiclass",
        help="Predict latest 5-class next-close result",
    )
    predict_multiclass.add_argument("--ticker", required=True)
    predict_multiclass.add_argument("--model-path", default=None)
    predict_multiclass.add_argument(
        "--threshold-profile",
        default=label_distribution.DEFAULT_THRESHOLD_PROFILE,
        choices=label_distribution.threshold_profile_options(),
        help="5-class label threshold profile",
    )
    predict_multiclass.add_argument(
        "--weight-mode",
        default=label_distribution.DEFAULT_WEIGHT_MODE,
        choices=label_distribution.weight_mode_options(),
        help="Multiclass rare-event class weighting mode",
    )

    backfill_predictions = subparsers.add_parser(
        "backfill-predictions",
        help="Create pseudo historical binary prediction history rows",
    )
    backfill_predictions.add_argument("--ticker", required=True)
    backfill_predictions.add_argument("--start", required=True)
    backfill_predictions.add_argument("--end", default=None)
    backfill_predictions.add_argument("--model-name", default="lightgbm_binary")
    backfill_predictions.add_argument("--min-train-rows", type=int, default=252)

    append_only_check = subparsers.add_parser(
        "prediction-append-only-check",
        help="Check append-only prediction tables without changing prediction save behavior",
    )
    append_only_check.add_argument("--ticker", default="7203.T")

    append_only_write_check = subparsers.add_parser(
        "prediction-append-only-write-check",
        help="Dry-run or manually write latest snapshot predictions to append-only tables",
    )
    append_only_write_check.add_argument("--ticker", default="7203.T")
    append_only_write_check.add_argument("--dry-run", action="store_true")
    append_only_write_check.add_argument("--write", action="store_true")
    append_only_write_check.add_argument("--database-path", default=None)
    append_only_write_check.add_argument("--run-id", default=None)
    append_only_write_check.add_argument("--run-type", default=None)
    append_only_write_check.add_argument("--official", action="store_true")
    append_only_write_check.add_argument("--latest", action="store_true")

    append_only_dual_write = subparsers.add_parser(
        "prediction-append-only-dual-write",
        help="Manually dual-write official latest snapshot rows to append-only tables",
    )
    append_only_dual_write.add_argument("--ticker", default="7203.T")
    append_only_dual_write.add_argument("--dry-run", action="store_true")
    append_only_dual_write.add_argument("--write", action="store_true")
    append_only_dual_write.add_argument("--database-path", default=None)
    append_only_dual_write.add_argument("--run-id", default=None)
    append_only_dual_write.add_argument("--idempotency-key", default=None)

    append_only_compare = subparsers.add_parser(
        "prediction-append-only-compare",
        help="Compare append-only latest official records with the current predictions snapshot",
    )
    append_only_compare.add_argument("--ticker", default="7203.T")
    append_only_compare.add_argument("--database-path", default=None)

    append_only_official_selection = subparsers.add_parser(
        "prediction-append-only-official-selection-check",
        help="Diagnose future append-only official candidates without updating the database",
    )
    append_only_official_selection.add_argument("--ticker", default="7203.T")
    append_only_official_selection.add_argument("--database-path", default=None)

    scheduled_contract_check = subparsers.add_parser(
        "prediction-append-only-scheduled-contract-check",
        help="Dry-run or opt-in test the scheduled append-only contract",
    )
    scheduled_contract_check.add_argument("--ticker", default="7203.T")
    scheduled_contract_check.add_argument("--feature-date", default=None)
    scheduled_contract_check.add_argument("--expected-feature-date", default=None)
    scheduled_contract_check.add_argument("--scheduled-for", default=None)
    scheduled_contract_check.add_argument("--execution-started-at", default=None)
    scheduled_contract_check.add_argument("--execution-completed-at", default=None)
    scheduled_contract_check.add_argument(
        "--execution-mode",
        default="on_schedule",
        choices=(
            "on_schedule",
            "delayed_start",
            "catch_up_after_missed_schedule",
            "manual",
            "retry",
            "unknown",
        ),
    )
    scheduled_contract_check.add_argument("--idempotency-key", default=None)
    scheduled_contract_check.add_argument("--database-path", default=None)
    scheduled_contract_check.add_argument("--dry-run", action="store_true")
    scheduled_contract_check.add_argument("--write", action="store_true")
    scheduled_contract_check.add_argument(
        "--binary-status",
        choices=("success", "failed"),
        default="success",
    )
    scheduled_contract_check.add_argument(
        "--multiclass-status",
        choices=("success", "failed"),
        default="success",
    )
    scheduled_contract_check.add_argument("--data-freshness-status", default=None)
    scheduled_contract_check.add_argument("--stale-at-run-time", action="store_true")
    scheduled_contract_check.add_argument("--task-name", default="stock-ai daily predict")
    scheduled_contract_check.add_argument("--task-last-result", type=int, default=None)
    scheduled_contract_check.add_argument("--notes", default=None)

    autonomous_operations = subparsers.add_parser(
        "autonomous-operations",
        help="Run the read-only Autonomous Operations monitoring cycle",
    )
    autonomous_operations.add_argument("--ticker", default="7203.T")
    autonomous_operations.add_argument(
        "--dry-run",
        action="store_true",
        help="Explicitly request read-only mode (the command is always read-only in v1)",
    )
    autonomous_operations.add_argument(
        "--state-path",
        default=str(Path("reports") / "autonomous" / "operations_state.json"),
    )
    autonomous_operations.add_argument(
        "--task-queue-path",
        default=str(Path("reports") / "autonomous" / "operations_task_queue.json"),
    )
    autonomous_operations.add_argument(
        "--report-dir",
        default=str(Path("reports") / "autonomous"),
    )
    autonomous_operations.add_argument("--max-runtime-minutes", type=float, default=10.0)
    autonomous_operations.add_argument(
        "--force",
        action="store_true",
        help="Remove only a stale operations lock; an active lock is never overridden",
    )

    autonomous_research = subparsers.add_parser(
        "autonomous-research",
        help="Run gated offline and shadow Research without production changes",
    )
    autonomous_research.add_argument("--ticker", default="7203.T")
    autonomous_research.add_argument("--dry-run", action="store_true")
    autonomous_research.add_argument("--force", action="store_true")
    autonomous_research.add_argument(
        "--state-path",
        default=str(Path("reports") / "autonomous" / "research_state.json"),
    )
    autonomous_research.add_argument(
        "--history-path",
        default=str(Path("reports") / "autonomous" / "research_history.jsonl"),
    )
    autonomous_research.add_argument(
        "--report-dir",
        default=str(Path("reports") / "autonomous"),
    )
    autonomous_research.add_argument("--max-runtime-minutes", type=float, default=120.0)
    autonomous_research.add_argument("--since-date", default=None)
    autonomous_research.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Limit Research to a supported candidate. Repeat to select multiple candidates.",
    )
    autonomous_research.add_argument("--skip-shadow", action="store_true")
    autonomous_research.add_argument("--skip-offline", action="store_true")

    autonomous_codex_agent = subparsers.add_parser(
        "autonomous-codex-agent",
        help="Run one gated Codex research-agent cycle without production changes",
    )
    autonomous_codex_agent.add_argument("--ticker", default="7203.T")
    autonomous_codex_agent.add_argument("--dry-run", action="store_true")
    autonomous_codex_agent.add_argument(
        "--state-path",
        default=str(Path("reports") / "autonomous" / "codex_research_agent_state.json"),
    )
    autonomous_codex_agent.add_argument(
        "--hypotheses-path",
        default=str(Path("reports") / "autonomous" / "codex_research_hypotheses.json"),
    )
    autonomous_codex_agent.add_argument(
        "--task-queue-path",
        default=str(Path("reports") / "autonomous" / "operations_task_queue.json"),
    )
    autonomous_codex_agent.add_argument(
        "--report-dir",
        default=str(Path("reports") / "autonomous"),
    )
    autonomous_codex_agent.add_argument(
        "--prompt-template",
        default=str(Path("prompts") / "autonomous_codex_research_agent.md"),
    )
    autonomous_codex_agent.add_argument(
        "--worktree-root",
        default=str(PROJECT_ROOT.parent / "stock-ai-worktrees"),
    )
    autonomous_codex_agent.add_argument("--max-runtime-minutes", type=float, default=90.0)
    autonomous_codex_agent.add_argument("--codex-executable", default="codex")
    autonomous_codex_agent.add_argument("--force", action="store_true")

    autonomous_codex_status = subparsers.add_parser(
        "autonomous-codex-research-status",
        help="Show read-only Codex research-agent due, lifecycle, and worktree status",
    )
    autonomous_codex_status.add_argument("--ticker", default="7203.T")
    autonomous_codex_status.add_argument(
        "--state-path",
        default=str(Path("reports") / "autonomous" / "codex_research_agent_state.json"),
    )
    autonomous_codex_status.add_argument(
        "--hypotheses-path",
        default=str(Path("reports") / "autonomous" / "codex_research_hypotheses.json"),
    )
    autonomous_codex_status.add_argument(
        "--task-queue-path",
        default=str(Path("reports") / "autonomous" / "operations_task_queue.json"),
    )
    autonomous_codex_status.add_argument(
        "--report-dir", default=str(Path("reports") / "autonomous")
    )
    autonomous_codex_status.add_argument(
        "--worktree-root", default=str(PROJECT_ROOT.parent / "stock-ai-worktrees")
    )
    autonomous_codex_status.add_argument("--force", action="store_true")

    autonomous_codex_cleanup = subparsers.add_parser(
        "autonomous-codex-research-cleanup",
        help="Audit a research worktree and remove it only after explicit confirmation",
    )
    autonomous_codex_cleanup.add_argument("--hypothesis-id", required=True)
    autonomous_codex_cleanup.add_argument("--confirm", action="store_true")
    autonomous_codex_cleanup.add_argument(
        "--hypotheses-path",
        default=str(Path("reports") / "autonomous" / "codex_research_hypotheses.json"),
    )
    autonomous_codex_cleanup.add_argument(
        "--report-dir", default=str(Path("reports") / "autonomous")
    )
    autonomous_codex_cleanup.add_argument(
        "--worktree-root", default=str(PROJECT_ROOT.parent / "stock-ai-worktrees")
    )

    add_health_check_parser(subparsers)

    args = parser.parse_args()
    settings = load_settings()
    ensure_project_dirs(settings)

    if args.command == "health-check":
        handle_health_check_command(args)
        return

    if args.command == "autonomous-operations":
        from stock_ai.operations.autonomous import run_autonomous_operations

        result = run_autonomous_operations(
            ticker=args.ticker,
            state_path=Path(args.state_path),
            task_queue_path=Path(args.task_queue_path),
            report_dir=Path(args.report_dir),
            max_runtime_minutes=args.max_runtime_minutes,
            force=args.force,
        )
        print("Autonomous Operations v1")
        print("mode: Level 0 read-only")
        print(f"ticker: {result['ticker']}")
        print(
            "classification: "
            f"{json.dumps(result['overall_classification'], ensure_ascii=False)}"
        )
        print(f"stop reason: {result['stop_reason']}")
        print(f"state: {Path(args.state_path).resolve()}")
        print(f"task queue: {Path(args.task_queue_path).resolve()}")
        print(f"markdown report: {result['report_paths']['markdown']}")
        print(f"json report: {result['report_paths']['json']}")
        print(f"recommended next action: {result['recommended_next_action']}")
        if result["stop_reason"] != "completed_read_only_cycle":
            raise SystemExit(2)
        return

    if args.command == "autonomous-research":
        from stock_ai.operations.autonomous_research import (
            ResearchLockError,
            ResearchStateError,
            ResearchTimeoutError,
            run_autonomous_research,
        )

        try:
            result = run_autonomous_research(
                ticker=args.ticker,
                state_path=Path(args.state_path),
                history_path=Path(args.history_path),
                report_dir=Path(args.report_dir),
                max_runtime_minutes=args.max_runtime_minutes,
                since_date=args.since_date,
                candidate_names=args.candidate,
                skip_shadow=args.skip_shadow,
                skip_offline=args.skip_offline,
                dry_run=args.dry_run,
                force=args.force,
            )
        except ResearchTimeoutError as error:
            print(f"autonomous research timeout: {error}")
            raise SystemExit(124) from error
        except ResearchLockError as error:
            print(f"autonomous research dependency/lock stop: {error}")
            raise SystemExit(125) from error
        except ResearchStateError as error:
            print(f"autonomous research state failure: {error}")
            raise SystemExit(1) from error
        print("Autonomous Research v1")
        print(f"ticker: {result['ticker']}")
        print(f"classification: {result['classification']}")
        print(f"experiment executed: {str(result['experiment_executed']).lower()}")
        print(f"experiment fingerprint: {result['experiment_fingerprint']}")
        print(f"trigger reasons: {json.dumps(result['trigger_reasons'], ensure_ascii=False)}")
        print(f"recommendation: {result['recommendation']}")
        print(f"markdown report: {result['report_paths']['markdown']}")
        print(f"json report: {result['report_paths']['json']}")
        print("production change applied: false")
        if int(result.get("exit_code", 0)):
            raise SystemExit(int(result["exit_code"]))
        return

    if args.command == "autonomous-codex-research-status":
        from stock_ai.operations.autonomous_codex_agent import autonomous_codex_research_status

        result = autonomous_codex_research_status(
            ticker=args.ticker,
            state_path=Path(args.state_path),
            hypotheses_path=Path(args.hypotheses_path),
            task_queue_path=Path(args.task_queue_path),
            report_dir=Path(args.report_dir),
            worktree_root=Path(args.worktree_root),
            force=args.force,
        )
        due = result["due"]
        print("Autonomous Codex Research Agent v1.1 preflight")
        print(f"ticker: {result['ticker']}")
        print(f"main clean: {str(result['main_git'].get('clean')).lower()}")
        print(f"active lock: {str(result['lock_active']).lower()}")
        print(f"gate healthy: {str(result['gate'].get('healthy')).lower()}")
        print(f"latest research fingerprint: {due.get('latest_research_fingerprint')}")
        print(f"last agent research fingerprint: {due.get('last_agent_research_fingerprint')}")
        print(f"live confirmed delta: {due.get('live_confirmed_delta')}")
        print(f"shadow confirmed delta: {due.get('shadow_confirmed_delta')}")
        print(f"approval-required tasks: {len(due.get('new_approval_required_tasks') or [])}")
        print(
            "selected next hypothesis: "
            f"{(due.get('selected_next_hypothesis') or {}).get('hypothesis_id') or 'none'}"
        )
        print(f"classification: {due.get('classification')}")
        print(f"due: {str(due.get('due')).lower()}")
        print(f"Codex invocation required: {str(due.get('codex_invocation_required')).lower()}")
        print(f"worktree creation required: {str(due.get('worktree_creation_required')).lower()}")
        print(f"reason codes: {json.dumps(due.get('reason_codes') or [], ensure_ascii=False)}")
        print(f"worktrees: {json.dumps(result.get('worktrees') or [], ensure_ascii=False)}")
        return

    if args.command == "autonomous-codex-research-cleanup":
        from stock_ai.operations.autonomous_codex_agent import (
            cleanup_research_worktree,
            load_hypothesis_backlog,
        )

        backlog, _ = load_hypothesis_backlog(Path(args.hypotheses_path))
        hypothesis = next(
            (
                item
                for item in backlog.get("hypotheses", [])
                if item.get("hypothesis_id") == args.hypothesis_id
            ),
            None,
        )
        if hypothesis is None:
            raise SystemExit(f"hypothesis not found: {args.hypothesis_id}")
        result = cleanup_research_worktree(
            hypothesis,
            worktree_root=Path(args.worktree_root),
            report_dir=Path(args.report_dir),
            confirm=args.confirm,
        )
        print("Autonomous Codex research worktree cleanup")
        print(f"mode: {result['mode']}")
        print(f"status: {result['audit']['lifecycle_status']}")
        print(f"worktree: {result['audit']['path']}")
        print(f"would remove: {str(result['would_remove']).lower()}")
        print(f"removed: {str(result['removed']).lower()}")
        print(f"reason: {result['reason']}")
        print(f"audit: {json.dumps(result['audit'], ensure_ascii=False)}")
        return

    if args.command == "autonomous-codex-agent":
        from stock_ai.operations.autonomous_codex_agent import (
            CodexAgentError,
            CodexAgentLockError,
            CodexAgentTimeoutError,
            run_autonomous_codex_research_agent,
        )

        try:
            result = run_autonomous_codex_research_agent(
                ticker=args.ticker,
                state_path=Path(args.state_path),
                hypotheses_path=Path(args.hypotheses_path),
                task_queue_path=Path(args.task_queue_path),
                report_dir=Path(args.report_dir),
                prompt_template_path=Path(args.prompt_template),
                worktree_root=Path(args.worktree_root),
                max_runtime_minutes=args.max_runtime_minutes,
                codex_executable=args.codex_executable,
                dry_run=args.dry_run,
                force=args.force,
            )
        except CodexAgentTimeoutError as error:
            print(f"autonomous Codex Research Agent timeout: {error}")
            raise SystemExit(124) from error
        except CodexAgentLockError as error:
            print(f"autonomous Codex Research Agent lock stop: {error}")
            raise SystemExit(125) from error
        except CodexAgentError as error:
            print(f"autonomous Codex Research Agent failure: {error}")
            raise SystemExit(1) from error
        print("Autonomous Codex Research Agent v1.1")
        print(f"ticker: {result['ticker']}")
        print(f"gate healthy: {str(result['gate']['healthy']).lower()}")
        print(
            "selected hypothesis: "
            f"{(result.get('selected_hypothesis') or {}).get('hypothesis_id') or 'none'}"
        )
        print(f"classification: {result['classification']}")
        print(f"stop reason: {result['stop_reason']}")
        print(f"worktree: {(result.get('worktree') or {}).get('path') or 'not created'}")
        print(f"markdown report: {result['report_paths']['markdown']}")
        print(f"json report: {result['report_paths']['json']}")
        print("production change applied: false")
        if int(result.get("exit_code", 0)):
            raise SystemExit(int(result["exit_code"]))
        return

    if args.command == "download":
        start = args.start or settings.data.get("default_start", "2010-01-01")
        data = fetch_daily_ohlcv(args.ticker, start=start, end=args.end)
        output_path = settings.paths.raw_data_dir / f"{args.ticker}.parquet"
        write_parquet(data, output_path)
        print(f"saved: {output_path}")
        return

    if args.command == "download-universe":
        start = args.start or settings.data.get("default_start", "2010-01-01")
        tickers = load_universe()
        if args.limit is not None:
            tickers = tickers[: args.limit]
        for ticker in tickers:
            data = fetch_daily_ohlcv(ticker.code, start=start, end=args.end)
            output_path = settings.paths.raw_data_dir / f"{ticker.code}.parquet"
            write_parquet(data, output_path)
            print(f"saved: {output_path}")
        return

    if args.command == "download-external":
        start = args.start or settings.data.get("default_start", "2010-01-01")
        report = download_external_market_data_report(
            settings.paths.raw_data_dir,
            start=start,
            end=args.end,
            symbols=args.symbols,
        )
        for output_path in report.successful_paths:
            print(f"saved: {output_path}")
        if report.status == "WARN":
            print("download-external completed with WARN. Existing Toyota prediction flow can continue.")
        if report.status == "FAIL":
            raise SystemExit("download-external failed: all external symbols failed.")
        return

    if args.command == "fetch-news":
        from stock_ai.news.fetcher import fetch_google_news_rss, ticker_default_query
        from stock_ai.news.query_profiles import (
            daily_news_query_profiles,
            get_news_query_profile,
            load_news_query_profiles,
        )
        from stock_ai.news.profile_quality import append_fetch_history

        news_dir = default_news_dir(settings.paths.raw_data_dir.parent)
        if args.all_profiles and args.daily_profiles:
            raise SystemExit("Use either --all-profiles or --daily-profiles, not both.")
        if args.all_profiles or args.daily_profiles:
            profile_mode = "all profiles" if args.all_profiles else "daily profiles"
            profiles = load_news_query_profiles() if args.all_profiles else daily_news_query_profiles()
            if not profiles:
                raise SystemExit(f"No news query profiles are configured for {profile_mode}.")
            print(f"news profile mode: {profile_mode}")
            total_fetched = 0
            total_added = 0
            total_duplicates = 0
            for profile in profiles:
                try:
                    result = fetch_google_news_rss(
                        query=profile.query,
                        ticker=profile.ticker or args.ticker,
                        save_ticker=profile.storage_ticker(args.ticker),
                        item_ticker=profile.row_ticker(),
                        news_dir=news_dir,
                        limit=args.limit,
                        category=profile.category,
                        query_profile=profile.name,
                        language=profile.language or args.language or "en-US",
                        region=profile.region or args.region or "US",
                    )
                except Exception as exc:
                    append_fetch_history(
                        news_dir,
                        profile=profile,
                        query=profile.query,
                        error=str(exc),
                    )
                    raise
                append_fetch_history(
                    news_dir,
                    profile=profile,
                    query=profile.query,
                    fetched_count=result.fetched_count,
                    added_count=result.added_count,
                    duplicate_count=result.duplicate_count,
                )
                _print_fetch_news_result(
                    result,
                    profile_name=profile.name,
                    storage_ticker=profile.storage_ticker(args.ticker),
                    row_ticker=profile.row_ticker(),
                )
                total_fetched += result.fetched_count
                total_added += result.added_count
                total_duplicates += result.duplicate_count
            print(f"news profiles fetched: {len(profiles)}")
            print(f"news profiles total fetched: {total_fetched}")
            print(f"news profiles total added: {total_added}")
            print(f"news profiles total duplicates: {total_duplicates}")
            return

        profile_name = None
        storage_ticker = args.ticker
        row_ticker = args.ticker
        category = args.category
        language = args.language or "en-US"
        region = args.region or "US"
        profile_obj = None
        if args.query_profile:
            profile = get_news_query_profile(args.query_profile)
            profile_obj = profile
            profile_name = profile.name
            query = args.query or profile.query
            storage_ticker = profile.storage_ticker(args.ticker)
            row_ticker = profile.row_ticker()
            category = profile.category
            language = profile.language or language
            region = profile.region or region
            ticker_for_scoring = profile.ticker or args.ticker
        else:
            query = args.query or ticker_default_query(args.ticker)
            ticker_for_scoring = args.ticker

        try:
            result = fetch_google_news_rss(
                query=query,
                ticker=ticker_for_scoring,
                save_ticker=storage_ticker,
                item_ticker=row_ticker,
                news_dir=news_dir,
                limit=args.limit,
                category=category,
                query_profile=profile_name,
                language=language,
                region=region,
            )
        except Exception as exc:
            append_fetch_history(
                news_dir,
                profile=profile_obj,
                query=query,
                error=str(exc),
            )
            raise
        append_fetch_history(
            news_dir,
            profile=profile_obj,
            query=query,
            fetched_count=result.fetched_count,
            added_count=result.added_count,
            duplicate_count=result.duplicate_count,
        )
        _print_fetch_news_result(
            result,
            profile_name=profile_name,
            storage_ticker=storage_ticker,
            row_ticker=row_ticker,
        )
        return

    if args.command == "check-news":
        from stock_ai.news.profile_quality import update_latest_fetch_history_quality
        from stock_ai.news.quality import check_news_for_ticker

        news_dir = default_news_dir(settings.paths.raw_data_dir.parent)
        report = check_news_for_ticker(news_dir, args.ticker)
        updated_history_path = update_latest_fetch_history_quality(news_dir, report)
        print(f"news quality status: {report['status']}")
        print(f"news quality message: {report['message']}")
        print(f"news path: {report['path']}")
        print(f"news rows: {report['rows']}")
        print(f"latest news date: {report['latest_news_date'] or '-'}")
        print(f"latest fetched_at: {report['latest_fetched_at'] or '-'}")
        for key in (
            "duplicate_count",
            "invalid_date_count",
            "mojibake_suspect_count",
            "empty_title_count",
            "empty_source_count",
            "empty_url_count",
            "missing_fetched_at_count",
            "missing_category_count",
            "missing_sentiment_reason_count",
            "missing_importance_reason_count",
            "missing_source_credibility_count",
            "missing_source_market_weight_count",
            "unknown_source_count",
            "source_credibility_out_of_range_count",
            "source_weight_out_of_range_count",
            "sentiment_out_of_range_count",
            "importance_out_of_range_count",
        ):
            print(f"{key}: {report[key]}")
        print(
            "unknown_sources: "
            f"{json.dumps(report.get('unknown_sources') or [], ensure_ascii=False)}"
        )
        print(
            "fetch_history quality update: "
            f"{updated_history_path if updated_history_path is not None else '-'}"
        )
        print(f"sentiment range: {report['sentiment_range_status']}")
        print(f"importance range: {report['importance_range_status']}")
        print(f"source credibility range: {report['source_credibility_range_status']}")
        print(f"source weight range: {report['source_weight_range_status']}")
        print(f"source type distribution: {json.dumps(report['source_type_distribution'], ensure_ascii=False)}")
        issue_rows = report.get("issue_rows") or []
        if issue_rows:
            print("issue rows:")
            for row in issue_rows[:10]:
                print(
                    f"- {row.get('date')}: {row.get('issues')} | "
                    f"{row.get('title')} | {row.get('url')}"
                )
        return

    if args.command == "refresh-news-scores":
        from stock_ai.news.refresh_scores import refresh_news_scores_for_ticker

        result = refresh_news_scores_for_ticker(
            default_news_dir(settings.paths.raw_data_dir.parent),
            args.ticker,
            dry_run=args.dry_run,
        )
        print(f"news refresh path: {result.path}")
        print(f"dry run: {result.dry_run}")
        print(f"target rows: {result.row_count}")
        print(f"rows with changes: {result.updated_count}")
        print(f"category distribution: {json.dumps(result.category_distribution, ensure_ascii=False)}")
        print(f"sentiment distribution: {json.dumps(result.sentiment_distribution, ensure_ascii=False)}")
        print(
            "importance average: "
            f"{result.importance_average:.3f}" if result.importance_average is not None else "importance average: -"
        )
        if result.change_examples:
            print("change examples:")
            for example in result.change_examples:
                print(json.dumps(example, ensure_ascii=False, indent=2))
        if result.dry_run:
            print("dry-run note: no file was written.")
        else:
            print("refresh complete: news scores were written. Run check-news to verify quality.")
        return

    if args.command == "list-universe":
        for ticker in load_universe():
            print(f"{ticker.code}\t{ticker.name}")
        return

    if args.command == "build-features":
        output_path = _build_features_for_ticker(args.ticker, settings)
        print(f"saved: {output_path}")
        return

    if args.command == "data-freshness":
        from stock_ai.operations.freshness import summarize_daily_data_freshness

        summary = summarize_daily_data_freshness(
            ticker=args.ticker,
            raw_data_dir=settings.paths.raw_data_dir,
            feature_data_dir=settings.paths.feature_data_dir,
            reference_date=args.reference_date,
        )
        print(f"data freshness ticker: {summary['ticker']}")
        print(f"expected latest trading date: {summary['expected_latest_trading_date']}")
        print(f"latest raw date: {summary['latest_raw_date'] or '-'}")
        print(f"latest feature date: {summary['latest_feature_date'] or '-'}")
        print(f"prediction feature date: {summary['prediction_feature_date'] or '-'}")
        print(f"saved prediction date expected: {summary['saved_prediction_date_expected'] or '-'}")
        print(f"raw lag trading days: {summary['raw_lag_trading_days']}")
        print(f"freshness lag trading days: {summary['freshness_lag_trading_days']}")
        print(f"stale at run time: {str(summary['stale_at_run_time']).lower()}")
        print(f"freshness status: {summary['freshness_status']}")
        print(f"freshness message: {summary['freshness_message']}")
        return

    if args.command == "prediction-append-only-check":
        from stock_ai.backtest.append_only_history import build_append_only_health_visibility
        from stock_ai.storage.sqlite_store import init_database, summarize_append_only_records

        init_database(settings.paths.database_path)
        summary = summarize_append_only_records(
            settings.paths.database_path,
            ticker=args.ticker,
        )
        visibility = build_append_only_health_visibility(
            settings.paths.database_path,
            ticker=args.ticker,
        )
        print("prediction append-only check")
        print(f"database path: {summary['database_path']}")
        print(f"database exists: {str(summary['database_exists']).lower()}")
        print(
            "append-only tables exist: "
            f"{str(summary['append_only_tables_exist']).lower()}"
        )
        print(f"prediction_runs rows: {summary['prediction_runs_count']}")
        print(f"prediction_records rows: {summary['prediction_records_count']}")
        print(f"existing predictions rows: {summary['existing_predictions_count']}")
        print(f"official records rows: {summary['official_records_count']}")
        print(f"latest records rows: {summary['latest_records_count']}")
        print(f"source counts: {json.dumps(summary['source_counts'], ensure_ascii=False)}")
        print(f"run type counts: {json.dumps(summary['run_type_counts'], ensure_ascii=False)}")
        print(f"visibility status: {visibility['visibility_status']}")
        print(
            "visibility classifications: "
            f"{json.dumps(visibility['visibility_classifications'], ensure_ascii=False)}"
        )
        print(f"latest run id: {visibility['latest_run_id'] or '-'}")
        print(f"latest run timestamp: {visibility['latest_run_timestamp'] or '-'}")
        print(f"latest run type: {visibility['latest_run_type'] or '-'}")
        print(f"latest feature date: {visibility['latest_feature_date'] or '-'}")
        print(
            "append-only vs snapshot matches: "
            f"{str(visibility['append_only_vs_snapshot_matches']).lower()}"
        )
        print(f"mismatch count: {visibility['mismatch_count']}")
        print(f"pending count: {visibility['pending_count']}")
        print(f"hit count: {visibility['hit_count']}")
        print(f"miss count: {visibility['miss_count']}")
        print(f"unevaluated count: {visibility['unevaluated_count']}")
        print(f"latest compare status: {visibility['latest_compare_status']}")
        print("visibility note: append-only is informational and is not an Action required source.")
        if summary["prediction_runs_count"] == 0 and summary["prediction_records_count"] == 0:
            print("status: OK - append-only tables are initialized but not yet populated.")
        else:
            print("status: OK - append-only tables are readable.")
        print("note: existing predictions remain the latest snapshot source.")
        return

    if args.command == "prediction-append-only-write-check":
        from stock_ai.storage.sqlite_store import (
            filter_official_prediction_snapshot_records,
            read_latest_predictions,
            summarize_append_only_records,
            write_append_only_prediction_run,
        )

        database_path = Path(args.database_path) if args.database_path else settings.paths.database_path
        dry_run = not args.write
        run_type = args.run_type or ("dry_run" if dry_run else "manual_live")
        latest = read_latest_predictions(database_path)
        if not latest.empty:
            latest = latest[latest["ticker"] == args.ticker].reset_index(drop=True)
        included, excluded = filter_official_prediction_snapshot_records(
            latest,
            ticker=args.ticker,
        )
        before = summarize_append_only_records(database_path, ticker=args.ticker)
        print("prediction append-only write check")
        print(f"database path: {database_path}")
        print(f"ticker: {args.ticker}")
        print(f"mode: {'dry-run' if dry_run else 'write'}")
        print(f"run type: {run_type}")
        print(f"run id: {args.run_id or '(new)'}")
        print(f"raw latest snapshot candidates: {len(latest)}")
        print(f"included official candidates: {len(included)}")
        print(f"excluded candidates: {len(excluded)}")
        print(f"append-only runs before: {before['prediction_runs_count']}")
        print(f"append-only records before: {before['prediction_records_count']}")
        print(f"existing predictions rows: {before['existing_predictions_count']}")
        if latest.empty:
            print("status: OK - no latest snapshot rows were available for append-only candidates.")
            return
        print("included records:")
        if included.empty:
            print("  (none)")
        for row in included.itertuples(index=False):
            print(
                "  included: "
                f"feature_date={row.date}, target={row.target}, model={row.model_name}, "
                f"model_type={getattr(row, 'model_type', '-')}, "
                f"predicted_class={row.predicted_class}, source={getattr(row, 'snapshot_source', '-')}, "
                f"reason={getattr(row, 'official_filter_reason', '-')}"
            )
        print("excluded records:")
        if excluded.empty:
            print("  (none)")
        for row in excluded.itertuples(index=False):
            print(
                "  excluded: "
                f"feature_date={row.date}, target={row.target}, model={row.model_name}, "
                f"model_type={getattr(row, 'model_type', '-')}, "
                f"predicted_class={row.predicted_class}, source={getattr(row, 'snapshot_source', '-')}, "
                f"reason={getattr(row, 'excluded_reason', '-')}"
            )
        if dry_run:
            print("status: OK - dry-run only; append-only tables were not written.")
            print("note: use --write for manual append-only verification writes.")
            return
        if included.empty:
            print("status: OK - no official candidates to write.")
            print("note: existing predictions remain the latest snapshot source.")
            return
        result = write_append_only_prediction_run(
            database_path,
            included,
            ticker=args.ticker,
            run_type=run_type,
            command_name="prediction-append-only-write-check",
            source="live",
            run_id=args.run_id,
            working_directory=str(Path.cwd()),
            is_official=True,
            is_latest=True,
            notes="manual append-only write check",
        )
        after = result["after"]
        print(f"created run id: {result['run_id']}")
        print(f"record candidates: {result['record_candidates']}")
        print(f"inserted records: {result['inserted_records']}")
        print(f"duplicate records ignored: {result['duplicate_records']}")
        print(f"append-only runs after: {after['prediction_runs_count']}")
        print(f"append-only records after: {after['prediction_records_count']}")
        print(f"existing predictions rows after: {after['existing_predictions_count']}")
        print("status: OK - append-only manual verification write completed.")
        print("note: existing predictions remain the latest snapshot source.")
        return

    if args.command == "prediction-append-only-dual-write":
        from stock_ai.operations.freshness import summarize_daily_data_freshness
        from stock_ai.storage.sqlite_store import (
            filter_official_prediction_snapshot_records,
            read_latest_predictions,
            summarize_append_only_records,
            write_append_only_prediction_run,
        )

        if args.dry_run and args.write:
            raise SystemExit("Use either --dry-run or --write, not both.")
        database_path = Path(args.database_path) if args.database_path else settings.paths.database_path
        dry_run = not args.write
        latest = read_latest_predictions(database_path)
        if not latest.empty:
            latest = latest[latest["ticker"] == args.ticker].reset_index(drop=True)
        included, excluded = filter_official_prediction_snapshot_records(latest, ticker=args.ticker)
        before = summarize_append_only_records(database_path, ticker=args.ticker)
        freshness = summarize_daily_data_freshness(
            ticker=args.ticker,
            raw_data_dir=settings.paths.raw_data_dir,
            feature_data_dir=settings.paths.feature_data_dir,
        )
        run_id = args.run_id or _append_only_dual_write_run_id(
            args.ticker,
            args.idempotency_key,
        )

        print("prediction append-only dual-write")
        print(f"database path: {database_path}")
        print(f"ticker: {args.ticker}")
        print(f"mode: {'dry-run' if dry_run else 'write'}")
        print("run type: manual_live_dual_write")
        print(f"run id: {run_id or '(new)'}")
        print(f"raw latest snapshot candidates: {len(latest)}")
        print(f"included official candidates: {len(included)}")
        print(f"excluded candidates: {len(excluded)}")
        print(f"append-only runs before: {before['prediction_runs_count']}")
        print(f"append-only records before: {before['prediction_records_count']}")
        print(f"existing predictions rows: {before['existing_predictions_count']}")
        print(f"data freshness status: {freshness['freshness_status']}")
        print(f"latest raw date: {freshness['latest_raw_date'] or '-'}")
        print(f"latest feature date: {freshness['latest_feature_date'] or '-'}")
        print(f"expected latest trading date: {freshness['expected_latest_trading_date']}")
        print(f"stale at run time: {str(freshness['stale_at_run_time']).lower()}")
        _print_official_filter_rows(included, excluded)

        if dry_run:
            print("status: OK - dry-run only; append-only tables were not written.")
            return
        if included.empty:
            print("status: OK - no official candidates to write.")
            return
        result = write_append_only_prediction_run(
            database_path,
            included,
            ticker=args.ticker,
            run_type="manual_live_dual_write",
            command_name="prediction-append-only-dual-write",
            source="live",
            run_id=run_id,
            working_directory=str(Path.cwd()),
            data_freshness_status=str(freshness["freshness_status"]),
            latest_raw_date=freshness["latest_raw_date"],
            latest_feature_date=freshness["latest_feature_date"],
            expected_latest_trading_date=freshness["expected_latest_trading_date"],
            stale_at_run_time=bool(freshness["stale_at_run_time"]),
            is_official=True,
            is_latest=True,
            notes="manual controlled append-only dual-write",
        )
        after = result["after"]
        print(f"created run id: {result['run_id']}")
        print(f"record candidates: {result['record_candidates']}")
        print(f"inserted records: {result['inserted_records']}")
        print(f"duplicate records ignored: {result['duplicate_records']}")
        print(f"append-only runs after: {after['prediction_runs_count']}")
        print(f"append-only records after: {after['prediction_records_count']}")
        print(f"existing predictions rows after: {after['existing_predictions_count']}")
        print("status: OK - manual append-only dual-write completed.")
        print("note: existing predictions remain the latest snapshot source.")
        return

    if args.command == "prediction-append-only-compare":
        from stock_ai.storage.sqlite_store import compare_append_only_official_to_snapshot

        database_path = Path(args.database_path) if args.database_path else settings.paths.database_path
        comparison = compare_append_only_official_to_snapshot(
            database_path,
            ticker=args.ticker,
        )
        print("prediction append-only compare")
        print(f"database path: {database_path}")
        print(f"ticker: {args.ticker}")
        print(f"snapshot official count: {comparison['snapshot_official_count']}")
        print(f"snapshot excluded count: {comparison['snapshot_excluded_count']}")
        print(f"append official count: {comparison['append_official_count']}")
        print(f"matches: {str(comparison['matches']).lower()}")
        for row in comparison["rows"]:
            print(
                "compare: "
                f"feature_date={row['feature_date']}, target={row['target']}, "
                f"model={row['model_name']}, snapshot_label={row['snapshot_predicted_label']}, "
                f"append_label={row['append_predicted_label']}, "
                f"snapshot_confidence={row['snapshot_confidence']}, "
                f"append_confidence={row['append_confidence']}, "
                f"run_id={row['append_run_id']}, run_timestamp={row['append_run_timestamp']}, "
                f"match={str(row['match']).lower()}, reason={row['reason']}"
            )
        return

    if args.command == "prediction-append-only-official-selection-check":
        from stock_ai.backtest.append_only_policy import (
            build_official_selection_diagnostics_from_db,
        )

        database_path = Path(args.database_path) if args.database_path else settings.paths.database_path
        diagnostics = build_official_selection_diagnostics_from_db(
            database_path,
            ticker=args.ticker,
        )
        print("prediction append-only official selection check")
        print(f"database path: {database_path}")
        print(f"ticker: {args.ticker}")
        print("mode: read-only")
        print(f"feature dates: {diagnostics['feature_date_count']}")
        print(f"scheduled runs: {diagnostics['scheduled_run_count']}")
        print(f"manual runs: {diagnostics['manual_run_count']}")
        print(f"complete scheduled runs: {diagnostics['complete_scheduled_run_count']}")
        print(f"partial scheduled runs: {diagnostics['partial_scheduled_run_count']}")
        print(
            "selected official candidates: "
            f"{diagnostics['selected_official_candidate_count']}"
        )
        print(f"latest run id: {diagnostics['latest_run_id'] or '-'}")
        print(f"latest run type: {diagnostics['latest_run_type'] or '-'}")
        print(
            "classifications: "
            f"{json.dumps(diagnostics['classifications'], ensure_ascii=False)}"
        )
        for item in diagnostics["feature_dates"]:
            print(f"feature date: {item['feature_date']}")
            print(f"  scheduled run count: {item['scheduled_run_count']}")
            print(f"  manual run count: {item['manual_run_count']}")
            print(f"  complete scheduled run count: {item['complete_scheduled_run_count']}")
            print(f"  partial scheduled run count: {item['partial_scheduled_run_count']}")
            print(f"  selected official run id: {item['selected_official_run_id'] or 'none'}")
            print(f"  latest run id: {item['latest_run_id'] or 'none'}")
            print(f"  selected binary record id: {item['selected_binary_record_id'] or 'none'}")
            print(
                "  selected multiclass record id: "
                f"{item['selected_multiclass_record_id'] or 'none'}"
            )
            print(f"  missing slots: {json.dumps(item['missing_slots'])}")
            print(f"  duplicate slots: {json.dumps(item['duplicate_slots'])}")
            print(
                "  reason codes: "
                f"{json.dumps(item['reason_codes'], ensure_ascii=False)}"
            )
            print(
                "  current DB official record count: "
                f"{item['current_db_official_record_count']}"
            )
            print(
                "  selected records currently official: "
                f"{_optional_bool_text(item['selected_records_currently_official'])}"
            )
            print(
                "  selected records currently latest: "
                f"{_optional_bool_text(item['selected_records_currently_latest'])}"
            )
            print(f"  snapshot compare status: {item['snapshot_compare_status']}")
            print(
                "  snapshot matches: "
                f"{_optional_bool_text(item['snapshot_matches'])}"
            )
            print(f"  snapshot mismatch count: {item['snapshot_mismatch_count']}")
        print("status: OK - future official selection diagnostics completed without DB writes.")
        print("note: DB is_official/is_latest flags and prediction metrics sources are unchanged.")
        return

    if args.command == "prediction-append-only-scheduled-contract-check":
        from stock_ai.backtest.scheduled_dual_write import (
            build_scheduled_dual_write_contract,
            run_scheduled_contract_check,
        )
        from stock_ai.operations.freshness import summarize_daily_data_freshness
        from stock_ai.storage.sqlite_store import (
            filter_official_prediction_snapshot_records,
            read_latest_predictions,
        )

        if args.dry_run and args.write:
            raise SystemExit("Use either --dry-run or --write, not both.")
        if args.write and not args.database_path:
            raise SystemExit("--write requires --database-path pointing to a test or copied DB.")

        database_path = (
            Path(args.database_path) if args.database_path else settings.paths.database_path
        )
        snapshot = read_latest_predictions(database_path)
        ticker_snapshot = (
            snapshot[snapshot["ticker"].astype(str).eq(args.ticker)].reset_index(drop=True)
            if not snapshot.empty
            else snapshot
        )
        current_candidates, _ = filter_official_prediction_snapshot_records(
            ticker_snapshot,
            ticker=args.ticker,
        )
        if args.feature_date:
            feature_date = args.feature_date
        elif not current_candidates.empty:
            feature_date = str(current_candidates["date"].max())
        else:
            raise SystemExit("No current production snapshot feature date is available.")

        freshness = summarize_daily_data_freshness(
            ticker=args.ticker,
            raw_data_dir=settings.paths.raw_data_dir,
            feature_data_dir=settings.paths.feature_data_dir,
        )
        expected_feature_date = (
            args.expected_feature_date
            or freshness.get("expected_latest_trading_date")
            or feature_date
        )
        contract = build_scheduled_dual_write_contract(
            ticker=args.ticker,
            feature_date=feature_date,
            expected_feature_date=expected_feature_date,
            scheduled_for=args.scheduled_for,
            execution_started_at=args.execution_started_at,
            execution_completed_at=args.execution_completed_at,
            execution_mode=args.execution_mode,
            binary_success=args.binary_status == "success",
            multiclass_success=args.multiclass_status == "success",
            data_freshness_status=(
                args.data_freshness_status or freshness.get("freshness_status") or "UNKNOWN"
            ),
            stale_at_run_time=bool(
                args.stale_at_run_time
                if args.data_freshness_status is not None
                else (args.stale_at_run_time or freshness.get("stale_at_run_time"))
            ),
            latest_raw_date=freshness.get("latest_raw_date"),
            latest_feature_date=freshness.get("latest_feature_date"),
            expected_latest_trading_date=freshness.get("expected_latest_trading_date"),
            task_name=args.task_name,
            task_last_result=args.task_last_result,
            notes=args.notes,
            idempotency_key=args.idempotency_key,
        )
        result = run_scheduled_contract_check(
            database_path,
            contract,
            write=bool(args.write),
            production_database_path=settings.paths.database_path,
        )
        eligibility = result["eligibility"]
        before = result["before"]
        after = result["after"]

        print("prediction append-only scheduled contract check")
        print(f"database path: {result['database_path']}")
        print(f"ticker: {args.ticker}")
        print(f"mode: {result['mode']}")
        print(f"resolved feature date: {contract.feature_date}")
        print(f"expected feature date: {contract.expected_feature_date}")
        print(f"scheduled for: {contract.scheduled_for}")
        print(f"execution started at: {contract.execution_started_at}")
        print(f"execution completed at: {contract.execution_completed_at}")
        print(f"execution mode: {contract.execution_mode}")
        print(f"run type: {contract.run_type}")
        print(f"source: {contract.source}")
        print(f"generated idempotency key: {contract.idempotency_key}")
        print(f"resolved run id: {result['run_id']}")
        print(f"raw snapshot candidates: {result['raw_snapshot_candidates']}")
        print(f"official filtered candidates: {result['official_filtered_candidates']}")
        print(f"write candidates: {result['write_candidates']}")
        print(f"complete target set: {str(eligibility['complete_target_set']).lower()}")
        print(f"present slots: {json.dumps(eligibility['present_slots'])}")
        print(f"missing slots: {json.dumps(eligibility['missing_slots'])}")
        print(f"duplicate slots: {json.dumps(eligibility['duplicate_slots'])}")
        print(
            "eligible for official: "
            f"{str(eligibility['eligible_for_official']).lower()}"
        )
        print(
            "contract selected official candidate: "
            f"{str(eligibility['selected_official_candidate']).lower()}"
        )
        print(f"selected official run id in DB: {result['selected_official_run_id'] or 'none'}")
        print(f"attempt status: {eligibility['attempt_status']}")
        print(f"delayed or catch-up: {str(eligibility['delayed_or_catch_up']).lower()}")
        print(
            "reason codes: "
            f"{json.dumps(eligibility['reason_codes'], ensure_ascii=False)}"
        )
        print(f"append-only runs before: {before['prediction_runs_count']}")
        print(f"append-only records before: {before['prediction_records_count']}")
        print(f"append-only runs after: {after['prediction_runs_count']}")
        print(f"append-only records after: {after['prediction_records_count']}")
        print(f"existing predictions rows: {after['existing_predictions_count']}")
        print(
            "existing predictions unchanged: "
            f"{str(result['existing_predictions_unchanged']).lower()}"
        )
        print(f"database modified: {str(result['database_modified']).lower()}")
        print("is_official/is_latest updated: false")
        if result["write_result"] is not None:
            print(f"inserted records: {result['write_result']['inserted_records']}")
            print(
                "duplicate records ignored: "
                f"{result['write_result']['duplicate_records']}"
            )
        print("status: OK - scheduled contract diagnostics completed.")
        print("note: production DB writes are refused; daily automatic dual-write remains off.")
        return

    if args.command == "build-features-universe":
        tickers = load_universe()
        if args.limit is not None:
            tickers = tickers[: args.limit]
        for ticker in tickers:
            output_path = _build_features_for_ticker(ticker.code, settings)
            print(f"saved: {output_path}")
        return

    if args.command == "train":
        from stock_ai.models.baseline import (
            evaluate_classifier,
            save_model,
            train_lightgbm_classifier,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        train_data, valid_data, test_data = chronological_train_valid_test_split(dataset)
        model, feature_columns = train_lightgbm_classifier(train_data, valid_data)
        metrics = evaluate_classifier(model, test_data, feature_columns)
        model_path = settings.paths.model_dir / f"lightgbm_{args.ticker}.joblib"
        save_model(model, feature_columns, model_path)
        metrics_path = Path("reports") / "metrics" / f"lightgbm_{args.ticker}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"model: {model_path}")
        print(f"metrics: {metrics_path}")
        print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))
        return

    if args.command == "train-binary":
        from stock_ai.models.baseline import (
            evaluate_binary_classifier,
            get_feature_importance,
            save_model,
            train_lightgbm_binary_classifier,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        if "target_next_close_binary" not in dataset.columns:
            dataset = add_next_close_binary_target(dataset)
        train_data, valid_data, test_data = chronological_train_valid_test_split(dataset)
        model, feature_columns = train_lightgbm_binary_classifier(train_data, valid_data)
        metrics = evaluate_binary_classifier(model, test_data, feature_columns)
        top_features = get_feature_importance(model, feature_columns, top_n=10)
        model_path = settings.paths.model_dir / f"lightgbm_binary_{args.ticker}.joblib"
        save_model(
            model,
            feature_columns,
            model_path,
            metadata={
                "target": "target_next_close_binary",
                "metrics": metrics,
                "top_features": top_features,
            },
        )
        metrics_path = Path("reports") / "metrics" / f"lightgbm_binary_{args.ticker}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "feature_columns": feature_columns,
                    "top_features": top_features,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"model: {model_path}")
        print(f"metrics: {metrics_path}")
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-overnight":
        model_path, metrics_path, metrics = train_overnight_model_for_ticker(args.ticker, settings)
        print(f"overnight model: {model_path}")
        print(f"overnight metrics: {metrics_path}")
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "train-multiclass":
        from stock_ai.models.baseline import (
            evaluate_classifier,
            get_feature_importance,
            save_model,
            train_lightgbm_classifier,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        threshold_profile = label_distribution.normalize_threshold_profile(args.threshold_profile)
        weight_mode = label_distribution.normalize_weight_mode(args.weight_mode)
        scheme = label_distribution.get_threshold_profile(threshold_profile)
        dataset = label_distribution.ensure_next_close_return(read_parquet(feature_path))
        base_train, _base_valid, _base_test = chronological_train_valid_test_split(dataset)
        label_thresholds = label_distribution.resolve_label_thresholds(base_train, scheme)
        dataset = label_distribution.add_target_for_scheme(
            dataset,
            scheme,
            thresholds=label_thresholds,
        )
        train_data, valid_data, test_data = chronological_train_valid_test_split(dataset)
        model, feature_columns = train_lightgbm_classifier(
            train_data,
            valid_data,
            target_column="target_next_close_class",
            weight_mode=weight_mode,
        )
        metrics = evaluate_classifier(
            model,
            test_data,
            feature_columns,
            target_column="target_next_close_class",
        )
        metrics["threshold_profile"] = threshold_profile
        metrics["label_thresholds"] = label_thresholds
        metrics["weight_mode"] = weight_mode
        top_features = get_feature_importance(model, feature_columns, top_n=10)
        model_name = label_distribution.multiclass_model_name(threshold_profile, weight_mode)
        artifact_stem = label_distribution.multiclass_artifact_stem(
            args.ticker,
            threshold_profile,
            weight_mode,
        )
        model_path = settings.paths.model_dir / f"{artifact_stem}.joblib"
        save_model(
            model,
            feature_columns,
            model_path,
            metadata={
                "target": "target_next_close_class",
                "model_name": model_name,
                "threshold_profile": threshold_profile,
                "weight_mode": weight_mode,
                "class_weight": label_distribution.multiclass_class_weight(weight_mode),
                "threshold_profile_scheme": label_distribution.scheme_to_dict(scheme),
                "label_thresholds": label_thresholds,
                "metrics": metrics,
                "top_features": top_features,
            },
        )
        metrics_path = Path("reports") / "metrics" / f"{artifact_stem}.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "threshold_profile": threshold_profile,
                    "weight_mode": weight_mode,
                    "class_weight": label_distribution.multiclass_class_weight(weight_mode),
                    "threshold_profile_scheme": label_distribution.scheme_to_dict(scheme),
                    "label_thresholds": label_thresholds,
                    "metrics": metrics,
                    "feature_columns": feature_columns,
                    "top_features": top_features,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"model: {model_path}")
        print(f"metrics: {metrics_path}")
        print(
            json.dumps(
                {
                    "threshold_profile": threshold_profile,
                    "weight_mode": weight_mode,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "weighted_f1": metrics["weighted_f1"],
                    "surge_recall": metrics["surge_recall"],
                    "crash_recall": metrics["crash_recall"],
                    "surge_precision": metrics["surge_precision"],
                    "crash_precision": metrics["crash_precision"],
                },
                indent=2,
            )
        )
        return

    if args.command == "backtest-binary":
        from stock_ai.anomaly.rule_based import add_anomaly_scores
        from stock_ai.backtest.walk_forward import (
            WalkForwardConfig,
            run_walk_forward_backtest,
            write_backtest_csvs,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        if "target_next_close_binary" not in dataset.columns:
            dataset = add_next_close_binary_target(dataset)
        dataset = add_anomaly_scores(dataset)
        config = WalkForwardConfig(
            ticker=args.ticker,
            train_years=args.train_years,
            test_step_months=args.test_step_months,
            probability_threshold=args.threshold,
            start_date=args.start,
            end_date=args.end,
        )
        result = run_walk_forward_backtest(dataset, config=config)
        output_paths = write_backtest_csvs(
            result,
            Path("reports") / "backtests",
            ticker=args.ticker,
            threshold=args.threshold,
        )
        print(json.dumps(result.metrics, indent=2))
        for artifact, path in output_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "offline-model-experiments":
        from stock_ai.anomaly.rule_based import add_anomaly_scores
        from stock_ai.evaluation.offline_model_experiments import (
            OfflineExperimentConfig,
            default_candidate_configs,
            run_offline_model_experiments,
            write_offline_experiment_reports,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        if "target_next_close_binary" not in dataset.columns:
            dataset = add_next_close_binary_target(dataset)
        dataset = add_anomaly_scores(dataset)
        config = OfflineExperimentConfig(
            ticker=args.ticker,
            train_years=args.train_years,
            test_step_months=args.test_step_months,
            start_date=args.start,
            end_date=args.end,
            min_train_rows=args.min_train_rows,
        )
        candidates = default_candidate_configs()
        if args.candidates:
            wanted = set(args.candidates)
            candidates = [candidate for candidate in candidates if candidate.name in wanted]
            if not candidates:
                raise SystemExit(f"No matching offline experiment candidates: {sorted(wanted)}")
        result = run_offline_model_experiments(dataset, config=config, candidates=candidates)
        output_paths = write_offline_experiment_reports(result, args.output_dir)
        print(json.dumps({"conclusions": result.conclusions, "rows": len(result.summary)}, indent=2))
        for artifact, path in output_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "binary-up-detection-experiments":
        from stock_ai.anomaly.rule_based import add_anomaly_scores
        from stock_ai.evaluation import accuracy_diagnostics
        from stock_ai.evaluation.binary_up_detection_experiments import (
            BinaryUpExperimentConfig,
            default_up_detection_candidates,
            run_binary_up_detection_experiments,
            write_binary_up_detection_reports,
        )
        from stock_ai.regime.rule_based import add_regime_labels
        from stock_ai.storage.sqlite_store import read_prediction_history

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        if "target_next_close_binary" not in dataset.columns:
            dataset = add_next_close_binary_target(dataset)
        context_data = add_anomaly_scores(add_regime_labels(dataset))

        live_frame = None
        if not args.skip_live_comparison:
            live_predictions = read_prediction_history(
                settings.paths.database_path,
                ticker=args.ticker,
                target="next_close_binary",
                model_name="lightgbm_binary",
                prediction_type="live",
            )
            live_frame = accuracy_diagnostics.build_binary_diagnostic_frame(
                live_predictions,
                context_data,
                prediction_type="live",
            )

        candidates = default_up_detection_candidates()
        if args.candidates:
            wanted = set(args.candidates)
            candidates = [candidate for candidate in candidates if candidate.name in wanted]
            if not candidates:
                raise SystemExit(f"No matching binary up-detection candidates: {sorted(wanted)}")

        result = run_binary_up_detection_experiments(
            context_data,
            config=BinaryUpExperimentConfig(
                ticker=args.ticker,
                train_years=args.train_years,
                test_step_months=args.test_step_months,
                start_date=args.start,
                end_date=args.end,
                min_train_rows=args.min_train_rows,
            ),
            candidates=candidates,
            live_frame=live_frame,
        )
        output_paths = write_binary_up_detection_reports(result, args.output_dir)
        print(
            json.dumps(
                {
                    "conclusions": result.conclusions,
                    "summary_rows": len(result.summary),
                    "candidate_count": len(candidates),
                    "historical_predictions": len(result.predictions),
                    "live_rows": 0 if live_frame is None else len(live_frame),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        for artifact, path in output_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "multiclass-walk-forward-diagnostics":
        from stock_ai.anomaly.rule_based import add_anomaly_scores
        from stock_ai.backtest.walk_forward import (
            WalkForwardConfig,
            run_walk_forward_backtest,
        )
        from stock_ai.evaluation import accuracy_diagnostics
        from stock_ai.evaluation.multiclass_walk_forward_diagnostics import (
            MulticlassWalkForwardConfig,
            append_binary_direction_reference,
            classify_multiclass_conclusions,
            run_multiclass_walk_forward_diagnostics,
            write_multiclass_walk_forward_reports,
        )
        from stock_ai.regime.rule_based import add_regime_labels
        from stock_ai.storage.sqlite_store import read_prediction_history

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        context_data = add_anomaly_scores(add_regime_labels(dataset))

        threshold_profile = label_distribution.normalize_threshold_profile(args.threshold_profile)
        weight_mode = label_distribution.normalize_weight_mode(args.weight_mode)
        live_frame = None
        if not args.skip_live_comparison:
            model_name = label_distribution.multiclass_model_name(threshold_profile, weight_mode)
            live_predictions = read_prediction_history(
                settings.paths.database_path,
                ticker=args.ticker,
                target="next_close_multiclass",
                model_name=model_name,
                prediction_type="live",
            )
            live_frame = accuracy_diagnostics.build_multiclass_diagnostic_frame(
                live_predictions,
                context_data,
                prediction_type="live",
            )

        config = MulticlassWalkForwardConfig(
            ticker=args.ticker,
            train_years=args.train_years,
            test_step_months=args.test_step_months,
            start_date=args.start,
            end_date=args.end,
            min_train_rows=args.min_train_rows,
            threshold_profile=threshold_profile,
            weight_mode=weight_mode,
        )
        result = run_multiclass_walk_forward_diagnostics(
            context_data,
            config=config,
            live_frame=live_frame,
        )

        if not args.skip_binary_comparison:
            binary_data = context_data
            if "target_next_close_binary" not in binary_data.columns:
                binary_data = add_next_close_binary_target(binary_data)
            binary_result = run_walk_forward_backtest(
                binary_data,
                config=WalkForwardConfig(
                    ticker=args.ticker,
                    train_years=args.train_years,
                    test_step_months=args.test_step_months,
                    start_date=args.start,
                    end_date=args.end,
                    min_train_rows=args.min_train_rows,
                ),
            )
            binary_frame = accuracy_diagnostics.build_walk_forward_binary_diagnostic_frame(
                binary_result.predictions,
                binary_data,
            )
            append_binary_direction_reference(result, binary_frame)
            result.conclusions = classify_multiclass_conclusions(
                result.summary,
                result.direction_comparison,
            )

        output_paths = write_multiclass_walk_forward_reports(result, args.output_dir)
        print(
            json.dumps(
                {
                    "conclusions": result.conclusions,
                    "summary_rows": len(result.summary),
                    "historical_predictions": len(result.predictions),
                    "live_rows": 0 if live_frame is None else len(live_frame),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        for artifact, path in output_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "multiclass-offline-boundary-experiments":
        from stock_ai.anomaly.rule_based import add_anomaly_scores
        from stock_ai.backtest.walk_forward import (
            WalkForwardConfig,
            run_walk_forward_backtest,
        )
        from stock_ai.evaluation import accuracy_diagnostics
        from stock_ai.evaluation.multiclass_offline_boundary_experiments import (
            BoundaryExperimentConfig,
            default_boundary_candidates,
            run_multiclass_offline_boundary_experiments,
            write_boundary_experiment_reports,
        )
        from stock_ai.regime.rule_based import add_regime_labels
        from stock_ai.storage.sqlite_store import read_prediction_history

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        context_data = add_anomaly_scores(add_regime_labels(dataset))

        live_frame = None
        if not args.skip_live_comparison:
            live_predictions = read_prediction_history(
                settings.paths.database_path,
                ticker=args.ticker,
                target="next_close_multiclass",
                model_name=label_distribution.multiclass_model_name(
                    label_distribution.DEFAULT_THRESHOLD_PROFILE,
                    label_distribution.DEFAULT_WEIGHT_MODE,
                ),
                prediction_type="live",
            )
            live_frame = accuracy_diagnostics.build_multiclass_diagnostic_frame(
                live_predictions,
                context_data,
                prediction_type="live",
            )

        binary_reference = None
        if not args.skip_binary_comparison:
            binary_data = context_data
            if "target_next_close_binary" not in binary_data.columns:
                binary_data = add_next_close_binary_target(binary_data)
            binary_result = run_walk_forward_backtest(
                binary_data,
                config=WalkForwardConfig(
                    ticker=args.ticker,
                    train_years=args.train_years,
                    test_step_months=args.test_step_months,
                    start_date=args.start,
                    end_date=args.end,
                    min_train_rows=args.min_train_rows,
                ),
            )
            binary_reference = accuracy_diagnostics.build_walk_forward_binary_diagnostic_frame(
                binary_result.predictions,
                binary_data,
            )

        candidates = default_boundary_candidates()
        if args.candidates:
            wanted = set(args.candidates)
            candidates = [candidate for candidate in candidates if candidate.name in wanted]
            if not candidates:
                raise SystemExit(f"No matching boundary experiment candidates: {sorted(wanted)}")

        result = run_multiclass_offline_boundary_experiments(
            context_data,
            config=BoundaryExperimentConfig(
                ticker=args.ticker,
                train_years=args.train_years,
                test_step_months=args.test_step_months,
                start_date=args.start,
                end_date=args.end,
                min_train_rows=args.min_train_rows,
            ),
            candidates=candidates,
            live_frame=live_frame,
            binary_reference=binary_reference,
        )
        output_paths = write_boundary_experiment_reports(result, args.output_dir)
        print(
            json.dumps(
                {
                    "conclusions": result.conclusions,
                    "summary_rows": len(result.summary),
                    "prediction_rows": len(result.predictions),
                    "candidate_count": len(candidates),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        for artifact, path in output_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "predict":
        from stock_ai.models.predict import load_model_bundle, predict_latest_rows

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        model_path = (
            Path(args.model_path)
            if args.model_path
            else settings.paths.model_dir / f"lightgbm_{args.ticker}.joblib"
        )
        dataset = read_parquet(feature_path)
        bundle = load_model_bundle(model_path)
        predictions = predict_latest_rows(
            bundle["model"],
            bundle["feature_columns"],
            dataset,
            target="next_close",
            model_name="lightgbm",
        )
        count = upsert_predictions(predictions, settings.paths.database_path)
        print(f"saved predictions: {count}")
        return

    if args.command == "predict-binary":
        from stock_ai.models.predict import load_model_bundle, predict_latest_binary_rows

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        model_path = (
            Path(args.model_path)
            if args.model_path
            else settings.paths.model_dir / f"lightgbm_binary_{args.ticker}.joblib"
        )
        dataset = read_parquet(feature_path)
        bundle = load_model_bundle(model_path)
        top_features = bundle.get("metadata", {}).get("top_features", [])
        predictions = predict_latest_binary_rows(
            bundle["model"],
            bundle["feature_columns"],
            dataset,
            top_features=top_features,
            target="next_close_binary",
            model_name="lightgbm_binary",
        )
        count = upsert_predictions(predictions, settings.paths.database_path)
        if count <= 0:
            raise SystemExit("No binary prediction rows were saved.")
        print(f"saved binary predictions: {count}")
        if not predictions.empty:
            row = predictions.iloc[0]
            print(
                "saved binary prediction detail: "
                f"date={pd_to_date_string(row.get('date'))}, "
                f"ticker={row.get('ticker')}, "
                f"target={row.get('target')}, "
                f"model={row.get('model_name')}, "
                "source=live"
            )
        return

    if args.command == "predict-overnight":
        from stock_ai.data.universe import load_universe

        target_tickers: list[str] = []
        if getattr(args, "all_universe", False) or not args.ticker:
            target_tickers = [t.code for t in load_universe()]
        else:
            target_tickers = [args.ticker]

        total_saved = 0
        for t in target_tickers:
            m_path = Path(args.model_path) if args.model_path else None
            count, predictions = predict_overnight_for_ticker(t, settings, model_path=m_path)
            total_saved += count
            if not predictions.empty:
                row = predictions.iloc[0]
                notes = json.loads(row.get("notes") or "{}")
                print(
                    "saved overnight prediction detail: "
                    f"date={pd_to_date_string(row.get('date'))}, "
                    f"ticker={row.get('ticker')}, "
                    f"target={row.get('target')}, "
                    f"model={row.get('model_name')}, "
                    f"p_up={row.get('probability_up'):.4f}, "
                    f"quantile={notes.get('quantile', 'N/A')} ({notes.get('label', '')}), "
                    f"hist_win_rate={notes.get('historical_win_rate', 'N/A')}"
                )
        if total_saved <= 0:
            raise SystemExit("No overnight prediction rows were saved.")
        print(f"saved overnight predictions: {total_saved} (tickers: {len(target_tickers)})")
        return

    if args.command == "sync-overnight-universe":
        from stock_ai.data.universe import load_universe

        if args.tickers:
            ticker_list = [t.strip() for t in args.tickers.split(",") if t.strip()]
        else:
            ticker_list = [t.code for t in load_universe()]

        start = args.start or settings.data.get("default_start", "2010-01-01")
        print(f"=== Sync Overnight Universe ({len(ticker_list)} tickers) ===")

        for idx, ticker in enumerate(ticker_list, 1):
            print(f"[{idx}/{len(ticker_list)}] Processing {ticker}...")
            raw_path = settings.paths.raw_data_dir / f"{ticker}.parquet"
            if not raw_path.exists() or args.force_download:
                print(f"  Downloading {ticker}...")
                raw_data = fetch_daily_ohlcv(ticker, start=start, end=args.end)
                write_parquet(raw_data, raw_path)

            feat_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
            needs_feat = (
                not feat_path.exists()
                or args.force_features
                or "target_next_open_binary" not in read_parquet(feat_path).columns
            )
            if needs_feat:
                print(f"  Building 163 features for {ticker}...")
                _build_features_for_ticker(ticker, settings)

            if not args.skip_train:
                from stock_ai.models.overnight import OVERNIGHT_ARTIFACT_FILENAME

                m_file = settings.paths.model_dir / OVERNIGHT_ARTIFACT_FILENAME.format(ticker=ticker)
                if not m_file.exists() or args.force_train:
                    print(f"  Training overnight model for {ticker}...")
                    _, _, metrics = train_overnight_model_for_ticker(ticker, settings)
                    print(f"  AUC: {metrics.get('auc', 0.0):.4f}")

            if not args.skip_predict:
                count, preds = predict_overnight_for_ticker(ticker, settings)
                if not preds.empty:
                    r = preds.iloc[0]
                    n = json.loads(r.get("notes") or "{}")
                    print(
                        f"  Predicted {ticker}: p_up={r.get('probability_up'):.4f} | "
                        f"{n.get('quantile', 'N/A')} ({n.get('label', '')})"
                    )

        print("=== Overnight Universe Sync Complete ===")
        return

    if args.command == "overnight-status":
        from stock_ai.models.overnight import (
            OVERNIGHT_ARTIFACT_FILENAME,
            OVERNIGHT_METRICS_FILENAME,
            OVERNIGHT_MODEL_NAME,
            OVERNIGHT_TARGET,
        )
        from stock_ai.storage.sqlite_store import read_prediction_history

        model_filename = OVERNIGHT_ARTIFACT_FILENAME.format(ticker=args.ticker)
        model_path = settings.paths.model_dir / model_filename
        metrics_filename = OVERNIGHT_METRICS_FILENAME.format(ticker=args.ticker)
        metrics_path = Path("reports") / "metrics" / metrics_filename

        history = read_prediction_history(
            settings.paths.database_path,
            ticker=args.ticker,
            target=OVERNIGHT_TARGET,
            model_name=OVERNIGHT_MODEL_NAME,
            prediction_type="live",
        )

        status_info: dict[str, object] = {
            "ticker": args.ticker,
            "target": OVERNIGHT_TARGET,
            "model_name": OVERNIGHT_MODEL_NAME,
            "artifact_exists": model_path.exists(),
            "artifact_path": str(model_path) if model_path.exists() else None,
            "metrics_exists": metrics_path.exists(),
            "metrics_path": str(metrics_path) if metrics_path.exists() else None,
            "history_rows": len(history),
            "latest_prediction": None,
        }
        if metrics_path.exists():
            try:
                metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
                status_info["metrics"] = metrics_data.get("metrics")
            except Exception:
                pass
        if not history.empty:
            from stock_ai.models.overnight import classify_overnight_quantile

            latest_row = history.sort_values("date").iloc[-1].to_dict()
            p_up = latest_row.get("probability_up")
            q_info = classify_overnight_quantile(float(p_up)) if p_up is not None else {}
            status_info["latest_prediction"] = {
                "date": str(latest_row.get("date")),
                "probability_up": p_up,
                "predicted_class": latest_row.get("predicted_class"),
                "quantile": q_info.get("quantile"),
                "label": q_info.get("label"),
                "historical_win_rate": q_info.get("historical_win_rate"),
                "notes": q_info,
            }
        print(json.dumps(status_info, indent=2, ensure_ascii=False))
        return

    if args.command == "run-overnight-paper-trader":
        from stock_ai.backtest.overnight_portfolio import (
            export_portfolio_simulation_reports,
            run_full_overnight_portfolio_backtest,
        )

        sim_data = run_full_overnight_portfolio_backtest(
            initial_capital=args.initial_capital,
        )
        export_portfolio_simulation_reports(
            sim_data,
            output_json_path=args.output_json,
            output_md_path=args.output_md,
        )
        res_5 = sim_data["results_by_cost"]["5bps"]
        print("=== Overnight Paper Trader Simulation Complete ===")
        print(f"Period: {sim_data['start_date']} to {sim_data['end_date']} ({sim_data['total_days']} days)")
        for s, s_data in res_5.items():
            m = s_data["metrics"]
            print(
                f"[{s}] Return: {m['cumulative_return']:+.1%} | Final: {m['final_equity']:,.0f} JPY | "
                f"Sharpe: {m['sharpe_ratio']:.2f} | MaxDD: {m['max_drawdown']:.1%} | WinRate: {m['win_rate']:.1%}"
            )
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved Report: {args.output_md}")
        return

    if args.command == "scan-turnaround-universe":
        from stock_ai.models.turnaround import (
            export_turnaround_reports,
            scan_turnaround_universe,
            train_turnaround_model,
        )

        if getattr(args, "force_train", False):
            print("Force re-training turnaround LightGBM model...")
            train_turnaround_model()

        scan_data = scan_turnaround_universe()
        export_turnaround_reports(
            scan_data,
            output_json_path=args.output_json,
            output_md_path=args.output_md,
        )
        print("=== Value & Turnaround Scanner Complete ===")
        print(f"Date: {scan_data.get('date', '-')}")
        rankings = scan_data.get("rankings", [])
        print(f"Scanned {len(rankings)} tickers.")
        for r in rankings[:5]:
            pbr_s = f"{r['pbr']:.2f}x" if r.get("pbr") is not None else "-"
            # ASCII/safe status for console logging to prevent cp932 encode issues
            stat_clean = r["status"].replace("💎", "[Value]").replace("👀", "[Watch]").replace("📈", "[Trend]").replace("⭐", "[Golden]")
            print(
                f"[{r['ticker']}] Score: {r['bargain_score']:.1f} | 52wDD: {r['drawdown_252d']:+.1%} | "
                f"PBR: {pbr_s} | p_turn: {r['p_turnaround']:.1%} | {stat_clean}"
            )
        print(f"Saved JSON: {args.output_json}")
        print(f"Saved Report: {args.output_md}")
        return

    if args.command == "run-cloud-daily-pipeline":
        from stock_ai.cloud.pipeline import run_cloud_daily_pipeline

        run_cloud_daily_pipeline(
            start=args.start,
            force_download=args.force_download,
            force_features=args.force_features,
            force_train=args.force_train,
            snapshot_path=args.snapshot_path,
            webhook_url=args.webhook_url,
        )
        return

    if args.command == "binary-shadow-predict":
        from stock_ai.evaluation.binary_shadow import (
            build_binary_shadow_predictions,
            save_shadow_predictions,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        model_path = (
            Path(args.model_path)
            if args.model_path
            else settings.paths.model_dir / f"lightgbm_binary_{args.ticker}.joblib"
        )
        dataset = read_parquet(feature_path)
        predictions = build_binary_shadow_predictions(
            dataset,
            ticker=args.ticker,
            production_model_path=model_path,
        )
        paths = {} if args.dry_run else save_shadow_predictions(predictions, args.output_dir)
        print(
            json.dumps(
                {
                    "rows": len(predictions),
                    "feature_dates": sorted(predictions["feature_date"].dropna().astype(str).unique().tolist())
                    if not predictions.empty
                    else [],
                    "candidates": sorted(predictions["candidate_name"].dropna().astype(str).unique().tolist())
                    if not predictions.empty
                    else [],
                    "sqlite_saved": False,
                    "dry_run": bool(args.dry_run),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        for artifact, path in paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "binary-shadow-status":
        from stock_ai.evaluation.binary_shadow import (
            read_shadow_predictions,
            summarize_shadow_lifecycle,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        predictions = read_shadow_predictions(args.shadow_path)
        summary = summarize_shadow_lifecycle(predictions, dataset, ticker=args.ticker)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "binary-shadow-evaluate":
        from stock_ai.evaluation.binary_shadow import evaluate_shadow_predictions
        from stock_ai.storage.sqlite_store import read_prediction_history

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        live_predictions = read_prediction_history(
            settings.paths.database_path,
            ticker=args.ticker,
            target="next_close_binary",
            model_name="lightgbm_binary",
            prediction_type="live",
        )
        result = evaluate_shadow_predictions(
            args.shadow_path,
            dataset,
            ticker=args.ticker,
            output_dir=args.output_dir,
            live_predictions=live_predictions,
            persist=not args.dry_run,
        )
        print(
            json.dumps(
                {
                    "conclusions": result.conclusions,
                    "summary_rows": len(result.summary),
                    "prediction_rows": len(result.predictions),
                    "sqlite_saved": False,
                    "dry_run": bool(args.dry_run),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        for artifact, path in result.saved_paths.items():
            print(f"{artifact}: {path}")
        return

    if args.command == "predict-multiclass":
        from stock_ai.models.predict import (
            load_model_bundle,
            predict_latest_multiclass_rows,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        threshold_profile = label_distribution.normalize_threshold_profile(args.threshold_profile)
        weight_mode = label_distribution.normalize_weight_mode(args.weight_mode)
        model_name = label_distribution.multiclass_model_name(threshold_profile, weight_mode)
        artifact_stem = label_distribution.multiclass_artifact_stem(
            args.ticker,
            threshold_profile,
            weight_mode,
        )
        model_path = (
            Path(args.model_path)
            if args.model_path
            else settings.paths.model_dir / f"{artifact_stem}.joblib"
        )
        dataset = read_parquet(feature_path)
        bundle = load_model_bundle(model_path)
        top_features = bundle.get("metadata", {}).get("top_features", [])
        model_name = str(bundle.get("metadata", {}).get("model_name") or model_name)
        predictions = predict_latest_multiclass_rows(
            bundle["model"],
            bundle["feature_columns"],
            dataset,
            top_features=top_features,
            target="next_close_multiclass",
            model_name=model_name,
        )
        count = upsert_predictions(predictions, settings.paths.database_path)
        if count <= 0:
            raise SystemExit("No multiclass prediction rows were saved.")
        print(f"saved multiclass predictions: {count}")
        if not predictions.empty:
            row = predictions.iloc[0]
            print(
                "saved multiclass prediction detail: "
                f"date={pd_to_date_string(row.get('date'))}, "
                f"ticker={row.get('ticker')}, "
                f"target={row.get('target')}, "
                f"model={row.get('model_name')}, "
                f"profile={threshold_profile}, "
                f"weight_mode={weight_mode}, "
                "source=live"
            )
        return

    if args.command == "backfill-predictions":
        from stock_ai.backfill.prediction_backfill import (
            BackfillConfig,
            build_binary_prediction_backfill,
        )

        feature_path = settings.paths.feature_data_dir / f"{args.ticker}.parquet"
        dataset = read_parquet(feature_path)
        config = BackfillConfig(
            ticker=args.ticker,
            start_date=args.start,
            end_date=args.end,
            model_name=args.model_name,
            min_train_rows=args.min_train_rows,
        )
        result = build_binary_prediction_backfill(dataset, config)
        count = (
            upsert_predictions(result.predictions, settings.paths.database_path)
            if not result.predictions.empty
            else 0
        )
        print("backfill note: generated rows are prediction_type=backfill for history inspection.")
        print("backfill note: each date is trained only with rows before that prediction date.")
        print(f"backfill attempted dates: {result.attempted_dates}")
        print(f"backfill saved predictions: {count}")
        if not result.skipped_rows.empty:
            print(f"backfill skipped dates: {len(result.skipped_rows)}")
            print(result.skipped_rows.head(20).to_string(index=False))
        return


def _print_fetch_news_result(
    result,
    *,
    profile_name: str | None = None,
    storage_ticker: str = "",
    row_ticker: str = "",
) -> None:
    print(f"news profile: {profile_name or '-'}")
    print(f"news query: {result.query}")
    print(f"news storage ticker: {storage_ticker or '-'}")
    print(f"news row ticker: {row_ticker or '-'}")
    print(f"news output: {result.output_path}")
    print(f"news fetched: {result.fetched_count}")
    print(f"news added: {result.added_count}")
    print(f"news duplicates: {result.duplicate_count}")
    print(f"news total: {result.total_count}")
    print(f"news latest fetched_at: {result.latest_fetched_at or '-'}")


def _append_only_dual_write_run_id(ticker: str, idempotency_key: str | None) -> str | None:
    if not idempotency_key:
        return None
    raw = f"manual_live_dual_write_{ticker}_{idempotency_key}"
    safe = "".join(character if character.isalnum() else "_" for character in raw)
    return "_".join(part for part in safe.split("_") if part)


def _optional_bool_text(value: object) -> str:
    if value is None:
        return "not applicable"
    return str(bool(value)).lower()


def _print_official_filter_rows(included, excluded) -> None:
    print("included records:")
    if included.empty:
        print("  (none)")
    for row in included.itertuples(index=False):
        print(
            "  included: "
            f"feature_date={row.date}, target={row.target}, model={row.model_name}, "
            f"model_type={getattr(row, 'model_type', '-')}, "
            f"predicted_class={row.predicted_class}, source={getattr(row, 'snapshot_source', '-')}, "
            f"reason={getattr(row, 'official_filter_reason', '-')}"
        )
    print("excluded records:")
    if excluded.empty:
        print("  (none)")
    for row in excluded.itertuples(index=False):
        print(
            "  excluded: "
            f"feature_date={row.date}, target={row.target}, model={row.model_name}, "
            f"model_type={getattr(row, 'model_type', '-')}, "
            f"predicted_class={row.predicted_class}, source={getattr(row, 'snapshot_source', '-')}, "
            f"reason={getattr(row, 'excluded_reason', '-')}"
        )


def _build_features_for_ticker(ticker: str, settings) -> Path:
    raw_path = settings.paths.raw_data_dir / f"{ticker}.parquet"
    raw = read_parquet(raw_path)
    features = add_technical_features(
        raw,
        moving_average_windows=settings.features.moving_average_windows,
        rsi_window=settings.features.rsi_window,
        volatility_window=settings.features.volatility_window,
        bollinger_window=settings.features.bollinger_window,
    )
    features = add_external_market_features(features, settings.paths.raw_data_dir)
    from stock_ai.features.overnight import add_overnight_features
    from stock_ai.features.intraday import add_intraday_premarket_features

    features = add_overnight_features(features)
    features = add_intraday_premarket_features(features)
    features = add_earnings_event_features(
        features,
        default_events_dir(settings.paths.raw_data_dir.parent),
        ticker,
    )
    features = add_news_features(
        features,
        default_news_dir(settings.paths.raw_data_dir.parent),
        ticker,
    )
    features = add_narrative_features(
        features,
        default_news_dir(settings.paths.raw_data_dir.parent),
        ticker,
    )
    features = add_next_close_target(
        features,
        surge_threshold=settings.labels.surge_threshold,
        up_threshold=settings.labels.up_threshold,
        down_threshold=settings.labels.down_threshold,
        crash_threshold=settings.labels.crash_threshold,
    )
    features = add_next_close_binary_target(features)
    features = add_next_open_binary_target(features)
    features = add_open_to_close_target(
        features,
        surge_threshold=settings.labels.surge_threshold,
        up_threshold=settings.labels.up_threshold,
        down_threshold=settings.labels.down_threshold,
        crash_threshold=settings.labels.crash_threshold,
    )
    output_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    write_parquet(features, output_path)
    return output_path


def train_overnight_model_for_ticker(
    ticker: str, settings
) -> tuple[Path, Path, dict]:
    from stock_ai.labels.targets import TARGET_NEXT_OPEN_BINARY, add_next_open_binary_target
    from stock_ai.models.baseline import (
        evaluate_binary_classifier,
        get_feature_importance,
        save_model,
    )
    from stock_ai.models.overnight import (
        OVERNIGHT_ARTIFACT_FILENAME,
        OVERNIGHT_METRICS_FILENAME,
        OVERNIGHT_QUANTILE_CUTOFFS,
        compute_empirical_quantile_cutoffs,
        train_lightgbm_overnight_classifier,
    )

    feature_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    dataset = read_parquet(feature_path)
    if TARGET_NEXT_OPEN_BINARY not in dataset.columns:
        dataset = add_next_open_binary_target(dataset)
    train_data, valid_data, test_data = chronological_train_valid_test_split(dataset)
    model, feature_columns = train_lightgbm_overnight_classifier(train_data, valid_data)
    metrics = evaluate_binary_classifier(
        model,
        test_data,
        feature_columns,
        target_column=TARGET_NEXT_OPEN_BINARY,
    )
    # Calibrate quantile cutoffs strictly on the historical validation window (In-sample Selection Bias prevention)
    valid_sub = valid_data.dropna(subset=feature_columns + [TARGET_NEXT_OPEN_BINARY])
    valid_sub = valid_sub[valid_sub[TARGET_NEXT_OPEN_BINARY].isin([0, 1])]
    if not valid_sub.empty:
        valid_probs = model.predict_proba(valid_sub[feature_columns])[:, 1]
        quantile_cutoffs = compute_empirical_quantile_cutoffs(
            valid_sub[TARGET_NEXT_OPEN_BINARY].to_numpy(),
            valid_probs,
        )
    else:
        quantile_cutoffs = OVERNIGHT_QUANTILE_CUTOFFS

    top_features = get_feature_importance(model, feature_columns, top_n=10)
    model_filename = OVERNIGHT_ARTIFACT_FILENAME.format(ticker=ticker)
    model_path = settings.paths.model_dir / model_filename
    save_model(
        model,
        feature_columns,
        model_path,
        metadata={
            "target": TARGET_NEXT_OPEN_BINARY,
            "metrics": metrics,
            "quantile_cutoffs": quantile_cutoffs,
            "top_features": top_features,
        },
    )
    metrics_filename = OVERNIGHT_METRICS_FILENAME.format(ticker=ticker)
    metrics_path = Path("reports") / "metrics" / metrics_filename
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "quantile_cutoffs": quantile_cutoffs,
                "feature_columns": feature_columns,
                "top_features": top_features,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return model_path, metrics_path, metrics


def predict_overnight_for_ticker(
    ticker: str, settings, model_path: Path | None = None
) -> tuple[int, Any]:
    from stock_ai.models.overnight import (
        OVERNIGHT_ARTIFACT_FILENAME,
        OVERNIGHT_MODEL_NAME,
        OVERNIGHT_TARGET,
        predict_latest_overnight_rows,
    )
    from stock_ai.models.predict import load_model_bundle

    feature_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    target_model_path = (
        model_path
        if model_path
        else settings.paths.model_dir / OVERNIGHT_ARTIFACT_FILENAME.format(ticker=ticker)
    )
    dataset = read_parquet(feature_path)
    bundle = load_model_bundle(target_model_path)
    top_features = bundle.get("metadata", {}).get("top_features", [])
    predictions = predict_latest_overnight_rows(
        bundle["model"],
        bundle["feature_columns"],
        dataset,
        top_features=top_features,
        target=OVERNIGHT_TARGET,
        model_name=OVERNIGHT_MODEL_NAME,
        ticker=ticker,
    )
    count = upsert_predictions(predictions, settings.paths.database_path)
    return count, predictions


def pd_to_date_string(value) -> str:
    import pandas as pd

    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return "-"
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()

