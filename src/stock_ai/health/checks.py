from __future__ import annotations

import argparse
import importlib
import json
import re
import sqlite3
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.backtest.prediction_history import (
    diagnose_missing_live_dates,
    summarize_prediction_operations,
)
from stock_ai.config import PROJECT_ROOT, load_settings
from stock_ai.external.market_data import (
    AUTO_SECTOR_SPECS,
    CORE_MARKET_SPECS,
    external_data_path,
)
from stock_ai.news.manual_csv import default_news_dir
from stock_ai.news.quality import check_news_for_ticker
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.storage.sqlite_store import read_prediction_history
from stock_ai.utils.safe_cast import safe_int


DEFAULT_EXPECTED_ROOT = Path(r"C:\Users\kmats\Documents\ai-stock\stock-ai")
DEFAULT_TICKER = "7203.T"
OLD_PATH_MARKERS = (
    str(Path("C:/Users/kmats/Documents") / "ai 譬ｪ" / "stock-ai"),
    str(Path("C:/Users/kmats/Desktop") / "stock-ai"),
)
PROJECT_TEXT_CHECKS = (
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-freeze.txt",
    "configs",
    "scripts",
    "tests",
    "src",
)
REQUIRED_CLI_COMMANDS = (
    "download-external",
    "download",
    "fetch-news",
    "check-news",
    "refresh-news-scores",
    "build-features",
    "data-freshness",
    "train-binary",
    "predict-binary",
    "train-multiclass",
    "predict-multiclass",
    "backfill-predictions",
)
STREAMLIT_REQUIRED_FUNCTIONS = (
    "main",
    "_render_prediction_tab",
    "_render_prediction_history_tab",
    "_render_backtest_tab",
    "_render_evaluation_tab",
    "_render_model_compare_tab",
    "_render_ablation_tab",
    "_render_similar_patterns_tab",
    "_render_cross_ticker_similarity_tab",
    "_render_lead_lag_tab",
    "_render_anomaly_tab",
    "_render_operations_tab",
)


@dataclass
class HealthItem:
    section: str
    name: str
    status: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


def run_health_check(
    *,
    ticker: str = DEFAULT_TICKER,
    expected_root: str | Path = DEFAULT_EXPECTED_ROOT,
    log_tail: int = 40,
) -> dict[str, Any]:
    expected = Path(expected_root)
    settings = load_settings()
    items: list[HealthItem] = []

    items.extend(_check_project_root(expected))
    items.extend(_check_python_and_imports(expected))
    feature_data = _read_features_for_health(settings.paths.feature_data_dir / f"{ticker}.parquet")
    items.extend(_check_data(settings, ticker, feature_data))
    items.extend(_check_news_quality(settings, ticker))
    items.extend(_check_models(settings, ticker))
    items.extend(_check_sqlite(settings, ticker, feature_data))
    items.extend(_check_daily_predict(log_tail=log_tail))
    items.extend(_check_cli_commands())
    items.extend(_check_streamlit_app())

    overall = _overall_status(items)
    item_dicts = [_json_ready(asdict(item)) for item in items]
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "overall_status": overall,
        "expected_root": str(expected),
        "project_root": str(PROJECT_ROOT),
        "current_working_directory": str(Path.cwd()),
        "items": item_dicts,
        "warning_summary": classify_health_warnings(item_dicts),
    }


def print_health_report(report: dict[str, Any]) -> None:
    print("stock-ai health check")
    print(f"Generated at: {report.get('generated_at')}")
    print(f"Expected root: {report.get('expected_root')}")
    print(f"Project root:  {report.get('project_root')}")
    print(f"Current dir:   {report.get('current_working_directory')}")
    print("")

    current_section = None
    for item in report.get("items", []):
        section = item.get("section", "General")
        if section != current_section:
            current_section = section
            print(f"== {section} ==")
        print(f"[{item.get('status')}] {item.get('name')}: {item.get('message')}")
        details = item.get("details") or {}
        for key, value in details.items():
            if key == "log_tail" and isinstance(value, list):
                print("  log_tail:")
                for line in value:
                    print(f"    {line}")
            elif isinstance(value, (list, tuple)):
                joined = ", ".join(str(part) for part in value)
                print(f"  {key}: {joined}")
            else:
                print(f"  {key}: {value}")
        print("")

    summary = report.get("warning_summary") or {}
    if summary:
        print("== Warning summary ==")
        print(f"Action required: {summary.get('action_required_count', 0)}")
        print(f"Known missed sections: {summary.get('known_missed_section_count', 0)}")
        print(f"Known missed dates: {summary.get('known_missed_date_count', 0)}")
        print(f"Known OK: {summary.get('known_ok_count', 0)}")
        print("")

    print(f"Overall status: {report.get('overall_status')}")


def classify_health_warnings(items: list[dict[str, Any] | HealthItem]) -> dict[str, Any]:
    action_required: list[dict[str, Any]] = []
    known_missed: list[dict[str, Any]] = []
    known_ok: list[dict[str, Any]] = []

    for item in items:
        item_dict = _health_item_to_dict(item)
        status = str(item_dict.get("status") or "").upper()
        if status == "OK":
            continue
        classification, reason = _classify_health_warning(item_dict)
        record = {
            "section": item_dict.get("section"),
            "name": item_dict.get("name"),
            "status": status,
            "message": item_dict.get("message"),
            "classification": classification,
            "reason": reason,
            "details": item_dict.get("details") or {},
        }
        if classification == "known_ok":
            known_ok.append(record)
        elif classification == "known_missed":
            known_missed.append(record)
        else:
            action_required.append(record)

    known_missed_date_count = _known_missed_date_count(known_missed)
    return {
        "action_required_count": len(action_required),
        "known_missed_count": known_missed_date_count,
        "known_missed_section_count": len(known_missed),
        "known_missed_date_count": known_missed_date_count,
        "known_ok_count": len(known_ok),
        "action_required": action_required,
        "known_missed": known_missed,
        "known_ok": known_ok,
    }


def _known_missed_date_count(items: list[dict[str, Any]]) -> int:
    count = 0
    fallback_count = 0
    for item in items:
        details = item.get("details") or {}
        rows = details.get("known_missed_live_dates")
        if isinstance(rows, list):
            count += len([row for row in rows if isinstance(row, dict)])
        else:
            fallback_count += 1
    return count if count > 0 else fallback_count


def _health_item_to_dict(item: dict[str, Any] | HealthItem) -> dict[str, Any]:
    if isinstance(item, HealthItem):
        return _json_ready(asdict(item))
    return item


def _classify_health_warning(item: dict[str, Any]) -> tuple[str, str]:
    status = str(item.get("status") or "").upper()
    section = str(item.get("section") or "")
    name = str(item.get("name") or "")
    details = item.get("details") or {}
    message = str(item.get("message") or "")

    if status == "FAIL":
        return "action_required", "FAIL items should be checked before relying on automation."

    if section == "External data" and name == "TOPIX parquet":
        if str(details.get("data_status") or "").lower() == "fallback" or "fallback" in message.lower():
            return (
                "known_ok",
                "TOPIX is continuing through the configured fallback source.",
            )

    if section == "SQLite" and name == "Missing live dates":
        unknown_missing_after_start = safe_int(
            details.get("unknown_missing_prediction_count_after_start")
        )
        if unknown_missing_after_start is None:
            unknown_missing_after_start = safe_int(
                details.get(
                    "missing_prediction_count_after_start",
                    details.get("missing_prediction_count"),
                )
            )
        known_missed_count = safe_int(details.get("known_missed_live_dates_count")) or 0
        if unknown_missing_after_start is None or unknown_missing_after_start <= 0:
            if known_missed_count > 0:
                return (
                    "known_missed",
                    "Missing live prediction dates are registered with explicit known-missed reasons.",
                )
            return (
                "known_ok",
                "Only pre-operation live prediction gaps are present or the configured window is covered.",
            )
        return (
            "action_required",
            "Unknown live prediction dates are missing after the configured operation start date.",
        )

    if section == "News" and name == "News CSV quality":
        unknown_sources = safe_int(details.get("unknown_source_count")) or 0
        if unknown_sources > 0:
            return "action_required", "Unknown news sources need source profile review."

    if section == "daily_predict" and name in {
        "Latest run",
        "Optional news latest run",
        "Windows scheduled task",
    }:
        return "action_required", "Daily automation or its scheduled run should be reviewed."

    if name in {"Latest binary prediction", "Latest multiclass prediction"}:
        return "action_required", "Latest saved predictions are required for daily operation."

    if section in {"Python / venv", "Models", "Data", "SQLite", "CLI", "Streamlit"}:
        return "action_required", "Core runtime, data, model, storage, CLI, or UI check needs review."

    return "action_required", "WARN item is not in the known-OK list."


def add_health_check_parser(subparsers: argparse._SubParsersAction) -> None:
    health = subparsers.add_parser(
        "health-check",
        help="Run diagnostic checks for paths, venv, data, models, SQLite, daily job, and UI",
    )
    health.add_argument("--ticker", default=DEFAULT_TICKER)
    health.add_argument("--expected-root", default=str(DEFAULT_EXPECTED_ROOT))
    health.add_argument("--tail", type=int, default=40, help="Number of daily log tail lines to show")
    health.add_argument("--json", action="store_true", help="Print machine-readable JSON")


def handle_health_check_command(args) -> None:
    _configure_safe_stdout()
    report = run_health_check(
        ticker=args.ticker,
        expected_root=args.expected_root,
        log_tail=args.tail,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_health_report(report)


def _configure_safe_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(errors="replace")


def _check_project_root(expected: Path) -> list[HealthItem]:
    items: list[HealthItem] = []
    project_root = PROJECT_ROOT.resolve()
    cwd = Path.cwd().resolve()
    expected_resolved = expected.resolve()

    items.append(
        HealthItem(
            section="Project root",
            name="PROJECT_ROOT",
            status="OK" if project_root == expected_resolved else "FAIL",
            message=(
                "PROJECT_ROOT matches the expected folder."
                if project_root == expected_resolved
                else "PROJECT_ROOT does not match the expected folder."
            ),
            details={"actual": project_root, "expected": expected_resolved},
        )
    )
    items.append(
        HealthItem(
            section="Project root",
            name="Current working directory",
            status="OK" if cwd == expected_resolved else "WARN",
            message=(
                "Current directory is the expected project folder."
                if cwd == expected_resolved
                else "Command is running from a different directory."
            ),
            details={"actual": cwd, "expected": expected_resolved},
        )
    )

    old_path_refs = _find_old_path_references(PROJECT_ROOT)
    items.append(
        HealthItem(
            section="Project root",
            name="Old path references",
            status="OK" if not old_path_refs else "WARN",
            message=(
                "No old project path references found in checked text files."
                if not old_path_refs
                else "Old project path references remain in checked text files."
            ),
            details={"matches": old_path_refs[:20], "match_count": len(old_path_refs)},
        )
    )
    return items


def _check_python_and_imports(expected: Path) -> list[HealthItem]:
    items: list[HealthItem] = []
    python_path = expected / ".venv" / "Scripts" / "python.exe"
    python_exists = python_path.exists()
    version = _run_command([str(python_path), "--version"]) if python_exists else None
    items.append(
        HealthItem(
            section="Python / venv",
            name=".venv python",
            status="OK" if python_exists and version and version["returncode"] == 0 else "FAIL",
            message=(
                version["stdout"] or version["stderr"]
                if python_exists and version
                else ".venv\\Scripts\\python.exe was not found."
            ),
            details={"path": python_path, "sys_executable": sys.executable},
        )
    )

    stock_ai_module = _import_module("stock_ai")
    items.append(
        HealthItem(
            section="Python / venv",
            name="stock_ai import",
            status="OK" if stock_ai_module["ok"] else "FAIL",
            message=stock_ai_module["message"],
            details={"module_file": stock_ai_module.get("file")},
        )
    )

    for module_name in ("streamlit", "pytest", "lightgbm", "shap", "yfinance"):
        result = _import_module(module_name)
        items.append(
            HealthItem(
                section="Python / venv",
                name=f"import {module_name}",
                status="OK" if result["ok"] else "FAIL",
                message=result["message"],
                details={"module_file": result.get("file")},
            )
        )
    return items


def _check_data(settings, ticker: str, feature_data: pd.DataFrame | None) -> list[HealthItem]:
    items: list[HealthItem] = []
    raw_path = settings.paths.raw_data_dir / f"{ticker}.parquet"
    feature_path = settings.paths.feature_data_dir / f"{ticker}.parquet"
    items.append(_file_item("Data", "Toyota raw data", raw_path, fail_if_missing=True))
    items.append(_file_item("Data", "Toyota features", feature_path, fail_if_missing=True))

    if feature_data is None or feature_data.empty:
        items.append(
            HealthItem(
                section="Data",
                name="Feature summary",
                status="FAIL",
                message="Feature parquet could not be read or contains no rows.",
                details={"path": feature_path},
            )
        )
    else:
        latest_date = _latest_date_string(feature_data.get("date"))
        items.append(
            HealthItem(
                section="Data",
                name="Feature summary",
                status="OK",
                message=f"Feature data loaded. Latest feature date is {latest_date}.",
                details={"rows": len(feature_data), "latest_feature_date": latest_date},
            )
        )
        items.extend(_check_feature_columns(feature_data))

    feature_latest_date = _latest_date_string(feature_data.get("date")) if feature_data is not None else None
    items.extend(
        _external_data_items(
            settings.paths.raw_data_dir,
            CORE_MARKET_SPECS,
            section="External data",
            feature_latest_date=feature_latest_date,
        )
    )
    items.extend(
        _external_data_items(
            settings.paths.raw_data_dir,
            AUTO_SECTOR_SPECS,
            section="External data",
            feature_latest_date=feature_latest_date,
        )
    )

    events_dir = settings.paths.raw_data_dir.parent / "events"
    news_dir = settings.paths.raw_data_dir.parent / "news"
    items.append(
        _group_files_item(
            "Data",
            "Earnings CSV",
            {
                "earnings_7203.T.csv": events_dir / "earnings_7203.T.csv",
                "toyota_earnings.csv": events_dir / "toyota_earnings.csv",
            },
            any_one_ok=True,
            warn_only=True,
        )
    )
    items.append(
        _group_files_item(
            "Data",
            "News CSV",
            {"toyota_news.csv": news_dir / "toyota_news.csv"},
            warn_only=True,
        )
    )
    return items


def _check_news_quality(settings, ticker: str) -> list[HealthItem]:
    news_dir = default_news_dir(settings.paths.raw_data_dir.parent)
    report = check_news_for_ticker(news_dir, ticker)
    status = str(report.get("status") or "WARN")
    if status not in {"OK", "WARN", "FAIL"}:
        status = "WARN"
    unknown_source_count = safe_int(report.get("unknown_source_count")) or 0
    message = str(report.get("message") or "News CSV quality status is unavailable.")
    if unknown_source_count > 0 and status == "OK":
        status = "WARN"
        message = (
            f"{message} {unknown_source_count} unknown news source(s) need profile review."
        )
    return [
        HealthItem(
            section="News",
            name="News CSV quality",
            status=status,
            message=message,
            details=report,
        )
    ]


def _check_feature_columns(data: pd.DataFrame) -> list[HealthItem]:
    groups = {
        "External market feature columns": ("usdjpy_return_1d", "n225_return_1d"),
        "Auto sector feature columns": ("sector_mean_return", "tesla_return_1d", "honda_return_1d"),
        "Earnings feature columns": ("days_to_earnings", "earnings_week_flag"),
        "News feature columns": ("news_count_1d", "avg_sentiment_3d", "negative_news_flag"),
        "Narrative feature columns": ("dominant_narrative_code", "dominant_narrative_trend"),
    }
    items: list[HealthItem] = []
    columns = set(data.columns)
    for name, expected_columns in groups.items():
        present = [column for column in expected_columns if column in columns]
        missing = [column for column in expected_columns if column not in columns]
        items.append(
            HealthItem(
                section="Data",
                name=name,
                status="OK" if not missing else "WARN",
                message=(
                    "Expected feature columns are present."
                    if not missing
                    else "Some expected feature columns are missing."
                ),
                details={"present": present, "missing": missing},
            )
        )
    return items


def _external_data_items(
    raw_data_dir: Path,
    specs,
    *,
    section: str,
    feature_latest_date: str | None,
) -> list[HealthItem]:
    return [
        _external_data_item(
            raw_data_dir,
            spec,
            section=section,
            feature_latest_date=feature_latest_date,
        )
        for spec in specs
    ]


def _external_data_item(
    raw_data_dir: Path,
    spec,
    *,
    section: str,
    feature_latest_date: str | None,
) -> HealthItem:
    path = external_data_path(raw_data_dir, spec)
    if not path.exists():
        return HealthItem(
            section=section,
            name=f"{spec.name} parquet",
            status="WARN",
            message=f"{spec.name} parquet is missing.",
            details={
                "symbol": spec.symbol,
                "fallback_symbols": list(getattr(spec, "fallback_symbols", ())),
                "path": path,
                "data_status": "missing",
            },
        )

    try:
        data = read_parquet(path)
    except Exception as error:
        return HealthItem(
            section=section,
            name=f"{spec.name} parquet",
            status="WARN",
            message=f"{spec.name} parquet exists but could not be read.",
            details={"symbol": spec.symbol, "path": path, "error": str(error), "data_status": "failed"},
        )

    rows = int(len(data))
    latest_date = _latest_date_string(data.get("date"))
    stale = _is_external_stale(latest_date, feature_latest_date)
    fallback_warning = _latest_external_fallback_warning(spec.symbol)
    status = "WARN" if stale or fallback_warning else "OK"
    data_status = "stale" if stale else "fallback" if fallback_warning else "ok"
    stat = path.stat()
    return HealthItem(
        section=section,
        name=f"{spec.name} parquet",
        status=status,
        message=(
            f"{spec.name} parquet is stale."
            if stale
            else f"{spec.name} parquet is available via fallback."
            if fallback_warning
            else f"{spec.name} parquet is available."
        ),
        details={
            "symbol": spec.symbol,
            "fallback_symbols": list(getattr(spec, "fallback_symbols", ())),
            "path": path,
            "rows": rows,
            "latest_date": latest_date,
            "feature_latest_date": feature_latest_date,
            "data_status": data_status,
            "fallback_warning": fallback_warning,
            "modified_at": _timestamp(stat.st_mtime),
        },
    )


def _is_external_stale(latest_date: str | None, feature_latest_date: str | None) -> bool:
    if latest_date is None or feature_latest_date is None:
        return False
    latest = pd.to_datetime(latest_date, errors="coerce")
    feature_latest = pd.to_datetime(feature_latest_date, errors="coerce")
    if pd.isna(latest) or pd.isna(feature_latest):
        return False
    return (feature_latest - latest).days > 7


def _check_models(settings, ticker: str) -> list[HealthItem]:
    model_dir = settings.paths.model_dir
    metrics_dir = PROJECT_ROOT / "reports" / "metrics"
    paths = {
        "Binary model": model_dir / f"lightgbm_binary_{ticker}.joblib",
        "Multiclass model": model_dir / f"lightgbm_multiclass_{ticker}.joblib",
        "Binary metrics": metrics_dir / f"lightgbm_binary_{ticker}.json",
        "Multiclass metrics": metrics_dir / f"lightgbm_multiclass_{ticker}.json",
    }
    return [_file_item("Models", name, path, fail_if_missing=True) for name, path in paths.items()]


def _check_sqlite(
    settings,
    ticker: str,
    feature_data: pd.DataFrame | None,
) -> list[HealthItem]:
    items: list[HealthItem] = []
    db_path = settings.paths.database_path
    items.append(_file_item("SQLite", "Database file", db_path, fail_if_missing=True))
    if not db_path.exists():
        return items

    try:
        with sqlite3.connect(db_path) as connection:
            table_exists = _sqlite_table_exists(connection, "predictions")
            items.append(
                HealthItem(
                    section="SQLite",
                    name="predictions table",
                    status="OK" if table_exists else "FAIL",
                    message=(
                        "predictions table exists."
                        if table_exists
                        else "predictions table is missing."
                    ),
                )
            )
            if not table_exists:
                return items
            items.extend(_sqlite_prediction_items(connection, ticker))
    except sqlite3.Error as error:
        items.append(
            HealthItem(
                section="SQLite",
                name="SQLite read",
                status="FAIL",
                message=f"SQLite read failed: {error}",
                details={"path": db_path},
            )
        )
        return items

    try:
        predictions = read_prediction_history(
            db_path,
            ticker=ticker,
            target="next_close_binary",
            model_name="lightgbm_binary",
        )
        all_predictions = read_prediction_history(db_path, ticker=ticker)
        if feature_data is not None and not feature_data.empty:
            operations = summarize_prediction_operations(
                predictions,
                feature_data,
                ticker=ticker,
                lookback_trading_days=60,
                live_prediction_start_date=settings.operations.live_prediction_start_date,
            )
            missing_count = int(operations.get("missing_prediction_count", 0))
            pre_start_excluded_count = int(
                operations.get(
                    "pre_start_excluded_count",
                    operations.get("ignored_pre_start_missing_count", 0),
                )
            )
            start_date = _date_like(operations.get("live_prediction_start_date"))
            daily_state = _latest_daily_predict_state()
            diagnostics = diagnose_missing_live_dates(
                all_predictions,
                feature_data,
                operations.get("missing_prediction_dates", []) or [],
                ticker=ticker,
                known_missed_live_dates=settings.operations.known_missed_live_dates,
            )
            known_missed_count = _known_missed_count(diagnostics)
            unknown_missing_count = max(missing_count - known_missed_count, 0)
            known_config_warnings = _known_missed_config_warnings(
                settings.operations.known_missed_live_dates,
                ticker=ticker,
            )
            if missing_count == 0:
                message = (
                    "No live prediction dates are missing after the configured "
                    f"operation start date ({start_date or 'not set'})."
                )
            elif unknown_missing_count == 0:
                message = (
                    "All missing live prediction dates after the configured operation start "
                    "are registered as known missed live dates with reasons."
                )
            elif daily_state.get("latest_success"):
                message = (
                    "daily automation is currently working, but live predictions are missing "
                    f"after the configured operation start date ({unknown_missing_count} unknown dates)."
                )
            else:
                message = (
                    f"{unknown_missing_count} unknown live prediction dates are missing after the configured start date. "
                    "The latest daily automation success could not be confirmed."
                )
            items.append(
                HealthItem(
                    section="SQLite",
                    name="Missing live dates",
                    status="OK" if missing_count == 0 else "WARN",
                    message=message,
                    details={
                        "live_prediction_start_date": start_date,
                        "pre_start_excluded_date_count": pre_start_excluded_count,
                        "missing_prediction_count_after_start": missing_count,
                        "missing_prediction_count": missing_count,
                        "unknown_missing_prediction_count_after_start": unknown_missing_count,
                        "known_missed_live_dates_count": known_missed_count,
                        "known_missed_live_dates": _known_missed_json_records(diagnostics),
                        "known_missed_config_warnings": known_config_warnings,
                        "latest_prediction_date": _date_like(operations.get("latest_prediction_date")),
                        "latest_saved_at": _date_like(operations.get("latest_saved_at")),
                        "next_confirmable_prediction_date": _date_like(
                            operations.get("next_confirmable_prediction_date")
                        ),
                        "backfilled_missing_count": operations.get("backfilled_missing_count", 0),
                        "backfillable_missing_count": operations.get("backfillable_missing_count", 0),
                        "daily_latest_success": daily_state.get("latest_success"),
                        "missing_live_date_diagnostics": _diagnostics_json_records(diagnostics),
                    },
                )
            )
    except Exception as error:
        items.append(
            HealthItem(
                section="SQLite",
                name="Missing live dates",
                status="WARN",
                message=f"Could not calculate missing live dates: {error}",
            )
        )
    return items


def _diagnostics_json_records(diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    if diagnostics.empty:
        return []
    records: list[dict[str, Any]] = []
    date_columns = {
        "missing_date",
        "previous_live_prediction_date",
        "previous_live_saved_at",
        "next_live_prediction_date",
    }
    for row in diagnostics.to_dict(orient="records"):
        item: dict[str, Any] = {}
        for key, value in row.items():
            if key in date_columns:
                item[key] = _date_like(value)
            else:
                item[key] = value
        records.append(item)
    return records


def _known_missed_count(diagnostics: pd.DataFrame) -> int:
    if diagnostics.empty or "known_missed" not in diagnostics.columns:
        return 0
    return int(diagnostics["known_missed"].fillna(False).astype(bool).sum())


def _known_missed_json_records(diagnostics: pd.DataFrame) -> list[dict[str, Any]]:
    if diagnostics.empty or "known_missed" not in diagnostics.columns:
        return []
    known = diagnostics[diagnostics["known_missed"].fillna(False).astype(bool)].copy()
    return _diagnostics_json_records(known)


def _known_missed_config_warnings(known_missed_live_dates, *, ticker: str) -> list[str]:
    warnings: list[str] = []
    seen: set[tuple[str, str]] = set()
    for item in known_missed_live_dates or []:
        date_text = str(getattr(item, "date", "") or "")
        item_ticker = str(getattr(item, "ticker", "") or "")
        reason_code = str(getattr(item, "reason_code", "") or "")
        action = str(getattr(item, "action", "") or "")
        parsed = pd.to_datetime(date_text, errors="coerce")
        if pd.isna(parsed):
            warnings.append(f"date parse failed: {date_text or '<empty>'}")
        if item_ticker and item_ticker != ticker:
            continue
        if not reason_code:
            warnings.append(f"reason_code is empty for {date_text or '<empty>'}")
        if not action:
            warnings.append(f"action is empty for {date_text or '<empty>'}")
        key = (item_ticker or ticker, pd.Timestamp(parsed).strftime("%Y-%m-%d") if not pd.isna(parsed) else date_text)
        if key in seen:
            warnings.append(f"duplicate known missed date: {key[0]} {key[1]}")
        seen.add(key)
    return warnings


def _sqlite_prediction_items(connection: sqlite3.Connection, ticker: str) -> list[HealthItem]:
    items: list[HealthItem] = []
    for name, target, model_name in (
        ("Latest binary prediction", "next_close_binary", "lightgbm_binary"),
        ("Latest multiclass prediction", "next_close_multiclass", "lightgbm_multiclass"),
    ):
        row = connection.execute(
            """
            SELECT date, created_at, predicted_class, confidence
            FROM predictions
            WHERE ticker = ? AND target = ? AND model_name = ?
            ORDER BY date DESC, created_at DESC
            LIMIT 1
            """,
            (ticker, target, model_name),
        ).fetchone()
        items.append(
            HealthItem(
                section="SQLite",
                name=name,
                status="OK" if row else "FAIL",
                message=(
                    f"Latest prediction date is {row[0]}."
                    if row
                    else "No saved prediction row found."
                ),
                details=(
                    {
                        "date": row[0],
                        "created_at": row[1],
                        "predicted_class": row[2],
                        "confidence": row[3],
                    }
                    if row
                    else {"target": target, "model_name": model_name}
                ),
            )
        )

    counts = connection.execute(
        """
        SELECT COALESCE(prediction_type, 'live') AS prediction_type, COUNT(*)
        FROM predictions
        WHERE ticker = ?
        GROUP BY COALESCE(prediction_type, 'live')
        ORDER BY prediction_type
        """,
        (ticker,),
    ).fetchall()
    items.append(
        HealthItem(
            section="SQLite",
            name="prediction_type counts",
            status="OK" if counts else "WARN",
            message="Prediction type counts were read." if counts else "No prediction rows found.",
            details={"counts": {row[0]: row[1] for row in counts}},
        )
    )
    return items


def _check_daily_predict(log_tail: int) -> list[HealthItem]:
    items: list[HealthItem] = []
    script_path = PROJECT_ROOT / "scripts" / "daily_predict.ps1"
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    items.append(_file_item("daily_predict", "daily_predict.ps1", script_path, fail_if_missing=True))
    items.append(
        _file_item(
            "daily_predict",
            "register_daily_predict_task.ps1",
            PROJECT_ROOT / "scripts" / "register_daily_predict_task.ps1",
            fail_if_missing=False,
        )
    )
    items.append(
        _file_item(
            "daily_predict",
            "show_daily_predict_task.ps1",
            PROJECT_ROOT / "scripts" / "show_daily_predict_task.ps1",
            fail_if_missing=False,
        )
    )

    if script_path.exists():
        script_text = _read_text(script_path)
        old_matches = [marker for marker in OLD_PATH_MARKERS if marker in script_text]
        items.append(
            HealthItem(
                section="daily_predict",
                name="Old paths in daily_predict.ps1",
                status="OK" if not old_matches else "FAIL",
                message=(
                    "No old project path references found in daily_predict.ps1."
                    if not old_matches
                    else "Old project path references remain in daily_predict.ps1."
                ),
                details={"matches": old_matches},
            )
        )
        warning_safe = (
            '$ErrorActionPreference = "Continue"' in script_text
            and "$ExitCode = $LASTEXITCODE" in script_text
        )
        items.append(
            HealthItem(
                section="daily_predict",
                name="Warnings are non-fatal",
                status="OK" if warning_safe else "WARN",
                message=(
                    "Python stderr warnings are logged and exit code decides failure."
                    if warning_safe
                    else "Could not confirm warning-safe Python step handling."
                ),
            )
        )
        fetch_news_option = "[switch]$FetchNews" in script_text and "fetch-news" in script_text
        items.append(
            HealthItem(
                section="daily_predict",
                name="Optional news fetch support",
                status="OK" if fetch_news_option else "WARN",
                message=(
                    "daily_predict supports optional FetchNews without making news mandatory."
                    if fetch_news_option
                    else "Could not confirm optional FetchNews support in daily_predict.ps1."
                ),
            )
        )

    items.append(_file_item("daily_predict", "daily_predict.log", log_path, fail_if_missing=False))
    if log_path.exists():
        log_lines = _read_text(log_path).splitlines()
        summary = _daily_log_summary(log_lines, log_tail=log_tail)
        status = "OK"
        message = "Latest daily_predict run finished successfully."
        if summary["latest_error"]:
            status = "FAIL"
            message = "Latest daily_predict run failed."
        elif summary["incomplete_step"]:
            status = "FAIL"
            message = f"Latest daily_predict run is incomplete at {summary['incomplete_step']}."
        elif not summary["latest_success"]:
            status = "WARN"
            message = "No successful daily_predict run was found in the latest log block."
        items.append(
            HealthItem(
                section="daily_predict",
                name="Latest run",
                status=status,
                message=message,
                details=summary,
            )
        )
        news_warning = summary.get("latest_news_warning")
        fetch_news_enabled = summary.get("fetch_news_enabled")
        news_status = "OK"
        news_message = "Optional news fetch was disabled in the latest daily_predict run."
        if fetch_news_enabled is True and news_warning:
            news_status = "WARN"
            news_message = "Optional news fetch/check-news had warnings, but prediction flow can continue."
        elif fetch_news_enabled is True:
            news_message = "Optional news fetch/check-news completed in the latest daily_predict run."
        elif fetch_news_enabled is None:
            news_message = "Latest log predates FetchNews tracking; current script keeps news fetch disabled by default."
        items.append(
            HealthItem(
                section="daily_predict",
                name="Optional news latest run",
                status=news_status,
                message=news_message,
                details={
                    "fetch_news_enabled": fetch_news_enabled,
                    "news_query_profile": summary.get("news_query_profile"),
                    "daily_news_profiles": summary.get("daily_news_profiles"),
                    "all_news_profiles": summary.get("all_news_profiles"),
                    "news_profile_mode": summary.get("news_profile_mode"),
                    "latest_fetch_news_ok": summary.get("latest_fetch_news_ok"),
                    "latest_check_news_ok": summary.get("latest_check_news_ok"),
                    "latest_news_warning": news_warning,
                },
            )
        )
    items.append(_check_windows_scheduled_task())
    return items


def _check_windows_scheduled_task() -> HealthItem:
    task_name = "stock-ai daily predict"
    show_script = PROJECT_ROOT / "scripts" / "show_daily_predict_task.ps1"
    if not show_script.exists():
        return HealthItem(
            section="daily_predict",
            name="Windows scheduled task",
            status="WARN",
            message="Task Scheduler helper script is missing.",
            details={"task_name": task_name, "registered": False, "script": show_script},
        )

    command = [
        "powershell.exe",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(show_script),
        "-AsJson",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return HealthItem(
            section="daily_predict",
            name="Windows scheduled task",
            status="WARN",
            message="Could not read Windows Task Scheduler status.",
            details={"task_name": task_name, "registered": None, "error": str(error)},
        )

    if result.returncode != 0:
        return HealthItem(
            section="daily_predict",
            name="Windows scheduled task",
            status="WARN",
            message="Task Scheduler status command returned a warning.",
            details={
                "task_name": task_name,
                "registered": None,
                "returncode": result.returncode,
                "stderr": result.stderr.strip(),
            },
        )

    try:
        details = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        return HealthItem(
            section="daily_predict",
            name="Windows scheduled task",
            status="WARN",
            message="Task Scheduler status output could not be parsed.",
            details={"task_name": task_name, "registered": None, "error": str(error)},
        )

    registered = bool(details.get("registered"))
    command_text = str(details.get("command") or "")
    working_directory = str(details.get("working_directory") or "")
    required_fragments = (
        "daily_predict.ps1",
        "-FetchNews",
        "-DailyNewsProfiles",
        "-NewsLimit",
        "3",
    )
    command_configured = registered and all(fragment in command_text for fragment in required_fragments)
    working_directory_configured = registered and Path(working_directory).resolve() == PROJECT_ROOT.resolve()
    configured = command_configured and working_directory_configured
    details["expected_command"] = (
        r".\scripts\daily_predict.ps1 -FetchNews -DailyNewsProfiles -NewsLimit 3"
    )
    details["expected_working_directory"] = str(PROJECT_ROOT)
    details["configuration_status"] = "recommended" if configured else "needs_review"

    if not registered:
        status = "WARN"
        message = "Windows scheduled task is not registered yet."
    elif configured:
        status = "OK"
        message = "Windows scheduled task is registered with the recommended NewsLimit 3 flow."
    else:
        status = "WARN"
        message = "Windows scheduled task is registered, but its command should be reviewed."

    return HealthItem(
        section="daily_predict",
        name="Windows scheduled task",
        status=status,
        message=message,
        details=details,
    )


def _check_cli_commands() -> list[HealthItem]:
    cli_path = PROJECT_ROOT / "src" / "stock_ai" / "cli.py"
    if not cli_path.exists():
        return [
            HealthItem(
                section="CLI",
                name="CLI source",
                status="FAIL",
                message="src/stock_ai/cli.py is missing.",
                details={"path": cli_path},
            )
        ]

    text = _read_text(cli_path)
    commands = sorted(set(re.findall(r"add_parser\(\s*[\"']([^\"']+)[\"']", text)))
    items: list[HealthItem] = []
    for command in REQUIRED_CLI_COMMANDS:
        exists = command in commands
        items.append(
            HealthItem(
                section="CLI",
                name=command,
                status="OK" if exists else "FAIL",
                message="CLI command is registered." if exists else "CLI command is missing.",
            )
        )
    return items


def _check_streamlit_app() -> list[HealthItem]:
    items: list[HealthItem] = []
    app_path = PROJECT_ROOT / "src" / "stock_ai" / "app" / "main.py"
    items.append(_file_item("Streamlit", "main.py", app_path, fail_if_missing=True))
    if not app_path.exists():
        return items

    result = _subprocess_import_module("stock_ai.app.main")
    items.append(
        HealthItem(
            section="Streamlit",
            name="import stock_ai.app.main",
            status="OK" if result["ok"] else "FAIL",
            message=result["message"],
            details={"module_file": result.get("file")},
        )
    )
    if not result["ok"]:
        return items

    missing = _missing_functions_in_source(app_path, STREAMLIT_REQUIRED_FUNCTIONS)
    items.append(
        HealthItem(
            section="Streamlit",
            name="Required tab functions",
            status="OK" if not missing else "FAIL",
            message=(
                "Required Streamlit tab functions are present."
                if not missing
                else "Some required Streamlit tab functions are missing."
            ),
            details={"missing": missing},
        )
    )
    return items


def _daily_log_summary(lines: list[str], *, log_tail: int) -> dict[str, Any]:
    latest_block = _latest_daily_log_block(lines)
    starts = _step_names(latest_block, "START:")
    dones = set(_step_names(latest_block, "DONE:"))
    incomplete_steps = [step for step in starts if step not in dones]
    latest_error = _last_line_containing(latest_block, "daily_predict failed")
    ticker_error = _last_line_containing(latest_block, "ERROR for ")
    fetch_news_enabled_line = _last_line_containing(latest_block, "FetchNews enabled:")
    fetch_news_enabled = _parse_bool_from_log_value(fetch_news_enabled_line)
    news_query_profile = _parse_text_from_log_value(
        _last_line_containing(latest_block, "News query profile:")
    )
    daily_news_profiles = _parse_bool_from_log_value(
        _last_line_containing(latest_block, "Daily news profiles:")
    )
    all_news_profiles = _parse_bool_from_log_value(
        _last_line_containing(latest_block, "All news profiles:")
    )
    news_profile_mode = _parse_text_from_log_value(
        _last_line_containing(latest_block, "News profile mode:")
    )
    data_freshness_status = _parse_text_from_log_value(
        _last_line_containing(latest_block, "freshness status:")
    )
    return {
        "latest_success": _last_line_containing(
            latest_block,
            "daily_predict finished successfully",
        ),
        "latest_error": latest_error or ticker_error,
        "latest_event": next((line for line in reversed(latest_block) if line.strip()), None),
        "last_started_step": starts[-1] if starts else None,
        "last_done_step": next((step for step in reversed(starts) if step in dones), None),
        "incomplete_step": incomplete_steps[-1] if incomplete_steps else None,
        "user_warning_count": sum("UserWarning" in line for line in latest_block),
        "lightgbm_warning_count": sum("[LightGBM] [Warning]" in line for line in latest_block),
        "fetch_news_enabled": fetch_news_enabled,
        "news_query_profile": news_query_profile,
        "daily_news_profiles": daily_news_profiles,
        "all_news_profiles": all_news_profiles,
        "news_profile_mode": news_profile_mode,
        "latest_fetch_news_ok": _last_line_containing(latest_block, "NEWS OK: fetch-news"),
        "latest_check_news_ok": _last_line_containing(latest_block, "NEWS OK: check-news"),
        "latest_fetch_news_warning": _last_line_containing(latest_block, "NEWS WARN: fetch-news"),
        "latest_check_news_warning": _last_line_containing(latest_block, "NEWS WARN: check-news"),
        "latest_news_warning": _last_line_containing(latest_block, "NEWS WARN:"),
        "data_freshness_status": data_freshness_status,
        "expected_latest_trading_date": _parse_text_from_log_value(
            _last_line_containing(latest_block, "expected latest trading date:")
        ),
        "latest_raw_date_at_run": _parse_text_from_log_value(
            _last_line_containing(latest_block, "latest raw date:")
        ),
        "latest_feature_date_at_run": _parse_text_from_log_value(
            _last_line_containing(latest_block, "latest feature date:")
        ),
        "prediction_feature_date_at_run": _parse_text_from_log_value(
            _last_line_containing(latest_block, "prediction feature date:")
        ),
        "saved_prediction_date_expected": _parse_text_from_log_value(
            _last_line_containing(latest_block, "saved prediction date expected:")
        ),
        "freshness_lag_trading_days": _parse_int_from_log_value(
            _last_line_containing(latest_block, "freshness lag trading days:")
        ),
        "raw_lag_trading_days": _parse_int_from_log_value(
            _last_line_containing(latest_block, "raw lag trading days:")
        ),
        "stale_at_run_time": _parse_bool_from_log_value(
            _last_line_containing(latest_block, "stale at run time:")
        ),
        "data_freshness_message": _parse_text_from_log_value(
            _last_line_containing(latest_block, "freshness message:")
        ),
        "log_tail": lines[-max(1, log_tail) :],
    }


def _latest_daily_predict_state() -> dict[str, str | None]:
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    if not log_path.exists():
        return {"latest_success": None, "latest_error": None}
    latest_block = _latest_daily_log_block(_read_text(log_path).splitlines())
    return {
        "latest_success": _last_line_containing(
            latest_block,
            "daily_predict finished successfully",
        ),
        "latest_error": _last_line_containing(latest_block, "daily_predict failed"),
    }


def _latest_external_fallback_warning(symbol: str) -> str | None:
    log_path = PROJECT_ROOT / "logs" / "daily_predict.log"
    if not log_path.exists():
        return None
    latest_block = _latest_daily_log_block(_read_text(log_path).splitlines())
    marker = f"{symbol} was unavailable; using fallback"
    return _last_line_containing(latest_block, marker)


def _latest_daily_log_block(lines: list[str]) -> list[str]:
    for index in range(len(lines) - 1, -1, -1):
        if "daily_predict started" in lines[index]:
            return lines[index:]
    return lines


def _step_names(lines: list[str], marker: str) -> list[str]:
    names: list[str] = []
    for line in lines:
        content = line.split("] ", maxsplit=1)[-1]
        if not content.startswith(marker):
            continue
        names.append(content.split(marker, maxsplit=1)[1].strip())
    return names


def _last_line_containing(lines: list[str], marker: str) -> str | None:
    return next((line for line in reversed(lines) if marker in line), None)


def _parse_bool_from_log_value(line: str | None) -> bool | None:
    if not line or ":" not in line:
        return None
    value = line.rsplit(":", maxsplit=1)[-1].strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _parse_int_from_log_value(line: str | None) -> int | None:
    text = _parse_text_from_log_value(line)
    if text is None or text == "None":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_text_from_log_value(line: str | None) -> str | None:
    if not line or ":" not in line:
        return None
    value = line.rsplit(":", maxsplit=1)[-1].strip()
    if not value or value == "<none>":
        return None
    return value


def _file_item(section: str, name: str, path: Path, *, fail_if_missing: bool) -> HealthItem:
    exists = path.exists()
    status = "OK" if exists else ("FAIL" if fail_if_missing else "WARN")
    details: dict[str, Any] = {"path": path}
    if exists:
        stat = path.stat()
        details.update({"size_bytes": stat.st_size, "modified_at": _timestamp(stat.st_mtime)})
    return HealthItem(
        section=section,
        name=name,
        status=status,
        message=f"{name} exists." if exists else f"{name} is missing.",
        details=details,
    )


def _group_files_item(
    section: str,
    name: str,
    paths: dict[str, Path],
    *,
    any_one_ok: bool = False,
    warn_only: bool = False,
) -> HealthItem:
    present = {label: str(path) for label, path in paths.items() if path.exists()}
    missing = {label: str(path) for label, path in paths.items() if not path.exists()}
    ok = bool(present) if any_one_ok else not missing
    if ok:
        status = "OK"
    else:
        status = "WARN" if warn_only else "FAIL"
    return HealthItem(
        section=section,
        name=name,
        status=status,
        message=(
            "Expected files are present."
            if ok
            else "Some expected files are missing."
        ),
        details={"present": present, "missing": missing},
    )


def _sqlite_table_exists(connection: sqlite3.Connection, table: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone()
    return row is not None


def _read_features_for_health(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return read_parquet(path)
    except Exception:
        return None


def _find_old_path_references(root: Path) -> list[str]:
    matches: list[str] = []
    for relative in PROJECT_TEXT_CHECKS:
        path = root / relative
        if not path.exists():
            continue
        candidates = [path] if path.is_file() else _iter_text_files(path)
        for candidate in candidates:
            if candidate == PROJECT_ROOT / "src" / "stock_ai" / "health" / "checks.py":
                continue
            text = _read_text(candidate)
            for marker in OLD_PATH_MARKERS:
                if marker in text:
                    matches.append(str(candidate.relative_to(root)))
                    break
    return sorted(set(matches))


def _iter_text_files(path: Path) -> list[Path]:
    suffixes = {".py", ".ps1", ".md", ".yml", ".yaml", ".toml", ".txt", ".json", ".cmd", ".bat"}
    ignored_parts = {".venv", ".git", "__pycache__", ".pytest_cache", ".ruff_cache"}
    files: list[Path] = []
    for candidate in path.rglob("*"):
        if not candidate.is_file():
            continue
        if any(part in ignored_parts for part in candidate.parts):
            continue
        if candidate.suffix.lower() in suffixes:
            files.append(candidate)
    return files


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=20,
        )
    except Exception as error:
        return {"returncode": -1, "stdout": "", "stderr": str(error)}
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _import_module(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        return {"ok": False, "message": f"Import failed: {error}", "file": None}
    return {
        "ok": True,
        "message": "Import succeeded.",
        "file": getattr(module, "__file__", None),
        "module": module,
    }


def _subprocess_import_module(module_name: str) -> dict[str, Any]:
    snippet = (
        "import pathlib, sys; "
        f"sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r}); "
        f"import {module_name} as module; "
        "print(getattr(module, '__file__', ''))"
    )
    result = _run_command([sys.executable, "-c", snippet])
    ok = result["returncode"] == 0
    return {
        "ok": ok,
        "message": "Import succeeded." if ok else f"Import failed: {result['stderr']}",
        "file": result["stdout"].splitlines()[-1] if ok and result["stdout"] else None,
    }


def _missing_functions_in_source(path: Path, function_names: tuple[str, ...]) -> list[str]:
    text = _read_text(path)
    defined = set(re.findall(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", text, flags=re.MULTILINE))
    return [name for name in function_names if name not in defined]


def _latest_date_string(values) -> str | None:
    if values is None:
        return None
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


def _date_like(value) -> str | None:
    if value is None:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)
    if pd.isna(timestamp):
        return None
    if timestamp.time() == datetime.min.time():
        return timestamp.strftime("%Y-%m-%d")
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def _timestamp(value: float) -> str:
    return datetime.fromtimestamp(value).isoformat(timespec="seconds")


def _overall_status(items: list[HealthItem]) -> str:
    if any(item.status == "FAIL" for item in items):
        return "FAIL"
    if any(item.status == "WARN" for item in items):
        return "WARN"
    return "OK"


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if not isinstance(value, (str, bytes, bytearray, dict, list, tuple)):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
    return value

