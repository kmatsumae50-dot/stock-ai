from __future__ import annotations

from datetime import UTC, datetime
import sqlite3
from pathlib import Path
from uuid import uuid4

import pandas as pd

from stock_ai.utils.safe_cast import safe_float, safe_int


PREDICTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    target TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_class INTEGER NOT NULL,
    confidence REAL,
    probability_surge REAL,
    probability_up REAL,
    probability_flat REAL,
    probability_down REAL,
    probability_crash REAL,
    top_features TEXT,
    prediction_type TEXT NOT NULL DEFAULT 'live',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, ticker, target, model_name)
)
"""

PREDICTION_EXTRA_COLUMNS = {
    "probability_surge": "REAL",
    "probability_up": "REAL",
    "probability_flat": "REAL",
    "probability_down": "REAL",
    "probability_crash": "REAL",
    "top_features": "TEXT",
    "prediction_type": "TEXT NOT NULL DEFAULT 'live'",
}

PREDICTION_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_runs (
    run_id TEXT NOT NULL PRIMARY KEY,
    run_timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    run_type TEXT NOT NULL,
    command_name TEXT,
    source TEXT,
    working_directory TEXT,
    data_freshness_status TEXT,
    latest_raw_date TEXT,
    latest_feature_date TEXT,
    expected_latest_trading_date TEXT,
    stale_at_run_time INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    notes TEXT
)
"""

PREDICTION_RECORDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_records (
    record_id TEXT NOT NULL PRIMARY KEY,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    feature_date TEXT NOT NULL,
    target_close_date TEXT,
    target TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_type TEXT,
    predicted_label TEXT,
    predicted_direction TEXT,
    probability_up REAL,
    probability_down REAL,
    confidence REAL,
    source TEXT NOT NULL,
    is_official INTEGER NOT NULL DEFAULT 0,
    is_latest INTEGER NOT NULL DEFAULT 0,
    actual_return REAL,
    actual_label TEXT,
    actual_direction TEXT,
    result TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    evaluated_at TEXT,
    notes TEXT,
    FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
)
"""

PREDICTION_APPEND_ONLY_INDEXES = [
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_runs_ticker_timestamp
    ON prediction_runs(ticker, run_timestamp)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_runs_type
    ON prediction_runs(run_type)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_records_run_id
    ON prediction_records(run_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_records_feature
    ON prediction_records(ticker, feature_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_records_source
    ON prediction_records(source)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_records_official
    ON prediction_records(ticker, target, model_name, is_official, feature_date)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_prediction_records_latest
    ON prediction_records(ticker, target, model_name, is_latest, feature_date)
    """,
    """
    CREATE UNIQUE INDEX IF NOT EXISTS idx_prediction_records_run_unique
    ON prediction_records(run_id, ticker, feature_date, target, model_name, source)
    """,
]

PREDICTION_RUN_COLUMNS = [
    "run_id",
    "run_timestamp",
    "ticker",
    "run_type",
    "command_name",
    "source",
    "working_directory",
    "data_freshness_status",
    "latest_raw_date",
    "latest_feature_date",
    "expected_latest_trading_date",
    "stale_at_run_time",
    "created_at",
    "notes",
]

PREDICTION_RECORD_COLUMNS = [
    "record_id",
    "run_id",
    "ticker",
    "feature_date",
    "target_close_date",
    "target",
    "model_name",
    "model_type",
    "predicted_label",
    "predicted_direction",
    "probability_up",
    "probability_down",
    "confidence",
    "source",
    "is_official",
    "is_latest",
    "actual_return",
    "actual_label",
    "actual_direction",
    "result",
    "created_at",
    "evaluated_at",
    "notes",
]

OFFICIAL_PREDICTION_SNAPSHOT_KEYS = {
    ("next_close_binary", "lightgbm_binary"),
    ("next_close_multiclass", "lightgbm_multiclass"),
}

PROBABILITY_COLUMNS = {
    "probability_surge",
    "probability_up",
    "probability_flat",
    "probability_down",
    "probability_crash",
}


def init_database(path: str | Path) -> Path:
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(PREDICTIONS_SCHEMA)
        _ensure_prediction_columns(connection)
        _ensure_append_only_schema(connection)
    return db_path


def upsert_predictions(predictions: pd.DataFrame, path: str | Path) -> int:
    db_path = init_database(path)
    required = {"date", "ticker", "target", "model_name", "predicted_class"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")

    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"]).dt.strftime("%Y-%m-%d")
    rows["predicted_class"] = rows["predicted_class"].map(_required_python_int)
    if "confidence" not in rows.columns:
        rows["confidence"] = None
    rows["confidence"] = rows["confidence"].map(_optional_python_float)
    for column in PREDICTION_EXTRA_COLUMNS:
        if column not in rows.columns:
            rows[column] = None
    for column in PROBABILITY_COLUMNS:
        rows[column] = rows[column].map(_optional_python_float)
    if "prediction_type" not in rows.columns:
        rows["prediction_type"] = "live"
    rows["prediction_type"] = rows["prediction_type"].fillna("live").astype(str)

    payload = [
        (
            row.date,
            row.ticker,
            row.target,
            row.model_name,
            row.predicted_class,
            row.confidence,
            row.probability_surge,
            row.probability_up,
            row.probability_flat,
            row.probability_down,
            row.probability_crash,
            row.top_features,
            row.prediction_type,
        )
        for row in rows[
            [
                "date",
                "ticker",
                "target",
                "model_name",
                "predicted_class",
                "confidence",
                "probability_surge",
                "probability_up",
                "probability_flat",
                "probability_down",
                "probability_crash",
                "top_features",
                "prediction_type",
            ]
        ].itertuples(index=False)
    ]

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO predictions
                (
                    date,
                    ticker,
                    target,
                    model_name,
                    predicted_class,
                    confidence,
                    probability_surge,
                    probability_up,
                    probability_flat,
                    probability_down,
                    probability_crash,
                    top_features,
                    prediction_type
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(date, ticker, target, model_name) DO UPDATE SET
                predicted_class = excluded.predicted_class,
                confidence = excluded.confidence,
                probability_surge = excluded.probability_surge,
                probability_up = excluded.probability_up,
                probability_flat = excluded.probability_flat,
                probability_down = excluded.probability_down,
                probability_crash = excluded.probability_crash,
                top_features = excluded.top_features,
                prediction_type = excluded.prediction_type,
                created_at = CURRENT_TIMESTAMP
            """,
            payload,
        )
    return len(rows)


def _required_python_int(value) -> int:
    result = safe_int(value)
    if result is None:
        raise ValueError("predicted_class cannot be converted to int.")
    return result


def _optional_python_float(value) -> float | None:
    return safe_float(value)


def read_latest_predictions(path: str | Path) -> pd.DataFrame:
    db_path = Path(path)
    if not db_path.exists():
        return pd.DataFrame()

    query = """
    SELECT p.*
    FROM predictions p
    INNER JOIN (
        SELECT ticker, target, model_name, MAX(date) AS max_date
        FROM predictions
        GROUP BY ticker, target, model_name
    ) latest
      ON p.ticker = latest.ticker
     AND p.target = latest.target
     AND p.model_name = latest.model_name
     AND p.date = latest.max_date
    ORDER BY p.date DESC, p.confidence DESC
    """
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(query, connection)


def read_prediction_history(
    path: str | Path,
    ticker: str | None = None,
    target: str | None = None,
    model_name: str | None = None,
    prediction_type: str | None = None,
) -> pd.DataFrame:
    db_path = Path(path)
    if not db_path.exists():
        return pd.DataFrame()

    query = """
    SELECT *
    FROM predictions
    ORDER BY date DESC, created_at DESC
    """
    with sqlite3.connect(db_path) as connection:
        rows = pd.read_sql_query(query, connection)

    if rows.empty:
        return rows
    if ticker is not None:
        rows = rows[rows["ticker"] == ticker]
    if target is not None:
        rows = rows[rows["target"] == target]
    if model_name is not None:
        rows = rows[rows["model_name"] == model_name]
    if prediction_type is not None and "prediction_type" in rows.columns:
        rows = rows[rows["prediction_type"] == prediction_type]
    return rows.reset_index(drop=True)


def create_prediction_run(
    path: str | Path,
    *,
    run_id: str | None = None,
    run_timestamp: str | pd.Timestamp | None = None,
    ticker: str,
    run_type: str,
    command_name: str | None = None,
    source: str | None = None,
    working_directory: str | None = None,
    data_freshness_status: str | None = None,
    latest_raw_date: str | pd.Timestamp | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    expected_latest_trading_date: str | pd.Timestamp | None = None,
    stale_at_run_time: bool = False,
    notes: str | None = None,
) -> str:
    db_path = init_database(path)
    final_run_id = run_id or _new_run_id()
    final_timestamp = _timestamp_text(run_timestamp)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO prediction_runs
                (
                    run_id,
                    run_timestamp,
                    ticker,
                    run_type,
                    command_name,
                    source,
                    working_directory,
                    data_freshness_status,
                    latest_raw_date,
                    latest_feature_date,
                    expected_latest_trading_date,
                    stale_at_run_time,
                    notes
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                final_run_id,
                final_timestamp,
                ticker,
                run_type,
                command_name,
                source,
                working_directory,
                data_freshness_status,
                _optional_date_text(latest_raw_date),
                _optional_date_text(latest_feature_date),
                _optional_date_text(expected_latest_trading_date),
                1 if stale_at_run_time else 0,
                notes,
            ),
        )
    return final_run_id


def get_or_create_prediction_run(
    path: str | Path,
    *,
    run_id: str | None = None,
    run_timestamp: str | pd.Timestamp | None = None,
    ticker: str,
    run_type: str,
    command_name: str | None = None,
    source: str | None = None,
    working_directory: str | None = None,
    data_freshness_status: str | None = None,
    latest_raw_date: str | pd.Timestamp | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    expected_latest_trading_date: str | pd.Timestamp | None = None,
    stale_at_run_time: bool = False,
    notes: str | None = None,
) -> str:
    if run_id is not None and get_prediction_run(path, run_id) is not None:
        return run_id
    return create_prediction_run(
        path,
        run_id=run_id,
        run_timestamp=run_timestamp,
        ticker=ticker,
        run_type=run_type,
        command_name=command_name,
        source=source,
        working_directory=working_directory,
        data_freshness_status=data_freshness_status,
        latest_raw_date=latest_raw_date,
        latest_feature_date=latest_feature_date,
        expected_latest_trading_date=expected_latest_trading_date,
        stale_at_run_time=stale_at_run_time,
        notes=notes,
    )


def update_prediction_run_notes(
    path: str | Path,
    *,
    run_id: str,
    notes: str,
) -> int:
    db_path = init_database(path)
    with sqlite3.connect(db_path) as connection:
        before = connection.total_changes
        connection.execute(
            "UPDATE prediction_runs SET notes = ? WHERE run_id = ?",
            (notes, run_id),
        )
        return int(connection.total_changes - before)


def append_prediction_record(
    path: str | Path,
    *,
    run_id: str,
    ticker: str,
    feature_date: str | pd.Timestamp,
    target: str,
    model_name: str,
    source: str,
    record_id: str | None = None,
    target_close_date: str | pd.Timestamp | None = None,
    model_type: str | None = None,
    predicted_label: object | None = None,
    predicted_direction: str | None = None,
    probability_up: object | None = None,
    probability_down: object | None = None,
    confidence: object | None = None,
    is_official: bool = False,
    is_latest: bool = False,
    actual_return: object | None = None,
    actual_label: object | None = None,
    actual_direction: str | None = None,
    result: str = "pending",
    evaluated_at: str | pd.Timestamp | None = None,
    notes: str | None = None,
) -> int:
    return append_prediction_records(
        path,
        pd.DataFrame(
            [
                {
                    "record_id": record_id or _new_record_id(),
                    "run_id": run_id,
                    "ticker": ticker,
                    "feature_date": _date_text(feature_date),
                    "target_close_date": _optional_date_text(target_close_date),
                    "target": target,
                    "model_name": model_name,
                    "model_type": model_type,
                    "predicted_label": None if predicted_label is None else str(predicted_label),
                    "predicted_direction": predicted_direction,
                    "probability_up": _optional_python_float(probability_up),
                    "probability_down": _optional_python_float(probability_down),
                    "confidence": _optional_python_float(confidence),
                    "source": source,
                    "is_official": 1 if is_official else 0,
                    "is_latest": 1 if is_latest else 0,
                    "actual_return": _optional_python_float(actual_return),
                    "actual_label": None if actual_label is None else str(actual_label),
                    "actual_direction": actual_direction,
                    "result": result or "pending",
                    "evaluated_at": _optional_timestamp_text(evaluated_at),
                    "notes": notes,
                }
            ]
        ),
    )


def append_prediction_records(path: str | Path, records: pd.DataFrame) -> int:
    db_path = init_database(path)
    if records.empty:
        return 0
    required = {"run_id", "ticker", "feature_date", "target", "model_name", "source"}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f"Missing append-only prediction record columns: {sorted(missing)}")

    rows = records.copy()
    for column in PREDICTION_RECORD_COLUMNS:
        if column not in rows.columns:
            rows[column] = None
    rows["record_id"] = rows["record_id"].map(lambda value: str(value or _new_record_id()))
    rows["feature_date"] = rows["feature_date"].map(_date_text)
    rows["target_close_date"] = rows["target_close_date"].map(_optional_date_text)
    rows["predicted_label"] = rows["predicted_label"].map(
        lambda value: None if pd.isna(value) else str(value)
    )
    rows["actual_label"] = rows["actual_label"].map(lambda value: None if pd.isna(value) else str(value))
    rows["probability_up"] = rows["probability_up"].map(_optional_python_float)
    rows["probability_down"] = rows["probability_down"].map(_optional_python_float)
    rows["confidence"] = rows["confidence"].map(_optional_python_float)
    rows["actual_return"] = rows["actual_return"].map(_optional_python_float)
    rows["is_official"] = rows["is_official"].map(_bool_to_int)
    rows["is_latest"] = rows["is_latest"].map(_bool_to_int)
    rows["result"] = rows["result"].fillna("pending").astype(str)
    rows["created_at"] = rows["created_at"].map(
        lambda value: _optional_timestamp_text(value) or _now_timestamp()
    )
    rows["evaluated_at"] = rows["evaluated_at"].map(_optional_timestamp_text)

    payload = [
        tuple(row)
        for row in rows[PREDICTION_RECORD_COLUMNS].itertuples(index=False, name=None)
    ]
    placeholders = ", ".join(["?"] * len(PREDICTION_RECORD_COLUMNS))
    columns = ", ".join(PREDICTION_RECORD_COLUMNS)
    with sqlite3.connect(db_path) as connection:
        before = connection.total_changes
        connection.executemany(
            f"""
            INSERT OR IGNORE INTO prediction_records
                ({columns})
            VALUES ({placeholders})
            """,
            payload,
        )
        return int(connection.total_changes - before)


def mark_latest_official_records(
    path: str | Path,
    *,
    ticker: str,
    target: str,
    model_name: str,
    record_ids: list[str],
) -> int:
    db_path = init_database(path)
    if not record_ids:
        return 0
    with sqlite3.connect(db_path) as connection:
        before = connection.total_changes
        connection.execute(
            """
            UPDATE prediction_records
            SET is_latest = 0
            WHERE ticker = ?
              AND target = ?
              AND model_name = ?
            """,
            (ticker, target, model_name),
        )
        placeholders = ", ".join(["?"] * len(record_ids))
        connection.execute(
            f"""
            UPDATE prediction_records
            SET is_latest = 1,
                is_official = 1
            WHERE record_id IN ({placeholders})
              AND ticker = ?
              AND target = ?
              AND model_name = ?
            """,
            [*record_ids, ticker, target, model_name],
        )
        return int(connection.total_changes - before)


def update_prediction_record_evaluation(
    path: str | Path,
    *,
    record_id: str,
    actual_return: object | None = None,
    actual_label: object | None = None,
    actual_direction: str | None = None,
    result: str,
    evaluated_at: str | pd.Timestamp | None = None,
) -> int:
    db_path = init_database(path)
    with sqlite3.connect(db_path) as connection:
        before = connection.total_changes
        connection.execute(
            """
            UPDATE prediction_records
            SET actual_return = ?,
                actual_label = ?,
                actual_direction = ?,
                result = ?,
                evaluated_at = ?
            WHERE record_id = ?
            """,
            (
                _optional_python_float(actual_return),
                None if actual_label is None else str(actual_label),
                actual_direction,
                result,
                _optional_timestamp_text(evaluated_at) or _now_timestamp(),
                record_id,
            ),
        )
        return int(connection.total_changes - before)


def evaluate_prediction_records(
    path: str | Path,
    actuals: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> int:
    if actuals.empty:
        return 0
    required = {"date", "actual_direction"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"Missing actual result columns: {sorted(missing)}")

    actual_rows = actuals.copy()
    actual_rows["date"] = actual_rows["date"].map(_date_text)
    actual_by_date = {row.date: row for row in actual_rows.itertuples(index=False)}

    pending = list_prediction_records(path, ticker=ticker)
    if pending.empty:
        return 0
    pending = pending[pending["result"].fillna("pending").astype(str) == "pending"]
    updated = 0
    for row in pending.itertuples(index=False):
        target_date = getattr(row, "target_close_date", None)
        if not target_date or pd.isna(target_date) or target_date not in actual_by_date:
            continue
        actual = actual_by_date[target_date]
        actual_direction = str(getattr(actual, "actual_direction"))
        actual_label = getattr(actual, "actual_label", actual_direction)
        actual_return = getattr(actual, "actual_return", None)
        predicted_direction = getattr(row, "predicted_direction", None)
        predicted_label = getattr(row, "predicted_label", None)
        if predicted_direction:
            result = "hit" if str(predicted_direction) == actual_direction else "miss"
        else:
            result = "hit" if str(predicted_label) == str(actual_label) else "miss"
        updated += update_prediction_record_evaluation(
            path,
            record_id=str(getattr(row, "record_id")),
            actual_return=actual_return,
            actual_label=actual_label,
            actual_direction=actual_direction,
            result=result,
        )
    return updated


def write_append_only_prediction_run(
    path: str | Path,
    predictions: pd.DataFrame,
    *,
    ticker: str,
    run_type: str,
    command_name: str,
    source: str = "live",
    run_id: str | None = None,
    run_timestamp: str | pd.Timestamp | None = None,
    working_directory: str | None = None,
    data_freshness_status: str | None = None,
    latest_raw_date: str | pd.Timestamp | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    expected_latest_trading_date: str | pd.Timestamp | None = None,
    stale_at_run_time: bool = False,
    is_official: bool = False,
    is_latest: bool = False,
    notes: str | None = None,
) -> dict[str, object]:
    before = summarize_append_only_records(path, ticker=ticker)
    final_run_id = get_or_create_prediction_run(
        path,
        run_id=run_id,
        run_timestamp=run_timestamp,
        ticker=ticker,
        run_type=run_type,
        command_name=command_name,
        source=source,
        working_directory=working_directory,
        data_freshness_status=data_freshness_status,
        latest_raw_date=latest_raw_date,
        latest_feature_date=latest_feature_date,
        expected_latest_trading_date=expected_latest_trading_date,
        stale_at_run_time=stale_at_run_time,
        notes=notes,
    )
    records = _append_only_records_from_snapshot(
        predictions,
        run_id=final_run_id,
        source=source,
        is_official=is_official,
        is_latest=is_latest,
        notes=notes,
    )
    inserted = append_prediction_records(path, records)
    after = summarize_append_only_records(path, ticker=ticker)
    return {
        "run_id": final_run_id,
        "record_candidates": int(len(records)),
        "inserted_records": inserted,
        "duplicate_records": int(len(records) - inserted),
        "before": before,
        "after": after,
    }


def filter_official_prediction_snapshot_records(
    predictions: pd.DataFrame,
    *,
    ticker: str | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    official_keys: set[tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        return _empty_filtered_snapshot(), _empty_filtered_snapshot()
    required = {"date", "ticker", "target", "model_name"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing snapshot filter columns: {sorted(missing)}")

    rows = predictions.copy()
    rows["date"] = rows["date"].map(_date_text)
    rows["snapshot_source"] = rows.apply(_snapshot_source, axis=1)
    rows["model_type"] = rows.apply(
        lambda row: _infer_model_type(str(row["target"]), str(row["model_name"])),
        axis=1,
    )
    active_keys = official_keys or OFFICIAL_PREDICTION_SNAPSHOT_KEYS
    expected_feature_date = _resolve_latest_snapshot_feature_date(
        rows,
        ticker=ticker,
        latest_feature_date=latest_feature_date,
    )

    statuses = []
    reasons = []
    excluded_reasons = []
    for row in rows.itertuples(index=False):
        reason = _official_snapshot_exclusion_reason(
            row,
            ticker=ticker,
            latest_feature_date=expected_feature_date,
            official_keys=active_keys,
        )
        if reason is None:
            statuses.append("included")
            reasons.append("official_production_current")
            excluded_reasons.append("")
        else:
            statuses.append("excluded")
            reasons.append("")
            excluded_reasons.append(reason)
    rows["official_filter_status"] = statuses
    rows["official_filter_reason"] = reasons
    rows["excluded_reason"] = excluded_reasons
    included = rows[rows["official_filter_status"] == "included"].reset_index(drop=True)
    excluded = rows[rows["official_filter_status"] == "excluded"].reset_index(drop=True)
    return included, excluded


def list_official_latest_prediction_snapshot_records(
    path: str | Path,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    snapshot = read_latest_predictions(path)
    included, _excluded = filter_official_prediction_snapshot_records(
        snapshot,
        ticker=ticker,
    )
    return included


def is_official_prediction_snapshot_record(
    row: pd.Series | dict[str, object],
    *,
    ticker: str | None = None,
    latest_feature_date: str | pd.Timestamp | None = None,
    official_keys: set[tuple[str, str]] | None = None,
) -> tuple[bool, str]:
    frame = pd.DataFrame([dict(row)])
    included, excluded = filter_official_prediction_snapshot_records(
        frame,
        ticker=ticker,
        latest_feature_date=latest_feature_date,
        official_keys=official_keys,
    )
    if not included.empty:
        return True, str(included.loc[0, "official_filter_reason"])
    if not excluded.empty:
        return False, str(excluded.loc[0, "excluded_reason"])
    return False, "empty_snapshot"


def get_append_only_candidate_records_from_snapshot(
    predictions: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    included, _excluded = filter_official_prediction_snapshot_records(
        predictions,
        ticker=ticker,
    )
    return included


def compare_append_only_official_to_snapshot(
    path: str | Path,
    *,
    ticker: str,
) -> dict[str, object]:
    snapshot = read_latest_predictions(path)
    snapshot = snapshot[snapshot["ticker"] == ticker].reset_index(drop=True) if not snapshot.empty else snapshot
    included, excluded = filter_official_prediction_snapshot_records(snapshot, ticker=ticker)
    append_rows = get_latest_official_prediction_records(path, ticker=ticker, source="live")
    run_rows = list_prediction_runs(path, ticker=ticker)
    run_timestamps = (
        dict(zip(run_rows["run_id"], run_rows["run_timestamp"], strict=True))
        if not run_rows.empty
        else {}
    )

    comparison_rows: list[dict[str, object]] = []
    for row in included.itertuples(index=False):
        matches = append_rows[
            (append_rows["ticker"] == getattr(row, "ticker"))
            & (append_rows["feature_date"] == getattr(row, "date"))
            & (append_rows["target"] == getattr(row, "target"))
            & (append_rows["model_name"] == getattr(row, "model_name"))
        ]
        if matches.empty:
            comparison_rows.append(
                {
                    "feature_date": getattr(row, "date"),
                    "target": getattr(row, "target"),
                    "model_name": getattr(row, "model_name"),
                    "snapshot_predicted_label": str(getattr(row, "predicted_class")),
                    "append_predicted_label": None,
                    "snapshot_confidence": _optional_python_float(getattr(row, "confidence", None)),
                    "append_confidence": None,
                    "append_run_id": None,
                    "append_run_timestamp": None,
                    "match": False,
                    "reason": "missing_append_only_official_record",
                }
            )
            continue
        append = matches.sort_values(
            ["feature_date", "created_at", "record_id"],
            ascending=[False, False, False],
        ).iloc[0]
        label_match = str(getattr(row, "predicted_class")) == str(append["predicted_label"])
        snapshot_confidence = _optional_python_float(getattr(row, "confidence", None))
        append_confidence = _optional_python_float(append["confidence"])
        confidence_match = snapshot_confidence == append_confidence
        row_match = bool(label_match and confidence_match)
        comparison_rows.append(
            {
                "feature_date": getattr(row, "date"),
                "target": getattr(row, "target"),
                "model_name": getattr(row, "model_name"),
                "snapshot_predicted_label": str(getattr(row, "predicted_class")),
                "append_predicted_label": append["predicted_label"],
                "snapshot_confidence": snapshot_confidence,
                "append_confidence": append_confidence,
                "append_run_id": append["run_id"],
                "append_run_timestamp": run_timestamps.get(append["run_id"]),
                "match": row_match,
                "reason": "match" if row_match else "value_mismatch",
            }
        )

    return {
        "snapshot_official_count": int(len(included)),
        "snapshot_excluded_count": int(len(excluded)),
        "append_official_count": int(len(append_rows)),
        "matches": bool(comparison_rows and all(row["match"] for row in comparison_rows)),
        "rows": comparison_rows,
    }


def list_prediction_runs(
    path: str | Path,
    ticker: str | None = None,
    run_type: str | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    db_path = Path(path)
    if not _has_append_only_tables(db_path):
        return _empty_prediction_runs()

    filters = []
    params: list[object] = []
    if ticker is not None:
        filters.append("ticker = ?")
        params.append(ticker)
    if run_type is not None:
        filters.append("run_type = ?")
        params.append(run_type)
    if source is not None:
        filters.append("source = ?")
        params.append(source)

    query = "SELECT * FROM prediction_runs"
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY run_timestamp DESC, created_at DESC, run_id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(db_path) as connection:
        rows = pd.read_sql_query(query, connection, params=params)
    return _ensure_columns(rows, PREDICTION_RUN_COLUMNS)


def get_prediction_run(path: str | Path, run_id: str) -> dict[str, object] | None:
    rows = list_prediction_runs(path)
    if rows.empty:
        return None
    matches = rows[rows["run_id"] == run_id]
    if matches.empty:
        return None
    return matches.iloc[0].to_dict()


def list_prediction_records(
    path: str | Path,
    ticker: str | None = None,
    feature_date: str | pd.Timestamp | None = None,
    target: str | None = None,
    model_name: str | None = None,
    source: str | None = None,
    is_official: bool | None = None,
    run_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    db_path = Path(path)
    if not _has_append_only_tables(db_path):
        return _empty_prediction_records()

    filters = []
    params: list[object] = []
    if ticker is not None:
        filters.append("ticker = ?")
        params.append(ticker)
    if feature_date is not None:
        filters.append("feature_date = ?")
        params.append(_date_text(feature_date))
    if target is not None:
        filters.append("target = ?")
        params.append(target)
    if model_name is not None:
        filters.append("model_name = ?")
        params.append(model_name)
    if source is not None:
        filters.append("source = ?")
        params.append(source)
    if is_official is not None:
        filters.append("is_official = ?")
        params.append(1 if is_official else 0)
    if run_id is not None:
        filters.append("run_id = ?")
        params.append(run_id)

    query = "SELECT * FROM prediction_records"
    if filters:
        query += " WHERE " + " AND ".join(filters)
    query += " ORDER BY feature_date DESC, created_at DESC, record_id DESC"
    if limit is not None:
        query += " LIMIT ?"
        params.append(int(limit))

    with sqlite3.connect(db_path) as connection:
        rows = pd.read_sql_query(query, connection, params=params)
    return _ensure_columns(rows, PREDICTION_RECORD_COLUMNS)


def get_latest_official_prediction_records(
    path: str | Path,
    ticker: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    rows = list_prediction_records(
        path,
        ticker=ticker,
        source=source,
        is_official=True,
    )
    if rows.empty:
        return rows
    rows = rows.sort_values(
        ["ticker", "target", "model_name", "feature_date", "created_at", "record_id"],
        ascending=[True, True, True, False, False, False],
    )
    latest = rows.drop_duplicates(["ticker", "target", "model_name"], keep="first")
    return latest.reset_index(drop=True)


def get_records_for_feature_date(
    path: str | Path,
    feature_date: str | pd.Timestamp,
    ticker: str | None = None,
    source: str | None = None,
) -> pd.DataFrame:
    return list_prediction_records(
        path,
        ticker=ticker,
        feature_date=feature_date,
        source=source,
    )


def get_run_history_for_feature_date(
    path: str | Path,
    feature_date: str | pd.Timestamp,
    ticker: str | None = None,
) -> pd.DataFrame:
    db_path = Path(path)
    if not _has_append_only_tables(db_path):
        columns = [f"record_{column}" for column in PREDICTION_RECORD_COLUMNS]
        columns += [f"run_{column}" for column in PREDICTION_RUN_COLUMNS]
        return pd.DataFrame(columns=columns)

    params: list[object] = [_date_text(feature_date)]
    filters = ["r.feature_date = ?"]
    if ticker is not None:
        filters.append("r.ticker = ?")
        params.append(ticker)
    record_columns = ", ".join(
        f"r.{column} AS record_{column}" for column in PREDICTION_RECORD_COLUMNS
    )
    run_columns = ", ".join(f"pr.{column} AS run_{column}" for column in PREDICTION_RUN_COLUMNS)
    query = f"""
    SELECT {record_columns}, {run_columns}
    FROM prediction_records r
    LEFT JOIN prediction_runs pr
      ON r.run_id = pr.run_id
    WHERE {" AND ".join(filters)}
    ORDER BY pr.run_timestamp DESC, r.created_at DESC, r.record_id DESC
    """
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(query, connection, params=params)


def get_latest_live_record_by_feature_date(
    path: str | Path,
    feature_date: str | pd.Timestamp,
    ticker: str | None = None,
    target: str | None = None,
    model_name: str | None = None,
) -> pd.DataFrame:
    rows = list_prediction_records(
        path,
        ticker=ticker,
        feature_date=feature_date,
        target=target,
        model_name=model_name,
        source="live",
    )
    if rows.empty:
        return rows
    rows = rows.sort_values(
        ["ticker", "target", "model_name", "created_at", "record_id"],
        ascending=[True, True, True, False, False],
    )
    latest = rows.drop_duplicates(["ticker", "target", "model_name"], keep="first")
    return latest.reset_index(drop=True)


def summarize_append_only_records(
    path: str | Path,
    ticker: str | None = None,
) -> dict[str, object]:
    db_path = Path(path)
    summary: dict[str, object] = {
        "database_path": str(db_path),
        "database_exists": db_path.exists(),
        "append_only_tables_exist": False,
        "prediction_runs_count": 0,
        "prediction_records_count": 0,
        "existing_predictions_count": 0,
        "official_records_count": 0,
        "latest_records_count": 0,
        "source_counts": {},
        "run_type_counts": {},
    }
    if not db_path.exists():
        return summary

    with sqlite3.connect(db_path) as connection:
        summary["existing_predictions_count"] = _count_rows(
            connection,
            "predictions",
            ticker=ticker,
        )
        has_runs = _table_exists(connection, "prediction_runs")
        has_records = _table_exists(connection, "prediction_records")
        summary["append_only_tables_exist"] = has_runs and has_records
        if not has_runs or not has_records:
            return summary
        summary["prediction_runs_count"] = _count_rows(
            connection,
            "prediction_runs",
            ticker=ticker,
        )
        summary["prediction_records_count"] = _count_rows(
            connection,
            "prediction_records",
            ticker=ticker,
        )
        summary["official_records_count"] = _count_rows(
            connection,
            "prediction_records",
            ticker=ticker,
            extra_filter="is_official = 1",
        )
        summary["latest_records_count"] = _count_rows(
            connection,
            "prediction_records",
            ticker=ticker,
            extra_filter="is_latest = 1",
        )
        summary["source_counts"] = _group_counts(
            connection,
            "prediction_records",
            "source",
            ticker=ticker,
        )
        summary["run_type_counts"] = _group_counts(
            connection,
            "prediction_runs",
            "run_type",
            ticker=ticker,
        )
    return summary


def _ensure_prediction_columns(connection: sqlite3.Connection) -> None:
    existing = {
        row[1]
        for row in connection.execute("PRAGMA table_info(predictions)").fetchall()
    }
    for column, column_type in PREDICTION_EXTRA_COLUMNS.items():
        if column not in existing:
            connection.execute(f"ALTER TABLE predictions ADD COLUMN {column} {column_type}")


def _ensure_append_only_schema(connection: sqlite3.Connection) -> None:
    connection.execute(PREDICTION_RUNS_SCHEMA)
    connection.execute(PREDICTION_RECORDS_SCHEMA)
    for index_sql in PREDICTION_APPEND_ONLY_INDEXES:
        connection.execute(index_sql)


def _has_append_only_tables(db_path: Path) -> bool:
    if not db_path.exists():
        return False
    with sqlite3.connect(db_path) as connection:
        return _table_exists(connection, "prediction_runs") and _table_exists(
            connection,
            "prediction_records",
        )


def _table_exists(connection: sqlite3.Connection, table_name: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _empty_prediction_runs() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_RUN_COLUMNS)


def _empty_prediction_records() -> pd.DataFrame:
    return pd.DataFrame(columns=PREDICTION_RECORD_COLUMNS)


def _ensure_columns(rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in rows.columns:
            rows[column] = pd.Series(dtype="object")
    return rows[columns].reset_index(drop=True)


def _date_text(value: str | pd.Timestamp) -> str:
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _optional_date_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return _date_text(value)


def _now_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _timestamp_text(value: str | pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return _now_timestamp()
    return pd.to_datetime(value).isoformat()


def _optional_timestamp_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).isoformat()


def _new_run_id() -> str:
    return f"run_{uuid4().hex}"


def _new_record_id() -> str:
    return f"record_{uuid4().hex}"


def _bool_to_int(value: object) -> int:
    if value is None or pd.isna(value):
        return 0
    if isinstance(value, str):
        return 1 if value.strip().lower() in {"1", "true", "yes", "y"} else 0
    return 1 if bool(value) else 0


def _append_only_records_from_snapshot(
    predictions: pd.DataFrame,
    *,
    run_id: str,
    source: str,
    is_official: bool,
    is_latest: bool,
    notes: str | None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=PREDICTION_RECORD_COLUMNS)
    required = {"date", "ticker", "target", "model_name", "predicted_class"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing snapshot prediction columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for row in predictions.itertuples(index=False):
        feature_date = _date_text(getattr(row, "date"))
        target = str(getattr(row, "target"))
        model_name = str(getattr(row, "model_name"))
        predicted_class = getattr(row, "predicted_class")
        model_type = _infer_model_type(target, model_name)
        rows.append(
            {
                "record_id": _new_record_id(),
                "run_id": run_id,
                "ticker": str(getattr(row, "ticker")),
                "feature_date": feature_date,
                "target_close_date": _next_business_date_text(feature_date),
                "target": target,
                "model_name": model_name,
                "model_type": model_type,
                "predicted_label": str(predicted_class),
                "predicted_direction": _predicted_direction(predicted_class, model_type),
                "probability_up": _optional_python_float(getattr(row, "probability_up", None)),
                "probability_down": _optional_python_float(getattr(row, "probability_down", None)),
                "confidence": _optional_python_float(getattr(row, "confidence", None)),
                "source": source,
                "is_official": 1 if is_official else 0,
                "is_latest": 1 if is_latest else 0,
                "actual_return": None,
                "actual_label": None,
                "actual_direction": None,
                "result": "pending",
                "created_at": _now_timestamp(),
                "evaluated_at": None,
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=PREDICTION_RECORD_COLUMNS)


def _empty_filtered_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "target",
            "model_name",
            "predicted_class",
            "confidence",
            "prediction_type",
            "created_at",
            "snapshot_source",
            "model_type",
            "official_filter_status",
            "official_filter_reason",
            "excluded_reason",
        ]
    )


def _snapshot_source(row: pd.Series) -> str:
    for column in ("prediction_type", "source"):
        if column in row.index:
            value = row.get(column)
            if value is not None and not pd.isna(value):
                text = str(value).strip().lower()
                if text:
                    return text
    return "unknown"


def _resolve_latest_snapshot_feature_date(
    rows: pd.DataFrame,
    *,
    ticker: str | None,
    latest_feature_date: str | pd.Timestamp | None,
) -> str | None:
    if latest_feature_date is not None:
        return _date_text(latest_feature_date)
    candidates = rows.copy()
    if ticker is not None:
        candidates = candidates[candidates["ticker"] == ticker]
    live = candidates[candidates["snapshot_source"] == "live"]
    if not live.empty:
        candidates = live
    if candidates.empty:
        return None
    return str(candidates["date"].max())


def _official_snapshot_exclusion_reason(
    row,
    *,
    ticker: str | None,
    latest_feature_date: str | None,
    official_keys: set[tuple[str, str]],
) -> str | None:
    if ticker is not None and str(getattr(row, "ticker")) != ticker:
        return "ticker_mismatch"
    source = str(getattr(row, "snapshot_source", "") or "").strip().lower()
    if source != "live":
        if source in {"backfill", "shadow", "experiment"}:
            return f"non_live_source_{source}"
        return "source_not_live"
    row_date = _date_text(getattr(row, "date"))
    if latest_feature_date is not None and row_date != latest_feature_date:
        return "stale_feature_date"
    target = str(getattr(row, "target"))
    model_name = str(getattr(row, "model_name"))
    if (target, model_name) not in official_keys:
        return "non_official_target_model"
    return None


def _infer_model_type(target: str, model_name: str) -> str:
    text = f"{target} {model_name}".lower()
    if "multiclass" in text:
        return "multiclass"
    if "binary" in text:
        return "binary"
    return "unknown"


def _predicted_direction(predicted_class: object, model_type: str | None) -> str | None:
    label = safe_int(predicted_class)
    if label is None:
        return None
    if model_type == "binary":
        return "up" if label == 1 else "down"
    if model_type == "multiclass":
        return {
            0: "crash",
            1: "down",
            2: "flat",
            3: "up",
            4: "surge",
        }.get(label)
    return None


def _next_business_date_text(feature_date: str) -> str:
    return (pd.to_datetime(feature_date) + pd.offsets.BDay(1)).strftime("%Y-%m-%d")


def _count_rows(
    connection: sqlite3.Connection,
    table_name: str,
    ticker: str | None = None,
    extra_filter: str | None = None,
) -> int:
    if not _table_exists(connection, table_name):
        return 0
    filters = []
    params: list[object] = []
    if ticker is not None:
        filters.append("ticker = ?")
        params.append(ticker)
    if extra_filter is not None:
        filters.append(extra_filter)
    query = f"SELECT COUNT(*) FROM {table_name}"
    if filters:
        query += " WHERE " + " AND ".join(filters)
    row = connection.execute(query, params).fetchone()
    return int(row[0]) if row is not None else 0


def _group_counts(
    connection: sqlite3.Connection,
    table_name: str,
    column_name: str,
    ticker: str | None = None,
) -> dict[str, int]:
    if not _table_exists(connection, table_name):
        return {}
    params: list[object] = []
    query = f"SELECT {column_name}, COUNT(*) FROM {table_name}"
    if ticker is not None:
        query += " WHERE ticker = ?"
        params.append(ticker)
    query += f" GROUP BY {column_name} ORDER BY COUNT(*) DESC, {column_name}"
    return {
        str(row[0]): int(row[1])
        for row in connection.execute(query, params).fetchall()
        if row[0] is not None
    }
