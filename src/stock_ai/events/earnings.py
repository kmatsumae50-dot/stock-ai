from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.utils.safe_cast import safe_float


EVENTS_DATA_SUBDIR = "events"
EVENT_STATE_NORMAL = "normal"
EVENT_STATE_PRE = "pre_earnings_week"
EVENT_STATE_DAY = "earnings_day"
EVENT_STATE_POST = "post_earnings"
EARNINGS_FEATURE_COLUMNS = (
    "days_to_earnings",
    "days_after_earnings",
    "earnings_week_flag",
    "earnings_day_flag",
    "post_earnings_mode",
    "earnings_event_state",
    "earnings_event_state_code",
    "earnings_pre_volatility",
    "earnings_post_gap",
    "earnings_volume_spike",
    "earnings_after_trend_5",
)
EARNINGS_DATE_COLUMNS = ("date", "earnings_date")
EARNINGS_TEMPLATE_COLUMNS = (
    "date",
    "ticker",
    "event_type",
    "fiscal_year",
    "quarter",
    "period",
    "source",
    "note",
)


@dataclass(frozen=True)
class EarningsEventContext:
    event_state: str
    next_earnings_date: str | None
    days_to_earnings: int | None
    days_after_earnings: int | None
    earnings_week_flag: bool
    earnings_day_flag: bool
    post_earnings_mode: bool
    summary: str
    metrics: dict[str, float | None]


def add_earnings_event_features(
    data: pd.DataFrame,
    events_dir: str | Path,
    ticker: str,
) -> pd.DataFrame:
    """Add manual earnings-calendar features without using future price outcomes."""
    result = data.copy()
    if result.empty or "date" not in result.columns:
        return result

    result = result.drop(columns=[c for c in EARNINGS_FEATURE_COLUMNS if c in result.columns])
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values("date").reset_index(drop=True)

    events = read_earnings_events(events_dir, ticker)
    if events.empty:
        return _add_empty_event_features(result)

    event_dates = pd.to_datetime(events["date"], errors="coerce").dropna().sort_values()
    if event_dates.empty:
        return _add_empty_event_features(result)

    dates = pd.to_datetime(result["date"], errors="coerce")
    next_dates: list[pd.Timestamp | pd.NaT] = []
    previous_dates: list[pd.Timestamp | pd.NaT] = []
    for current_date in dates:
        future = event_dates[event_dates >= current_date]
        past = event_dates[event_dates <= current_date]
        next_dates.append(future.iloc[0] if not future.empty else pd.NaT)
        previous_dates.append(past.iloc[-1] if not past.empty else pd.NaT)

    next_series = pd.Series(next_dates, index=result.index)
    previous_series = pd.Series(previous_dates, index=result.index)
    result["days_to_earnings"] = (next_series - dates).dt.days
    result["days_after_earnings"] = (dates - previous_series).dt.days

    result["earnings_day_flag"] = (result["days_to_earnings"] == 0).astype(int)
    result["earnings_week_flag"] = (
        result["days_to_earnings"].between(0, 7, inclusive="both")
        | result["days_after_earnings"].between(0, 7, inclusive="both")
    ).astype(int)
    result["post_earnings_mode"] = result["days_after_earnings"].between(
        0,
        5,
        inclusive="both",
    ).astype(int)
    result["earnings_event_state"] = _event_state_series(result)
    result["earnings_event_state_code"] = result["earnings_event_state"].map(_event_state_code)

    result["earnings_pre_volatility"] = _pre_earnings_volatility(result)
    result["earnings_post_gap"] = _post_earnings_gap(result, event_dates)
    result["earnings_volume_spike"] = _earnings_volume_spike(result)
    result["earnings_after_trend_5"] = _earnings_after_trend(result, event_dates)

    return result.replace([float("inf"), -float("inf")], pd.NA)


def build_earnings_event_context(data: pd.DataFrame) -> EarningsEventContext:
    if data.empty:
        return _empty_context("Earnings event data is not available yet.")

    latest = data.sort_values("date").iloc[-1] if "date" in data.columns else data.iloc[-1]
    state = str(latest.get("earnings_event_state") or EVENT_STATE_NORMAL)
    metrics = {
        "days_to_earnings": _value(latest, "days_to_earnings"),
        "days_after_earnings": _value(latest, "days_after_earnings"),
        "earnings_pre_volatility": _value(latest, "earnings_pre_volatility"),
        "earnings_post_gap": _value(latest, "earnings_post_gap"),
        "earnings_volume_spike": _value(latest, "earnings_volume_spike"),
        "earnings_after_trend_5": _value(latest, "earnings_after_trend_5"),
    }
    days_to = _safe_int(metrics["days_to_earnings"])
    days_after = _safe_int(metrics["days_after_earnings"])
    next_date = _next_earnings_date(data)
    earnings_day = bool(safe_float(latest.get("earnings_day_flag")) == 1)
    earnings_week = bool(safe_float(latest.get("earnings_week_flag")) == 1)
    post_mode = bool(safe_float(latest.get("post_earnings_mode")) == 1)

    return EarningsEventContext(
        event_state=state,
        next_earnings_date=next_date,
        days_to_earnings=days_to,
        days_after_earnings=days_after,
        earnings_week_flag=earnings_week,
        earnings_day_flag=earnings_day,
        post_earnings_mode=post_mode,
        summary=_summary_text(state, days_to, days_after, metrics),
        metrics=metrics,
    )


def read_earnings_events(events_dir: str | Path, ticker: str) -> pd.DataFrame:
    for path in earnings_event_path_candidates(events_dir, ticker):
        if not path.exists():
            continue
        events = _read_earnings_csv(path)
        if not events.empty:
            return events
    return pd.DataFrame(columns=["date", "event_type", "period", "note"])


def earnings_events_path(events_dir: str | Path, ticker: str) -> Path:
    return Path(events_dir) / f"earnings_{ticker}.csv"


def earnings_event_path_candidates(events_dir: str | Path, ticker: str) -> list[Path]:
    events_path = Path(events_dir)
    candidates = [earnings_events_path(events_path, ticker)]
    if ticker == "7203.T":
        candidates.append(events_path / "toyota_earnings.csv")
    candidates.append(events_path / f"{ticker}_earnings.csv")
    return _dedupe_paths(candidates)


def default_events_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / EVENTS_DATA_SUBDIR


def summarize_earnings_event_files(
    events_dir: str | Path,
    ticker: str,
    as_of_date: str | pd.Timestamp | None = None,
) -> dict[str, object]:
    """Summarize manual earnings CSV readiness without mutating feature data."""
    candidates = earnings_event_path_candidates(events_dir, ticker)
    active_path = _active_earnings_events_path(candidates)
    file_rows = [
        validate_earnings_csv(
            path,
            ticker,
            as_of_date=as_of_date,
            build_features_candidate=True,
            build_features_active=active_path == path,
        )
        for path in candidates
    ]
    existing_rows = [row for row in file_rows if row["exists"]]
    populated_rows = [row for row in file_rows if row["status"] == "populated"]
    invalid_rows = [
        row for row in file_rows if row["status"] in {"invalid", "needs_review"}
    ]
    header_only_rows = [row for row in file_rows if row["status"] == "header_only"]
    parseable_dates = [
        date
        for row in file_rows
        for date in (
            row.get("earliest_event_date"),
            row.get("latest_event_date"),
            row.get("next_known_event_date"),
        )
        if date
    ]

    if invalid_rows:
        status = "needs_review"
        message = "One or more earnings CSV files need review before event features can be trusted."
    elif populated_rows:
        status = "populated"
        message = "Earnings event dates are available for feature generation."
    elif header_only_rows:
        status = "header_only"
        message = (
            "Earnings CSV files exist but contain no event rows. "
            "Event-distance features are not applicable for the current period."
        )
    elif existing_rows:
        status = "empty"
        message = "Earnings CSV files exist but do not contain usable event rows."
    else:
        status = "missing"
        message = "No earnings event CSV file was found."

    latest_dates = [
        str(row["latest_event_date"]) for row in file_rows if row.get("latest_event_date")
    ]
    earliest_dates = [
        str(row["earliest_event_date"]) for row in file_rows if row.get("earliest_event_date")
    ]
    next_dates = [
        str(row["next_known_event_date"])
        for row in file_rows
        if row.get("next_known_event_date")
    ]

    return {
        "status": status,
        "message": message,
        "events_dir": str(Path(events_dir)),
        "active_path": str(active_path) if active_path is not None else None,
        "candidate_count": len(candidates),
        "rows": int(sum(safe_float(row.get("rows")) or 0 for row in file_rows)),
        "files": file_rows,
        "earliest_event_date": min(earliest_dates) if earliest_dates else None,
        "latest_event_date": max(latest_dates) if latest_dates else None,
        "next_known_event_date": min(next_dates) if next_dates else None,
        "source_column_available": any(
            bool(row.get("source_column_available")) for row in file_rows
        ),
        "event_type_column_available": any(
            bool(row.get("event_type_column_available")) for row in file_rows
        ),
        "future_event_count": int(
            sum(safe_float(row.get("future_event_count")) or 0 for row in file_rows)
        ),
        "invalid_date_count": int(
            sum(safe_float(row.get("invalid_date_count")) or 0 for row in file_rows)
        ),
        "duplicate_event_date_count": int(
            sum(
                safe_float(row.get("duplicate_event_date_count")) or 0
                for row in file_rows
            )
        ),
        "ticker_mismatch_count": int(
            sum(safe_float(row.get("ticker_mismatch_count")) or 0 for row in file_rows)
        ),
        "template_columns": list(EARNINGS_TEMPLATE_COLUMNS),
        "has_parseable_event_dates": bool(parseable_dates),
    }


def validate_earnings_csv(
    path: str | Path,
    ticker: str,
    as_of_date: str | pd.Timestamp | None = None,
    *,
    build_features_candidate: bool = False,
    build_features_active: bool = False,
) -> dict[str, object]:
    """Validate an earnings CSV for display and operations checks."""
    csv_path = Path(path)
    if not csv_path.exists():
        return {
            "path": str(csv_path),
            "exists": False,
            "status": "missing",
            "rows": 0,
            "columns": [],
            "earliest_event_date": None,
            "latest_event_date": None,
            "next_known_event_date": None,
            "required_columns_present": False,
            "missing_required_columns": ["date or earnings_date"],
            "invalid_date_count": None,
            "duplicate_event_date_count": None,
            "future_event_count": None,
            "ticker_mismatch_count": None,
            "source_column_available": False,
            "event_type_column_available": False,
            "build_features_candidate": build_features_candidate,
            "build_features_active": build_features_active,
            "message": "File is missing.",
        }

    try:
        events = pd.read_csv(csv_path)
    except pd.errors.EmptyDataError:
        return {
            "path": str(csv_path),
            "exists": True,
            "status": "empty",
            "rows": 0,
            "columns": [],
            "earliest_event_date": None,
            "latest_event_date": None,
            "next_known_event_date": None,
            "required_columns_present": False,
            "missing_required_columns": ["date or earnings_date"],
            "invalid_date_count": None,
            "duplicate_event_date_count": None,
            "future_event_count": None,
            "ticker_mismatch_count": None,
            "source_column_available": False,
            "event_type_column_available": False,
            "build_features_candidate": build_features_candidate,
            "build_features_active": build_features_active,
            "message": "File exists but is empty.",
        }

    columns = [str(column) for column in events.columns]
    rows = int(len(events))
    date_column = "date" if "date" in events.columns else "earnings_date"
    required_columns_present = date_column in events.columns
    missing_required_columns = [] if required_columns_present else ["date or earnings_date"]
    source_column_available = "source" in events.columns
    event_type_column_available = "event_type" in events.columns

    if rows == 0:
        return {
            "path": str(csv_path),
            "exists": True,
            "status": "header_only",
            "rows": 0,
            "columns": columns,
            "earliest_event_date": None,
            "latest_event_date": None,
            "next_known_event_date": None,
            "required_columns_present": required_columns_present,
            "missing_required_columns": missing_required_columns,
            "invalid_date_count": 0 if required_columns_present else None,
            "duplicate_event_date_count": 0 if required_columns_present else None,
            "future_event_count": 0 if required_columns_present else None,
            "ticker_mismatch_count": 0 if "ticker" in events.columns else None,
            "source_column_available": source_column_available,
            "event_type_column_available": event_type_column_available,
            "build_features_candidate": build_features_candidate,
            "build_features_active": build_features_active,
            "message": "File has headers but no event rows.",
        }

    if not required_columns_present:
        return {
            "path": str(csv_path),
            "exists": True,
            "status": "invalid",
            "rows": rows,
            "columns": columns,
            "earliest_event_date": None,
            "latest_event_date": None,
            "next_known_event_date": None,
            "required_columns_present": False,
            "missing_required_columns": missing_required_columns,
            "invalid_date_count": None,
            "duplicate_event_date_count": None,
            "future_event_count": None,
            "ticker_mismatch_count": _ticker_mismatch_count(events, ticker),
            "source_column_available": source_column_available,
            "event_type_column_available": event_type_column_available,
            "build_features_candidate": build_features_candidate,
            "build_features_active": build_features_active,
            "message": "Missing required date column. Use date or earnings_date.",
        }

    parsed_dates = pd.to_datetime(events[date_column], errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())
    valid_dates = parsed_dates.dropna()
    duplicate_count = int(valid_dates.dt.normalize().duplicated().sum())
    as_of = _as_timestamp(as_of_date)
    future_event_count = int((valid_dates >= as_of).sum()) if as_of is not None else 0
    future_dates = valid_dates[valid_dates >= as_of] if as_of is not None else pd.Series(dtype="datetime64[ns]")
    ticker_mismatch_count = _ticker_mismatch_count(events, ticker)

    status = "populated"
    message = "Earnings event rows are available."
    if invalid_date_count:
        status = "invalid"
        message = "One or more earnings event dates could not be parsed."
    elif duplicate_count:
        status = "needs_review"
        message = "Duplicate earnings event dates were found."
    elif ticker_mismatch_count not in {None, 0}:
        status = "needs_review"
        message = "One or more event rows have a different ticker."

    return {
        "path": str(csv_path),
        "exists": True,
        "status": status,
        "rows": rows,
        "columns": columns,
        "earliest_event_date": _date_string(valid_dates.min()),
        "latest_event_date": _date_string(valid_dates.max()),
        "next_known_event_date": _date_string(future_dates.min()),
        "required_columns_present": required_columns_present,
        "missing_required_columns": missing_required_columns,
        "invalid_date_count": invalid_date_count,
        "duplicate_event_date_count": duplicate_count,
        "future_event_count": future_event_count,
        "ticker_mismatch_count": ticker_mismatch_count,
        "source_column_available": source_column_available,
        "event_type_column_available": event_type_column_available,
        "build_features_candidate": build_features_candidate,
        "build_features_active": build_features_active,
        "message": message,
    }


def _read_earnings_csv(path: Path) -> pd.DataFrame:
    try:
        events = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=["date", "event_type", "period", "note"])

    if events.empty:
        return pd.DataFrame(columns=["date", "event_type", "period", "note"])

    date_column = "date" if "date" in events.columns else "earnings_date"
    if date_column not in events.columns:
        raise ValueError(f"Missing date column in earnings event CSV: {path}")

    result = events.rename(columns={date_column: "date"}).copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "event_type" not in result.columns:
        result["event_type"] = "earnings"
    return result


def _active_earnings_events_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if not path.exists():
            continue
        try:
            events = _read_earnings_csv(path)
        except ValueError:
            return path
        if not events.empty:
            return path
    return None


def _ticker_mismatch_count(events: pd.DataFrame, ticker: str) -> int | None:
    if "ticker" not in events.columns:
        return None
    tickers = events["ticker"].astype("string").str.strip()
    valid = tickers.dropna()
    if valid.empty:
        return 0
    return int((valid != ticker).sum())


def _as_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _date_string(value: object) -> str | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).strftime("%Y-%m-%d")


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def _add_empty_event_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result["days_to_earnings"] = pd.NA
    result["days_after_earnings"] = pd.NA
    result["earnings_week_flag"] = 0
    result["earnings_day_flag"] = 0
    result["post_earnings_mode"] = 0
    result["earnings_event_state"] = EVENT_STATE_NORMAL
    result["earnings_event_state_code"] = 0
    result["earnings_pre_volatility"] = 0.0
    result["earnings_post_gap"] = 0.0
    result["earnings_volume_spike"] = 0.0
    result["earnings_after_trend_5"] = 0.0
    return result


def _event_state_series(data: pd.DataFrame) -> pd.Series:
    states = pd.Series(EVENT_STATE_NORMAL, index=data.index, dtype="object")
    states[data["days_to_earnings"].between(1, 7, inclusive="both")] = EVENT_STATE_PRE
    states[data["days_after_earnings"].between(1, 5, inclusive="both")] = EVENT_STATE_POST
    states[data["earnings_day_flag"] == 1] = EVENT_STATE_DAY
    return states


def _event_state_code(state: str) -> int:
    return {
        EVENT_STATE_NORMAL: 0,
        EVENT_STATE_PRE: -1,
        EVENT_STATE_DAY: 1,
        EVENT_STATE_POST: 2,
    }.get(state, 0)


def _pre_earnings_volatility(data: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(data.get("return_1d"), errors="coerce")
    volatility = returns.rolling(window=10, min_periods=5).std()
    mask = data["days_to_earnings"].between(0, 10, inclusive="both")
    return volatility.where(mask, 0.0).fillna(0.0)


def _post_earnings_gap(data: pd.DataFrame, event_dates: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=data.index)
    event_indices = _aligned_event_indices(data, event_dates)
    previous_close = pd.to_numeric(data["close"], errors="coerce").shift(1)
    open_price = pd.to_numeric(data.get("open"), errors="coerce")
    for event_index in event_indices:
        if event_index <= 0 or event_index >= len(data):
            continue
        base = previous_close.iloc[event_index]
        gap = None if pd.isna(base) or base == 0 else open_price.iloc[event_index] / base - 1
        if gap is None or pd.isna(gap):
            continue
        end_index = min(event_index + 5, len(data) - 1)
        result.iloc[event_index : end_index + 1] = float(gap)
    return result


def _earnings_volume_spike(data: pd.DataFrame) -> pd.Series:
    if "volume" not in data.columns:
        return pd.Series(0.0, index=data.index)
    volume = pd.to_numeric(data["volume"], errors="coerce")
    average = volume.rolling(window=20, min_periods=5).mean()
    spike = volume / average - 1
    mask = (
        data["days_to_earnings"].between(0, 7, inclusive="both")
        | data["days_after_earnings"].between(0, 5, inclusive="both")
    )
    return spike.where(mask, 0.0).fillna(0.0)


def _earnings_after_trend(data: pd.DataFrame, event_dates: pd.Series) -> pd.Series:
    result = pd.Series(0.0, index=data.index)
    close = pd.to_numeric(data["close"], errors="coerce")
    for event_index in _aligned_event_indices(data, event_dates):
        if event_index >= len(data):
            continue
        event_close = close.iloc[event_index]
        if pd.isna(event_close) or event_close == 0:
            continue
        end_index = min(event_index + 5, len(data) - 1)
        result.iloc[event_index : end_index + 1] = close.iloc[event_index : end_index + 1] / event_close - 1
    return result.fillna(0.0)


def _aligned_event_indices(data: pd.DataFrame, event_dates: pd.Series) -> list[int]:
    dates = pd.to_datetime(data["date"], errors="coerce").reset_index(drop=True)
    valid_dates = dates.dropna()
    indices: list[int] = []
    for event_date in event_dates:
        position = valid_dates.searchsorted(event_date, side="left")
        if position < len(valid_dates):
            indices.append(int(valid_dates.index[position]))
    return sorted(set(indices))


def _next_earnings_date(data: pd.DataFrame) -> str | None:
    if "days_to_earnings" not in data.columns or "date" not in data.columns:
        return None
    latest = data.sort_values("date").iloc[-1]
    days_to = _safe_int(safe_float(latest.get("days_to_earnings")))
    if days_to is None:
        return None
    date = pd.to_datetime(latest.get("date"), errors="coerce")
    if pd.isna(date):
        return None
    return (date + pd.Timedelta(days=days_to)).strftime("%Y-%m-%d")


def _summary_text(
    state: str,
    days_to: int | None,
    days_after: int | None,
    metrics: dict[str, float | None],
) -> str:
    if state == EVENT_STATE_DAY:
        return "Today is treated as an earnings event day, so gap, volume, and volatility signals need extra care."
    if state == EVENT_STATE_PRE:
        base = "Earnings are approaching"
        if days_to is not None:
            base += f" in {days_to} days"
        volatility = metrics.get("earnings_pre_volatility")
        if volatility is not None and volatility > 0:
            return f"{base}; pre-earnings volatility is elevated at {volatility:.2%}."
        return f"{base}; event risk may be higher than usual."
    if state == EVENT_STATE_POST:
        trend = metrics.get("earnings_after_trend_5")
        if trend is not None and trend > 0:
            return "Post-earnings mode is active and the post-event trend is positive."
        if trend is not None and trend < 0:
            return "Post-earnings mode is active and the post-event trend is negative."
        if days_after is not None:
            return f"Post-earnings mode is active, {days_after} days after the event."
        return "Post-earnings mode is active."
    return "No near-term earnings event state is active."


def _empty_context(summary: str) -> EarningsEventContext:
    return EarningsEventContext(
        event_state=EVENT_STATE_NORMAL,
        next_earnings_date=None,
        days_to_earnings=None,
        days_after_earnings=None,
        earnings_week_flag=False,
        earnings_day_flag=False,
        post_earnings_mode=False,
        summary=summary,
        metrics={},
    )


def _value(row: pd.Series, key: str) -> float | None:
    return safe_float(row.get(key))


def _safe_int(value: float | None) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)
