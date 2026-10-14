from __future__ import annotations

from typing import Any

import pandas as pd

from stock_ai.labels.targets import BINARY_CLASS_NAMES
from stock_ai.utils.safe_cast import safe_float, safe_int


HISTORY_COLUMNS = [
    "date",
    "ticker",
    "target",
    "model_name",
    "prediction_type",
    "created_at",
    "predicted_class",
    "prediction_direction",
    "probability_up",
    "probability_down",
    "confidence",
    "close",
    "next_close",
    "next_open",
    "actual_return",
    "actual_class",
    "actual_direction",
    "status",
    "is_correct",
]

MISSING_LOOKBACK_TRADING_DAYS = 60


def evaluate_prediction_history(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    target: str = "next_close_binary",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Compare saved predictions with the next available market close.

    This is an operational check: it only evaluates rows whose next trading day
    is already present in the local price data. The latest unresolved prediction
    remains pending and is excluded from accuracy calculations.
    """
    history = build_prediction_outcomes(predictions, prices, target=target)
    metrics = summarize_prediction_history(history)
    return history, metrics


def build_prediction_outcomes(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    target: str = "next_close_binary",
) -> pd.DataFrame:
    if predictions.empty:
        return _empty_history()

    rows = predictions.copy()
    if "target" in rows.columns:
        rows = rows[rows["target"] == target].copy()
    if rows.empty:
        return _empty_history()

    rows = _normalize_predictions(rows)
    eval_frame = (
        _next_open_frame(prices)
        if target in ("next_open_binary", "target_next_open_binary")
        else _next_close_frame(prices)
    )
    outcome = rows.merge(
        eval_frame,
        on=["date", "ticker"],
        how="left",
    )
    if outcome.empty:
        return _empty_history()

    outcome["actual_class"] = outcome["actual_return"].map(_actual_binary_class)
    outcome["prediction_direction"] = outcome["predicted_class"].map(_binary_label)
    outcome["actual_direction"] = outcome["actual_class"].map(_binary_label)
    outcome["status"] = outcome["actual_class"].map(
        lambda value: "pending" if safe_int(value) is None else "confirmed"
    )
    outcome["is_correct"] = outcome.apply(_is_correct, axis=1)
    return (
        outcome[[column for column in HISTORY_COLUMNS if column in outcome.columns]]
        .sort_values("date", ascending=False)
        .reset_index(drop=True)
    )


def summarize_prediction_history(history: pd.DataFrame) -> dict[str, object]:
    if history.empty:
        return _empty_metrics()

    settled = history[
        (history["status"] == "confirmed")
        & history["predicted_class"].isin([0, 1])
        & history["actual_class"].isin([0, 1])
    ].copy()
    correct = int(settled["is_correct"].sum()) if not settled.empty else 0

    return {
        "total_predictions": int(len(history)),
        "settled_predictions": int(len(settled)),
        "pending_predictions": int((history["status"] == "pending").sum()),
        "correct_predictions": correct,
        "accuracy": _accuracy(settled),
        "up_prediction_accuracy": _direction_accuracy(settled, predicted_class=1),
        "down_prediction_accuracy": _direction_accuracy(settled, predicted_class=0),
        "average_confidence": _average_confidence(settled),
        "confidence_accuracy": _confidence_accuracy(settled),
    }


def summarize_prediction_operations(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str | None = None,
    lookback_trading_days: int = MISSING_LOOKBACK_TRADING_DAYS,
    live_prediction_start_date=None,
) -> dict[str, object]:
    """Summarize operational prediction coverage for saved prediction rows."""
    if predictions.empty:
        live_predictions = predictions.copy()
        backfill_predictions = predictions.copy()
    else:
        live_predictions = _filter_live_predictions(predictions)
        backfill_predictions = _filter_backfill_predictions(predictions)
    missing = missing_prediction_dates(
        live_predictions,
        prices,
        ticker=ticker,
        lookback_trading_days=lookback_trading_days,
        live_prediction_start_date=live_prediction_start_date,
    )
    pre_start_excluded = ignored_pre_start_missing_prediction_dates(
        live_predictions,
        prices,
        ticker=ticker,
        lookback_trading_days=lookback_trading_days,
        live_prediction_start_date=live_prediction_start_date,
    )
    backfilled_dates = _saved_prediction_dates(backfill_predictions, ticker=ticker)
    backfilled_missing = [date for date in missing if date in backfilled_dates]
    backfillable_missing = [date for date in missing if date not in backfilled_dates]
    pending = _pending_prediction_dates(predictions, prices)
    latest_prediction_date = _latest_date(
        live_predictions.get("date") if not live_predictions.empty else None
    )
    latest_saved_at = _latest_datetime(
        live_predictions.get("created_at") if not live_predictions.empty else None
    )
    return {
        "latest_prediction_date": latest_prediction_date,
        "latest_saved_at": latest_saved_at,
        "live_prediction_start_date": _normalize_start_date(live_prediction_start_date),
        "missing_prediction_dates": missing,
        "missing_prediction_count": int(len(missing)),
        "pre_start_excluded_dates": pre_start_excluded,
        "pre_start_excluded_count": int(len(pre_start_excluded)),
        # Backward-compatible aliases for older UI/tests. New display code should use
        # pre_start_excluded_* so pre-operation dates are not described as missing.
        "ignored_pre_start_missing_dates": pre_start_excluded,
        "ignored_pre_start_missing_count": int(len(pre_start_excluded)),
        "backfilled_missing_dates": backfilled_missing,
        "backfilled_missing_count": int(len(backfilled_missing)),
        "backfillable_missing_dates": backfillable_missing,
        "backfillable_missing_count": int(len(backfillable_missing)),
        "first_backfillable_missing_date": backfillable_missing[0] if backfillable_missing else None,
        "next_confirmable_prediction_date": pending[0] if pending else None,
    }


def diagnose_missing_live_dates(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    missing_dates: list[pd.Timestamp],
    ticker: str | None = None,
    known_missed_live_dates: list[dict[str, Any]] | tuple[Any, ...] | None = None,
) -> pd.DataFrame:
    """Explain why each missing live prediction date looks missing.

    This does not decide whether to suppress a warning. It only separates
    operational cases such as backfill-only rows, one target missing, or a daily
    run that saved the previous trading date because newer market data was not
    available at run time.
    """
    columns = [
        "missing_date",
        "in_feature_data",
        "any_prediction_exists",
        "binary_live_exists",
        "multiclass_live_exists",
        "backfill_exists",
        "source_breakdown",
        "previous_live_prediction_date",
        "previous_live_saved_at",
        "next_live_prediction_date",
        "inferred_reason",
        "known_missed",
        "known_reason_code",
        "known_reason",
        "known_action",
        "known_allow_backfill",
        "recommended_action",
    ]
    if not missing_dates:
        return pd.DataFrame(columns=columns)

    rows = _normalize_diagnostic_predictions(predictions, ticker=ticker)
    trading_dates = _trading_dates(prices, ticker=ticker)
    trading_date_set = set(trading_dates)
    live_rows = _filter_live_predictions(rows) if not rows.empty else rows.copy()
    live_dates = sorted(_saved_prediction_dates(live_rows, ticker=ticker))
    known_lookup = _known_missed_lookup(known_missed_live_dates, ticker=ticker)

    diagnostics = []
    for value in missing_dates:
        missing_date = pd.Timestamp(value).normalize()
        known = known_lookup.get(missing_date, {})
        date_rows = rows[rows["date"] == missing_date].copy() if not rows.empty else rows.copy()
        date_live = _filter_live_predictions(date_rows) if not date_rows.empty else date_rows.copy()
        date_backfill = _filter_backfill_predictions(date_rows) if not date_rows.empty else date_rows.copy()
        previous_date = _nearest_live_date(live_dates, missing_date, before=True)
        next_date = _nearest_live_date(live_dates, missing_date, before=False)
        previous_saved_at = _latest_saved_at_for_date(live_rows, previous_date)

        binary_live = _has_target_family(date_live, "binary")
        multiclass_live = _has_target_family(date_live, "multiclass")
        reason = _infer_missing_live_reason(
            missing_date=missing_date,
            in_feature_data=missing_date in trading_date_set,
            any_prediction_exists=not date_rows.empty,
            binary_live_exists=binary_live,
            multiclass_live_exists=multiclass_live,
            backfill_exists=not date_backfill.empty,
            previous_saved_at=previous_saved_at,
        )
        diagnostics.append(
            {
                "missing_date": missing_date,
                "in_feature_data": bool(missing_date in trading_date_set),
                "any_prediction_exists": bool(not date_rows.empty),
                "binary_live_exists": bool(binary_live),
                "multiclass_live_exists": bool(multiclass_live),
                "backfill_exists": bool(not date_backfill.empty),
                "source_breakdown": _source_breakdown(date_rows),
                "previous_live_prediction_date": previous_date,
                "previous_live_saved_at": previous_saved_at,
                "next_live_prediction_date": next_date,
                "inferred_reason": reason,
                "known_missed": bool(known),
                "known_reason_code": known.get("reason_code"),
                "known_reason": known.get("reason"),
                "known_action": known.get("action"),
                "known_allow_backfill": known.get("allow_backfill"),
                "recommended_action": _missing_live_recommended_action(reason),
            }
        )

    return pd.DataFrame(diagnostics, columns=columns)


def missing_prediction_dates(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str | None = None,
    lookback_trading_days: int = MISSING_LOOKBACK_TRADING_DAYS,
    live_prediction_start_date=None,
) -> list[pd.Timestamp]:
    """Find local trading dates without saved live predictions.

    The local feature/price dataset is used as the trading calendar, so this
    avoids guessing Japanese market holidays.
    """
    trading_dates = _operation_scope_trading_dates(
        prices,
        ticker=ticker,
        lookback_trading_days=lookback_trading_days,
    )
    if not trading_dates:
        return []
    start_date = _normalize_start_date(live_prediction_start_date)
    if start_date is not None:
        trading_dates = [date for date in trading_dates if date >= start_date]

    saved = _saved_prediction_dates(predictions, ticker=ticker)
    return [date for date in trading_dates if date not in saved]


def ignored_pre_start_missing_prediction_dates(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    ticker: str | None = None,
    lookback_trading_days: int = MISSING_LOOKBACK_TRADING_DAYS,
    live_prediction_start_date=None,
) -> list[pd.Timestamp]:
    start_date = _normalize_start_date(live_prediction_start_date)
    if start_date is None:
        return []
    trading_dates = _operation_scope_trading_dates(
        prices,
        ticker=ticker,
        lookback_trading_days=lookback_trading_days,
    )
    saved = _saved_prediction_dates(predictions, ticker=ticker)
    return [date for date in trading_dates if date < start_date and date not in saved]


def _normalize_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["date"])
    rows["ticker"] = rows["ticker"].astype(str)

    for column in ("predicted_class",):
        rows[column] = rows[column].map(safe_int)
    for column in ("probability_up", "probability_down", "confidence"):
        if column not in rows.columns:
            rows[column] = None
        rows[column] = rows[column].map(safe_float)
    rows["confidence"] = rows.apply(_confidence_from_row, axis=1)
    if "prediction_type" not in rows.columns:
        rows["prediction_type"] = "live"
    rows["prediction_type"] = rows["prediction_type"].fillna("live").astype(str)
    return rows


def _next_close_frame(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(columns=["date", "ticker", "close", "next_close", "actual_return"])

    frame = prices[["date", "ticker", "close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["close"] = frame["close"].map(safe_float)
    frame = frame.dropna(subset=["date", "ticker", "close"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["next_close"] = frame.groupby("ticker")["close"].shift(-1)
    frame["actual_return"] = frame["next_close"] / frame["close"] - 1
    return frame


def _next_open_frame(prices: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close", "open"}
    if prices.empty or not required.issubset(prices.columns):
        return pd.DataFrame(columns=["date", "ticker", "close", "next_open", "actual_return"])

    frame = prices[["date", "ticker", "close", "open"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame["close"] = frame["close"].map(safe_float)
    frame["open"] = frame["open"].map(safe_float)
    frame = frame.dropna(subset=["date", "ticker", "close", "open"])
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    frame["next_open"] = frame.groupby("ticker")["open"].shift(-1)
    frame["actual_return"] = frame["next_open"] / frame["close"] - 1
    return frame


def _confidence_from_row(row: pd.Series) -> float | None:
    confidence = safe_float(row.get("confidence"))
    if confidence is not None:
        return confidence

    values = [
        value
        for value in (safe_float(row.get("probability_up")), safe_float(row.get("probability_down")))
        if value is not None
    ]
    return max(values) if values else None


def _actual_binary_class(value) -> int | None:
    actual_return = safe_float(value)
    if actual_return is None:
        return None
    return int(actual_return > 0)


def _binary_label(value) -> str:
    class_id = safe_int(value)
    if class_id is None:
        return "unknown"
    return BINARY_CLASS_NAMES.get(class_id, "unknown")


def _is_correct(row: pd.Series) -> bool | None:
    predicted = safe_int(row.get("predicted_class"))
    actual = safe_int(row.get("actual_class"))
    if predicted is None or actual is None:
        return None
    return predicted == actual


def _accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float(frame["is_correct"].mean())


def _direction_accuracy(frame: pd.DataFrame, predicted_class: int) -> float | None:
    subset = frame[frame["predicted_class"] == predicted_class]
    if subset.empty:
        return None
    return float(subset["is_correct"].mean())


def _average_confidence(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    confidence = pd.to_numeric(frame["confidence"], errors="coerce").dropna()
    if confidence.empty:
        return None
    return float(confidence.mean())


def _confidence_accuracy(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["confidence_bucket", "prediction_count", "correct_count", "accuracy"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    rows = frame.copy()
    rows["confidence"] = pd.to_numeric(rows["confidence"], errors="coerce")
    rows = rows.dropna(subset=["confidence"])
    if rows.empty:
        return pd.DataFrame(columns=columns)

    rows["confidence_bucket"] = pd.cut(
        rows["confidence"],
        bins=[-float("inf"), 0.5, 0.6, 0.7, 0.8, 0.9, float("inf")],
        labels=["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90%+"],
        right=False,
    )
    summary = (
        rows.groupby("confidence_bucket", observed=True)["is_correct"]
        .agg(prediction_count="count", correct_count="sum", accuracy="mean")
        .reset_index()
    )
    summary["correct_count"] = summary["correct_count"].astype(int)
    return summary[columns]


def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _empty_metrics() -> dict[str, object]:
    return {
        "total_predictions": 0,
        "settled_predictions": 0,
        "pending_predictions": 0,
        "correct_predictions": 0,
        "accuracy": None,
        "up_prediction_accuracy": None,
        "down_prediction_accuracy": None,
        "average_confidence": None,
        "confidence_accuracy": pd.DataFrame(
            columns=["confidence_bucket", "prediction_count", "correct_count", "accuracy"]
        ),
    }


def _filter_live_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    if "prediction_type" not in predictions.columns:
        return predictions.copy()
    return predictions[predictions["prediction_type"].fillna("live") == "live"].copy()


def _filter_backfill_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "prediction_type" not in predictions.columns:
        return pd.DataFrame(columns=predictions.columns)
    return predictions[predictions["prediction_type"].fillna("live") == "backfill"].copy()


def _normalize_diagnostic_predictions(
    predictions: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=predictions.columns)
    rows = predictions.copy()
    if "date" not in rows.columns:
        return pd.DataFrame(columns=rows.columns)
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows = rows.dropna(subset=["date"])
    if ticker is not None and "ticker" in rows.columns:
        rows = rows[rows["ticker"].astype(str) == ticker].copy()
    if "prediction_type" not in rows.columns:
        rows["prediction_type"] = "live"
    rows["prediction_type"] = rows["prediction_type"].fillna("live").astype(str)
    if "target" not in rows.columns:
        rows["target"] = ""
    if "created_at" in rows.columns:
        rows["created_at"] = pd.to_datetime(rows["created_at"], errors="coerce")
    return rows.reset_index(drop=True)


def _has_target_family(rows: pd.DataFrame, family: str) -> bool:
    if rows.empty or "target" not in rows.columns:
        return False
    return bool(rows["target"].astype(str).str.contains(family, case=False, na=False).any())


def _source_breakdown(rows: pd.DataFrame) -> str:
    if rows.empty or "prediction_type" not in rows.columns:
        return "none"
    counts = rows["prediction_type"].fillna("live").astype(str).value_counts().sort_index()
    return "; ".join(f"{name}={int(count)}" for name, count in counts.items())


def _nearest_live_date(
    live_dates: list[pd.Timestamp],
    missing_date: pd.Timestamp,
    *,
    before: bool,
) -> pd.Timestamp | None:
    if before:
        candidates = [date for date in live_dates if date < missing_date]
        return candidates[-1] if candidates else None
    candidates = [date for date in live_dates if date > missing_date]
    return candidates[0] if candidates else None


def _latest_saved_at_for_date(rows: pd.DataFrame, date: pd.Timestamp | None) -> pd.Timestamp | None:
    if date is None or rows.empty or "created_at" not in rows.columns:
        return None
    saved = pd.to_datetime(rows.loc[rows["date"] == date, "created_at"], errors="coerce").dropna()
    if saved.empty:
        return None
    return pd.Timestamp(saved.max())


def _infer_missing_live_reason(
    *,
    missing_date: pd.Timestamp,
    in_feature_data: bool,
    any_prediction_exists: bool,
    binary_live_exists: bool,
    multiclass_live_exists: bool,
    backfill_exists: bool,
    previous_saved_at: pd.Timestamp | None,
) -> str:
    if not in_feature_data:
        return "market_calendar_mismatch"
    if binary_live_exists and not multiclass_live_exists:
        return "multiclass_missing_only"
    if multiclass_live_exists and not binary_live_exists:
        return "binary_missing_only"
    if backfill_exists and not binary_live_exists and not multiclass_live_exists:
        return "backfill_only"
    if any_prediction_exists and not binary_live_exists and not multiclass_live_exists:
        return "prediction_exists_but_not_live"
    if previous_saved_at is not None and previous_saved_at.normalize() == missing_date:
        return "data_not_available_at_run_time"
    return "no_live_prediction_saved"


def _missing_live_recommended_action(reason: str) -> str:
    if reason == "data_not_available_at_run_time":
        return (
            "Do not create a live row manually. Treat this as a missed operational live date; "
            "use backfill only for research, or add a future catch-up design."
        )
    if reason in {"binary_missing_only", "multiclass_missing_only", "prediction_exists_but_not_live"}:
        return "Investigate prediction save flow for the missing target/source before changing history."
    if reason == "backfill_only":
        return "Keep backfill separate from live history; live remains missing for operations."
    if reason == "market_calendar_mismatch":
        return "Check the local feature calendar before treating this as a live prediction gap."
    return "Check daily_predict logs and scheduler history; do not fabricate a live prediction."


def _known_missed_lookup(
    known_missed_live_dates: list[dict[str, Any]] | tuple[Any, ...] | None,
    ticker: str | None = None,
) -> dict[pd.Timestamp, dict[str, Any]]:
    if not known_missed_live_dates:
        return {}
    result: dict[pd.Timestamp, dict[str, Any]] = {}
    for item in known_missed_live_dates:
        row = _known_missed_item_to_dict(item)
        known_ticker = row.get("ticker")
        if ticker is not None and known_ticker not in (None, "", ticker):
            continue
        date = _normalize_start_date(row.get("date"))
        if date is None or date in result:
            continue
        result[date] = row
    return result


def _known_missed_item_to_dict(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    return {
        "date": getattr(item, "date", None),
        "ticker": getattr(item, "ticker", None),
        "reason_code": getattr(item, "reason_code", None),
        "reason": getattr(item, "reason", None),
        "action": getattr(item, "action", None),
        "allow_backfill": getattr(item, "allow_backfill", None),
    }


def _trading_dates(prices: pd.DataFrame, ticker: str | None = None) -> list[pd.Timestamp]:
    if prices.empty or "date" not in prices.columns:
        return []
    rows = prices.copy()
    if ticker is not None and "ticker" in rows.columns:
        rows = rows[rows["ticker"].astype(str) == ticker]
    dates = pd.to_datetime(rows["date"], errors="coerce").dt.normalize().dropna()
    return sorted(pd.Timestamp(date) for date in dates.unique())


def _operation_scope_trading_dates(
    prices: pd.DataFrame,
    ticker: str | None = None,
    lookback_trading_days: int = MISSING_LOOKBACK_TRADING_DAYS,
) -> list[pd.Timestamp]:
    trading_dates = _trading_dates(prices, ticker=ticker)
    if lookback_trading_days > 0:
        return trading_dates[-lookback_trading_days:]
    return trading_dates


def _normalize_start_date(value) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).normalize()


def _saved_prediction_dates(predictions: pd.DataFrame, ticker: str | None = None) -> set[pd.Timestamp]:
    if predictions.empty or "date" not in predictions.columns:
        return set()
    rows = predictions.copy()
    if ticker is not None and "ticker" in rows.columns:
        rows = rows[rows["ticker"].astype(str) == ticker]
    dates = pd.to_datetime(rows["date"], errors="coerce").dt.normalize().dropna()
    return {pd.Timestamp(date) for date in dates.unique()}


def _pending_prediction_dates(predictions: pd.DataFrame, prices: pd.DataFrame) -> list[pd.Timestamp]:
    if predictions.empty:
        return []
    history = build_prediction_outcomes(predictions, prices)
    if history.empty or "status" not in history.columns:
        return []
    pending = history[history["status"] == "pending"].copy()
    if pending.empty:
        return []
    dates = pd.to_datetime(pending["date"], errors="coerce").dropna().sort_values()
    return [pd.Timestamp(date) for date in dates]


def _latest_date(values) -> pd.Timestamp | None:
    if values is None:
        return None
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).normalize()


def _latest_datetime(values) -> pd.Timestamp | None:
    if values is None:
        return None
    dates = pd.to_datetime(values, errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max())
