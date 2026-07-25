from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from stock_ai.labels.targets import BINARY_CLASS_NAMES, CLASS_NAMES, classify_return
from stock_ai.utils.safe_cast import safe_float, safe_int


CONFIDENCE_THRESHOLDS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70)
HIGH_CONFIDENCE_THRESHOLD = 0.60
LARGE_MOVE_THRESHOLD = 0.02


@dataclass(frozen=True)
class EvaluationTargetSummary:
    binary_rows: int
    binary_confirmed_rows: int
    binary_pending_rows: int
    multiclass_rows: int
    multiclass_confirmed_rows: int
    multiclass_pending_rows: int
    live_rows: int
    backfill_rows: int
    historical_rows: int
    evaluated_start: str | None
    evaluated_end: str | None


def build_binary_diagnostic_frame(
    predictions: pd.DataFrame,
    feature_data: pd.DataFrame,
    *,
    prediction_type: str | None = None,
) -> pd.DataFrame:
    rows = _prediction_rows(predictions, "next_close_binary", prediction_type=prediction_type)
    if rows.empty:
        return _empty_binary_frame()
    outcomes = _attach_next_close(rows, feature_data)
    if outcomes.empty:
        return _empty_binary_frame()

    outcomes["actual_class"] = outcomes["actual_return"].map(_actual_binary_class)
    outcomes["status"] = outcomes["actual_class"].map(
        lambda value: "pending" if safe_int(value) is None else "confirmed"
    )
    outcomes["is_correct"] = outcomes.apply(_is_correct, axis=1)
    outcomes = _attach_context(outcomes, feature_data)
    outcomes = _attach_baseline_predictions(outcomes, feature_data)
    outcomes = _normalize_prediction_columns(outcomes)
    return outcomes.sort_values(["date", "created_at"], ascending=[False, False]).reset_index(drop=True)


def build_multiclass_diagnostic_frame(
    predictions: pd.DataFrame,
    feature_data: pd.DataFrame,
    *,
    prediction_type: str | None = None,
) -> pd.DataFrame:
    rows = _prediction_rows(predictions, "next_close_multiclass", prediction_type=prediction_type)
    if rows.empty:
        return _empty_multiclass_frame()
    outcomes = _attach_next_close(rows, feature_data)
    if outcomes.empty:
        return _empty_multiclass_frame()

    outcomes["actual_class"] = outcomes["actual_return"].map(
        lambda value: None if safe_float(value) is None else classify_return(float(value))
    )
    outcomes["status"] = outcomes["actual_class"].map(
        lambda value: "pending" if safe_int(value) is None else "confirmed"
    )
    outcomes["is_correct"] = outcomes.apply(_is_correct, axis=1)
    outcomes = _attach_context(outcomes, feature_data)
    outcomes = _normalize_prediction_columns(outcomes)
    return outcomes.sort_values(["date", "created_at"], ascending=[False, False]).reset_index(drop=True)


def build_walk_forward_binary_diagnostic_frame(
    predictions: pd.DataFrame,
    feature_data: pd.DataFrame,
    *,
    prediction_type: str = "historical_walk_forward",
) -> pd.DataFrame:
    """Normalize walk-forward rows so they can use the saved-history diagnostics."""
    if predictions.empty:
        return _empty_binary_frame()

    rows = predictions.copy()
    rows["target"] = rows.get("target", "next_close_binary")
    rows["model_name"] = rows.get("model_name", "walk_forward_binary")
    rows["prediction_type"] = rows.get("prediction_type", prediction_type)
    rows["created_at"] = rows.get("created_at", None)
    if "actual_return" not in rows.columns and "next_close_return" in rows.columns:
        rows["actual_return"] = rows["next_close_return"]
    if "actual_class" not in rows.columns and "actual_return" in rows.columns:
        rows["actual_class"] = rows["actual_return"].map(_actual_binary_class)
    if "confidence" not in rows.columns or rows["confidence"].isna().all():
        probability_columns = [column for column in ("probability_up", "probability_down") if column in rows.columns]
        if probability_columns:
            rows["confidence"] = rows[probability_columns].max(axis=1)

    rows["status"] = rows.get("actual_class", pd.Series(index=rows.index)).map(
        lambda value: "pending" if safe_int(value) is None else "confirmed"
    )
    rows["is_correct"] = rows.apply(_is_correct, axis=1)
    rows = _attach_next_close_metadata(rows, feature_data)
    rows = _attach_context_without_duplicates(rows, feature_data)
    rows = _attach_baseline_predictions_without_duplicates(rows, feature_data)
    rows = _normalize_prediction_columns(rows)
    return rows.sort_values(["date"], ascending=False).reset_index(drop=True)


def summarize_evaluation_targets(
    binary_frame: pd.DataFrame,
    multiclass_frame: pd.DataFrame,
) -> EvaluationTargetSummary:
    binary_confirmed = _confirmed(binary_frame)
    multiclass_confirmed = _confirmed(multiclass_frame)
    combined = pd.concat(
        [
            binary_confirmed[["date"]] if not binary_confirmed.empty else pd.DataFrame(columns=["date"]),
            multiclass_confirmed[["date"]]
            if not multiclass_confirmed.empty
            else pd.DataFrame(columns=["date"]),
        ],
        ignore_index=True,
    )
    dates = pd.to_datetime(combined.get("date", pd.Series(dtype=object)), errors="coerce").dropna()
    all_rows = pd.concat(
        [
            binary_frame[["prediction_type"]] if "prediction_type" in binary_frame else pd.DataFrame(),
            multiclass_frame[["prediction_type"]] if "prediction_type" in multiclass_frame else pd.DataFrame(),
        ],
        ignore_index=True,
    )
    types = all_rows.get("prediction_type", pd.Series(dtype=str)).fillna("live").astype(str)
    return EvaluationTargetSummary(
        binary_rows=int(len(binary_frame)),
        binary_confirmed_rows=int(len(binary_confirmed)),
        binary_pending_rows=int((binary_frame.get("status", pd.Series(dtype=str)) == "pending").sum()),
        multiclass_rows=int(len(multiclass_frame)),
        multiclass_confirmed_rows=int(len(multiclass_confirmed)),
        multiclass_pending_rows=int((multiclass_frame.get("status", pd.Series(dtype=str)) == "pending").sum()),
        live_rows=int(types.eq("live").sum()),
        backfill_rows=int(types.eq("backfill").sum()),
        historical_rows=int(types.str.contains("historical|walk_forward|backtest", case=False, regex=True).sum()),
        evaluated_start=pd.Timestamp(dates.min()).strftime("%Y-%m-%d") if not dates.empty else None,
        evaluated_end=pd.Timestamp(dates.max()).strftime("%Y-%m-%d") if not dates.empty else None,
    )


def baseline_accuracy_frame(frame: pd.DataFrame, *, seed: int = 42) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = [
        "baseline",
        "model_accuracy",
        "baseline_accuracy",
        "model_minus_baseline",
        "sample_count",
        "evaluated_start",
        "evaluated_end",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    actual = rows["actual_class"].map(safe_int).astype(int)
    model = rows["predicted_class"].map(safe_int).astype(int)
    model_accuracy = _accuracy(actual, model)
    start, end = _period(rows)
    baseline_predictions = {
        "always_up": pd.Series(1, index=rows.index),
        "always_down": pd.Series(0, index=rows.index),
        "previous_day_direction": rows.get("baseline_previous_day_direction", pd.Series(index=rows.index)).map(
            safe_int
        ),
        "moving_average_trend_only": rows.get("baseline_moving_average_trend", pd.Series(index=rows.index)).map(
            safe_int
        ),
        "random_baseline": _random_baseline(actual, seed=seed),
        "class_distribution_baseline": pd.Series(int(actual.mean() >= 0.5), index=rows.index),
    }
    result = []
    for name, prediction in baseline_predictions.items():
        prediction = pd.Series(prediction, index=rows.index).map(safe_int)
        valid = actual.notna() & prediction.notna()
        baseline_accuracy = _accuracy(actual[valid], prediction[valid]) if bool(valid.any()) else None
        result.append(
            {
                "baseline": name,
                "model_accuracy": model_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "model_minus_baseline": (
                    model_accuracy - baseline_accuracy
                    if model_accuracy is not None and baseline_accuracy is not None
                    else None
                ),
                "sample_count": int(valid.sum()),
                "evaluated_start": start,
                "evaluated_end": end,
            }
        )
    return pd.DataFrame(result, columns=columns)


def confidence_threshold_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = [
        "threshold",
        "sample_count",
        "accuracy",
        "precision",
        "recall",
        "hit_rate",
        "average_confidence",
        "coverage",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    confidence = pd.to_numeric(rows.get("confidence"), errors="coerce")
    total = len(rows)
    result = []
    for threshold in CONFIDENCE_THRESHOLDS:
        subset = rows[confidence >= threshold].copy()
        metrics = _binary_metrics(subset)
        result.append(
            {
                "threshold": threshold,
                "sample_count": int(len(subset)),
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "hit_rate": metrics["accuracy"],
                "average_confidence": _mean(subset.get("confidence")),
                "coverage": _safe_divide(len(subset), total),
            }
        )
    return pd.DataFrame(result, columns=columns)


def regime_accuracy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return _context_accuracy_frame(frame, "regime")


def anomaly_accuracy_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if rows.empty:
        return _empty_context_frame("anomaly_level")
    rows["anomaly_level"] = rows.apply(_anomaly_level, axis=1)
    return _context_accuracy_frame(rows, "anomaly_level")


def binary_class_performance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = ["metric", "class", "sample_count", "value"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    result = []
    for class_id, label in BINARY_CLASS_NAMES.items():
        predicted_mask = predicted == class_id
        predicted_count = int(predicted_mask.sum())
        result.append(
            {
                "metric": "predicted_class_accuracy",
                "class": label,
                "sample_count": predicted_count,
                "value": _safe_divide(int((actual[predicted_mask] == class_id).sum()), predicted_count),
            }
        )
        actual_mask = actual == class_id
        actual_count = int(actual_mask.sum())
        result.append(
            {
                "metric": "actual_class_recall",
                "class": label,
                "sample_count": actual_count,
                "value": _safe_divide(int((predicted[actual_mask] == class_id).sum()), actual_count),
            }
        )
    return pd.DataFrame(result, columns=columns)


def multiclass_performance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = ["class_id", "class", "precision", "recall", "support", "predicted_count"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    result = []
    for class_id, label in CLASS_NAMES.items():
        predicted_mask = predicted == class_id
        actual_mask = actual == class_id
        tp = int((predicted_mask & actual_mask).sum())
        predicted_count = int(predicted_mask.sum())
        support = int(actual_mask.sum())
        result.append(
            {
                "class_id": class_id,
                "class": label,
                "precision": _safe_divide(tp, predicted_count),
                "recall": _safe_divide(tp, support),
                "support": support,
                "predicted_count": predicted_count,
            }
        )
    return pd.DataFrame(result, columns=columns)


def multiclass_confusion_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    labels = list(CLASS_NAMES.values())
    matrix = pd.DataFrame(0, index=labels, columns=labels)
    if rows.empty:
        return matrix
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    for actual_id, predicted_id in zip(actual, predicted, strict=False):
        actual_label = CLASS_NAMES.get(actual_id)
        predicted_label = CLASS_NAMES.get(predicted_id)
        if actual_label in matrix.index and predicted_label in matrix.columns:
            matrix.loc[actual_label, predicted_label] += 1
    matrix.index.name = "Actual"
    matrix.columns.name = "Predicted"
    return matrix


def biggest_hits_misses_frame(frame: pd.DataFrame, *, limit: int = 40) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = [
        "category",
        "feature_date",
        "target_close_date",
        "predicted_class",
        "actual_direction",
        "actual_return",
        "confidence",
        "regime",
        "anomaly_score",
        "top_features",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    result = rows.copy()
    confidence = pd.to_numeric(result.get("confidence"), errors="coerce").fillna(0.0)
    actual_return = pd.to_numeric(result.get("actual_return"), errors="coerce")
    correct = result["is_correct"].fillna(False).astype(bool)
    large_move = actual_return.abs() >= LARGE_MOVE_THRESHOLD
    result["category"] = np.select(
        [
            large_move & correct,
            large_move & ~correct,
            (confidence >= HIGH_CONFIDENCE_THRESHOLD) & correct,
            (confidence >= HIGH_CONFIDENCE_THRESHOLD) & ~correct,
            correct,
            ~correct,
        ],
        [
            "large_move_hit",
            "large_move_miss",
            "high_confidence_hit",
            "high_confidence_miss",
            "low_confidence_hit",
            "low_confidence_miss",
        ],
        default="unknown",
    )
    result["_rank"] = actual_return.abs().fillna(0.0) * 10 + confidence
    result = result.sort_values("_rank", ascending=False).head(limit)
    result["feature_date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["target_close_date"] = pd.to_datetime(result["target_close_date"], errors="coerce").dt.strftime(
        "%Y-%m-%d"
    )
    result["predicted_class"] = result["predicted_class"].map(lambda value: BINARY_CLASS_NAMES.get(safe_int(value), value))
    result["actual_direction"] = result["actual_class"].map(lambda value: BINARY_CLASS_NAMES.get(safe_int(value), value))
    result["top_features"] = result.get("top_features", pd.Series(index=result.index)).map(_top_feature_text)
    if "regime" not in result.columns:
        result["regime"] = "Unknown"
    if "anomaly_score" not in result.columns:
        result["anomaly_score"] = None
    return result[[column for column in columns if column in result.columns]].reset_index(drop=True)


def source_breakdown_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "prediction_type" not in frame.columns:
        return pd.DataFrame(columns=["source", "rows", "confirmed_rows", "accuracy"])
    rows = []
    for source, group in frame.groupby(frame["prediction_type"].fillna("live").astype(str), dropna=False):
        confirmed = _confirmed(group)
        rows.append(
            {
                "source": source,
                "rows": int(len(group)),
                "confirmed_rows": int(len(confirmed)),
                "accuracy": _binary_metrics(confirmed)["accuracy"] if not confirmed.empty else None,
            }
        )
    return pd.DataFrame(rows, columns=["source", "rows", "confirmed_rows", "accuracy"])


def source_comparison_frame(frames: dict[str, pd.DataFrame], *, min_sample_warning: int = 30) -> pd.DataFrame:
    columns = _summary_columns(include_source=True)
    result = []
    for source, frame in frames.items():
        row = _summary_row(frame, source=str(source), min_sample_warning=min_sample_warning)
        result.append(row)
    return pd.DataFrame(result, columns=columns)


def period_summary_frame(
    frame: pd.DataFrame,
    *,
    reference_date=None,
    min_sample_warning: int = 30,
) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = _summary_columns(include_source=False, label_column="period")
    if rows.empty:
        return pd.DataFrame(columns=columns)

    dates = pd.to_datetime(rows.get("date"), errors="coerce")
    latest = pd.Timestamp(reference_date).normalize() if reference_date is not None else pd.Timestamp(dates.max()).normalize()
    period_masks = {
        "2023-": dates >= pd.Timestamp("2023-01-01"),
        "2024-": dates >= pd.Timestamp("2024-01-01"),
        "2025-": dates >= pd.Timestamp("2025-01-01"),
        "2026-": dates >= pd.Timestamp("2026-01-01"),
        "Last 6 months": dates >= latest - pd.DateOffset(months=6),
        "Last 3 months": dates >= latest - pd.DateOffset(months=3),
    }
    result = []
    for label, mask in period_masks.items():
        result.append(
            _summary_row(
                rows[mask].copy(),
                period=label,
                min_sample_warning=min_sample_warning,
            )
        )
    return pd.DataFrame(result, columns=columns)


def confidence_distribution_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = ["metric", "count", "min", "max", "mean", "median", "count_48_52", "count_ge_55"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    result = []
    for metric in ("probability_up", "probability_down", "confidence"):
        values = pd.to_numeric(rows.get(metric), errors="coerce").dropna()
        if values.empty:
            result.append(
                {
                    "metric": metric,
                    "count": 0,
                    "min": None,
                    "max": None,
                    "mean": None,
                    "median": None,
                    "count_48_52": 0,
                    "count_ge_55": 0,
                }
            )
            continue
        result.append(
            {
                "metric": metric,
                "count": int(len(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "count_48_52": int(((values >= 0.48) & (values <= 0.52)).sum()),
                "count_ge_55": int((values >= 0.55).sum()),
            }
        )
    return pd.DataFrame(result, columns=columns)


def multiclass_flat_summary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(frame)
    columns = ["sample_count", "flat_count", "flat_rate", "flat_avg_actual_return", "flat_avg_abs_actual_return"]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    predicted = rows["predicted_class"].map(safe_int)
    flat = rows[predicted == 2].copy()
    returns = pd.to_numeric(flat.get("actual_return"), errors="coerce")
    return pd.DataFrame(
        [
            {
                "sample_count": int(len(rows)),
                "flat_count": int(len(flat)),
                "flat_rate": _safe_divide(len(flat), len(rows)),
                "flat_avg_actual_return": float(returns.mean()) if not returns.dropna().empty else None,
                "flat_avg_abs_actual_return": float(returns.abs().mean()) if not returns.dropna().empty else None,
            }
        ],
        columns=columns,
    )


def _prediction_rows(
    predictions: pd.DataFrame,
    target: str,
    *,
    prediction_type: str | None = None,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = predictions.copy()
    if "target" in rows.columns:
        rows = rows[rows["target"].astype(str).eq(target)].copy()
    if prediction_type is not None and "prediction_type" in rows.columns:
        rows = rows[rows["prediction_type"].fillna("live").astype(str).eq(prediction_type)].copy()
    if "prediction_type" not in rows.columns:
        rows["prediction_type"] = "live"
    return rows.reset_index(drop=True)


def _attach_next_close(predictions: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    if predictions.empty or feature_data.empty or not required.issubset(feature_data.columns):
        return pd.DataFrame()
    prices = feature_data[["date", "ticker", "close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce").dt.normalize()
    prices["ticker"] = prices["ticker"].astype(str)
    prices["close"] = prices["close"].map(safe_float)
    prices = prices.dropna(subset=["date", "ticker", "close"])
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["target_close_date"] = prices.groupby("ticker")["date"].shift(-1)
    prices["next_close"] = prices.groupby("ticker")["close"].shift(-1)
    prices["actual_return"] = prices["next_close"] / prices["close"] - 1

    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str)
    return rows.merge(
        prices[["date", "ticker", "close", "target_close_date", "next_close", "actual_return"]],
        on=["date", "ticker"],
        how="left",
    )


def _attach_next_close_metadata(rows: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or feature_data.empty or not {"date", "ticker", "close"}.issubset(feature_data.columns):
        return rows.copy()
    metadata = _attach_next_close(rows[["date", "ticker"]].drop_duplicates().copy(), feature_data)
    metadata_columns = ["date", "ticker", "close", "target_close_date", "next_close"]
    if "actual_return" not in rows.columns:
        metadata_columns.append("actual_return")
    result = rows.copy()
    for column in ("close", "target_close_date", "next_close"):
        if column in result.columns:
            result = result.drop(columns=[column])
    return result.merge(metadata[[column for column in metadata_columns if column in metadata.columns]], on=["date", "ticker"], how="left")


def _attach_context(rows: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or feature_data.empty or "date" not in feature_data.columns:
        return rows.copy()
    context_columns = [
        "date",
        "ticker",
        "regime",
        "anomaly_score",
        "anomaly_level",
        "anomaly_types",
        "surge_risk_level",
        "crash_risk_level",
    ]
    context_columns = [column for column in context_columns if column in feature_data.columns]
    context = feature_data[context_columns].copy()
    context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.normalize()
    if "ticker" in context.columns:
        context["ticker"] = context["ticker"].astype(str)
    else:
        context["ticker"] = rows["ticker"].iloc[0] if "ticker" in rows.columns and not rows.empty else ""
    result = rows.merge(context, on=["date", "ticker"], how="left")
    for column in ("regime", "anomaly_level", "anomaly_types"):
        if column not in result.columns:
            result[column] = "Unknown"
        result[column] = result[column].fillna("Unknown").replace("", "Unknown").astype(str)
    return result


def _attach_context_without_duplicates(rows: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in (
        "regime",
        "anomaly_score",
        "anomaly_level",
        "anomaly_types",
        "surge_risk_level",
        "crash_risk_level",
    ):
        if column in result.columns:
            result = result.drop(columns=[column])
    return _attach_context(result, feature_data)


def _attach_baseline_predictions(rows: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or feature_data.empty or not {"date", "ticker", "close"}.issubset(feature_data.columns):
        return rows.copy()
    market = feature_data[["date", "ticker", "close"]].copy()
    market["date"] = pd.to_datetime(market["date"], errors="coerce").dt.normalize()
    market["ticker"] = market["ticker"].astype(str)
    market["close"] = market["close"].map(safe_float)
    market = market.dropna(subset=["date", "ticker", "close"]).sort_values(["ticker", "date"])
    market["previous_return"] = market.groupby("ticker")["close"].pct_change()
    market["baseline_previous_day_direction"] = (market["previous_return"] > 0).astype(int)
    if {"ma_5", "ma_25"}.issubset(feature_data.columns):
        ma = feature_data[["date", "ticker", "ma_5", "ma_25"]].copy()
        ma["date"] = pd.to_datetime(ma["date"], errors="coerce").dt.normalize()
        ma["ticker"] = ma["ticker"].astype(str)
        ma["baseline_moving_average_trend"] = (
            pd.to_numeric(ma["ma_5"], errors="coerce") >= pd.to_numeric(ma["ma_25"], errors="coerce")
        ).astype(int)
        market = market.merge(ma[["date", "ticker", "baseline_moving_average_trend"]], on=["date", "ticker"], how="left")
    else:
        market["ma_5"] = market.groupby("ticker")["close"].transform(lambda values: values.rolling(5, min_periods=1).mean())
        market["ma_25"] = market.groupby("ticker")["close"].transform(
            lambda values: values.rolling(25, min_periods=1).mean()
        )
        market["baseline_moving_average_trend"] = (market["ma_5"] >= market["ma_25"]).astype(int)
    return rows.merge(
        market[["date", "ticker", "baseline_previous_day_direction", "baseline_moving_average_trend"]],
        on=["date", "ticker"],
        how="left",
    )


def _attach_baseline_predictions_without_duplicates(rows: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in ("baseline_previous_day_direction", "baseline_moving_average_trend"):
        if column in result.columns:
            result = result.drop(columns=[column])
    return _attach_baseline_predictions(result, feature_data)


def _normalize_prediction_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    for column in ("predicted_class", "actual_class"):
        if column in result.columns:
            result[column] = result[column].map(safe_int)
    for column in ("confidence", "probability_up", "probability_down", "actual_return"):
        if column not in result.columns:
            result[column] = None
        result[column] = result[column].map(safe_float)
    if "confidence" not in result.columns or result["confidence"].isna().all():
        probability_columns = [column for column in ("probability_up", "probability_down") if column in result.columns]
        if probability_columns:
            result["confidence"] = result[probability_columns].max(axis=1)
    if "prediction_type" not in result.columns:
        result["prediction_type"] = "live"
    result["prediction_type"] = result["prediction_type"].fillna("live").astype(str)
    if "created_at" not in result.columns:
        result["created_at"] = None
    result["created_at"] = pd.to_datetime(result["created_at"], errors="coerce")
    return result


def _context_accuracy_frame(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = _confirmed(frame)
    if rows.empty or group_column not in rows.columns:
        return _empty_context_frame(group_column)
    result = []
    for value, group in rows.groupby(group_column, dropna=False):
        metrics = _binary_metrics(group)
        predicted = group["predicted_class"].map(safe_int)
        actual = group["actual_class"].map(safe_int)
        up_mask = predicted == 1
        down_mask = predicted == 0
        result.append(
            {
                group_column: value,
                "sample_count": int(len(group)),
                "accuracy": metrics["accuracy"],
                "up_accuracy": _safe_divide(int((actual[up_mask] == 1).sum()), int(up_mask.sum())),
                "down_accuracy": _safe_divide(int((actual[down_mask] == 0).sum()), int(down_mask.sum())),
                "average_confidence": _mean(group.get("confidence")),
            }
        )
    return pd.DataFrame(result).sort_values(["sample_count", "accuracy"], ascending=[False, True]).reset_index(drop=True)


def _binary_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty:
        return {"accuracy": None, "precision": None, "recall": None}
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    valid = actual.notna() & predicted.notna()
    if not bool(valid.any()):
        return {"accuracy": None, "precision": None, "recall": None}
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    correct = int((actual == predicted).sum())
    count = int(len(actual))
    return {
        "accuracy": _safe_divide(correct, count),
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
    }


def _binary_extended_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    if rows.empty:
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "up_recall": None,
            "down_recall": None,
            "predicted_up_accuracy": None,
            "predicted_down_accuracy": None,
        }
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    valid = actual.notna() & predicted.notna()
    if not bool(valid.any()):
        return {
            "accuracy": None,
            "balanced_accuracy": None,
            "up_recall": None,
            "down_recall": None,
            "predicted_up_accuracy": None,
            "predicted_down_accuracy": None,
        }
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    up_actual = actual == 1
    down_actual = actual == 0
    up_predicted = predicted == 1
    down_predicted = predicted == 0
    up_recall = _safe_divide(int((predicted[up_actual] == 1).sum()), int(up_actual.sum()))
    down_recall = _safe_divide(int((predicted[down_actual] == 0).sum()), int(down_actual.sum()))
    return {
        "accuracy": _safe_divide(int((actual == predicted).sum()), len(actual)),
        "balanced_accuracy": (
            (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "predicted_up_accuracy": _safe_divide(int((actual[up_predicted] == 1).sum()), int(up_predicted.sum())),
        "predicted_down_accuracy": _safe_divide(int((actual[down_predicted] == 0).sum()), int(down_predicted.sum())),
    }


def _summary_columns(*, include_source: bool, label_column: str = "source") -> list[str]:
    columns = [
        label_column,
        "sample_count",
        "warning",
        "evaluated_start",
        "evaluated_end",
        "model_accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "predicted_up_accuracy",
        "predicted_down_accuracy",
        "actual_up_rate",
        "actual_down_rate",
        "predicted_up_rate",
        "predicted_down_rate",
        "average_confidence",
        "confidence_min",
        "confidence_max",
        "confidence_median",
        "confidence_ge_55_rate",
        "always_up",
        "always_down",
        "previous_day_direction",
        "moving_average_trend_only",
        "class_distribution_baseline",
    ]
    if include_source:
        return columns
    return [label_column, *columns[1:]]


def _summary_row(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    period: str | None = None,
    min_sample_warning: int,
) -> dict[str, object]:
    rows = _confirmed(frame)
    label_key = "source" if source is not None else "period"
    label_value = source if source is not None else period
    if rows.empty:
        return {column: None for column in _summary_columns(include_source=source is not None, label_column=label_key)} | {
            label_key: label_value,
            "sample_count": 0,
            "warning": "no confirmed rows",
        }

    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    valid = rows[actual.notna() & predicted.notna()].copy()
    start, end = _period(valid)
    metrics = _binary_extended_metrics(valid)
    actual = valid["actual_class"].map(safe_int).astype(int) if not valid.empty else pd.Series(dtype=int)
    predicted = valid["predicted_class"].map(safe_int).astype(int) if not valid.empty else pd.Series(dtype=int)
    baselines = baseline_accuracy_frame(valid).set_index("baseline") if not valid.empty else pd.DataFrame()
    confidence = pd.to_numeric(valid.get("confidence"), errors="coerce").dropna()
    sample_count = int(len(valid))
    row = {
        label_key: label_value,
        "sample_count": sample_count,
        "warning": "low sample" if sample_count and sample_count < min_sample_warning else "",
        "evaluated_start": start,
        "evaluated_end": end,
        "model_accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "up_recall": metrics["up_recall"],
        "down_recall": metrics["down_recall"],
        "predicted_up_accuracy": metrics["predicted_up_accuracy"],
        "predicted_down_accuracy": metrics["predicted_down_accuracy"],
        "actual_up_rate": _safe_divide(int((actual == 1).sum()), len(actual)) if len(actual) else None,
        "actual_down_rate": _safe_divide(int((actual == 0).sum()), len(actual)) if len(actual) else None,
        "predicted_up_rate": _safe_divide(int((predicted == 1).sum()), len(predicted)) if len(predicted) else None,
        "predicted_down_rate": _safe_divide(int((predicted == 0).sum()), len(predicted)) if len(predicted) else None,
        "average_confidence": float(confidence.mean()) if not confidence.empty else None,
        "confidence_min": float(confidence.min()) if not confidence.empty else None,
        "confidence_max": float(confidence.max()) if not confidence.empty else None,
        "confidence_median": float(confidence.median()) if not confidence.empty else None,
        "confidence_ge_55_rate": _safe_divide(int((confidence >= 0.55).sum()), len(confidence)) if len(confidence) else None,
    }
    for name in (
        "always_up",
        "always_down",
        "previous_day_direction",
        "moving_average_trend_only",
        "class_distribution_baseline",
    ):
        row[name] = (
            baselines.loc[name, "baseline_accuracy"]
            if not baselines.empty and name in baselines.index
            else None
        )
    return row


def _confirmed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "status" not in frame.columns:
        return frame.copy()
    return frame[frame["status"] == "confirmed"].copy()


def _random_baseline(actual: pd.Series, *, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    up_rate = float(actual.mean()) if len(actual) else 0.5
    return pd.Series((rng.random(len(actual)) < up_rate).astype(int), index=actual.index)


def _actual_binary_class(value) -> int | None:
    actual_return = safe_float(value)
    if actual_return is None:
        return None
    return int(actual_return > 0)


def _is_correct(row: pd.Series) -> bool | None:
    predicted = safe_int(row.get("predicted_class"))
    actual = safe_int(row.get("actual_class"))
    if predicted is None or actual is None:
        return None
    return predicted == actual


def _accuracy(actual: pd.Series, predicted: pd.Series) -> float | None:
    if len(actual) == 0:
        return None
    return float((actual.astype(int) == predicted.astype(int)).mean())


def _period(rows: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(rows.get("date", pd.Series(dtype=object)), errors="coerce").dropna()
    if dates.empty:
        return None, None
    return pd.Timestamp(dates.min()).strftime("%Y-%m-%d"), pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


def _mean(values) -> float | None:
    if values is None:
        return None
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return None
    return float(numbers.mean())


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _anomaly_level(row: pd.Series) -> str:
    existing = str(row.get("anomaly_level") or "").strip()
    if existing and existing.lower() != "unknown":
        return existing
    score = safe_float(row.get("anomaly_score")) or 0.0
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def _top_feature_text(value) -> str:
    if value in (None, ""):
        return ""
    if isinstance(value, list):
        items = value
    else:
        try:
            items = json.loads(str(value))
        except (TypeError, ValueError, json.JSONDecodeError):
            return str(value)[:160]
    if not isinstance(items, list):
        return str(value)[:160]
    names = []
    for item in items[:3]:
        if isinstance(item, dict):
            name = item.get("feature")
            if name:
                names.append(str(name))
        elif item:
            names.append(str(item))
    return ", ".join(names)


def _empty_binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "target",
            "model_name",
            "prediction_type",
            "created_at",
            "predicted_class",
            "confidence",
            "target_close_date",
            "actual_return",
            "actual_class",
            "status",
            "is_correct",
            "baseline_previous_day_direction",
            "baseline_moving_average_trend",
        ]
    )


def _empty_multiclass_frame() -> pd.DataFrame:
    return _empty_binary_frame()


def _empty_context_frame(group_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            group_column,
            "sample_count",
            "accuracy",
            "up_accuracy",
            "down_accuracy",
            "average_confidence",
        ]
    )
