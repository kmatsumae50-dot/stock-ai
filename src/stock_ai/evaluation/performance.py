from __future__ import annotations

import pandas as pd

from stock_ai.utils.safe_cast import safe_float, safe_int


CONTEXT_COLUMNS = [
    "regime",
    "surge_risk_level",
    "crash_risk_level",
    "anomaly_types",
    "dominant_narrative",
    "earnings_event_state",
    "negative_news_flag",
    "positive_news_flag",
    "importance_weighted_sentiment",
]
GROUP_LABELS = {
    "year": "Year",
    "month": "Month",
    "regime": "Regime",
    "surge_risk_level": "Surge Risk",
    "crash_risk_level": "Crash Risk",
    "anomaly_type": "Anomaly Type",
    "dominant_narrative": "Narrative",
    "earnings_event_state": "Earnings Event",
    "news_sentiment_bucket": "News Sentiment",
    "confidence_bucket": "Confidence",
}


def prepare_backtest_evaluation_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return _empty_eval_frame()
    rows = predictions.copy()
    rows = _normalize_common_columns(rows)
    rows = rows[rows["actual_class"].isin([0, 1]) & rows["predicted_class"].isin([0, 1])]
    return _add_evaluation_dimensions(rows)


def prepare_prediction_history_evaluation_frame(
    history: pd.DataFrame,
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        return _empty_eval_frame()

    rows = history.copy()
    if "status" in rows.columns:
        rows = rows[rows["status"] == "confirmed"].copy()
    if rows.empty:
        return _empty_eval_frame()

    rows = _join_context(rows, feature_data)
    rows = _normalize_common_columns(rows)
    rows = rows[rows["actual_class"].isin([0, 1]) & rows["predicted_class"].isin([0, 1])]
    rows["trade"] = rows["predicted_class"] == 1
    rows["strategy_return"] = rows["actual_return"].where(rows["trade"], 0.0).fillna(0.0)
    rows["buy_hold_return"] = rows["actual_return"].fillna(0.0)
    return _add_evaluation_dimensions(rows)


def build_evaluation_summaries(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if frame.empty:
        return {key: _empty_summary(key) for key in GROUP_LABELS}

    summaries: dict[str, pd.DataFrame] = {
        "year": summarize_by_group(frame, "year"),
        "month": summarize_by_group(frame, "month"),
        "regime": summarize_by_group(frame, "regime"),
        "surge_risk_level": summarize_by_group(frame, "surge_risk_level"),
        "crash_risk_level": summarize_by_group(frame, "crash_risk_level"),
        "dominant_narrative": summarize_by_group(frame, "dominant_narrative"),
        "earnings_event_state": summarize_by_group(frame, "earnings_event_state"),
        "news_sentiment_bucket": summarize_by_group(frame, "news_sentiment_bucket"),
        "confidence_bucket": summarize_by_group(frame, "confidence_bucket"),
    }
    summaries["anomaly_type"] = summarize_by_group(_explode_anomaly_types(frame), "anomaly_type")
    return summaries


def summarize_by_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if frame.empty or group_column not in frame.columns:
        return _empty_summary(group_column)

    rows: list[dict[str, object]] = []
    for group, group_rows in frame.groupby(group_column, dropna=False):
        metrics = summarize_rows(group_rows)
        rows.append({group_column: group, **metrics})
    if not rows:
        return _empty_summary(group_column)
    result = pd.DataFrame(rows)
    if group_column in {"year", "month", "confidence_bucket"}:
        return result.sort_values(group_column).reset_index(drop=True)
    return result.sort_values(["count", "accuracy"], ascending=[False, True]).reset_index(drop=True)


def summarize_rows(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return _empty_metric_values()

    actual = rows["actual_class"].map(safe_int).dropna().astype(int)
    predicted = rows.loc[actual.index, "predicted_class"].map(safe_int).astype(int)
    probability_up = rows.loc[actual.index, "probability_up"].map(safe_float)
    tn, fp, fn, tp = _confusion_counts(actual, predicted)
    count = len(actual)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = (
        _safe_divide(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    returns = _return_series(rows)
    trade_rows = rows[pd.Series(rows["trade"], index=rows.index).astype(bool)] if "trade" in rows.columns else rows
    return {
        "count": int(count),
        "accuracy": _safe_divide(tp + tn, count),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auc": _auc(actual, probability_up),
        "average_return": _mean_or_none(returns),
        "cumulative_return": _compound_return(returns),
        "win_rate": _win_rate(trade_rows),
        "max_drawdown": _max_drawdown(returns),
    }


def build_weakness_summary(
    summaries: dict[str, pd.DataFrame],
    overall: dict[str, object],
    min_count: int = 10,
) -> list[str]:
    messages: list[str] = []
    overall_accuracy = safe_float(overall.get("accuracy"))
    if overall_accuracy is None:
        return ["Not enough evaluated rows to summarize strengths and weaknesses yet."]

    for key in (
        "regime",
        "surge_risk_level",
        "crash_risk_level",
        "anomaly_type",
        "dominant_narrative",
        "earnings_event_state",
        "news_sentiment_bucket",
        "confidence_bucket",
    ):
        summary = summaries.get(key)
        if summary is None or summary.empty or key not in summary.columns:
            continue
        eligible = summary[pd.to_numeric(summary["count"], errors="coerce") >= min_count].copy()
        if eligible.empty:
            continue
        eligible["accuracy"] = pd.to_numeric(eligible["accuracy"], errors="coerce")
        eligible["cumulative_return"] = pd.to_numeric(eligible["cumulative_return"], errors="coerce")
        worst_accuracy = eligible.dropna(subset=["accuracy"]).sort_values("accuracy").head(1)
        if not worst_accuracy.empty:
            row = worst_accuracy.iloc[0]
            accuracy = safe_float(row.get("accuracy"))
            if accuracy is not None and accuracy <= min(0.5, overall_accuracy - 0.08):
                messages.append(
                    f"{GROUP_LABELS.get(key, key)} '{row.get(key)}' is a weak condition: "
                    f"accuracy {_format_percent(accuracy)} over {int(row.get('count'))} rows."
                )
        worst_return = eligible.dropna(subset=["cumulative_return"]).sort_values("cumulative_return").head(1)
        if not worst_return.empty:
            row = worst_return.iloc[0]
            cumulative_return = safe_float(row.get("cumulative_return"))
            if cumulative_return is not None and cumulative_return < -0.03:
                messages.append(
                    f"{GROUP_LABELS.get(key, key)} '{row.get(key)}' has weak simulated returns: "
                    f"cumulative return {_format_percent(cumulative_return)}."
                )
        precision_values = (
            pd.to_numeric(eligible["precision"], errors="coerce")
            if "precision" in eligible.columns
            else pd.Series(index=eligible.index, dtype=float)
        )
        strong_precision = eligible[precision_values >= 0.65]
        if key in {"crash_risk_level", "regime"} and not strong_precision.empty:
            row = strong_precision.sort_values("precision", ascending=False).iloc[0]
            messages.append(
                f"{GROUP_LABELS.get(key, key)} '{row.get(key)}' shows relatively strong up-prediction precision "
                f"({_format_percent(row.get('precision'))})."
            )

    if not messages:
        messages.append("No obvious weak condition is detected yet. More evaluated rows may be needed.")
    return messages[:6]


def _normalize_common_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in ("predicted_class", "actual_class"):
        if column in result.columns:
            result[column] = result[column].map(safe_int)
    for column in ("probability_up", "probability_down", "confidence", "actual_return"):
        if column not in result.columns:
            result[column] = None
        result[column] = result[column].map(safe_float)
    if "confidence" not in result.columns or result["confidence"].isna().all():
        result["confidence"] = result[["probability_up", "probability_down"]].max(axis=1)
    if "strategy_return" not in result.columns:
        trade = result.get("trade")
        if trade is None:
            trade = result["predicted_class"] == 1
        result["strategy_return"] = result["actual_return"].where(trade, 0.0).fillna(0.0)
    else:
        result["strategy_return"] = result["strategy_return"].map(safe_float)
    if "buy_hold_return" not in result.columns:
        result["buy_hold_return"] = result["actual_return"].fillna(0.0)
    if "trade" not in result.columns:
        result["trade"] = result["predicted_class"] == 1
    return result


def _add_evaluation_dimensions(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    result["year"] = result["date"].dt.year.astype(str)
    result["month"] = result["date"].dt.to_period("M").astype(str)
    for column in (
        "regime",
        "surge_risk_level",
        "crash_risk_level",
        "anomaly_types",
        "dominant_narrative",
        "earnings_event_state",
    ):
        if column not in result.columns:
            result[column] = "Unknown"
        result[column] = result[column].fillna("Unknown").replace("", "Unknown").astype(str)
    result["news_sentiment_bucket"] = result.apply(_news_sentiment_bucket, axis=1)
    result["confidence_bucket"] = pd.cut(
        pd.to_numeric(result["confidence"], errors="coerce"),
        bins=[-float("inf"), 0.5, 0.6, 0.7, 0.8, 0.9, float("inf")],
        labels=["<50%", "50-60%", "60-70%", "70-80%", "80-90%", "90%+"],
        right=False,
    ).astype(str)
    result["confidence_bucket"] = result["confidence_bucket"].replace("nan", "Unknown")
    return result


def _join_context(history: pd.DataFrame, feature_data: pd.DataFrame) -> pd.DataFrame:
    if feature_data.empty or "date" not in feature_data.columns:
        return history.copy()
    context_columns = ["date", "ticker", *[column for column in CONTEXT_COLUMNS if column in feature_data.columns]]
    context = feature_data[context_columns].copy()
    context["date"] = pd.to_datetime(context["date"], errors="coerce").dt.normalize()
    if "ticker" in context.columns:
        context["ticker"] = context["ticker"].astype(str)
    rows = history.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    rows["ticker"] = rows["ticker"].astype(str)
    return rows.merge(context, on=["date", "ticker"], how="left")


def _explode_anomaly_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "anomaly_types" not in frame.columns:
        return pd.DataFrame(columns=[*frame.columns, "anomaly_type"])
    rows = frame.copy()
    rows["anomaly_type"] = rows["anomaly_types"].map(_split_anomaly_types)
    rows = rows.explode("anomaly_type")
    rows["anomaly_type"] = rows["anomaly_type"].fillna("None").replace("", "None")
    return rows


def _split_anomaly_types(value) -> list[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return ["None"]
    return [part.strip() for part in str(value).split("|") if part.strip()] or ["None"]


def _news_sentiment_bucket(row: pd.Series) -> str:
    if bool(row.get("negative_news_flag")):
        return "Negative"
    if bool(row.get("positive_news_flag")):
        return "Positive"
    weighted = safe_float(row.get("importance_weighted_sentiment"))
    if weighted is None:
        return "Unknown"
    if weighted <= -0.25:
        return "Negative"
    if weighted >= 0.25:
        return "Positive"
    return "Neutral"


def _return_series(rows: pd.DataFrame) -> pd.Series:
    column = "strategy_return" if "strategy_return" in rows.columns else "actual_return"
    return pd.to_numeric(rows[column], errors="coerce").fillna(0.0)


def _confusion_counts(actual: pd.Series, predicted: pd.Series) -> tuple[int, int, int, int]:
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())
    return tn, fp, fn, tp


def _auc(actual: pd.Series, probability: pd.Series) -> float | None:
    rows = pd.DataFrame({"actual": actual, "probability": probability}).dropna()
    if rows.empty or rows["actual"].nunique() < 2:
        return None
    positives = rows[rows["actual"] == 1]
    negatives = rows[rows["actual"] == 0]
    if positives.empty or negatives.empty:
        return None
    ranks = rows["probability"].rank(method="average")
    positive_rank_sum = float(ranks[rows["actual"] == 1].sum())
    n_pos = len(positives)
    n_neg = len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _compound_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if clean.empty:
        return 0.0
    return float((1 + clean).prod() - 1)


def _win_rate(rows: pd.DataFrame) -> float | None:
    if rows.empty:
        return None
    returns = _return_series(rows)
    if returns.empty:
        return None
    return float((returns > 0).mean())


def _max_drawdown(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if clean.empty:
        return None
    equity = (1 + clean).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _empty_eval_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "predicted_class",
            "actual_class",
            "probability_up",
            "confidence",
            "actual_return",
            "strategy_return",
            "trade",
        ]
    )


def _empty_summary(group_column: str) -> pd.DataFrame:
    return pd.DataFrame(columns=[group_column, *_empty_metric_values().keys()])


def _empty_metric_values() -> dict[str, object]:
    return {
        "count": 0,
        "accuracy": None,
        "f1": None,
        "precision": None,
        "recall": None,
        "auc": None,
        "average_return": None,
        "cumulative_return": None,
        "win_rate": None,
        "max_drawdown": None,
    }
