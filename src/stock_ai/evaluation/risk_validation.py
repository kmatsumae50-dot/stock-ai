from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ai.labels.targets import CLASS_NAMES
from stock_ai.utils.safe_cast import safe_float, safe_int


RISK_BUCKET_ORDER = ["Low", "Medium", "High"]
DEFAULT_BUCKET_MODE = "fixed_33_66"
BUCKET_MODE_DEFINITIONS = {
    "fixed_33_66": {
        "name": "Fixed 33 / 66",
        "description": "Low 0-33, Medium 34-66, High 67-100.",
        "interpretability": "Absolute threshold. Easy to read, but can be too conservative.",
        "low_quantile": None,
        "high_quantile": None,
    },
    "quantile_50_85": {
        "name": "Quantile 50 / 85",
        "description": "Low bottom 50%, Medium 50-85%, High top 15%.",
        "interpretability": "Balanced warning buckets. High usually has enough samples.",
        "low_quantile": 0.50,
        "high_quantile": 0.85,
    },
    "quantile_60_90": {
        "name": "Quantile 60 / 90",
        "description": "Low bottom 60%, Medium 60-90%, High top 10%.",
        "interpretability": "Stricter warning bucket. High is more selective but still data-driven.",
        "low_quantile": 0.60,
        "high_quantile": 0.90,
    },
}


@dataclass(frozen=True)
class RiskValidationConfig:
    min_history_rows: int = 60
    forward_days: int = 20
    surge_return_threshold: float = 0.03
    crash_return_threshold: float = -0.03


def risk_bucket(score) -> str:
    value = safe_float(score)
    if value is None:
        return "Unknown"
    if value >= 67:
        return "High"
    if value >= 34:
        return "Medium"
    return "Low"


def bucket_mode_options() -> list[str]:
    return list(BUCKET_MODE_DEFINITIONS)


def bucket_mode_description(bucket_mode: str) -> str:
    definition = BUCKET_MODE_DEFINITIONS.get(normalize_bucket_mode(bucket_mode), {})
    return str(definition.get("description") or "")


def normalize_bucket_mode(bucket_mode: str | None) -> str:
    if bucket_mode is None or not str(bucket_mode).strip():
        return DEFAULT_BUCKET_MODE
    value = str(bucket_mode).strip()
    aliases = {
        "fixed": "fixed_33_66",
        "quantile": "quantile_50_85",
        "q50_85": "quantile_50_85",
        "q60_90": "quantile_60_90",
    }
    value = aliases.get(value, value)
    if value not in BUCKET_MODE_DEFINITIONS:
        raise ValueError(
            f"Unknown risk bucket mode: {bucket_mode}. Valid modes: {bucket_mode_options()}"
        )
    return value


def apply_bucket_mode(validation: pd.DataFrame, bucket_mode: str | None = None) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    mode = normalize_bucket_mode(bucket_mode)
    frame = validation.copy()
    for risk_type in ("surge", "crash"):
        score_column = f"{risk_type}_risk_score"
        bucket_column = f"{risk_type}_risk_bucket"
        if score_column not in frame.columns:
            frame[bucket_column] = "Unknown"
            continue
        if mode == "fixed_33_66":
            frame[bucket_column] = frame[score_column].map(risk_bucket)
        else:
            frame[bucket_column] = _quantile_buckets(
                frame[score_column],
                low_quantile=float(BUCKET_MODE_DEFINITIONS[mode]["low_quantile"]),
                high_quantile=float(BUCKET_MODE_DEFINITIONS[mode]["high_quantile"]),
            )
    frame["risk_bucket_mode"] = mode
    return frame


def build_risk_validation_frame(
    data: pd.DataFrame,
    *,
    config: RiskValidationConfig | None = None,
    multiclass_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Recompute historical Risk Scores without using future rows.

    The historical MVP validates the rule-based risk core. Saved multiclass
    prediction rows can be merged by date, but no current model is applied to
    historical rows because that would use a future-trained model.
    """
    active_config = config or RiskValidationConfig()
    frame = _prepare_frame(data)
    if frame.empty:
        return pd.DataFrame()

    frame = _add_forward_returns(frame, active_config.forward_days)
    frame = _add_rule_based_risk_scores(frame)
    prediction_lookup = _multiclass_prediction_lookup(multiclass_predictions)
    rows: list[dict[str, object]] = []

    for _ticker, group in frame.groupby("ticker", sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        for position, row in group.iterrows():
            if position < active_config.min_history_rows:
                continue
            if pd.isna(row.get("return_1d")):
                continue

            surge_score = safe_int(row.get("surge_risk_score_core")) or 0
            crash_score = safe_int(row.get("crash_risk_score_core")) or 0

            prediction = prediction_lookup.get(
                (
                    str(row.get("ticker")),
                    pd.to_datetime(row.get("date")).strftime("%Y-%m-%d"),
                )
            )
            if prediction is not None:
                surge_score, crash_score = _add_multiclass_signal_scores(
                    surge_score,
                    crash_score,
                    prediction,
                )

            rows.append(
                {
                    "date": row.get("date"),
                    "ticker": row.get("ticker"),
                    "surge_risk_score": min(100, surge_score),
                    "crash_risk_score": min(100, crash_score),
                    "surge_risk_bucket": risk_bucket(surge_score),
                    "crash_risk_bucket": risk_bucket(crash_score),
                    "return_1d": row.get("return_1d"),
                    "return_5d": row.get("return_5d"),
                    "return_20d": row.get("return_20d"),
                    "future_min_return_20d": row.get("future_min_return_20d"),
                    "future_max_return_20d": row.get("future_max_return_20d"),
                    "actual_down_1d": bool((safe_float(row.get("return_1d")) or 0.0) < 0),
                    "actual_up_1d": bool((safe_float(row.get("return_1d")) or 0.0) > 0),
                    "actual_crash_1d": bool(
                        (safe_float(row.get("return_1d")) or 0.0)
                        <= active_config.crash_return_threshold
                    ),
                    "actual_surge_1d": bool(
                        (safe_float(row.get("return_1d")) or 0.0)
                        >= active_config.surge_return_threshold
                    ),
                    "multiclass_signal_used": prediction is not None,
                }
            )

    return apply_bucket_mode(pd.DataFrame(rows), DEFAULT_BUCKET_MODE)


def summarize_risk_buckets(
    validation: pd.DataFrame,
    *,
    risk_type: str,
    bucket_mode: str | None = None,
) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    if risk_type not in {"surge", "crash"}:
        raise ValueError("risk_type must be 'surge' or 'crash'.")

    frame = apply_bucket_mode(validation, bucket_mode)
    bucket_column = f"{risk_type}_risk_bucket"
    score_column = f"{risk_type}_risk_score"
    event_column = "actual_surge_1d" if risk_type == "surge" else "actual_crash_1d"
    direction_column = "actual_up_1d" if risk_type == "surge" else "actual_down_1d"
    extreme_column = "future_max_return_20d" if risk_type == "surge" else "future_min_return_20d"
    extreme_name = "max_upside_20d" if risk_type == "surge" else "max_drawdown_20d"

    rows: list[dict[str, object]] = []
    for bucket in RISK_BUCKET_ORDER:
        subset = frame[frame[bucket_column] == bucket].copy()
        rows.append(
            {
                "bucket": bucket,
                "count": int(len(subset)),
                "average_score": _mean(subset.get(score_column)),
                "average_return_1d": _mean(subset.get("return_1d")),
                "average_return_5d": _mean(subset.get("return_5d")),
                "average_return_20d": _mean(subset.get("return_20d")),
                "direction_rate": _mean(subset.get(direction_column)),
                "event_rate": _mean(subset.get(event_column)),
                extreme_name: _extreme(subset.get(extreme_column), risk_type=risk_type),
                "multiclass_signal_days": int(subset.get("multiclass_signal_used", pd.Series()).sum())
                if not subset.empty
                else 0,
            }
        )
    return pd.DataFrame(rows)


def high_risk_days(
    validation: pd.DataFrame,
    *,
    risk_type: str,
    min_score: int = 67,
    bucket_mode: str | None = None,
    limit: int = 30,
) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    score_column = f"{risk_type}_risk_score"
    bucket_column = f"{risk_type}_risk_bucket"
    if score_column not in validation.columns:
        return pd.DataFrame()
    mode = normalize_bucket_mode(bucket_mode)
    frame = apply_bucket_mode(validation, mode)
    if mode == "fixed_33_66":
        result = frame[pd.to_numeric(frame[score_column], errors="coerce") >= min_score].copy()
    else:
        result = frame[frame[bucket_column] == "High"].copy()
    if result.empty:
        return result
    return result.sort_values(["date", score_column], ascending=[False, False]).head(limit)


def compare_bucket_modes(validation: pd.DataFrame, *, risk_type: str) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for mode in bucket_mode_options():
        summary = summarize_risk_buckets(validation, risk_type=risk_type, bucket_mode=mode)
        high = _summary_bucket_row(summary, "High")
        medium = _summary_bucket_row(summary, "Medium")
        if high is None:
            high = pd.Series(dtype=object)
        if medium is None:
            medium = pd.Series(dtype=object)
        rows.append(
            {
                "bucket_mode": mode,
                "description": bucket_mode_description(mode),
                "high_count": safe_int(high.get("count")) or 0,
                "medium_count": safe_int(medium.get("count")) or 0,
                "high_event_rate": safe_float(high.get("event_rate")),
                "high_average_return_5d": safe_float(high.get("average_return_5d")),
                "high_average_return_20d": safe_float(high.get("average_return_20d")),
                "high_extreme_20d": safe_float(
                    high.get("max_upside_20d")
                    if risk_type == "surge"
                    else high.get("max_drawdown_20d")
                ),
                "interpretability": BUCKET_MODE_DEFINITIONS[mode]["interpretability"],
            }
        )
    return pd.DataFrame(rows)


def risk_score_distribution(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for risk_type in ("surge", "crash"):
        column = f"{risk_type}_risk_score"
        if column not in validation.columns:
            continue
        scores = pd.to_numeric(validation[column], errors="coerce").dropna()
        rows.append(
            {
                "risk_type": risk_type,
                "count": int(len(scores)),
                "mean": float(scores.mean()) if not scores.empty else None,
                "max": float(scores.max()) if not scores.empty else None,
                "p50": _quantile_value(scores, 0.50),
                "p60": _quantile_value(scores, 0.60),
                "p85": _quantile_value(scores, 0.85),
                "p90": _quantile_value(scores, 0.90),
                "p95": _quantile_value(scores, 0.95),
            }
        )
    return pd.DataFrame(rows)


def latest_risk_percentile(
    validation: pd.DataFrame,
    *,
    risk_type: str,
) -> float | None:
    if validation.empty:
        return None
    score_column = f"{risk_type}_risk_score"
    if score_column not in validation.columns:
        return None
    clean = validation.dropna(subset=[score_column]).copy()
    if clean.empty:
        return None
    latest = clean.sort_values("date").iloc[-1]
    latest_score = safe_float(latest.get(score_column))
    scores = pd.to_numeric(clean[score_column], errors="coerce").dropna()
    if latest_score is None or scores.empty:
        return None
    return float((scores <= latest_score).mean())


def score_percentile(
    validation: pd.DataFrame,
    *,
    risk_type: str,
    score,
) -> float | None:
    if validation.empty:
        return None
    score_column = f"{risk_type}_risk_score"
    if score_column not in validation.columns:
        return None
    value = safe_float(score)
    scores = pd.to_numeric(validation[score_column], errors="coerce").dropna()
    if value is None or scores.empty:
        return None
    return float((scores <= value).mean())


def bucket_for_score(
    validation: pd.DataFrame,
    *,
    risk_type: str,
    score,
    bucket_mode: str | None = None,
) -> str:
    mode = normalize_bucket_mode(bucket_mode)
    value = safe_float(score)
    if value is None:
        return "Unknown"
    if mode == "fixed_33_66":
        return risk_bucket(value)
    percentile = score_percentile(validation, risk_type=risk_type, score=value)
    if percentile is None:
        return "Unknown"
    definition = BUCKET_MODE_DEFINITIONS[mode]
    low_quantile = float(definition["low_quantile"])
    high_quantile = float(definition["high_quantile"])
    if percentile <= low_quantile:
        return "Low"
    if percentile <= high_quantile:
        return "Medium"
    return "High"


def build_risk_validation_messages(
    surge_summary: pd.DataFrame,
    crash_summary: pd.DataFrame,
) -> list[str]:
    messages: list[str] = []
    crash_high = _summary_bucket_row(crash_summary, "High")
    surge_high = _summary_bucket_row(surge_summary, "High")

    if crash_high is None or int(crash_high.get("count", 0)) == 0:
        messages.append("Crash Risk High has no historical samples in this MVP validation window.")
    else:
        count = int(crash_high.get("count", 0))
        avg_5d = safe_float(crash_high.get("average_return_5d"))
        event_rate = safe_float(crash_high.get("event_rate"))
        messages.append(
            f"Crash Risk High has {count} samples; average 5D return is "
            f"{_format_percent(avg_5d)} and next-day crash event rate is {_format_percent(event_rate)}."
        )

    if surge_high is None or int(surge_high.get("count", 0)) == 0:
        messages.append("Surge Risk High has no historical samples in this MVP validation window.")
    else:
        count = int(surge_high.get("count", 0))
        avg_5d = safe_float(surge_high.get("average_return_5d"))
        event_rate = safe_float(surge_high.get("event_rate"))
        messages.append(
            f"Surge Risk High has {count} samples; average 5D return is "
            f"{_format_percent(avg_5d)} and next-day surge event rate is {_format_percent(event_rate)}."
        )

    if messages:
        messages.append(
            "MVP note: historical validation avoids applying a future-trained multiclass model to old dates."
        )
    return messages


def _prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "date" not in data.columns or "close" not in data.columns:
        return pd.DataFrame()
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    if "ticker" not in frame.columns:
        frame["ticker"] = "UNKNOWN"
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame.dropna(subset=["date", "close"]).sort_values(["ticker", "date"])
    return frame.reset_index(drop=True)


def _quantile_buckets(
    scores: pd.Series,
    *,
    low_quantile: float,
    high_quantile: float,
) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    result = pd.Series("Unknown", index=scores.index, dtype="object")
    valid = numeric.dropna()
    if valid.empty:
        return result
    percentiles = valid.rank(method="first", pct=True)
    result.loc[percentiles.index[percentiles <= low_quantile]] = "Low"
    result.loc[percentiles.index[(percentiles > low_quantile) & (percentiles <= high_quantile)]] = "Medium"
    result.loc[percentiles.index[percentiles > high_quantile]] = "High"
    return result


def _quantile_value(scores: pd.Series, value: float) -> float | None:
    clean = pd.to_numeric(scores, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.quantile(value))


def _add_forward_returns(data: pd.DataFrame, forward_days: int) -> pd.DataFrame:
    groups = []
    for _ticker, group in data.groupby("ticker", sort=False):
        item = group.sort_values("date").copy()
        close = pd.to_numeric(item["close"], errors="coerce")
        for horizon in (1, 5, 20):
            item[f"return_{horizon}d"] = close.shift(-horizon) / close - 1
        max_returns: list[float | None] = []
        min_returns: list[float | None] = []
        values = close.to_list()
        for index, close_value in enumerate(values):
            current = safe_float(close_value)
            future = [safe_float(value) for value in values[index + 1 : index + 1 + forward_days]]
            future = [value for value in future if value is not None]
            if current is None or current == 0 or not future:
                max_returns.append(None)
                min_returns.append(None)
                continue
            forward_returns = [(value / current) - 1 for value in future]
            max_returns.append(max(forward_returns))
            min_returns.append(min(forward_returns))
        item["future_max_return_20d"] = max_returns
        item["future_min_return_20d"] = min_returns
        groups.append(item)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


def _add_rule_based_risk_scores(data: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _ticker, group in data.groupby("ticker", sort=False):
        item = group.sort_values("date").copy()
        item = _coerce_numeric_columns(
            item,
            [
                "volume",
                "volume_change_1d",
                "rsi_14",
                "macd",
                "macd_signal",
                "macd_hist",
                "volatility_20",
                "positive_news_shock",
                "negative_news_shock",
                "importance_weighted_sentiment",
                "sentiment_trend_3d",
                "sector_news_strength",
                "narrative_bullish_pressure",
                "narrative_bearish_pressure",
                "dominant_narrative_sentiment",
                "dominant_narrative_importance",
            ],
        )

        surge = pd.Series(0, index=item.index, dtype="int64")
        crash = pd.Series(0, index=item.index, dtype="int64")

        volume_spike = _volume_spike_series(item)
        surge += volume_spike.astype(int) * 20
        crash += volume_spike.astype(int) * 15

        surge += _rsi_up_series(item).astype(int) * 15
        crash += _rsi_down_series(item).astype(int) * 15
        surge += _macd_improving_series(item).astype(int) * 15
        crash += _macd_worsening_series(item).astype(int) * 15
        crash += _volatility_spike_series(item).astype(int) * 20

        surge += _positive_news_shock_series(item).astype(int) * 20
        crash += _negative_news_shock_series(item).astype(int) * 20
        surge += _positive_news_context_series(item).astype(int) * 10
        crash += _negative_news_context_series(item).astype(int) * 10
        surge += _positive_narrative_context_series(item).astype(int) * 10
        crash += _negative_narrative_context_series(item).astype(int) * 10

        item["surge_risk_score_core"] = surge.clip(lower=0, upper=100)
        item["crash_risk_score_core"] = crash.clip(lower=0, upper=100)
        groups.append(item)
    return pd.concat(groups, ignore_index=True) if groups else pd.DataFrame()


def _coerce_numeric_columns(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    frame = data.copy()
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _volume_spike_series(data: pd.DataFrame) -> pd.Series:
    direct = _series(data, "volume_change_1d") >= 0.30
    if "volume" not in data.columns:
        return direct.fillna(False)
    volume = _series(data, "volume")
    recent_average = volume.rolling(20, min_periods=20).mean()
    relative = volume >= recent_average * 1.5
    return (direct | relative).fillna(False)


def _volatility_spike_series(data: pd.DataFrame) -> pd.Series:
    volatility = _series(data, "volatility_20")
    baseline = volatility.rolling(60, min_periods=10).median()
    return ((baseline > 0) & (volatility >= baseline * 1.25)).fillna(False)


def _rsi_up_series(data: pd.DataFrame) -> pd.Series:
    rsi = _series(data, "rsi_14")
    return ((rsi >= 60) | ((rsi - rsi.shift(5)) >= 5)).fillna(False)


def _rsi_down_series(data: pd.DataFrame) -> pd.Series:
    rsi = _series(data, "rsi_14")
    return ((rsi <= 40) | ((rsi.shift(5) - rsi) >= 5)).fillna(False)


def _macd_improving_series(data: pd.DataFrame) -> pd.Series:
    hist = _series(data, "macd_hist")
    if hist.notna().any():
        return ((hist > 0) | (hist > hist.shift(3))).fillna(False)
    macd = _series(data, "macd")
    signal = _series(data, "macd_signal")
    return (macd > signal).fillna(False)


def _macd_worsening_series(data: pd.DataFrame) -> pd.Series:
    hist = _series(data, "macd_hist")
    if hist.notna().any():
        return ((hist < 0) | (hist < hist.shift(3))).fillna(False)
    macd = _series(data, "macd")
    signal = _series(data, "macd_signal")
    return (macd < signal).fillna(False)


def _positive_news_shock_series(data: pd.DataFrame) -> pd.Series:
    return (_series(data, "positive_news_shock") >= 1.0).fillna(False)


def _negative_news_shock_series(data: pd.DataFrame) -> pd.Series:
    return (_series(data, "negative_news_shock") >= 1.0).fillna(False)


def _positive_news_context_series(data: pd.DataFrame) -> pd.Series:
    weighted = _series(data, "importance_weighted_sentiment")
    trend = _series(data, "sentiment_trend_3d")
    sector = _series(data, "sector_news_strength")
    return ((weighted >= 0.25) | (trend >= 0.25) | (sector >= 0.25)).fillna(False)


def _negative_news_context_series(data: pd.DataFrame) -> pd.Series:
    weighted = _series(data, "importance_weighted_sentiment")
    trend = _series(data, "sentiment_trend_3d")
    sector = _series(data, "sector_news_strength")
    return ((weighted <= -0.25) | (trend <= -0.25) | (sector <= -0.25)).fillna(False)


def _positive_narrative_context_series(data: pd.DataFrame) -> pd.Series:
    pressure = _series(data, "narrative_bullish_pressure")
    sentiment = _series(data, "dominant_narrative_sentiment")
    importance = _series(data, "dominant_narrative_importance")
    return ((pressure >= 0.35) | ((sentiment >= 0.25) & (importance >= 0.5))).fillna(False)


def _negative_narrative_context_series(data: pd.DataFrame) -> pd.Series:
    pressure = _series(data, "narrative_bearish_pressure")
    sentiment = _series(data, "dominant_narrative_sentiment")
    importance = _series(data, "dominant_narrative_importance")
    return ((pressure >= 0.35) | ((sentiment <= -0.25) & (importance >= 0.5))).fillna(False)


def _series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(float("nan"), index=data.index, dtype="float64")
    return pd.to_numeric(data[column], errors="coerce")


def _multiclass_prediction_lookup(predictions: pd.DataFrame | None) -> dict[tuple[str, str], pd.Series]:
    if predictions is None or predictions.empty:
        return {}
    frame = predictions.copy()
    if "date" not in frame.columns or "ticker" not in frame.columns:
        return {}
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.dropna(subset=["date", "ticker"])
    lookup: dict[tuple[str, str], pd.Series] = {}
    for _, row in frame.iterrows():
        lookup[(str(row.get("ticker")), str(row.get("date")))] = row
    return lookup


def _add_multiclass_signal_scores(
    surge_score: int,
    crash_score: int,
    prediction: pd.Series,
) -> tuple[int, int]:
    predicted_class = safe_int(prediction.get("predicted_class"))
    predicted_label = CLASS_NAMES.get(predicted_class, "unknown")

    surge_probability = safe_float(prediction.get("probability_surge"))
    up_probability = safe_float(prediction.get("probability_up"))
    crash_probability = safe_float(prediction.get("probability_crash"))
    down_probability = safe_float(prediction.get("probability_down"))

    surge_score += _tail_probability_points(surge_probability)
    crash_score += _tail_probability_points(crash_probability)
    if predicted_label == "surge":
        surge_score += 15
    if predicted_label == "crash":
        crash_score += 15
    surge_score += _direction_probability_points(up_probability)
    crash_score += _direction_probability_points(down_probability)
    return min(100, surge_score), min(100, crash_score)


def _tail_probability_points(value: float | None) -> int:
    if value is None or value < 0.20:
        return 0
    return 20 if value >= 0.35 else 10


def _direction_probability_points(value: float | None) -> int:
    if value is None or value < 0.35:
        return 0
    return 10 if value >= 0.45 else 5


def _mean(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _extreme(values, *, risk_type: str) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max() if risk_type == "surge" else numeric.min())


def _summary_bucket_row(summary: pd.DataFrame, bucket: str) -> pd.Series | None:
    if summary.empty or "bucket" not in summary.columns:
        return None
    rows = summary[summary["bucket"] == bucket]
    if rows.empty:
        return None
    return rows.iloc[0]


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"
