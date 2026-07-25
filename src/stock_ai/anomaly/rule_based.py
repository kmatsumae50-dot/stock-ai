from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from stock_ai.utils.safe_cast import safe_float


ANOMALY_COMPONENTS = (
    "price_shock_score",
    "volume_spike_score",
    "volatility_spike_score",
    "sector_divergence_score",
    "market_divergence_score",
    "news_shock_score",
    "narrative_shift_score",
)
ANOMALY_TYPE_LABELS = {
    "price_shock_score": "Price Shock",
    "volume_spike_score": "Volume Spike",
    "volatility_spike_score": "Volatility Spike",
    "sector_divergence_score": "Sector Divergence",
    "market_divergence_score": "Market Divergence",
    "news_shock_score": "News Shock",
    "narrative_shift_score": "Narrative Shift",
}
ANOMALY_FEATURES = {
    "price_shock_score": ("return_1d", "rsi_14", "macd_hist"),
    "volume_spike_score": ("volume", "volume_change_1d"),
    "volatility_spike_score": ("volatility_20",),
    "sector_divergence_score": (
        "sector_mean_return",
        "honda_return_1d",
        "nissan_return_1d",
        "tesla_return_1d",
        "byd_return_1d",
    ),
    "market_divergence_score": ("n225_return_1d", "topix_return_1d", "usdjpy_return_1d"),
    "news_shock_score": (
        "max_importance_3d",
        "negative_news_shock",
        "positive_news_shock",
        "news_count_3d",
    ),
    "narrative_shift_score": (
        "dominant_narrative_trend",
        "dominant_narrative_sentiment",
        "narrative_bullish_pressure",
        "narrative_bearish_pressure",
    ),
}


@dataclass(frozen=True)
class AnomalyContext:
    date: object
    score: int
    level: str
    types: list[str]
    reasons: list[str]
    features: list[str]
    component_scores: dict[str, int]
    summary: str


def add_anomaly_scores(
    data: pd.DataFrame,
    lookback: int = 60,
    forward_days: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    """Add lightweight rule/statistical anomaly scores.

    Rolling baselines are shifted by one row so the current row is never used
    to define its own normal range.
    """
    if data.empty or "date" not in data.columns:
        return _empty_frame(data)

    frame = _prepare_frame(data)
    if frame.empty:
        return _empty_frame(data)

    frame["price_shock_score"] = _price_shock_score(frame, lookback)
    frame["volume_spike_score"] = _volume_spike_score(frame, lookback)
    frame["volatility_spike_score"] = _volatility_spike_score(frame, lookback)
    frame["sector_divergence_score"] = _sector_divergence_score(frame, lookback)
    frame["market_divergence_score"] = _market_divergence_score(frame, lookback)
    frame["news_shock_score"] = _news_shock_score(frame, lookback)
    frame["narrative_shift_score"] = _narrative_shift_score(frame, lookback)

    component_scores = frame[list(ANOMALY_COMPONENTS)].fillna(0.0)
    frame["anomaly_score"] = component_scores.apply(_overall_score, axis=1).astype(int)
    frame["anomaly_level"] = frame["anomaly_score"].map(_level)

    row_details = frame.apply(_row_anomaly_details, axis=1)
    frame["anomaly_types"] = row_details.map(lambda item: " | ".join(item["types"]))
    frame["anomaly_reasons"] = row_details.map(lambda item: " | ".join(item["reasons"]))
    frame["anomaly_features"] = row_details.map(lambda item: ", ".join(item["features"]))

    if "close" in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        for days in forward_days:
            frame[f"anomaly_return_{days}d_after"] = close.shift(-days) / close - 1

    return frame


def build_anomaly_context(data: pd.DataFrame) -> AnomalyContext | None:
    frame = data if "anomaly_score" in data.columns else add_anomaly_scores(data)
    if frame.empty:
        return None
    latest = frame.sort_values("date").iloc[-1]
    score = int(safe_float(latest.get("anomaly_score")) or 0)
    types = _split_text(latest.get("anomaly_types"), separator="|")
    reasons = _split_text(latest.get("anomaly_reasons"), separator="|")
    features = _split_text(latest.get("anomaly_features"), separator=",")
    component_scores = {
        ANOMALY_TYPE_LABELS[column]: int(safe_float(latest.get(column)) or 0)
        for column in ANOMALY_COMPONENTS
    }
    level = str(latest.get("anomaly_level") or _level(score))
    summary = build_anomaly_reason_from_context(
        AnomalyContext(
            date=latest.get("date"),
            score=score,
            level=level,
            types=types,
            reasons=reasons,
            features=features,
            component_scores=component_scores,
            summary="",
        )
    )
    return AnomalyContext(
        date=latest.get("date"),
        score=score,
        level=level,
        types=types,
        reasons=reasons,
        features=features,
        component_scores=component_scores,
        summary=summary,
    )


def build_anomaly_reason_from_context(context: AnomalyContext | None) -> str:
    if context is None:
        return "Anomaly context is not available yet."
    if context.score < 40:
        return (
            f"Current Anomaly Score is {context.score}/100 ({context.level}). "
            "No strong abnormal condition is detected by the MVP rules."
        )
    types = ", ".join(context.types) if context.types else "mixed anomaly"
    reason = context.reasons[0] if context.reasons else "Multiple indicators are outside normal ranges."
    return (
        f"Current Anomaly Score is {context.score}/100 ({context.level}). "
        f"Detected type: {types}. Main reason: {reason}"
    )


def find_anomaly_events(
    data: pd.DataFrame,
    min_score: int = 50,
    top_n: int | None = 100,
) -> pd.DataFrame:
    frame = data if "anomaly_score" in data.columns else add_anomaly_scores(data)
    if frame.empty or "anomaly_score" not in frame.columns:
        return _empty_events()
    events = frame[pd.to_numeric(frame["anomaly_score"], errors="coerce").fillna(0) >= min_score].copy()
    if events.empty:
        return _empty_events()
    events = events.sort_values(["date", "anomaly_score"], ascending=[False, False])
    if top_n is not None:
        events = events.head(top_n)
    columns = [
        "date",
        "anomaly_score",
        "anomaly_level",
        "anomaly_types",
        "anomaly_reasons",
        "anomaly_features",
        "return_1d",
        "volume_change_1d",
        "volatility_20",
        "anomaly_return_1d_after",
        "anomaly_return_5d_after",
        "anomaly_return_20d_after",
        *ANOMALY_COMPONENTS,
    ]
    return events[[column for column in columns if column in events.columns]].reset_index(drop=True)


def summarize_anomaly_forward_returns(
    events: pd.DataFrame,
    forward_days: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    columns = ["horizon", "count", "average_return", "positive_rate", "worst_return", "best_return"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for days in forward_days:
        values = pd.to_numeric(events.get(f"anomaly_return_{days}d_after"), errors="coerce").dropna()
        rows.append(
            {
                "horizon": f"{days}D",
                "count": int(len(values)),
                "average_return": _mean_or_none(values),
                "positive_rate": _positive_rate_or_none(values),
                "worst_return": _min_or_none(values),
                "best_return": _max_or_none(values),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def anomaly_direction(context: AnomalyContext | None) -> str:
    if context is None or context.score < 40:
        return "unknown"
    bearish_types = {"Volatility Spike", "Market Divergence", "News Shock", "Narrative Shift"}
    if "Price Shock" in context.types:
        return "mixed"
    if any(item in bearish_types for item in context.types):
        return "down"
    if "Volume Spike" in context.types:
        return "up"
    return "mixed"


def build_anomaly_comparison_frame(
    data: pd.DataFrame,
    event_date,
    lookback: int = 20,
    forward: int = 20,
) -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame(columns=["day", "current", "selected_anomaly"])

    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if frame.empty:
        return pd.DataFrame(columns=["day", "current", "selected_anomaly"])

    event_timestamp = pd.to_datetime(event_date, errors="coerce")
    if pd.isna(event_timestamp):
        return pd.DataFrame(columns=["day", "current", "selected_anomaly"])

    event_matches = frame.index[frame["date"] <= event_timestamp].tolist()
    if not event_matches:
        return pd.DataFrame(columns=["day", "current", "selected_anomaly"])
    event_index = int(event_matches[-1])
    current_index = len(frame) - 1

    selected = _normalized_window(frame, event_index, lookback=lookback, forward=forward)
    current = _normalized_window(frame, current_index, lookback=lookback, forward=0)
    result = pd.DataFrame({"day": range(-lookback + 1, forward + 1)})
    result = result.merge(selected.rename(columns={"value": "selected_anomaly"}), on="day", how="left")
    result = result.merge(current.rename(columns={"value": "current"}), on="day", how="left")
    return result


def _prepare_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "volume_change_1d",
        "rsi_14",
        "macd",
        "macd_hist",
        "volatility_20",
        "n225_return_1d",
        "topix_return_1d",
        "usdjpy_return_1d",
        "sector_mean_return",
        "honda_return_1d",
        "nissan_return_1d",
        "tesla_return_1d",
        "byd_return_1d",
        "news_count_3d",
        "max_importance_3d",
        "negative_news_shock",
        "positive_news_shock",
        "dominant_narrative_sentiment",
        "dominant_narrative_importance",
        "dominant_narrative_trend",
        "narrative_bullish_pressure",
        "narrative_bearish_pressure",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "return_1d" not in frame.columns and "close" in frame.columns:
        frame["return_1d"] = pd.to_numeric(frame["close"], errors="coerce").pct_change()
    if "volume_change_1d" not in frame.columns and "volume" in frame.columns:
        frame["volume_change_1d"] = pd.to_numeric(frame["volume"], errors="coerce").pct_change()
    return frame.replace([float("inf"), -float("inf")], pd.NA)


def _price_shock_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    score = _score_abs_z(_rolling_z(frame.get("return_1d"), lookback), moderate=1.8, extreme=4.0)
    if "return_1d" in frame.columns:
        score = score.combine(pd.to_numeric(frame["return_1d"], errors="coerce").abs().map(_return_score), max)
    if "rsi_14" in frame.columns:
        score = score.combine(pd.to_numeric(frame["rsi_14"], errors="coerce").map(_rsi_score), max)
    if "macd_hist" in frame.columns:
        macd_delta = pd.to_numeric(frame["macd_hist"], errors="coerce").diff()
        score = score.combine(_score_abs_z(_rolling_z(macd_delta, lookback), moderate=1.8, extreme=4.0), max)
    return score.fillna(0.0)


def _volume_spike_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    if "volume" in frame.columns:
        ratio = _ratio_to_prior_median(pd.to_numeric(frame["volume"], errors="coerce"), lookback)
        score = score.combine(ratio.map(lambda value: _ratio_score(value, moderate=1.5, extreme=3.0)), max)
    if "volume_change_1d" in frame.columns:
        volume_change = pd.to_numeric(frame["volume_change_1d"], errors="coerce")
        score = score.combine(_score_abs_z(_rolling_z(volume_change, lookback), moderate=2.0, extreme=4.0), max)
        score = score.combine(volume_change.map(lambda value: _positive_ratio_score(value, moderate=0.3, extreme=1.0)), max)
    return score.fillna(0.0)


def _volatility_spike_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    if "volatility_20" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    ratio = _ratio_to_prior_median(pd.to_numeric(frame["volatility_20"], errors="coerce"), lookback)
    return ratio.map(lambda value: _ratio_score(value, moderate=1.25, extreme=2.2)).fillna(0.0)


def _sector_divergence_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    if "return_1d" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    sector_return = _sector_return(frame)
    if sector_return is None:
        return pd.Series(0.0, index=frame.index)
    divergence = pd.to_numeric(frame["return_1d"], errors="coerce") - sector_return
    score = _score_abs_z(_rolling_z(divergence, lookback), moderate=1.8, extreme=4.0)
    score = score.combine(divergence.abs().map(_divergence_score), max)
    return score.fillna(0.0)


def _market_divergence_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    if "return_1d" not in frame.columns:
        return pd.Series(0.0, index=frame.index)
    market_columns = [column for column in ("n225_return_1d", "topix_return_1d") if column in frame.columns]
    if not market_columns:
        return pd.Series(0.0, index=frame.index)
    market_return = frame[market_columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    divergence = pd.to_numeric(frame["return_1d"], errors="coerce") - market_return
    score = _score_abs_z(_rolling_z(divergence, lookback), moderate=1.8, extreme=4.0)
    score = score.combine(divergence.abs().map(_divergence_score), max)
    if "usdjpy_return_1d" in frame.columns:
        usdjpy = pd.to_numeric(frame["usdjpy_return_1d"], errors="coerce")
        yen_divergence = pd.to_numeric(frame["return_1d"], errors="coerce") - usdjpy
        score = score.combine(_score_abs_z(_rolling_z(yen_divergence, lookback), moderate=2.2, extreme=4.5), max)
        score = score.combine(yen_divergence.abs().map(lambda value: _linear_score(value, moderate=0.04, extreme=0.08)), max)
    return score.fillna(0.0)


def _news_shock_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    if "max_importance_3d" in frame.columns:
        importance = pd.to_numeric(frame["max_importance_3d"], errors="coerce").fillna(0.0)
        score = score.combine(importance.map(lambda value: _linear_score(value, moderate=0.65, extreme=0.95)), max)
    for column in ("negative_news_shock", "positive_news_shock"):
        if column in frame.columns:
            shock = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
            score = score.combine(shock.map(lambda value: 85.0 if value >= 1.0 else 0.0), max)
    if "news_count_3d" in frame.columns:
        news_count = pd.to_numeric(frame["news_count_3d"], errors="coerce")
        score = score.combine(_score_abs_z(_rolling_z(news_count, lookback), moderate=2.0, extreme=4.0), max)
    return score.fillna(0.0)


def _narrative_shift_score(frame: pd.DataFrame, lookback: int) -> pd.Series:
    score = pd.Series(0.0, index=frame.index)
    for column in (
        "dominant_narrative_trend",
        "dominant_narrative_sentiment",
        "narrative_bullish_pressure",
        "narrative_bearish_pressure",
    ):
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        score = score.combine(_score_abs_z(_rolling_z(series.diff(), lookback), moderate=1.8, extreme=4.0), max)
        score = score.combine(series.abs().map(lambda value: _linear_score(value, moderate=0.45, extreme=1.0)), max)
    return score.fillna(0.0)


def _sector_return(frame: pd.DataFrame) -> pd.Series | None:
    if "sector_mean_return" in frame.columns:
        return pd.to_numeric(frame["sector_mean_return"], errors="coerce")
    columns = [
        column
        for column in ("honda_return_1d", "nissan_return_1d", "tesla_return_1d", "byd_return_1d")
        if column in frame.columns
    ]
    if not columns:
        return None
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def _rolling_z(series: pd.Series | None, lookback: int) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    values = pd.to_numeric(series, errors="coerce")
    baseline = values.shift(1)
    mean = baseline.rolling(window=lookback, min_periods=max(10, lookback // 3)).mean()
    std = baseline.rolling(window=lookback, min_periods=max(10, lookback // 3)).std()
    return (values - mean) / std.replace(0, pd.NA)


def _ratio_to_prior_median(series: pd.Series, lookback: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    median = values.shift(1).rolling(window=lookback, min_periods=max(10, lookback // 3)).median()
    return values / median.replace(0, pd.NA)


def _score_abs_z(z: pd.Series, moderate: float, extreme: float) -> pd.Series:
    return z.abs().map(lambda value: _linear_score(value, moderate=moderate, extreme=extreme))


def _linear_score(value, moderate: float, extreme: float) -> float:
    number = safe_float(value)
    if number is None or not math.isfinite(number):
        return 0.0
    if number < moderate:
        return 0.0
    if number >= extreme:
        return 100.0
    return float((number - moderate) / (extreme - moderate) * 100.0)


def _ratio_score(value, moderate: float, extreme: float) -> float:
    number = safe_float(value)
    if number is None:
        return 0.0
    return _linear_score(number, moderate=moderate, extreme=extreme)


def _positive_ratio_score(value, moderate: float, extreme: float) -> float:
    number = safe_float(value)
    if number is None or number < moderate:
        return 0.0
    if number >= extreme:
        return 100.0
    return float((number - moderate) / (extreme - moderate) * 100.0)


def _return_score(value) -> float:
    number = safe_float(value)
    if number is None:
        return 0.0
    absolute = abs(number)
    if absolute >= 0.07:
        return 100.0
    if absolute >= 0.05:
        return 80.0
    if absolute >= 0.03:
        return 50.0
    return 0.0


def _rsi_score(value) -> float:
    number = safe_float(value)
    if number is None:
        return 0.0
    if number >= 80 or number <= 20:
        return 100.0
    if number >= 75 or number <= 25:
        return 80.0
    if number >= 70 or number <= 30:
        return 55.0
    return 0.0


def _divergence_score(value) -> float:
    number = safe_float(value)
    if number is None:
        return 0.0
    return _linear_score(abs(number), moderate=0.025, extreme=0.06)


def _overall_score(row: pd.Series) -> float:
    values = sorted([safe_float(value) or 0.0 for value in row], reverse=True)
    if not values:
        return 0.0
    padded = values + [0.0, 0.0, 0.0]
    top3_mean = sum(padded[:3]) / 3.0
    breadth = sum(value >= 50.0 for value in values)
    return min(100.0, round(values[0] * 0.60 + top3_mean * 0.40 + breadth * 4.0))


def _row_anomaly_details(row: pd.Series) -> dict[str, list[str]]:
    types: list[str] = []
    reasons: list[str] = []
    features: list[str] = []
    for component in ANOMALY_COMPONENTS:
        score = safe_float(row.get(component)) or 0.0
        if score < 50:
            continue
        anomaly_type = ANOMALY_TYPE_LABELS[component]
        types.append(anomaly_type)
        reasons.append(_component_reason(anomaly_type, row))
        features.extend(column for column in ANOMALY_FEATURES[component] if column in row.index)
    return {
        "types": types,
        "reasons": reasons,
        "features": sorted(set(features)),
    }


def _component_reason(anomaly_type: str, row: pd.Series) -> str:
    if anomaly_type == "Price Shock":
        return f"Price or momentum is outside its normal range; latest return is {_format_percent(row.get('return_1d'))}."
    if anomaly_type == "Volume Spike":
        return f"Volume activity is unusually high; volume change is {_format_percent(row.get('volume_change_1d'))}."
    if anomaly_type == "Volatility Spike":
        return f"Volatility is elevated; 20D volatility is {_format_percent(row.get('volatility_20'))}."
    if anomaly_type == "Sector Divergence":
        return "Toyota is moving differently from the auto-sector context."
    if anomaly_type == "Market Divergence":
        return "Toyota is diverging from broad market or FX context."
    if anomaly_type == "News Shock":
        return "Recent news importance or shock flags are elevated."
    if anomaly_type == "Narrative Shift":
        return "Dominant narrative or narrative pressure changed sharply."
    return f"{anomaly_type} detected."


def _level(score: int | float) -> str:
    value = int(score)
    if value >= 70:
        return "High"
    if value >= 40:
        return "Medium"
    return "Low"


def _split_text(value, separator: str) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(separator) if part.strip()]


def _normalized_window(frame: pd.DataFrame, center_index: int, lookback: int, forward: int) -> pd.DataFrame:
    start = max(0, center_index - lookback + 1)
    end = min(len(frame) - 1, center_index + forward)
    window = frame.iloc[start : end + 1].copy()
    base = safe_float(frame.loc[center_index, "close"])
    if base is None or base == 0:
        return pd.DataFrame(columns=["day", "value"])
    window["day"] = window.index - center_index
    window["value"] = pd.to_numeric(window["close"], errors="coerce") / base * 100.0
    return window[["day", "value"]]


def _empty_frame(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    for column in (*ANOMALY_COMPONENTS, "anomaly_score"):
        frame[column] = pd.Series(dtype=float)
    frame["anomaly_level"] = pd.Series(dtype=str)
    frame["anomaly_types"] = pd.Series(dtype=str)
    frame["anomaly_reasons"] = pd.Series(dtype=str)
    frame["anomaly_features"] = pd.Series(dtype=str)
    return frame


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "anomaly_score",
            "anomaly_level",
            "anomaly_types",
            "anomaly_reasons",
            "anomaly_features",
            "anomaly_return_1d_after",
            "anomaly_return_5d_after",
            "anomaly_return_20d_after",
        ]
    )


def _mean_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.mean())


def _positive_rate_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float((values > 0).mean())


def _min_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.min())


def _max_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.max())


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"
