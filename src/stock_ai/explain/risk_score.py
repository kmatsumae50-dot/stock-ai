from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ai.explain.shap_reason import split_shap_contributions
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class RiskScore:
    name: str
    score: int
    level: str
    reasons: list[str]
    features: list[str]


def build_surge_crash_scores(
    data: pd.DataFrame,
    shap_items: list[dict[str, object]],
    similar_summary: pd.DataFrame,
    cross_ticker_summary: pd.DataFrame | None = None,
    lead_lag_summary: pd.DataFrame | None = None,
    anomaly_context: object | None = None,
) -> dict[str, RiskScore]:
    """Build lightweight rule-based surge/crash early-warning scores.

    The output is intentionally model-agnostic so it can later be backed by a
    5-class classifier without changing the Prediction tab contract.
    """
    frame = _prepare_price_frame(data)
    if frame.empty:
        return {
            "surge": _risk_score("Surge", 0, [], []),
            "crash": _risk_score("Crash", 0, [], []),
        }

    surge_points, surge_reasons, surge_features = _surge_components(
        frame,
        shap_items,
        similar_summary,
        cross_ticker_summary,
        lead_lag_summary,
        anomaly_context,
    )
    crash_points, crash_reasons, crash_features = _crash_components(
        frame,
        shap_items,
        similar_summary,
        cross_ticker_summary,
        lead_lag_summary,
        anomaly_context,
    )
    return {
        "surge": _risk_score("Surge", surge_points, surge_reasons, surge_features),
        "crash": _risk_score("Crash", crash_points, crash_reasons, crash_features),
    }


def _surge_components(
    frame: pd.DataFrame,
    shap_items: list[dict[str, object]],
    similar_summary: pd.DataFrame,
    cross_ticker_summary: pd.DataFrame | None,
    lead_lag_summary: pd.DataFrame | None,
    anomaly_context: object | None,
) -> tuple[int, list[str], list[str]]:
    points = 0
    reasons: list[str] = []
    features: list[str] = []

    if _volume_spike(frame):
        points += 20
        reasons.append("Volume is expanding sharply.")
        features.append("volume_change_1d")

    if _rsi_up(frame):
        points += 15
        reasons.append("RSI is rising or already in a strong zone.")
        features.append("rsi_14")

    if _macd_improving(frame):
        points += 15
        reasons.append("MACD momentum is improving.")
        features.append("macd")

    if _similarity_points(similar_summary, direction="surge"):
        points += 20
        reasons.append("Similar historical patterns tended to rise afterward.")
        features.append("similar_patterns")

    if _cross_ticker_points(cross_ticker_summary, direction="surge"):
        points += 15
        reasons.append("Cross-ticker similar patterns tended to rise afterward.")
        features.append("cross_ticker_similarity")

    if _lead_lag_points(lead_lag_summary, direction="surge"):
        points += 10
        reasons.append("Current lead-lag signals from peers or markets lean bullish.")
        features.append("lead_lag")

    if _anomaly_points(anomaly_context, direction="surge"):
        points += 10
        reasons.append("Current anomaly context can amplify upside volatility.")
        features.append("anomaly")

    if _shap_points(shap_items, direction="surge"):
        points += 25
        reasons.append("SHAP drivers are strongly tilted toward the up class.")
        features.append("shap")

    if _positive_news_shock(frame):
        points += 20
        reasons.append("Important positive news shock is present.")
        features.append("positive_news_shock")

    if _positive_news_context(frame):
        points += 10
        reasons.append("News sentiment and sector news context lean positive.")
        features.extend(["importance_weighted_sentiment", "sector_news_strength"])

    if _positive_narrative_context(frame):
        points += 10
        reasons.append("Dominant market narrative is supportive.")
        features.extend(["narrative_bullish_pressure", "dominant_narrative_sentiment"])

    return points, reasons, features


def _crash_components(
    frame: pd.DataFrame,
    shap_items: list[dict[str, object]],
    similar_summary: pd.DataFrame,
    cross_ticker_summary: pd.DataFrame | None,
    lead_lag_summary: pd.DataFrame | None,
    anomaly_context: object | None,
) -> tuple[int, list[str], list[str]]:
    points = 0
    reasons: list[str] = []
    features: list[str] = []

    if _volatility_spike(frame):
        points += 20
        reasons.append("Volatility is elevated versus recent history.")
        features.append("volatility_20")

    if _volume_spike(frame):
        points += 15
        reasons.append("Volume is expanding sharply.")
        features.append("volume_change_1d")

    if _rsi_down(frame):
        points += 15
        reasons.append("RSI is weak or falling.")
        features.append("rsi_14")

    if _macd_worsening(frame):
        points += 15
        reasons.append("MACD momentum is deteriorating.")
        features.append("macd")

    if _similarity_points(similar_summary, direction="crash"):
        points += 20
        reasons.append("Similar historical patterns tended to fall afterward.")
        features.append("similar_patterns")

    if _cross_ticker_points(cross_ticker_summary, direction="crash"):
        points += 15
        reasons.append("Cross-ticker similar patterns tended to fall afterward.")
        features.append("cross_ticker_similarity")

    if _lead_lag_points(lead_lag_summary, direction="crash"):
        points += 10
        reasons.append("Current lead-lag signals from peers or markets lean bearish.")
        features.append("lead_lag")

    if _anomaly_points(anomaly_context, direction="crash"):
        points += 10
        reasons.append("Current anomaly context can amplify downside or event risk.")
        features.append("anomaly")

    if _shap_points(shap_items, direction="crash"):
        points += 25
        reasons.append("SHAP drivers are strongly tilted toward the down class.")
        features.append("shap")

    if _negative_news_shock(frame):
        points += 20
        reasons.append("Important negative news shock is present.")
        features.append("negative_news_shock")

    if _negative_news_context(frame):
        points += 10
        reasons.append("News sentiment and sector news context lean negative.")
        features.extend(["importance_weighted_sentiment", "sector_news_strength"])

    if _negative_narrative_context(frame):
        points += 10
        reasons.append("Dominant market narrative is bearish.")
        features.extend(["narrative_bearish_pressure", "dominant_narrative_sentiment"])

    return points, reasons, features


def _risk_score(name: str, points: int, reasons: list[str], features: list[str]) -> RiskScore:
    score = max(0, min(100, int(points)))
    if score >= 70:
        level = "High"
    elif score >= 40:
        level = "Medium"
    else:
        level = "Low"
    if not reasons:
        reasons = ["No strong early-warning signals detected by the MVP rules."]
    return RiskScore(
        name=name,
        score=score,
        level=level,
        reasons=reasons,
        features=sorted(set(features)),
    )


def _prepare_price_frame(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    for column in (
        "volume",
        "volume_change_1d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_20",
        "sentiment_trend_3d",
        "sentiment_trend_7d",
        "importance_weighted_sentiment",
        "negative_news_shock",
        "positive_news_shock",
        "sector_news_strength",
        "dominant_narrative_code",
        "dominant_narrative_sentiment",
        "dominant_narrative_importance",
        "dominant_narrative_trend",
        "narrative_bullish_pressure",
        "narrative_bearish_pressure",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _volume_spike(frame: pd.DataFrame) -> bool:
    latest_change = _latest(frame, "volume_change_1d")
    if latest_change is not None and latest_change >= 0.30:
        return True
    if "volume" not in frame.columns or len(frame) < 20:
        return False
    latest_volume = _latest(frame, "volume")
    recent_average = safe_float(frame["volume"].tail(20).mean())
    if latest_volume is None or recent_average is None or recent_average == 0:
        return False
    return latest_volume >= recent_average * 1.5


def _volatility_spike(frame: pd.DataFrame) -> bool:
    latest = _latest(frame, "volatility_20")
    if latest is None:
        return False
    recent = pd.to_numeric(frame["volatility_20"].tail(60), errors="coerce").dropna()
    if len(recent) < 10:
        return False
    baseline = safe_float(recent.median())
    return baseline is not None and baseline > 0 and latest >= baseline * 1.25


def _rsi_up(frame: pd.DataFrame) -> bool:
    latest = _latest(frame, "rsi_14")
    previous = _previous(frame, "rsi_14", periods=5)
    if latest is None:
        return False
    if latest >= 60:
        return True
    return previous is not None and latest - previous >= 5


def _rsi_down(frame: pd.DataFrame) -> bool:
    latest = _latest(frame, "rsi_14")
    previous = _previous(frame, "rsi_14", periods=5)
    if latest is None:
        return False
    if latest <= 40:
        return True
    return previous is not None and previous - latest >= 5


def _macd_improving(frame: pd.DataFrame) -> bool:
    latest_hist = _latest(frame, "macd_hist")
    previous_hist = _previous(frame, "macd_hist", periods=3)
    if latest_hist is not None:
        if latest_hist > 0:
            return True
        return previous_hist is not None and latest_hist > previous_hist
    macd = _latest(frame, "macd")
    signal = _latest(frame, "macd_signal")
    return macd is not None and signal is not None and macd > signal


def _macd_worsening(frame: pd.DataFrame) -> bool:
    latest_hist = _latest(frame, "macd_hist")
    previous_hist = _previous(frame, "macd_hist", periods=3)
    if latest_hist is not None:
        if latest_hist < 0:
            return True
        return previous_hist is not None and latest_hist < previous_hist
    macd = _latest(frame, "macd")
    signal = _latest(frame, "macd_signal")
    return macd is not None and signal is not None and macd < signal


def _similarity_points(similar_summary: pd.DataFrame, direction: str) -> bool:
    row = _summary_row(similar_summary, "5D") or _summary_row(similar_summary, "1D")
    if row is None:
        return False
    average_return = safe_float(row.get("average_return"))
    positive_rate = safe_float(row.get("positive_rate"))
    if average_return is None:
        return False
    if direction == "surge":
        return average_return >= 0.02 or (positive_rate is not None and positive_rate >= 0.60)
    return average_return <= -0.02 or (positive_rate is not None and positive_rate <= 0.40)


def _cross_ticker_points(cross_ticker_summary: pd.DataFrame | None, direction: str) -> bool:
    if cross_ticker_summary is None:
        return False
    row = _summary_row(cross_ticker_summary, "5D") or _summary_row(cross_ticker_summary, "1D")
    if row is None:
        return False
    toyota_return = safe_float(row.get("toyota_average_return"))
    peer_return = safe_float(row.get("peer_average_return"))
    toyota_positive = safe_float(row.get("toyota_positive_rate"))
    peer_positive = safe_float(row.get("peer_positive_rate"))
    signal_return = toyota_return if toyota_return is not None else peer_return
    signal_positive = toyota_positive if toyota_positive is not None else peer_positive
    if signal_return is None:
        return False
    if direction == "surge":
        return signal_return >= 0.02 or (signal_positive is not None and signal_positive >= 0.60)
    return signal_return <= -0.02 or (signal_positive is not None and signal_positive <= 0.40)


def _lead_lag_points(lead_lag_summary: pd.DataFrame | None, direction: str) -> bool:
    if lead_lag_summary is None or lead_lag_summary.empty:
        return False
    if "current_signal" not in lead_lag_summary.columns:
        return False

    summary = lead_lag_summary.copy()
    summary["score"] = pd.to_numeric(summary.get("score"), errors="coerce").fillna(0.0)
    summary = summary[summary["score"] >= 10.0]
    if summary.empty:
        return False

    target_signal = "bullish" if direction == "surge" else "bearish"
    return bool((summary["current_signal"] == target_signal).any())


def _anomaly_points(anomaly_context: object | None, direction: str) -> bool:
    if anomaly_context is None:
        return False
    score = safe_float(_object_attr(anomaly_context, "score")) or 0.0
    if score < 50:
        return False
    types = _object_attr(anomaly_context, "types") or []
    if isinstance(types, str):
        types = [part.strip() for part in types.split("|") if part.strip()]
    if direction == "surge":
        return bool({"Volume Spike", "Price Shock"} & set(types))
    return bool(
        {
            "Volatility Spike",
            "Sector Divergence",
            "Market Divergence",
            "News Shock",
            "Narrative Shift",
            "Price Shock",
        }
        & set(types)
    )


def _shap_points(shap_items: list[dict[str, object]], direction: str) -> bool:
    upward, downward = split_shap_contributions(shap_items, top_n=10)
    up_strength = sum(abs(safe_float(item.get("shap_value")) or 0.0) for item in upward)
    down_strength = sum(abs(safe_float(item.get("shap_value")) or 0.0) for item in downward)
    if up_strength == 0 and down_strength == 0:
        return False
    if direction == "surge":
        return up_strength >= max(0.05, down_strength * 1.25)
    return down_strength >= max(0.05, up_strength * 1.25)


def _positive_news_shock(frame: pd.DataFrame) -> bool:
    return (_latest(frame, "positive_news_shock") or 0.0) >= 1.0


def _negative_news_shock(frame: pd.DataFrame) -> bool:
    return (_latest(frame, "negative_news_shock") or 0.0) >= 1.0


def _positive_news_context(frame: pd.DataFrame) -> bool:
    weighted = _latest(frame, "importance_weighted_sentiment")
    trend = _latest(frame, "sentiment_trend_3d")
    sector = _latest(frame, "sector_news_strength")
    return (
        (weighted is not None and weighted >= 0.25)
        or (trend is not None and trend >= 0.25)
        or (sector is not None and sector >= 0.25)
    )


def _negative_news_context(frame: pd.DataFrame) -> bool:
    weighted = _latest(frame, "importance_weighted_sentiment")
    trend = _latest(frame, "sentiment_trend_3d")
    sector = _latest(frame, "sector_news_strength")
    return (
        (weighted is not None and weighted <= -0.25)
        or (trend is not None and trend <= -0.25)
        or (sector is not None and sector <= -0.25)
    )


def _positive_narrative_context(frame: pd.DataFrame) -> bool:
    pressure = _latest(frame, "narrative_bullish_pressure")
    sentiment = _latest(frame, "dominant_narrative_sentiment")
    importance = _latest(frame, "dominant_narrative_importance")
    return (
        (pressure is not None and pressure >= 0.35)
        or (
            sentiment is not None
            and sentiment >= 0.25
            and importance is not None
            and importance >= 0.5
        )
    )


def _negative_narrative_context(frame: pd.DataFrame) -> bool:
    pressure = _latest(frame, "narrative_bearish_pressure")
    sentiment = _latest(frame, "dominant_narrative_sentiment")
    importance = _latest(frame, "dominant_narrative_importance")
    return (
        (pressure is not None and pressure >= 0.35)
        or (
            sentiment is not None
            and sentiment <= -0.25
            and importance is not None
            and importance >= 0.5
        )
    )


def _summary_row(summary: pd.DataFrame, horizon: str) -> dict[str, object] | None:
    if summary.empty or "horizon" not in summary.columns:
        return None
    rows = summary[summary["horizon"] == horizon]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _latest(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns or frame.empty:
        return None
    return safe_float(frame.iloc[-1].get(column))


def _previous(frame: pd.DataFrame, column: str, periods: int) -> float | None:
    if column not in frame.columns or len(frame) <= periods:
        return None
    return safe_float(frame.iloc[-1 - periods].get(column))


def _object_attr(item: object, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)
