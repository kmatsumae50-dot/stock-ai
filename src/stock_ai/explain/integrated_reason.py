from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ai.explain.shap_reason import split_shap_contributions
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class IntegratedReasoning:
    prediction_direction: str
    confidence_level: str
    probability_text: str
    shap_direction: str
    similarity_direction: str
    news_direction: str
    narrative_direction: str
    agreement: str
    summary: str
    shap_text: str
    similarity_text: str
    external_text: str
    event_text: str
    news_text: str
    news_alignment_text: str
    narrative_text: str
    cross_ticker_direction: str
    cross_ticker_text: str
    lead_lag_direction: str
    lead_lag_text: str
    anomaly_direction: str
    anomaly_text: str
    regime_text: str
    risk_text: str
    conflict_text: str


def build_integrated_reasoning(
    prediction_label: str,
    up_probability,
    down_probability,
    shap_items: list[dict[str, object]],
    similar_patterns: pd.DataFrame,
    similar_summary: pd.DataFrame,
    external_context: object | None = None,
    event_context: object | None = None,
    news_context: object | None = None,
    narrative_context: object | None = None,
    cross_ticker_patterns: pd.DataFrame | None = None,
    cross_ticker_summary: pd.DataFrame | None = None,
    lead_lag_summary: pd.DataFrame | None = None,
    anomaly_context: object | None = None,
    regime_state: object | None = None,
    risk_scores: dict[str, object] | None = None,
) -> IntegratedReasoning:
    prediction_direction = _prediction_direction(prediction_label, up_probability, down_probability)
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    confidence = max([value for value in (up, down) if value is not None], default=None)
    shap_direction = _shap_direction(shap_items)
    similarity_direction = _similarity_direction(similar_summary)
    news_direction = _news_direction(news_context)
    narrative_direction = _narrative_direction(narrative_context)
    cross_ticker_direction = _cross_ticker_direction(cross_ticker_summary)
    lead_lag_direction = _lead_lag_direction(lead_lag_summary)
    anomaly_direction = _anomaly_direction(anomaly_context)
    confidence_level = _confidence_level(
        prediction_direction=prediction_direction,
        probability=confidence,
        shap_direction=shap_direction,
        similarity_direction=similarity_direction,
        news_direction=news_direction,
        lead_lag_direction=lead_lag_direction,
        anomaly_direction=anomaly_direction,
    )
    agreement = _agreement_label(
        prediction_direction,
        shap_direction,
        similarity_direction,
        news_direction,
        lead_lag_direction,
        anomaly_direction,
    )
    probability_text = _probability_text(up, down, prediction_direction)
    shap_text = _shap_text(shap_items, prediction_direction)
    similarity_text = _similarity_text(similar_patterns, similar_summary)
    external_text = _external_text(external_context)
    event_text = _event_text(event_context)
    news_text = _news_text(news_context)
    narrative_text = _narrative_text(narrative_context)
    cross_ticker_text = _cross_ticker_text(cross_ticker_patterns, cross_ticker_summary)
    lead_lag_text = _lead_lag_text(lead_lag_summary)
    anomaly_text = _anomaly_text(anomaly_context)
    news_alignment_text = _news_alignment_text(
        prediction_direction=prediction_direction,
        shap_direction=shap_direction,
        similarity_direction=similarity_direction,
        news_direction=news_direction,
    )
    regime_text = _regime_text(regime_state)
    risk_text = _risk_text(risk_scores)
    conflict_text = _conflict_text(
        prediction_direction,
        shap_direction,
        similarity_direction,
        news_direction,
        lead_lag_direction,
        anomaly_direction,
    )
    summary = _summary_text(
        prediction_direction=prediction_direction,
        confidence_level=confidence_level,
        probability_text=probability_text,
        shap_text=shap_text,
        similarity_text=similarity_text,
        external_text=external_text,
        event_text=event_text,
        news_text=news_text,
        news_alignment_text=news_alignment_text,
        narrative_text=narrative_text,
        cross_ticker_text=cross_ticker_text,
        lead_lag_text=lead_lag_text,
        anomaly_text=anomaly_text,
        regime_text=regime_text,
        risk_text=risk_text,
        conflict_text=conflict_text,
    )
    return IntegratedReasoning(
        prediction_direction=prediction_direction,
        confidence_level=confidence_level,
        probability_text=probability_text,
        shap_direction=shap_direction,
        similarity_direction=similarity_direction,
        news_direction=news_direction,
        narrative_direction=narrative_direction,
        cross_ticker_direction=cross_ticker_direction,
        lead_lag_direction=lead_lag_direction,
        anomaly_direction=anomaly_direction,
        agreement=agreement,
        summary=summary,
        shap_text=shap_text,
        similarity_text=similarity_text,
        external_text=external_text,
        event_text=event_text,
        news_text=news_text,
        news_alignment_text=news_alignment_text,
        narrative_text=narrative_text,
        cross_ticker_text=cross_ticker_text,
        lead_lag_text=lead_lag_text,
        anomaly_text=anomaly_text,
        regime_text=regime_text,
        risk_text=risk_text,
        conflict_text=conflict_text,
    )


def _prediction_direction(prediction_label: str, up_probability, down_probability) -> str:
    if prediction_label in {"up", "down"}:
        return prediction_label
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    if up is None or down is None:
        return "unknown"
    return "up" if up >= down else "down"


def _shap_direction(shap_items: list[dict[str, object]]) -> str:
    upward, downward = split_shap_contributions(shap_items, top_n=10)
    up_strength = sum(abs(safe_float(item.get("shap_value")) or 0.0) for item in upward)
    down_strength = sum(abs(safe_float(item.get("shap_value")) or 0.0) for item in downward)
    if up_strength == 0 and down_strength == 0:
        return "unknown"
    if up_strength >= down_strength * 1.15:
        return "up"
    if down_strength >= up_strength * 1.15:
        return "down"
    return "mixed"


def _similarity_direction(similar_summary: pd.DataFrame) -> str:
    row = _summary_row(similar_summary, horizon="5D")
    if row is None:
        row = _summary_row(similar_summary, horizon="1D")
    if row is None:
        return "unknown"

    average_return = safe_float(row.get("average_return"))
    positive_rate = safe_float(row.get("positive_rate"))
    if average_return is None:
        return "unknown"
    if average_return > 0 and (positive_rate is None or positive_rate >= 0.5):
        return "up"
    if average_return < 0 and (positive_rate is None or positive_rate <= 0.5):
        return "down"
    return "mixed"


def _confidence_level(
    prediction_direction: str,
    probability: float | None,
    shap_direction: str,
    similarity_direction: str,
    news_direction: str,
    lead_lag_direction: str,
    anomaly_direction: str,
) -> str:
    if prediction_direction == "unknown":
        return "Low"

    shap_matches = shap_direction == prediction_direction
    similarity_matches = similarity_direction == prediction_direction
    news_matches = news_direction == prediction_direction
    lead_lag_matches = lead_lag_direction == prediction_direction
    anomaly_matches = anomaly_direction == prediction_direction
    shap_conflicts = _conflicts(prediction_direction, shap_direction)
    similarity_conflicts = _conflicts(prediction_direction, similarity_direction)
    news_conflicts = _conflicts(prediction_direction, news_direction)
    lead_lag_conflicts = _conflicts(prediction_direction, lead_lag_direction)
    anomaly_conflicts = _conflicts(prediction_direction, anomaly_direction)

    if probability is not None and probability >= 0.65 and shap_matches and similarity_matches:
        if news_conflicts or lead_lag_conflicts or anomaly_conflicts:
            return "Medium"
        return "High"
    if (
        shap_conflicts
        or similarity_conflicts
        or news_conflicts
        or lead_lag_conflicts
        or anomaly_conflicts
    ):
        return "Low"
    if probability is not None and probability >= 0.55 and (
        shap_matches or similarity_matches or news_matches or lead_lag_matches or anomaly_matches
    ):
        return "Medium"
    return "Low"


def _agreement_label(
    prediction_direction: str,
    shap_direction: str,
    similarity_direction: str,
    news_direction: str,
    lead_lag_direction: str,
    anomaly_direction: str,
) -> str:
    if prediction_direction == "unknown":
        return "unknown"
    matches = sum(
        direction == prediction_direction
        for direction in (
            shap_direction,
            similarity_direction,
            news_direction,
            lead_lag_direction,
            anomaly_direction,
        )
    )
    conflicts = sum(
        _conflicts(prediction_direction, direction)
        for direction in (
            shap_direction,
            similarity_direction,
            news_direction,
            lead_lag_direction,
            anomaly_direction,
        )
    )
    if matches >= 5:
        return "model, SHAP, similar patterns, news, lead-lag, and anomaly context align"
    if matches >= 2:
        return "most evidence aligns"
    if conflicts:
        return "mixed evidence"
    if matches == 1:
        return "partially aligned"
    return "weak evidence"


def _probability_text(up: float | None, down: float | None, prediction_direction: str) -> str:
    direction_text = _direction_text(prediction_direction)
    if up is None and down is None:
        return f"現在の予測は{direction_text}ですが、確率情報は不足しています。"
    return (
        f"現在の予測は{direction_text}です"
        f"（上昇確率 {_format_percent(up)}、下落確率 {_format_percent(down)}）。"
    )


def _shap_text(shap_items: list[dict[str, object]], prediction_direction: str) -> str:
    upward, downward = split_shap_contributions(shap_items, top_n=3)
    drivers = upward if prediction_direction == "up" else downward
    opposing = downward if prediction_direction == "up" else upward
    if not drivers and not opposing:
        return "SHAPの局所寄与はまだ十分に保存されていません。"

    parts: list[str] = []
    if drivers:
        parts.append(
            "SHAPでは"
            + "、".join(_feature_name(item) for item in drivers)
            + f"が{_direction_text(prediction_direction)}へ寄与しています。"
        )
    if opposing:
        parts.append(
            "一方で"
            + "、".join(_feature_name(item) for item in opposing[:2])
            + f"は{_opposite_direction_text(prediction_direction)}方向の材料です。"
        )
    return " ".join(parts)


def _similarity_text(similar_patterns: pd.DataFrame, similar_summary: pd.DataFrame) -> str:
    if similar_patterns.empty:
        return "過去の類似局面はまだ十分に見つかっていません。"

    row_1d = _summary_row(similar_summary, "1D")
    row_5d = _summary_row(similar_summary, "5D")
    row_20d = _summary_row(similar_summary, "20D")
    best = similar_patterns.iloc[0]
    best_start = pd.to_datetime(best.get("similar_start_date")).strftime("%Y-%m-%d")
    best_end = pd.to_datetime(best.get("similar_end_date")).strftime("%Y-%m-%d")
    similarity = _format_number(best.get("similarity"), digits=3)

    return (
        f"類似局面Top5では、最も近い局面は{best_start}から{best_end}"
        f"（類似度 {similarity}）です。"
        f"その後の平均騰落率は1日 {_summary_return(row_1d)}、"
        f"5日 {_summary_return(row_5d)}、20日 {_summary_return(row_20d)}で、"
        f"5日後の上昇率は{_summary_positive_rate(row_5d)}です。"
    )


def _conflict_text(
    prediction_direction: str,
    shap_direction: str,
    similarity_direction: str,
    news_direction: str,
    lead_lag_direction: str,
    anomaly_direction: str,
) -> str:
    conflicts: list[str] = []
    if _conflicts(prediction_direction, shap_direction):
        conflicts.append("SHAPの総合方向")
    if _conflicts(prediction_direction, similarity_direction):
        conflicts.append("類似局面の事後推移")
    if _conflicts(prediction_direction, news_direction):
        conflicts.append("ニュース文脈")
    if _conflicts(prediction_direction, lead_lag_direction):
        conflicts.append("lead-lag signals")
    if _conflicts(prediction_direction, anomaly_direction):
        conflicts.append("anomaly context")
    if not conflicts:
        return ""
    return (
        f"ただし、{ 'と'.join(conflicts) }は予測方向と矛盾しているため、"
        "信頼度は低めに見ます。"
    )


def _regime_text(regime_state: object | None) -> str:
    if regime_state is None:
        return ""

    name = _regime_attr(regime_state, "name")
    if not name:
        return ""

    if name == "Panic":
        context = "panic-like stress is already visible, so downside and liquidity risk deserve priority."
    elif name == "High Volatility":
        context = "volatility is elevated, so similar-pattern and crash-risk signals should be read carefully."
    elif name == "Bull Trend":
        context = "trend conditions are supportive, which can strengthen upward signals."
    elif name == "Bear Trend":
        context = "trend conditions are weak, which can strengthen downward signals."
    elif name == "Sideways":
        context = "trend direction is not strong, so probability confidence should be treated conservatively."
    else:
        context = "regime context is mixed."

    return f"Current market regime is {name}; {context}"


def _external_text(external_context: object | None) -> str:
    if external_context is None:
        return ""

    summary = _external_attr(external_context, "summary")
    if summary:
        return str(summary)

    usdjpy_status = _external_attr(external_context, "usdjpy_status") or "Unavailable"
    market_trend = _external_attr(external_context, "market_trend") or "Unavailable"
    auto_sector_trend = _external_attr(external_context, "auto_sector_trend") or "Unavailable"
    tesla_trend = _external_attr(external_context, "tesla_trend") or "Unavailable"
    if (
        usdjpy_status == "Unavailable"
        and market_trend == "Unavailable"
        and auto_sector_trend == "Unavailable"
    ):
        return ""
    return (
        f"External market context: USDJPY is {usdjpy_status}; "
        f"market trend is {market_trend}; auto sector trend is {auto_sector_trend}; "
        f"Tesla trend is {tesla_trend}."
    )


def _event_text(event_context: object | None) -> str:
    if event_context is None:
        return ""

    summary = _external_attr(event_context, "summary")
    if summary:
        return str(summary)

    event_state = _external_attr(event_context, "event_state") or "normal"
    days_to = _external_attr(event_context, "days_to_earnings")
    days_after = _external_attr(event_context, "days_after_earnings")
    if event_state == "normal":
        return ""
    if event_state == "pre_earnings_week":
        return f"Earnings are approaching in {days_to} days, so volatility risk may rise."
    if event_state == "earnings_day":
        return "Today is an earnings event day, so price gaps and volume spikes deserve extra care."
    if event_state == "post_earnings":
        return f"Post-earnings mode is active, {days_after} days after the event."
    return f"Earnings event state is {event_state}."


def _news_text(news_context: object | None) -> str:
    if news_context is None:
        return ""

    impact_summary = _external_attr(news_context, "impact_summary")
    if impact_summary:
        return str(impact_summary)

    summary = _external_attr(news_context, "summary")
    if summary:
        return str(summary)

    count_3d = safe_float(_external_attr(news_context, "news_count_3d")) or 0.0
    if count_3d <= 0:
        return ""

    sentiment = safe_float(_external_attr(news_context, "avg_sentiment_3d"))
    importance = safe_float(_external_attr(news_context, "max_importance_3d"))
    negative = bool(_external_attr(news_context, "negative_news_flag"))
    positive = bool(_external_attr(news_context, "positive_news_flag"))

    if negative and importance is not None and importance >= 0.7:
        return "Recent manual news includes high-importance negative items."
    if positive and importance is not None and importance >= 0.7:
        return "Recent manual news includes high-importance positive items."
    if negative:
        return "Recent manual news flow is leaning negative."
    if positive:
        return "Recent manual news flow is leaning positive."
    if sentiment is not None:
        return f"Recent manual news count is {int(count_3d)} with sentiment {sentiment:.2f}."
    return f"Recent manual news count is {int(count_3d)}."


def _news_direction(news_context: object | None) -> str:
    if news_context is None:
        return "unknown"

    negative_shock = safe_float(_external_attr(news_context, "negative_news_shock")) or 0.0
    positive_shock = safe_float(_external_attr(news_context, "positive_news_shock")) or 0.0
    weighted = safe_float(_external_attr(news_context, "importance_weighted_sentiment"))
    trend = safe_float(_external_attr(news_context, "sentiment_trend_3d"))
    sector = safe_float(_external_attr(news_context, "sector_news_strength"))

    down_score = sum(
        (
            negative_shock >= 1.0,
            weighted is not None and weighted <= -0.25,
            trend is not None and trend <= -0.25,
            sector is not None and sector <= -0.25,
        )
    )
    up_score = sum(
        (
            positive_shock >= 1.0,
            weighted is not None and weighted >= 0.25,
            trend is not None and trend >= 0.25,
            sector is not None and sector >= 0.25,
        )
    )
    if up_score == 0 and down_score == 0:
        return "unknown"
    if up_score >= down_score + 1:
        return "up"
    if down_score >= up_score + 1:
        return "down"
    return "mixed"


def _news_alignment_text(
    prediction_direction: str,
    shap_direction: str,
    similarity_direction: str,
    news_direction: str,
) -> str:
    if news_direction == "unknown":
        return ""
    direction_text = _direction_text(news_direction)
    parts = [f"News direction is {direction_text}."]
    if news_direction == prediction_direction:
        parts.append("It supports the model prediction.")
    elif _conflicts(prediction_direction, news_direction):
        parts.append("It conflicts with the model prediction, so confidence should be reduced.")

    if news_direction == shap_direction:
        parts.append("It also aligns with SHAP local drivers.")
    elif _conflicts(shap_direction, news_direction):
        parts.append("It does not align with SHAP local drivers.")

    if news_direction == similarity_direction:
        parts.append("Similar-pattern evidence points in the same direction.")
    elif _conflicts(similarity_direction, news_direction):
        parts.append("Similar-pattern evidence points in the opposite direction.")
    return " ".join(parts)


def _narrative_text(narrative_context: object | None) -> str:
    if narrative_context is None:
        return ""
    summary = _external_attr(narrative_context, "summary")
    if summary:
        return str(summary)
    label = _external_attr(narrative_context, "dominant_label") or "None"
    sentiment = safe_float(_external_attr(narrative_context, "dominant_sentiment"))
    importance = safe_float(_external_attr(narrative_context, "dominant_importance"))
    if label == "None" or sentiment is None:
        return ""
    return (
        f"Dominant market narrative is {label}; sentiment is {_format_number(sentiment)} "
        f"and importance is {_format_number(importance)}."
    )


def _narrative_direction(narrative_context: object | None) -> str:
    if narrative_context is None:
        return "unknown"
    bullish = safe_float(_external_attr(narrative_context, "bullish_pressure"))
    bearish = safe_float(_external_attr(narrative_context, "bearish_pressure"))
    sentiment = safe_float(_external_attr(narrative_context, "dominant_sentiment"))
    if bullish is None and bearish is None and sentiment is None:
        return "unknown"
    bullish = bullish or 0.0
    bearish = bearish or 0.0
    if bullish >= bearish + 0.15 and bullish >= 0.25:
        return "up"
    if bearish >= bullish + 0.15 and bearish >= 0.25:
        return "down"
    if sentiment is not None and sentiment >= 0.25:
        return "up"
    if sentiment is not None and sentiment <= -0.25:
        return "down"
    return "mixed"


def _cross_ticker_text(
    cross_ticker_patterns: pd.DataFrame | None,
    cross_ticker_summary: pd.DataFrame | None,
) -> str:
    if cross_ticker_patterns is None or cross_ticker_patterns.empty:
        return ""
    best = cross_ticker_patterns.iloc[0]
    ticker = best.get("ticker_name") or best.get("ticker") or "peer"
    start = pd.to_datetime(best.get("similar_start_date")).strftime("%Y-%m-%d")
    end = pd.to_datetime(best.get("similar_end_date")).strftime("%Y-%m-%d")
    similarity = _format_number(best.get("similarity"), digits=3)
    row_5d = _summary_row(cross_ticker_summary, "5D") if cross_ticker_summary is not None else None
    toyota_5d = _summary_cross_return(row_5d, "toyota_average_return")
    peer_5d = _summary_cross_return(row_5d, "peer_average_return")
    return (
        f"Cross-ticker similarity: current Toyota is closest to {ticker} "
        f"from {start} to {end} (similarity {similarity}). "
        f"After similar peer windows, peer 5D average return was {peer_5d}; "
        f"Toyota same-date 5D average return was {toyota_5d}."
    )


def _cross_ticker_direction(cross_ticker_summary: pd.DataFrame | None) -> str:
    if cross_ticker_summary is None:
        return "unknown"
    row = _summary_row(cross_ticker_summary, "5D") or _summary_row(cross_ticker_summary, "1D")
    if row is None:
        return "unknown"
    toyota_return = safe_float(row.get("toyota_average_return"))
    peer_return = safe_float(row.get("peer_average_return"))
    value = toyota_return if toyota_return is not None else peer_return
    if value is None:
        return "unknown"
    if value >= 0.02:
        return "up"
    if value <= -0.02:
        return "down"
    return "mixed"


def _lead_lag_text(lead_lag_summary: pd.DataFrame | None) -> str:
    if lead_lag_summary is None or lead_lag_summary.empty:
        return ""
    best = lead_lag_summary.iloc[0]
    name = best.get("name") or best.get("symbol") or "candidate"
    lag = int(safe_float(best.get("lag")) or 0)
    correlation = _format_number(best.get("correlation"), digits=3)
    agreement = _format_percent(best.get("direction_agreement"))
    current_signal = str(best.get("current_signal") or "neutral")
    if current_signal == "bullish":
        signal_text = "The latest move is a bullish lead-lag signal."
    elif current_signal == "bearish":
        signal_text = "The latest move is a bearish lead-lag signal."
    else:
        signal_text = "The latest move is not a strong current lead-lag signal."
    return (
        f"Lead-lag context: {name} has the strongest {lag}D exploratory relationship "
        f"with Toyota future returns (correlation {correlation}, direction agreement {agreement}). "
        f"{signal_text}"
    )


def _lead_lag_direction(lead_lag_summary: pd.DataFrame | None) -> str:
    if lead_lag_summary is None or lead_lag_summary.empty:
        return "unknown"
    if "current_signal" not in lead_lag_summary.columns:
        return "mixed"
    summary = lead_lag_summary.copy()
    summary["score"] = pd.to_numeric(summary.get("score"), errors="coerce").fillna(0.0)
    current = summary[summary["current_signal"].isin(["bullish", "bearish"])]
    if current.empty:
        return "mixed"
    bullish = float(current.loc[current["current_signal"] == "bullish", "score"].sum())
    bearish = float(current.loc[current["current_signal"] == "bearish", "score"].sum())
    if bullish == 0 and bearish == 0:
        return "unknown"
    if bullish >= bearish * 1.2:
        return "up"
    if bearish >= bullish * 1.2:
        return "down"
    return "mixed"


def _anomaly_text(anomaly_context: object | None) -> str:
    if anomaly_context is None:
        return ""
    score = safe_float(_external_attr(anomaly_context, "score"))
    level = _external_attr(anomaly_context, "level") or "Unknown"
    if score is None:
        return ""
    types = _external_attr(anomaly_context, "types") or []
    if isinstance(types, str):
        types = [part.strip() for part in types.split("|") if part.strip()]
    summary = _external_attr(anomaly_context, "summary")
    if summary:
        return str(summary)
    if score < 40:
        return f"Anomaly context is quiet: score is {int(score)}/100 ({level})."
    type_text = ", ".join(types[:3]) if types else "mixed anomaly"
    return f"Anomaly context is elevated: score is {int(score)}/100 ({level}); detected {type_text}."


def _anomaly_direction(anomaly_context: object | None) -> str:
    if anomaly_context is None:
        return "unknown"
    score = safe_float(_external_attr(anomaly_context, "score")) or 0.0
    if score < 40:
        return "unknown"
    types = _external_attr(anomaly_context, "types") or []
    if isinstance(types, str):
        types = [part.strip() for part in types.split("|") if part.strip()]
    if "Price Shock" in types:
        return "mixed"
    bearish = {
        "Volatility Spike",
        "Sector Divergence",
        "Market Divergence",
        "News Shock",
        "Narrative Shift",
    }
    if any(item in bearish for item in types):
        return "down"
    if "Volume Spike" in types:
        return "up"
    return "mixed"


def _external_attr(item: object, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _regime_attr(item: object, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _risk_text(risk_scores: dict[str, object] | None) -> str:
    if not risk_scores:
        return ""
    surge = risk_scores.get("surge")
    crash = risk_scores.get("crash")
    if surge is None or crash is None:
        return ""

    surge_score = _risk_attr(surge, "score")
    crash_score = _risk_attr(crash, "score")
    surge_level = _risk_attr(surge, "level") or "Unknown"
    crash_level = _risk_attr(crash, "level") or "Unknown"
    if surge_score is None or crash_score is None:
        return ""

    if crash_level == "High" and surge_level != "High":
        stance = "Crash early-warning signals are stronger than surge signals."
    elif surge_level == "High" and crash_level != "High":
        stance = "Surge early-warning signals are stronger than crash signals."
    elif surge_level == "High" and crash_level == "High":
        stance = "Both surge and crash early-warning signals are elevated, so volatility risk is high."
    else:
        stance = "Early-warning scores do not show an extreme signal yet."

    return (
        f"Surge Risk Score is {surge_score}/100 ({surge_level}) and "
        f"Crash Risk Score is {crash_score}/100 ({crash_level}). {stance}"
    )


def _risk_attr(item: object, name: str):
    if isinstance(item, dict):
        return item.get(name)
    return getattr(item, name, None)


def _summary_text(
    prediction_direction: str,
    confidence_level: str,
    probability_text: str,
    shap_text: str,
    similarity_text: str,
    external_text: str,
    event_text: str,
    news_text: str,
    news_alignment_text: str,
    narrative_text: str,
    cross_ticker_text: str,
    lead_lag_text: str,
    anomaly_text: str,
    regime_text: str,
    risk_text: str,
    conflict_text: str,
) -> str:
    parts = [
        probability_text,
        f"統合信頼度は{confidence_level}です。",
        shap_text,
        similarity_text,
    ]
    if external_text:
        parts.append(external_text)
    if event_text:
        parts.append(event_text)
    if news_text:
        parts.append(news_text)
    if news_alignment_text:
        parts.append(news_alignment_text)
    if narrative_text:
        parts.append(narrative_text)
    if cross_ticker_text:
        parts.append(cross_ticker_text)
    if lead_lag_text:
        parts.append(lead_lag_text)
    if anomaly_text:
        parts.append(anomaly_text)
    if regime_text:
        parts.append(regime_text)
    if risk_text:
        parts.append(risk_text)
    if conflict_text:
        parts.append(conflict_text)
    elif confidence_level == "High":
        parts.append("予測確率、SHAP、類似局面が同方向で、根拠は比較的そろっています。")
    elif confidence_level == "Medium":
        parts.append("根拠は一部そろっていますが、強い確信まではありません。")
    elif prediction_direction in {"up", "down"}:
        parts.append("根拠の足並みが弱いため、低めの確信として扱います。")
    return " ".join(parts)


def _summary_row(summary: pd.DataFrame, horizon: str) -> dict[str, object] | None:
    if summary.empty or "horizon" not in summary.columns:
        return None
    rows = summary[summary["horizon"] == horizon]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _conflicts(prediction_direction: str, evidence_direction: str) -> bool:
    if prediction_direction not in {"up", "down"}:
        return False
    return evidence_direction in {"up", "down"} and evidence_direction != prediction_direction


def _feature_name(item: dict[str, object]) -> str:
    return str(item.get("label") or item.get("feature") or "unknown")


def _direction_text(direction: str) -> str:
    if direction == "up":
        return "上昇寄り"
    if direction == "down":
        return "下落寄り"
    return "方向不明"


def _opposite_direction_text(direction: str) -> str:
    if direction == "up":
        return "下落"
    if direction == "down":
        return "上昇"
    return "反対"


def _summary_return(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    return _format_percent(row.get("average_return"))


def _summary_positive_rate(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    return _format_percent(row.get("positive_rate"))


def _summary_cross_return(row: dict[str, object] | None, key: str) -> str:
    if row is None:
        return "-"
    return _format_percent(row.get(key))


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _format_number(value, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"
