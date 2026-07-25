from __future__ import annotations

from stock_ai.utils.safe_cast import safe_float


FEATURE_LABELS = {
    "open": "始値",
    "high": "高値",
    "low": "安値",
    "close": "終値",
    "volume": "出来高",
    "return_1d": "日次リターン",
    "volume_change_1d": "出来高変化率",
    "intraday_return": "日中リターン",
    "high_low_range": "日中値幅",
    "close_position": "終値位置",
    "rsi_14": "RSI",
    "macd": "MACD",
    "macd_signal": "MACDシグナル",
    "macd_hist": "MACDヒストグラム",
    "volatility_20": "ボラティリティ",
    "bb_percent_b": "ボリンジャーバンド位置",
}


def normalize_shap_contributions(items: list[dict[str, object]]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for item in items:
        contribution = safe_float(item.get("contribution", item.get("shap_value")))
        if contribution is None:
            continue
        feature = item.get("feature")
        if feature is None:
            continue
        value = safe_float(item.get("value"))
        direction = "up" if contribution >= 0 else "down"
        normalized.append(
            {
                "feature": str(feature),
                "label": feature_label(str(feature)),
                "value": value,
                "shap_value": contribution,
                "contribution": contribution,
                "abs_contribution": abs(contribution),
                "direction": direction,
            }
        )
    normalized.sort(key=lambda row: row["abs_contribution"], reverse=True)
    return normalized


def split_shap_contributions(
    items: list[dict[str, object]],
    top_n: int = 10,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    normalized = normalize_shap_contributions(items)
    upward = [item for item in normalized if item["shap_value"] > 0]
    downward = [item for item in normalized if item["shap_value"] < 0]
    upward.sort(key=lambda row: row["shap_value"], reverse=True)
    downward.sort(key=lambda row: row["shap_value"])
    return upward[:top_n], downward[:top_n]


def build_shap_reason_text(
    items: list[dict[str, object]],
    prediction_label: str,
    top_n: int = 3,
) -> str:
    upward, downward = split_shap_contributions(items, top_n=top_n)
    parts: list[str] = []

    if prediction_label == "up":
        parts.append("今回の予測は上昇方向の寄与が強い特徴量を重視しています。")
    elif prediction_label == "down":
        parts.append("今回の予測は下落方向の寄与が強い特徴量を重視しています。")
    else:
        parts.append("今回の予測方向は不明ですが、SHAP寄与から判断材料を確認できます。")

    for item in upward:
        parts.append(_feature_sentence(item, direction_label="上昇方向"))
    for item in downward:
        parts.append(_feature_sentence(item, direction_label="下落方向"))

    return " ".join(parts)


def feature_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    event_label = _event_feature_label(feature)
    if event_label:
        return event_label
    news_label = _news_feature_label(feature)
    if news_label:
        return news_label
    narrative_label = _narrative_feature_label(feature)
    if narrative_label:
        return narrative_label
    external_label = _external_feature_label(feature)
    if external_label:
        return external_label
    if feature.startswith("ma_"):
        return feature.upper().replace("_", " ")
    if feature.startswith("close_vs_ma_"):
        return feature.replace("close_vs_ma_", "MA") + "乖離率"
    return feature


def _event_feature_label(feature: str) -> str:
    labels = {
        "days_to_earnings": "Days to earnings",
        "days_after_earnings": "Days after earnings",
        "earnings_week_flag": "Earnings week",
        "earnings_day_flag": "Earnings day",
        "post_earnings_mode": "Post-earnings mode",
        "earnings_event_state_code": "Earnings event state",
        "earnings_pre_volatility": "Pre-earnings volatility",
        "earnings_post_gap": "Post-earnings gap",
        "earnings_volume_spike": "Earnings volume spike",
        "earnings_after_trend_5": "Post-earnings trend",
    }
    return labels.get(feature, "")


def _news_feature_label(feature: str) -> str:
    labels = {
        "news_count_1d": "News count 1D",
        "news_count_3d": "News count 3D",
        "avg_sentiment_1d": "News sentiment 1D",
        "avg_sentiment_3d": "News sentiment 3D",
        "max_importance_3d": "News importance 3D",
        "negative_news_flag": "Negative news flag",
        "positive_news_flag": "Positive news flag",
        "sentiment_trend_3d": "News sentiment trend 3D",
        "sentiment_trend_7d": "News sentiment trend 7D",
        "importance_weighted_sentiment": "News weighted sentiment",
        "negative_news_shock": "Negative news shock",
        "positive_news_shock": "Positive news shock",
        "sector_news_strength": "Sector news strength",
    }
    return labels.get(feature, "")


def _narrative_feature_label(feature: str) -> str:
    labels = {
        "dominant_narrative_code": "Dominant narrative",
        "dominant_narrative_sentiment": "Narrative sentiment",
        "dominant_narrative_importance": "Narrative importance",
        "dominant_narrative_trend": "Narrative trend",
        "narrative_bullish_pressure": "Narrative bullish pressure",
        "narrative_bearish_pressure": "Narrative bearish pressure",
    }
    if feature in labels:
        return labels[feature]
    if feature.startswith("narrative_theme_"):
        return feature.removeprefix("narrative_theme_").replace("_", " ").title()
    return ""


def _external_feature_label(feature: str) -> str:
    prefixes = {
        "usdjpy_": "USDJPY",
        "n225_": "Nikkei 225",
        "topix_": "TOPIX",
        "honda_": "Honda",
        "nissan_": "Nissan",
        "tesla_": "Tesla",
        "byd_": "BYD",
        "sector_": "Auto sector",
    }
    for prefix, label in prefixes.items():
        if feature.startswith(prefix):
            suffix = feature.removeprefix(prefix).replace("_", " ")
            return f"{label} {suffix}"
    return ""


def _feature_sentence(item: dict[str, object], direction_label: str) -> str:
    label = str(item.get("label") or item.get("feature"))
    feature = str(item.get("feature") or "")
    value = safe_float(item.get("value"))
    shap_value = safe_float(item.get("shap_value"))
    value_phrase = _value_phrase(feature, value)
    shap_phrase = "" if shap_value is None else f"SHAP値 {shap_value:.4f}で"
    return f"{label}{value_phrase}が{shap_phrase}{direction_label}へ寄与しています。"


def _value_phrase(feature: str, value: float | None) -> str:
    if value is None:
        return ""

    if feature == "rsi_14":
        if value >= 70:
            return f"が高めです（{value:.1f}）"
        if value <= 30:
            return f"が低めです（{value:.1f}）"
        return f"は中立圏です（{value:.1f}）"

    if "return" in feature or "change" in feature or "volatility" in feature or "close_vs" in feature:
        return f"は{value * 100:.2f}%です"

    return f"は{value:.4f}です"
