from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.news.manual_csv import read_news_items
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class NarrativeTheme:
    key: str
    label: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeContext:
    dominant_key: str
    dominant_label: str
    dominant_sentiment: float | None
    dominant_importance: float | None
    dominant_trend: float | None
    bullish_pressure: float | None
    bearish_pressure: float | None
    summary: str
    themes: list[dict[str, object]]
    metrics: dict[str, float | None]


THEMES: tuple[NarrativeTheme, ...] = (
    NarrativeTheme("yen_weakness", "Yen Weakness", ("円安", "yen weak", "weak yen", "usdjpy")),
    NarrativeTheme("ev", "EV", ("ev", "electric vehicle", "battery", "電気自動車", "bev")),
    NarrativeTheme("semiconductor", "Semiconductor", ("semiconductor", "chip", "半導体")),
    NarrativeTheme("interest_rates", "Interest Rates", ("rate", "rates", "金利", "boj", "fed")),
    NarrativeTheme("recession", "Recession", ("recession", "景気後退", "slowdown", "減速")),
    NarrativeTheme("raw_materials", "Raw Materials", ("原材料", "commodity", "steel", "oil", "crude")),
    NarrativeTheme("geopolitics", "Geopolitics", ("地政学", "war", "tariff", "sanction", "trade tension")),
    NarrativeTheme("auto_demand", "Auto Demand", ("自動車需要", "sales", "demand", "production")),
    NarrativeTheme("china_market", "China Market", ("中国", "china", "chinese")),
)

NARRATIVE_FEATURE_COLUMNS = (
    "dominant_narrative_code",
    "dominant_narrative_sentiment",
    "dominant_narrative_importance",
    "dominant_narrative_trend",
    "narrative_bullish_pressure",
    "narrative_bearish_pressure",
)
NARRATIVE_STRING_COLUMNS = ("dominant_narrative", "dominant_narrative_label")
THEME_METRIC_SUFFIXES = ("count_3d", "sentiment_3d", "importance_3d", "trend_3d")
THEME_CODE = {theme.key: index + 1 for index, theme in enumerate(THEMES)}
THEME_LABEL = {theme.key: theme.label for theme in THEMES}


def add_narrative_features(data: pd.DataFrame, news_dir: str | Path, ticker: str) -> pd.DataFrame:
    """Add trailing market narrative features from manual news CSV."""
    result = data.copy()
    if result.empty or "date" not in result.columns:
        return result

    drop_columns = [
        column
        for column in result.columns
        if column in NARRATIVE_FEATURE_COLUMNS
        or column in NARRATIVE_STRING_COLUMNS
        or column.startswith("narrative_theme_")
    ]
    result = result.drop(columns=drop_columns)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.sort_values("date").reset_index(drop=True)

    news = read_news_items(news_dir, ticker)
    if news.empty:
        return _add_empty_narrative_features(result)

    tagged = tag_news_narratives(news)
    if tagged.empty:
        return _add_empty_narrative_features(result)

    features = _build_daily_narrative_features(result["date"], tagged)
    merged = result.merge(features, on="date", how="left")
    for column in NARRATIVE_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    for column in NARRATIVE_STRING_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
        merged[column] = merged[column].fillna("").astype(str)
    for column in [c for c in merged.columns if c.startswith("narrative_theme_")]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def build_narrative_context(data: pd.DataFrame) -> NarrativeContext:
    if data.empty:
        return _empty_context("Narrative data is not available yet.")

    latest = data.sort_values("date").iloc[-1] if "date" in data.columns else data.iloc[-1]
    dominant_key = str(latest.get("dominant_narrative") or "")
    dominant_label = str(latest.get("dominant_narrative_label") or "")
    if not dominant_key:
        dominant_label = "None"

    metrics = {
        "dominant_narrative_code": _value(latest, "dominant_narrative_code"),
        "dominant_narrative_sentiment": _value(latest, "dominant_narrative_sentiment"),
        "dominant_narrative_importance": _value(latest, "dominant_narrative_importance"),
        "dominant_narrative_trend": _value(latest, "dominant_narrative_trend"),
        "narrative_bullish_pressure": _value(latest, "narrative_bullish_pressure"),
        "narrative_bearish_pressure": _value(latest, "narrative_bearish_pressure"),
    }
    themes = _theme_rows(latest)
    return NarrativeContext(
        dominant_key=dominant_key,
        dominant_label=dominant_label or "None",
        dominant_sentiment=metrics["dominant_narrative_sentiment"],
        dominant_importance=metrics["dominant_narrative_importance"],
        dominant_trend=metrics["dominant_narrative_trend"],
        bullish_pressure=metrics["narrative_bullish_pressure"],
        bearish_pressure=metrics["narrative_bearish_pressure"],
        summary=_summary_text(dominant_label, metrics, themes),
        themes=themes,
        metrics=metrics,
    )


def tag_news_narratives(news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        return pd.DataFrame(columns=[*news.columns, "narrative_key", "narrative_label"])

    rows: list[dict[str, object]] = []
    for _, item in news.iterrows():
        text = _combined_text(item)
        matched = [theme for theme in THEMES if _matches_theme(text, theme)]
        for theme in matched:
            row = item.to_dict()
            row["narrative_key"] = theme.key
            row["narrative_label"] = theme.label
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=[*news.columns, "narrative_key", "narrative_label"])
    return pd.DataFrame(rows)


def _build_daily_narrative_features(stock_dates: pd.Series, tagged_news: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(stock_dates, errors="coerce").dt.normalize()
    start = dates.min()
    end = dates.max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["date", *NARRATIVE_FEATURE_COLUMNS, *NARRATIVE_STRING_COLUMNS])

    tagged = tagged_news.copy()
    tagged["date"] = pd.to_datetime(tagged["date"], errors="coerce").dt.normalize()
    tagged["sentiment_score"] = pd.to_numeric(tagged["sentiment_score"], errors="coerce").fillna(0.0)
    tagged["importance_score"] = pd.to_numeric(tagged["importance_score"], errors="coerce").fillna(0.0)
    tagged["weighted_sentiment"] = tagged["sentiment_score"] * tagged["importance_score"]

    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    theme_frames: list[pd.DataFrame] = []
    for theme in THEMES:
        theme_daily = _theme_daily_frame(calendar, tagged, theme)
        theme_frames.append(theme_daily)
    result = calendar.copy()
    for frame in theme_frames:
        result = result.merge(frame, on="date", how="left")
    result = result.fillna(0.0)

    result = _add_dominant_narrative_columns(result)
    stock_date_frame = pd.DataFrame({"date": dates})
    return stock_date_frame.merge(result, on="date", how="left")


def _theme_daily_frame(calendar: pd.DataFrame, tagged: pd.DataFrame, theme: NarrativeTheme) -> pd.DataFrame:
    theme_news = tagged[tagged["narrative_key"] == theme.key]
    base_columns = ["date", *(_theme_column(theme.key, suffix) for suffix in THEME_METRIC_SUFFIXES)]
    if theme_news.empty:
        empty = calendar[["date"]].copy()
        for column in base_columns[1:]:
            empty[column] = 0.0
        return empty

    daily = (
        theme_news.groupby("date")
        .agg(
            count=("title", "size"),
            sentiment_sum=("sentiment_score", "sum"),
            weighted_sentiment_sum=("weighted_sentiment", "sum"),
            importance_sum=("importance_score", "sum"),
            importance=("importance_score", "max"),
        )
        .reset_index()
    )
    frame = calendar.merge(daily, on="date", how="left").fillna(0.0)
    count_3d = frame["count"].rolling(window=3, min_periods=1).sum()
    sentiment_sum_3d = frame["sentiment_sum"].rolling(window=3, min_periods=1).sum()
    importance_sum_3d = frame["importance_sum"].rolling(window=3, min_periods=1).sum()
    weighted_sum_3d = frame["weighted_sentiment_sum"].rolling(window=3, min_periods=1).sum()
    sentiment_3d = sentiment_sum_3d / count_3d.replace(0, pd.NA)
    weighted_sentiment_3d = weighted_sum_3d / importance_sum_3d.replace(0, pd.NA)
    previous_sentiment_3d = sentiment_3d.shift(3)

    result = frame[["date"]].copy()
    result[_theme_column(theme.key, "count_3d")] = count_3d.fillna(0.0)
    result[_theme_column(theme.key, "sentiment_3d")] = weighted_sentiment_3d.fillna(
        sentiment_3d,
    ).fillna(0.0)
    result[_theme_column(theme.key, "importance_3d")] = (
        frame["importance"].rolling(window=3, min_periods=1).max().fillna(0.0)
    )
    result[_theme_column(theme.key, "trend_3d")] = (sentiment_3d - previous_sentiment_3d).fillna(0.0)
    return result


def _add_dominant_narrative_columns(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    dominant_keys: list[str] = []
    dominant_labels: list[str] = []
    dominant_codes: list[int] = []
    dominant_sentiments: list[float] = []
    dominant_importances: list[float] = []
    dominant_trends: list[float] = []
    bullish_pressures: list[float] = []
    bearish_pressures: list[float] = []

    for _, row in result.iterrows():
        theme_rows = _theme_rows(row)
        active = [theme for theme in theme_rows if safe_float(theme.get("news_count")) not in (None, 0.0)]
        if not active:
            dominant_keys.append("")
            dominant_labels.append("")
            dominant_codes.append(0)
            dominant_sentiments.append(0.0)
            dominant_importances.append(0.0)
            dominant_trends.append(0.0)
            bullish_pressures.append(0.0)
            bearish_pressures.append(0.0)
            continue

        dominant = max(
            active,
            key=lambda item: (
                safe_float(item.get("importance")) or 0.0,
                safe_float(item.get("news_count")) or 0.0,
            ),
        )
        dominant_key = str(dominant["key"])
        dominant_sentiment = safe_float(dominant.get("sentiment")) or 0.0
        dominant_importance = safe_float(dominant.get("importance")) or 0.0
        dominant_trend = safe_float(dominant.get("trend")) or 0.0
        bullish_pressure = sum(_theme_pressure(theme, direction="bullish") for theme in active)
        bearish_pressure = sum(_theme_pressure(theme, direction="bearish") for theme in active)
        dominant_keys.append(dominant_key)
        dominant_labels.append(THEME_LABEL.get(dominant_key, dominant_key))
        dominant_codes.append(THEME_CODE.get(dominant_key, 0))
        dominant_sentiments.append(dominant_sentiment)
        dominant_importances.append(dominant_importance)
        dominant_trends.append(dominant_trend)
        bullish_pressures.append(float(min(1.0, bullish_pressure)))
        bearish_pressures.append(float(min(1.0, bearish_pressure)))

    result["dominant_narrative"] = dominant_keys
    result["dominant_narrative_label"] = dominant_labels
    result["dominant_narrative_code"] = dominant_codes
    result["dominant_narrative_sentiment"] = dominant_sentiments
    result["dominant_narrative_importance"] = dominant_importances
    result["dominant_narrative_trend"] = dominant_trends
    result["narrative_bullish_pressure"] = bullish_pressures
    result["narrative_bearish_pressure"] = bearish_pressures
    return result


def _theme_rows(row) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for theme in THEMES:
        count = _row_value(row, _theme_column(theme.key, "count_3d"))
        sentiment = _row_value(row, _theme_column(theme.key, "sentiment_3d"))
        importance = _row_value(row, _theme_column(theme.key, "importance_3d"))
        trend = _row_value(row, _theme_column(theme.key, "trend_3d"))
        if count is None and sentiment is None and importance is None and trend is None:
            continue
        rows.append(
            {
                "key": theme.key,
                "label": theme.label,
                "news_count": count or 0.0,
                "sentiment": sentiment or 0.0,
                "importance": importance or 0.0,
                "trend": trend or 0.0,
            }
        )
    rows.sort(
        key=lambda item: (
            safe_float(item.get("importance")) or 0.0,
            safe_float(item.get("news_count")) or 0.0,
        ),
        reverse=True,
    )
    return rows


def _theme_pressure(theme: dict[str, object], direction: str) -> float:
    sentiment = safe_float(theme.get("sentiment")) or 0.0
    importance = safe_float(theme.get("importance")) or 0.0
    count = safe_float(theme.get("news_count")) or 0.0
    if count <= 0:
        return 0.0
    if direction == "bullish" and sentiment <= 0:
        return 0.0
    if direction == "bearish" and sentiment >= 0:
        return 0.0
    return min(1.0, abs(sentiment) * max(0.2, importance) * min(1.0, count / 3.0))


def _summary_text(
    dominant_label: str,
    metrics: dict[str, float | None],
    themes: list[dict[str, object]],
) -> str:
    active = [theme for theme in themes if (safe_float(theme.get("news_count")) or 0.0) > 0]
    if not active:
        return "No active market narrative is detected from manual news."
    sentiment = metrics.get("dominant_narrative_sentiment") or 0.0
    importance = metrics.get("dominant_narrative_importance") or 0.0
    trend = metrics.get("dominant_narrative_trend") or 0.0
    direction = "positive" if sentiment > 0.15 else "negative" if sentiment < -0.15 else "neutral"
    return (
        f"Dominant narrative is {dominant_label} with {direction} sentiment "
        f"({sentiment:.2f}), importance {importance:.2f}, and trend {trend:.2f}."
    )


def _add_empty_narrative_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in NARRATIVE_FEATURE_COLUMNS:
        result[column] = 0.0
    for column in NARRATIVE_STRING_COLUMNS:
        result[column] = ""
    for theme in THEMES:
        for suffix in THEME_METRIC_SUFFIXES:
            result[_theme_column(theme.key, suffix)] = 0.0
    return result


def _empty_context(summary: str) -> NarrativeContext:
    return NarrativeContext(
        dominant_key="",
        dominant_label="None",
        dominant_sentiment=None,
        dominant_importance=None,
        dominant_trend=None,
        bullish_pressure=None,
        bearish_pressure=None,
        summary=summary,
        themes=[],
        metrics={},
    )


def _combined_text(item: pd.Series) -> str:
    fields = ("title", "summary", "category")
    return " ".join(str(item.get(field, "") or "") for field in fields).lower()


def _matches_theme(text: str, theme: NarrativeTheme) -> bool:
    return any(keyword.lower() in text for keyword in theme.keywords)


def _theme_column(key: str, suffix: str) -> str:
    return f"narrative_theme_{key}_{suffix}"


def _value(row: pd.Series, key: str) -> float | None:
    return safe_float(row.get(key))


def _row_value(row, key: str) -> float | None:
    if isinstance(row, pd.Series):
        return safe_float(row.get(key))
    return safe_float(row[key]) if key in row else None
