from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.utils.safe_cast import safe_float


NEWS_DATA_SUBDIR = "news"
NEWS_FEATURE_COLUMNS = (
    "news_count_1d",
    "news_count_3d",
    "avg_sentiment_1d",
    "avg_sentiment_3d",
    "max_importance_3d",
    "negative_news_flag",
    "positive_news_flag",
    "sentiment_trend_3d",
    "sentiment_trend_7d",
    "importance_weighted_sentiment",
    "negative_news_shock",
    "positive_news_shock",
    "sector_news_strength",
)
NEWS_DISPLAY_COLUMNS = (
    "date",
    "ticker",
    "title",
    "source",
    "url",
    "source_credibility_score",
    "source_market_impact_weight",
    "source_type",
    "category",
    "sentiment_score",
    "importance_score",
    "sentiment_reason",
    "importance_reason",
    "summary",
    "fetched_at",
    "query_profile",
)
NEWS_INTERNAL_COLUMNS = (
    "weighted_sentiment",
    "sector_importance",
    "sector_weighted_sentiment",
)


@dataclass(frozen=True)
class NewsContext:
    news_count_1d: int
    news_count_3d: int
    avg_sentiment_1d: float | None
    avg_sentiment_3d: float | None
    max_importance_3d: float | None
    negative_news_flag: bool
    positive_news_flag: bool
    sentiment_trend_3d: float | None
    sentiment_trend_7d: float | None
    importance_weighted_sentiment: float | None
    negative_news_shock: bool
    positive_news_shock: bool
    sector_news_strength: float | None
    summary: str
    impact_summary: str
    recent_items: list[dict[str, object]]
    important_items: list[dict[str, object]]
    sentiment_history: list[dict[str, object]]
    metrics: dict[str, float | None]


def add_news_features(data: pd.DataFrame, news_dir: str | Path, ticker: str) -> pd.DataFrame:
    """Add trailing manual-news features using only news dated on or before each row."""
    result = data.copy()
    if result.empty or "date" not in result.columns:
        return result

    result = result.drop(columns=[c for c in NEWS_FEATURE_COLUMNS if c in result.columns])
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.normalize()
    result = result.sort_values("date").reset_index(drop=True)

    news = read_news_items(news_dir, ticker)
    if news.empty:
        return _add_empty_news_features(result)

    features = _build_daily_news_features(result["date"], news)
    merged = result.merge(features, on="date", how="left")
    for column in NEWS_FEATURE_COLUMNS:
        if column not in merged.columns:
            merged[column] = 0.0
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    return merged


def build_news_context(data: pd.DataFrame, news_dir: str | Path, ticker: str) -> NewsContext:
    if data.empty:
        return _empty_context("News data is not available yet.")

    latest = data.sort_values("date").iloc[-1] if "date" in data.columns else data.iloc[-1]
    latest_date = pd.to_datetime(latest.get("date"), errors="coerce")
    metrics = {
        "news_count_1d": _value(latest, "news_count_1d"),
        "news_count_3d": _value(latest, "news_count_3d"),
        "avg_sentiment_1d": _value(latest, "avg_sentiment_1d"),
        "avg_sentiment_3d": _value(latest, "avg_sentiment_3d"),
        "max_importance_3d": _value(latest, "max_importance_3d"),
        "negative_news_flag": _value(latest, "negative_news_flag"),
        "positive_news_flag": _value(latest, "positive_news_flag"),
        "sentiment_trend_3d": _value(latest, "sentiment_trend_3d"),
        "sentiment_trend_7d": _value(latest, "sentiment_trend_7d"),
        "importance_weighted_sentiment": _value(latest, "importance_weighted_sentiment"),
        "negative_news_shock": _value(latest, "negative_news_shock"),
        "positive_news_shock": _value(latest, "positive_news_shock"),
        "sector_news_strength": _value(latest, "sector_news_strength"),
    }
    news = read_news_items(news_dir, ticker)
    recent_items = _recent_news_items(news, latest_date, days=7)
    important_items = _important_news_items(news, latest_date, days=14)
    sentiment_history = _sentiment_history(data, days=14)
    count_1d = int(metrics["news_count_1d"] or 0)
    count_3d = int(metrics["news_count_3d"] or 0)
    negative = bool(metrics["negative_news_flag"] == 1)
    positive = bool(metrics["positive_news_flag"] == 1)
    negative_shock = bool(metrics["negative_news_shock"] == 1)
    positive_shock = bool(metrics["positive_news_shock"] == 1)

    return NewsContext(
        news_count_1d=count_1d,
        news_count_3d=count_3d,
        avg_sentiment_1d=metrics["avg_sentiment_1d"],
        avg_sentiment_3d=metrics["avg_sentiment_3d"],
        max_importance_3d=metrics["max_importance_3d"],
        negative_news_flag=negative,
        positive_news_flag=positive,
        sentiment_trend_3d=metrics["sentiment_trend_3d"],
        sentiment_trend_7d=metrics["sentiment_trend_7d"],
        importance_weighted_sentiment=metrics["importance_weighted_sentiment"],
        negative_news_shock=negative_shock,
        positive_news_shock=positive_shock,
        sector_news_strength=metrics["sector_news_strength"],
        summary=_summary_text(count_3d, metrics, negative, positive),
        impact_summary=_impact_summary_text(count_3d, metrics, negative_shock, positive_shock),
        recent_items=recent_items,
        important_items=important_items,
        sentiment_history=sentiment_history,
        metrics=metrics,
    )


def read_news_items(news_dir: str | Path, ticker: str) -> pd.DataFrame:
    for path in news_path_candidates(news_dir, ticker):
        if not path.exists():
            continue
        news = _read_news_csv(path, ticker)
        if not news.empty:
            return news
    return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)


def news_path_candidates(news_dir: str | Path, ticker: str) -> list[Path]:
    news_path = Path(news_dir)
    candidates = [
        news_path / f"news_{ticker}.csv",
        news_path / f"{ticker}_news.csv",
    ]
    if ticker == "7203.T":
        candidates.insert(0, news_path / "toyota_news.csv")
    return _dedupe_paths(candidates)


def default_news_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / NEWS_DATA_SUBDIR


def _build_daily_news_features(stock_dates: pd.Series, news: pd.DataFrame) -> pd.DataFrame:
    dates = pd.to_datetime(stock_dates, errors="coerce").dt.normalize()
    start = dates.min()
    end = dates.max()
    if pd.isna(start) or pd.isna(end):
        return pd.DataFrame(columns=["date", *NEWS_FEATURE_COLUMNS])

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    daily = (
        news.dropna(subset=["date"])
        .groupby("date")
        .agg(
            news_count=("title", "size"),
            sentiment_sum=("sentiment_score", "sum"),
            weighted_sentiment_sum=("weighted_sentiment", "sum"),
            importance_sum=("importance_score", "sum"),
            sector_weighted_sentiment_sum=("sector_weighted_sentiment", "sum"),
            sector_importance_sum=("sector_importance", "sum"),
            avg_sentiment=("sentiment_score", "mean"),
            max_importance=("importance_score", "max"),
            min_sentiment=("sentiment_score", "min"),
            max_sentiment=("sentiment_score", "max"),
        )
        .reset_index()
    )
    calendar = pd.DataFrame({"date": pd.date_range(start=start, end=end, freq="D")})
    calendar = calendar.merge(daily, on="date", how="left")
    calendar["news_count"] = calendar["news_count"].fillna(0.0)
    calendar["sentiment_sum"] = calendar["sentiment_sum"].fillna(0.0)
    calendar["weighted_sentiment_sum"] = calendar["weighted_sentiment_sum"].fillna(0.0)
    calendar["importance_sum"] = calendar["importance_sum"].fillna(0.0)
    calendar["sector_weighted_sentiment_sum"] = calendar[
        "sector_weighted_sentiment_sum"
    ].fillna(0.0)
    calendar["sector_importance_sum"] = calendar["sector_importance_sum"].fillna(0.0)
    calendar["avg_sentiment"] = calendar["avg_sentiment"].fillna(0.0)
    calendar["max_importance"] = calendar["max_importance"].fillna(0.0)
    calendar["min_sentiment"] = calendar["min_sentiment"].fillna(0.0)
    calendar["max_sentiment"] = calendar["max_sentiment"].fillna(0.0)

    calendar["news_count_1d"] = calendar["news_count"]
    calendar["news_count_3d"] = calendar["news_count"].rolling(window=3, min_periods=1).sum()
    sentiment_sum_3d = calendar["sentiment_sum"].rolling(window=3, min_periods=1).sum()
    sentiment_sum_7d = calendar["sentiment_sum"].rolling(window=7, min_periods=1).sum()
    count_7d = calendar["news_count"].rolling(window=7, min_periods=1).sum()
    previous_sentiment_sum_3d = sentiment_sum_3d.shift(3)
    previous_count_3d = calendar["news_count_3d"].shift(3)
    previous_sentiment_sum_7d = sentiment_sum_7d.shift(7)
    previous_count_7d = count_7d.shift(7)
    calendar["avg_sentiment_1d"] = calendar["avg_sentiment"]
    calendar["avg_sentiment_3d"] = sentiment_sum_3d / calendar["news_count_3d"].replace(0, pd.NA)
    calendar["avg_sentiment_3d"] = calendar["avg_sentiment_3d"].fillna(0.0)
    avg_sentiment_7d = sentiment_sum_7d / count_7d.replace(0, pd.NA)
    avg_sentiment_7d = avg_sentiment_7d.fillna(0.0)
    previous_avg_3d = previous_sentiment_sum_3d / previous_count_3d.replace(0, pd.NA)
    previous_avg_7d = previous_sentiment_sum_7d / previous_count_7d.replace(0, pd.NA)
    calendar["sentiment_trend_3d"] = (calendar["avg_sentiment_3d"] - previous_avg_3d).fillna(0.0)
    calendar["sentiment_trend_7d"] = (avg_sentiment_7d - previous_avg_7d).fillna(0.0)
    calendar["max_importance_3d"] = calendar["max_importance"].rolling(
        window=3,
        min_periods=1,
    ).max()
    weighted_sentiment_sum_3d = calendar["weighted_sentiment_sum"].rolling(
        window=3,
        min_periods=1,
    ).sum()
    importance_sum_3d = calendar["importance_sum"].rolling(window=3, min_periods=1).sum()
    calendar["importance_weighted_sentiment"] = (
        weighted_sentiment_sum_3d / importance_sum_3d.replace(0, pd.NA)
    ).fillna(0.0)
    sector_weighted_sentiment_sum_3d = calendar["sector_weighted_sentiment_sum"].rolling(
        window=3,
        min_periods=1,
    ).sum()
    sector_importance_sum_3d = calendar["sector_importance_sum"].rolling(
        window=3,
        min_periods=1,
    ).sum()
    calendar["sector_news_strength"] = (
        sector_weighted_sentiment_sum_3d / sector_importance_sum_3d.replace(0, pd.NA)
    ).fillna(0.0)
    min_sentiment_3d = calendar["min_sentiment"].rolling(window=3, min_periods=1).min()
    max_sentiment_3d = calendar["max_sentiment"].rolling(window=3, min_periods=1).max()
    calendar["negative_news_flag"] = (
        (min_sentiment_3d <= -0.3) | (calendar["avg_sentiment_3d"] <= -0.2)
    ).astype(int)
    calendar["positive_news_flag"] = (
        (max_sentiment_3d >= 0.3) | (calendar["avg_sentiment_3d"] >= 0.2)
    ).astype(int)
    calendar["negative_news_shock"] = (
        (
            (min_sentiment_3d <= -0.6)
            & (calendar["max_importance_3d"] >= 0.7)
        )
        | (
            (calendar["avg_sentiment_3d"] <= -0.35)
            & (calendar["sentiment_trend_3d"] <= -0.25)
        )
    ).astype(int)
    calendar["positive_news_shock"] = (
        (
            (max_sentiment_3d >= 0.6)
            & (calendar["max_importance_3d"] >= 0.7)
        )
        | (
            (calendar["avg_sentiment_3d"] >= 0.35)
            & (calendar["sentiment_trend_3d"] >= 0.25)
        )
    ).astype(int)

    features = calendar[["date", *NEWS_FEATURE_COLUMNS]].copy()
    stock_date_frame = pd.DataFrame({"date": dates})
    return stock_date_frame.merge(features, on="date", how="left")


def _read_news_csv(path: Path, ticker: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)
    if raw.empty:
        return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)

    result = raw.copy()
    if "date" not in result.columns:
        raise ValueError(f"Missing date column in news CSV: {path}")
    for column in NEWS_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["sentiment_score"] = pd.to_numeric(result["sentiment_score"], errors="coerce").fillna(0.0)
    result["importance_score"] = pd.to_numeric(result["importance_score"], errors="coerce").fillna(0.0)
    result = result.dropna(subset=["date"])
    result = _filter_news_for_ticker(result, ticker)
    result["weighted_sentiment"] = result["sentiment_score"] * result["importance_score"]
    sector_mask = _sector_news_mask(result)
    result["sector_importance"] = result["importance_score"].where(sector_mask, 0.0)
    result["sector_weighted_sentiment"] = result["weighted_sentiment"].where(sector_mask, 0.0)
    output_columns = [*NEWS_DISPLAY_COLUMNS, *NEWS_INTERNAL_COLUMNS]
    return result[output_columns].sort_values("date").reset_index(drop=True)


def _filter_news_for_ticker(news: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if "ticker" not in news.columns or news["ticker"].fillna("").astype(str).str.strip().eq("").all():
        return news
    ticker_values = news["ticker"].fillna("").astype(str).str.strip().str.lower()
    allowed = {"", "all", "auto", "sector", ticker.lower()}
    if ticker == "7203.T":
        allowed.update({"toyota", "tm", "7203"})
    return news[ticker_values.isin(allowed)].copy()


def _recent_news_items(news: pd.DataFrame, latest_date: pd.Timestamp, days: int) -> list[dict[str, object]]:
    if news.empty or pd.isna(latest_date):
        return []
    start = latest_date.normalize() - pd.Timedelta(days=days - 1)
    end = latest_date.normalize()
    recent = news[
        (pd.to_datetime(news["date"], errors="coerce").dt.normalize() >= start)
        & (pd.to_datetime(news["date"], errors="coerce").dt.normalize() <= end)
    ].copy()
    recent = recent.sort_values(["date", "importance_score"], ascending=[False, False]).head(10)
    recent["date"] = pd.to_datetime(recent["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return recent[list(NEWS_DISPLAY_COLUMNS)].to_dict(orient="records")


def _important_news_items(
    news: pd.DataFrame,
    latest_date: pd.Timestamp,
    days: int,
) -> list[dict[str, object]]:
    if news.empty or pd.isna(latest_date):
        return []
    start = latest_date.normalize() - pd.Timedelta(days=days - 1)
    end = latest_date.normalize()
    recent = news[
        (pd.to_datetime(news["date"], errors="coerce").dt.normalize() >= start)
        & (pd.to_datetime(news["date"], errors="coerce").dt.normalize() <= end)
    ].copy()
    recent = recent.sort_values(["importance_score", "date"], ascending=[False, False]).head(5)
    recent["date"] = pd.to_datetime(recent["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return recent[list(NEWS_DISPLAY_COLUMNS)].to_dict(orient="records")


def _sentiment_history(data: pd.DataFrame, days: int) -> list[dict[str, object]]:
    if data.empty or "date" not in data.columns:
        return []
    history_columns = [
        "date",
        "avg_sentiment_1d",
        "avg_sentiment_3d",
        "sentiment_trend_3d",
        "importance_weighted_sentiment",
        "sector_news_strength",
        "news_count_1d",
    ]
    available = [column for column in history_columns if column in data.columns]
    if "date" not in available:
        return []
    history = data[available].copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history = history.dropna(subset=["date"]).sort_values("date").tail(days)
    history["date"] = history["date"].dt.strftime("%Y-%m-%d")
    for column in available:
        if column != "date":
            history[column] = pd.to_numeric(history[column], errors="coerce").fillna(0.0)
    return history.to_dict(orient="records")


def _add_empty_news_features(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in NEWS_FEATURE_COLUMNS:
        result[column] = 0.0
    return result


def _summary_text(
    count_3d: int,
    metrics: dict[str, float | None],
    negative: bool,
    positive: bool,
) -> str:
    if count_3d == 0:
        return "No recent manual news items are registered."
    avg_sentiment = metrics.get("avg_sentiment_3d") or 0.0
    max_importance = metrics.get("max_importance_3d") or 0.0
    if negative and max_importance >= 0.7:
        return "Recent 3-day news flow includes high-importance negative items."
    if positive and max_importance >= 0.7:
        return "Recent 3-day news flow includes high-importance positive items."
    if negative:
        return "Recent 3-day news flow is leaning negative."
    if positive:
        return "Recent 3-day news flow is leaning positive."
    return f"Recent 3-day news flow has {count_3d} items with average sentiment {avg_sentiment:.2f}."


def _impact_summary_text(
    count_3d: int,
    metrics: dict[str, float | None],
    negative_shock: bool,
    positive_shock: bool,
) -> str:
    if count_3d == 0:
        return "News impact is neutral because no recent manual news is registered."

    weighted = metrics.get("importance_weighted_sentiment") or 0.0
    trend_3d = metrics.get("sentiment_trend_3d") or 0.0
    sector_strength = metrics.get("sector_news_strength") or 0.0
    if negative_shock:
        return (
            "News impact is negative: important negative news is present and "
            f"3-day sentiment trend is {trend_3d:.2f}."
        )
    if positive_shock:
        return (
            "News impact is positive: important positive news is present and "
            f"3-day sentiment trend is {trend_3d:.2f}."
        )
    if weighted <= -0.2:
        return (
            "News impact leans negative based on importance-weighted sentiment "
            f"({weighted:.2f})."
        )
    if weighted >= 0.2:
        return (
            "News impact leans positive based on importance-weighted sentiment "
            f"({weighted:.2f})."
        )
    if abs(sector_strength) >= 0.2:
        return f"Sector news context is notable with strength {sector_strength:.2f}."
    return "News impact is currently mild and does not show a strong directional shock."


def _sector_news_mask(news: pd.DataFrame) -> pd.Series:
    ticker_text = news.get("ticker", pd.Series(index=news.index, dtype="object"))
    category_text = news.get("category", pd.Series(index=news.index, dtype="object"))
    title_text = news.get("title", pd.Series(index=news.index, dtype="object"))
    combined = (
        ticker_text.fillna("").astype(str)
        + " "
        + category_text.fillna("").astype(str)
        + " "
        + title_text.fillna("").astype(str)
    ).str.lower()
    keywords = ("auto", "sector", "industry", "honda", "nissan", "tesla", "byd", "ev")
    return combined.apply(lambda text: any(keyword in text for keyword in keywords))


def _empty_context(summary: str) -> NewsContext:
    return NewsContext(
        news_count_1d=0,
        news_count_3d=0,
        avg_sentiment_1d=None,
        avg_sentiment_3d=None,
        max_importance_3d=None,
        negative_news_flag=False,
        positive_news_flag=False,
        sentiment_trend_3d=None,
        sentiment_trend_7d=None,
        importance_weighted_sentiment=None,
        negative_news_shock=False,
        positive_news_shock=False,
        sector_news_strength=None,
        summary=summary,
        impact_summary=summary,
        recent_items=[],
        important_items=[],
        sentiment_history=[],
        metrics={},
    )


def _value(row: pd.Series, key: str) -> float | None:
    return safe_float(row.get(key))


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
