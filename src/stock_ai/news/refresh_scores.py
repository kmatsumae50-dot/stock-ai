from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.news.fetcher import (
    infer_news_category,
    score_news_importance_with_reason,
    score_news_sentiment_with_reason,
)
from stock_ai.news.manual_csv import NEWS_DISPLAY_COLUMNS, news_path_candidates
from stock_ai.news.sources import resolve_news_source_profile


@dataclass(frozen=True)
class RefreshNewsScoresResult:
    path: Path
    dry_run: bool
    row_count: int
    updated_count: int
    category_distribution: dict[str, int]
    sentiment_distribution: dict[str, int]
    importance_average: float | None
    change_examples: list[dict[str, object]]


def refresh_news_scores_for_ticker(
    news_dir: str | Path,
    ticker: str,
    *,
    dry_run: bool = False,
    example_limit: int = 5,
) -> RefreshNewsScoresResult:
    path = news_path_candidates(news_dir, ticker)[0]
    return refresh_news_scores_csv(
        path,
        ticker=ticker,
        dry_run=dry_run,
        example_limit=example_limit,
    )


def refresh_news_scores_csv(
    news_path: str | Path,
    *,
    ticker: str,
    dry_run: bool = False,
    example_limit: int = 5,
) -> RefreshNewsScoresResult:
    path = Path(news_path)
    if not path.exists():
        raise FileNotFoundError(f"News CSV does not exist: {path}")

    try:
        original = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        original = pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)

    frame = _ensure_news_columns(original)
    refreshed = frame.copy().astype(object)
    change_examples: list[dict[str, object]] = []
    updated_count = 0

    for index, row in frame.iterrows():
        recalculated = _recalculate_row(row, ticker=ticker)
        changed_fields = {
            column: {
                "old": str(row.get(column, "")),
                "new": str(value),
            }
            for column, value in recalculated.items()
            if str(row.get(column, "")) != str(value)
        }
        if changed_fields:
            updated_count += 1
            if len(change_examples) < example_limit:
                change_examples.append(
                    {
                        "row": int(index),
                        "date": row.get("date", ""),
                        "title": row.get("title", ""),
                        "changes": changed_fields,
                    }
                )
        for column, value in recalculated.items():
            refreshed.at[index, column] = value

    refreshed = refreshed[list(NEWS_DISPLAY_COLUMNS)]
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        refreshed.to_csv(path, index=False)

    return RefreshNewsScoresResult(
        path=path,
        dry_run=dry_run,
        row_count=int(len(refreshed)),
        updated_count=updated_count,
        category_distribution=_value_counts(refreshed.get("category")),
        sentiment_distribution=_sentiment_distribution(refreshed.get("sentiment_score")),
        importance_average=_importance_average(refreshed.get("importance_score")),
        change_examples=change_examples,
    )


def _ensure_news_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in NEWS_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    return result[list(NEWS_DISPLAY_COLUMNS)].fillna("")


def _recalculate_row(row: pd.Series, *, ticker: str) -> dict[str, object]:
    row_ticker = str(row.get("ticker", "") or "").strip()
    scoring_ticker = row_ticker if row_ticker not in {"", "all", "auto", "sector"} else ticker
    title = str(row.get("title", "") or "")
    summary = str(row.get("summary", "") or "")
    source = str(row.get("source", "") or "")
    text = f"{title} {summary}"
    source_profile = resolve_news_source_profile(source)

    sentiment = score_news_sentiment_with_reason(text)
    importance = score_news_importance_with_reason(
        text,
        ticker=scoring_ticker,
        title=title,
        source=source,
    )
    category = infer_news_category(
        title,
        summary=summary,
        fallback=str(row.get("category", "") or "other"),
    )
    return {
        "sentiment_score": sentiment.score,
        "importance_score": importance.score,
        "category": category,
        "sentiment_reason": sentiment.reason,
        "importance_reason": importance.reason,
        "source_credibility_score": source_profile.credibility_score,
        "source_market_impact_weight": source_profile.market_impact_weight,
        "source_type": source_profile.source_type,
    }


def _value_counts(values: pd.Series | None) -> dict[str, int]:
    if values is None:
        return {}
    clean = values.fillna("").astype(str).str.strip()
    clean = clean.where(clean.ne(""), "missing")
    return {str(key): int(value) for key, value in clean.value_counts().sort_index().items()}


def _sentiment_distribution(values: pd.Series | None) -> dict[str, int]:
    if values is None:
        return {"negative": 0, "neutral": 0, "positive": 0}
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return {
        "negative": int((numeric <= -0.1).sum()),
        "neutral": int(((numeric > -0.1) & (numeric < 0.1)).sum()),
        "positive": int((numeric >= 0.1).sum()),
    }


def _importance_average(values: pd.Series | None) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())
