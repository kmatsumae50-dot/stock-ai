from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ai.news.manual_csv import NEWS_DISPLAY_COLUMNS, news_path_candidates


MOJIBAKE_MARKERS = (
    "\ufffd",
    "窶",
    "窶冱",
    "窶冩",
    "窶冦",
    "Ã",
    "Â",
    "譁",
    "繧",
    "縺",
    "蜿",
    "窶",
    "冱",
)


def check_news_csv(news_path: str | Path) -> dict[str, object]:
    path = Path(news_path)
    if not path.exists():
        return _empty_report(path, status="WARN", message="News CSV does not exist yet.")
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return _empty_report(path, status="WARN", message="News CSV is empty.")

    for column in NEWS_DISPLAY_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[list(NEWS_DISPLAY_COLUMNS)].copy()

    dates = pd.to_datetime(frame["date"], errors="coerce", format="mixed")
    fetched_at = pd.to_datetime(frame["fetched_at"], errors="coerce", format="mixed")
    title_text = frame["title"].fillna("").astype(str)
    source_text = frame["source"].fillna("").astype(str)
    url_text = frame["url"].fillna("").astype(str)
    source_credibility_text = frame["source_credibility_score"].fillna("").astype(str)
    source_market_weight_text = frame["source_market_impact_weight"].fillna("").astype(str)
    source_type_text = frame["source_type"].fillna("").astype(str)
    category_text = frame["category"].fillna("").astype(str)
    sentiment_reason_text = frame["sentiment_reason"].fillna("").astype(str)
    importance_reason_text = frame["importance_reason"].fillna("").astype(str)
    sentiment = pd.to_numeric(frame["sentiment_score"], errors="coerce")
    importance = pd.to_numeric(frame["importance_score"], errors="coerce")
    source_credibility = pd.to_numeric(source_credibility_text, errors="coerce")
    source_market_weight = pd.to_numeric(source_market_weight_text, errors="coerce")

    empty_title_mask = title_text.str.strip().eq("")
    missing_category_mask = category_text.str.strip().eq("")
    missing_sentiment_reason_mask = sentiment_reason_text.str.strip().eq("")
    missing_importance_reason_mask = importance_reason_text.str.strip().eq("")
    empty_source_mask = source_text.str.strip().eq("")
    empty_url_mask = url_text.str.strip().eq("")
    missing_source_credibility_mask = source_credibility_text.str.strip().eq("")
    missing_source_market_weight_mask = source_market_weight_text.str.strip().eq("")
    source_credibility_bad_mask = (
        ~missing_source_credibility_mask
        & (source_credibility.isna() | (source_credibility < 0.0) | (source_credibility > 1.0))
    )
    source_market_weight_bad_mask = (
        ~missing_source_market_weight_mask
        & (source_market_weight.isna() | (source_market_weight < 0.0) | (source_market_weight > 1.0))
    )
    unknown_source_mask = source_type_text.str.strip().str.lower().isin({"", "unknown"})
    unknown_sources = sorted(
        {
            source.strip()
            for source in source_text[unknown_source_mask].astype(str)
            if source.strip()
        }
    )
    invalid_date_mask = dates.isna()
    missing_fetched_mask = frame["fetched_at"].fillna("").astype(str).str.strip().eq("") | fetched_at.isna()
    duplicate_title_mask = title_text.str.strip().str.lower().duplicated(keep=False) & ~empty_title_mask
    duplicate_url_mask = url_text.str.strip().str.lower().duplicated(keep=False) & ~empty_url_mask
    sentiment_bad_mask = sentiment.isna() | (sentiment < -1.0) | (sentiment > 1.0)
    importance_bad_mask = importance.isna() | (importance < 0.0) | (importance > 1.0)
    mojibake_mask = title_text.combine(frame["summary"].fillna("").astype(str), lambda a, b: f"{a} {b}").map(
        is_mojibake_suspect
    )

    issue_mask = (
        empty_title_mask
        | invalid_date_mask
        | empty_source_mask
        | empty_url_mask
        | duplicate_title_mask
        | duplicate_url_mask
        | sentiment_bad_mask
        | importance_bad_mask
        | source_credibility_bad_mask
        | source_market_weight_bad_mask
        | missing_fetched_mask
        | mojibake_mask
    )
    issue_rows = _issue_rows(
        frame,
        issue_mask=issue_mask,
        masks={
            "empty_title": empty_title_mask,
            "invalid_date": invalid_date_mask,
            "empty_source": empty_source_mask,
            "empty_url": empty_url_mask,
            "duplicate_title": duplicate_title_mask,
            "duplicate_url": duplicate_url_mask,
            "sentiment_range": sentiment_bad_mask,
            "importance_range": importance_bad_mask,
            "source_credibility_range": source_credibility_bad_mask,
            "source_market_weight_range": source_market_weight_bad_mask,
            "missing_fetched_at": missing_fetched_mask,
            "mojibake_suspect": mojibake_mask,
        },
    )
    fail_count = int(
        empty_title_mask.sum()
        + invalid_date_mask.sum()
        + empty_source_mask.sum()
        + empty_url_mask.sum()
        + sentiment_bad_mask.sum()
        + importance_bad_mask.sum()
        + source_credibility_bad_mask.sum()
        + source_market_weight_bad_mask.sum()
    )
    warn_count = int(
        duplicate_title_mask.sum()
        + duplicate_url_mask.sum()
        + missing_fetched_mask.sum()
        + mojibake_mask.sum()
    )
    status = "OK"
    if fail_count > 0:
        status = "FAIL"
    elif warn_count > 0:
        status = "WARN"

    return {
        "path": str(path),
        "status": status,
        "message": _status_message(status, fail_count=fail_count, warn_count=warn_count),
        "rows": int(len(frame)),
        "latest_news_date": _format_timestamp(dates.dropna().max()),
        "latest_fetched_at": _format_timestamp(fetched_at.dropna().max()),
        "empty_title_count": int(empty_title_mask.sum()),
        "invalid_date_count": int(invalid_date_mask.sum()),
        "empty_source_count": int(empty_source_mask.sum()),
        "empty_url_count": int(empty_url_mask.sum()),
        "duplicate_title_count": int(duplicate_title_mask.sum()),
        "duplicate_url_count": int(duplicate_url_mask.sum()),
        "duplicate_count": int((duplicate_title_mask | duplicate_url_mask).sum()),
        "sentiment_range_status": "OK" if int(sentiment_bad_mask.sum()) == 0 else "FAIL",
        "importance_range_status": "OK" if int(importance_bad_mask.sum()) == 0 else "FAIL",
        "source_credibility_range_status": "OK" if int(source_credibility_bad_mask.sum()) == 0 else "FAIL",
        "source_weight_range_status": "OK" if int(source_market_weight_bad_mask.sum()) == 0 else "FAIL",
        "sentiment_out_of_range_count": int(sentiment_bad_mask.sum()),
        "importance_out_of_range_count": int(importance_bad_mask.sum()),
        "source_credibility_out_of_range_count": int(source_credibility_bad_mask.sum()),
        "source_weight_out_of_range_count": int(source_market_weight_bad_mask.sum()),
        "missing_fetched_at_count": int(missing_fetched_mask.sum()),
        "missing_category_count": int(missing_category_mask.sum()),
        "missing_sentiment_reason_count": int(missing_sentiment_reason_mask.sum()),
        "missing_importance_reason_count": int(missing_importance_reason_mask.sum()),
        "missing_source_credibility_count": int(missing_source_credibility_mask.sum()),
        "missing_source_market_weight_count": int(missing_source_market_weight_mask.sum()),
        "unknown_source_count": int(unknown_source_mask.sum()),
        "unknown_sources": unknown_sources,
        "source_type_distribution": _source_type_distribution(source_type_text),
        "mojibake_suspect_count": int(mojibake_mask.sum()),
        "issue_rows": issue_rows,
    }


def check_news_for_ticker(news_dir: str | Path, ticker: str) -> dict[str, object]:
    return check_news_csv(news_path_candidates(news_dir, ticker)[0])


def is_mojibake_suspect(text: str) -> bool:
    value = str(text or "")
    if not value.strip():
        return False
    if any(marker in value for marker in MOJIBAKE_MARKERS):
        return True
    symbol_count = sum(1 for char in value if not char.isalnum() and not char.isspace())
    ratio = symbol_count / max(1, len(value))
    return len(value) >= 30 and ratio >= 0.35


def _issue_rows(
    frame: pd.DataFrame,
    *,
    issue_mask: pd.Series,
    masks: dict[str, pd.Series],
    limit: int = 20,
) -> list[dict[str, object]]:
    rows = frame[issue_mask].head(limit).copy()
    result: list[dict[str, object]] = []
    for index, row in rows.iterrows():
        issues = [name for name, mask in masks.items() if bool(mask.loc[index])]
        result.append(
            {
                "date": row.get("date", ""),
                "title": row.get("title", ""),
                "source": row.get("source", ""),
                "url": row.get("url", ""),
                "issues": ", ".join(issues),
            }
        )
    return result


def _empty_report(path: Path, *, status: str, message: str) -> dict[str, object]:
    return {
        "path": str(path),
        "status": status,
        "message": message,
        "rows": 0,
        "latest_news_date": None,
        "latest_fetched_at": None,
        "empty_title_count": 0,
        "invalid_date_count": 0,
        "empty_source_count": 0,
        "empty_url_count": 0,
        "duplicate_title_count": 0,
        "duplicate_url_count": 0,
        "duplicate_count": 0,
        "sentiment_range_status": "OK",
        "importance_range_status": "OK",
        "source_credibility_range_status": "OK",
        "source_weight_range_status": "OK",
        "sentiment_out_of_range_count": 0,
        "importance_out_of_range_count": 0,
        "source_credibility_out_of_range_count": 0,
        "source_weight_out_of_range_count": 0,
        "missing_fetched_at_count": 0,
        "missing_category_count": 0,
        "missing_sentiment_reason_count": 0,
        "missing_importance_reason_count": 0,
        "missing_source_credibility_count": 0,
        "missing_source_market_weight_count": 0,
        "unknown_source_count": 0,
        "unknown_sources": [],
        "source_type_distribution": {},
        "mojibake_suspect_count": 0,
        "issue_rows": [],
    }


def _status_message(status: str, *, fail_count: int, warn_count: int) -> str:
    if status == "OK":
        return "News CSV quality checks passed."
    if status == "FAIL":
        return f"News CSV has {fail_count} blocking quality issues and {warn_count} warnings."
    return f"News CSV has {warn_count} warnings."


def _format_timestamp(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _source_type_distribution(values: pd.Series) -> dict[str, int]:
    clean = values.fillna("").astype(str).str.strip().str.lower()
    clean = clean.where(clean.ne(""), "unknown")
    return {str(key): int(value) for key, value in clean.value_counts().sort_index().items()}
