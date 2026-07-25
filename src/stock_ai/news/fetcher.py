from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import html
from pathlib import Path
import re
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

import pandas as pd

from stock_ai.news.manual_csv import NEWS_DISPLAY_COLUMNS, news_path_candidates
from stock_ai.news.sources import resolve_news_source_profile
from stock_ai.utils.safe_cast import safe_float


DEFAULT_NEWS_QUERY = "Toyota stock"
DEFAULT_FETCH_LIMIT = 20
DEFAULT_NEWS_LANGUAGE = "en-US"
DEFAULT_NEWS_REGION = "US"

STRONG_NEGATIVE_KEYWORDS = (
    "recall",
    "lawsuit",
    "scandal",
    "fraud",
    "strike",
    "production halt",
    "profit warning",
    "downgrade",
    "loss",
    "tariff",
    "investigation",
    "safety issue",
    "リコール",
    "不正",
    "不具合",
    "減益",
    "下方修正",
    "販売減",
    "生産停止",
    "訴訟",
    "調査",
    "関税",
)
WEAK_NEGATIVE_KEYWORDS = (
    "decline",
    "slowdown",
    "concern",
    "pressure",
    "uncertainty",
    "cut",
    "delay",
    "bankruptcy",
    "defect",
    "fine",
    "halt",
    "miss",
    "probe",
    "slump",
    "weak",
    "懸念",
    "減少",
    "遅れ",
    "低迷",
    "圧迫",
    "不透明",
)
STRONG_POSITIVE_KEYWORDS = (
    "record profit",
    "upgrade",
    "buyback",
    "dividend hike",
    "strong demand",
    "beat expectations",
    "partnership",
    "expansion",
    "最高益",
    "増益",
    "上方修正",
    "自社株買い",
    "増配",
    "販売好調",
    "提携",
    "拡大",
)
WEAK_POSITIVE_KEYWORDS = (
    "growth",
    "recovery",
    "increase",
    "improve",
    "launch",
    "investment",
    "beat",
    "raise",
    "raised",
    "upgraded",
    "成長",
    "回復",
    "増加",
    "改善",
    "投資",
    "発表",
)
NEGATIVE_KEYWORDS = {
    *STRONG_NEGATIVE_KEYWORDS,
    *WEAK_NEGATIVE_KEYWORDS,
    "bankruptcy",
    "cut",
    "defect",
    "delay",
    "downgrade",
    "fine",
    "fraud",
    "halt",
    "investigation",
    "lawsuit",
    "loss",
    "miss",
    "probe",
    "recall",
    "scandal",
    "strike",
    "slump",
    "tariff",
    "weak",
}
POSITIVE_KEYWORDS = {
    *STRONG_POSITIVE_KEYWORDS,
    *WEAK_POSITIVE_KEYWORDS,
    "beat",
    "buyback",
    "growth",
    "record profit",
    "raise",
    "raised",
    "strong demand",
    "upgrade",
    "upgraded",
}
IMPORTANT_KEYWORDS = {
    "auto",
    "byd",
    "china",
    "earnings",
    "ev",
    "guidance",
    "honda",
    "hybrid",
    "nissan",
    "tariff",
    "tesla",
    "toyota",
    "usdjpy",
    "yen",
    "ev",
    "サプライチェーン",
    "トヨタ",
    "ハイブリッド",
    "リコール",
    "中国",
    "円安",
    "円高",
    "半導体",
    "業績",
    "決算",
    "生産",
    "見通し",
    "販売",
    "金利",
    "関税",
    "電動車",
}
IMPORTANT_TERMS = (
    "earnings",
    "guidance",
    "profit",
    "sales",
    "production",
    "recall",
    "lawsuit",
    "tariff",
    "china",
    "ev",
    "hybrid",
    "usdjpy",
    "interest rates",
    "supply chain",
    "auto",
    "byd",
    "honda",
    "nissan",
    "tesla",
    "toyota",
    "yen",
    "サプライチェーン",
    "トヨタ",
    "ハイブリッド",
    "リコール",
    "中国",
    "円安",
    "円高",
    "半導体",
    "業績",
    "決算",
    "生産",
    "見通し",
    "販売",
    "金利",
    "関税",
    "電動車",
)
MAJOR_NEWS_SOURCES = (
    "reuters",
    "bloomberg",
    "nikkei",
    "financial times",
    "wall street journal",
    "cnbc",
    "associated press",
    "ap news",
    "marketwatch",
    "yahoo finance",
    "automotive news",
)
CATEGORY_KEYWORDS = (
    ("recall_safety", ("recall", "safety", "defect", "crash", "investigation", "probe", "リコール", "不具合", "安全")),
    ("legal_regulatory", ("lawsuit", "fraud", "fine", "regulation", "regulatory", "antitrust")),
    ("earnings", ("earnings", "guidance", "profit", "results", "dividend", "buyback", "決算", "業績", "利益", "見通し", "上方修正", "下方修正")),
    ("macro", ("usdjpy", "yen", "tariff", "interest rates", "rates", "inflation", "currency", "円安", "円高", "関税", "金利")),
    ("china_market", ("china", "chinese", "byd", "中国市場", "中国販売", "中国")),
    ("ev_hybrid", ("ev", "electric vehicle", "hybrid", "battery", "ハイブリッド", "電動車")),
    ("production_supply", ("production", "supply chain", "factory", "plant", "chip", "semiconductor", "halt", "strike", "生産", "工場", "半導体", "サプライチェーン")),
    ("product", ("launch", "model", "vehicle", "suv", "sedan", "truck")),
    ("auto_sector", ("honda", "nissan", "tesla", "byd", "auto industry", "automaker", "vehicle sales")),
)
MOJIBAKE_REPLACEMENTS = {
    "窶冱": "'s",
    "窶冩": "'",
    "窶冦": '"',
    "窶": "'",
    "縺": "",
    "繧": "",
    "譁": "",
}


@dataclass(frozen=True)
class FetchNewsResult:
    output_path: Path
    fetched_count: int
    added_count: int
    duplicate_count: int
    total_count: int
    latest_fetched_at: str | None
    query: str


@dataclass(frozen=True)
class NewsScoreResult:
    score: float
    reason: str


def fetch_google_news_rss(
    *,
    query: str,
    ticker: str = "7203.T",
    news_dir: str | Path,
    limit: int = DEFAULT_FETCH_LIMIT,
    category: str = "auto",
    timeout: int = 20,
    save_ticker: str | None = None,
    item_ticker: str | None = None,
    query_profile: str | None = None,
    language: str = DEFAULT_NEWS_LANGUAGE,
    region: str = DEFAULT_NEWS_REGION,
) -> FetchNewsResult:
    storage_ticker = str(save_ticker or ticker or "7203.T").strip() or "7203.T"
    row_ticker = ticker if item_ticker is None else item_ticker
    scoring_ticker = str(ticker or storage_ticker)
    news_path = _target_news_path(news_dir, storage_ticker)
    existing = _read_existing_news(news_path)
    fetched = _fetch_rss_items(
        query=query,
        ticker=scoring_ticker,
        item_ticker=str(row_ticker or "").strip(),
        limit=limit,
        category=category,
        timeout=timeout,
        query_profile=str(query_profile or "").strip(),
        language=language,
        region=region,
    )
    combined, added_count, duplicate_count = merge_news_items(existing, fetched)
    news_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(news_path, index=False)
    latest = latest_news_status(news_path)
    return FetchNewsResult(
        output_path=news_path,
        fetched_count=int(len(fetched)),
        added_count=added_count,
        duplicate_count=duplicate_count,
        total_count=int(len(combined)),
        latest_fetched_at=latest.get("latest_fetched_at"),
        query=query,
    )


def merge_news_items(existing: pd.DataFrame, fetched: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    existing = _normalize_news_frame(existing)
    fetched = _normalize_news_frame(fetched)
    if fetched.empty:
        return existing, 0, 0

    existing_urls, existing_titles = _dedupe_token_sets(existing)
    rows: list[dict[str, object]] = []
    duplicate_count = 0
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for row in fetched.to_dict(orient="records"):
        url = str(row.get("url") or "").strip().lower()
        title = _normalize_title(str(row.get("title") or ""))
        if (
            not title
            or (url and (url in existing_urls or url in seen_urls))
            or title in existing_titles
            or title in seen_titles
        ):
            duplicate_count += 1
            continue
        if url:
            seen_urls.add(url)
        seen_titles.add(title)
        rows.append(row)

    if rows:
        combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        combined = existing
    combined = _normalize_news_frame(combined)
    combined = combined.sort_values(["date", "importance_score"], ascending=[False, False])
    return combined.reset_index(drop=True), len(rows), duplicate_count


def latest_news_status(news_path: str | Path) -> dict[str, object]:
    path = Path(news_path)
    if not path.exists():
        return {
            "path": str(path),
            "count": 0,
            "latest_date": None,
            "latest_fetched_at": None,
        }
    frame = _read_existing_news(path)
    if frame.empty:
        return {
            "path": str(path),
            "count": 0,
            "latest_date": None,
            "latest_fetched_at": None,
        }
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
    fetched = pd.to_datetime(frame.get("fetched_at", pd.Series(dtype="object")), errors="coerce").dropna()
    return {
        "path": str(path),
        "count": int(len(frame)),
        "latest_date": _format_timestamp(dates.max()) if not dates.empty else None,
        "latest_fetched_at": _format_timestamp(fetched.max()) if not fetched.empty else None,
    }


def ticker_default_query(ticker: str | None) -> str:
    if str(ticker or "").upper() == "7203.T":
        return DEFAULT_NEWS_QUERY
    return str(ticker or DEFAULT_NEWS_QUERY)


def _fetch_rss_items(
    *,
    query: str,
    ticker: str,
    item_ticker: str,
    limit: int,
    category: str,
    timeout: int,
    query_profile: str,
    language: str,
    region: str,
) -> pd.DataFrame:
    url = _google_news_rss_url(query, language=language, region=region)
    request = Request(
        url,
        headers={"User-Agent": "stock-ai-news-fetcher/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        body = response.read()
    root = ET.fromstring(body)
    fetched_at = datetime.now(UTC).replace(microsecond=0).isoformat()
    rows: list[dict[str, object]] = []
    for item in root.findall("./channel/item")[: max(1, int(limit))]:
        title = _clean_text(_text(item.find("title")))
        summary = _clean_summary(_text(item.find("description")))
        source = _clean_text(_source_text(item, title))
        published_at = _parse_rss_date(_text(item.find("pubDate")))
        link = _text(item.find("link"))
        scoring_text = f"{title} {summary}"
        source_profile = resolve_news_source_profile(source)
        sentiment = score_news_sentiment_with_reason(scoring_text)
        importance = score_news_importance_with_reason(
            scoring_text,
            ticker=ticker,
            title=title,
            source=source,
        )
        inferred_category = infer_news_category(title, summary=summary, fallback=category)
        rows.append(
            {
                "date": published_at,
                "ticker": item_ticker,
                "title": title,
                "source": source,
                "url": link,
                "source_credibility_score": source_profile.credibility_score,
                "source_market_impact_weight": source_profile.market_impact_weight,
                "source_type": source_profile.source_type,
                "category": inferred_category,
                "sentiment_score": sentiment.score,
                "importance_score": importance.score,
                "sentiment_reason": sentiment.reason,
                "importance_reason": importance.reason,
                "summary": summary,
                "fetched_at": fetched_at,
                "query_profile": query_profile,
            }
        )
    return _normalize_news_frame(pd.DataFrame(rows))


def score_news_sentiment(text: str) -> float:
    return score_news_sentiment_with_reason(text).score


def score_news_sentiment_with_reason(text: str) -> NewsScoreResult:
    normalized = _normalize_for_matching(text)
    strong_negative = _matched_keywords(normalized, STRONG_NEGATIVE_KEYWORDS)
    weak_negative = _matched_keywords(normalized, WEAK_NEGATIVE_KEYWORDS)
    strong_positive = _matched_keywords(normalized, STRONG_POSITIVE_KEYWORDS)
    weak_positive = _matched_keywords(normalized, WEAK_POSITIVE_KEYWORDS)

    score = (
        len(strong_positive) * 0.45
        + len(weak_positive) * 0.18
        - len(strong_negative) * 0.45
        - len(weak_negative) * 0.18
    )
    score = round(_clamp(score, -1.0, 1.0), 3)
    reasons = []
    if strong_negative:
        reasons.append(f"strong negative: {', '.join(strong_negative[:4])}")
    if weak_negative:
        reasons.append(f"weak negative: {', '.join(weak_negative[:4])}")
    if strong_positive:
        reasons.append(f"strong positive: {', '.join(strong_positive[:4])}")
    if weak_positive:
        reasons.append(f"weak positive: {', '.join(weak_positive[:4])}")
    reason = "; ".join(reasons) if reasons else "no directional sentiment keywords"
    return NewsScoreResult(score=score, reason=reason)


def score_news_importance(text: str, *, ticker: str = "") -> float:
    return score_news_importance_with_reason(text, ticker=ticker).score


def score_news_importance_with_reason(
    text: str,
    *,
    ticker: str = "",
    title: str = "",
    source: str = "",
) -> NewsScoreResult:
    normalized_text = _normalize_for_matching(text)
    normalized = _normalize_for_matching(f"{ticker} {text}")
    title_normalized = _normalize_for_matching(title)
    important_terms = _matched_keywords(normalized, IMPORTANT_TERMS)
    title_terms = _matched_keywords(title_normalized, IMPORTANT_TERMS)
    direct_toyota = bool(_matched_keywords(normalized_text, ("toyota", "トヨタ", "7203.t", "7203")))
    source_profile = resolve_news_source_profile(source)

    score = 0.2
    reasons = []
    if direct_toyota:
        score += 0.25
        reasons.append("direct Toyota mention")
    if important_terms:
        score += min(0.35, len(important_terms) * 0.08)
        reasons.append(f"important terms: {', '.join(important_terms[:5])}")
    if title_terms:
        score += min(0.15, len(title_terms) * 0.05)
        reasons.append(f"title terms: {', '.join(title_terms[:4])}")
    source_adjustment = source_profile.market_impact_weight * 0.10
    if source_adjustment > 0:
        score += source_adjustment
        reasons.append(
            "source impact: "
            f"{source_profile.name} ({source_profile.source_type}, "
            f"weight {source_profile.market_impact_weight:.2f})"
        )
    if (
        source_profile.source_type in {"official", "wire", "financial_media"}
        and source_profile.market_impact_weight >= 0.60
    ):
        reasons.append(f"major source: {source_profile.name}")

    score = round(_clamp(score, 0.0, 1.0), 3)
    reason = "; ".join(reasons) if reasons else "baseline importance only"
    return NewsScoreResult(score=score, reason=reason)


def infer_news_category(title: str, *, summary: str = "", fallback: str = "other") -> str:
    normalized = _normalize_for_matching(f"{title} {summary}")
    for category, keywords in CATEGORY_KEYWORDS:
        if _matched_keywords(normalized, keywords):
            return category
    fallback_value = str(fallback or "").strip().lower()
    if fallback_value and fallback_value != "auto":
        return fallback_value
    return "other"


def _google_news_rss_url(
    query: str,
    *,
    language: str = DEFAULT_NEWS_LANGUAGE,
    region: str = DEFAULT_NEWS_REGION,
) -> str:
    encoded = quote_plus(query)
    normalized_region = str(region or DEFAULT_NEWS_REGION).strip().upper() or DEFAULT_NEWS_REGION
    normalized_language = str(language or DEFAULT_NEWS_LANGUAGE).strip() or DEFAULT_NEWS_LANGUAGE
    ceid_language = normalized_language.split("-", maxsplit=1)[0].lower() or "en"
    return (
        "https://news.google.com/rss/search"
        f"?q={encoded}&hl={quote_plus(normalized_language)}"
        f"&gl={quote_plus(normalized_region)}"
        f"&ceid={quote_plus(f'{normalized_region}:{ceid_language}')}"
    )


def _target_news_path(news_dir: str | Path, ticker: str) -> Path:
    candidates = news_path_candidates(news_dir, ticker)
    return candidates[0]


def _read_existing_news(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)
    try:
        return _normalize_news_frame(pd.read_csv(path))
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)


def _normalize_news_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy() if frame is not None else pd.DataFrame()
    for column in NEWS_DISPLAY_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    if result.empty:
        return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)
    result["date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["sentiment_score"] = result["sentiment_score"].map(safe_float).fillna(0.0)
    result["importance_score"] = result["importance_score"].map(safe_float).fillna(0.0)
    result["source_credibility_score"] = result["source_credibility_score"].map(safe_float).fillna(0.0)
    result["source_market_impact_weight"] = result["source_market_impact_weight"].map(safe_float).fillna(0.0)
    for column in (
        "ticker",
        "title",
        "source",
        "url",
        "source_type",
        "category",
        "sentiment_reason",
        "importance_reason",
        "summary",
        "fetched_at",
        "query_profile",
    ):
        result[column] = result[column].fillna("").astype(str)
    for column in ("title", "source", "summary"):
        result[column] = result[column].map(_clean_text)
    result = result.dropna(subset=["date"])
    return result[list(NEWS_DISPLAY_COLUMNS)].reset_index(drop=True)


def _normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def _matched_keywords(text: str, keywords: tuple[str, ...] | set[str]) -> list[str]:
    return [keyword for keyword in keywords if keyword and keyword in text]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _dedupe_token_sets(frame: pd.DataFrame) -> tuple[set[str], set[str]]:
    if frame.empty:
        return set(), set()
    urls = set(frame.get("url", pd.Series(dtype="object")).fillna("").astype(str).str.strip().str.lower())
    titles = {
        _normalize_title(title)
        for title in frame.get("title", pd.Series(dtype="object")).fillna("").astype(str)
    }
    urls.discard("")
    titles.discard("")
    return urls, titles


def _normalize_title(title: str) -> str:
    return re.sub(r"\s+", " ", title.strip().lower())


def _text(element: ET.Element | None) -> str:
    return "" if element is None or element.text is None else str(element.text).strip()


def _source_text(item: ET.Element, title: str) -> str:
    source = _text(item.find("source"))
    if source:
        return source
    if " - " in title:
        return title.rsplit(" - ", maxsplit=1)[-1].strip()
    return "Google News"


def _clean_summary(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:500]


def _clean_text(value: str) -> str:
    text = html.unescape(str(value or ""))
    for old, new in MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def _parse_rss_date(value: str) -> str:
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        parsed = datetime.now(UTC)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.date().isoformat()


def _format_timestamp(value: pd.Timestamp) -> str:
    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
