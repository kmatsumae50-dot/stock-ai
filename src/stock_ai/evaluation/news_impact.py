from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from stock_ai.evaluation import risk_validation
from stock_ai.news.manual_csv import read_news_items
from stock_ai.news.profile_quality import (
    build_profile_quality_frame,
    build_profile_recommendations,
)
from stock_ai.news.query_profiles import news_query_profile_rows
from stock_ai.news.quality import is_mojibake_suspect
from stock_ai.utils.safe_cast import safe_float


NEWS_IMPACT_FEATURES = (
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

EVENT_COLUMNS = (
    "negative_news_flag",
    "positive_news_flag",
    "negative_news_shock",
    "positive_news_shock",
)

NEWS_EXAMPLE_COLUMNS = (
    "date",
    "base_trading_date",
    "title",
    "source",
    "source_type",
    "category",
    "sentiment_score",
    "importance_score",
    "return_1d",
    "return_5d",
    "return_20d",
    "max_up_20d",
    "max_down_20d",
    "sentiment_reason",
    "importance_reason",
    "url",
    "quality_note",
    "future_data_note",
)

NEWS_PRICE_MOVE_RANKING_COLUMNS = (
    "rank",
    "date",
    "base_trading_date",
    "title",
    "source",
    "source_type",
    "category",
    "sentiment_score",
    "importance_score",
    "source_credibility_score",
    "return_1d",
    "return_5d",
    "return_20d",
    "max_up_20d",
    "max_down_20d",
    "url",
    "future_data_note",
)

NEWS_PRICE_MOVE_RANKING_MODES = {
    "return_5d_desc": "5営業日後リターンが大きい順",
    "return_5d_asc": "5営業日後リターンが小さい順",
    "return_20d_desc": "20営業日後リターンが大きい順",
    "return_20d_asc": "20営業日後リターンが小さい順",
    "max_up_20d_desc": "20営業日以内最大上昇率が大きい順",
    "max_down_20d_asc": "20営業日以内最大下落率が大きい順",
    "importance_move_desc": "importance_scoreが高く、その後の値動きも大きい順",
}

CATEGORY_PRICE_MOVE_RANKING_COLUMNS = (
    "category",
    "news_count",
    "average_sentiment_score",
    "average_importance_score",
    "average_source_credibility",
    "average_return_1d",
    "average_return_5d",
    "average_return_20d",
    "average_max_up_20d",
    "average_max_down_20d",
    "large_up_rate",
    "large_down_rate",
    "average_crash_risk",
    "average_surge_risk",
)

CATEGORY_PRICE_MOVE_RANKING_MODES = {
    "average_return_5d_desc": "5D average return high to low",
    "average_return_5d_asc": "5D average return low to high",
    "average_return_20d_desc": "20D average return high to low",
    "average_return_20d_asc": "20D average return low to high",
    "average_max_up_20d_desc": "20D max upside high to low",
    "average_max_down_20d_asc": "20D max drawdown high to low",
    "large_down_rate_desc": "large-down rate high to low",
    "large_up_rate_desc": "large-up rate high to low",
    "average_importance_score_desc": "average importance high to low",
}

CATEGORY_SOURCE_IMPACT_COLUMNS = (
    "category",
    "source_type",
    "news_count",
    "average_sentiment_score",
    "average_importance_score",
    "average_source_credibility",
    "average_source_market_impact_weight",
    "average_return_1d",
    "average_return_5d",
    "average_return_20d",
    "average_max_up_20d",
    "average_max_down_20d",
    "large_up_rate",
    "large_down_rate",
    "average_crash_risk",
    "average_surge_risk",
)

CATEGORY_SOURCE_IMPACT_MODES = {
    "average_return_5d_desc": "5D average return high to low",
    "average_return_5d_asc": "5D average return low to high",
    "average_max_up_20d_desc": "20D max upside high to low",
    "average_max_down_20d_asc": "20D max drawdown high to low",
    "large_down_rate_desc": "large-down rate high to low",
    "large_up_rate_desc": "large-up rate high to low",
    "average_importance_score_desc": "average importance high to low",
    "average_source_credibility_desc": "average source credibility high to low",
}

CATEGORY_SOURCE_HEATMAP_METRICS = {
    "average_return_5d": "Average 5D return",
    "news_count": "News count",
    "average_crash_risk": "Average Crash Risk",
}

PROFILE_NEWS_IMPACT_COLUMNS = (
    "query_profile",
    "group",
    "enabled_for_daily",
    "language",
    "region",
    "news_count",
    "average_sentiment_score",
    "average_importance_score",
    "average_source_credibility_score",
    "average_source_market_impact_weight",
    "average_return_1d",
    "average_return_5d",
    "average_return_20d",
    "average_max_up_20d",
    "average_max_down_20d",
    "large_up_rate",
    "large_down_rate",
    "average_crash_risk",
    "average_surge_risk",
    "total_fetched_rows",
    "total_saved_rows",
    "latest_fetched_at",
    "latest_fetched_count",
    "latest_added_count",
    "latest_duplicate_count",
    "duplicate_count",
    "duplicate_rate",
    "added_count",
    "quality_status",
    "recommendation",
    "daily_decision_note",
)

PROFILE_NEWS_IMPACT_MODES = {
    "average_return_5d_desc": "5D average return high to low",
    "average_return_5d_asc": "5D average return low to high",
    "average_return_20d_desc": "20D average return high to low",
    "average_return_20d_asc": "20D average return low to high",
    "average_max_up_20d_desc": "20D max upside high to low",
    "average_max_down_20d_asc": "20D max drawdown high to low",
    "large_down_rate_desc": "large-down rate high to low",
    "large_up_rate_desc": "large-up rate high to low",
    "average_importance_score_desc": "average importance high to low",
    "average_source_credibility_desc": "average source credibility high to low",
}

DAILY_NEWS_PROFILE_RECOMMENDATION_COLUMNS = (
    "query_profile",
    "group",
    "enabled_for_daily",
    "language",
    "region",
    "total_saved_rows",
    "latest_fetched_at",
    "added_count",
    "duplicate_count",
    "duplicate_rate",
    "quality_status",
    "current_recommendation",
    "news_count",
    "average_return_5d",
    "average_return_20d",
    "average_max_up_20d",
    "average_max_down_20d",
    "large_up_rate",
    "large_down_rate",
    "average_crash_risk",
    "average_surge_risk",
    "average_importance_score",
    "average_source_credibility_score",
    "daily_operation_recommendation",
    "priority",
    "suggested_enabled_for_daily",
    "config_diff",
    "recommendation_reason",
)

DAILY_NEWS_PROFILE_ACTIVE_LABELS = ("daily_keep", "daily_candidate", "reduce_limit")

NEWS_LIMIT_COMPARISON_COLUMNS = (
    "query_profile",
    "enabled_for_daily",
    "group",
    "latest_fetched_count",
    "latest_added_count",
    "latest_duplicate_count",
    "duplicate_rate",
    "total_saved_rows",
    "quality_status",
    "daily_operation_recommendation",
    "limit_3_assessment",
    "limit_5_assessment",
    "limit_10_assessment",
    "suggested_daily_news_limit",
    "suggested_research_news_limit",
    "reason",
)


@dataclass(frozen=True)
class NewsImpactConfig:
    min_history_rows: int = 60
    high_news_quantile: float = 0.75
    positive_sentiment_threshold: float = 0.10
    negative_sentiment_threshold: float = -0.10
    high_importance_threshold: float = 0.70
    large_up_threshold: float = 0.03
    large_down_threshold: float = -0.03


def build_news_impact_frame(
    data: pd.DataFrame,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()

    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        config=risk_validation.RiskValidationConfig(min_history_rows=active.min_history_rows),
    )
    if risk_frame.empty:
        return pd.DataFrame()

    feature_columns = [column for column in NEWS_IMPACT_FEATURES if column in data.columns]
    if not feature_columns:
        return pd.DataFrame()

    news_frame = data.copy()
    news_frame["date"] = pd.to_datetime(news_frame["date"], errors="coerce").dt.normalize()
    if "ticker" not in news_frame.columns:
        news_frame["ticker"] = "UNKNOWN"
    news_frame["ticker"] = news_frame["ticker"].astype(str)
    news_frame = news_frame[["date", "ticker", *feature_columns]].copy()
    news_frame = news_frame.dropna(subset=["date"])

    frame = risk_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["ticker"] = frame["ticker"].astype(str)
    frame = frame.merge(news_frame, on=["date", "ticker"], how="left")

    for column in NEWS_IMPACT_FEATURES:
        if column not in frame.columns:
            frame[column] = 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    frame = _add_news_buckets(frame, active)
    frame = _add_risk_score_changes(frame)
    return frame


def build_news_impact_summaries(
    frame: pd.DataFrame,
    *,
    config: NewsImpactConfig | None = None,
) -> dict[str, pd.DataFrame]:
    active = config or NewsImpactConfig()
    if frame.empty:
        empty = pd.DataFrame()
        return {
            "volume": empty,
            "sentiment": empty,
            "importance": empty,
            "events": empty,
            "feature_relationships": empty,
        }
    return {
        "volume": _summarize_group(frame, "news_volume_bucket", config=active),
        "sentiment": _summarize_group(frame, "news_sentiment_bucket", config=active),
        "importance": _summarize_group(frame, "news_importance_bucket", config=active),
        "events": summarize_news_events(frame, config=active),
        "feature_relationships": summarize_feature_relationships(frame),
    }


def summarize_news_events(
    frame: pd.DataFrame,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for column in EVENT_COLUMNS:
        if column not in frame.columns:
            continue
        active_subset = frame[pd.to_numeric(frame[column], errors="coerce").fillna(0.0) > 0].copy()
        row = _summary_row(active_subset, config=active)
        row["event"] = column
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_query_profile_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()
    news = read_news_items(news_dir, ticker)
    if news.empty or "query_profile" not in news.columns:
        return _profile_availability_frame(news_dir)

    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        config=risk_validation.RiskValidationConfig(min_history_rows=active.min_history_rows),
    )
    if risk_frame.empty:
        return _profile_availability_frame(news_dir)

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    news["query_profile"] = news["query_profile"].fillna("").astype(str).str.strip()
    news = news.dropna(subset=["date"])
    if news.empty:
        return _profile_availability_frame(news_dir)

    frame = risk_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    merged = news.merge(frame, on="date", how="left", suffixes=("_news", ""))
    if merged.empty:
        return _profile_availability_frame(news_dir)

    rows: list[dict[str, object]] = []
    profile_names = [row["name"] for row in news_query_profile_rows()]
    for profile_name in profile_names:
        subset = merged[merged["query_profile"] == profile_name].copy()
        row = _summary_row(subset, config=active)
        row["query_profile"] = profile_name
        row["tracking_status"] = "tracked" if int(row["count"]) > 0 else "no tracked rows yet"
        rows.append(row)
    untracked = merged[merged["query_profile"].eq("")].copy()
    if not untracked.empty:
        row = _summary_row(untracked, config=active)
        row["query_profile"] = "(untracked)"
        row["tracking_status"] = "legacy/manual rows without query_profile"
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_category_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()
    news = read_news_items(news_dir, ticker)
    if news.empty or "category" not in news.columns:
        return pd.DataFrame()

    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        config=risk_validation.RiskValidationConfig(min_history_rows=active.min_history_rows),
    )
    if risk_frame.empty:
        return pd.DataFrame()

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    news["category"] = news["category"].fillna("").astype(str).str.strip()
    news.loc[news["category"].eq(""), "category"] = "other"
    news = news.dropna(subset=["date"])
    if news.empty:
        return pd.DataFrame()

    frame = risk_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    merged = news.merge(frame, on="date", how="left", suffixes=("_news", ""))
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for category, subset in merged.groupby("category", dropna=False, sort=False):
        row = _summary_row(subset, config=active)
        row["category"] = str(category or "other")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def summarize_source_type_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    if data.empty or "date" not in data.columns:
        return pd.DataFrame()
    news = read_news_items(news_dir, ticker)
    if news.empty or "source_type" not in news.columns:
        return pd.DataFrame()

    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        config=risk_validation.RiskValidationConfig(min_history_rows=active.min_history_rows),
    )
    if risk_frame.empty:
        return pd.DataFrame()

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    news["source_type"] = news["source_type"].fillna("").astype(str).str.strip()
    news.loc[news["source_type"].eq(""), "source_type"] = "unknown"
    news = news.dropna(subset=["date"])
    if news.empty:
        return pd.DataFrame()

    frame = risk_frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    merged = news.merge(frame, on="date", how="left", suffixes=("_news", ""))
    if merged.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for source_type, subset in merged.groupby("source_type", dropna=False, sort=False):
        row = _summary_row(subset, config=active)
        row["source_type"] = str(source_type or "unknown")
        row["average_source_credibility"] = _mean(subset.get("source_credibility_score"))
        row["average_source_weight"] = _mean(subset.get("source_market_impact_weight"))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def summarize_feature_relationships(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    targets = ("return_1d", "return_5d", "return_20d", "surge_risk_score", "crash_risk_score")
    for feature in NEWS_IMPACT_FEATURES:
        if feature not in frame.columns:
            continue
        feature_values = pd.to_numeric(frame[feature], errors="coerce")
        for target in targets:
            if target not in frame.columns:
                continue
            target_values = pd.to_numeric(frame[target], errors="coerce")
            valid = pd.DataFrame({"feature": feature_values, "target": target_values}).dropna()
            corr = None
            if len(valid) >= 5 and valid["feature"].nunique() > 1 and valid["target"].nunique() > 1:
                corr = float(valid["feature"].corr(valid["target"]))
            rows.append(
                {
                    "feature": feature,
                    "target": target,
                    "correlation": corr,
                    "valid_rows": int(len(valid)),
                }
            )
    return pd.DataFrame(rows)


def news_examples_for_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    dimension: str,
    value: str,
    config: NewsImpactConfig | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    news = read_news_items(news_dir, ticker)
    if news.empty:
        return pd.DataFrame(columns=NEWS_EXAMPLE_COLUMNS)

    news = news.copy()
    news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.normalize()
    news = news.dropna(subset=["date"])
    if news.empty:
        return pd.DataFrame(columns=NEWS_EXAMPLE_COLUMNS)

    selected = _select_news_examples(data, news, dimension=dimension, value=value, config=active)
    if selected.empty:
        return pd.DataFrame(columns=NEWS_EXAMPLE_COLUMNS)

    selected = selected.copy()
    for column in NEWS_EXAMPLE_COLUMNS:
        if column not in selected.columns and column not in {"quality_note", "future_data_note"}:
            selected[column] = ""
    selected["quality_note"] = selected.apply(_news_quality_note, axis=1)
    selected = add_news_following_price_moves(selected, data)
    selected["date"] = pd.to_datetime(selected["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    selected["base_trading_date"] = pd.to_datetime(
        selected["base_trading_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    selected["sentiment_score"] = pd.to_numeric(selected["sentiment_score"], errors="coerce").fillna(0.0)
    selected["importance_score"] = pd.to_numeric(selected["importance_score"], errors="coerce").fillna(0.0)
    selected = selected.sort_values(["date", "importance_score"], ascending=[False, False]).head(limit)
    return selected[list(NEWS_EXAMPLE_COLUMNS)].reset_index(drop=True)


def build_news_price_move_ranking(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    ranking_mode: str = "return_5d_desc",
    limit: int = 20,
) -> pd.DataFrame:
    ranking = _news_with_price_moves(data, news_dir, ticker)
    if ranking.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)

    for column in NEWS_PRICE_MOVE_RANKING_COLUMNS:
        if column not in ranking.columns and column not in {"rank", "future_data_note"}:
            ranking[column] = ""
    ranking = _sort_price_move_ranking(ranking, ranking_mode)
    ranking = ranking.head(limit).copy()
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking["date"] = pd.to_datetime(ranking["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    ranking["base_trading_date"] = pd.to_datetime(
        ranking["base_trading_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    for column in ("sentiment_score", "importance_score", "source_credibility_score"):
        ranking[column] = pd.to_numeric(ranking[column], errors="coerce")
    return ranking[list(NEWS_PRICE_MOVE_RANKING_COLUMNS)].reset_index(drop=True)


def build_category_price_move_ranking(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    ranking_mode: str = "average_return_5d_desc",
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    news = _news_with_price_moves(data, news_dir, ticker, config=active)
    if news.empty:
        return pd.DataFrame(columns=CATEGORY_PRICE_MOVE_RANKING_COLUMNS)

    news["category"] = news.get("category", pd.Series(index=news.index, dtype="object")).fillna("")
    news["category"] = news["category"].astype(str).str.strip()
    news.loc[news["category"].eq(""), "category"] = "other"
    numeric_columns = (
        "sentiment_score",
        "importance_score",
        "source_credibility_score",
        "return_1d",
        "return_5d",
        "return_20d",
        "max_up_20d",
        "max_down_20d",
        "crash_risk_score",
        "surge_risk_score",
    )
    for column in numeric_columns:
        news[column] = pd.to_numeric(news.get(column), errors="coerce")

    rows: list[dict[str, object]] = []
    for category, subset in news.groupby("category", sort=False, dropna=False):
        rows.append(
            {
                "category": str(category or "other"),
                "news_count": int(len(subset)),
                "average_sentiment_score": _mean(subset.get("sentiment_score")),
                "average_importance_score": _mean(subset.get("importance_score")),
                "average_source_credibility": _mean(subset.get("source_credibility_score")),
                "average_return_1d": _mean(subset.get("return_1d")),
                "average_return_5d": _mean(subset.get("return_5d")),
                "average_return_20d": _mean(subset.get("return_20d")),
                "average_max_up_20d": _mean(subset.get("max_up_20d")),
                "average_max_down_20d": _mean(subset.get("max_down_20d")),
                "large_up_rate": _rate(subset["max_up_20d"] >= active.large_up_threshold),
                "large_down_rate": _rate(subset["max_down_20d"] <= active.large_down_threshold),
                "average_crash_risk": _mean(subset.get("crash_risk_score")),
                "average_surge_risk": _mean(subset.get("surge_risk_score")),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=CATEGORY_PRICE_MOVE_RANKING_COLUMNS)
    ranking = _sort_category_price_move_ranking(ranking, ranking_mode)
    return ranking[list(CATEGORY_PRICE_MOVE_RANKING_COLUMNS)].reset_index(drop=True)


def category_price_move_examples(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    category: str,
    *,
    limit: int = 3,
) -> pd.DataFrame:
    news = _news_with_price_moves(data, news_dir, ticker)
    if news.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    category_text = news.get("category", pd.Series(index=news.index, dtype="object")).fillna("")
    category_text = category_text.astype(str).str.strip()
    category_text = category_text.mask(category_text.eq(""), "other")
    examples = news[category_text.eq(str(category or "other"))].copy()
    if examples.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    examples = _sort_price_move_ranking(examples, "importance_move_desc").head(limit).copy()
    for column in NEWS_PRICE_MOVE_RANKING_COLUMNS:
        if column not in examples.columns and column not in {"rank", "future_data_note"}:
            examples[column] = ""
    examples.insert(0, "rank", range(1, len(examples) + 1))
    examples["date"] = pd.to_datetime(examples["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    examples["base_trading_date"] = pd.to_datetime(
        examples["base_trading_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    return examples[list(NEWS_PRICE_MOVE_RANKING_COLUMNS)].reset_index(drop=True)


def build_category_source_type_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    ranking_mode: str = "average_return_5d_desc",
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    news = _news_with_price_moves(data, news_dir, ticker, config=active)
    if news.empty:
        return pd.DataFrame(columns=CATEGORY_SOURCE_IMPACT_COLUMNS)

    news = _normalize_news_group_columns(news)
    numeric_columns = (
        "sentiment_score",
        "importance_score",
        "source_credibility_score",
        "source_market_impact_weight",
        "return_1d",
        "return_5d",
        "return_20d",
        "max_up_20d",
        "max_down_20d",
        "crash_risk_score",
        "surge_risk_score",
    )
    for column in numeric_columns:
        news[column] = pd.to_numeric(news.get(column), errors="coerce")

    rows: list[dict[str, object]] = []
    for (category, source_type), subset in news.groupby(
        ["category", "source_type"],
        sort=False,
        dropna=False,
    ):
        rows.append(
            {
                "category": str(category or "other"),
                "source_type": str(source_type or "unknown"),
                "news_count": int(len(subset)),
                "average_sentiment_score": _mean(subset.get("sentiment_score")),
                "average_importance_score": _mean(subset.get("importance_score")),
                "average_source_credibility": _mean(subset.get("source_credibility_score")),
                "average_source_market_impact_weight": _mean(
                    subset.get("source_market_impact_weight")
                ),
                "average_return_1d": _mean(subset.get("return_1d")),
                "average_return_5d": _mean(subset.get("return_5d")),
                "average_return_20d": _mean(subset.get("return_20d")),
                "average_max_up_20d": _mean(subset.get("max_up_20d")),
                "average_max_down_20d": _mean(subset.get("max_down_20d")),
                "large_up_rate": _rate(subset["max_up_20d"] >= active.large_up_threshold),
                "large_down_rate": _rate(subset["max_down_20d"] <= active.large_down_threshold),
                "average_crash_risk": _mean(subset.get("crash_risk_score")),
                "average_surge_risk": _mean(subset.get("surge_risk_score")),
            }
        )
    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=CATEGORY_SOURCE_IMPACT_COLUMNS)
    ranking = _sort_category_source_type_impact(ranking, ranking_mode)
    return ranking[list(CATEGORY_SOURCE_IMPACT_COLUMNS)].reset_index(drop=True)


def category_source_type_examples(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    category: str,
    source_type: str,
    *,
    limit: int = 3,
) -> pd.DataFrame:
    news = _news_with_price_moves(data, news_dir, ticker)
    if news.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    news = _normalize_news_group_columns(news)
    examples = news[
        news["category"].eq(str(category or "other"))
        & news["source_type"].eq(str(source_type or "unknown"))
    ].copy()
    if examples.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    examples = _sort_price_move_ranking(examples, "importance_move_desc").head(limit).copy()
    for column in NEWS_PRICE_MOVE_RANKING_COLUMNS:
        if column not in examples.columns and column not in {"rank", "future_data_note"}:
            examples[column] = ""
    examples.insert(0, "rank", range(1, len(examples) + 1))
    examples["date"] = pd.to_datetime(examples["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    examples["base_trading_date"] = pd.to_datetime(
        examples["base_trading_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    return examples[list(NEWS_PRICE_MOVE_RANKING_COLUMNS)].reset_index(drop=True)


def build_profile_news_impact(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    ranking_mode: str = "average_return_5d_desc",
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    news = _news_with_price_moves(data, news_dir, ticker, config=active)
    quality = build_profile_quality_frame(news_dir, ticker)
    recommendations = build_profile_recommendations(quality)
    profile_rows = news_query_profile_rows()
    if news.empty and not profile_rows:
        return pd.DataFrame(columns=PROFILE_NEWS_IMPACT_COLUMNS)

    news = news.copy() if not news.empty else pd.DataFrame()
    if not news.empty:
        news["query_profile"] = (
            news.get("query_profile", pd.Series(index=news.index, dtype="object"))
            .fillna("")
            .astype(str)
            .str.strip()
        )
        numeric_columns = (
            "sentiment_score",
            "importance_score",
            "source_credibility_score",
            "source_market_impact_weight",
            "return_1d",
            "return_5d",
            "return_20d",
            "max_up_20d",
            "max_down_20d",
            "crash_risk_score",
            "surge_risk_score",
        )
        for column in numeric_columns:
            news[column] = pd.to_numeric(news.get(column), errors="coerce")

    profile_meta = {str(row.get("name") or ""): row for row in profile_rows}
    configured_names = [name for name in profile_meta if name]
    observed_names = []
    if not news.empty and "query_profile" in news.columns:
        observed_names = [
            value
            for value in sorted(news["query_profile"].dropna().astype(str).str.strip().unique())
            if value
        ]
    profile_names = _dedupe_ordered([*configured_names, *observed_names])

    rows: list[dict[str, object]] = []
    for profile_name in profile_names:
        subset = (
            news[news["query_profile"].eq(profile_name)].copy()
            if not news.empty and "query_profile" in news.columns
            else pd.DataFrame()
        )
        rows.append(
            _profile_news_impact_row(
                profile_name,
                subset,
                profile_meta.get(profile_name, {}),
                quality,
                recommendations,
                config=active,
            )
        )

    if not news.empty:
        untracked = news[news["query_profile"].eq("")].copy()
        if not untracked.empty:
            rows.append(
                _profile_news_impact_row(
                    "(untracked)",
                    untracked,
                    {},
                    quality,
                    recommendations,
                    config=active,
                )
            )

    ranking = pd.DataFrame(rows)
    if ranking.empty:
        return pd.DataFrame(columns=PROFILE_NEWS_IMPACT_COLUMNS)
    ranking = _sort_profile_news_impact(ranking, ranking_mode)
    return ranking[list(PROFILE_NEWS_IMPACT_COLUMNS)].reset_index(drop=True)


def profile_news_impact_examples(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    query_profile: str,
    *,
    limit: int = 3,
) -> pd.DataFrame:
    news = _news_with_price_moves(data, news_dir, ticker)
    if news.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    profile = news.get("query_profile", pd.Series(index=news.index, dtype="object")).fillna("")
    profile = profile.astype(str).str.strip()
    value = str(query_profile or "").strip()
    if value == "(untracked)":
        examples = news[profile.eq("")].copy()
    else:
        examples = news[profile.eq(value)].copy()
    if examples.empty:
        return pd.DataFrame(columns=NEWS_PRICE_MOVE_RANKING_COLUMNS)
    examples = _sort_price_move_ranking(examples, "importance_move_desc").head(limit).copy()
    for column in NEWS_PRICE_MOVE_RANKING_COLUMNS:
        if column not in examples.columns and column not in {"rank", "future_data_note"}:
            examples[column] = ""
    examples.insert(0, "rank", range(1, len(examples) + 1))
    examples["date"] = pd.to_datetime(examples["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    examples["base_trading_date"] = pd.to_datetime(
        examples["base_trading_date"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    return examples[list(NEWS_PRICE_MOVE_RANKING_COLUMNS)].reset_index(drop=True)


def build_daily_news_profile_recommendations(profile_impact: pd.DataFrame) -> pd.DataFrame:
    """Combine fetch quality and impact signals into daily-operation suggestions."""
    if profile_impact.empty:
        return pd.DataFrame(columns=DAILY_NEWS_PROFILE_RECOMMENDATION_COLUMNS)

    rows = [_daily_news_profile_recommendation_row(row) for _, row in profile_impact.iterrows()]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=DAILY_NEWS_PROFILE_RECOMMENDATION_COLUMNS)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    label_order = {
        "review_needed": 0,
        "disable_candidate": 1,
        "reduce_limit": 2,
        "daily_candidate": 3,
        "daily_keep": 4,
        "research_only": 5,
    }
    result["_priority_order"] = result["priority"].map(priority_order).fillna(9)
    result["_label_order"] = result["daily_operation_recommendation"].map(label_order).fillna(9)
    result["_enabled_order"] = result["enabled_for_daily"].map(_as_bool).astype(int) * -1
    result = result.sort_values(
        ["_priority_order", "_label_order", "_enabled_order", "query_profile"],
        ascending=[True, True, True, True],
        na_position="last",
    ).drop(columns=["_priority_order", "_label_order", "_enabled_order"])
    return result[list(DAILY_NEWS_PROFILE_RECOMMENDATION_COLUMNS)].reset_index(drop=True)


def recommended_daily_profile_names(recommendations: pd.DataFrame) -> list[str]:
    if recommendations.empty:
        return []
    active = recommendations[
        recommendations["daily_operation_recommendation"].astype(str).isin(
            DAILY_NEWS_PROFILE_ACTIVE_LABELS
        )
    ].copy()
    return [
        str(value)
        for value in active.get("query_profile", pd.Series(dtype="object")).fillna("").tolist()
        if str(value).strip() and str(value) != "(untracked)"
    ]


def daily_news_profile_candidate_names(recommendations: pd.DataFrame) -> list[str]:
    if recommendations.empty:
        return []
    active = recommendations[
        recommendations["daily_operation_recommendation"].astype(str).eq("daily_candidate")
    ].copy()
    return [
        str(value)
        for value in active.get("query_profile", pd.Series(dtype="object")).fillna("").tolist()
        if str(value).strip() and str(value) != "(untracked)"
    ]


def build_daily_news_profile_recommendation_messages(
    recommendations: pd.DataFrame,
    *,
    max_items: int = 2,
) -> list[str]:
    if recommendations.empty:
        return []
    frame = recommendations.copy()
    focus = frame[
        frame["daily_operation_recommendation"].astype(str).isin(
            ("review_needed", "disable_candidate", "reduce_limit", "daily_candidate")
        )
    ].copy()
    if focus.empty:
        focus = frame[frame["enabled_for_daily"].map(_as_bool)].copy()
    if focus.empty:
        focus = frame.copy()
    messages: list[str] = []
    for _, row in focus.head(max_items).iterrows():
        profile = str(row.get("query_profile") or "-")
        label = str(row.get("daily_operation_recommendation") or "-")
        reason = str(row.get("recommendation_reason") or "")
        messages.append(f"{profile}: {label}. {reason}")
    return messages


def build_news_limit_recommendations(
    daily_recommendations: pd.DataFrame,
) -> pd.DataFrame:
    if daily_recommendations.empty:
        return pd.DataFrame(columns=NEWS_LIMIT_COMPARISON_COLUMNS)
    rows = [_news_limit_recommendation_row(row) for _, row in daily_recommendations.iterrows()]
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=NEWS_LIMIT_COMPARISON_COLUMNS)
    order = {"daily_keep": 0, "reduce_limit": 1, "daily_candidate": 2, "review_needed": 3, "research_only": 4, "disable_candidate": 5}
    result["_order"] = result["daily_operation_recommendation"].map(order).fillna(9)
    result["_enabled"] = result["enabled_for_daily"].map(_as_bool).astype(int) * -1
    result = result.sort_values(
        ["_order", "_enabled", "query_profile"],
        ascending=[True, True, True],
        na_position="last",
    ).drop(columns=["_order", "_enabled"])
    return result[list(NEWS_LIMIT_COMPARISON_COLUMNS)].reset_index(drop=True)


def recommended_daily_news_limit(news_limit_recommendations: pd.DataFrame) -> int | None:
    if news_limit_recommendations.empty:
        return None
    daily = news_limit_recommendations[
        news_limit_recommendations["enabled_for_daily"].map(_as_bool)
    ].copy()
    if daily.empty:
        return None
    suggested = pd.to_numeric(daily["suggested_daily_news_limit"], errors="coerce").dropna()
    if suggested.empty:
        return None
    return int(suggested.median())


def category_source_type_heatmap_frame(
    category_source_frame: pd.DataFrame,
    *,
    metric: str = "average_return_5d",
) -> pd.DataFrame:
    if category_source_frame.empty or metric not in CATEGORY_SOURCE_HEATMAP_METRICS:
        return pd.DataFrame()
    required = {"category", "source_type", metric}
    if not required.issubset(category_source_frame.columns):
        return pd.DataFrame()
    frame = category_source_frame[list(required)].copy()
    frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    if metric == "news_count":
        frame[metric] = frame[metric].fillna(0)
    return frame.pivot_table(
        index="category",
        columns="source_type",
        values=metric,
        aggfunc="mean",
        dropna=False,
    )


def news_price_move_ranking_mode_rows() -> list[dict[str, str]]:
    return [
        {"mode": mode, "label": label}
        for mode, label in NEWS_PRICE_MOVE_RANKING_MODES.items()
    ]


def category_price_move_ranking_mode_rows() -> list[dict[str, str]]:
    return [
        {"mode": mode, "label": label}
        for mode, label in CATEGORY_PRICE_MOVE_RANKING_MODES.items()
    ]


def category_source_type_impact_mode_rows() -> list[dict[str, str]]:
    return [
        {"mode": mode, "label": label}
        for mode, label in CATEGORY_SOURCE_IMPACT_MODES.items()
    ]


def profile_news_impact_mode_rows() -> list[dict[str, str]]:
    return [
        {"mode": mode, "label": label}
        for mode, label in PROFILE_NEWS_IMPACT_MODES.items()
    ]


def category_source_type_heatmap_metric_rows() -> list[dict[str, str]]:
    return [
        {"metric": metric, "label": label}
        for metric, label in CATEGORY_SOURCE_HEATMAP_METRICS.items()
    ]


def add_news_following_price_moves(news: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    result = news.copy()
    result["base_trading_date"] = pd.NaT
    for column in ("return_1d", "return_5d", "return_20d", "max_up_20d", "max_down_20d"):
        result[column] = float("nan")
    result["future_data_note"] = ""

    if result.empty:
        return result
    if data.empty or "date" not in data.columns or "close" not in data.columns:
        result["future_data_note"] = "price data unavailable"
        return result

    price = data[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if price.empty:
        result["future_data_note"] = "price data unavailable"
        return result

    trading_dates = price["date"]
    closes = price["close"]
    for index, row in result.iterrows():
        news_date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(news_date):
            result.at[index, "future_data_note"] = "invalid news date"
            continue
        base_position = int(trading_dates.searchsorted(news_date.normalize(), side="left"))
        if base_position >= len(price):
            result.at[index, "future_data_note"] = "not enough future data"
            continue

        base_close = safe_float(closes.iloc[base_position])
        base_date = trading_dates.iloc[base_position]
        result.at[index, "base_trading_date"] = base_date
        if base_close is None or base_close == 0:
            result.at[index, "future_data_note"] = "invalid base close"
            continue

        missing: list[str] = []
        for days in (1, 5, 20):
            target_position = base_position + days
            column = f"return_{days}d"
            if target_position >= len(price):
                result.at[index, column] = None
                missing.append(f"{days}d")
                continue
            target_close = safe_float(closes.iloc[target_position])
            result.at[index, column] = _return_from_close(base_close, target_close)

        future_end = min(base_position + 21, len(price))
        future_closes = closes.iloc[base_position + 1 : future_end]
        future_returns = [
            _return_from_close(base_close, safe_float(close))
            for close in future_closes
        ]
        future_returns = [value for value in future_returns if value is not None]
        if future_returns:
            window_returns = [0.0, *future_returns]
            result.at[index, "max_up_20d"] = max(window_returns)
            result.at[index, "max_down_20d"] = min(window_returns)
        else:
            result.at[index, "max_up_20d"] = None
            result.at[index, "max_down_20d"] = None
            if "20d path" not in missing:
                missing.append("20d path")

        result.at[index, "future_data_note"] = (
            f"not enough future data: {', '.join(missing)}" if missing else ""
        )
    return result


def _news_with_price_moves(
    data: pd.DataFrame,
    news_dir: str | Path,
    ticker: str,
    *,
    config: NewsImpactConfig | None = None,
) -> pd.DataFrame:
    active = config or NewsImpactConfig()
    news = read_news_items(news_dir, ticker)
    if news.empty:
        return pd.DataFrame()
    result = add_news_following_price_moves(news.copy(), data)
    result["quality_note"] = result.apply(_news_quality_note, axis=1)
    result = _add_price_move_ranking_score(result)
    return _add_risk_scores_for_news_base_dates(result, data, config=active)


def _add_risk_scores_for_news_base_dates(
    news: pd.DataFrame,
    data: pd.DataFrame,
    *,
    config: NewsImpactConfig,
) -> pd.DataFrame:
    result = news.copy()
    result["crash_risk_score"] = float("nan")
    result["surge_risk_score"] = float("nan")
    risk_frame = risk_validation.build_risk_validation_frame(
        data,
        config=risk_validation.RiskValidationConfig(min_history_rows=config.min_history_rows),
    )
    if risk_frame.empty or "date" not in risk_frame.columns:
        return result
    lookup = risk_frame[["date", "crash_risk_score", "surge_risk_score"]].copy()
    lookup["base_trading_date"] = pd.to_datetime(lookup["date"], errors="coerce").dt.normalize()
    result["base_trading_date"] = pd.to_datetime(result["base_trading_date"], errors="coerce").dt.normalize()
    result = result.merge(
        lookup[["base_trading_date", "crash_risk_score", "surge_risk_score"]],
        on="base_trading_date",
        how="left",
        suffixes=("", "_risk"),
    )
    for risk_type in ("crash", "surge"):
        base_column = f"{risk_type}_risk_score"
        merged_column = f"{risk_type}_risk_score_risk"
        if merged_column in result.columns:
            result[base_column] = pd.to_numeric(result[merged_column], errors="coerce")
            result = result.drop(columns=[merged_column])
    return result


def _add_price_move_ranking_score(news: pd.DataFrame) -> pd.DataFrame:
    result = news.copy()
    for column in ("return_5d", "return_20d", "max_up_20d", "max_down_20d", "importance_score"):
        result[column] = pd.to_numeric(result.get(column), errors="coerce")
    move_strength = pd.concat(
        [
            result["return_5d"].abs(),
            result["return_20d"].abs(),
            result["max_up_20d"].abs(),
            result["max_down_20d"].abs(),
        ],
        axis=1,
    ).max(axis=1, skipna=True)
    result["importance_move_score"] = result["importance_score"].fillna(0.0) * move_strength.fillna(0.0)
    return result


def _normalize_news_group_columns(news: pd.DataFrame) -> pd.DataFrame:
    result = news.copy()
    result["category"] = result.get("category", pd.Series(index=result.index, dtype="object")).fillna("")
    result["category"] = result["category"].astype(str).str.strip()
    result.loc[result["category"].eq(""), "category"] = "other"
    result["source_type"] = result.get(
        "source_type",
        pd.Series(index=result.index, dtype="object"),
    ).fillna("")
    result["source_type"] = result["source_type"].astype(str).str.strip()
    result.loc[result["source_type"].eq(""), "source_type"] = "unknown"
    return result


def _sort_category_price_move_ranking(news: pd.DataFrame, ranking_mode: str) -> pd.DataFrame:
    mode = ranking_mode if ranking_mode in CATEGORY_PRICE_MOVE_RANKING_MODES else "average_return_5d_desc"
    sort_map = {
        "average_return_5d_desc": ("average_return_5d", False),
        "average_return_5d_asc": ("average_return_5d", True),
        "average_return_20d_desc": ("average_return_20d", False),
        "average_return_20d_asc": ("average_return_20d", True),
        "average_max_up_20d_desc": ("average_max_up_20d", False),
        "average_max_down_20d_asc": ("average_max_down_20d", True),
        "large_down_rate_desc": ("large_down_rate", False),
        "large_up_rate_desc": ("large_up_rate", False),
        "average_importance_score_desc": ("average_importance_score", False),
    }
    sort_column, ascending = sort_map[mode]
    result = news.copy()
    result[sort_column] = pd.to_numeric(result.get(sort_column), errors="coerce")
    result = result.sort_values(
        [sort_column, "news_count", "average_importance_score", "category"],
        ascending=[ascending, False, False, True],
        na_position="last",
    )
    return result.reset_index(drop=True)


def _sort_category_source_type_impact(news: pd.DataFrame, ranking_mode: str) -> pd.DataFrame:
    mode = ranking_mode if ranking_mode in CATEGORY_SOURCE_IMPACT_MODES else "average_return_5d_desc"
    sort_map = {
        "average_return_5d_desc": ("average_return_5d", False),
        "average_return_5d_asc": ("average_return_5d", True),
        "average_max_up_20d_desc": ("average_max_up_20d", False),
        "average_max_down_20d_asc": ("average_max_down_20d", True),
        "large_down_rate_desc": ("large_down_rate", False),
        "large_up_rate_desc": ("large_up_rate", False),
        "average_importance_score_desc": ("average_importance_score", False),
        "average_source_credibility_desc": ("average_source_credibility", False),
    }
    sort_column, ascending = sort_map[mode]
    result = news.copy()
    result[sort_column] = pd.to_numeric(result.get(sort_column), errors="coerce")
    result = result.sort_values(
        [sort_column, "news_count", "average_importance_score", "category", "source_type"],
        ascending=[ascending, False, False, True, True],
        na_position="last",
    )
    return result.reset_index(drop=True)


def _sort_profile_news_impact(news: pd.DataFrame, ranking_mode: str) -> pd.DataFrame:
    mode = ranking_mode if ranking_mode in PROFILE_NEWS_IMPACT_MODES else "average_return_5d_desc"
    sort_map = {
        "average_return_5d_desc": ("average_return_5d", False),
        "average_return_5d_asc": ("average_return_5d", True),
        "average_return_20d_desc": ("average_return_20d", False),
        "average_return_20d_asc": ("average_return_20d", True),
        "average_max_up_20d_desc": ("average_max_up_20d", False),
        "average_max_down_20d_asc": ("average_max_down_20d", True),
        "large_down_rate_desc": ("large_down_rate", False),
        "large_up_rate_desc": ("large_up_rate", False),
        "average_importance_score_desc": ("average_importance_score", False),
        "average_source_credibility_desc": ("average_source_credibility_score", False),
    }
    sort_column, ascending = sort_map[mode]
    result = news.copy()
    result[sort_column] = pd.to_numeric(result.get(sort_column), errors="coerce")
    result = result.sort_values(
        [sort_column, "news_count", "average_importance_score", "query_profile"],
        ascending=[ascending, False, False, True],
        na_position="last",
    )
    return result.reset_index(drop=True)


def _sort_price_move_ranking(news: pd.DataFrame, ranking_mode: str) -> pd.DataFrame:
    mode = ranking_mode if ranking_mode in NEWS_PRICE_MOVE_RANKING_MODES else "return_5d_desc"
    sort_map = {
        "return_5d_desc": ("return_5d", False),
        "return_5d_asc": ("return_5d", True),
        "return_20d_desc": ("return_20d", False),
        "return_20d_asc": ("return_20d", True),
        "max_up_20d_desc": ("max_up_20d", False),
        "max_down_20d_asc": ("max_down_20d", True),
        "importance_move_desc": ("importance_move_score", False),
    }
    sort_column, ascending = sort_map[mode]
    result = news.copy()
    result[sort_column] = pd.to_numeric(result.get(sort_column), errors="coerce")
    result = result.sort_values(
        [sort_column, "importance_score", "date"],
        ascending=[ascending, False, False],
        na_position="last",
    )
    return result.reset_index(drop=True)


def build_news_price_move_chart_frame(
    data: pd.DataFrame,
    news_date,
    *,
    pre_days: int = 5,
    post_days: int = 20,
) -> pd.DataFrame:
    if data.empty or "date" not in data.columns or "close" not in data.columns:
        return pd.DataFrame()

    price = data[["date", "close"]].copy()
    price["date"] = pd.to_datetime(price["date"], errors="coerce").dt.normalize()
    price["close"] = pd.to_numeric(price["close"], errors="coerce")
    price = price.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    if price.empty:
        return pd.DataFrame()

    parsed_news_date = pd.to_datetime(news_date, errors="coerce")
    if pd.isna(parsed_news_date):
        return pd.DataFrame()

    trading_dates = price["date"]
    base_position = int(trading_dates.searchsorted(parsed_news_date.normalize(), side="left"))
    if base_position >= len(price):
        return pd.DataFrame()

    base_close = safe_float(price.loc[base_position, "close"])
    if base_close is None or base_close == 0:
        return pd.DataFrame()

    start_position = max(0, base_position - pre_days)
    end_position = min(len(price), base_position + post_days + 1)
    window = price.iloc[start_position:end_position].copy()
    window["day_offset"] = [position - base_position for position in range(start_position, end_position)]
    window["base_trading_date"] = trading_dates.iloc[base_position]
    window["is_base_trading_date"] = window["day_offset"].eq(0)
    window["cumulative_return"] = pd.to_numeric(window["close"], errors="coerce") / base_close - 1.0
    return window[
        [
            "date",
            "day_offset",
            "close",
            "cumulative_return",
            "base_trading_date",
            "is_base_trading_date",
        ]
    ].reset_index(drop=True)


def build_news_impact_messages(
    summaries: dict[str, pd.DataFrame],
    *,
    min_count: int = 5,
) -> list[str]:
    messages: list[str] = []
    events = summaries.get("events", pd.DataFrame())
    if not events.empty:
        negative = _event_row(events, "negative_news_flag")
        positive = _event_row(events, "positive_news_flag")
        negative_shock = _event_row(events, "negative_news_shock")
        positive_shock = _event_row(events, "positive_news_shock")
        if negative is not None:
            messages.append(_event_message("negative_news_flag", negative, min_count=min_count))
        if positive is not None:
            messages.append(_event_message("positive_news_flag", positive, min_count=min_count))
        if negative_shock is not None:
            messages.append(_event_message("negative_news_shock", negative_shock, min_count=min_count))
        if positive_shock is not None:
            messages.append(_event_message("positive_news_shock", positive_shock, min_count=min_count))

    importance = summaries.get("importance", pd.DataFrame())
    high_importance = _group_row(importance, "High Importance")
    if high_importance is not None:
        count = int(high_importance.get("count") or 0)
        avg_5d = safe_float(high_importance.get("average_return_5d"))
        if count >= min_count:
            messages.append(
                f"High-importance news has {count} samples; average 5D return is {_format_percent(avg_5d)}."
            )
        else:
            messages.append("High-importance news has too few samples for a stable read.")

    if not messages:
        messages.append("News Impact Validation has no stable sample yet.")
    return messages


def build_source_type_impact_messages(
    source_type_frame: pd.DataFrame,
    *,
    max_items: int = 3,
    min_count: int = 5,
) -> list[str]:
    if source_type_frame.empty:
        return []

    messages: list[str] = []
    frame = source_type_frame.copy()
    frame["count"] = pd.to_numeric(frame.get("count"), errors="coerce").fillna(0).astype(int)
    frame = frame.sort_values("count", ascending=False).head(max_items)

    for _, row in frame.iterrows():
        source_type = str(row.get("source_type") or "unknown")
        count = int(row.get("count") or 0)
        average_5d = _format_percent(row.get("average_return_5d"))
        crash_risk = _format_number(row.get("average_crash_risk"))
        surge_risk = _format_number(row.get("average_surge_risk"))
        credibility = _format_number(row.get("average_source_credibility"), digits=2)
        source_weight = _format_number(row.get("average_source_weight"), digits=2)
        if count < min_count:
            messages.append(
                f"{source_type}: {count} samples; interpretation is limited. "
                f"Average 5D return {average_5d}, Crash Risk {crash_risk}, "
                f"Surge Risk {surge_risk}."
            )
        else:
            messages.append(
                f"{source_type}: {count} samples; average 5D return {average_5d}, "
                f"Crash Risk {crash_risk}, Surge Risk {surge_risk}, "
                f"source credibility {credibility}, source weight {source_weight}."
            )
    return messages


def build_profile_news_impact_messages(
    profile_frame: pd.DataFrame,
    *,
    max_items: int = 2,
    min_count: int = 3,
) -> list[str]:
    if profile_frame.empty:
        return []
    frame = profile_frame.copy()
    frame["news_count"] = pd.to_numeric(frame.get("news_count"), errors="coerce").fillna(0).astype(int)
    if "enabled_for_daily" in frame.columns:
        daily = frame[frame["enabled_for_daily"].map(_as_bool)].copy()
    else:
        daily = pd.DataFrame()
    if daily.empty:
        daily = frame.copy()
    daily = daily.sort_values(["news_count", "average_importance_score"], ascending=[False, False]).head(max_items)
    messages: list[str] = []
    for _, row in daily.iterrows():
        profile = str(row.get("query_profile") or "-")
        count = int(row.get("news_count") or 0)
        average_5d = _format_percent(row.get("average_return_5d"))
        recommendation = str(row.get("recommendation") or "-")
        duplicate_rate = _format_percent(row.get("duplicate_rate"))
        if count < min_count:
            messages.append(
                f"{profile}: {count} saved rows; impact read is limited. "
                f"Recommendation {recommendation}, duplicate rate {duplicate_rate}."
            )
        else:
            messages.append(
                f"{profile}: {count} saved rows; average 5D return {average_5d}. "
                f"Recommendation {recommendation}, duplicate rate {duplicate_rate}."
            )
    return messages


def current_news_impact_context(
    frame: pd.DataFrame,
    summaries: dict[str, pd.DataFrame],
) -> str:
    if frame.empty:
        return "News Impact Validation is not available yet."
    latest = frame.sort_values("date").iloc[-1]
    parts: list[str] = []
    volume_bucket = str(latest.get("news_volume_bucket") or "Unknown")
    sentiment_bucket = str(latest.get("news_sentiment_bucket") or "Unknown")
    parts.append(f"Current news volume is {volume_bucket}; sentiment bucket is {sentiment_bucket}.")

    events = summaries.get("events", pd.DataFrame())
    if bool(safe_float(latest.get("negative_news_shock")) or 0.0):
        row = _event_row(events, "negative_news_shock")
        parts.append(_current_event_text("negative news shock", row))
    elif bool(safe_float(latest.get("positive_news_shock")) or 0.0):
        row = _event_row(events, "positive_news_shock")
        parts.append(_current_event_text("positive news shock", row))
    elif bool(safe_float(latest.get("negative_news_flag")) or 0.0):
        row = _event_row(events, "negative_news_flag")
        parts.append(_current_event_text("negative news flag", row))
    elif bool(safe_float(latest.get("positive_news_flag")) or 0.0):
        row = _event_row(events, "positive_news_flag")
        parts.append(_current_event_text("positive news flag", row))
    else:
        parts.append("No active news shock flag is present today.")
    return " ".join(parts)


def _add_news_buckets(frame: pd.DataFrame, config: NewsImpactConfig) -> pd.DataFrame:
    result = frame.copy()
    count_3d = pd.to_numeric(result["news_count_3d"], errors="coerce").fillna(0.0)
    positive_counts = count_3d[count_3d > 0]
    high_count = (
        float(positive_counts.quantile(config.high_news_quantile))
        if not positive_counts.empty
        else 1.0
    )
    high_count = max(1.0, high_count)
    result["news_volume_bucket"] = "No News"
    result.loc[(count_3d > 0) & (count_3d < high_count), "news_volume_bucket"] = "Low / Medium News"
    result.loc[count_3d >= high_count, "news_volume_bucket"] = "High News"

    sentiment = pd.to_numeric(result["avg_sentiment_3d"], errors="coerce").fillna(0.0)
    result["news_sentiment_bucket"] = "Neutral"
    result.loc[sentiment >= config.positive_sentiment_threshold, "news_sentiment_bucket"] = "Positive"
    result.loc[sentiment <= config.negative_sentiment_threshold, "news_sentiment_bucket"] = "Negative"

    importance = pd.to_numeric(result["max_importance_3d"], errors="coerce").fillna(0.0)
    result["news_importance_bucket"] = "Normal Importance"
    result.loc[importance >= config.high_importance_threshold, "news_importance_bucket"] = "High Importance"
    return result


def _add_risk_score_changes(frame: pd.DataFrame) -> pd.DataFrame:
    groups = []
    for _ticker, group in frame.groupby("ticker", sort=False):
        item = group.sort_values("date").copy()
        for risk_type in ("surge", "crash"):
            column = f"{risk_type}_risk_score"
            item[f"{risk_type}_risk_change_1d"] = (
                pd.to_numeric(item[column], errors="coerce").shift(-1)
                - pd.to_numeric(item[column], errors="coerce")
            )
        groups.append(item)
    return pd.concat(groups, ignore_index=True) if groups else frame


def _select_news_examples(
    data: pd.DataFrame,
    news: pd.DataFrame,
    *,
    dimension: str,
    value: str,
    config: NewsImpactConfig,
) -> pd.DataFrame:
    value_text = str(value or "").strip()
    if not value_text:
        return pd.DataFrame()

    if dimension == "category":
        category = news.get("category", pd.Series(index=news.index, dtype="object")).fillna("")
        return news[category.astype(str).str.strip().eq(value_text)].copy()

    if dimension == "source_type":
        source_type = news.get("source_type", pd.Series(index=news.index, dtype="object")).fillna("")
        source_type = source_type.astype(str).str.strip()
        source_type = source_type.mask(source_type.eq(""), "unknown")
        return news[source_type.eq(value_text)].copy()

    impact_frame = build_news_impact_frame(data, config=config)
    if impact_frame.empty or "date" not in impact_frame.columns:
        return pd.DataFrame()
    impact_frame = impact_frame.copy()
    impact_frame["date"] = pd.to_datetime(impact_frame["date"], errors="coerce").dt.normalize()

    if dimension == "sentiment_bucket":
        if "news_sentiment_bucket" not in impact_frame.columns:
            return pd.DataFrame()
        dates = impact_frame[impact_frame["news_sentiment_bucket"].astype(str).eq(value_text)]["date"]
    elif dimension == "importance_bucket":
        if "news_importance_bucket" not in impact_frame.columns:
            return pd.DataFrame()
        dates = impact_frame[impact_frame["news_importance_bucket"].astype(str).eq(value_text)]["date"]
    elif dimension == "event":
        if value_text == "news_shock":
            shock_columns = [column for column in ("negative_news_shock", "positive_news_shock") if column in impact_frame.columns]
            if not shock_columns:
                return pd.DataFrame()
            mask = pd.Series(False, index=impact_frame.index)
            for column in shock_columns:
                mask = mask | (pd.to_numeric(impact_frame[column], errors="coerce").fillna(0.0) > 0)
            dates = impact_frame[mask]["date"]
        elif value_text in EVENT_COLUMNS and value_text in impact_frame.columns:
            mask = pd.to_numeric(impact_frame[value_text], errors="coerce").fillna(0.0) > 0
            dates = impact_frame[mask]["date"]
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

    date_set = set(pd.to_datetime(dates, errors="coerce").dropna().dt.normalize())
    return news[news["date"].isin(date_set)].copy()


def _summarize_group(
    frame: pd.DataFrame,
    column: str,
    *,
    config: NewsImpactConfig,
) -> pd.DataFrame:
    if frame.empty or column not in frame.columns:
        return pd.DataFrame()
    rows = []
    for value, subset in frame.groupby(column, dropna=False, sort=False):
        row = _summary_row(subset, config=config)
        row["group"] = str(value)
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_row(subset: pd.DataFrame, *, config: NewsImpactConfig) -> dict[str, object]:
    if subset.empty:
        return {
            "count": 0,
            "average_return_1d": None,
            "average_return_5d": None,
            "average_return_20d": None,
            "up_rate_1d": None,
            "large_down_rate_20d": None,
            "large_up_rate_20d": None,
            "average_crash_risk": None,
            "average_surge_risk": None,
            "average_crash_risk_change_1d": None,
            "average_surge_risk_change_1d": None,
        }
    return {
        "count": int(len(subset)),
        "average_return_1d": _mean(subset.get("return_1d")),
        "average_return_5d": _mean(subset.get("return_5d")),
        "average_return_20d": _mean(subset.get("return_20d")),
        "up_rate_1d": _rate(pd.to_numeric(subset.get("return_1d"), errors="coerce") > 0),
        "large_down_rate_20d": _rate(
            pd.to_numeric(subset.get("future_min_return_20d"), errors="coerce")
            <= config.large_down_threshold
        ),
        "large_up_rate_20d": _rate(
            pd.to_numeric(subset.get("future_max_return_20d"), errors="coerce")
            >= config.large_up_threshold
        ),
        "average_crash_risk": _mean(subset.get("crash_risk_score")),
        "average_surge_risk": _mean(subset.get("surge_risk_score")),
        "average_crash_risk_change_1d": _mean(subset.get("crash_risk_change_1d")),
        "average_surge_risk_change_1d": _mean(subset.get("surge_risk_change_1d")),
    }


def _profile_news_impact_row(
    profile_name: str,
    subset: pd.DataFrame,
    meta: dict[str, object],
    quality: pd.DataFrame,
    recommendations: pd.DataFrame,
    *,
    config: NewsImpactConfig,
) -> dict[str, object]:
    quality_row = _profile_quality_row(quality, profile_name)
    recommendation_row = _profile_recommendation_row(recommendations, profile_name)
    enabled = _profile_enabled(meta, quality_row)
    row = {
        "query_profile": profile_name,
        "group": _profile_value(meta, quality_row, "group", "untracked"),
        "enabled_for_daily": enabled,
        "language": _profile_value(meta, quality_row, "language", ""),
        "region": _profile_value(meta, quality_row, "region", ""),
        "news_count": int(len(subset)),
        "average_sentiment_score": _mean(subset.get("sentiment_score")),
        "average_importance_score": _mean(subset.get("importance_score")),
        "average_source_credibility_score": _mean(subset.get("source_credibility_score")),
        "average_source_market_impact_weight": _mean(subset.get("source_market_impact_weight")),
        "average_return_1d": _mean(subset.get("return_1d")),
        "average_return_5d": _mean(subset.get("return_5d")),
        "average_return_20d": _mean(subset.get("return_20d")),
        "average_max_up_20d": _mean(subset.get("max_up_20d")),
        "average_max_down_20d": _mean(subset.get("max_down_20d")),
        "large_up_rate": _rate(
            pd.to_numeric(subset.get("max_up_20d"), errors="coerce") >= config.large_up_threshold
        ),
        "large_down_rate": _rate(
            pd.to_numeric(subset.get("max_down_20d"), errors="coerce") <= config.large_down_threshold
        ),
        "average_crash_risk": _mean(subset.get("crash_risk_score")),
        "average_surge_risk": _mean(subset.get("surge_risk_score")),
        "total_fetched_rows": _quality_number(quality_row, "history_fetched_count"),
        "total_saved_rows": _quality_number(quality_row, "saved_rows"),
        "latest_fetched_at": _quality_text(quality_row, "latest_fetched_at"),
        "latest_fetched_count": _quality_number(quality_row, "latest_fetched_count"),
        "latest_added_count": _quality_number(quality_row, "latest_added_count"),
        "latest_duplicate_count": _quality_number(quality_row, "latest_duplicate_count"),
        "duplicate_count": _quality_number(quality_row, "history_duplicate_count"),
        "duplicate_rate": _quality_number(quality_row, "duplicate_rate"),
        "added_count": _quality_number(quality_row, "history_added_count"),
        "quality_status": _quality_text(quality_row, "quality_status") or "unknown",
        "recommendation": _quality_text(recommendation_row, "recommendation") or _fallback_recommendation(enabled),
    }
    row["daily_decision_note"] = _profile_daily_decision_note(row)
    return row


def _profile_quality_row(frame: pd.DataFrame, profile_name: str) -> pd.Series | None:
    if frame.empty or "profile" not in frame.columns:
        return None
    rows = frame[frame["profile"].astype(str).eq(profile_name)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _profile_recommendation_row(frame: pd.DataFrame, profile_name: str) -> pd.Series | None:
    if frame.empty or "profile" not in frame.columns:
        return None
    rows = frame[frame["profile"].astype(str).eq(profile_name)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _profile_value(
    meta: dict[str, object],
    quality_row: pd.Series | None,
    key: str,
    default: object,
) -> object:
    value = meta.get(key)
    if value not in (None, ""):
        return value
    if quality_row is not None:
        quality_value = quality_row.get(key)
        if quality_value not in (None, ""):
            return quality_value
    return default


def _profile_enabled(meta: dict[str, object], quality_row: pd.Series | None) -> bool:
    if "enabled_for_daily" in meta:
        return _as_bool(meta.get("enabled_for_daily"))
    if quality_row is not None:
        return _as_bool(quality_row.get("enabled_for_daily"))
    return False


def _quality_text(row: pd.Series | None, key: str) -> str:
    if row is None:
        return ""
    value = row.get(key)
    return "" if value is None else str(value)


def _quality_number(row: pd.Series | None, key: str) -> float | None:
    if row is None:
        return None
    return safe_float(row.get(key))


def _fallback_recommendation(enabled_for_daily: bool) -> str:
    return "daily_keep" if enabled_for_daily else "research_only"


def _profile_daily_decision_note(row: dict[str, object]) -> str:
    profile = str(row.get("query_profile") or "-")
    count = int(safe_float(row.get("news_count")) or 0)
    recommendation = str(row.get("recommendation") or "")
    quality_status = str(row.get("quality_status") or "")
    duplicate_rate = safe_float(row.get("duplicate_rate"))
    average_5d = safe_float(row.get("average_return_5d"))
    enabled = _as_bool(row.get("enabled_for_daily"))
    if count < 3:
        role = "daily" if enabled else "research"
        return (
            f"{profile}: only {count} saved rows, so treat this as a {role} research candidate "
            "rather than a stable conclusion."
        )
    if recommendation == "review_needed":
        duplicate_text = f" duplicate rate {_format_percent(duplicate_rate)}" if duplicate_rate is not None else ""
        return f"{profile}: review_needed because quality is {quality_status}.{duplicate_text}"
    if enabled:
        return (
            f"{profile}: daily_keep candidate with {count} saved rows and average 5D return "
            f"{_format_percent(average_5d)} after its news dates."
        )
    return (
        f"{profile}: research_only candidate with {count} saved rows and average 5D return "
        f"{_format_percent(average_5d)} after its news dates."
    )


def _daily_news_profile_recommendation_row(row: pd.Series) -> dict[str, object]:
    profile = str(row.get("query_profile") or "")
    group = str(row.get("group") or "")
    enabled = _as_bool(row.get("enabled_for_daily"))
    news_count = int(safe_float(row.get("news_count")) or 0)
    total_saved = int(safe_float(row.get("total_saved_rows")) or 0)
    added_count = int(safe_float(row.get("added_count")) or 0)
    duplicate_count = int(safe_float(row.get("duplicate_count")) or 0)
    duplicate_rate = safe_float(row.get("duplicate_rate"))
    fetched_total = int(safe_float(row.get("total_fetched_rows")) or 0)
    quality_status = str(row.get("quality_status") or "unknown")
    current_recommendation = str(row.get("recommendation") or "")
    average_return_5d = safe_float(row.get("average_return_5d"))
    average_return_20d = safe_float(row.get("average_return_20d"))
    average_max_up_20d = safe_float(row.get("average_max_up_20d"))
    average_max_down_20d = safe_float(row.get("average_max_down_20d"))
    large_up_rate = safe_float(row.get("large_up_rate"))
    large_down_rate = safe_float(row.get("large_down_rate"))
    average_importance = safe_float(row.get("average_importance_score"))
    average_credibility = safe_float(row.get("average_source_credibility_score"))

    label = "daily_keep" if enabled else "research_only"
    priority = "Low"
    suggested_enabled = enabled

    has_useful_impact = (
        (average_importance is not None and average_importance >= 0.65)
        or (average_credibility is not None and average_credibility >= 0.70)
        or (average_return_5d is not None and abs(average_return_5d) >= 0.02)
        or (large_up_rate is not None and large_up_rate >= 0.20)
        or (large_down_rate is not None and large_down_rate >= 0.20)
    )
    high_duplicate = duplicate_rate is not None and duplicate_rate >= 0.85
    no_recent_additions = fetched_total >= 5 and added_count == 0
    hard_quality_issue = quality_status.upper() == "FAIL"
    untracked_profile = profile == "(untracked)" or group == "untracked"

    if untracked_profile:
        label = "review_needed"
        priority = "Medium"
        suggested_enabled = False
    elif hard_quality_issue:
        label = "review_needed"
        priority = "High"
        suggested_enabled = enabled
    elif enabled and total_saved == 0 and added_count == 0:
        label = "disable_candidate"
        priority = "High"
        suggested_enabled = False
    elif enabled and (high_duplicate or no_recent_additions):
        label = "reduce_limit"
        priority = "Medium"
        suggested_enabled = True
    elif enabled and news_count < 3:
        label = "reduce_limit"
        priority = "Medium"
        suggested_enabled = True
    elif enabled:
        label = "daily_keep"
        priority = "Low"
        suggested_enabled = True
    elif group == "research":
        label = "research_only"
        priority = "Low"
        suggested_enabled = False
    elif news_count >= 3 and has_useful_impact:
        label = "daily_candidate"
        priority = "Medium"
        suggested_enabled = True
    else:
        label = "research_only"
        priority = "Low"
        suggested_enabled = False

    reason = _daily_news_profile_recommendation_reason(
        profile=profile,
        label=label,
        enabled=enabled,
        group=group,
        news_count=news_count,
        added_count=added_count,
        duplicate_count=duplicate_count,
        duplicate_rate=duplicate_rate,
        quality_status=quality_status,
        current_recommendation=current_recommendation,
        average_return_5d=average_return_5d,
        average_max_down_20d=average_max_down_20d,
        average_importance=average_importance,
    )

    return {
        "query_profile": profile,
        "group": group,
        "enabled_for_daily": enabled,
        "language": row.get("language"),
        "region": row.get("region"),
        "total_saved_rows": total_saved,
        "latest_fetched_at": row.get("latest_fetched_at"),
        "added_count": added_count,
        "duplicate_count": duplicate_count,
        "duplicate_rate": duplicate_rate,
        "quality_status": quality_status,
        "current_recommendation": current_recommendation,
        "news_count": news_count,
        "average_return_5d": average_return_5d,
        "average_return_20d": average_return_20d,
        "average_max_up_20d": average_max_up_20d,
        "average_max_down_20d": average_max_down_20d,
        "large_up_rate": large_up_rate,
        "large_down_rate": large_down_rate,
        "average_crash_risk": safe_float(row.get("average_crash_risk")),
        "average_surge_risk": safe_float(row.get("average_surge_risk")),
        "average_importance_score": average_importance,
        "average_source_credibility_score": average_credibility,
        "daily_operation_recommendation": label,
        "priority": priority,
        "suggested_enabled_for_daily": suggested_enabled,
        "config_diff": _daily_news_profile_config_diff(
            enabled,
            suggested_enabled,
            label,
        ),
        "recommendation_reason": reason,
    }


def _daily_news_profile_recommendation_reason(
    *,
    profile: str,
    label: str,
    enabled: bool,
    group: str,
    news_count: int,
    added_count: int,
    duplicate_count: int,
    duplicate_rate: float | None,
    quality_status: str,
    current_recommendation: str,
    average_return_5d: float | None,
    average_max_down_20d: float | None,
    average_importance: float | None,
) -> str:
    sample_note = (
        " Samples are still limited, so treat this as a provisional operating hint."
        if news_count < 5
        else ""
    )
    duplicate_text = _format_percent(duplicate_rate)
    average_5d_text = _format_percent(average_return_5d)
    drawdown_text = _format_percent(average_max_down_20d)
    importance_text = _format_number(average_importance, digits=2)
    if label == "daily_keep":
        return (
            f"{profile} is adding usable daily-context news with {news_count} saved rows, "
            f"duplicate rate {duplicate_text}, and average 5D post-news return {average_5d_text}."
            f"{sample_note}"
        )
    if label == "daily_candidate":
        return (
            f"{profile} is currently research-side but has {news_count} saved rows and "
            f"impact/importance signals worth testing in daily operation "
            f"(importance {importance_text}, avg 5D {average_5d_text}).{sample_note}"
        )
    if label == "reduce_limit":
        return (
            f"{profile} is daily-enabled, but recent fetches are duplicate-heavy or sparse "
            f"(added {added_count}, duplicates {duplicate_count}, duplicate rate {duplicate_text}). "
            "Keep it only with a lower NewsLimit or review cadence."
            f"{sample_note}"
        )
    if label == "research_only":
        return (
            f"{profile} fits research/manual use for now "
            f"(group {group or '-'}, {news_count} saved rows, avg 5D {average_5d_text})."
            f"{sample_note}"
        )
    if label == "disable_candidate":
        return (
            f"{profile} is daily-enabled but has no saved usable rows yet. "
            "Consider moving it out of daily operation until it adds new articles."
        )
    if label == "review_needed":
        return (
            f"{profile} needs review before changing daily settings "
            f"(quality {quality_status}, current recommendation {current_recommendation or '-'}, "
            f"avg max down 20D {drawdown_text}).{sample_note}"
        )
    return f"{profile} has no strong daily-operation signal yet.{sample_note}"


def _daily_news_profile_config_diff(
    enabled: bool,
    suggested_enabled: bool,
    label: str,
) -> str:
    if label == "reduce_limit":
        return "Keep enabled_for_daily=true; consider lowering NewsLimit."
    if label == "review_needed":
        return "Review data quality before changing enabled_for_daily."
    if enabled and not suggested_enabled:
        return "enabled_for_daily true -> false recommended."
    if not enabled and suggested_enabled:
        return "enabled_for_daily false -> true candidate."
    if label == "research_only":
        return "No change; keep research/manual."
    return "No config change suggested."


def _news_limit_recommendation_row(row: pd.Series) -> dict[str, object]:
    profile = str(row.get("query_profile") or "")
    enabled = _as_bool(row.get("enabled_for_daily"))
    group = str(row.get("group") or "")
    daily_label = str(row.get("daily_operation_recommendation") or "")
    latest_fetched = int(safe_float(row.get("latest_fetched_count")) or 0)
    latest_added = int(safe_float(row.get("latest_added_count")) or 0)
    latest_duplicates = int(safe_float(row.get("latest_duplicate_count")) or 0)
    if latest_fetched == 0:
        latest_fetched = int(safe_float(row.get("added_count")) or 0) + int(
            safe_float(row.get("duplicate_count")) or 0
        )
        latest_added = int(safe_float(row.get("added_count")) or 0)
        latest_duplicates = int(safe_float(row.get("duplicate_count")) or 0)
    duplicate_rate = safe_float(row.get("duplicate_rate"))
    saved_rows = int(safe_float(row.get("total_saved_rows")) or 0)
    quality_status = str(row.get("quality_status") or "unknown")
    average_importance = safe_float(row.get("average_importance_score"))

    if not enabled or daily_label == "research_only":
        daily_limit: int | str = "daily_off"
        research_limit: int | str = 10
        reason = (
            f"{profile} is better suited to manual research for now. "
            "Use limit 10 when you intentionally want broader coverage."
        )
    elif daily_label in {"disable_candidate", "review_needed"} and saved_rows == 0:
        daily_limit = "monitor_more"
        research_limit = 10
        reason = (
            f"{profile} needs more usable fetch history before a daily NewsLimit is chosen."
        )
    elif daily_label == "reduce_limit" or (
        duplicate_rate is not None and duplicate_rate >= 0.80 and latest_added <= 1
    ):
        daily_limit = 3
        research_limit = 10
        reason = (
            f"{profile} is duplicate-heavy or adds few new rows, so limit 3 is the safest daily option."
        )
    elif latest_added >= 2:
        daily_limit = 5
        research_limit = 10
        reason = (
            f"{profile} is adding news, so limit 5 is a balanced daily option."
        )
    elif average_importance is not None and average_importance >= 0.75:
        daily_limit = 3
        research_limit = 10
        reason = (
            f"{profile} looks important but low-frequency; use limit 3 daily and limit 10 manually."
        )
    else:
        daily_limit = "monitor_more"
        research_limit = 10
        reason = (
            f"{profile} has limited history, so keep monitoring before increasing daily fetch volume."
        )

    return {
        "query_profile": profile,
        "enabled_for_daily": enabled,
        "group": group,
        "latest_fetched_count": latest_fetched,
        "latest_added_count": latest_added,
        "latest_duplicate_count": latest_duplicates,
        "duplicate_rate": duplicate_rate,
        "total_saved_rows": saved_rows,
        "quality_status": quality_status,
        "daily_operation_recommendation": daily_label,
        "limit_3_assessment": _news_limit_assessment(3, latest_added, duplicate_rate, enabled, daily_label),
        "limit_5_assessment": _news_limit_assessment(5, latest_added, duplicate_rate, enabled, daily_label),
        "limit_10_assessment": _news_limit_assessment(10, latest_added, duplicate_rate, enabled, daily_label),
        "suggested_daily_news_limit": daily_limit,
        "suggested_research_news_limit": research_limit,
        "reason": reason,
    }


def _news_limit_assessment(
    limit: int,
    latest_added: int,
    duplicate_rate: float | None,
    enabled_for_daily: bool,
    daily_label: str,
) -> str:
    if not enabled_for_daily or daily_label == "research_only":
        return "research/manual"
    if daily_label in {"disable_candidate", "review_needed"}:
        return "review first"
    if limit == 3:
        return "light daily"
    if limit == 5:
        if duplicate_rate is not None and duplicate_rate >= 0.85 and latest_added <= 1:
            return "may still duplicate"
        return "standard daily"
    if duplicate_rate is not None and duplicate_rate >= 0.70:
        return "research only; duplicate-heavy"
    return "broad research"


def _profile_availability_frame(news_dir: str | Path) -> pd.DataFrame:
    rows = news_query_profile_rows()
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["count"] = 0
    result["tracking_status"] = "query_profile tracking not available yet"
    result["news_dir"] = str(news_dir)
    return result


def _event_row(events: pd.DataFrame, event: str) -> pd.Series | None:
    if events.empty or "event" not in events.columns:
        return None
    rows = events[events["event"] == event]
    if rows.empty:
        return None
    return rows.iloc[0]


def _group_row(frame: pd.DataFrame, group: str) -> pd.Series | None:
    if frame.empty or "group" not in frame.columns:
        return None
    rows = frame[frame["group"] == group]
    if rows.empty:
        return None
    return rows.iloc[0]


def _event_message(event: str, row: pd.Series, *, min_count: int) -> str:
    count = int(row.get("count") or 0)
    average_5d = safe_float(row.get("average_return_5d"))
    large_down = safe_float(row.get("large_down_rate_20d"))
    large_up = safe_float(row.get("large_up_rate_20d"))
    label = event.replace("_", " ")
    if count < min_count:
        return f"{label} has only {count} samples; treat this read as limited."
    return (
        f"{label} has {count} samples; average 5D return is {_format_percent(average_5d)}, "
        f"large-down rate is {_format_percent(large_down)}, and large-up rate is {_format_percent(large_up)}."
    )


def _current_event_text(label: str, row: pd.Series | None) -> str:
    if row is None:
        return f"Current {label} is active, but validation samples are unavailable."
    count = int(row.get("count") or 0)
    average_5d = safe_float(row.get("average_return_5d"))
    if count < 5:
        return f"Current {label} is active, but validation has only {count} samples."
    return (
        f"Historical {label} samples: {count}; average 5D return "
        f"{_format_percent(average_5d)}."
    )


def _mean(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _rate(mask) -> float | None:
    if mask is None:
        return None
    clean = pd.Series(mask).dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _dedupe_ordered(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _return_from_close(base_close: float, target_close: float | None) -> float | None:
    if target_close is None or base_close == 0:
        return None
    return float(target_close / base_close - 1.0)


def _news_quality_note(row: pd.Series) -> str:
    notes: list[str] = []
    title = str(row.get("title") or "").strip()
    summary = str(row.get("summary") or "").strip()
    if not title:
        notes.append("missing title")
    if is_mojibake_suspect(f"{title} {summary}"):
        notes.append("mojibake suspect")
    return ", ".join(notes)


def _format_percent(value) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric * 100:.2f}%"


def _format_number(value, *, digits: int = 1) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return "-"
    return f"{numeric:.{digits}f}"
