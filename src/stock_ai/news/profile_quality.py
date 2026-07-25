from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_ai.news.manual_csv import NEWS_DISPLAY_COLUMNS, news_path_candidates
from stock_ai.news.quality import is_mojibake_suspect
from stock_ai.news.query_profiles import NewsQueryProfile, load_news_query_profiles
from stock_ai.utils.safe_cast import safe_float


FETCH_HISTORY_COLUMNS = (
    "fetched_at",
    "profile",
    "query",
    "group",
    "enabled_for_daily",
    "language",
    "region",
    "fetched_count",
    "added_count",
    "duplicate_count",
    "unknown_source_count",
    "unknown_sources",
    "check_news_status",
    "check_news_message",
    "warning",
    "error",
)
NEWS_QUALITY_BATCH_COLUMNS = (
    "fetched_at",
    "query_profile",
    "fetched_count",
    "added_count",
    "duplicate_count",
    "check_news_status",
    "check_news_message",
    "unknown_source_count",
    "unknown_sources",
)

PROFILE_QUALITY_COLUMNS = (
    "profile",
    "group",
    "enabled_for_daily",
    "language",
    "region",
    "saved_rows",
    "history_fetched_count",
    "history_added_count",
    "history_duplicate_count",
    "latest_fetched_count",
    "latest_added_count",
    "latest_duplicate_count",
    "latest_news_date",
    "latest_fetched_at",
    "duplicate_count",
    "duplicate_rate",
    "unique_title_count",
    "unique_url_count",
    "average_sentiment_score",
    "average_importance_score",
    "average_source_credibility_score",
    "average_source_market_impact_weight",
    "category_distribution",
    "source_type_distribution",
    "mojibake_suspect_count",
    "invalid_date_count",
    "unknown_source_count",
    "quality_status",
    "quality_message",
)

RECOMMENDATION_COLUMNS = (
    "profile",
    "group",
    "enabled_for_daily",
    "recommendation",
    "priority",
    "reason",
)
NEWSLIMIT3_DAILY_COLUMNS = (
    "date",
    "fetched",
    "added",
    "duplicates",
    "duplicate_rate",
    "profiles_count",
    "unknown_source_count",
)
NEWSLIMIT3_PROFILE_COLUMNS = (
    "profile",
    "group",
    "enabled_for_daily",
    "runs",
    "total_fetched",
    "total_added",
    "total_duplicates",
    "duplicate_rate",
    "added_per_run",
    "latest_fetched_at",
    "daily_operation_recommendation",
    "reason",
)
NEWSLIMIT3_REPORT_COLUMNS = (
    "profile",
    "group",
    "enabled_for_daily",
    "runs",
    "total_fetched",
    "total_added",
    "total_duplicates",
    "duplicate_rate",
    "added_per_run",
    "latest_fetched_at",
    "daily_operation_recommendation",
    "reason",
    "overall_recommendation",
    "overall_reason",
    "observation_days",
)
DAILY_PROFILE_CONFIG_REVIEW_COLUMNS = (
    "profile",
    "group",
    "observed_enabled_for_daily",
    "current_enabled_for_daily",
    "current_decision",
    "total_fetched",
    "total_added",
    "duplicate_rate",
    "recommendation",
    "suggested_enabled_for_daily",
    "config_change",
    "reason",
)
DAILY_PROFILE_CHANGE_SUMMARY_COLUMNS = (
    "period",
    "start_date",
    "end_date",
    "observation_days",
    "daily_profile_count",
    "theoretical_fetched_per_day",
    "total_fetched",
    "total_added",
    "total_duplicates",
    "duplicate_rate",
    "added_per_day",
    "added_per_profile_run",
    "unknown_source_count",
    "unknown_source_occurrences",
    "unknown_source_observed_batches",
    "fetch_batch_count",
    "unknown_source_unrecorded_batches",
    "latest_unknown_sources",
    "check_news_status",
    "check_news_ok_count",
    "check_news_warn_count",
    "check_news_error_count",
    "check_news_unrecorded_count",
)
DAILY_PROFILE_AFTER_COLUMNS = (
    "profile",
    "group",
    "enabled_for_daily",
    "run_count",
    "total_fetched",
    "total_added",
    "total_duplicates",
    "duplicate_rate",
    "added_per_run",
    "latest_fetched_at",
    "unknown_source_count",
    "unknown_source_occurrences",
    "unknown_source_observed_batches",
    "unknown_source_unrecorded_batches",
    "check_news_ok_count",
    "check_news_warn_count",
    "check_news_error_count",
    "recommendation",
    "reason",
)


def default_fetch_history_path(news_dir: str | Path) -> Path:
    return Path(news_dir) / "fetch_history.csv"


def append_fetch_history(
    news_dir: str | Path,
    *,
    profile: NewsQueryProfile | None,
    query: str,
    fetched_count: int = 0,
    added_count: int = 0,
    duplicate_count: int = 0,
    warning: str = "",
    error: str = "",
) -> Path:
    path = default_fetch_history_path(news_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "fetched_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "profile": profile.name if profile is not None else "",
        "query": query,
        "group": profile.group if profile is not None else "manual",
        "enabled_for_daily": bool(profile.enabled_for_daily) if profile is not None else False,
        "language": profile.language if profile is not None else "",
        "region": profile.region if profile is not None else "",
        "fetched_count": int(fetched_count),
        "added_count": int(added_count),
        "duplicate_count": int(duplicate_count),
        "unknown_source_count": "",
        "unknown_sources": "",
        "check_news_status": "",
        "check_news_message": "",
        "warning": warning,
        "error": error,
    }
    existing = read_fetch_history(path)
    updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    updated = _normalize_fetch_history(updated)
    updated.to_csv(path, index=False)
    return path


def read_fetch_history(path_or_news_dir: str | Path) -> pd.DataFrame:
    path = Path(path_or_news_dir)
    if path.is_dir():
        path = default_fetch_history_path(path)
    if not path.exists():
        return pd.DataFrame(columns=FETCH_HISTORY_COLUMNS)
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=FETCH_HISTORY_COLUMNS)
    return _normalize_fetch_history(frame)


def update_latest_fetch_history_quality(
    news_dir: str | Path,
    quality_report: dict[str, object],
) -> Path | None:
    path = default_fetch_history_path(news_dir)
    history = read_fetch_history(path)
    if history.empty:
        return None
    batch_mask = _latest_fetch_batch_mask(history)
    if not batch_mask.any():
        return None

    unknown_count = quality_report.get("unknown_source_count")
    unknown_sources = quality_report.get("unknown_sources") or []
    history.loc[batch_mask, "unknown_source_count"] = (
        "" if unknown_count is None else int(unknown_count)
    )
    history.loc[batch_mask, "unknown_sources"] = json.dumps(
        list(unknown_sources),
        ensure_ascii=False,
    )
    history.loc[batch_mask, "check_news_status"] = str(
        quality_report.get("status") or ""
    )
    history.loc[batch_mask, "check_news_message"] = str(
        quality_report.get("message") or ""
    )
    _normalize_fetch_history(history).to_csv(path, index=False)
    return path


def build_profile_quality_frame(
    news_dir: str | Path,
    ticker: str,
    *,
    history_path: str | Path | None = None,
    profiles: list[NewsQueryProfile] | None = None,
) -> pd.DataFrame:
    configured_profiles = profiles if profiles is not None else load_news_query_profiles()
    news = _read_news_quality_items(news_dir, ticker)
    history = read_fetch_history(history_path or default_fetch_history_path(news_dir))
    if news.empty and not configured_profiles:
        return pd.DataFrame(columns=PROFILE_QUALITY_COLUMNS)

    rows: list[dict[str, object]] = []
    configured_names = {profile.name for profile in configured_profiles}
    observed_names = {
        name
        for name in news.get("query_profile", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
        if name
    }
    observed_names.update(
        name
        for name in history.get("profile", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
        if name
    )

    profile_lookup = {profile.name: profile for profile in configured_profiles}
    for profile_name in [profile.name for profile in configured_profiles] + sorted(observed_names - configured_names):
        profile = profile_lookup.get(profile_name)
        news_subset = _news_for_profile(news, profile_name)
        history_subset = _history_for_profile(history, profile_name)
        rows.append(_profile_quality_row(profile_name, profile, news_subset, history_subset))
    return pd.DataFrame(rows, columns=PROFILE_QUALITY_COLUMNS)


def build_profile_recommendations(quality: pd.DataFrame) -> pd.DataFrame:
    if quality.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)
    rows: list[dict[str, object]] = []
    for _, row in quality.iterrows():
        rows.append(_recommendation_row(row))
    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def recent_fetch_history(
    news_dir: str | Path,
    *,
    limit: int = 20,
) -> pd.DataFrame:
    history = read_fetch_history(news_dir)
    if history.empty:
        return history
    history = history.copy()
    history["_fetched_at"] = pd.to_datetime(history["fetched_at"], errors="coerce", format="mixed")
    history = history.sort_values("_fetched_at", ascending=False).drop(columns=["_fetched_at"])
    return history.head(max(1, int(limit))).reset_index(drop=True)


def build_news_quality_history(
    path_or_history: str | Path | pd.DataFrame,
) -> dict[str, object]:
    if isinstance(path_or_history, pd.DataFrame):
        history = _normalize_fetch_history(path_or_history)
    else:
        history = read_fetch_history(path_or_history)
    empty_batches = pd.DataFrame(columns=NEWS_QUALITY_BATCH_COLUMNS)
    empty_summary = {
        "latest_check_news_status": "",
        "latest_check_news_message": "",
        "latest_unknown_source_count": None,
        "latest_unknown_sources": "",
        "observed_batch_count": 0,
        "unrecorded_batch_count": 0,
        "total_batch_count": 0,
        "ok_count": 0,
        "warn_count": 0,
        "error_count": 0,
        "non_ok_count": 0,
    }
    if history.empty:
        return {"summary": empty_summary, "batches": empty_batches}

    rows = history.copy()
    rows["_fetched_at"] = pd.to_datetime(
        rows["fetched_at"],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    rows = rows.dropna(subset=["_fetched_at"]).sort_values("_fetched_at").reset_index(drop=True)
    if rows.empty:
        return {"summary": empty_summary, "batches": empty_batches}

    gaps = rows["_fetched_at"].diff().dt.total_seconds().fillna(float("inf"))
    rows["_quality_batch_id"] = gaps.gt(300).cumsum()
    batch_rows = [
        _news_quality_batch_row(batch)
        for _, batch in rows.groupby("_quality_batch_id", sort=True)
    ]
    batches = pd.DataFrame(batch_rows, columns=NEWS_QUALITY_BATCH_COLUMNS)
    batches = batches.sort_values("fetched_at", ascending=False).reset_index(drop=True)

    observed = pd.to_numeric(batches["unknown_source_count"], errors="coerce").notna()
    statuses = batches["check_news_status"].fillna("").astype(str).str.upper()
    latest = batches.iloc[0]
    latest_unknown_count = pd.to_numeric(
        pd.Series([latest.get("unknown_source_count")]),
        errors="coerce",
    ).iloc[0]
    summary = {
        "latest_check_news_status": str(latest.get("check_news_status") or ""),
        "latest_check_news_message": str(latest.get("check_news_message") or ""),
        "latest_unknown_source_count": (
            None if pd.isna(latest_unknown_count) else int(latest_unknown_count)
        ),
        "latest_unknown_sources": str(latest.get("unknown_sources") or ""),
        "observed_batch_count": int(observed.sum()),
        "unrecorded_batch_count": int((~observed).sum()),
        "total_batch_count": int(len(batches)),
        "ok_count": int(statuses.eq("OK").sum()),
        "warn_count": int(statuses.eq("WARN").sum()),
        "error_count": int(statuses.eq("ERROR").sum()),
        "non_ok_count": int(statuses.isin(("WARN", "ERROR")).sum()),
    }
    return {"summary": summary, "batches": batches}


def build_newslimit3_daily_summary(
    news_dir: str | Path,
    *,
    start_date: str = "2026-06-05",
) -> pd.DataFrame:
    history = _newslimit3_history(news_dir, start_date=start_date)
    if history.empty:
        return pd.DataFrame(columns=NEWSLIMIT3_DAILY_COLUMNS)
    grouped = (
        history.groupby("date", dropna=False)
        .agg(
            fetched=("fetched_count", "sum"),
            added=("added_count", "sum"),
            duplicates=("duplicate_count", "sum"),
            profiles_count=("profile", "nunique"),
        )
        .reset_index()
        .sort_values("date")
    )
    grouped["duplicate_rate"] = grouped.apply(
        lambda row: _safe_ratio(row["duplicates"], row["fetched"]),
        axis=1,
    )
    grouped["unknown_source_count"] = ""
    return grouped[list(NEWSLIMIT3_DAILY_COLUMNS)].reset_index(drop=True)


def build_newslimit3_profile_evaluation(
    news_dir: str | Path,
    *,
    start_date: str = "2026-06-05",
) -> pd.DataFrame:
    history = _newslimit3_history(news_dir, start_date=start_date)
    if history.empty:
        return pd.DataFrame(columns=NEWSLIMIT3_PROFILE_COLUMNS)
    grouped = (
        history.groupby("profile", dropna=False)
        .agg(
            group=("group", "last"),
            enabled_for_daily=("enabled_for_daily", "last"),
            runs=("profile", "size"),
            total_fetched=("fetched_count", "sum"),
            total_added=("added_count", "sum"),
            total_duplicates=("duplicate_count", "sum"),
            latest_fetched_at=("fetched_at_local", "max"),
        )
        .reset_index()
        .sort_values("profile")
    )
    grouped["duplicate_rate"] = grouped.apply(
        lambda row: _safe_ratio(row["total_duplicates"], row["total_fetched"]),
        axis=1,
    )
    grouped["added_per_run"] = grouped.apply(
        lambda row: _safe_ratio(row["total_added"], row["runs"]),
        axis=1,
    )
    recommendations = grouped.apply(_newslimit3_profile_recommendation, axis=1, result_type="expand")
    grouped["daily_operation_recommendation"] = recommendations["recommendation"]
    grouped["reason"] = recommendations["reason"]
    grouped["latest_fetched_at"] = grouped["latest_fetched_at"].map(_format_datetime)
    return grouped[list(NEWSLIMIT3_PROFILE_COLUMNS)].reset_index(drop=True)


def build_newslimit3_evaluation(
    news_dir: str | Path,
    *,
    start_date: str = "2026-06-05",
) -> dict[str, object]:
    daily = build_newslimit3_daily_summary(news_dir, start_date=start_date)
    profiles = build_newslimit3_profile_evaluation(news_dir, start_date=start_date)
    overall = _newslimit3_overall_recommendation(daily, profiles)
    return {
        "start_date": start_date,
        "daily_summary": daily,
        "profile_evaluation": profiles,
        **overall,
    }


def save_newslimit3_evaluation_report(
    evaluation: dict[str, object],
    report_dir: str | Path,
    *,
    generated_date: str | None = None,
) -> Path | None:
    profiles = evaluation.get("profile_evaluation")
    if not isinstance(profiles, pd.DataFrame) or profiles.empty:
        return None
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_text = generated_date or datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"newslimit3_evaluation_{date_text}.csv"
    frame = profiles.copy()
    frame["overall_recommendation"] = evaluation.get("overall_recommendation", "")
    frame["overall_reason"] = evaluation.get("overall_reason", "")
    frame["observation_days"] = evaluation.get("observation_days", 0)
    frame = frame[[column for column in NEWSLIMIT3_REPORT_COLUMNS if column in frame.columns]]
    frame.to_csv(path, index=False)
    return path


def build_daily_profile_config_review(
    news_dir: str | Path,
    *,
    start_date: str = "2026-06-05",
    profiles: list[NewsQueryProfile] | None = None,
) -> pd.DataFrame:
    evaluation = build_newslimit3_profile_evaluation(news_dir, start_date=start_date)
    configured_profiles = profiles if profiles is not None else load_news_query_profiles()
    profile_lookup = {profile.name: profile for profile in configured_profiles}
    if evaluation.empty:
        return pd.DataFrame(columns=DAILY_PROFILE_CONFIG_REVIEW_COLUMNS)

    rows: list[dict[str, object]] = []
    for _, row in evaluation.iterrows():
        profile_name = str(row.get("profile") or "")
        profile = profile_lookup.get(profile_name)
        current_enabled = bool(profile.enabled_for_daily) if profile is not None else bool(row.get("enabled_for_daily"))
        suggested_enabled, recommendation, reason = _daily_profile_config_suggestion(row, current_enabled)
        rows.append(
            {
                "profile": profile_name,
                "group": profile.group if profile is not None else row.get("group", ""),
                "observed_enabled_for_daily": bool(row.get("enabled_for_daily")),
                "current_enabled_for_daily": current_enabled,
                "current_decision": row.get("daily_operation_recommendation", ""),
                "total_fetched": row.get("total_fetched", 0),
                "total_added": row.get("total_added", 0),
                "duplicate_rate": row.get("duplicate_rate"),
                "recommendation": recommendation,
                "suggested_enabled_for_daily": suggested_enabled,
                "config_change": _config_change_text(current_enabled, suggested_enabled),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows, columns=DAILY_PROFILE_CONFIG_REVIEW_COLUMNS)


def save_daily_profile_config_review_report(
    review: pd.DataFrame,
    report_dir: str | Path,
    *,
    generated_date: str | None = None,
) -> Path | None:
    if review.empty:
        return None
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_text = generated_date or datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"daily_profile_config_review_{date_text}.csv"
    review.to_csv(path, index=False)
    return path


def build_daily_profile_before_after_evaluation(
    news_dir: str | Path,
    *,
    start_date: str = "2026-06-05",
    removed_profile: str = "toyota_earnings",
    profiles: list[NewsQueryProfile] | None = None,
) -> dict[str, object]:
    configured_profiles = profiles if profiles is not None else load_news_query_profiles()
    current_daily_profiles = {
        profile.name for profile in configured_profiles if profile.enabled_for_daily
    }
    history = _newslimit3_history(news_dir, start_date=start_date)
    before, after, boundary = _split_daily_profile_history(
        history,
        removed_profile=removed_profile,
        current_daily_profiles=current_daily_profiles,
    )
    before_summary = _profile_change_period_summary(before, period="before")
    after_summary = _profile_change_period_summary(after, period="after")
    summary = pd.DataFrame(
        [before_summary, after_summary],
        columns=DAILY_PROFILE_CHANGE_SUMMARY_COLUMNS,
    )
    profile_evaluation = _profile_change_after_evaluation(
        after,
        configured_profiles=configured_profiles,
    )
    decision = _profile_change_overall_recommendation(
        before_summary,
        after_summary,
        profile_evaluation,
        removed_profile=removed_profile,
    )
    return {
        "start_date": start_date,
        "removed_profile": removed_profile,
        "change_boundary": boundary,
        "summary": summary,
        "profile_evaluation": profile_evaluation,
        **decision,
        "caveats": (
            "This is an operational data-quality evaluation, not proof of market predictive power. "
            "Older fetch batches without saved check-news fields remain not available and are not treated as zero."
        ),
    }


def save_daily_profile_before_after_report(
    evaluation: dict[str, object],
    report_dir: str | Path,
    *,
    generated_date: str | None = None,
) -> Path | None:
    summary = evaluation.get("summary")
    profiles = evaluation.get("profile_evaluation")
    if not isinstance(summary, pd.DataFrame) or summary.empty:
        return None
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    date_text = generated_date or datetime.now().strftime("%Y-%m-%d")
    path = output_dir / f"daily_profile_before_after_{date_text}.csv"

    summary_rows = summary.copy()
    summary_rows.insert(0, "row_type", "period_summary")
    profile_rows = profiles.copy() if isinstance(profiles, pd.DataFrame) else pd.DataFrame()
    if not profile_rows.empty:
        profile_rows.insert(0, "row_type", "after_profile")
    combined = pd.concat([summary_rows, profile_rows], ignore_index=True, sort=False)
    combined["overall_recommendation"] = evaluation.get("overall_recommendation", "")
    combined["overall_reason"] = evaluation.get("overall_reason", "")
    combined["removed_profile_effect"] = evaluation.get("removed_profile_effect", "")
    combined["change_boundary"] = evaluation.get("change_boundary", "")
    combined.to_csv(path, index=False)
    return path


def _profile_quality_row(
    profile_name: str,
    profile: NewsQueryProfile | None,
    news_subset: pd.DataFrame,
    history_subset: pd.DataFrame,
) -> dict[str, object]:
    dates = pd.to_datetime(news_subset.get("date", pd.Series(dtype="object")), errors="coerce", format="mixed")
    fetched_at = pd.to_datetime(
        news_subset.get("fetched_at", pd.Series(dtype="object")),
        errors="coerce",
        format="mixed",
    )
    titles = news_subset.get("title", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    urls = news_subset.get("url", pd.Series(dtype="object")).fillna("").astype(str).str.strip()
    profile_display = profile_name or "(untracked)"
    fetched_total = _sum_numeric(history_subset.get("fetched_count"))
    added_total = _sum_numeric(history_subset.get("added_count"))
    duplicate_total = _sum_numeric(history_subset.get("duplicate_count"))
    duplicate_rate = duplicate_total / fetched_total if fetched_total > 0 else None
    latest_history = _latest_history_row(history_subset)
    combined_text = (
        titles
        + " "
        + news_subset.get("summary", pd.Series(index=news_subset.index, dtype="object")).fillna("").astype(str)
    )
    invalid_dates = int(dates.isna().sum()) if not news_subset.empty else 0
    unknown_sources = _unknown_source_count(news_subset)
    mojibake_count = int(combined_text.map(is_mojibake_suspect).sum()) if not news_subset.empty else 0
    quality_status, quality_message = _quality_status(
        duplicate_rate=duplicate_rate,
        added_total=added_total,
        fetched_total=fetched_total,
        mojibake_count=mojibake_count,
        invalid_dates=invalid_dates,
        unknown_sources=unknown_sources,
        latest_news_date=dates.max() if not dates.empty else pd.NaT,
        enabled_for_daily=bool(profile.enabled_for_daily) if profile else False,
    )
    return {
        "profile": profile_display,
        "group": profile.group if profile else "untracked",
        "enabled_for_daily": bool(profile.enabled_for_daily) if profile else False,
        "language": profile.language if profile else "",
        "region": profile.region if profile else "",
        "saved_rows": int(len(news_subset)),
        "history_fetched_count": fetched_total,
        "history_added_count": added_total,
        "history_duplicate_count": duplicate_total,
        "latest_fetched_count": _history_count(latest_history, "fetched_count"),
        "latest_added_count": _history_count(latest_history, "added_count"),
        "latest_duplicate_count": _history_count(latest_history, "duplicate_count"),
        "latest_news_date": _format_date(dates.max() if not dates.empty else pd.NaT),
        "latest_fetched_at": _format_datetime(fetched_at.max() if not fetched_at.empty else pd.NaT),
        "duplicate_count": duplicate_total,
        "duplicate_rate": duplicate_rate,
        "unique_title_count": int(titles[titles.ne("")].nunique()),
        "unique_url_count": int(urls[urls.ne("")].nunique()),
        "average_sentiment_score": _mean(news_subset.get("sentiment_score")),
        "average_importance_score": _mean(news_subset.get("importance_score")),
        "average_source_credibility_score": _mean(news_subset.get("source_credibility_score")),
        "average_source_market_impact_weight": _mean(news_subset.get("source_market_impact_weight")),
        "category_distribution": _distribution_text(news_subset.get("category")),
        "source_type_distribution": _distribution_text(news_subset.get("source_type")),
        "mojibake_suspect_count": mojibake_count,
        "invalid_date_count": invalid_dates,
        "unknown_source_count": unknown_sources,
        "quality_status": quality_status,
        "quality_message": quality_message,
    }


def _newslimit3_history(news_dir: str | Path, *, start_date: str) -> pd.DataFrame:
    history = read_fetch_history(news_dir)
    if history.empty:
        return history
    result = history.copy()
    fetched_at = pd.to_datetime(result["fetched_at"], errors="coerce", format="mixed", utc=True)
    result = result[fetched_at.notna()].copy()
    if result.empty:
        return result
    result["fetched_at_local"] = fetched_at[fetched_at.notna()].dt.tz_convert("Asia/Tokyo")
    result["date"] = result["fetched_at_local"].dt.strftime("%Y-%m-%d")
    start = pd.Timestamp(start_date).strftime("%Y-%m-%d")
    result = result[result["date"].ge(start)].copy()
    result = result[result["enabled_for_daily"].astype(bool)].copy()
    result = result[result["fetched_count"].eq(3)].copy()
    result["profile"] = result["profile"].fillna("").astype(str).str.strip()
    result = result[result["profile"].ne("")]
    return result.reset_index(drop=True)


def _split_daily_profile_history(
    history: pd.DataFrame,
    *,
    removed_profile: str,
    current_daily_profiles: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if history.empty:
        return history.copy(), history.copy(), ""
    rows = history.copy().sort_values("fetched_at_local").reset_index(drop=True)
    gaps = rows["fetched_at_local"].diff().dt.total_seconds().fillna(float("inf"))
    rows["_batch_id"] = gaps.gt(300).cumsum()

    removed_batches = rows.loc[rows["profile"].eq(removed_profile), "_batch_id"]
    if removed_batches.empty:
        return rows.copy(), rows.iloc[0:0].copy(), ""

    last_before_batch = int(removed_batches.max())
    before = rows[rows["_batch_id"].le(last_before_batch)].copy()
    after = rows[rows["_batch_id"].gt(last_before_batch)].copy()
    if current_daily_profiles:
        after = after[after["profile"].isin(current_daily_profiles)].copy()
    boundary_value = after["fetched_at_local"].min() if not after.empty else pd.NaT
    boundary = _format_datetime(boundary_value)
    return (
        before.drop(columns=["_batch_id"], errors="ignore").reset_index(drop=True),
        after.drop(columns=["_batch_id"], errors="ignore").reset_index(drop=True),
        boundary,
    )


def _profile_change_period_summary(frame: pd.DataFrame, *, period: str) -> dict[str, object]:
    if frame.empty:
        return {
            "period": period,
            "start_date": "",
            "end_date": "",
            "observation_days": 0,
            "daily_profile_count": 0,
            "theoretical_fetched_per_day": 0,
            "total_fetched": 0,
            "total_added": 0,
            "total_duplicates": 0,
            "duplicate_rate": None,
            "added_per_day": None,
            "added_per_profile_run": None,
            "unknown_source_count": None,
            "unknown_source_occurrences": None,
            "unknown_source_observed_batches": 0,
            "fetch_batch_count": 0,
            "unknown_source_unrecorded_batches": 0,
            "latest_unknown_sources": "",
            "check_news_status": "",
            "check_news_ok_count": 0,
            "check_news_warn_count": 0,
            "check_news_error_count": 0,
            "check_news_unrecorded_count": 0,
        }
    total_fetched = _sum_numeric(frame["fetched_count"])
    total_added = _sum_numeric(frame["added_count"])
    total_duplicates = _sum_numeric(frame["duplicate_count"])
    observation_days = int(frame["date"].nunique())
    profile_count = int(frame.groupby("date")["profile"].nunique().max())
    profile_runs = int(len(frame))
    quality_summary = _fetch_batch_quality_summary(frame)
    return {
        "period": period,
        "start_date": str(frame["date"].min()),
        "end_date": str(frame["date"].max()),
        "observation_days": observation_days,
        "daily_profile_count": profile_count,
        "theoretical_fetched_per_day": profile_count * 3,
        "total_fetched": total_fetched,
        "total_added": total_added,
        "total_duplicates": total_duplicates,
        "duplicate_rate": _safe_ratio(total_duplicates, total_fetched),
        "added_per_day": _safe_ratio(total_added, observation_days),
        "added_per_profile_run": _safe_ratio(total_added, profile_runs),
        **quality_summary,
    }


def _profile_change_after_evaluation(
    frame: pd.DataFrame,
    *,
    configured_profiles: list[NewsQueryProfile],
) -> pd.DataFrame:
    profile_lookup = {profile.name: profile for profile in configured_profiles}
    current_daily_names = [
        profile.name for profile in configured_profiles if profile.enabled_for_daily
    ]
    rows: list[dict[str, object]] = []
    for profile_name in current_daily_names:
        subset = frame[frame["profile"].eq(profile_name)].copy()
        configured = profile_lookup[profile_name]
        run_count = int(len(subset))
        total_fetched = _sum_numeric(subset.get("fetched_count"))
        total_added = _sum_numeric(subset.get("added_count"))
        total_duplicates = _sum_numeric(subset.get("duplicate_count"))
        recommendation = _profile_change_recommendation(
            run_count=run_count,
            total_fetched=total_fetched,
            total_added=total_added,
            duplicate_rate=_safe_ratio(total_duplicates, total_fetched),
        )
        quality_summary = _fetch_batch_quality_summary(subset)
        rows.append(
            {
                "profile": profile_name,
                "group": configured.group,
                "enabled_for_daily": configured.enabled_for_daily,
                "run_count": run_count,
                "total_fetched": total_fetched,
                "total_added": total_added,
                "total_duplicates": total_duplicates,
                "duplicate_rate": _safe_ratio(total_duplicates, total_fetched),
                "added_per_run": _safe_ratio(total_added, run_count),
                "latest_fetched_at": _format_datetime(
                    subset["fetched_at_local"].max() if not subset.empty else pd.NaT
                ),
                "unknown_source_count": quality_summary["unknown_source_count"],
                "unknown_source_occurrences": quality_summary["unknown_source_occurrences"],
                "unknown_source_observed_batches": quality_summary[
                    "unknown_source_observed_batches"
                ],
                "unknown_source_unrecorded_batches": quality_summary[
                    "unknown_source_unrecorded_batches"
                ],
                "check_news_ok_count": quality_summary["check_news_ok_count"],
                "check_news_warn_count": quality_summary["check_news_warn_count"],
                "check_news_error_count": quality_summary["check_news_error_count"],
                "recommendation": recommendation["recommendation"],
                "reason": recommendation["reason"],
            }
        )
    return pd.DataFrame(rows, columns=DAILY_PROFILE_AFTER_COLUMNS)


def _profile_change_recommendation(
    *,
    run_count: int,
    total_fetched: int,
    total_added: int,
    duplicate_rate: float | None,
) -> dict[str, str]:
    rate = duplicate_rate or 0.0
    if run_count < 3:
        return {
            "recommendation": "needs_more_data",
            "reason": "Fewer than 3 post-change runs are available.",
        }
    if total_fetched <= 0:
        return {
            "recommendation": "needs_more_data",
            "reason": "No usable post-change fetch rows are available.",
        }
    if total_added == 0 and rate >= 0.95:
        return {
            "recommendation": "research_only_candidate",
            "reason": "Post-change runs added no rows and were almost entirely duplicates.",
        }
    if rate >= 0.85 and _safe_ratio(total_added, run_count) < 0.5:
        return {
            "recommendation": "daily_reduce_limit",
            "reason": "Duplicate rate remains high and added rows per run are low.",
        }
    if total_added > 0:
        return {
            "recommendation": "daily_keep",
            "reason": "The profile continues to add new rows under NewsLimit 3.",
        }
    return {
        "recommendation": "needs_more_data",
        "reason": "The post-change result is inconclusive.",
    }


def _profile_change_overall_recommendation(
    before: dict[str, object],
    after: dict[str, object],
    profiles: pd.DataFrame,
    *,
    removed_profile: str,
) -> dict[str, object]:
    after_days = int(after.get("observation_days") or 0)
    duplicate_change = _comparison_label(
        safe_float(before.get("duplicate_rate")),
        safe_float(after.get("duplicate_rate")),
        lower_is_better=True,
    )
    added_change = _comparison_label(
        safe_float(before.get("added_per_day")),
        safe_float(after.get("added_per_day")),
        lower_is_better=False,
    )
    if after_days < 3:
        recommendation = "needs_more_data"
        reason = (
            f"Only {after_days} post-change observation days are available. "
            "Keep the 4-profile configuration unchanged until at least 3 days are collected."
        )
        removed_effect = f"Keep {removed_profile} disabled for now and collect more observations."
    elif duplicate_change == "improved" and added_change != "worsened":
        recommendation = "keep_4_profiles"
        reason = "Duplicate rate improved without a clear decline in added rows per day."
        removed_effect = f"Maintain the exclusion of {removed_profile}."
    elif duplicate_change == "worsened" and added_change == "worsened":
        recommendation = "review_4_profiles"
        reason = "Both duplicate rate and added rows per day worsened after the change."
        removed_effect = (
            f"Do not restore {removed_profile} automatically; review the remaining profiles first."
        )
    else:
        recommendation = (
            "keep_4_profiles" if after_days >= 5 else "keep_4_profiles_provisionally"
        )
        reason = (
            "Five or more post-change days show a mixed quality result, but the lighter "
            "configuration reduces daily fetch volume without evidence that restoring the removed "
            "zero-add profile would help."
            if after_days >= 5
            else "The operational effect is mixed; keep the lighter configuration while observing more days."
        )
        removed_effect = (
            f"Maintain the exclusion of {removed_profile}; review remaining duplicate-only profiles separately."
            if after_days >= 5
            else f"Maintain the exclusion of {removed_profile} provisionally."
        )

    if not profiles.empty and profiles["recommendation"].eq("research_only_candidate").any():
        reason += " At least one remaining profile may later be better suited to research-only use."
    return {
        "overall_recommendation": recommendation,
        "overall_reason": reason,
        "removed_profile_effect": removed_effect,
        "duplicate_rate_change": duplicate_change,
        "added_per_day_change": added_change,
        "unknown_source_change": _unknown_source_comparison(before, after),
        "after_observation_days": after_days,
    }


def _unknown_source_comparison(
    before: dict[str, object],
    after: dict[str, object],
) -> str:
    before_count = safe_float(before.get("unknown_source_count"))
    after_count = safe_float(after.get("unknown_source_count"))
    if before_count is None or after_count is None:
        return "unavailable"
    before_batches = int(safe_float(before.get("unknown_source_observed_batches")) or 0)
    after_batches = int(safe_float(after.get("unknown_source_observed_batches")) or 0)
    if before_batches <= 0 or after_batches <= 0:
        return "unavailable"
    before_rate = before_count / before_batches
    after_rate = after_count / after_batches
    return _comparison_label(before_rate, after_rate, lower_is_better=True, tolerance=0.1)


def _comparison_label(
    before: float | None,
    after: float | None,
    *,
    lower_is_better: bool,
    tolerance: float = 0.02,
) -> str:
    if before is None or after is None:
        return "unavailable"
    delta = after - before
    if abs(delta) <= tolerance:
        return "flat"
    improved = delta < 0 if lower_is_better else delta > 0
    return "improved" if improved else "worsened"


def _latest_fetch_batch_mask(history: pd.DataFrame) -> pd.Series:
    if history.empty or "fetched_at" not in history.columns:
        return pd.Series(False, index=history.index)
    timestamps = pd.to_datetime(
        history["fetched_at"],
        errors="coerce",
        format="mixed",
        utc=True,
    )
    valid = timestamps.dropna().sort_values()
    if valid.empty:
        return pd.Series(False, index=history.index)
    batch_start = valid.iloc[-1]
    for index in range(len(valid) - 1, 0, -1):
        current = valid.iloc[index]
        previous = valid.iloc[index - 1]
        if (current - previous).total_seconds() > 30:
            break
        batch_start = previous
    return timestamps.ge(batch_start) & timestamps.le(valid.iloc[-1])


def _fetch_batch_quality_summary(frame: pd.DataFrame) -> dict[str, object]:
    unavailable = {
        "unknown_source_count": None,
        "unknown_source_occurrences": None,
        "unknown_source_observed_batches": 0,
        "fetch_batch_count": 0,
        "unknown_source_unrecorded_batches": 0,
        "latest_unknown_sources": "",
        "check_news_status": "",
        "check_news_ok_count": 0,
        "check_news_warn_count": 0,
        "check_news_error_count": 0,
        "check_news_unrecorded_count": 0,
    }
    if frame.empty or "unknown_source_count" not in frame.columns:
        return unavailable

    rows = frame.copy().sort_values("fetched_at_local").reset_index(drop=True)
    gaps = rows["fetched_at_local"].diff().dt.total_seconds().fillna(float("inf"))
    rows["_quality_batch_id"] = gaps.gt(300).cumsum()
    fetch_batch_count = int(rows["_quality_batch_id"].nunique())
    rows["_unknown_count"] = pd.to_numeric(
        rows["unknown_source_count"],
        errors="coerce",
    )
    batches = (
        rows.groupby("_quality_batch_id", dropna=False)
        .agg(
            unknown_source_count=("_unknown_count", "max"),
            unknown_sources=("unknown_sources", "last"),
            check_news_status=("check_news_status", "last"),
            fetched_at_local=("fetched_at_local", "max"),
        )
        .reset_index(drop=True)
        .sort_values("fetched_at_local")
    )
    unknown_counts = pd.to_numeric(batches["unknown_source_count"], errors="coerce")
    unknown_observed = unknown_counts.notna()
    statuses = batches["check_news_status"].fillna("").astype(str).str.strip().str.upper()
    status_observed = statuses.ne("")
    latest_unknown = batches[unknown_observed].iloc[-1] if unknown_observed.any() else None
    latest_status = batches[status_observed].iloc[-1] if status_observed.any() else None
    return {
        "unknown_source_count": (
            int(unknown_counts[unknown_observed].sum()) if unknown_observed.any() else None
        ),
        "unknown_source_occurrences": (
            int(unknown_counts[unknown_observed].gt(0).sum())
            if unknown_observed.any()
            else None
        ),
        "unknown_source_observed_batches": int(unknown_observed.sum()),
        "fetch_batch_count": fetch_batch_count,
        "unknown_source_unrecorded_batches": int((~unknown_observed).sum()),
        "latest_unknown_sources": (
            _format_unknown_sources(latest_unknown.get("unknown_sources"))
            if latest_unknown is not None
            else ""
        ),
        "check_news_status": (
            str(latest_status.get("check_news_status") or "")
            if latest_status is not None
            else ""
        ),
        "check_news_ok_count": int(statuses.eq("OK").sum()),
        "check_news_warn_count": int(statuses.eq("WARN").sum()),
        "check_news_error_count": int(statuses.eq("ERROR").sum()),
        "check_news_unrecorded_count": int((~status_observed).sum()),
    }


def _format_unknown_sources(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(parsed, list):
        return text
    return "; ".join(str(item) for item in parsed if str(item).strip())


def _news_quality_batch_row(batch: pd.DataFrame) -> dict[str, object]:
    statuses = batch["check_news_status"].fillna("").astype(str).str.strip().str.upper()
    status = _highest_check_news_status(statuses)
    status_rows = batch[statuses.eq(status)] if status else batch.iloc[0:0]
    message_values = (
        status_rows["check_news_message"]
        if not status_rows.empty
        else batch["check_news_message"]
    )
    messages = message_values.fillna("").astype(str).str.strip()
    messages = messages[messages.ne("")]

    unknown_counts = pd.to_numeric(batch["unknown_source_count"], errors="coerce")
    unknown_count = int(unknown_counts.max()) if unknown_counts.notna().any() else None
    unknown_sources: list[str] = []
    for value in batch["unknown_sources"]:
        formatted = _format_unknown_sources(value)
        for source in formatted.split(";"):
            source = source.strip()
            if source and source not in unknown_sources:
                unknown_sources.append(source)

    profiles = [
        value
        for value in batch["profile"].fillna("").astype(str).str.strip().unique()
        if value
    ]
    return {
        "fetched_at": batch["_fetched_at"].max().isoformat(),
        "query_profile": "; ".join(profiles),
        "fetched_count": int(batch["fetched_count"].sum()),
        "added_count": int(batch["added_count"].sum()),
        "duplicate_count": int(batch["duplicate_count"].sum()),
        "check_news_status": status,
        "check_news_message": messages.iloc[-1] if not messages.empty else "",
        "unknown_source_count": unknown_count,
        "unknown_sources": "; ".join(unknown_sources),
    }


def _highest_check_news_status(statuses: pd.Series) -> str:
    values = set(statuses[statuses.ne("")])
    for status in ("ERROR", "WARN", "OK"):
        if status in values:
            return status
    return ""


def _newslimit3_profile_recommendation(row: pd.Series) -> dict[str, str]:
    profile = str(row.get("profile") or "")
    group = str(row.get("group") or "")
    runs = int(safe_float(row.get("runs")) or 0)
    total_added = int(safe_float(row.get("total_added")) or 0)
    total_fetched = int(safe_float(row.get("total_fetched")) or 0)
    duplicate_rate = safe_float(row.get("duplicate_rate")) or 0.0
    added_per_run = safe_float(row.get("added_per_run")) or 0.0

    if runs < 3:
        return {
            "recommendation": "needs_more_data",
            "reason": "Observation history is still short; keep collecting before changing config.",
        }
    if total_fetched <= 0:
        return {
            "recommendation": "needs_more_data",
            "reason": "No usable NewsLimit 3 fetch rows were found.",
        }
    if total_added == 0 and duplicate_rate >= 0.90:
        if group in {"core", "jp"}:
            return {
                "recommendation": "daily_reduce_limit",
                "reason": (
                    f"{profile} is daily-relevant, but NewsLimit 3 added no new rows "
                    "in this window; keep it light or review later."
                ),
            }
        return {
            "recommendation": "research_only_candidate",
            "reason": f"{profile} added no new rows and is better suited for manual research checks.",
        }
    if duplicate_rate >= 0.85 and added_per_run < 0.5:
        return {
            "recommendation": "daily_reduce_limit",
            "reason": "Duplicate rate is high and added rows per run are low; keep NewsLimit at 3.",
        }
    if added_per_run >= 0.5 and duplicate_rate <= 0.85:
        return {
            "recommendation": "daily_keep",
            "reason": "Profile is still adding rows under NewsLimit 3 with an acceptable duplicate rate.",
        }
    if total_added > 0:
        return {
            "recommendation": "daily_keep",
            "reason": "Profile has added rows in this window; continue collecting at NewsLimit 3.",
        }
    return {
        "recommendation": "needs_more_data",
        "reason": "Signal is inconclusive; keep collecting a few more daily runs.",
    }


def _daily_profile_config_suggestion(
    row: pd.Series,
    current_enabled: bool,
) -> tuple[bool, str, str]:
    profile = str(row.get("profile") or "")
    current_decision = str(row.get("daily_operation_recommendation") or "")
    runs = int(safe_float(row.get("runs")) or 0)
    total_added = int(safe_float(row.get("total_added")) or 0)
    duplicate_rate = safe_float(row.get("duplicate_rate")) or 0.0

    if profile in {"toyota_jp", "toyota_stock"}:
        return True, "daily_keep", "Core useful profile; keep it enabled for daily collection."
    if profile == "toyota_earnings" and runs >= 5 and total_added == 0 and duplicate_rate >= 0.90:
        return False, "disable_candidate", "Six-day NewsLimit 3 window added no rows; move it out of daily for now."
    if current_decision == "daily_reduce_limit" and total_added <= 1:
        return current_enabled, "needs_more_data", "Duplicate-heavy, but keep observing before removing more daily profiles."
    if current_decision == "daily_keep":
        return current_enabled, "daily_keep", "Profile still adds rows under NewsLimit 3."
    return current_enabled, "needs_more_data", "Keep current setting until more observations are available."


def _config_change_text(current_enabled: bool, suggested_enabled: bool) -> str:
    if current_enabled == suggested_enabled:
        return "no_change"
    if current_enabled and not suggested_enabled:
        return "enabled_for_daily true -> false"
    if not current_enabled and suggested_enabled:
        return "enabled_for_daily false -> true"
    return "no_change"


def _newslimit3_overall_recommendation(
    daily: pd.DataFrame,
    profiles: pd.DataFrame,
) -> dict[str, object]:
    if daily.empty or profiles.empty:
        return {
            "overall_recommendation": "needs_more_data",
            "overall_reason": "No usable NewsLimit 3 daily-profile history was found.",
            "observation_days": 0,
            "total_fetched": 0,
            "total_added": 0,
            "total_duplicates": 0,
            "duplicate_rate": None,
            "caveats": "This is an operational data-quality evaluation, not proof of market predictive power.",
        }
    total_fetched = int(pd.to_numeric(daily["fetched"], errors="coerce").fillna(0).sum())
    total_added = int(pd.to_numeric(daily["added"], errors="coerce").fillna(0).sum())
    total_duplicates = int(pd.to_numeric(daily["duplicates"], errors="coerce").fillna(0).sum())
    duplicate_rate = _safe_ratio(total_duplicates, total_fetched)
    observation_days = int(daily["date"].nunique())
    profile_recs = profiles["daily_operation_recommendation"].fillna("").astype(str)
    keep_count = int(profile_recs.eq("daily_keep").sum())
    reduce_count = int(profile_recs.eq("daily_reduce_limit").sum())
    more_count = int(profile_recs.eq("needs_more_data").sum())

    if observation_days < 3:
        recommendation = "needs_more_data"
        reason = "NewsLimit 3 has started working, but fewer than 3 observation days are available."
    elif keep_count >= 1 and reduce_count >= 1:
        recommendation = "mixed"
        reason = (
            "Keep NewsLimit 3 overall, but some daily profiles look duplicate-heavy. "
            "Consider profile-level review rather than raising the global limit."
        )
    elif keep_count >= 1 and duplicate_rate is not None and duplicate_rate <= 0.90:
        recommendation = "keep_newslimit_3"
        reason = "NewsLimit 3 is adding rows while keeping daily fetch volume light."
    elif total_added == 0:
        recommendation = "keep_newslimit_3"
        reason = "Recent runs are duplicate-heavy; increasing the global limit is unlikely to help yet."
    elif more_count:
        recommendation = "keep_newslimit_3"
        reason = "Keep NewsLimit 3 while collecting more observations for inconclusive profiles."
    else:
        recommendation = "keep_newslimit_3"
        reason = "Current data does not justify increasing the global NewsLimit."

    return {
        "overall_recommendation": recommendation,
        "overall_reason": reason,
        "observation_days": observation_days,
        "total_fetched": total_fetched,
        "total_added": total_added,
        "total_duplicates": total_duplicates,
        "duplicate_rate": duplicate_rate,
        "caveats": "This is an operational data-quality evaluation, not proof of market predictive power.",
    }


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    top = safe_float(numerator)
    bottom = safe_float(denominator)
    if top is None or bottom in (None, 0):
        return None
    return float(top / bottom)


def _recommendation_row(row: pd.Series) -> dict[str, object]:
    profile = str(row.get("profile") or "")
    group = str(row.get("group") or "")
    enabled = bool(row.get("enabled_for_daily"))
    saved_rows = int(safe_float(row.get("saved_rows")) or 0)
    fetched_count = int(safe_float(row.get("history_fetched_count")) or 0)
    added_count = int(safe_float(row.get("history_added_count")) or 0)
    duplicate_rate = safe_float(row.get("duplicate_rate"))
    mojibake_count = int(safe_float(row.get("mojibake_suspect_count")) or 0)
    invalid_dates = int(safe_float(row.get("invalid_date_count")) or 0)
    unknown_sources = int(safe_float(row.get("unknown_source_count")) or 0)
    status = str(row.get("quality_status") or "OK")

    recommendation = "daily_keep" if enabled else "research_only"
    priority = "Low"
    reason = "Profile is suitable for its current role."
    if mojibake_count or invalid_dates:
        recommendation = "review_needed"
        priority = "High"
        reason = "Data quality issue detected: mojibake or invalid dates."
    elif unknown_sources >= 3:
        recommendation = "review_needed"
        priority = "Medium"
        reason = "Many saved rows have unknown sources; add source profiles or keep this research-only."
    elif fetched_count >= 5 and added_count == 0:
        recommendation = "review_needed" if enabled else "research_only"
        priority = "Medium" if enabled else "Low"
        reason = "Recent fetch history added no new rows."
    elif duplicate_rate is not None and duplicate_rate >= 0.80:
        recommendation = "review_needed" if enabled else "research_only"
        priority = "Medium" if enabled else "Low"
        reason = "Duplicate rate is high; consider lowering limit or moving to research-only."
    elif saved_rows == 0 and enabled:
        recommendation = "review_needed"
        priority = "Medium"
        reason = "Daily-enabled profile has no saved rows yet."
    elif status == "WARN" and enabled:
        recommendation = "review_needed"
        priority = "Medium"
        reason = str(row.get("quality_message") or "Profile quality warning.")
    return {
        "profile": profile,
        "group": group,
        "enabled_for_daily": enabled,
        "recommendation": recommendation,
        "priority": priority,
        "reason": reason,
    }


def _normalize_fetch_history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy() if frame is not None else pd.DataFrame()
    for column in FETCH_HISTORY_COLUMNS:
        if column not in result.columns:
            result[column] = ""
    result = result[list(FETCH_HISTORY_COLUMNS)].copy()
    for column in ("fetched_count", "added_count", "duplicate_count"):
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    result["unknown_source_count"] = pd.to_numeric(
        result["unknown_source_count"],
        errors="coerce",
    ).astype("Int64")
    for column in (
        "fetched_at",
        "profile",
        "query",
        "group",
        "language",
        "region",
        "unknown_sources",
        "check_news_status",
        "check_news_message",
        "warning",
        "error",
    ):
        result[column] = result[column].fillna("").astype(str)
    result["enabled_for_daily"] = result["enabled_for_daily"].map(_as_bool)
    return result


def _latest_history_row(history_subset: pd.DataFrame) -> pd.Series | None:
    if history_subset.empty or "fetched_at" not in history_subset.columns:
        return None
    history = history_subset.copy()
    history["_fetched_at"] = pd.to_datetime(
        history["fetched_at"],
        errors="coerce",
        format="mixed",
    )
    history = history.sort_values("_fetched_at", ascending=False)
    if history.empty:
        return None
    return history.iloc[0]


def _history_count(row: pd.Series | None, column: str) -> int:
    if row is None:
        return 0
    return int(safe_float(row.get(column)) or 0)


def _news_for_profile(news: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    if news.empty or "query_profile" not in news.columns:
        return pd.DataFrame()
    query_profile = news["query_profile"].fillna("").astype(str).str.strip()
    return news[query_profile.eq(profile_name)].copy()


def _history_for_profile(history: pd.DataFrame, profile_name: str) -> pd.DataFrame:
    if history.empty or "profile" not in history.columns:
        return pd.DataFrame(columns=FETCH_HISTORY_COLUMNS)
    profile = history["profile"].fillna("").astype(str).str.strip()
    return history[profile.eq(profile_name)].copy()


def _read_news_quality_items(news_dir: str | Path, ticker: str) -> pd.DataFrame:
    for path in news_path_candidates(news_dir, ticker):
        if not path.exists():
            continue
        try:
            raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)
        if raw.empty:
            return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)
        result = raw.copy()
        for column in NEWS_DISPLAY_COLUMNS:
            if column not in result.columns:
                result[column] = ""
        return result[list(NEWS_DISPLAY_COLUMNS)].copy()
    return pd.DataFrame(columns=NEWS_DISPLAY_COLUMNS)


def _quality_status(
    *,
    duplicate_rate: float | None,
    added_total: int,
    fetched_total: int,
    mojibake_count: int,
    invalid_dates: int,
    unknown_sources: int,
    latest_news_date,
    enabled_for_daily: bool,
) -> tuple[str, str]:
    if mojibake_count or invalid_dates:
        return "FAIL", "Mojibake suspects or invalid dates were found."
    warnings: list[str] = []
    if unknown_sources:
        warnings.append(f"unknown sources={unknown_sources}")
    if duplicate_rate is not None and duplicate_rate >= 0.80:
        warnings.append(f"high duplicate rate={duplicate_rate:.0%}")
    if fetched_total >= 5 and added_total == 0:
        warnings.append("recent history added no rows")
    if enabled_for_daily and pd.notna(latest_news_date):
        age_days = (pd.Timestamp.now().normalize() - pd.Timestamp(latest_news_date).normalize()).days
        if age_days >= 14:
            warnings.append(f"latest news is {age_days} days old")
    if warnings:
        return "WARN", "; ".join(warnings)
    return "OK", "Profile quality looks acceptable for its current role."


def _sum_numeric(values) -> int:
    if values is None:
        return 0
    return int(pd.to_numeric(values, errors="coerce").fillna(0).sum())


def _mean(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _distribution_text(values) -> str:
    if values is None:
        return ""
    clean = pd.Series(values).fillna("").astype(str).str.strip()
    clean = clean.where(clean.ne(""), "unknown")
    counts = clean.value_counts().sort_index()
    return ", ".join(f"{key}:{int(value)}" for key, value in counts.items())


def _unknown_source_count(news: pd.DataFrame) -> int:
    if news.empty or "source_type" not in news.columns:
        return 0
    source_type = news["source_type"].fillna("").astype(str).str.strip().str.lower()
    return int(source_type.isin({"", "unknown"}).sum())


def _format_date(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _format_datetime(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
