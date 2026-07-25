from __future__ import annotations

import math

import pandas as pd


DEFAULT_SIMILARITY_FEATURES = (
    "return_1d",
    "volume_change_1d",
    "rsi_14",
    "macd",
    "volatility_20",
    "usdjpy_return_1d",
    "usdjpy_trend_direction",
    "n225_return_1d",
    "n225_trend_direction",
    "topix_return_1d",
    "topix_trend_direction",
    "sector_mean_return",
    "sector_trend",
    "sector_strength",
    "honda_return_1d",
    "nissan_return_1d",
    "tesla_return_1d",
    "byd_return_1d",
    "days_to_earnings",
    "days_after_earnings",
    "earnings_week_flag",
    "earnings_day_flag",
    "post_earnings_mode",
    "earnings_event_state_code",
    "earnings_pre_volatility",
    "earnings_post_gap",
    "earnings_volume_spike",
    "earnings_after_trend_5",
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
    "dominant_narrative_code",
    "dominant_narrative_sentiment",
    "dominant_narrative_importance",
    "dominant_narrative_trend",
    "narrative_bullish_pressure",
    "narrative_bearish_pressure",
)

FEATURE_LABELS = {
    "return_1d": "Close return",
    "volume_change_1d": "Volume change",
    "rsi_14": "RSI",
    "macd": "MACD",
    "volatility_20": "Volatility",
    "usdjpy_return_1d": "USDJPY return",
    "usdjpy_trend_direction": "USDJPY trend",
    "n225_return_1d": "Nikkei return",
    "n225_trend_direction": "Nikkei trend",
    "topix_return_1d": "TOPIX return",
    "topix_trend_direction": "TOPIX trend",
    "sector_mean_return": "Auto sector return",
    "sector_trend": "Auto sector trend",
    "sector_strength": "Auto sector strength",
    "honda_return_1d": "Honda return",
    "nissan_return_1d": "Nissan return",
    "tesla_return_1d": "Tesla return",
    "byd_return_1d": "BYD return",
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
    "dominant_narrative_code": "Dominant narrative",
    "dominant_narrative_sentiment": "Narrative sentiment",
    "dominant_narrative_importance": "Narrative importance",
    "dominant_narrative_trend": "Narrative trend",
    "narrative_bullish_pressure": "Narrative bullish pressure",
    "narrative_bearish_pressure": "Narrative bearish pressure",
}


def find_similar_patterns(
    data: pd.DataFrame,
    window_size: int = 20,
    top_n: int = 5,
    forward_days: tuple[int, ...] = (1, 5, 20),
    feature_columns: tuple[str, ...] | list[str] | None = None,
    regime_column: str | None = None,
    prefer_regime: str | None = None,
    event_state_column: str | None = None,
    prefer_event_state: str | None = None,
    narrative_column: str | None = None,
    prefer_narrative: str | None = None,
) -> pd.DataFrame:
    """Find historical windows similar to the latest window.

    The initial MVP uses cosine similarity over normalized feature windows.
    Candidate windows and their forward returns are kept fully before the latest
    comparison window to avoid mixing the current pattern into historical cases.
    """
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")

    max_forward = max(forward_days)

    if not {"date", "close"}.issubset(data.columns):
        return _empty_result(forward_days)

    ordered = data.sort_values("date").reset_index(drop=True).copy()
    features = _available_features(ordered, feature_columns)
    if len(ordered) < window_size * 2 + max_forward:
        return _empty_result(forward_days)
    if not features:
        return _empty_result(forward_days)

    latest_start = len(ordered) - window_size
    latest_window = ordered.iloc[latest_start:]
    latest_vector = _window_vector(latest_window, features)
    latest_norm = _vector_norm(latest_vector)
    if latest_norm == 0:
        return _empty_result(forward_days)

    max_candidate_end = latest_start - max_forward - 1
    min_candidate_end = window_size - 1
    if max_candidate_end < min_candidate_end:
        return _empty_result(forward_days)

    rows: list[dict[str, object]] = []
    for candidate_end in range(min_candidate_end, max_candidate_end + 1):
        candidate_start = candidate_end - window_size + 1
        candidate_window = ordered.iloc[candidate_start : candidate_end + 1]
        candidate_vector = _window_vector(candidate_window, features)
        similarity = _cosine_similarity(latest_vector, candidate_vector)
        if similarity is None:
            continue

        row: dict[str, object] = {
            "similar_start_date": ordered.loc[candidate_start, "date"],
            "similar_end_date": ordered.loc[candidate_end, "date"],
            "similarity": similarity,
        }
        if regime_column and regime_column in ordered.columns:
            candidate_regime = ordered.loc[candidate_end, regime_column]
            row["regime"] = candidate_regime
            row["regime_match"] = _regime_matches(candidate_regime, prefer_regime)
        if event_state_column and event_state_column in ordered.columns:
            candidate_event_state = ordered.loc[candidate_end, event_state_column]
            row["event_state"] = candidate_event_state
            row["event_state_match"] = _regime_matches(
                candidate_event_state,
                prefer_event_state,
            )
        if narrative_column and narrative_column in ordered.columns:
            candidate_narrative = ordered.loc[candidate_end, narrative_column]
            row["narrative"] = candidate_narrative
            row["narrative_match"] = _regime_matches(candidate_narrative, prefer_narrative)
        if "anomaly_score" in ordered.columns:
            row["anomaly_score"] = ordered.loc[candidate_end, "anomaly_score"]
        if "anomaly_types" in ordered.columns:
            row["anomaly_types"] = ordered.loc[candidate_end, "anomaly_types"]
        feature_scores = _feature_similarity_scores(latest_window, candidate_window, features)
        row["similar_features"] = ", ".join(
            str(item["feature"]) for item in feature_scores[:3]
        )
        row["similar_feature_scores"] = feature_scores
        row["pattern_start_date"] = row["similar_start_date"]
        row["similar_date"] = row["similar_end_date"]
        for days in forward_days:
            row[f"return_{days}d_after"] = _forward_return(ordered, candidate_end, days)
        rows.append(row)

    if not rows:
        return _empty_result(forward_days)

    result = pd.DataFrame(rows)
    sort_columns: list[str] = []
    ascending: list[bool] = []
    if prefer_event_state and "event_state_match" in result.columns:
        sort_columns.append("event_state_match")
        ascending.append(False)
    if prefer_narrative and "narrative_match" in result.columns:
        sort_columns.append("narrative_match")
        ascending.append(False)
    if prefer_regime and "regime_match" in result.columns:
        sort_columns.append("regime_match")
        ascending.append(False)
    sort_columns.append("similarity")
    ascending.append(False)
    if sort_columns:
        result = result.sort_values(sort_columns, ascending=ascending)
    else:
        result = result.sort_values("similarity", ascending=False)
    result = result.head(top_n).reset_index(drop=True)
    return result


def summarize_forward_returns(
    patterns: pd.DataFrame,
    forward_days: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    columns = [
        "horizon",
        "count",
        "average_return",
        "median_return",
        "positive_rate",
        "best_return",
        "worst_return",
    ]
    if patterns.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for days in forward_days:
        column = f"return_{days}d_after"
        if column not in patterns.columns:
            continue
        values = pd.to_numeric(patterns[column], errors="coerce").dropna()
        if values.empty:
            rows.append(
                {
                    "horizon": f"{days}D",
                    "count": 0,
                    "average_return": None,
                    "median_return": None,
                    "positive_rate": None,
                    "best_return": None,
                    "worst_return": None,
                }
            )
            continue
        rows.append(
            {
                "horizon": f"{days}D",
                "count": int(len(values)),
                "average_return": float(values.mean()),
                "median_return": float(values.median()),
                "positive_rate": float((values > 0).mean()),
                "best_return": float(values.max()),
                "worst_return": float(values.min()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_pattern_comparison_frame(
    data: pd.DataFrame,
    pattern: pd.Series | dict[str, object],
    window_size: int = 20,
    forward_days: int = 20,
) -> pd.DataFrame:
    columns = [
        "series",
        "phase",
        "day",
        "date",
        "close",
        "normalized_close",
    ]
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame(columns=columns)

    ordered = data.sort_values("date").reset_index(drop=True).copy()
    ordered["date"] = pd.to_datetime(ordered["date"], errors="coerce")
    ordered["close"] = pd.to_numeric(ordered["close"], errors="coerce")
    ordered = ordered.dropna(subset=["date", "close"]).reset_index(drop=True)
    if len(ordered) < window_size:
        return pd.DataFrame(columns=columns)

    latest_start = len(ordered) - window_size
    latest_end = len(ordered) - 1
    match_start, match_end = _match_window_indices(ordered, pattern)
    if match_start is None or match_end is None:
        return pd.DataFrame(columns=columns)

    rows = _price_path_rows(
        ordered,
        start_index=latest_start,
        end_index=latest_end,
        label="Current",
        window_size=window_size,
    )
    rows.extend(
        _price_path_rows(
            ordered,
            start_index=match_start,
            end_index=min(match_end + forward_days, len(ordered) - 1),
            label="Matched past",
            window_size=window_size,
        )
    )
    return pd.DataFrame(rows, columns=columns)


def _available_features(
    data: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] | None,
) -> list[str]:
    candidates = feature_columns or DEFAULT_SIMILARITY_FEATURES
    features: list[str] = []
    for column in candidates:
        if column not in data.columns:
            continue
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.notna().any():
            features.append(column)
    return features


def _empty_result(forward_days: tuple[int, ...]) -> pd.DataFrame:
    columns = [
        "similar_start_date",
        "similar_end_date",
        "similarity",
        "regime",
        "regime_match",
        "event_state",
        "event_state_match",
        "narrative",
        "narrative_match",
        "similar_features",
        "similar_feature_scores",
        "pattern_start_date",
        "similar_date",
    ]
    columns.extend(f"return_{days}d_after" for days in forward_days)
    return pd.DataFrame(columns=columns)


def _forward_return(data: pd.DataFrame, candidate_end: int, days: int) -> float | None:
    close_at_end = pd.to_numeric(
        pd.Series([data.loc[candidate_end, "close"]]), errors="coerce"
    ).iloc[0]
    future_close = pd.to_numeric(
        pd.Series([data.loc[candidate_end + days, "close"]]), errors="coerce"
    ).iloc[0]
    if pd.isna(close_at_end) or pd.isna(future_close) or close_at_end == 0:
        return None
    return float(future_close / close_at_end - 1)


def _window_vector(window: pd.DataFrame, features: list[str]) -> list[float]:
    values: list[float] = []
    for feature in features:
        series = pd.to_numeric(window[feature], errors="coerce")
        normalized = _normalize_series(series)
        values.extend(normalized)
    return values


def _normalize_series(series: pd.Series) -> list[float]:
    clean = series.replace([float("inf"), -float("inf")], pd.NA)
    clean = clean.ffill().bfill().fillna(0.0).astype(float)
    mean = float(clean.mean())
    std = float(clean.std(ddof=0))
    if std == 0 or math.isnan(std):
        return [0.0 for _ in clean]
    return [float((value - mean) / std) for value in clean]


def _cosine_similarity(left: list[float], right: list[float]) -> float | None:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0 or right_norm == 0:
        return None
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    return float(dot / (left_norm * right_norm))


def _vector_norm(values: list[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _feature_similarity_scores(
    latest_window: pd.DataFrame,
    candidate_window: pd.DataFrame,
    features: list[str],
) -> list[dict[str, object]]:
    feature_scores: list[dict[str, object]] = []
    for feature in features:
        latest_vector = _normalize_series(pd.to_numeric(latest_window[feature], errors="coerce"))
        candidate_vector = _normalize_series(
            pd.to_numeric(candidate_window[feature], errors="coerce")
        )
        score = _cosine_similarity(latest_vector, candidate_vector)
        if score is None:
            continue
        feature_scores.append(
            {
                "feature": _feature_label(feature),
                "column": feature,
                "similarity": score,
            }
        )

    feature_scores.sort(key=lambda item: float(item["similarity"]), reverse=True)
    return feature_scores


def _match_window_indices(
    data: pd.DataFrame,
    pattern: pd.Series | dict[str, object],
) -> tuple[int | None, int | None]:
    start_date = _pattern_value(pattern, "similar_start_date")
    end_date = _pattern_value(pattern, "similar_end_date")
    if start_date is None or end_date is None:
        return None, None

    start = pd.to_datetime(start_date, errors="coerce")
    end = pd.to_datetime(end_date, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return None, None

    start_matches = data.index[data["date"] == start].tolist()
    end_matches = data.index[data["date"] == end].tolist()
    if not start_matches or not end_matches:
        return None, None
    return int(start_matches[0]), int(end_matches[0])


def _pattern_value(pattern: pd.Series | dict[str, object], key: str):
    if isinstance(pattern, pd.Series):
        return pattern.get(key)
    return pattern.get(key)


def _price_path_rows(
    data: pd.DataFrame,
    start_index: int,
    end_index: int,
    label: str,
    window_size: int,
) -> list[dict[str, object]]:
    if start_index < 0 or end_index < start_index:
        return []

    window = data.iloc[start_index : end_index + 1]
    base_close = float(window.iloc[0]["close"])
    if base_close == 0 or math.isnan(base_close):
        return []

    rows: list[dict[str, object]] = []
    for offset, row in enumerate(window.itertuples(index=False)):
        rows.append(
            {
                "series": label,
                "phase": "future" if offset >= window_size else "pattern",
                "day": offset,
                "date": row.date,
                "close": row.close,
                "normalized_close": float(row.close / base_close - 1),
            }
        )
    return rows


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


def _regime_matches(candidate_regime, prefer_regime: str | None) -> bool:
    if not prefer_regime:
        return False
    if pd.isna(candidate_regime):
        return False
    return str(candidate_regime) == str(prefer_regime)
