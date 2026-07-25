from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ai.utils.safe_cast import safe_float


REGIME_BULL = "Bull Trend"
REGIME_BEAR = "Bear Trend"
REGIME_SIDEWAYS = "Sideways"
REGIME_HIGH_VOL = "High Volatility"
REGIME_PANIC = "Panic"
REGIME_COLUMN = "regime"


@dataclass(frozen=True)
class MarketRegime:
    name: str
    level: str
    reasons: list[str]
    features: list[str]
    metrics: dict[str, float | None]


def add_regime_labels(data: pd.DataFrame) -> pd.DataFrame:
    """Add rule-based market regime labels without using future data."""
    if data.empty:
        return data.copy()

    prepared = _prepare_metrics(data)
    labels: list[str] = []
    for index in prepared.index:
        labels.append(_classify_row(prepared, index).name)

    result = prepared.copy()
    result[REGIME_COLUMN] = labels
    return result


def classify_latest_regime(data: pd.DataFrame) -> MarketRegime:
    """Classify the latest row's market regime."""
    if data.empty:
        return _fallback_regime("No data is available.")

    prepared = _prepare_metrics(data)
    valid = prepared.dropna(subset=["date", "close"], how="any")
    if valid.empty:
        return _fallback_regime("No valid latest price row is available.")

    return _classify_row(prepared, valid.index[-1])


def _prepare_metrics(data: pd.DataFrame) -> pd.DataFrame:
    sort_columns = ["date"]
    if "ticker" in data.columns:
        sort_columns = ["ticker", "date"]

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values(sort_columns).reset_index(drop=True)

    for column in (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1d",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "volatility_20",
        "close_vs_ma_25",
        "usdjpy_return_1d",
        "usdjpy_trend_direction",
        "n225_return_1d",
        "n225_volatility_20",
        "n225_trend_direction",
        "topix_return_1d",
        "topix_volatility_20",
        "topix_trend_direction",
        "honda_return_1d",
        "honda_trend_direction",
        "nissan_return_1d",
        "nissan_trend_direction",
        "tesla_return_1d",
        "tesla_trend_direction",
        "byd_return_1d",
        "byd_trend_direction",
        "sector_mean_return",
        "sector_volatility",
        "sector_trend",
        "sector_strength",
        "days_to_earnings",
        "days_after_earnings",
        "earnings_week_flag",
        "earnings_day_flag",
        "post_earnings_mode",
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
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if "return_1d" not in result.columns and "close" in result.columns:
        result["return_1d"] = _grouped_pct_change(result, "close")

    if "volatility_20" not in result.columns and "return_1d" in result.columns:
        result["volatility_20"] = _grouped_rolling_std(result, "return_1d", window=20)

    if "macd_hist" not in result.columns and {"macd", "macd_signal"}.issubset(result.columns):
        result["macd_hist"] = result["macd"] - result["macd_signal"]

    if "close_vs_ma_25" not in result.columns and "close" in result.columns:
        ma_25 = _grouped_rolling_mean(result, "close", window=25)
        result["close_vs_ma_25"] = result["close"] / ma_25 - 1

    result["price_change_5d"] = _grouped_pct_change(result, "close", periods=5)
    result["volume_ratio_20"] = _volume_ratio(result)
    result["atr_14"] = _atr(result, window=14)
    result["atr_pct_14"] = result["atr_14"] / result["close"]

    volatility_median = _grouped_rolling_median(
        result,
        "volatility_20",
        window=60,
        min_periods=20,
    )
    result["volatility_ratio_60"] = result["volatility_20"] / volatility_median
    return result.replace([float("inf"), -float("inf")], pd.NA)


def _classify_row(data: pd.DataFrame, index: int) -> MarketRegime:
    row = data.loc[index]
    metrics = {
        "volatility": _value(row, "volatility_20"),
        "volatility_ratio": _value(row, "volatility_ratio_60"),
        "atr_pct": _value(row, "atr_pct_14"),
        "rsi": _value(row, "rsi_14"),
        "ma_deviation": _value(row, "close_vs_ma_25"),
        "volume_ratio": _value(row, "volume_ratio_20"),
        "macd_hist": _value(row, "macd_hist"),
        "price_change_1d": _value(row, "return_1d"),
        "price_change_5d": _value(row, "price_change_5d"),
        "usdjpy_return": _value(row, "usdjpy_return_1d"),
        "usdjpy_trend": _value(row, "usdjpy_trend_direction"),
        "n225_return": _value(row, "n225_return_1d"),
        "n225_volatility": _value(row, "n225_volatility_20"),
        "n225_trend": _value(row, "n225_trend_direction"),
        "topix_return": _value(row, "topix_return_1d"),
        "topix_volatility": _value(row, "topix_volatility_20"),
        "topix_trend": _value(row, "topix_trend_direction"),
        "honda_return": _value(row, "honda_return_1d"),
        "honda_trend": _value(row, "honda_trend_direction"),
        "nissan_return": _value(row, "nissan_return_1d"),
        "nissan_trend": _value(row, "nissan_trend_direction"),
        "tesla_return": _value(row, "tesla_return_1d"),
        "tesla_trend": _value(row, "tesla_trend_direction"),
        "byd_return": _value(row, "byd_return_1d"),
        "byd_trend": _value(row, "byd_trend_direction"),
        "sector_mean_return": _value(row, "sector_mean_return"),
        "sector_volatility": _value(row, "sector_volatility"),
        "sector_trend": _value(row, "sector_trend"),
        "sector_strength": _value(row, "sector_strength"),
        "days_to_earnings": _value(row, "days_to_earnings"),
        "days_after_earnings": _value(row, "days_after_earnings"),
        "earnings_week_flag": _value(row, "earnings_week_flag"),
        "earnings_day_flag": _value(row, "earnings_day_flag"),
        "post_earnings_mode": _value(row, "post_earnings_mode"),
        "earnings_pre_volatility": _value(row, "earnings_pre_volatility"),
        "earnings_post_gap": _value(row, "earnings_post_gap"),
        "earnings_volume_spike": _value(row, "earnings_volume_spike"),
        "earnings_after_trend_5": _value(row, "earnings_after_trend_5"),
        "news_count_1d": _value(row, "news_count_1d"),
        "news_count_3d": _value(row, "news_count_3d"),
        "avg_sentiment_1d": _value(row, "avg_sentiment_1d"),
        "avg_sentiment_3d": _value(row, "avg_sentiment_3d"),
        "max_importance_3d": _value(row, "max_importance_3d"),
        "negative_news_flag": _value(row, "negative_news_flag"),
        "positive_news_flag": _value(row, "positive_news_flag"),
        "sentiment_trend_3d": _value(row, "sentiment_trend_3d"),
        "sentiment_trend_7d": _value(row, "sentiment_trend_7d"),
        "importance_weighted_sentiment": _value(row, "importance_weighted_sentiment"),
        "negative_news_shock": _value(row, "negative_news_shock"),
        "positive_news_shock": _value(row, "positive_news_shock"),
        "sector_news_strength": _value(row, "sector_news_strength"),
        "dominant_narrative_code": _value(row, "dominant_narrative_code"),
        "dominant_narrative_sentiment": _value(row, "dominant_narrative_sentiment"),
        "dominant_narrative_importance": _value(row, "dominant_narrative_importance"),
        "dominant_narrative_trend": _value(row, "dominant_narrative_trend"),
        "narrative_bullish_pressure": _value(row, "narrative_bullish_pressure"),
        "narrative_bearish_pressure": _value(row, "narrative_bearish_pressure"),
    }
    market_trend_score = _market_trend_score(metrics)
    metrics["market_trend_score"] = float(market_trend_score) if market_trend_score is not None else None
    auto_trend_score = _auto_trend_score(metrics)
    metrics["auto_trend_score"] = float(auto_trend_score) if auto_trend_score is not None else None

    if _is_panic(metrics):
        return MarketRegime(
            name=REGIME_PANIC,
            level="High stress",
            reasons=_panic_reasons(metrics),
            features=_features(
                metrics,
                "price_change_1d",
                "price_change_5d",
                "volume_ratio",
                "rsi",
                "n225_return",
                "topix_return",
                "sector_mean_return",
                "tesla_return",
                "earnings_day_flag",
                "post_earnings_mode",
                "negative_news_flag",
                "negative_news_shock",
                "max_importance_3d",
            ),
            metrics=metrics,
        )

    if _is_high_volatility(metrics):
        return MarketRegime(
            name=REGIME_HIGH_VOL,
            level="Elevated stress",
            reasons=_high_volatility_reasons(metrics),
            features=_features(
                metrics,
                "volatility",
                "volatility_ratio",
                "atr_pct",
                "volume_ratio",
                "n225_volatility",
                "topix_volatility",
                "sector_volatility",
                "earnings_week_flag",
                "earnings_pre_volatility",
                "earnings_volume_spike",
                "news_count_3d",
                "max_importance_3d",
                "negative_news_shock",
                "positive_news_shock",
            ),
            metrics=metrics,
        )

    bull_score = sum(
        (
            _gte(metrics["ma_deviation"], 0.03),
            _gte(metrics["rsi"], 55.0),
            _gte(metrics["macd_hist"], 0.0),
            _gte(metrics["price_change_5d"], 0.02),
            market_trend_score is not None and market_trend_score >= 1,
            auto_trend_score is not None and auto_trend_score >= 1,
            _gte(metrics["positive_news_flag"], 1.0),
            _gte(metrics["positive_news_shock"], 1.0),
            _gte(metrics["importance_weighted_sentiment"], 0.25),
            _gte(metrics["sector_news_strength"], 0.25),
            _gte(metrics["narrative_bullish_pressure"], 0.35),
        )
    )
    bear_score = sum(
        (
            _lte(metrics["ma_deviation"], -0.03),
            _lte(metrics["rsi"], 45.0),
            _lte(metrics["macd_hist"], 0.0),
            _lte(metrics["price_change_5d"], -0.02),
            market_trend_score is not None and market_trend_score <= -1,
            auto_trend_score is not None and auto_trend_score <= -1,
            _gte(metrics["negative_news_flag"], 1.0),
            _gte(metrics["negative_news_shock"], 1.0),
            _lte(metrics["importance_weighted_sentiment"], -0.25),
            _lte(metrics["sector_news_strength"], -0.25),
            _gte(metrics["narrative_bearish_pressure"], 0.35),
        )
    )

    if bull_score >= 3:
        return MarketRegime(
            name=REGIME_BULL,
            level="Trend",
            reasons=_trend_reasons(metrics, direction="bull"),
            features=_features(
                metrics,
                "ma_deviation",
                "rsi",
                "macd_hist",
                "price_change_5d",
                "market_trend_score",
                "auto_trend_score",
                "sector_strength",
                "earnings_after_trend_5",
                "avg_sentiment_3d",
                "positive_news_flag",
                "positive_news_shock",
                "importance_weighted_sentiment",
                "sector_news_strength",
                "narrative_bullish_pressure",
            ),
            metrics=metrics,
        )

    if bear_score >= 3:
        return MarketRegime(
            name=REGIME_BEAR,
            level="Trend",
            reasons=_trend_reasons(metrics, direction="bear"),
            features=_features(
                metrics,
                "ma_deviation",
                "rsi",
                "macd_hist",
                "price_change_5d",
                "market_trend_score",
                "auto_trend_score",
                "sector_strength",
                "earnings_after_trend_5",
                "avg_sentiment_3d",
                "negative_news_flag",
                "negative_news_shock",
                "importance_weighted_sentiment",
                "sector_news_strength",
                "narrative_bearish_pressure",
            ),
            metrics=metrics,
        )

    return MarketRegime(
        name=REGIME_SIDEWAYS,
        level="Neutral",
        reasons=_sideways_reasons(metrics),
        features=_features(
            metrics,
            "ma_deviation",
            "rsi",
            "volatility",
            "volume_ratio",
            "earnings_week_flag",
            "post_earnings_mode",
            "news_count_3d",
            "avg_sentiment_3d",
            "sentiment_trend_3d",
            "importance_weighted_sentiment",
            "dominant_narrative_code",
        ),
        metrics=metrics,
    )


def _is_panic(metrics: dict[str, float | None]) -> bool:
    return (
        _lte(metrics["price_change_1d"], -0.05)
        or _lte(metrics["n225_return"], -0.04)
        or _lte(metrics["topix_return"], -0.04)
        or _lte(metrics["sector_mean_return"], -0.04)
        or (
            _lte(metrics["price_change_5d"], -0.10)
            and (_gte(metrics["volume_ratio"], 1.5) or _gte(metrics["volatility_ratio"], 1.5))
        )
        or (
            _lte(metrics["rsi"], 30.0)
            and _lte(metrics["macd_hist"], 0.0)
            and _gte(metrics["volume_ratio"], 1.5)
            and _gte(metrics["volatility_ratio"], 1.4)
        )
        or (
            _gte(metrics["negative_news_flag"], 1.0)
            and _gte(metrics["max_importance_3d"], 0.85)
            and (_lte(metrics["price_change_1d"], -0.02) or _lte(metrics["price_change_5d"], -0.05))
        )
        or (
            _gte(metrics["negative_news_shock"], 1.0)
            and (
                _lte(metrics["importance_weighted_sentiment"], -0.35)
                or _lte(metrics["sector_news_strength"], -0.35)
            )
        )
    )


def _is_high_volatility(metrics: dict[str, float | None]) -> bool:
    one_day = metrics["price_change_1d"]
    return (
        _gte(metrics["volatility_ratio"], 1.8)
        or _gte(metrics["atr_pct"], 0.04)
        or _gte(metrics["n225_volatility"], 0.03)
        or _gte(metrics["topix_volatility"], 0.03)
        or _gte(metrics["sector_volatility"], 0.035)
        or (_gte(metrics["earnings_week_flag"], 1.0) and _gte(metrics["earnings_pre_volatility"], 0.02))
        or (_gte(metrics["post_earnings_mode"], 1.0) and _gte(metrics["earnings_volume_spike"], 0.5))
        or (
            _gte(metrics["news_count_3d"], 3.0)
            and _gte(metrics["max_importance_3d"], 0.7)
        )
        or _gte(metrics["negative_news_shock"], 1.0)
        or _gte(metrics["positive_news_shock"], 1.0)
        or (
            _gte(metrics["volume_ratio"], 2.0)
            and one_day is not None
            and abs(one_day) >= 0.03
        )
    )


def _panic_reasons(metrics: dict[str, float | None]) -> list[str]:
    reasons: list[str] = []
    if _lte(metrics["price_change_1d"], -0.05):
        reasons.append("Daily return is below -5%.")
    if _lte(metrics["price_change_5d"], -0.10):
        reasons.append("Five-day price change is deeply negative.")
    if _lte(metrics["n225_return"], -0.04):
        reasons.append("Nikkei 225 return is sharply negative.")
    if _lte(metrics["topix_return"], -0.04):
        reasons.append("TOPIX return is sharply negative.")
    if _lte(metrics["sector_mean_return"], -0.04):
        reasons.append("Auto sector peer return is sharply negative.")
    if _gte(metrics["volume_ratio"], 1.5):
        reasons.append("Volume is elevated versus the recent average.")
    if _lte(metrics["rsi"], 30.0):
        reasons.append("RSI is in a stressed low range.")
    if _gte(metrics["negative_news_flag"], 1.0) and _gte(metrics["max_importance_3d"], 0.85):
        reasons.append("High-importance negative news is present.")
    if _gte(metrics["negative_news_shock"], 1.0):
        reasons.append("Negative news shock is active.")
    if not reasons:
        reasons.append("Multiple stress signals are elevated.")
    return reasons


def _high_volatility_reasons(metrics: dict[str, float | None]) -> list[str]:
    reasons: list[str] = []
    if _gte(metrics["volatility_ratio"], 1.8):
        reasons.append("Volatility is high versus the recent baseline.")
    if _gte(metrics["atr_pct"], 0.04):
        reasons.append("ATR is large relative to price.")
    if _gte(metrics["n225_volatility"], 0.03):
        reasons.append("Nikkei 225 volatility is elevated.")
    if _gte(metrics["topix_volatility"], 0.03):
        reasons.append("TOPIX volatility is elevated.")
    if _gte(metrics["sector_volatility"], 0.035):
        reasons.append("Auto sector volatility is elevated.")
    if _gte(metrics["earnings_week_flag"], 1.0) and _gte(metrics["earnings_pre_volatility"], 0.02):
        reasons.append("Earnings are near and pre-event volatility is elevated.")
    if _gte(metrics["post_earnings_mode"], 1.0) and _gte(metrics["earnings_volume_spike"], 0.5):
        reasons.append("Post-earnings mode has elevated volume.")
    if _gte(metrics["news_count_3d"], 3.0) and _gte(metrics["max_importance_3d"], 0.7):
        reasons.append("Recent news flow is active and includes important items.")
    if _gte(metrics["negative_news_shock"], 1.0):
        reasons.append("Important negative news shock may raise volatility.")
    if _gte(metrics["positive_news_shock"], 1.0):
        reasons.append("Important positive news shock may raise volatility.")
    if _gte(metrics["volume_ratio"], 2.0):
        reasons.append("Volume is unusually high.")
    if not reasons:
        reasons.append("Volatility-related indicators are elevated.")
    return reasons


def _trend_reasons(metrics: dict[str, float | None], direction: str) -> list[str]:
    if direction == "bull":
        checks = (
            (_gte(metrics["ma_deviation"], 0.03), "Price is above the 25-day moving average."),
            (_gte(metrics["rsi"], 55.0), "RSI is in a positive range."),
            (_gte(metrics["macd_hist"], 0.0), "MACD histogram is positive."),
            (_gte(metrics["price_change_5d"], 0.02), "Five-day price change is positive."),
            (
                _gte(metrics["market_trend_score"], 1.0),
                "Nikkei 225 and TOPIX trend context is positive.",
            ),
            (
                _gte(metrics["auto_trend_score"], 1.0),
                "Auto sector peer trend context is positive.",
            ),
            (_gte(metrics["sector_strength"], 0.01), "Auto sector strength is positive."),
            (
                _gte(metrics["post_earnings_mode"], 1.0)
                and _gte(metrics["earnings_after_trend_5"], 0.01),
                "Post-earnings trend is positive.",
            ),
            (
                _gte(metrics["positive_news_flag"], 1.0),
                "Recent manual news flow is positive.",
            ),
            (
                _gte(metrics["importance_weighted_sentiment"], 0.25),
                "Importance-weighted news sentiment is positive.",
            ),
            (
                _gte(metrics["sector_news_strength"], 0.25),
                "Auto-sector news context is positive.",
            ),
            (
                _gte(metrics["narrative_bullish_pressure"], 0.35),
                "Market narrative pressure is bullish.",
            ),
        )
    else:
        checks = (
            (_lte(metrics["ma_deviation"], -0.03), "Price is below the 25-day moving average."),
            (_lte(metrics["rsi"], 45.0), "RSI is in a weak range."),
            (_lte(metrics["macd_hist"], 0.0), "MACD histogram is negative."),
            (_lte(metrics["price_change_5d"], -0.02), "Five-day price change is negative."),
            (
                _lte(metrics["market_trend_score"], -1.0),
                "Nikkei 225 and TOPIX trend context is negative.",
            ),
            (
                _lte(metrics["auto_trend_score"], -1.0),
                "Auto sector peer trend context is negative.",
            ),
            (_lte(metrics["sector_strength"], -0.01), "Auto sector strength is negative."),
            (
                _gte(metrics["post_earnings_mode"], 1.0)
                and _lte(metrics["earnings_after_trend_5"], -0.01),
                "Post-earnings trend is negative.",
            ),
            (
                _gte(metrics["negative_news_flag"], 1.0),
                "Recent manual news flow is negative.",
            ),
            (
                _lte(metrics["importance_weighted_sentiment"], -0.25),
                "Importance-weighted news sentiment is negative.",
            ),
            (
                _lte(metrics["sector_news_strength"], -0.25),
                "Auto-sector news context is negative.",
            ),
            (
                _gte(metrics["narrative_bearish_pressure"], 0.35),
                "Market narrative pressure is bearish.",
            ),
        )
    return [message for matched, message in checks if matched]


def _sideways_reasons(metrics: dict[str, float | None]) -> list[str]:
    reasons = ["Trend and stress signals are mixed or weak."]
    if _gte(metrics["earnings_week_flag"], 1.0):
        reasons.append("Earnings week is active, so event-specific moves may dominate.")
    if _gte(metrics["post_earnings_mode"], 1.0):
        reasons.append("Post-earnings mode is active.")
    if _gte(metrics["news_count_3d"], 1.0):
        reasons.append("Recent manual news exists, but trend signals remain mixed.")
    return reasons


def _market_trend_score(metrics: dict[str, float | None]) -> int | None:
    values = [
        value
        for value in (metrics.get("n225_trend"), metrics.get("topix_trend"))
        if value is not None
    ]
    if not values:
        return None
    return int(sum(1 if value > 0 else -1 if value < 0 else 0 for value in values))


def _auto_trend_score(metrics: dict[str, float | None]) -> int | None:
    if metrics.get("sector_trend") is not None:
        return int(metrics["sector_trend"] or 0)

    values = [
        value
        for value in (
            metrics.get("honda_trend"),
            metrics.get("nissan_trend"),
            metrics.get("tesla_trend"),
            metrics.get("byd_trend"),
        )
        if value is not None
    ]
    if not values:
        return None
    return int(sum(1 if value > 0 else -1 if value < 0 else 0 for value in values))


def _volume_ratio(data: pd.DataFrame) -> pd.Series:
    if "volume" not in data.columns:
        return pd.Series([pd.NA] * len(data), index=data.index)
    volume_average = _grouped_rolling_mean(data, "volume", window=20)
    return data["volume"] / volume_average


def _atr(data: pd.DataFrame, window: int) -> pd.Series:
    if not {"high", "low", "close"}.issubset(data.columns):
        return pd.Series([pd.NA] * len(data), index=data.index)

    previous_close = _grouped_shift(data, "close", periods=1)
    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    work = data.copy()
    work["_true_range"] = true_range
    return _grouped_rolling_mean(work, "_true_range", window=window)


def _grouped_rolling_mean(
    data: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
):
    min_periods = window if min_periods is None else min_periods
    if "ticker" in data.columns:
        return (
            data.groupby("ticker", group_keys=False)[column]
            .rolling(window=window, min_periods=min_periods)
            .mean()
            .reset_index(level=0, drop=True)
        )
    return data[column].rolling(window=window, min_periods=min_periods).mean()


def _grouped_rolling_std(
    data: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    if "ticker" in data.columns:
        return (
            data.groupby("ticker", group_keys=False)[column]
            .rolling(window=window, min_periods=min_periods)
            .std()
            .reset_index(level=0, drop=True)
        )
    return data[column].rolling(window=window, min_periods=min_periods).std()


def _grouped_rolling_median(
    data: pd.DataFrame,
    column: str,
    window: int,
    min_periods: int | None = None,
) -> pd.Series:
    min_periods = window if min_periods is None else min_periods
    if "ticker" in data.columns:
        return (
            data.groupby("ticker", group_keys=False)[column]
            .rolling(window=window, min_periods=min_periods)
            .median()
            .reset_index(level=0, drop=True)
        )
    return data[column].rolling(window=window, min_periods=min_periods).median()


def _grouped_shift(data: pd.DataFrame, column: str, periods: int) -> pd.Series:
    if "ticker" in data.columns:
        return data.groupby("ticker", group_keys=False)[column].shift(periods)
    return data[column].shift(periods)


def _grouped_pct_change(data: pd.DataFrame, column: str, periods: int = 1) -> pd.Series:
    shifted = _grouped_shift(data, column, periods=periods)
    return data[column] / shifted - 1


def _fallback_regime(reason: str) -> MarketRegime:
    return MarketRegime(
        name=REGIME_SIDEWAYS,
        level="Unknown",
        reasons=[reason],
        features=[],
        metrics={},
    )


def _features(metrics: dict[str, float | None], *names: str) -> list[str]:
    return [name for name in names if metrics.get(name) is not None]


def _value(row: pd.Series, key: str) -> float | None:
    return safe_float(row.get(key))


def _gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _lte(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold
