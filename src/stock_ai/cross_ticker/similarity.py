from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd

from stock_ai.features.technical import add_technical_features
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class CrossTickerSpec:
    ticker: str
    name: str
    storage_prefix: str


CROSS_TICKER_SPECS = (
    CrossTickerSpec("7203.T", "Toyota", "7203.T"),
    CrossTickerSpec("7267.T", "Honda", "honda"),
    CrossTickerSpec("7201.T", "Nissan", "nissan"),
    CrossTickerSpec("TSLA", "Tesla", "tesla"),
    CrossTickerSpec("1211.HK", "BYD", "byd"),
)
CROSS_TICKER_FEATURES = (
    "return_1d",
    "volume_change_1d",
    "rsi_14",
    "macd",
    "volatility_20",
    "sector_strength",
    "narrative_strength",
)
FEATURE_LABELS = {
    "return_1d": "Daily return",
    "volume_change_1d": "Volume change",
    "rsi_14": "RSI",
    "macd": "MACD",
    "volatility_20": "Volatility",
    "sector_strength": "Sector strength",
    "narrative_strength": "Narrative strength",
}


def find_cross_ticker_similar_patterns_from_storage(
    target_data: pd.DataFrame,
    raw_data_dir: str | Path,
    feature_data_dir: str | Path,
    target_ticker: str = "7203.T",
    window_size: int = 20,
    top_n: int = 5,
    forward_days: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    frames = load_cross_ticker_frames(
        raw_data_dir=raw_data_dir,
        feature_data_dir=feature_data_dir,
    )
    return find_cross_ticker_similar_patterns(
        target_data=target_data,
        candidate_frames=frames,
        target_ticker=target_ticker,
        window_size=window_size,
        top_n=top_n,
        forward_days=forward_days,
    )


def load_cross_ticker_frames(
    raw_data_dir: str | Path,
    feature_data_dir: str | Path,
    specs: tuple[CrossTickerSpec, ...] = CROSS_TICKER_SPECS,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    raw_dir = Path(raw_data_dir)
    feature_dir = Path(feature_data_dir)
    for spec in specs:
        frame = _read_first_available(
            [
                feature_dir / f"{spec.ticker}.parquet",
                raw_dir / f"{spec.ticker}.parquet",
                raw_dir / "external" / f"{spec.storage_prefix}.parquet",
            ],
        )
        if frame.empty:
            continue
        frames[spec.ticker] = prepare_cross_ticker_frame(frame, spec.ticker)
    return frames


def find_cross_ticker_similar_patterns(
    target_data: pd.DataFrame,
    candidate_frames: dict[str, pd.DataFrame],
    target_ticker: str = "7203.T",
    window_size: int = 20,
    top_n: int = 5,
    forward_days: tuple[int, ...] = (1, 5, 20),
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    if window_size <= 1:
        raise ValueError("window_size must be greater than 1.")
    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if target_data.empty or not {"date", "close"}.issubset(target_data.columns):
        return _empty_result(forward_days)

    target = prepare_cross_ticker_frame(target_data, target_ticker)
    max_forward = max(forward_days)
    if len(target) < window_size:
        return _empty_result(forward_days)

    latest_window = target.tail(window_size).copy()
    rows: list[dict[str, object]] = []
    for ticker, frame in candidate_frames.items():
        candidate = prepare_cross_ticker_frame(frame, ticker)
        if len(candidate) < window_size + max_forward:
            continue
        features = _available_features(latest_window, candidate, feature_columns)
        if not features:
            continue
        latest_vector = _window_vector(latest_window, features)
        if _vector_norm(latest_vector) == 0:
            continue

        max_candidate_end = len(candidate) - max_forward - 1
        if ticker == target_ticker:
            max_candidate_end = min(max_candidate_end, len(candidate) - window_size - max_forward - 1)
        for candidate_end in range(window_size - 1, max_candidate_end + 1):
            candidate_start = candidate_end - window_size + 1
            candidate_window = candidate.iloc[candidate_start : candidate_end + 1]
            candidate_vector = _window_vector(candidate_window, features)
            similarity = _cosine_similarity(latest_vector, candidate_vector)
            if similarity is None:
                continue

            feature_scores = _feature_similarity_scores(latest_window, candidate_window, features)
            row: dict[str, object] = {
                "ticker": ticker,
                "ticker_name": _ticker_name(ticker),
                "similar_start_date": candidate.loc[candidate_start, "date"],
                "similar_end_date": candidate.loc[candidate_end, "date"],
                "similarity": similarity,
                "similar_features": ", ".join(str(item["feature"]) for item in feature_scores[:3]),
                "similar_feature_scores": feature_scores,
            }
            for days in forward_days:
                row[f"peer_return_{days}d_after"] = _forward_return(candidate, candidate_end, days)
                row[f"toyota_return_{days}d_after"] = _target_return_from_date(
                    target,
                    candidate.loc[candidate_end, "date"],
                    days,
                )
            rows.append(row)

    if not rows:
        return _empty_result(forward_days)
    return pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


def summarize_cross_ticker_patterns(
    patterns: pd.DataFrame,
    forward_days: tuple[int, ...] = (1, 5, 20),
) -> pd.DataFrame:
    columns = [
        "horizon",
        "count",
        "peer_average_return",
        "toyota_average_return",
        "peer_positive_rate",
        "toyota_positive_rate",
    ]
    if patterns.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for days in forward_days:
        peer = pd.to_numeric(patterns.get(f"peer_return_{days}d_after"), errors="coerce").dropna()
        toyota = pd.to_numeric(patterns.get(f"toyota_return_{days}d_after"), errors="coerce").dropna()
        rows.append(
            {
                "horizon": f"{days}D",
                "count": int(len(peer)),
                "peer_average_return": _mean_or_none(peer),
                "toyota_average_return": _mean_or_none(toyota),
                "peer_positive_rate": _positive_rate_or_none(peer),
                "toyota_positive_rate": _positive_rate_or_none(toyota),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_cross_ticker_reason(patterns: pd.DataFrame, summary: pd.DataFrame) -> str:
    if patterns.empty:
        return "Cross-ticker similar patterns are not available yet."
    best = patterns.iloc[0]
    ticker_name = best.get("ticker_name") or best.get("ticker") or "peer"
    start = pd.to_datetime(best.get("similar_start_date")).strftime("%Y-%m-%d")
    end = pd.to_datetime(best.get("similar_end_date")).strftime("%Y-%m-%d")
    similarity = _format_number(best.get("similarity"), digits=3)
    row_5d = _summary_row(summary, "5D")
    toyota_5d = _format_percent(row_5d.get("toyota_average_return")) if row_5d else "-"
    peer_5d = _format_percent(row_5d.get("peer_average_return")) if row_5d else "-"
    features = best.get("similar_features") or "unknown"
    return (
        f"Current Toyota pattern is closest to {ticker_name} from {start} to {end} "
        f"(similarity {similarity}). Similar features are {features}. "
        f"After similar peer windows, average 5D peer return was {peer_5d}; "
        f"Toyota's same-date 5D return averaged {toyota_5d}."
    )


def cross_ticker_direction(summary: pd.DataFrame) -> str:
    row = _summary_row(summary, "5D") or _summary_row(summary, "1D")
    if row is None:
        return "unknown"
    toyota_return = safe_float(row.get("toyota_average_return"))
    peer_return = safe_float(row.get("peer_average_return"))
    if toyota_return is None and peer_return is None:
        return "unknown"
    value = toyota_return if toyota_return is not None else peer_return
    if value is not None and value >= 0.02:
        return "up"
    if value is not None and value <= -0.02:
        return "down"
    return "mixed"


def prepare_cross_ticker_frame(data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    result = data.copy()
    if result.empty or "date" not in result.columns:
        return pd.DataFrame()
    if "ticker" not in result.columns:
        result["ticker"] = ticker
    else:
        result["ticker"] = result["ticker"].fillna(ticker)
    if _needs_technical_features(result):
        try:
            result = add_technical_features(result)
        except ValueError:
            pass
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if {"narrative_bullish_pressure", "narrative_bearish_pressure"}.issubset(result.columns):
        result["narrative_strength"] = (
            pd.to_numeric(result["narrative_bullish_pressure"], errors="coerce").fillna(0.0)
            - pd.to_numeric(result["narrative_bearish_pressure"], errors="coerce").fillna(0.0)
        )
    return result.replace([float("inf"), -float("inf")], pd.NA)


def _read_first_available(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            try:
                return read_parquet(path)
            except Exception:
                continue
    return pd.DataFrame()


def _needs_technical_features(data: pd.DataFrame) -> bool:
    required = {"return_1d", "volume_change_1d", "rsi_14", "macd", "volatility_20"}
    return not required.issubset(data.columns) and {"date", "open", "high", "low", "close", "volume"}.issubset(
        data.columns,
    )


def _available_features(
    target_window: pd.DataFrame,
    candidate: pd.DataFrame,
    feature_columns: tuple[str, ...] | list[str] | None,
) -> list[str]:
    features: list[str] = []
    for column in feature_columns or CROSS_TICKER_FEATURES:
        if column not in target_window.columns or column not in candidate.columns:
            continue
        left = pd.to_numeric(target_window[column], errors="coerce")
        right = pd.to_numeric(candidate[column], errors="coerce")
        if left.notna().any() and right.notna().any():
            features.append(column)
    return features


def _window_vector(window: pd.DataFrame, features: list[str]) -> list[float]:
    values: list[float] = []
    for feature in features:
        values.extend(_normalize_series(pd.to_numeric(window[feature], errors="coerce")))
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
    scores: list[dict[str, object]] = []
    for feature in features:
        left = _normalize_series(pd.to_numeric(latest_window[feature], errors="coerce"))
        right = _normalize_series(pd.to_numeric(candidate_window[feature], errors="coerce"))
        score = _cosine_similarity(left, right)
        if score is None:
            continue
        scores.append(
            {
                "feature": FEATURE_LABELS.get(feature, feature),
                "column": feature,
                "similarity": score,
            }
        )
    return sorted(scores, key=lambda item: float(item["similarity"]), reverse=True)


def _forward_return(data: pd.DataFrame, end_index: int, days: int) -> float | None:
    if end_index + days >= len(data):
        return None
    base = safe_float(data.loc[end_index, "close"])
    future = safe_float(data.loc[end_index + days, "close"])
    if base is None or future is None or base == 0:
        return None
    return float(future / base - 1)


def _target_return_from_date(target: pd.DataFrame, date, days: int) -> float | None:
    if target.empty:
        return None
    end = pd.to_datetime(date, errors="coerce")
    if pd.isna(end):
        return None
    matches = target.index[target["date"] >= end].tolist()
    if not matches:
        return None
    return _forward_return(target, int(matches[0]), days)


def _empty_result(forward_days: tuple[int, ...]) -> pd.DataFrame:
    columns = [
        "ticker",
        "ticker_name",
        "similar_start_date",
        "similar_end_date",
        "similarity",
        "similar_features",
        "similar_feature_scores",
    ]
    for days in forward_days:
        columns.append(f"peer_return_{days}d_after")
        columns.append(f"toyota_return_{days}d_after")
    return pd.DataFrame(columns=columns)


def _ticker_name(ticker: str) -> str:
    for spec in CROSS_TICKER_SPECS:
        if spec.ticker == ticker:
            return spec.name
    return ticker


def _mean_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float(values.mean())


def _positive_rate_or_none(values: pd.Series) -> float | None:
    if values.empty:
        return None
    return float((values > 0).mean())


def _summary_row(summary: pd.DataFrame, horizon: str) -> dict[str, object] | None:
    if summary.empty or "horizon" not in summary.columns:
        return None
    rows = summary[summary["horizon"] == horizon]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _format_number(value, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"
