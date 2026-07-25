from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import pandas as pd

from stock_ai.features.technical import add_technical_features
from stock_ai.storage.parquet_store import read_parquet
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class LeadLagSpec:
    symbol: str
    name: str
    storage_prefix: str
    return_column: str


DEFAULT_LEAD_LAG_SPECS = (
    LeadLagSpec("7267.T", "Honda", "honda", "honda_return_1d"),
    LeadLagSpec("7201.T", "Nissan", "nissan", "nissan_return_1d"),
    LeadLagSpec("TSLA", "Tesla", "tesla", "tesla_return_1d"),
    LeadLagSpec("1211.HK", "BYD", "byd", "byd_return_1d"),
    LeadLagSpec("^N225", "Nikkei 225", "n225", "n225_return_1d"),
    LeadLagSpec("^TOPX", "TOPIX", "topix", "topix_return_1d"),
    LeadLagSpec("USDJPY=X", "USDJPY", "usdjpy", "usdjpy_return_1d"),
)
DEFAULT_LAGS = (1, 3, 5, 10)
MIN_SAMPLE_SIZE = 20


METRIC_COLUMNS = [
    "symbol",
    "name",
    "lag",
    "sample_size",
    "correlation",
    "direction_agreement",
    "large_move_threshold",
    "avg_return_after_large_move",
    "large_up_count",
    "toyota_avg_return_after_large_up",
    "toyota_up_probability_after_large_up",
    "large_down_count",
    "toyota_avg_return_after_large_down",
    "toyota_down_probability_after_large_down",
    "latest_candidate_return",
    "current_signal",
    "relationship_direction",
    "score",
]


def analyze_lead_lag_from_storage(
    target_data: pd.DataFrame,
    raw_data_dir: str | Path,
    feature_data_dir: str | Path | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    specs: tuple[LeadLagSpec, ...] = DEFAULT_LEAD_LAG_SPECS,
) -> pd.DataFrame:
    """Analyze candidate returns against Toyota future returns.

    Existing joined external columns are used first. Missing candidates are
    loaded from local parquet files under data/raw, data/features, or
    data/raw/external.
    """
    loaded_frames = load_lead_lag_candidate_frames(raw_data_dir, feature_data_dir, specs)
    return analyze_lead_lag(
        target_data=target_data,
        candidate_frames=loaded_frames,
        lags=lags,
        specs=specs,
    )


def load_lead_lag_candidate_frames(
    raw_data_dir: str | Path,
    feature_data_dir: str | Path | None = None,
    specs: tuple[LeadLagSpec, ...] = DEFAULT_LEAD_LAG_SPECS,
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    raw_dir = Path(raw_data_dir)
    feature_dir = Path(feature_data_dir) if feature_data_dir else None

    for spec in specs:
        paths = []
        if feature_dir is not None:
            paths.append(feature_dir / f"{spec.symbol}.parquet")
        paths.extend(
            [
                raw_dir / f"{spec.symbol}.parquet",
                raw_dir / "external" / f"{spec.storage_prefix}.parquet",
            ]
        )
        frame = _read_first_available(paths)
        if frame.empty:
            continue
        prepared = prepare_candidate_return_frame(frame, spec)
        if not prepared.empty:
            frames[spec.symbol] = prepared
    return frames


def analyze_lead_lag(
    target_data: pd.DataFrame,
    candidate_frames: dict[str, pd.DataFrame] | None = None,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    specs: tuple[LeadLagSpec, ...] = DEFAULT_LEAD_LAG_SPECS,
) -> pd.DataFrame:
    if not lags:
        return _empty_metrics()

    target = _prepare_target_frame(target_data, lags)
    if target.empty:
        return _empty_metrics()

    candidate_frames = candidate_frames or {}
    rows: list[dict[str, object]] = []
    for spec in specs:
        candidate = _candidate_from_target_columns(target_data, spec)
        if candidate.empty:
            candidate = candidate_frames.get(spec.symbol, pd.DataFrame())
        if candidate.empty:
            continue
        aligned = _align_candidate_to_target_dates(target, candidate)
        if aligned.empty:
            continue
        for lag in lags:
            row = _metric_row(spec, lag, aligned)
            if row is not None:
                rows.append(row)

    if not rows:
        return _empty_metrics()
    return pd.DataFrame(rows, columns=METRIC_COLUMNS).sort_values(
        ["score", "sample_size"],
        ascending=[False, False],
    ).reset_index(drop=True)


def prepare_candidate_return_frame(data: pd.DataFrame, spec: LeadLagSpec) -> pd.DataFrame:
    if data.empty or "date" not in data.columns:
        return pd.DataFrame(columns=["date", "candidate_return"])

    work = data.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(columns=["date", "candidate_return"])

    return_column = _find_return_column(work, spec)
    if return_column is None and {"open", "high", "low", "close", "volume"}.issubset(work.columns):
        try:
            work = add_technical_features(work)
        except ValueError:
            pass
        return_column = _find_return_column(work, spec)

    if return_column is None and "close" in work.columns:
        work["return_1d"] = pd.to_numeric(work["close"], errors="coerce").pct_change()
        return_column = "return_1d"

    if return_column is None:
        return pd.DataFrame(columns=["date", "candidate_return"])

    result = work[["date", return_column]].rename(columns={return_column: "candidate_return"})
    result["candidate_return"] = pd.to_numeric(result["candidate_return"], errors="coerce")
    result = result.replace([float("inf"), -float("inf")], pd.NA)
    return result.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")


def rank_lead_lag_signals(metrics: pd.DataFrame, top_n: int = 7) -> pd.DataFrame:
    if metrics.empty:
        return _empty_metrics()
    ranked = metrics.sort_values(["score", "sample_size"], ascending=[False, False]).copy()
    ranked = ranked.drop_duplicates(subset=["symbol"], keep="first")
    ranked = ranked.head(top_n).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def lead_lag_direction(ranking: pd.DataFrame | None) -> str:
    if ranking is None or ranking.empty:
        return "unknown"

    current = ranking[ranking["current_signal"].isin(["bullish", "bearish"])].copy()
    if current.empty:
        return "mixed"

    current["weight"] = pd.to_numeric(current["score"], errors="coerce").fillna(0.0)
    bullish = float(current.loc[current["current_signal"] == "bullish", "weight"].sum())
    bearish = float(current.loc[current["current_signal"] == "bearish", "weight"].sum())
    if bullish == 0 and bearish == 0:
        return "unknown"
    if bullish >= bearish * 1.2:
        return "up"
    if bearish >= bullish * 1.2:
        return "down"
    return "mixed"


def build_lead_lag_reason(metrics: pd.DataFrame, ranking: pd.DataFrame | None = None) -> str:
    if metrics.empty:
        return "Lead-lag signals are not available yet."
    ranked = ranking if ranking is not None and not ranking.empty else rank_lead_lag_signals(metrics)
    if ranked.empty:
        return "Lead-lag signals are not available yet."

    best = ranked.iloc[0]
    name = best.get("name") or best.get("symbol") or "candidate"
    lag = int(best.get("lag") or 0)
    correlation = _format_number(best.get("correlation"), digits=3)
    agreement = _format_percent(best.get("direction_agreement"))
    down_prob = _format_percent(best.get("toyota_down_probability_after_large_down"))
    up_prob = _format_percent(best.get("toyota_up_probability_after_large_up"))
    current_signal = str(best.get("current_signal") or "neutral")

    if current_signal == "bullish":
        current_text = "The latest move is a bullish lead-lag signal."
    elif current_signal == "bearish":
        current_text = "The latest move is a bearish lead-lag signal."
    else:
        current_text = "The latest move is not large enough to be treated as a strong signal."

    return (
        f"Best exploratory lead-lag relationship is {name} with a {lag}-day horizon. "
        f"Correlation is {correlation}, direction agreement is {agreement}. "
        f"After large {name} rises, Toyota rose {up_prob} of the time; "
        f"after large {name} drops, Toyota fell {down_prob} of the time. "
        f"{current_text}"
    )


def _prepare_target_frame(data: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame()

    target = data.copy()
    target["date"] = pd.to_datetime(target["date"], errors="coerce")
    target["close"] = pd.to_numeric(target["close"], errors="coerce")
    target = target.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    target = target.drop_duplicates(subset=["date"], keep="last")
    if target.empty:
        return pd.DataFrame()

    for lag in lags:
        target[f"toyota_future_return_{lag}d"] = target["close"].shift(-lag) / target["close"] - 1
    future_columns = [f"toyota_future_return_{lag}d" for lag in lags]
    return target[["date", "close", *future_columns]]


def _candidate_from_target_columns(target_data: pd.DataFrame, spec: LeadLagSpec) -> pd.DataFrame:
    if target_data.empty or "date" not in target_data.columns or spec.return_column not in target_data.columns:
        return pd.DataFrame(columns=["date", "candidate_return"])
    result = target_data[["date", spec.return_column]].rename(
        columns={spec.return_column: "candidate_return"}
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["candidate_return"] = pd.to_numeric(result["candidate_return"], errors="coerce")
    return result.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")


def _align_candidate_to_target_dates(target: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    if target.empty or candidate.empty:
        return pd.DataFrame()

    target_dates = target.sort_values("date").reset_index(drop=True)
    candidate_returns = candidate.copy()
    candidate_returns["date"] = pd.to_datetime(candidate_returns["date"], errors="coerce")
    candidate_returns["candidate_return"] = pd.to_numeric(
        candidate_returns["candidate_return"],
        errors="coerce",
    )
    candidate_returns = candidate_returns.dropna(subset=["date"]).sort_values("date")
    candidate_returns = candidate_returns.drop_duplicates(subset=["date"], keep="last")
    if candidate_returns.empty:
        return pd.DataFrame()

    aligned = pd.merge_asof(
        target_dates,
        candidate_returns[["date", "candidate_return"]],
        on="date",
        direction="backward",
        tolerance=pd.Timedelta(days=7),
    )
    return aligned


def _metric_row(spec: LeadLagSpec, lag: int, aligned: pd.DataFrame) -> dict[str, object] | None:
    future_column = f"toyota_future_return_{lag}d"
    if future_column not in aligned.columns:
        return None

    work = aligned[["date", "candidate_return", future_column]].copy()
    work["candidate_return"] = pd.to_numeric(work["candidate_return"], errors="coerce")
    work[future_column] = pd.to_numeric(work[future_column], errors="coerce")
    work = work.dropna(subset=["candidate_return", future_column])
    if len(work) < MIN_SAMPLE_SIZE:
        return None

    candidate = work["candidate_return"]
    toyota_future = work[future_column]
    correlation = _correlation(candidate, toyota_future)
    direction_agreement = _direction_agreement(candidate, toyota_future)
    threshold = _large_move_threshold(candidate)

    large = work[candidate.abs() >= threshold] if threshold is not None else work.iloc[0:0]
    large_up = work[candidate >= threshold] if threshold is not None else work.iloc[0:0]
    large_down = work[candidate <= -threshold] if threshold is not None else work.iloc[0:0]

    avg_large = _mean_or_none(large[future_column])
    avg_after_up = _mean_or_none(large_up[future_column])
    avg_after_down = _mean_or_none(large_down[future_column])
    up_probability = _positive_rate_or_none(large_up[future_column])
    down_probability = _negative_rate_or_none(large_down[future_column])
    latest_candidate_return = _latest_candidate_return(aligned)
    current_signal = _current_signal(
        latest_candidate_return,
        threshold,
        avg_after_up,
        up_probability,
        avg_after_down,
        down_probability,
    )
    relationship_direction = _relationship_direction(
        avg_after_up,
        up_probability,
        avg_after_down,
        down_probability,
    )

    return {
        "symbol": spec.symbol,
        "name": spec.name,
        "lag": int(lag),
        "sample_size": int(len(work)),
        "correlation": correlation,
        "direction_agreement": direction_agreement,
        "large_move_threshold": threshold,
        "avg_return_after_large_move": avg_large,
        "large_up_count": int(len(large_up)),
        "toyota_avg_return_after_large_up": avg_after_up,
        "toyota_up_probability_after_large_up": up_probability,
        "large_down_count": int(len(large_down)),
        "toyota_avg_return_after_large_down": avg_after_down,
        "toyota_down_probability_after_large_down": down_probability,
        "latest_candidate_return": latest_candidate_return,
        "current_signal": current_signal,
        "relationship_direction": relationship_direction,
        "score": _relationship_score(correlation, direction_agreement, avg_large, len(work)),
    }


def _find_return_column(data: pd.DataFrame, spec: LeadLagSpec) -> str | None:
    for column in (spec.return_column, "return_1d", f"{spec.storage_prefix}_return_1d"):
        if column in data.columns:
            return column
    return None


def _read_first_available(paths: list[Path]) -> pd.DataFrame:
    for path in paths:
        if path.exists():
            try:
                return read_parquet(path)
            except Exception:
                continue
    return pd.DataFrame()


def _correlation(left: pd.Series, right: pd.Series) -> float | None:
    if len(left) < 2 or left.std(ddof=0) == 0 or right.std(ddof=0) == 0:
        return None
    value = left.corr(right)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _direction_agreement(left: pd.Series, right: pd.Series) -> float | None:
    mask = (left != 0) & (right != 0)
    if not mask.any():
        return None
    return float(((left[mask] > 0) == (right[mask] > 0)).mean())


def _large_move_threshold(values: pd.Series) -> float | None:
    absolute = pd.to_numeric(values, errors="coerce").abs().dropna()
    if absolute.empty:
        return None
    threshold = float(absolute.quantile(0.75))
    if not math.isfinite(threshold):
        return None
    return max(0.005, threshold)


def _current_signal(
    latest_return: float | None,
    threshold: float | None,
    avg_after_up: float | None,
    up_probability: float | None,
    avg_after_down: float | None,
    down_probability: float | None,
) -> str:
    if latest_return is None or threshold is None or abs(latest_return) < threshold:
        return "neutral"
    if latest_return > 0:
        return _outcome_signal(avg_after_up, up_probability, positive_outcome=True)
    return _outcome_signal(avg_after_down, down_probability, positive_outcome=False)


def _outcome_signal(
    average_return: float | None,
    probability: float | None,
    positive_outcome: bool,
) -> str:
    if positive_outcome:
        if (average_return is not None and average_return >= 0.005) or (
            probability is not None and probability >= 0.55
        ):
            return "bullish"
        if (average_return is not None and average_return <= -0.005) or (
            probability is not None and probability <= 0.45
        ):
            return "bearish"
    else:
        if (average_return is not None and average_return <= -0.005) or (
            probability is not None and probability >= 0.55
        ):
            return "bearish"
        if (average_return is not None and average_return >= 0.005) or (
            probability is not None and probability <= 0.45
        ):
            return "bullish"
    return "neutral"


def _relationship_direction(
    avg_after_up: float | None,
    up_probability: float | None,
    avg_after_down: float | None,
    down_probability: float | None,
) -> str:
    bullish = 0
    bearish = 0
    if avg_after_up is not None and avg_after_up >= 0.005:
        bullish += 1
    if up_probability is not None and up_probability >= 0.55:
        bullish += 1
    if avg_after_down is not None and avg_after_down <= -0.005:
        bearish += 1
    if down_probability is not None and down_probability >= 0.55:
        bearish += 1
    if bullish > bearish:
        return "up"
    if bearish > bullish:
        return "down"
    if bullish or bearish:
        return "mixed"
    return "unknown"


def _relationship_score(
    correlation: float | None,
    direction_agreement: float | None,
    average_return_after_large_move: float | None,
    sample_size: int,
) -> float:
    corr_strength = abs(correlation or 0.0) * 50.0
    agreement_strength = abs((direction_agreement or 0.5) - 0.5) * 60.0
    large_move_strength = min(abs(average_return_after_large_move or 0.0) * 400.0, 25.0)
    sample_weight = min(1.0, sample_size / 120.0)
    return round((corr_strength + agreement_strength + large_move_strength) * sample_weight, 2)


def _latest_candidate_return(aligned: pd.DataFrame) -> float | None:
    series = pd.to_numeric(aligned.get("candidate_return"), errors="coerce").dropna()
    if series.empty:
        return None
    return safe_float(series.iloc[-1])


def _mean_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _positive_rate_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean > 0).mean())


def _negative_rate_or_none(values: pd.Series) -> float | None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float((clean < 0).mean())


def _empty_metrics() -> pd.DataFrame:
    return pd.DataFrame(columns=METRIC_COLUMNS)


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.1f}%"


def _format_number(value, digits: int = 2) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"
