from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd

from stock_ai.features.technical import add_technical_features
from stock_ai.storage.parquet_store import read_parquet, write_parquet
from stock_ai.utils.safe_cast import safe_float


@dataclass(frozen=True)
class ExternalMarketSpec:
    symbol: str
    prefix: str
    name: str
    group: str = "market"
    fallback_symbols: tuple[str, ...] = ()
    is_foreign: bool = False


@dataclass(frozen=True)
class ExternalDownloadResult:
    spec: ExternalMarketSpec
    status: str
    output_path: Path | None
    rows: int
    latest_date: str | None
    used_symbol: str | None
    attempted_symbols: tuple[str, ...]
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalDownloadReport:
    results: tuple[ExternalDownloadResult, ...]

    @property
    def successful_paths(self) -> list[Path]:
        return [result.output_path for result in self.results if result.output_path is not None]

    @property
    def status(self) -> str:
        if not self.results:
            return "FAIL"
        successes = [result for result in self.results if result.status == "OK"]
        if not successes:
            return "FAIL"
        if len(successes) != len(self.results):
            return "WARN"
        if any(result.errors for result in successes):
            return "WARN"
        return "OK"


@dataclass(frozen=True)
class ExternalMarketContext:
    usdjpy_status: str
    market_trend: str
    auto_sector_trend: str
    tesla_trend: str
    honda_trend: str
    summary: str
    metrics: dict[str, float | None]
    sox_trend: str = "Unavailable"
    nvda_trend: str = "Unavailable"
    tnx_trend: str = "Unavailable"
    vix_trend: str = "Unavailable"


CORE_MARKET_SPECS = (
    ExternalMarketSpec(symbol="USDJPY=X", prefix="usdjpy", name="USDJPY", is_foreign=True),
    ExternalMarketSpec(symbol="^N225", prefix="n225", name="Nikkei 225", is_foreign=False),
    ExternalMarketSpec(
        symbol="^TOPX",
        prefix="topix",
        name="TOPIX",
        fallback_symbols=("^TPX", "1306.T"),
        is_foreign=False,
    ),
)
MACRO_MARKET_SPECS = (
    ExternalMarketSpec(symbol="^SOX", prefix="sox", name="SOX Semiconductor", group="macro", is_foreign=True),
    ExternalMarketSpec(symbol="NVDA", prefix="nvda", name="NVIDIA", group="macro", is_foreign=True),
    ExternalMarketSpec(symbol="^TNX", prefix="tnx", name="US 10Y Yield", group="macro", is_foreign=True),
    ExternalMarketSpec(symbol="^VIX", prefix="vix", name="VIX Volatility Index", group="macro", is_foreign=True),
)
AUTO_SECTOR_SPECS = (
    ExternalMarketSpec(symbol="7267.T", prefix="honda", name="Honda", group="auto_sector", is_foreign=False),
    ExternalMarketSpec(symbol="7201.T", prefix="nissan", name="Nissan", group="auto_sector", is_foreign=False),
    ExternalMarketSpec(symbol="TSLA", prefix="tesla", name="Tesla", group="auto_sector", is_foreign=True),
    ExternalMarketSpec(symbol="1211.HK", prefix="byd", name="BYD", group="auto_sector", is_foreign=True),
)
DEFAULT_EXTERNAL_MARKETS = (*CORE_MARKET_SPECS, *MACRO_MARKET_SPECS, *AUTO_SECTOR_SPECS)
EXTERNAL_DATA_SUBDIR = "external"
MIN_EXTERNAL_ROWS = 100


def download_external_market_data(
    raw_data_dir: str | Path,
    start: str,
    end: str | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Download configured external market series into data/raw/external."""
    report = download_external_market_data_report(
        raw_data_dir=raw_data_dir,
        start=start,
        end=end,
        symbols=symbols,
    )
    return report.successful_paths


def download_external_market_data_report(
    raw_data_dir: str | Path,
    start: str,
    end: str | None = None,
    symbols: list[str] | tuple[str, ...] | None = None,
) -> ExternalDownloadReport:
    """Download configured external market series and return per-symbol diagnostics."""
    from stock_ai.data.yfinance_provider import fetch_daily_ohlcv

    results: list[ExternalDownloadResult] = []
    for spec in specs_for_symbols(symbols):
        results.append(
            _download_external_spec(
                raw_data_dir=raw_data_dir,
                spec=spec,
                start=start,
                end=end,
                fetch_daily_ohlcv=fetch_daily_ohlcv,
            )
        )
    report = ExternalDownloadReport(tuple(results))
    print(f"download-external overall status: {report.status}")
    return report


def _download_external_spec(
    *,
    raw_data_dir: str | Path,
    spec: ExternalMarketSpec,
    start: str,
    end: str | None,
    fetch_daily_ohlcv,
) -> ExternalDownloadResult:
    attempted_symbols: list[str] = []
    errors: list[str] = []
    symbols_to_try = (spec.symbol, *spec.fallback_symbols)

    for candidate_symbol in symbols_to_try:
        attempted_symbols.append(candidate_symbol)
        fallback_label = " fallback" if candidate_symbol != spec.symbol else ""
        print(f"[START] download external {candidate_symbol}{fallback_label} ({spec.name})")
        try:
            data = fetch_daily_ohlcv(candidate_symbol, start=start, end=end)
        except Exception as error:
            message = f"{candidate_symbol}: {error}"
            errors.append(message)
            print(f"[FAIL] {candidate_symbol} error={error}")
            continue

        rows = int(len(data))
        latest_date = _latest_date_string(data)
        if rows < MIN_EXTERNAL_ROWS:
            message = (
                f"{candidate_symbol}: only {rows} rows returned "
                f"(latest_date={latest_date}); minimum is {MIN_EXTERNAL_ROWS}"
            )
            errors.append(message)
            print(f"[FAIL] {candidate_symbol} error={message}")
            continue

        path = external_data_path(raw_data_dir, spec)
        output_path = write_parquet(data, path)
        print(
            f"[OK] {candidate_symbol} rows={rows} latest_date={latest_date} "
            f"saved={output_path}"
        )
        if candidate_symbol != spec.symbol:
            print(
                f"[WARN] {spec.symbol} was unavailable; "
                f"using fallback {candidate_symbol} for {spec.name}."
            )
        return ExternalDownloadResult(
            spec=spec,
            status="OK",
            output_path=output_path,
            rows=rows,
            latest_date=latest_date,
            used_symbol=candidate_symbol,
            attempted_symbols=tuple(attempted_symbols),
            errors=tuple(errors),
        )

    print(f"[FAIL] {spec.symbol} all attempts failed: {'; '.join(errors)}")
    return ExternalDownloadResult(
        spec=spec,
        status="FAIL",
        output_path=None,
        rows=0,
        latest_date=None,
        used_symbol=None,
        attempted_symbols=tuple(attempted_symbols),
        errors=tuple(errors),
    )


def is_foreign_market_spec(spec: ExternalMarketSpec) -> bool:
    """Check if market spec represents non-Tokyo hours requiring shift(1) at 15:30 close."""
    if getattr(spec, "is_foreign", False):
        return True
    return not (spec.symbol.endswith(".T") or spec.symbol in ("^N225", "^TOPX", "^TPX", "1306.T"))


def add_external_market_features(
    stock_data: pd.DataFrame,
    raw_data_dir: str | Path,
    specs: tuple[ExternalMarketSpec, ...] | None = None,
    lag_foreign: bool = True,
) -> pd.DataFrame:
    """Merge available external market features into a stock feature frame.

    When lag_foreign=True (the default for 15:30 JST close prediction), all overseas
    series (TSLA, BYD, USDJPY) are shifted by 1 trading session (shift(1)) so that
    uncompleted overnight moves cannot leak into the 15:30 close features.
    """
    feature_frames: list[pd.DataFrame] = []
    feature_frames_by_prefix: dict[str, pd.DataFrame] = {}
    for spec in specs or DEFAULT_EXTERNAL_MARKETS:
        path = external_data_path(raw_data_dir, spec)
        if not path.exists():
            continue
        raw = read_parquet(path)
        feature_frame = build_external_market_feature_frame(raw, spec, lag_foreign=lag_foreign)
        feature_frames.append(feature_frame)
        feature_frames_by_prefix[spec.prefix] = feature_frame

    auto_sector_frame = build_auto_sector_feature_frame(feature_frames_by_prefix)
    if not auto_sector_frame.empty:
        feature_frames.append(auto_sector_frame)

    return merge_external_feature_frames(stock_data, feature_frames)


def build_external_market_feature_frame(
    data: pd.DataFrame,
    spec: ExternalMarketSpec,
    lag_foreign: bool = True,
) -> pd.DataFrame:
    """Create past-and-current-day external features for one market series.

    If lag_foreign is True and spec is foreign/US/overnight, the external features
    are shifted by 1 trading session (shift(1)) to prevent Look-Ahead Bias at 15:30 JST.
    """
    if data.empty:
        return pd.DataFrame(columns=["date"])

    work = data.copy()
    work["ticker"] = spec.symbol
    technical = add_technical_features(
        work,
        moving_average_windows=(5, 25),
        rsi_window=14,
        volatility_window=20,
        bollinger_window=20,
    )
    technical[f"{spec.prefix}_trend_direction"] = _trend_direction(technical)

    columns = {
        "close": f"{spec.prefix}_close",
        "return_1d": f"{spec.prefix}_return_1d",
        "ma_5": f"{spec.prefix}_ma_5",
        "ma_25": f"{spec.prefix}_ma_25",
        "close_vs_ma_25": f"{spec.prefix}_close_vs_ma_25",
        "rsi_14": f"{spec.prefix}_rsi_14",
        "macd": f"{spec.prefix}_macd",
        "macd_hist": f"{spec.prefix}_macd_hist",
        "volatility_20": f"{spec.prefix}_volatility_20",
        f"{spec.prefix}_trend_direction": f"{spec.prefix}_trend_direction",
    }
    available = ["date", *[column for column in columns if column in technical.columns]]
    result = technical[available].rename(columns=columns)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values("date").reset_index(drop=True)

    if lag_foreign and is_foreign_market_spec(spec):
        feature_cols = [c for c in result.columns if c != "date"]
        result[feature_cols] = result[feature_cols].shift(1)

    return result


def add_premarket_overnight_gap_features(
    stock_data: pd.DataFrame,
    raw_data_dir: str | Path,
    specs: tuple[ExternalMarketSpec, ...] | None = None,
) -> pd.DataFrame:
    """Build leak-free 07:30 JST pre-market gap and intraday reversal features.

    At 07:30 JST, the previous night's US market (dated t-1 relative to Tokyo date t)
    is completely finalized (closed at 05:00 JST).
    This function computes the gap shock, expected opening gap, and intraday
    reversal / momentum indicators for the upcoming 09:00 Open to 15:30 Close session.
    """
    result = stock_data.copy()
    if result.empty or "date" not in result.columns:
        return result

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values("date").reset_index(drop=True)

    overnight_frames: list[pd.DataFrame] = []
    for spec in specs or DEFAULT_EXTERNAL_MARKETS:
        path = external_data_path(raw_data_dir, spec)
        if not path.exists():
            continue
        raw = read_parquet(path)
        feat = build_external_market_feature_frame(raw, spec, lag_foreign=False)
        pre_cols = {c: f"premarket_{c}" for c in feat.columns if c != "date"}
        overnight_frames.append(feat.rename(columns=pre_cols))

    if overnight_frames:
        result = merge_external_feature_frames(result, overnight_frames)

    us_lead = (
        result["premarket_tesla_return_1d"]
        if "premarket_tesla_return_1d" in result.columns
        else pd.Series(0.0, index=result.index)
    )
    fx_lead = (
        result["premarket_usdjpy_return_1d"]
        if "premarket_usdjpy_return_1d" in result.columns
        else pd.Series(0.0, index=result.index)
    )
    dom_ret = result["return_1d"] if "return_1d" in result.columns else pd.Series(0.0, index=result.index)

    result["premarket_overnight_us_shock"] = us_lead.fillna(0.0)
    result["premarket_overnight_fx_shock"] = fx_lead.fillna(0.0)
    result["premarket_expected_gap"] = 0.60 * us_lead.fillna(0.0) + 0.40 * fx_lead.fillna(0.0)
    result["premarket_gap_divergence"] = result["premarket_expected_gap"] - dom_ret.fillna(0.0)
    result["intraday_gap_reversal_pressure"] = np.where(
        result["premarket_expected_gap"].abs() > 0.015,
        -np.sign(result["premarket_expected_gap"]),
        0.0,
    )

    if "open" in result.columns and "close" in result.columns:
        result["intraday_open_to_close_return"] = (result["close"] - result["open"]) / (result["open"] + 1e-8)
        result["target_open_to_close_binary"] = (result["close"] > result["open"]).astype(int)

    return result


def build_auto_sector_feature_frame(
    feature_frames_by_prefix: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate competitor features into a lightweight auto sector context."""
    sector_prefixes = [spec.prefix for spec in AUTO_SECTOR_SPECS]
    frames = [
        feature_frames_by_prefix[prefix]
        for prefix in sector_prefixes
        if prefix in feature_frames_by_prefix and not feature_frames_by_prefix[prefix].empty
    ]
    if not frames:
        return pd.DataFrame(columns=["date"])

    aggregate = pd.DataFrame({"date": pd.concat([frame["date"] for frame in frames]).dropna().unique()})
    aggregate["date"] = pd.to_datetime(aggregate["date"], errors="coerce")
    aggregate = aggregate.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    for prefix in sector_prefixes:
        frame = feature_frames_by_prefix.get(prefix)
        if frame is None or frame.empty:
            continue
        selected = frame[
            [
                column
                for column in (
                    "date",
                    f"{prefix}_return_1d",
                    f"{prefix}_volatility_20",
                    f"{prefix}_trend_direction",
                    f"{prefix}_close_vs_ma_25",
                )
                if column in frame.columns
            ]
        ].copy()
        aggregate = aggregate.merge(selected, on="date", how="left")

    return_columns = [f"{prefix}_return_1d" for prefix in sector_prefixes if f"{prefix}_return_1d" in aggregate]
    volatility_columns = [
        f"{prefix}_volatility_20"
        for prefix in sector_prefixes
        if f"{prefix}_volatility_20" in aggregate
    ]
    trend_columns = [
        f"{prefix}_trend_direction"
        for prefix in sector_prefixes
        if f"{prefix}_trend_direction" in aggregate
    ]
    strength_columns = [
        f"{prefix}_close_vs_ma_25"
        for prefix in sector_prefixes
        if f"{prefix}_close_vs_ma_25" in aggregate
    ]

    aggregate["sector_mean_return"] = _row_mean(aggregate, return_columns)
    aggregate["sector_volatility"] = _row_mean(aggregate, volatility_columns)
    aggregate["sector_trend_score"] = _row_mean(aggregate, trend_columns)
    aggregate["sector_trend"] = _trend_from_score(aggregate["sector_trend_score"])
    aggregate["sector_strength"] = _row_mean(aggregate, strength_columns)

    return aggregate[
        [
            "date",
            "sector_mean_return",
            "sector_volatility",
            "sector_trend",
            "sector_trend_score",
            "sector_strength",
        ]
    ]


def merge_external_feature_frames(
    stock_data: pd.DataFrame,
    feature_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    """Left join external features by date and forward-fill only past values."""
    result = stock_data.copy()
    if result.empty or "date" not in result.columns:
        return result

    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.sort_values("date").reset_index(drop=True)

    for frame in feature_frames:
        if frame.empty or "date" not in frame.columns:
            continue
        external = frame.copy()
        external["date"] = pd.to_datetime(external["date"], errors="coerce")
        external = external.dropna(subset=["date"]).sort_values("date")
        external = external.drop_duplicates(subset=["date"], keep="last")
        external_columns = [column for column in external.columns if column != "date"]
        if not external_columns:
            continue
        result = result.drop(columns=[c for c in external_columns if c in result.columns])
        result = result.merge(external, on="date", how="left")
        result[external_columns] = result[external_columns].ffill()

    return result


def build_external_market_context(data: pd.DataFrame) -> ExternalMarketContext:
    """Summarize latest external market conditions for UI and reasoning."""
    if data.empty:
        return ExternalMarketContext(
            usdjpy_status="Unavailable",
            market_trend="Unavailable",
            auto_sector_trend="Unavailable",
            tesla_trend="Unavailable",
            honda_trend="Unavailable",
            summary="External market data is not available yet.",
            metrics={},
        )

    latest = data.sort_values("date").iloc[-1] if "date" in data.columns else data.iloc[-1]
    metrics = {
        "usdjpy_return_1d": _value(latest, "usdjpy_return_1d"),
        "usdjpy_close_vs_ma_25": _value(latest, "usdjpy_close_vs_ma_25"),
        "usdjpy_trend_direction": _value(latest, "usdjpy_trend_direction"),
        "n225_return_1d": _value(latest, "n225_return_1d"),
        "n225_trend_direction": _value(latest, "n225_trend_direction"),
        "topix_return_1d": _value(latest, "topix_return_1d"),
        "topix_trend_direction": _value(latest, "topix_trend_direction"),
        "sox_return_1d": _value(latest, "sox_return_1d"),
        "sox_trend_direction": _value(latest, "sox_trend_direction"),
        "nvda_return_1d": _value(latest, "nvda_return_1d"),
        "nvda_trend_direction": _value(latest, "nvda_trend_direction"),
        "tnx_return_1d": _value(latest, "tnx_return_1d"),
        "tnx_trend_direction": _value(latest, "tnx_trend_direction"),
        "vix_return_1d": _value(latest, "vix_return_1d"),
        "vix_trend_direction": _value(latest, "vix_trend_direction"),
        "honda_return_1d": _value(latest, "honda_return_1d"),
        "honda_trend_direction": _value(latest, "honda_trend_direction"),
        "nissan_return_1d": _value(latest, "nissan_return_1d"),
        "nissan_trend_direction": _value(latest, "nissan_trend_direction"),
        "tesla_return_1d": _value(latest, "tesla_return_1d"),
        "tesla_trend_direction": _value(latest, "tesla_trend_direction"),
        "byd_return_1d": _value(latest, "byd_return_1d"),
        "byd_trend_direction": _value(latest, "byd_trend_direction"),
        "sector_mean_return": _value(latest, "sector_mean_return"),
        "sector_volatility": _value(latest, "sector_volatility"),
        "sector_trend": _value(latest, "sector_trend"),
        "sector_strength": _value(latest, "sector_strength"),
    }

    usdjpy_status = _usdjpy_status(metrics)
    market_trend = _market_trend(metrics)
    auto_sector_trend = _auto_sector_trend(metrics)
    tesla_trend = _single_trend(metrics.get("tesla_trend_direction"))
    honda_trend = _single_trend(metrics.get("honda_trend_direction"))
    sox_trend = _single_trend(metrics.get("sox_trend_direction"))
    nvda_trend = _single_trend(metrics.get("nvda_trend_direction"))
    tnx_trend = _single_trend(metrics.get("tnx_trend_direction"))
    vix_trend = _single_trend(metrics.get("vix_trend_direction"))
    summary = _market_summary(
        usdjpy_status=usdjpy_status,
        market_trend=market_trend,
        auto_sector_trend=auto_sector_trend,
        tesla_trend=tesla_trend,
    )
    return ExternalMarketContext(
        usdjpy_status=usdjpy_status,
        market_trend=market_trend,
        auto_sector_trend=auto_sector_trend,
        tesla_trend=tesla_trend,
        honda_trend=honda_trend,
        summary=summary,
        metrics=metrics,
        sox_trend=sox_trend,
        nvda_trend=nvda_trend,
        tnx_trend=tnx_trend,
        vix_trend=vix_trend,
    )


def external_data_path(raw_data_dir: str | Path, spec: ExternalMarketSpec) -> Path:
    return Path(raw_data_dir) / EXTERNAL_DATA_SUBDIR / f"{spec.prefix}.parquet"


def specs_for_symbols(
    symbols: list[str] | tuple[str, ...] | None,
) -> tuple[ExternalMarketSpec, ...]:
    if not symbols:
        return DEFAULT_EXTERNAL_MARKETS

    known = {
        value: spec
        for spec in DEFAULT_EXTERNAL_MARKETS
        for value in (spec.symbol, spec.prefix, spec.name, *spec.fallback_symbols)
    }
    specs: list[ExternalMarketSpec] = []
    for symbol in symbols:
        clean = symbol.strip()
        if not clean:
            continue
        specs.append(known.get(clean, ExternalMarketSpec(clean, _safe_prefix(clean), clean)))
    return tuple(specs)


def _latest_date_string(data: pd.DataFrame) -> str | None:
    if data.empty or "date" not in data.columns:
        return None
    dates = pd.to_datetime(data["date"], errors="coerce").dropna()
    if dates.empty:
        return None
    return pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


def external_prefixes() -> tuple[str, ...]:
    return tuple(spec.prefix for spec in DEFAULT_EXTERNAL_MARKETS)


def _trend_direction(data: pd.DataFrame) -> pd.Series:
    trend = pd.Series(0, index=data.index, dtype="int64")
    ma_deviation = _numeric_series(data, "close_vs_ma_25")
    macd_hist = _numeric_series(data, "macd_hist")
    trend[(ma_deviation > 0) & (macd_hist >= 0)] = 1
    trend[(ma_deviation < 0) & (macd_hist <= 0)] = -1
    return trend


def _usdjpy_status(metrics: dict[str, float | None]) -> str:
    trend = metrics.get("usdjpy_trend_direction")
    ma_deviation = metrics.get("usdjpy_close_vs_ma_25")
    if trend is None and ma_deviation is None:
        return "Unavailable"
    if (trend is not None and trend > 0) or (ma_deviation is not None and ma_deviation > 0.01):
        return "Yen weakening"
    if (trend is not None and trend < 0) or (ma_deviation is not None and ma_deviation < -0.01):
        return "Yen strengthening"
    return "Neutral"


def _market_trend(metrics: dict[str, float | None]) -> str:
    values = [
        value
        for value in (metrics.get("n225_trend_direction"), metrics.get("topix_trend_direction"))
        if value is not None
    ]
    if not values:
        return "Unavailable"
    score = sum(values)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def _auto_sector_trend(metrics: dict[str, float | None]) -> str:
    sector_trend = metrics.get("sector_trend")
    if sector_trend is not None:
        return _single_trend(sector_trend)

    values = [
        value
        for value in (
            metrics.get("honda_trend_direction"),
            metrics.get("nissan_trend_direction"),
            metrics.get("tesla_trend_direction"),
            metrics.get("byd_trend_direction"),
        )
        if value is not None
    ]
    if not values:
        return "Unavailable"
    score = sum(values)
    if score > 0:
        return "Positive"
    if score < 0:
        return "Negative"
    return "Neutral"


def _single_trend(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    if value > 0:
        return "Positive"
    if value < 0:
        return "Negative"
    return "Neutral"


def _market_summary(
    usdjpy_status: str,
    market_trend: str,
    auto_sector_trend: str,
    tesla_trend: str,
) -> str:
    parts: list[str] = []
    if usdjpy_status == "Yen weakening":
        parts.append("USDJPY shows yen weakening, which can be supportive for exporters.")
    elif usdjpy_status == "Yen strengthening":
        parts.append("USDJPY shows yen strengthening, which can pressure exporters.")
    elif usdjpy_status == "Neutral":
        parts.append("USDJPY is neutral versus its recent trend.")

    if market_trend == "Positive":
        parts.append("Nikkei 225 and TOPIX trend signals are broadly positive.")
    elif market_trend == "Negative":
        parts.append("Nikkei 225 and TOPIX trend signals are broadly negative.")
    elif market_trend == "Neutral":
        parts.append("Broad Japanese market trend signals are mixed.")

    if auto_sector_trend == "Positive":
        parts.append("Auto sector peer signals are broadly strong.")
    elif auto_sector_trend == "Negative":
        parts.append("Auto sector peer signals are broadly weak.")
    elif auto_sector_trend == "Neutral":
        parts.append("Auto sector peer signals are mixed.")

    if tesla_trend == "Negative":
        parts.append("Tesla trend is negative, which may reflect global EV/auto pressure.")
    elif tesla_trend == "Positive":
        parts.append("Tesla trend is positive, which may support global auto sentiment.")

    return " ".join(parts) if parts else "External market data is not available yet."


def _value(row: pd.Series, key: str) -> float | None:
    return safe_float(row.get(key))


def _numeric_series(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data.columns:
        return pd.Series(pd.NA, index=data.index)
    return pd.to_numeric(data[column], errors="coerce")


def _row_mean(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(pd.NA, index=data.index)
    return data[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1, skipna=True)


def _trend_from_score(score: pd.Series) -> pd.Series:
    result = pd.Series(0, index=score.index, dtype="int64")
    result[score > 0.25] = 1
    result[score < -0.25] = -1
    return result


def _safe_prefix(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", symbol).strip("_").lower()
    return cleaned or "external"
