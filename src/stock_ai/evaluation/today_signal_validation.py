from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import logging
from pathlib import Path

import pandas as pd

from stock_ai.config import PROJECT_ROOT
from stock_ai.evaluation import risk_validation
from stock_ai.utils.safe_cast import safe_float, safe_int

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - PyYAML is listed in requirements.
    yaml = None


LOGGER = logging.getLogger(__name__)


SIGNAL_ORDER = ["Bullish", "Bearish", "Neutral", "Caution", "High Risk", "Watch Only"]
SIGNAL_IMPROVEMENT_COLUMNS = [
    "today_signal",
    "count",
    "issue",
    "suggestion",
    "config_keys",
    "priority",
]
TODAY_SIGNAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "today_signal.yaml"
TODAY_SIGNAL_EXPERIMENTS_PATH = PROJECT_ROOT / "configs" / "today_signal_experiments.yaml"


@dataclass(frozen=True)
class TodaySignalRuleConfig:
    bullish_up_probability_threshold: float = 0.58
    bearish_down_probability_threshold: float = 0.58
    high_crash_percentile_threshold: float = 0.90
    high_surge_percentile_threshold: float = 0.90
    caution_conflict_threshold: float = 0.0
    neutral_probability_gap_threshold: float = 0.08
    watch_only_both_risks_high_threshold: float = 0.90
    surge_score_threshold: float = 40.0
    crash_score_threshold: float = 40.0
    anomaly_caution_score_threshold: float = 50.0


DEFAULT_TODAY_SIGNAL_CONFIG = TodaySignalRuleConfig()


def load_today_signal_config(path: str | Path | None = None) -> TodaySignalRuleConfig:
    config, _status = _load_today_signal_config_with_status(path)
    return config


def describe_today_signal_config(path: str | Path | None = None) -> dict[str, object]:
    config, status = _load_today_signal_config_with_status(path)
    return {
        **status,
        "values": asdict(config),
    }


def today_signal_config_rows(config: TodaySignalRuleConfig | None = None) -> list[dict[str, object]]:
    source = config or load_today_signal_config()
    values = asdict(source)
    descriptions = {
        "bullish_up_probability_threshold": "Minimum up probability used for stronger Bullish confirmation.",
        "bearish_down_probability_threshold": "Minimum down probability used for Bearish / High Risk confirmation.",
        "high_crash_percentile_threshold": "Historical percentile used when judging Crash Risk as high.",
        "high_surge_percentile_threshold": "Historical percentile used when judging Surge Risk as high.",
        "caution_conflict_threshold": "Minimum probability gap for binary-up versus high-crash conflict.",
        "neutral_probability_gap_threshold": "Probability gap below this is treated as low-conviction Neutral.",
        "watch_only_both_risks_high_threshold": "Percentile threshold for Watch Only when both tail risks are high.",
        "surge_score_threshold": "Absolute Surge Risk Score threshold for Bullish fallback.",
        "crash_score_threshold": "Absolute Crash Risk Score threshold for Bearish fallback.",
        "anomaly_caution_score_threshold": "Anomaly Score threshold for Caution fallback.",
    }
    return [
        {
            "config_key": key,
            "value": value,
            "description": descriptions.get(key, ""),
        }
        for key, value in values.items()
    ]


def load_today_signal_experiments(
    path: str | Path | None = None,
    *,
    current_config: TodaySignalRuleConfig | None = None,
) -> dict[str, TodaySignalRuleConfig]:
    base = current_config or load_today_signal_config()
    candidates = _default_experiment_configs(base)
    config_path = Path(path) if path is not None else TODAY_SIGNAL_EXPERIMENTS_PATH
    if not config_path.exists() or yaml is None:
        return candidates
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        LOGGER.debug("Could not load Today Signal experiment config: %s", error, exc_info=True)
        return candidates
    raw_experiments = raw.get("experiments", raw) if isinstance(raw, dict) else {}
    if not isinstance(raw_experiments, dict):
        return candidates
    loaded: dict[str, TodaySignalRuleConfig] = {}
    for name, overrides in raw_experiments.items():
        if not isinstance(name, str):
            continue
        if not isinstance(overrides, dict):
            overrides = {}
        loaded[name] = _config_with_overrides(base, overrides)
    return loaded or candidates


def compare_today_signal_configs(
    binary_predictions: pd.DataFrame,
    risk_frame: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    bucket_mode: str | None = None,
    experiments: dict[str, TodaySignalRuleConfig] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    candidates = experiments or load_today_signal_experiments()
    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    baseline = candidates.get("current") or next(iter(candidates.values()))
    comparison_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    change_rows: list[dict[str, object]] = []

    for name, config in candidates.items():
        validation = build_today_signal_validation_frame(
            binary_predictions,
            risk_frame,
            data=data,
            bucket_mode=bucket_mode,
            config=config,
        )
        summary = summarize_today_signals(validation)
        suggestions = build_signal_improvement_suggestions(summary)
        comparison_rows.append(_config_experiment_row(name, summary, suggestions))
        distribution_rows.extend(_config_distribution_rows(name, summary))
        change_rows.extend(_config_change_rows(name, config, baseline))

    comparison = pd.DataFrame(comparison_rows)
    distribution = pd.DataFrame(distribution_rows)
    changes = pd.DataFrame(change_rows)
    recommendation = recommend_today_signal_config(comparison)
    return comparison, distribution, changes, recommendation


def recommend_today_signal_config(comparison: pd.DataFrame) -> dict[str, object]:
    if comparison.empty:
        return {
            "config_name": "",
            "reason": "No experiment rows are available.",
        }
    rows = comparison.copy()
    rows["_score"] = rows.apply(_experiment_score, axis=1)
    rows = rows.sort_values(["_score", "config_name"], ascending=[False, True])
    best = rows.iloc[0]
    return {
        "config_name": str(best.get("config_name") or ""),
        "score": safe_float(best.get("_score")),
        "reason": _experiment_recommendation_reason(best),
    }


def build_config_adoption_recommendations(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "config_name",
        "adoption_rating",
        "adoption_score",
        "adoption_reason",
        "validation_rows",
        "bullish_average_return_5d",
        "high_risk_average_return_5d",
        "high_risk_max_drawdown_20d",
        "neutral_share",
        "caution_watch_share",
        "improvement_suggestion_count",
        "high_priority_suggestions",
    ]
    if comparison.empty:
        return pd.DataFrame(columns=columns)

    rows = comparison.copy()
    rows["_score"] = rows.apply(_experiment_score, axis=1)
    max_score = safe_float(rows["_score"].max()) or 0.0
    results: list[dict[str, object]] = []
    for _index, row in rows.iterrows():
        rating, reason = _adoption_rating_and_reason(row, max_score=max_score)
        validation_rows = safe_int(row.get("validation_rows")) or 0
        caution_watch_count = safe_int(row.get("caution_watch_count")) or 0
        results.append(
            {
                "config_name": row.get("config_name"),
                "adoption_rating": rating,
                "adoption_score": safe_float(row.get("_score")),
                "adoption_reason": reason,
                "validation_rows": validation_rows,
                "bullish_average_return_5d": safe_float(row.get("bullish_average_return_5d")),
                "high_risk_average_return_5d": safe_float(
                    row.get("high_risk_average_return_5d")
                ),
                "high_risk_max_drawdown_20d": safe_float(row.get("high_risk_max_drawdown_20d")),
                "neutral_share": safe_float(row.get("neutral_share")),
                "caution_watch_share": _safe_divide(caution_watch_count, validation_rows),
                "improvement_suggestion_count": safe_int(row.get("improvement_suggestion_count"))
                or 0,
                "high_priority_suggestions": safe_int(row.get("high_priority_suggestions")) or 0,
            }
        )
    return pd.DataFrame(results, columns=columns).sort_values(
        ["adoption_score", "config_name"],
        ascending=[False, True],
    )


def build_today_signal(
    *,
    prediction_label: str,
    up_probability,
    down_probability,
    surge_score,
    crash_score,
    surge_percentile=None,
    crash_percentile=None,
    surge_bucket: str = "Unknown",
    crash_bucket: str = "Unknown",
    regime_name: str | None = None,
    anomaly_score=None,
    operations_status: str = "",
    config: TodaySignalRuleConfig | None = None,
) -> dict[str, str]:
    rule_config = config or load_today_signal_config()
    if str(operations_status).upper() == "FAIL":
        return {
            "signal": "Watch Only",
            "confidence": "Low",
            "reason": "System health is FAIL, so the daily read should not be used until Operations Snapshot is checked.",
        }

    direction = _binary_direction_label(prediction_label, up_probability, down_probability)
    up = safe_float(up_probability) or 0.0
    down = safe_float(down_probability) or 0.0
    probability_gap = abs(up - down)
    surge_percentile_value = safe_float(surge_percentile) or 0.0
    crash_percentile_value = safe_float(crash_percentile) or 0.0
    surge_score_value = safe_float(surge_score) or 0.0
    crash_score_value = safe_float(crash_score) or 0.0
    anomaly_score_value = safe_float(anomaly_score) or 0.0
    regime_label = regime_name or "Unknown regime"
    operations_note = _operations_note(operations_status)
    surge_high = _risk_is_high(
        surge_bucket,
        surge_percentile_value,
        rule_config.high_surge_percentile_threshold,
    )
    crash_high = _risk_is_high(
        crash_bucket,
        crash_percentile_value,
        rule_config.high_crash_percentile_threshold,
    )
    both_tail_risks_high = _both_tail_risks_high(
        surge_bucket=surge_bucket,
        crash_bucket=crash_bucket,
        surge_percentile=surge_percentile_value,
        crash_percentile=crash_percentile_value,
        threshold=rule_config.watch_only_both_risks_high_threshold,
    )

    if both_tail_risks_high:
        return {
            "signal": "Watch Only",
            "confidence": "Medium",
            "reason": (
                "Both Surge and Crash Risk are high relative to historical distribution, "
                f"so volatility risk is more important than choosing a single direction.{operations_note}"
            ),
        }
    if crash_high and direction == "down":
        confidence = (
            "High"
            if down >= rule_config.bearish_down_probability_threshold
            or crash_percentile_value >= rule_config.high_crash_percentile_threshold
            else "Medium"
        )
        return {
            "signal": "High Risk",
            "confidence": confidence,
            "reason": (
                f"Binary model leans down ({_format_percent(down)}), and Crash Risk is "
                f"{risk_percentile_position_text(crash_percentile_value)}. Caution has priority.{operations_note}"
            ),
        }
    if (
        crash_high
        and direction == "up"
        and probability_gap >= rule_config.caution_conflict_threshold
    ):
        return {
            "signal": "Caution",
            "confidence": "Medium",
            "reason": (
                f"Binary model leans up ({_format_percent(up)}), but Crash Risk is "
                f"{risk_percentile_position_text(crash_percentile_value)}. The signals conflict.{operations_note}"
            ),
        }
    if surge_high and direction == "up" and not crash_high:
        confidence = (
            "High"
            if up >= rule_config.bullish_up_probability_threshold
            and surge_percentile_value >= rule_config.high_surge_percentile_threshold
            else "Medium"
        )
        return {
            "signal": "Bullish",
            "confidence": confidence,
            "reason": (
                f"Binary model leans up ({_format_percent(up)}), while Surge Risk is "
                f"{risk_percentile_position_text(surge_percentile_value)} and Crash Risk is not high.{operations_note}"
            ),
        }
    if (
        direction == "down"
        and (
            down >= rule_config.bearish_down_probability_threshold
            or crash_score_value >= rule_config.crash_score_threshold
        )
    ):
        return {
            "signal": "Bearish",
            "confidence": (
                "Medium" if probability_gap >= rule_config.neutral_probability_gap_threshold else "Low"
            ),
            "reason": (
                f"Binary model leans down ({_format_percent(down)}). "
                f"Crash Risk is {crash_score_value:.0f}/100 under {regime_label}.{operations_note}"
            ),
        }
    if (
        direction == "up"
        and (
            up >= rule_config.bullish_up_probability_threshold
            or surge_score_value >= rule_config.surge_score_threshold
        )
    ):
        return {
            "signal": "Bullish",
            "confidence": (
                "Medium" if probability_gap >= rule_config.neutral_probability_gap_threshold else "Low"
            ),
            "reason": (
                f"Binary model leans up ({_format_percent(up)}). "
                f"Surge Risk is {surge_score_value:.0f}/100 under {regime_label}.{operations_note}"
            ),
        }
    if anomaly_score_value >= rule_config.anomaly_caution_score_threshold:
        return {
            "signal": "Caution",
            "confidence": "Medium",
            "reason": (
                f"Prediction confidence is not decisive, but anomaly score is {anomaly_score_value:.0f}/100. "
                f"Treat today as a special-condition day.{operations_note}"
            ),
        }
    return {
        "signal": "Neutral",
        "confidence": (
            "Low" if probability_gap < rule_config.neutral_probability_gap_threshold else "Medium"
        ),
        "reason": (
            "Binary probability and risk signals are not strong enough for a directional call. "
            f"Use the detailed tabs before acting.{operations_note}"
        ),
    }


def build_today_signal_validation_frame(
    binary_predictions: pd.DataFrame,
    risk_frame: pd.DataFrame,
    *,
    data: pd.DataFrame | None = None,
    bucket_mode: str | None = None,
    config: TodaySignalRuleConfig | None = None,
) -> pd.DataFrame:
    if binary_predictions.empty or risk_frame.empty:
        return pd.DataFrame()

    predictions = _prepare_predictions(binary_predictions)
    risk = risk_validation.apply_bucket_mode(risk_frame, bucket_mode).copy()
    risk["date"] = pd.to_datetime(risk["date"], errors="coerce")
    if "ticker" in risk.columns:
        risk["ticker"] = risk["ticker"].astype(str)

    columns = [
        "date",
        "ticker",
        "surge_risk_score",
        "crash_risk_score",
        "surge_risk_bucket",
        "crash_risk_bucket",
        "return_1d",
        "return_5d",
        "return_20d",
        "future_min_return_20d",
        "future_max_return_20d",
    ]
    merged = predictions.merge(
        risk[[column for column in columns if column in risk.columns]],
        on=["date", "ticker"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    context = _prepare_context(data)
    if not context.empty:
        context = context[
            [
                column
                for column in context.columns
                if column in {"date", "ticker"} or column not in merged.columns
            ]
        ]
        merged = merged.merge(context, on=["date", "ticker"], how="left")

    rows: list[dict[str, object]] = []
    rule_config = config or load_today_signal_config()
    for row in merged.itertuples(index=False):
        item = row._asdict()
        surge_score = item.get("surge_risk_score")
        crash_score = item.get("crash_risk_score")
        signal = build_today_signal(
            prediction_label=str(item.get("prediction_label") or "unknown"),
            up_probability=item.get("probability_up"),
            down_probability=item.get("probability_down"),
            surge_score=surge_score,
            crash_score=crash_score,
            surge_percentile=risk_validation.score_percentile(
                risk,
                risk_type="surge",
                score=surge_score,
            ),
            crash_percentile=risk_validation.score_percentile(
                risk,
                risk_type="crash",
                score=crash_score,
            ),
            surge_bucket=str(item.get("surge_risk_bucket") or "Unknown"),
            crash_bucket=str(item.get("crash_risk_bucket") or "Unknown"),
            regime_name=str(item.get("regime") or "Unknown regime"),
            anomaly_score=item.get("anomaly_score"),
            config=rule_config,
        )
        item["today_signal"] = signal["signal"]
        item["signal_confidence"] = signal["confidence"]
        item["signal_reason"] = signal["reason"]
        rows.append(item)

    result = pd.DataFrame(rows)
    return result.sort_values("date").reset_index(drop=True)


def summarize_today_signals(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "today_signal",
        "count",
        "average_return_1d",
        "average_return_5d",
        "average_return_20d",
        "up_rate_1d",
        "up_rate_5d",
        "max_drawdown_20d",
        "max_upside_20d",
    ]
    if frame.empty or "today_signal" not in frame.columns:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, object]] = []
    for signal in SIGNAL_ORDER:
        subset = frame[frame["today_signal"] == signal].copy()
        rows.append(
            {
                "today_signal": signal,
                "count": int(len(subset)),
                "average_return_1d": _mean(subset.get("return_1d")),
                "average_return_5d": _mean(subset.get("return_5d")),
                "average_return_20d": _mean(subset.get("return_20d")),
                "up_rate_1d": _positive_rate(subset.get("return_1d")),
                "up_rate_5d": _positive_rate(subset.get("return_5d")),
                "max_drawdown_20d": _minimum(subset.get("future_min_return_20d")),
                "max_upside_20d": _maximum(subset.get("future_max_return_20d")),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_today_signal_validation_messages(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["Today Signal validation is not available yet."]
    messages: list[str] = []
    high_risk = _summary_row(summary, "High Risk")
    bullish = _summary_row(summary, "Bullish")
    caution = _summary_row(summary, "Caution")
    watch = _summary_row(summary, "Watch Only")

    if high_risk is not None:
        count = safe_int(high_risk.get("count")) or 0
        if count > 0:
            messages.append(
                "High Risk validation: "
                f"{count} samples, average 5D return {_format_percent(high_risk.get('average_return_5d'))}, "
                f"max 20D drawdown {_format_percent(high_risk.get('max_drawdown_20d'))}."
            )
        else:
            messages.append("High Risk has no validation samples yet.")
    if bullish is not None:
        count = safe_int(bullish.get("count")) or 0
        if count > 0:
            messages.append(
                "Bullish validation: "
                f"{count} samples, average 5D return {_format_percent(bullish.get('average_return_5d'))}, "
                f"5D up rate {_format_percent(bullish.get('up_rate_5d'))}."
            )
        else:
            messages.append("Bullish has no validation samples yet.")
    for label, row in (("Caution", caution), ("Watch Only", watch)):
        if row is not None and (safe_int(row.get("count")) or 0) > 0:
            messages.append(
                f"{label} is best read as a risk-management state: "
                f"{safe_int(row.get('count')) or 0} samples, average 5D return {_format_percent(row.get('average_return_5d'))}."
            )
    return messages


def build_signal_improvement_suggestions(
    summary: pd.DataFrame,
    *,
    min_samples: int = 20,
) -> pd.DataFrame:
    """Build rule-based suggestions without changing Today Signal logic."""
    if summary.empty:
        return pd.DataFrame(columns=SIGNAL_IMPROVEMENT_COLUMNS)

    total_count = sum(safe_int(row.get("count")) or 0 for _index, row in summary.iterrows())
    rows: list[dict[str, object]] = []
    caution_share = _signal_share(summary, "Caution", total_count)
    watch_share = _signal_share(summary, "Watch Only", total_count)

    for signal in SIGNAL_ORDER:
        row = _summary_row(summary, signal)
        count = safe_int(row.get("count")) if row is not None else 0
        count = count or 0
        average_5d = safe_float(row.get("average_return_5d")) if row is not None else None
        up_rate_5d = safe_float(row.get("up_rate_5d")) if row is not None else None
        max_drawdown = safe_float(row.get("max_drawdown_20d")) if row is not None else None
        share = _safe_divide(count, total_count)

        issue, suggestion, priority = _signal_improvement_rule(
            signal=signal,
            count=count,
            share=share,
            average_5d=average_5d,
            up_rate_5d=up_rate_5d,
            max_drawdown=max_drawdown,
            caution_watch_share=(caution_share or 0.0) + (watch_share or 0.0),
            min_samples=min_samples,
        )
        rows.append(
            {
                "today_signal": signal,
                "count": count,
                "issue": issue,
                "suggestion": suggestion,
                "config_keys": _signal_improvement_config_keys(signal, issue),
                "priority": priority,
            }
        )
    return pd.DataFrame(rows, columns=SIGNAL_IMPROVEMENT_COLUMNS)


def current_signal_improvement_text(suggestions: pd.DataFrame, signal: str | None) -> str:
    if suggestions.empty or not signal:
        return ""
    row = _summary_row(suggestions, signal)
    if row is None:
        return ""
    issue = str(row.get("issue") or "")
    suggestion = str(row.get("suggestion") or "")
    priority = str(row.get("priority") or "Low")
    if not issue:
        return ""
    return f"Signal improvement note ({priority}): {issue} Suggested check: {suggestion}"


def current_signal_context_text(summary: pd.DataFrame, signal: str | None) -> str:
    if summary.empty or not signal:
        return "Run Evaluation > Today Signal Validation to attach historical context to this signal."
    row = _summary_row(summary, signal)
    if row is None or (safe_int(row.get("count")) or 0) == 0:
        return f"{signal} has no validation samples yet, so confidence is limited."
    return (
        f"Historical {signal} validation: {safe_int(row.get('count')) or 0} samples, "
        f"average 5D return {_format_percent(row.get('average_return_5d'))}, "
        f"5D up rate {_format_percent(row.get('up_rate_5d'))}."
    )


def risk_percentile_position_text(percentile) -> str:
    value = safe_float(percentile)
    if value is None:
        return "-"
    if value >= 0.95:
        return "Top 5%"
    if value >= 0.90:
        return "Top 10%"
    if value >= 0.85:
        return "Top 15%"
    if value >= 0.75:
        return "Top 25%"
    if value <= 0.10:
        return "Bottom 10%"
    if value <= 0.25:
        return "Bottom 25%"
    return "Middle range"


def _prepare_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = predictions.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    if "ticker" not in rows.columns:
        rows["ticker"] = ""
    rows["ticker"] = rows["ticker"].astype(str)
    if "probability_up" not in rows.columns:
        rows["probability_up"] = pd.NA
    rows["probability_up"] = rows["probability_up"].map(safe_float)
    if "probability_down" not in rows.columns:
        rows["probability_down"] = 1.0 - pd.to_numeric(rows["probability_up"], errors="coerce")
    rows["probability_down"] = rows["probability_down"].map(safe_float)
    rows["prediction_label"] = rows.apply(_prediction_label, axis=1)
    return rows.dropna(subset=["date"])


def _prepare_context(data: pd.DataFrame | None) -> pd.DataFrame:
    if data is None or data.empty or "date" not in data.columns:
        return pd.DataFrame()
    rows = data.copy()
    rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
    if "ticker" not in rows.columns:
        rows["ticker"] = ""
    rows["ticker"] = rows["ticker"].astype(str)
    columns = ["date", "ticker", "regime", "anomaly_score"]
    return rows[[column for column in columns if column in rows.columns]].dropna(subset=["date"])


def _prediction_label(row: pd.Series) -> str:
    if row.get("prediction_direction") in {"up", "down"}:
        return str(row.get("prediction_direction"))
    predicted_class = safe_int(row.get("predicted_class"))
    if predicted_class == 1:
        return "up"
    if predicted_class == 0:
        return "down"
    up = safe_float(row.get("probability_up"))
    down = safe_float(row.get("probability_down"))
    if up is None or down is None:
        return "unknown"
    return "up" if up >= down else "down"


def _binary_direction_label(prediction_label: str, up_probability, down_probability) -> str:
    if prediction_label in {"up", "down"}:
        return prediction_label
    up = safe_float(up_probability)
    down = safe_float(down_probability)
    if up is None or down is None:
        return "unknown"
    return "up" if up >= down else "down"


def _load_today_signal_config_with_status(
    path: str | Path | None = None,
) -> tuple[TodaySignalRuleConfig, dict[str, object]]:
    config_path = Path(path) if path is not None else TODAY_SIGNAL_CONFIG_PATH
    status = {
        "path": str(config_path),
        "status": "default",
        "message": "Using built-in safe defaults.",
    }
    if not config_path.exists():
        status["message"] = "Config file is missing; using built-in safe defaults."
        return DEFAULT_TODAY_SIGNAL_CONFIG, status
    if yaml is None:
        status["status"] = "fallback"
        status["message"] = "PyYAML is unavailable; using built-in safe defaults."
        return DEFAULT_TODAY_SIGNAL_CONFIG, status

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as error:
        LOGGER.debug("Could not load Today Signal config: %s", error, exc_info=True)
        status["status"] = "fallback"
        status["message"] = f"Config could not be read; using defaults. error={error}"
        return DEFAULT_TODAY_SIGNAL_CONFIG, status

    if not isinstance(raw, dict):
        status["status"] = "fallback"
        status["message"] = "Config root must be a mapping; using built-in safe defaults."
        return DEFAULT_TODAY_SIGNAL_CONFIG, status

    raw_rules = raw.get("rules", raw)
    if not isinstance(raw_rules, dict):
        status["status"] = "fallback"
        status["message"] = "Config rules section must be a mapping; using built-in safe defaults."
        return DEFAULT_TODAY_SIGNAL_CONFIG, status

    default_values = asdict(DEFAULT_TODAY_SIGNAL_CONFIG)
    values = {
        key: _coerce_config_value(
            raw_rules.get(key),
            default=default,
            min_value=0.0,
            max_value=100.0 if key.endswith("_score_threshold") else 1.0,
        )
        for key, default in default_values.items()
    }
    status["status"] = "configured"
    status["message"] = "Config file loaded successfully."
    return TodaySignalRuleConfig(**values), status


def _coerce_config_value(
    value,
    *,
    default: float,
    min_value: float,
    max_value: float,
) -> float:
    number = safe_float(value)
    if number is None:
        return float(default)
    return max(min_value, min(max_value, float(number)))


def _default_experiment_configs(
    current_config: TodaySignalRuleConfig,
) -> dict[str, TodaySignalRuleConfig]:
    return {
        "current": current_config,
        "conservative": _config_with_overrides(
            current_config,
            {
                "bullish_up_probability_threshold": 0.62,
                "bearish_down_probability_threshold": 0.62,
                "high_crash_percentile_threshold": 0.93,
                "high_surge_percentile_threshold": 0.93,
                "neutral_probability_gap_threshold": 0.10,
                "watch_only_both_risks_high_threshold": 0.93,
                "surge_score_threshold": 45.0,
                "crash_score_threshold": 45.0,
            },
        ),
        "aggressive": _config_with_overrides(
            current_config,
            {
                "bullish_up_probability_threshold": 0.54,
                "bearish_down_probability_threshold": 0.54,
                "high_crash_percentile_threshold": 0.85,
                "high_surge_percentile_threshold": 0.85,
                "neutral_probability_gap_threshold": 0.05,
                "watch_only_both_risks_high_threshold": 0.85,
                "surge_score_threshold": 35.0,
                "crash_score_threshold": 35.0,
            },
        ),
        "risk_averse": _config_with_overrides(
            current_config,
            {
                "bearish_down_probability_threshold": 0.55,
                "high_crash_percentile_threshold": 0.85,
                "high_surge_percentile_threshold": 0.92,
                "watch_only_both_risks_high_threshold": 0.88,
                "crash_score_threshold": 35.0,
                "anomaly_caution_score_threshold": 45.0,
            },
        ),
        "bullish_sensitive": _config_with_overrides(
            current_config,
            {
                "bullish_up_probability_threshold": 0.54,
                "high_surge_percentile_threshold": 0.85,
                "surge_score_threshold": 35.0,
                "neutral_probability_gap_threshold": 0.06,
            },
        ),
    }


def _config_with_overrides(
    base: TodaySignalRuleConfig,
    overrides: dict[str, object],
) -> TodaySignalRuleConfig:
    base_values = asdict(base)
    values = {}
    for key, default in base_values.items():
        values[key] = _coerce_config_value(
            overrides.get(key),
            default=float(default),
            min_value=0.0,
            max_value=100.0 if key.endswith("_score_threshold") else 1.0,
        )
    return replace(base, **values)


def _config_experiment_row(
    name: str,
    summary: pd.DataFrame,
    suggestions: pd.DataFrame,
) -> dict[str, object]:
    total_count = int(sum(safe_int(row.get("count")) or 0 for _index, row in summary.iterrows()))
    bullish = _summary_row(summary, "Bullish")
    high_risk = _summary_row(summary, "High Risk")
    caution = _summary_row(summary, "Caution")
    watch = _summary_row(summary, "Watch Only")
    neutral = _summary_row(summary, "Neutral")
    caution_count = safe_int(caution.get("count")) if caution is not None else 0
    watch_count = safe_int(watch.get("count")) if watch is not None else 0
    neutral_count = safe_int(neutral.get("count")) if neutral is not None else 0
    high_priority = 0
    medium_priority = 0
    if not suggestions.empty and "priority" in suggestions.columns:
        high_priority = int((suggestions["priority"] == "High").sum())
        medium_priority = int((suggestions["priority"] == "Medium").sum())
    return {
        "config_name": name,
        "validation_rows": total_count,
        "bullish_count": safe_int(bullish.get("count")) if bullish is not None else 0,
        "bullish_average_return_5d": (
            safe_float(bullish.get("average_return_5d")) if bullish is not None else None
        ),
        "bullish_up_rate_5d": safe_float(bullish.get("up_rate_5d")) if bullish is not None else None,
        "high_risk_count": safe_int(high_risk.get("count")) if high_risk is not None else 0,
        "high_risk_average_return_5d": (
            safe_float(high_risk.get("average_return_5d")) if high_risk is not None else None
        ),
        "high_risk_max_drawdown_20d": (
            safe_float(high_risk.get("max_drawdown_20d")) if high_risk is not None else None
        ),
        "caution_count": caution_count or 0,
        "watch_only_count": watch_count or 0,
        "caution_watch_count": (caution_count or 0) + (watch_count or 0),
        "neutral_count": neutral_count or 0,
        "neutral_share": _safe_divide(neutral_count or 0, total_count),
        "high_priority_suggestions": high_priority,
        "medium_priority_suggestions": medium_priority,
        "improvement_suggestion_count": high_priority + medium_priority,
    }


def _config_distribution_rows(name: str, summary: pd.DataFrame) -> list[dict[str, object]]:
    total_count = int(sum(safe_int(row.get("count")) or 0 for _index, row in summary.iterrows()))
    rows: list[dict[str, object]] = []
    for signal in SIGNAL_ORDER:
        row = _summary_row(summary, signal)
        count = safe_int(row.get("count")) if row is not None else 0
        rows.append(
            {
                "config_name": name,
                "today_signal": signal,
                "count": count or 0,
                "share": _safe_divide(count or 0, total_count),
            }
        )
    return rows


def _config_change_rows(
    name: str,
    config: TodaySignalRuleConfig,
    baseline: TodaySignalRuleConfig,
) -> list[dict[str, object]]:
    config_values = asdict(config)
    baseline_values = asdict(baseline)
    rows = []
    for key, value in config_values.items():
        baseline_value = baseline_values.get(key)
        if value == baseline_value:
            continue
        rows.append(
            {
                "config_name": name,
                "config_key": key,
                "value": value,
                "current_value": baseline_value,
                "change": _config_change_text(value, baseline_value),
            }
        )
    return rows


def _config_change_text(value: object, baseline_value: object) -> str:
    number = safe_float(value)
    baseline = safe_float(baseline_value)
    if number is None or baseline is None:
        return "-"
    delta = number - baseline
    if abs(delta) < 1e-12:
        return "same"
    return f"{delta:+.3f}"


def _experiment_score(row: pd.Series) -> float:
    score = 0.0
    bullish_return = safe_float(row.get("bullish_average_return_5d"))
    high_risk_return = safe_float(row.get("high_risk_average_return_5d"))
    high_drawdown = safe_float(row.get("high_risk_max_drawdown_20d"))
    neutral_share = safe_float(row.get("neutral_share"))
    high_priority = safe_int(row.get("high_priority_suggestions")) or 0
    medium_priority = safe_int(row.get("medium_priority_suggestions")) or 0
    bullish_count = safe_int(row.get("bullish_count")) or 0
    high_risk_count = safe_int(row.get("high_risk_count")) or 0

    if bullish_count >= 10 and bullish_return is not None:
        score += min(3.0, max(-3.0, bullish_return * 100.0))
    elif bullish_count == 0:
        score -= 0.5
    if high_risk_count >= 10 and high_risk_return is not None:
        score += min(3.0, max(-3.0, -high_risk_return * 100.0))
    elif high_risk_count == 0:
        score -= 0.5
    if high_drawdown is not None and high_drawdown <= -0.02:
        score += 1.0
    if neutral_share is not None and neutral_share >= 0.50:
        score -= 1.0
    score -= high_priority * 0.75
    score -= medium_priority * 0.25
    return float(score)


def _experiment_recommendation_reason(row: pd.Series) -> str:
    config_name = str(row.get("config_name") or "-")
    bullish = safe_float(row.get("bullish_average_return_5d"))
    high_risk = safe_float(row.get("high_risk_average_return_5d"))
    neutral_share = safe_float(row.get("neutral_share"))
    high_issues = safe_int(row.get("high_priority_suggestions")) or 0
    parts = [f"{config_name} has the best current experiment score."]
    if bullish is not None:
        parts.append(f"Bullish 5D average return is {_format_percent(bullish)}.")
    if high_risk is not None:
        parts.append(f"High Risk 5D average return is {_format_percent(high_risk)}.")
    if neutral_share is not None:
        parts.append(f"Neutral share is {_format_percent(neutral_share)}.")
    if high_issues:
        parts.append(f"It still has {high_issues} high-priority improvement flags.")
    return " ".join(parts)


def _adoption_rating_and_reason(row: pd.Series, *, max_score: float) -> tuple[str, str]:
    config_name = str(row.get("config_name") or "-")
    validation_rows = safe_int(row.get("validation_rows")) or 0
    bullish_count = safe_int(row.get("bullish_count")) or 0
    high_risk_count = safe_int(row.get("high_risk_count")) or 0
    caution_watch_count = safe_int(row.get("caution_watch_count")) or 0
    neutral_share = safe_float(row.get("neutral_share"))
    bullish_return = safe_float(row.get("bullish_average_return_5d"))
    high_risk_return = safe_float(row.get("high_risk_average_return_5d"))
    high_drawdown = safe_float(row.get("high_risk_max_drawdown_20d"))
    high_issues = safe_int(row.get("high_priority_suggestions")) or 0
    improvement_count = safe_int(row.get("improvement_suggestion_count")) or 0
    score = safe_float(row.get("_score")) or 0.0
    caution_watch_share = _safe_divide(caution_watch_count, validation_rows)
    directional_count = bullish_count + high_risk_count

    if validation_rows < 30 or directional_count < 3:
        return (
            "Insufficient Samples",
            (
                f"{config_name} does not create enough validation samples for stable adoption. "
                "Keep it as a research candidate until more walk-forward rows accumulate."
            ),
        )
    if high_issues >= 2:
        return (
            "Not Recommended",
            (
                f"{config_name} has multiple high-priority improvement flags. "
                "Do not adopt it until the flagged signal behavior is reviewed."
            ),
        )
    if neutral_share is not None and neutral_share >= 0.60:
        return (
            "Too Conservative",
            (
                f"{config_name} leaves too many days as Neutral "
                f"({ _format_percent(neutral_share) }), which may hide useful signals."
            ),
        )
    if caution_watch_share is not None and caution_watch_share >= 0.45:
        return (
            "Too Conservative",
            (
                f"{config_name} sends too many days into Caution / Watch Only "
                f"({ _format_percent(caution_watch_share) }), so daily operations may become too defensive."
            ),
        )
    if neutral_share is not None and neutral_share <= 0.10 and improvement_count >= 2:
        return (
            "Too Aggressive",
            (
                f"{config_name} creates very few Neutral days and still has improvement flags. "
                "It may be forcing directional or risk states too often."
            ),
        )
    if bullish_return is not None and bullish_return <= 0 and high_risk_return is not None and high_risk_return >= 0:
        return (
            "Not Recommended",
            (
                f"{config_name} does not separate bullish and high-risk outcomes well: "
                f"Bullish 5D is {_format_percent(bullish_return)} and High Risk 5D is {_format_percent(high_risk_return)}."
            ),
        )
    if high_risk_count >= 10 and high_drawdown is not None and high_drawdown > -0.02:
        return (
            "Too Aggressive",
            (
                f"{config_name} flags High Risk, but the worst 20D drawdown after those days is only "
                f"{_format_percent(high_drawdown)}. Treat this as potentially over-sensitive."
            ),
        )
    if score >= max_score - 0.25 and high_issues == 0:
        return (
            "Recommended",
            (
                f"{config_name} has the strongest balance in this experiment: "
                f"Bullish 5D {_format_percent(bullish_return)}, "
                f"High Risk 5D {_format_percent(high_risk_return)}, "
                f"Neutral share {_format_percent(neutral_share)}."
            ),
        )
    return (
        "Candidate",
        (
            f"{config_name} is usable for comparison but is not the strongest current choice. "
            f"Bullish 5D {_format_percent(bullish_return)}, "
            f"High Risk 5D {_format_percent(high_risk_return)}, "
            f"improvement flags {improvement_count}."
        ),
    )


def _risk_is_high(bucket: str, percentile: float, threshold: float) -> bool:
    return str(bucket) == "High" or percentile >= threshold


def _both_tail_risks_high(
    *,
    surge_bucket: str,
    crash_bucket: str,
    surge_percentile: float,
    crash_percentile: float,
    threshold: float,
) -> bool:
    bucket_high = str(surge_bucket) == "High" and str(crash_bucket) == "High"
    percentile_high = surge_percentile >= threshold and crash_percentile >= threshold
    return bucket_high or percentile_high


def _operations_note(operations_status: str) -> str:
    if str(operations_status).upper() == "WARN":
        return " Operations health is WARN, so confirm the Operations Snapshot before relying on it."
    return ""


def _summary_row(summary: pd.DataFrame, signal: str) -> pd.Series | None:
    signal_column = "today_signal" if "today_signal" in summary.columns else "Today Signal"
    if summary.empty or signal_column not in summary.columns:
        return None
    rows = summary[summary[signal_column] == signal]
    if rows.empty:
        return None
    return rows.iloc[0]


def _signal_improvement_rule(
    *,
    signal: str,
    count: int,
    share: float | None,
    average_5d: float | None,
    up_rate_5d: float | None,
    max_drawdown: float | None,
    caution_watch_share: float,
    min_samples: int,
) -> tuple[str, str, str]:
    if count == 0:
        return (
            "No validation samples.",
            "Keep this signal under evaluation until more walk-forward rows exist.",
            "Low",
        )
    if count < min_samples:
        return (
            "Sample size is too small for a stable conclusion.",
            "Treat this signal as provisional; avoid changing thresholds from this sample alone.",
            "Low",
        )

    if signal == "Bullish":
        if average_5d is not None and average_5d <= 0:
            return (
                "Bullish 5D average return is weak or negative.",
                "Bullish conditions may be too loose; consider requiring higher up probability or lower Crash Risk.",
                "High",
            )
        if up_rate_5d is not None and up_rate_5d < 0.52:
            return (
                "Bullish 5D up rate is not clearly above neutral.",
                "Add confirmation from Similar Patterns, Regime, or Surge Risk percentile before showing Bullish.",
                "Medium",
            )
    if signal == "Bearish" and average_5d is not None and average_5d >= 0:
        return (
            "Bearish 5D average return is not negative.",
            "Bearish conditions may need stronger down probability or stronger Crash Risk confirmation.",
            "Medium",
        )
    if signal == "High Risk":
        if max_drawdown is not None and max_drawdown > -0.02:
            return (
                "High Risk did not precede meaningful 20D drawdowns.",
                "High Risk may be oversensitive; consider raising Crash Risk percentile or requiring bearish binary alignment.",
                "High",
            )
        if average_5d is not None and average_5d > 0:
            return (
                "High Risk average 5D return is positive.",
                "Check whether High Risk is detecting volatility rather than downside risk.",
                "Medium",
            )
    if signal == "Neutral" and share is not None and share >= 0.50:
        return (
            "Neutral dominates the validation set.",
            "The signal logic may be too conservative; consider reducing the neutral zone or adding context confirmations.",
            "Medium",
        )
    if signal in {"Caution", "Watch Only"}:
        if share is not None and share >= 0.25:
            return (
                f"{signal} appears frequently.",
                "Contradiction or volatility detection may be too broad; inspect Crash/Surge percentile thresholds.",
                "Medium",
            )
        if caution_watch_share >= 0.40:
            return (
                "Caution / Watch Only together occupy a large share.",
                "The logic may be flagging too many mixed states; review conflict thresholds.",
                "Medium",
            )

    return (
        "No major issue detected by the current MVP rules.",
        "Keep monitoring this signal as more validation rows accumulate.",
        "Low",
    )


def _signal_improvement_config_keys(signal: str, issue: str) -> str:
    if "Sample size" in issue or "No validation samples" in issue:
        return ""
    if signal == "Bullish":
        return "bullish_up_probability_threshold, high_surge_percentile_threshold"
    if signal == "Bearish":
        return "bearish_down_probability_threshold, high_crash_percentile_threshold"
    if signal == "High Risk":
        return "high_crash_percentile_threshold, bearish_down_probability_threshold"
    if signal == "Neutral":
        return "neutral_probability_gap_threshold"
    if signal == "Caution":
        return "caution_conflict_threshold, high_crash_percentile_threshold"
    if signal == "Watch Only":
        return (
            "watch_only_both_risks_high_threshold, "
            "high_crash_percentile_threshold, high_surge_percentile_threshold"
        )
    return ""


def _signal_share(summary: pd.DataFrame, signal: str, total_count: int) -> float | None:
    row = _summary_row(summary, signal)
    if row is None or total_count <= 0:
        return None
    count = safe_int(row.get("count")) or 0
    return _safe_divide(count, total_count)


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.mean())


def _positive_rate(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float((numeric > 0).mean())


def _minimum(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.min())


def _maximum(values) -> float | None:
    if values is None:
        return None
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"
