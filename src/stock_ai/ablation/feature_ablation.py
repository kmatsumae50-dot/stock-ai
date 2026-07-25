from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ai.model_compare import compare as model_compare
from stock_ai.models.baseline import select_feature_columns
from stock_ai.utils.safe_cast import safe_float


TARGET_COLUMN = "target_next_close_binary"
FEATURE_GROUP_ORDER = (
    "technical",
    "external_market",
    "auto_sector",
    "earnings_event",
    "news",
    "narrative",
    "regime",
    "anomaly",
    "lead_lag",
)
FEATURE_GROUP_LABELS = {
    "technical": "Technical",
    "external_market": "External Market",
    "auto_sector": "Auto Sector",
    "earnings_event": "Earnings Event",
    "news": "News",
    "narrative": "Narrative",
    "regime": "Regime",
    "anomaly": "Anomaly",
    "lead_lag": "Lead-Lag",
}
COMBO_GROUPS = (
    "external_market",
    "auto_sector",
    "earnings_event",
    "news",
    "narrative",
    "regime",
    "anomaly",
    "lead_lag",
)
METRIC_COLUMNS = (
    "accuracy",
    "f1",
    "precision",
    "recall",
    "auc",
    "up_prediction_accuracy",
    "down_prediction_accuracy",
    "cumulative_return",
    "buy_hold_return",
    "max_drawdown",
    "win_rate",
    "trade_count",
)
ANOMALY_FEATURE_COLUMNS = {
    "price_shock_score",
    "volume_spike_score",
    "volatility_spike_score",
    "sector_divergence_score",
    "market_divergence_score",
    "news_shock_score",
    "narrative_shift_score",
    "anomaly_score",
}
EVENT_FEATURE_COLUMNS = {
    "days_to_earnings",
    "days_after_earnings",
    "post_earnings_mode",
}
NEWS_FEATURE_COLUMNS = {
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
}
REGIME_FLAG_COLUMNS = {
    "Bull Trend": "regime_bull_trend",
    "Bear Trend": "regime_bear_trend",
    "Sideways": "regime_sideways",
    "High Volatility": "regime_high_volatility",
    "Panic": "regime_panic",
}


@dataclass(frozen=True)
class AblationConfig:
    name: str
    label: str
    included_groups: tuple[str, ...]
    kind: str
    removed_group: str | None = None


@dataclass(frozen=True)
class AblationResult:
    predictions: pd.DataFrame
    score_table: pd.DataFrame
    group_impact: pd.DataFrame
    skipped_configs: pd.DataFrame
    feature_groups: dict[str, list[str]]
    configs: list[AblationConfig]
    split_summary: dict[str, object]
    summary_messages: list[str]


def run_feature_ablation(
    data: pd.DataFrame,
    ticker: str = "7203.T",
    model_name: str = "LightGBM",
    probability_threshold: float = 0.55,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    config_names: tuple[str, ...] | None = None,
) -> AblationResult:
    """Run lightweight feature-group ablation on one chronological split."""
    frame = prepare_ablation_frame(data, ticker=ticker)
    if frame.empty:
        return _empty_result()

    all_feature_columns = select_feature_columns(frame, target_column=TARGET_COLUMN)
    feature_groups = build_feature_groups(frame, all_feature_columns)
    configs = build_ablation_configs(feature_groups)
    if config_names:
        requested = set(config_names)
        configs = [config for config in configs if config.name in requested]
    if not configs:
        return _empty_result(feature_groups=feature_groups)

    score_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []
    split_summary: dict[str, object] = {}

    for config in configs:
        feature_columns = selected_feature_columns(feature_groups, config.included_groups)
        if not feature_columns:
            skipped_rows.append(
                {"config_name": config.name, "config_label": config.label, "reason": "No features."}
            )
            continue

        result = model_compare.run_model_comparison(
            frame,
            ticker=ticker,
            probability_threshold=probability_threshold,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            include_xgboost=model_name == "XGBoost",
            model_names=(model_name,),
            feature_columns_override=feature_columns,
        )
        if result.split_summary and not split_summary:
            split_summary = result.split_summary
        if result.score_table.empty:
            reason = _skipped_reason(result.skipped_models)
            skipped_rows.append(
                {"config_name": config.name, "config_label": config.label, "reason": reason}
            )
            continue

        score = result.score_table.iloc[0].to_dict()
        score.update(_config_metadata(config, feature_columns))
        score_rows.append(score)

        if not result.predictions.empty:
            predictions = result.predictions.copy()
            for key, value in _config_metadata(config, feature_columns).items():
                predictions[key] = value
            prediction_frames.append(predictions)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    score_table = add_ablation_deltas(pd.DataFrame(score_rows))
    group_impact = build_group_impact_table(score_table, feature_groups)
    return AblationResult(
        predictions=predictions,
        score_table=score_table,
        group_impact=group_impact,
        skipped_configs=pd.DataFrame(skipped_rows),
        feature_groups=feature_groups,
        configs=configs,
        split_summary=split_summary,
        summary_messages=build_ablation_summary(score_table, group_impact),
    )


def prepare_ablation_frame(data: pd.DataFrame, ticker: str = "7203.T") -> pd.DataFrame:
    frame = model_compare.prepare_model_compare_frame(data, ticker=ticker)
    if frame.empty:
        return frame
    return add_regime_feature_flags(frame)


def add_regime_feature_flags(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    if "regime" not in result.columns:
        return result

    regime = result["regime"].fillna("Unknown").astype(str)
    for label, column in REGIME_FLAG_COLUMNS.items():
        result[column] = (regime == label).astype(int)
    return result


def build_feature_groups(
    data: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, list[str]]:
    columns = feature_columns or select_feature_columns(data, target_column=TARGET_COLUMN)
    groups: dict[str, list[str]] = {name: [] for name in FEATURE_GROUP_ORDER}
    for column in columns:
        group = feature_group_for_column(column)
        groups[group].append(column)
    return {group: sorted(set(values)) for group, values in groups.items()}


def feature_group_for_column(column: str) -> str:
    name = column.lower()
    if _is_anomaly_feature(name):
        return "anomaly"
    if name.startswith(("lead_lag_", "lag_")) or "_lag_" in name:
        return "lead_lag"
    if name.startswith("regime_"):
        return "regime"
    if name.startswith(("narrative_", "dominant_narrative_")):
        return "narrative"
    if name in NEWS_FEATURE_COLUMNS or name.startswith("news_"):
        return "news"
    if name in EVENT_FEATURE_COLUMNS or name.startswith("earnings_"):
        return "earnings_event"
    if name.startswith(("usdjpy_", "n225_", "topix_")):
        return "external_market"
    if name.startswith(("honda_", "nissan_", "tesla_", "byd_", "sector_")):
        return "auto_sector"
    return "technical"


def build_ablation_configs(feature_groups: dict[str, list[str]]) -> list[AblationConfig]:
    available = tuple(group for group in FEATURE_GROUP_ORDER if feature_groups.get(group))
    if not available:
        return []

    configs: list[AblationConfig] = [
        AblationConfig("all_features", "All features", available, "all"),
    ]
    if feature_groups.get("technical"):
        configs.append(
            AblationConfig("technical_only", "Technical only", ("technical",), "baseline")
        )
        for group in COMBO_GROUPS:
            if feature_groups.get(group):
                configs.append(
                    AblationConfig(
                        f"technical_plus_{group}",
                        f"Technical + {FEATURE_GROUP_LABELS[group]}",
                        ("technical", group),
                        "add_one",
                    )
                )

    for group in available:
        remaining = tuple(item for item in available if item != group)
        if remaining:
            configs.append(
                AblationConfig(
                    f"without_{group}",
                    f"Without {FEATURE_GROUP_LABELS[group]}",
                    remaining,
                    "remove_one",
                    removed_group=group,
                )
            )
    return configs


def selected_feature_columns(
    feature_groups: dict[str, list[str]],
    included_groups: tuple[str, ...],
) -> list[str]:
    selected: list[str] = []
    for group in included_groups:
        selected.extend(feature_groups.get(group, []))
    return sorted(set(selected))


def add_ablation_deltas(score_table: pd.DataFrame) -> pd.DataFrame:
    if score_table.empty or "config_name" not in score_table.columns:
        return score_table

    result = score_table.copy()
    full = _first_config_row(result, "all_features")
    technical = _first_config_row(result, "technical_only")
    for metric in METRIC_COLUMNS:
        if metric not in result.columns:
            continue
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
        if full is not None:
            full_value = safe_float(full.get(metric))
            result[f"{metric}_delta_vs_full"] = (
                result[metric] - full_value if full_value is not None else None
            )
        if technical is not None:
            technical_value = safe_float(technical.get(metric))
            result[f"{metric}_delta_vs_technical"] = (
                result[metric] - technical_value if technical_value is not None else None
            )
    return result


def build_group_impact_table(
    score_table: pd.DataFrame,
    feature_groups: dict[str, list[str]],
) -> pd.DataFrame:
    if score_table.empty:
        return pd.DataFrame()

    full = _first_config_row(score_table, "all_features")
    technical = _first_config_row(score_table, "technical_only")
    rows: list[dict[str, object]] = []
    for group in FEATURE_GROUP_ORDER:
        without = _first_config_row(score_table, f"without_{group}")
        plus = _first_config_row(score_table, f"technical_plus_{group}")
        if without is None and plus is None:
            continue
        rows.append(
            {
                "group": group,
                "group_label": FEATURE_GROUP_LABELS[group],
                "feature_count": len(feature_groups.get(group, [])),
                "f1_drop_when_removed": _drop(full, without, "f1"),
                "auc_drop_when_removed": _drop(full, without, "auc"),
                "return_drop_when_removed": _drop(full, without, "cumulative_return"),
                "f1_gain_vs_technical": _gain(technical, plus, "f1"),
                "auc_gain_vs_technical": _gain(technical, plus, "auc"),
                "return_gain_vs_technical": _gain(technical, plus, "cumulative_return"),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["f1_drop_when_removed", "auc_drop_when_removed"],
        ascending=[False, False],
        na_position="last",
    )


def build_ablation_summary(
    score_table: pd.DataFrame,
    group_impact: pd.DataFrame,
) -> list[str]:
    if score_table.empty:
        return ["Ablation results are not available yet."]

    messages: list[str] = []
    if not group_impact.empty:
        helpful = group_impact.dropna(subset=["f1_drop_when_removed"]).copy()
        helpful = helpful[helpful["f1_drop_when_removed"] > 0]
        if not helpful.empty:
            row = helpful.sort_values("f1_drop_when_removed", ascending=False).iloc[0]
            messages.append(
                f"{row['group_label']} looks useful: removing it lowers F1 by "
                f"{_format_percent(row['f1_drop_when_removed'])}."
            )

        additive = group_impact.dropna(subset=["auc_gain_vs_technical"]).copy()
        additive = additive[additive["auc_gain_vs_technical"] > 0]
        if not additive.empty:
            row = additive.sort_values("auc_gain_vs_technical", ascending=False).iloc[0]
            messages.append(
                f"{row['group_label']} improves AUC over technical-only by "
                f"{_format_percent(row['auc_gain_vs_technical'])}."
            )

        return_helpful = group_impact.dropna(subset=["return_drop_when_removed"]).copy()
        return_helpful = return_helpful[return_helpful["return_drop_when_removed"] > 0]
        if not return_helpful.empty:
            row = return_helpful.sort_values("return_drop_when_removed", ascending=False).iloc[0]
            messages.append(
                f"{row['group_label']} contributes to simulated return: removing it changes return by "
                f"{_format_percent(row['return_drop_when_removed'])}."
            )

    best = _best_config(score_table, "f1")
    if best is not None:
        messages.append(
            f"Best F1 configuration is {best.get('config_label')} "
            f"({_format_percent(best.get('f1'))})."
        )

    return messages[:5] or ["No clear useful feature group is detected yet."]


def _is_anomaly_feature(name: str) -> bool:
    return (
        name in ANOMALY_FEATURE_COLUMNS
        or name.startswith("anomaly_")
        or name.endswith("_shock_score")
        or name.endswith("_spike_score")
        or name.endswith("_divergence_score")
    )


def _config_metadata(config: AblationConfig, feature_columns: list[str]) -> dict[str, object]:
    return {
        "config_name": config.name,
        "config_label": config.label,
        "config_kind": config.kind,
        "included_groups": ", ".join(config.included_groups),
        "removed_group": config.removed_group or "",
        "feature_count": int(len(feature_columns)),
    }


def _first_config_row(score_table: pd.DataFrame, config_name: str):
    if score_table.empty or "config_name" not in score_table.columns:
        return None
    rows = score_table[score_table["config_name"] == config_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def _best_config(score_table: pd.DataFrame, metric: str):
    if score_table.empty or metric not in score_table.columns:
        return None
    rows = score_table.copy()
    rows[metric] = pd.to_numeric(rows[metric], errors="coerce")
    rows = rows.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if rows.empty:
        return None
    return rows.iloc[0]


def _drop(full, without, metric: str) -> float | None:
    if full is None or without is None:
        return None
    full_value = safe_float(full.get(metric))
    without_value = safe_float(without.get(metric))
    if full_value is None or without_value is None:
        return None
    return full_value - without_value


def _gain(base, plus, metric: str) -> float | None:
    if base is None or plus is None:
        return None
    base_value = safe_float(base.get(metric))
    plus_value = safe_float(plus.get(metric))
    if base_value is None or plus_value is None:
        return None
    return plus_value - base_value


def _skipped_reason(skipped_models: pd.DataFrame) -> str:
    if skipped_models.empty or "reason" not in skipped_models.columns:
        return "No usable prediction rows."
    return "; ".join(skipped_models["reason"].dropna().astype(str).head(3))


def _format_percent(value) -> str:
    number = safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.2f}%"


def _empty_result(
    feature_groups: dict[str, list[str]] | None = None,
) -> AblationResult:
    return AblationResult(
        predictions=pd.DataFrame(),
        score_table=pd.DataFrame(),
        group_impact=pd.DataFrame(),
        skipped_configs=pd.DataFrame(),
        feature_groups=feature_groups or {name: [] for name in FEATURE_GROUP_ORDER},
        configs=[],
        split_summary={},
        summary_messages=["Ablation results are not available yet."],
    )
