from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.labels import distribution as label_distribution
from stock_ai.labels.targets import CLASS_NAMES
from stock_ai.models.baseline import select_feature_columns
from stock_ai.utils.safe_cast import safe_float, safe_int


TARGET_COLUMN = "target_next_close_class"
PROBABILITY_COLUMNS = {
    0: "probability_crash",
    1: "probability_down",
    2: "probability_flat",
    3: "probability_up",
    4: "probability_surge",
}
CONFIDENCE_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60)
HIGH_CONFIDENCE_THRESHOLD = 0.55
TAIL_SAMPLE_WARNING = 30


@dataclass(frozen=True)
class MulticlassWalkForwardConfig:
    ticker: str = "7203.T"
    train_years: int = 3
    test_step_months: int = 1
    start_date: str | None = None
    end_date: str | None = None
    min_train_rows: int = 252
    valid_fraction: float = 0.2
    threshold_profile: str = label_distribution.DEFAULT_THRESHOLD_PROFILE
    weight_mode: str = label_distribution.DEFAULT_WEIGHT_MODE
    min_sample_warning: int = 100


@dataclass
class MulticlassWalkForwardResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    class_distribution: pd.DataFrame
    confidence_buckets: pd.DataFrame
    probability_summary: pd.DataFrame
    direction_comparison: pd.DataFrame
    context_summary: pd.DataFrame
    biggest_moves: pd.DataFrame
    fold_details: pd.DataFrame
    conclusions: list[str] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)


def run_multiclass_walk_forward_diagnostics(
    data: pd.DataFrame,
    config: MulticlassWalkForwardConfig | None = None,
    *,
    live_frame: pd.DataFrame | None = None,
    model_factory=None,
) -> MulticlassWalkForwardResult:
    """Run historical multiclass walk-forward diagnostics without saving predictions.

    Label thresholds are resolved from each fold's training window only, then
    applied to that fold's validation/test rows. This keeps the diagnostic close
    to production training semantics while avoiding test-period threshold tuning.
    """
    config = config or MulticlassWalkForwardConfig()
    frame = _prepare_diagnostic_frame(data, config)
    if frame.empty:
        return _build_result(pd.DataFrame(), pd.DataFrame(), config, live_frame=live_frame)

    prediction_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    scheme = label_distribution.get_threshold_profile(config.threshold_profile)
    for test_start, test_end in _test_windows(frame, config):
        raw_train = _training_window(frame, test_start, config)
        raw_test = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        if len(raw_train) < config.min_train_rows or raw_test.empty:
            continue

        thresholds = label_distribution.resolve_label_thresholds(raw_train, scheme)
        train_window = label_distribution.add_target_for_scheme(
            raw_train,
            scheme,
            target_column=TARGET_COLUMN,
            thresholds=thresholds,
        )
        test_window = label_distribution.add_target_for_scheme(
            raw_test,
            scheme,
            target_column=TARGET_COLUMN,
            thresholds=thresholds,
        )
        test_window = test_window[test_window[TARGET_COLUMN].isin(CLASS_NAMES)].copy()
        if test_window.empty:
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        feature_columns = _available_feature_columns(train_window, test_window, feature_columns)
        if not feature_columns:
            continue

        train_window = train_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        test_window = test_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        train_window = train_window[train_window[TARGET_COLUMN].isin(CLASS_NAMES)].copy()
        test_window = test_window[test_window[TARGET_COLUMN].isin(CLASS_NAMES)].copy()
        if (
            len(train_window) < config.min_train_rows
            or test_window.empty
            or train_window[TARGET_COLUMN].nunique() < 2
        ):
            continue

        train_data, valid_data = _train_valid_split(train_window, config.valid_fraction)
        if train_data.empty or valid_data.empty or train_data[TARGET_COLUMN].nunique() < 2:
            continue

        model = (
            model_factory(train_data, valid_data, feature_columns, TARGET_COLUMN)
            if model_factory is not None
            else _fit_lightgbm_multiclass(train_data, valid_data, feature_columns, config.weight_mode)
        )
        prediction_frames.append(
            _predict_fold(
                model=model,
                test_data=test_window,
                feature_columns=feature_columns,
                config=config,
                train_data=train_data,
                valid_data=valid_data,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_rows.append(
            _fold_detail_row(
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_window,
                feature_columns=feature_columns,
                thresholds=thresholds,
                test_start=test_start,
                test_end=test_end,
            )
        )

    predictions = (
        pd.concat(prediction_frames, ignore_index=True).sort_values("date").reset_index(drop=True)
        if prediction_frames
        else pd.DataFrame()
    )
    fold_details = pd.DataFrame(fold_rows)
    return _build_result(predictions, fold_details, config, live_frame=live_frame)


def append_binary_direction_reference(
    result: MulticlassWalkForwardResult,
    binary_frame: pd.DataFrame,
    *,
    source: str = "historical",
    period: str = "Historical all",
) -> MulticlassWalkForwardResult:
    binary_direction = binary_direction_comparison_frame(binary_frame, source=source, period=period)
    if not binary_direction.empty:
        result.direction_comparison = pd.concat(
            [result.direction_comparison, binary_direction],
            ignore_index=True,
        )
    return result


def write_multiclass_walk_forward_reports(
    result: MulticlassWalkForwardResult,
    output_dir: str | Path,
    *,
    run_date: str | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = run_date or date.today().isoformat()
    csv_path = output / f"multiclass_walk_forward_diagnostics_{stamp}.csv"
    json_path = output / f"multiclass_walk_forward_diagnostics_{stamp}.json"

    result.summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "conclusions": result.conclusions,
        "summary": _json_records(result.summary),
        "class_metrics": _json_records(result.class_metrics),
        "confusion_matrix": _json_records(result.confusion_matrix),
        "class_distribution": _json_records(result.class_distribution),
        "confidence_buckets": _json_records(result.confidence_buckets),
        "probability_summary": _json_records(result.probability_summary),
        "direction_comparison": _json_records(result.direction_comparison),
        "context_summary": _json_records(result.context_summary),
        "biggest_moves": _json_records(result.biggest_moves),
        "fold_details": _json_records(result.fold_details),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result.report_paths = {"csv": str(csv_path), "json": str(json_path)}
    return result.report_paths


def multiclass_source_summary_frame(
    frames: dict[str, pd.DataFrame],
    *,
    min_sample_warning: int = 100,
) -> pd.DataFrame:
    rows = []
    for source, frame in frames.items():
        rows.append(
            _summary_row(
                frame,
                source=str(source),
                period=_source_period_label(str(source)),
                min_sample_warning=min_sample_warning,
            )
        )
    return pd.DataFrame(rows, columns=_summary_columns())


def multiclass_period_summary_frame(
    frame: pd.DataFrame,
    *,
    source: str = "historical",
    reference_date=None,
    min_sample_warning: int = 100,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = _summary_columns()
    if rows.empty:
        return pd.DataFrame(columns=columns)

    dates = pd.to_datetime(rows.get("date"), errors="coerce")
    latest = (
        pd.Timestamp(reference_date).normalize()
        if reference_date is not None
        else pd.Timestamp(dates.max()).normalize()
    )
    period_masks = {
        "Historical all": pd.Series(True, index=rows.index),
        "2023-": dates >= pd.Timestamp("2023-01-01"),
        "2024-": dates >= pd.Timestamp("2024-01-01"),
        "2025-": dates >= pd.Timestamp("2025-01-01"),
        "2026-": dates >= pd.Timestamp("2026-01-01"),
        "Last 6 months": dates >= latest - pd.DateOffset(months=6),
        "Last 3 months": dates >= latest - pd.DateOffset(months=3),
    }
    result = [
        _summary_row(
            rows[mask].copy(),
            source=source,
            period=period,
            min_sample_warning=min_sample_warning,
        )
        for period, mask in period_masks.items()
    ]
    return pd.DataFrame(result, columns=columns)


def multiclass_class_metrics_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = [
        "source",
        "period",
        "class_id",
        "class",
        "precision",
        "recall",
        "f1",
        "support",
        "predicted_count",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    actual = rows["actual_class"].map(safe_int).astype(int)
    predicted = rows["predicted_class"].map(safe_int).astype(int)
    result = []
    for class_id, class_name in CLASS_NAMES.items():
        actual_mask = actual == class_id
        predicted_mask = predicted == class_id
        true_positive = int((actual_mask & predicted_mask).sum())
        support = int(actual_mask.sum())
        predicted_count = int(predicted_mask.sum())
        precision = _safe_divide(true_positive, predicted_count)
        recall = _safe_divide(true_positive, support)
        result.append(
            {
                "source": source,
                "period": period,
                "class_id": class_id,
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": _f1(precision, recall),
                "support": support,
                "predicted_count": predicted_count,
            }
        )
    return pd.DataFrame(result, columns=columns)


def multiclass_confusion_matrix_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = ["source", "period", "actual_class", "predicted_class", "count"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    result = []
    for actual_id, actual_name in CLASS_NAMES.items():
        for predicted_id, predicted_name in CLASS_NAMES.items():
            result.append(
                {
                    "source": source,
                    "period": period,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "count": int(((actual == actual_id) & (predicted == predicted_id)).sum()),
                }
            )
    return pd.DataFrame(result, columns=columns)


def class_distribution_frame(
    *,
    historical_frame: pd.DataFrame,
    fold_details: pd.DataFrame | None = None,
    live_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = []
    historical = _confirmed_multiclass_rows(historical_frame)
    result.extend(_distribution_rows(historical.get("actual_class"), "historical_actual"))
    result.extend(_distribution_rows(historical.get("predicted_class"), "historical_predicted"))
    if fold_details is not None and not fold_details.empty:
        result.extend(_fold_train_distribution_rows(fold_details))
    if live_frame is not None:
        live = _confirmed_multiclass_rows(live_frame)
        result.extend(_distribution_rows(live.get("actual_class"), "live_actual"))
        result.extend(_distribution_rows(live.get("predicted_class"), "live_predicted"))
    return pd.DataFrame(result, columns=["dataset", "class_id", "class", "count", "share"])


def confidence_bucket_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = [
        "source",
        "period",
        "threshold",
        "sample_count",
        "accuracy",
        "coverage",
        "average_confidence",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    confidence = _confidence(rows)
    total = int(len(rows))
    result = []
    for threshold in CONFIDENCE_THRESHOLDS:
        subset = rows[confidence >= threshold].copy()
        result.append(
            {
                "source": source,
                "period": period,
                "threshold": threshold,
                "sample_count": int(len(subset)),
                "accuracy": _accuracy(subset),
                "coverage": _safe_divide(len(subset), total),
                "average_confidence": _mean(subset.get("confidence")),
            }
        )
    return pd.DataFrame(result, columns=columns)


def probability_summary_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = ["source", "period", "metric", "count", "min", "max", "mean", "median"]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    result = []
    metrics = {class_name: column for _, (class_id, class_name) in enumerate(CLASS_NAMES.items()) for column in [PROBABILITY_COLUMNS[class_id]]}
    metrics["confidence"] = "confidence"
    metrics["max_probability"] = "_max_probability"
    rows = rows.copy()
    rows["_max_probability"] = _max_probability(rows)
    for metric, column in metrics.items():
        values = pd.to_numeric(rows.get(column), errors="coerce").dropna()
        result.append(
            {
                "source": source,
                "period": period,
                "metric": metric,
                "count": int(len(values)),
                "min": float(values.min()) if not values.empty else None,
                "max": float(values.max()) if not values.empty else None,
                "mean": float(values.mean()) if not values.empty else None,
                "median": float(values.median()) if not values.empty else None,
            }
        )
    return pd.DataFrame(result, columns=columns)


def context_summary_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = [
        "source",
        "period",
        "context_type",
        "context_value",
        "sample_count",
        "accuracy",
        "macro_f1",
        "predicted_flat_ratio",
        "actual_flat_ratio",
        "average_confidence",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    result = []
    for context_type, column in (("regime", "regime"), ("anomaly", "anomaly_level")):
        if column not in rows.columns:
            continue
        values = rows[column].fillna("Unknown").replace("", "Unknown").astype(str)
        for context_value, group in rows.groupby(values, dropna=False):
            actual = group["actual_class"].map(safe_int).astype(int)
            predicted = group["predicted_class"].map(safe_int).astype(int)
            metrics = _multiclass_metric_values(actual, predicted)
            result.append(
                {
                    "source": source,
                    "period": period,
                    "context_type": context_type,
                    "context_value": context_value,
                    "sample_count": int(len(group)),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "predicted_flat_ratio": _safe_divide(
                        int((predicted == 2).sum()),
                        len(predicted),
                    ),
                    "actual_flat_ratio": _safe_divide(int((actual == 2).sum()), len(actual)),
                    "average_confidence": _mean(group.get("confidence")),
                }
            )
    return pd.DataFrame(result, columns=columns).sort_values(
        ["context_type", "sample_count"],
        ascending=[True, False],
    )


def direction_collapsed_comparison_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = [
        "source",
        "period",
        "model",
        "mode",
        "sample_count",
        "coverage",
        "direction_accuracy",
        "flat_prediction_count",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    actual_direction = rows["actual_class"].map(_multiclass_direction)
    predicted_direction = rows["predicted_class"].map(_multiclass_direction)
    total = int(len(rows))
    exact = actual_direction == predicted_direction
    non_flat = actual_direction.ne("flat") & predicted_direction.ne("flat")
    actual_directional = actual_direction.ne("flat")
    flat_predictions = int(predicted_direction.eq("flat").sum())
    result = [
        {
            "source": source,
            "period": period,
            "model": "multiclass",
            "mode": "flat_as_neutral",
            "sample_count": total,
            "coverage": 1.0,
            "direction_accuracy": _safe_divide(int(exact.sum()), total),
            "flat_prediction_count": flat_predictions,
        },
        {
            "source": source,
            "period": period,
            "model": "multiclass",
            "mode": "flat_excluded",
            "sample_count": int(non_flat.sum()),
            "coverage": _safe_divide(int(non_flat.sum()), total),
            "direction_accuracy": _safe_divide(
                int(exact[non_flat].sum()),
                int(non_flat.sum()),
            ),
            "flat_prediction_count": flat_predictions,
        },
        {
            "source": source,
            "period": period,
            "model": "multiclass",
            "mode": "actual_direction_only",
            "sample_count": int(actual_directional.sum()),
            "coverage": _safe_divide(int(actual_directional.sum()), total),
            "direction_accuracy": _safe_divide(
                int(exact[actual_directional].sum()),
                int(actual_directional.sum()),
            ),
            "flat_prediction_count": flat_predictions,
        },
    ]
    return pd.DataFrame(result, columns=columns)


def binary_direction_comparison_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
) -> pd.DataFrame:
    rows = frame.copy()
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "period",
                "model",
                "mode",
                "sample_count",
                "coverage",
                "direction_accuracy",
                "flat_prediction_count",
            ]
        )
    if "status" in rows.columns:
        rows = rows[rows["status"].astype(str).eq("confirmed")].copy()
    actual = rows.get("actual_class", pd.Series(dtype=object)).map(safe_int)
    predicted = rows.get("predicted_class", pd.Series(dtype=object)).map(safe_int)
    valid = actual.isin([0, 1]) & predicted.isin([0, 1])
    sample_count = int(valid.sum())
    accuracy = _safe_divide(int((actual[valid] == predicted[valid]).sum()), sample_count)
    return pd.DataFrame(
        [
            {
                "source": source,
                "period": period,
                "model": "binary",
                "mode": "binary_up_down",
                "sample_count": sample_count,
                "coverage": 1.0 if sample_count else None,
                "direction_accuracy": accuracy,
                "flat_prediction_count": 0,
            }
        ]
    )


def biggest_multiclass_moves_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
    limit: int = 40,
) -> pd.DataFrame:
    rows = _confirmed_multiclass_rows(frame)
    columns = [
        "source",
        "period",
        "category",
        "feature_date",
        "target_close_date",
        "predicted_class",
        "actual_class",
        "actual_return",
        "confidence",
        "regime",
        "anomaly_score",
    ]
    if rows.empty:
        return pd.DataFrame(columns=columns)
    result = rows.copy()
    actual = result["actual_class"].map(safe_int)
    predicted = result["predicted_class"].map(safe_int)
    actual_return = pd.to_numeric(result.get("actual_return"), errors="coerce")
    correct = actual == predicted
    large_move = actual_return.abs() >= 0.02
    result["category"] = np.select(
        [
            large_move & correct,
            large_move & ~correct,
            correct,
            ~correct,
        ],
        [
            "large_move_hit",
            "large_move_miss",
            "low_confidence_hit",
            "low_confidence_miss",
        ],
        default="unknown",
    )
    confidence = _confidence(result).fillna(0.0)
    result["_rank"] = actual_return.abs().fillna(0.0) * 10 + confidence
    result = result.sort_values("_rank", ascending=False).head(limit)
    result["source"] = source
    result["period"] = period
    result["feature_date"] = pd.to_datetime(result["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    result["target_close_date"] = pd.to_datetime(
        result.get("target_close_date"),
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")
    result["predicted_class"] = result["predicted_class"].map(lambda value: CLASS_NAMES.get(safe_int(value), value))
    result["actual_class"] = result["actual_class"].map(lambda value: CLASS_NAMES.get(safe_int(value), value))
    if "regime" not in result.columns:
        result["regime"] = "Unknown"
    if "anomaly_score" not in result.columns:
        result["anomaly_score"] = None
    return result[[column for column in columns if column in result.columns]].reset_index(drop=True)


def classify_multiclass_conclusions(
    summary: pd.DataFrame,
    direction_comparison: pd.DataFrame,
) -> list[str]:
    if summary.empty:
        return ["no_clear_multiclass_signal", "production_change_not_ready"]

    conclusions: list[str] = []
    historical = _summary_match(summary, source="historical", period="Historical all")
    live = _summary_match(summary, source="live", period="Live only")
    if historical is None:
        conclusions.extend(["no_clear_multiclass_signal", "production_change_not_ready"])
        return conclusions

    historical_flat = safe_float(historical.get("predicted_flat_ratio"))
    actual_flat = safe_float(historical.get("actual_flat_ratio"))
    live_flat = safe_float(live.get("predicted_flat_ratio")) if live is not None else None
    if historical_flat is not None and actual_flat is not None:
        if historical_flat >= max(0.45, actual_flat + 0.15):
            conclusions.append("multiclass_flat_bias_structural")
        elif live_flat is not None and live_flat >= max(0.45, historical_flat + 0.15):
            conclusions.append("multiclass_flat_bias_live_only")

    surge_support = safe_int(historical.get("surge_support")) or 0
    crash_support = safe_int(historical.get("crash_support")) or 0
    if surge_support < TAIL_SAMPLE_WARNING or crash_support < TAIL_SAMPLE_WARNING:
        conclusions.append("tail_classes_too_sparse")

    flat_abs_return = safe_float(historical.get("predicted_flat_actual_abs_return_mean"))
    if historical_flat is not None and flat_abs_return is not None:
        if historical_flat >= 0.45 and flat_abs_return >= 0.01:
            conclusions.append("flat_boundary_needs_review")

    confidence_median = safe_float(historical.get("confidence_median"))
    high_confidence_rate = safe_float(historical.get("high_confidence_rate"))
    if confidence_median is not None and confidence_median < 0.45:
        conclusions.append("probability_compression_multiclass")
    if high_confidence_rate is not None and high_confidence_rate < 0.05:
        conclusions.append("high_confidence_multiclass_not_available")

    binary_direction = direction_comparison[
        (direction_comparison.get("model") == "binary")
        & (direction_comparison.get("period") == "Historical all")
    ]
    multiclass_direction = direction_comparison[
        (direction_comparison.get("model") == "multiclass")
        & (direction_comparison.get("mode") == "actual_direction_only")
        & (direction_comparison.get("period") == "Historical all")
    ]
    if not binary_direction.empty and not multiclass_direction.empty:
        binary_accuracy = safe_float(binary_direction.iloc[0].get("direction_accuracy"))
        multiclass_accuracy = safe_float(multiclass_direction.iloc[0].get("direction_accuracy"))
        if binary_accuracy is not None and multiclass_accuracy is not None:
            if multiclass_accuracy <= binary_accuracy + 0.005:
                conclusions.append("multiclass_direction_not_better_than_binary")

    if not conclusions:
        conclusions.append("no_clear_multiclass_signal")
    conclusions.append("production_change_not_ready")
    return list(dict.fromkeys(conclusions))


def _build_result(
    historical_predictions: pd.DataFrame,
    fold_details: pd.DataFrame,
    config: MulticlassWalkForwardConfig,
    *,
    live_frame: pd.DataFrame | None = None,
) -> MulticlassWalkForwardResult:
    frames: dict[str, pd.DataFrame] = {"historical": historical_predictions}
    if live_frame is not None and not live_frame.empty:
        frames["live"] = live_frame
        combined = pd.concat([historical_predictions, live_frame], ignore_index=True)
        frames["historical + live"] = combined

    summary = multiclass_source_summary_frame(frames, min_sample_warning=config.min_sample_warning)
    period = multiclass_period_summary_frame(
        historical_predictions,
        source="historical",
        min_sample_warning=config.min_sample_warning,
    )
    if not period.empty:
        summary = pd.concat(
            [
                summary[~summary["period"].eq("Historical all")],
                period,
            ],
            ignore_index=True,
        )

    class_metrics_frames = []
    confusion_frames = []
    confidence_frames = []
    probability_frames = []
    direction_frames = []
    context_frames = []
    biggest_frames = []
    for source, frame in frames.items():
        period_label = _source_period_label(source)
        class_metrics_frames.append(
            multiclass_class_metrics_frame(frame, source=source, period=period_label)
        )
        confusion_frames.append(
            multiclass_confusion_matrix_frame(frame, source=source, period=period_label)
        )
        confidence_frames.append(confidence_bucket_frame(frame, source=source, period=period_label))
        probability_frames.append(probability_summary_frame(frame, source=source, period=period_label))
        direction_frames.append(
            direction_collapsed_comparison_frame(frame, source=source, period=period_label)
        )
        context_frames.append(context_summary_frame(frame, source=source, period=period_label))
        biggest_frames.append(biggest_multiclass_moves_frame(frame, source=source, period=period_label))

    result = MulticlassWalkForwardResult(
        summary=summary.reset_index(drop=True),
        predictions=historical_predictions,
        class_metrics=_concat_frames(class_metrics_frames),
        confusion_matrix=_concat_frames(confusion_frames),
        class_distribution=class_distribution_frame(
            historical_frame=historical_predictions,
            fold_details=fold_details,
            live_frame=live_frame,
        ),
        confidence_buckets=_concat_frames(confidence_frames),
        probability_summary=_concat_frames(probability_frames),
        direction_comparison=_concat_frames(direction_frames),
        context_summary=_concat_frames(context_frames),
        biggest_moves=_concat_frames(biggest_frames),
        fold_details=fold_details,
        conclusions=[],
    )
    result.conclusions = classify_multiclass_conclusions(result.summary, result.direction_comparison)
    return result


def _prepare_diagnostic_frame(data: pd.DataFrame, config: MulticlassWalkForwardConfig) -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame()
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if config.ticker and "ticker" in frame.columns:
        frame = frame[frame["ticker"].astype(str).eq(config.ticker)].copy()
    if frame.empty:
        return pd.DataFrame()
    if "ticker" not in frame.columns:
        frame["ticker"] = config.ticker
    frame = label_distribution.ensure_next_close_return(frame)
    frame = _attach_target_close_metadata(frame)
    if "regime" not in frame.columns:
        frame["regime"] = "Unknown"
    if "anomaly_score" not in frame.columns:
        frame["anomaly_score"] = 0.0
    if "anomaly_level" not in frame.columns:
        frame["anomaly_level"] = frame["anomaly_score"].map(_anomaly_level)
    if config.end_date:
        frame = frame[frame["date"] <= pd.to_datetime(config.end_date)]
    return frame.reset_index(drop=True)


def _attach_target_close_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.sort_values(["ticker", "date"]).copy()
    grouped = rows.groupby("ticker", group_keys=False)
    rows["target_close_date"] = grouped["date"].shift(-1)
    if "next_close" not in rows.columns:
        rows["next_close"] = grouped["close"].shift(-1)
    if "next_close_return" not in rows.columns:
        rows["next_close_return"] = rows["next_close"] / rows["close"] - 1
    return rows


def _test_windows(
    frame: pd.DataFrame,
    config: MulticlassWalkForwardConfig,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if frame.empty:
        return []
    first_test_date = frame["date"].min() + pd.DateOffset(years=config.train_years)
    if config.start_date:
        first_test_date = max(first_test_date, pd.to_datetime(config.start_date))
    last_date = frame["date"].max()
    if config.end_date:
        last_date = min(last_date, pd.to_datetime(config.end_date))

    candidates = frame[(frame["date"] >= first_test_date) & (frame["date"] <= last_date)].copy()
    if candidates.empty:
        return []
    candidates["period"] = candidates["date"].dt.to_period("M")
    month_starts = candidates.groupby("period")["date"].min().tolist()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    step = max(1, config.test_step_months)
    for index in range(0, len(month_starts), step):
        start = pd.Timestamp(month_starts[index])
        next_index = index + step
        end = (
            pd.Timestamp(month_starts[next_index])
            if next_index < len(month_starts)
            else start + pd.DateOffset(months=step)
        )
        windows.append((start, end))
    return windows


def _training_window(
    frame: pd.DataFrame,
    test_start: pd.Timestamp,
    config: MulticlassWalkForwardConfig,
) -> pd.DataFrame:
    train_start = test_start - pd.DateOffset(years=config.train_years)
    train = frame[(frame["date"] >= train_start) & (frame["date"] < test_start)].copy()
    if len(train) > 1:
        train = train.iloc[:-1].copy()
    return train


def _available_feature_columns(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str],
) -> list[str]:
    available: list[str] = []
    for column in feature_columns:
        if column not in train.columns or column not in test.columns:
            continue
        train_values = pd.to_numeric(train[column], errors="coerce")
        test_values = pd.to_numeric(test[column], errors="coerce")
        if train_values.notna().any() and test_values.notna().any():
            available.append(column)
    return available


def _train_valid_split(train: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(train) * (1 - valid_fraction))
    split_index = max(1, min(split_index, len(train) - 1))
    return train.iloc[:split_index].copy(), train.iloc[split_index:].copy()


def _fit_lightgbm_multiclass(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    weight_mode: str,
):
    import lightgbm as lgb

    active_weight_mode = label_distribution.normalize_weight_mode(weight_mode)
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight=label_distribution.multiclass_class_weight(active_weight_mode),
        random_state=42,
        verbose=-1,
    )
    train_target = train[TARGET_COLUMN].astype(int)
    valid_target = valid[TARGET_COLUMN].astype(int)
    if set(valid_target.unique()).issubset(set(train_target.unique())):
        model.fit(
            train[feature_columns],
            train_target,
            eval_set=[(valid[feature_columns], valid_target)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )
    else:
        model.fit(train[feature_columns], train_target)
    return model


def _predict_fold(
    *,
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    config: MulticlassWalkForwardConfig,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_start,
    test_end,
) -> pd.DataFrame:
    probabilities = _predict_class_probabilities(model, test_data[feature_columns])
    rows = test_data.copy()
    for class_id, column in PROBABILITY_COLUMNS.items():
        rows[column] = probabilities[:, class_id]
    rows["predicted_class"] = probabilities.argmax(axis=1).astype(int)
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = probabilities.max(axis=1)
    rows["target"] = "next_close_multiclass"
    rows["model_name"] = "walk_forward_multiclass"
    rows["prediction_type"] = "historical_multiclass_walk_forward"
    rows["status"] = "confirmed"
    rows["is_correct"] = rows["predicted_class"] == rows["actual_class"]
    rows["threshold_profile"] = config.threshold_profile
    rows["weight_mode"] = config.weight_mode
    rows["feature_count"] = len(feature_columns)
    rows["train_start_date"] = pd.to_datetime(train_data["date"].min())
    rows["train_end_date"] = pd.to_datetime(train_data["date"].max())
    rows["valid_start_date"] = pd.to_datetime(valid_data["date"].min())
    rows["valid_end_date"] = pd.to_datetime(valid_data["date"].max())
    rows["test_start_date"] = pd.to_datetime(test_start)
    rows["test_end_date"] = pd.to_datetime(test_end)
    rows["actual_return"] = pd.to_numeric(rows["next_close_return"], errors="coerce")

    keep = [
        "date",
        "ticker",
        "target_close_date",
        "target",
        "model_name",
        "prediction_type",
        "predicted_class",
        "actual_class",
        "confidence",
        *PROBABILITY_COLUMNS.values(),
        "next_close_return",
        "actual_return",
        "status",
        "is_correct",
        "regime",
        "anomaly_level",
        "anomaly_score",
        "threshold_profile",
        "weight_mode",
        "feature_count",
        "train_start_date",
        "train_end_date",
        "valid_start_date",
        "valid_end_date",
        "test_start_date",
        "test_end_date",
    ]
    return rows[[column for column in keep if column in rows.columns]]


def _predict_class_probabilities(model, features: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Multiclass model predict_proba must return a 2D array.")

    normalized = np.zeros((len(features), len(CLASS_NAMES)), dtype=float)
    class_ids = _model_class_ids(model, probabilities.shape[1])
    for position, class_id in enumerate(class_ids):
        if class_id in CLASS_NAMES and position < probabilities.shape[1]:
            normalized[:, class_id] = probabilities[:, position]
    row_sums = normalized.sum(axis=1)
    missing = row_sums <= 0
    if missing.any():
        fallback = np.zeros(len(CLASS_NAMES), dtype=float)
        fallback[: min(probabilities.shape[1], len(CLASS_NAMES))] = probabilities[missing][0][
            : min(probabilities.shape[1], len(CLASS_NAMES))
        ]
        normalized[missing] = fallback
        row_sums = normalized.sum(axis=1)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    return normalized / row_sums.reshape(-1, 1)


def _model_class_ids(model, probability_column_count: int) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return list(range(probability_column_count))
    result = []
    for value in classes:
        if hasattr(value, "item"):
            value = value.item()
        result.append(int(value))
    if len(result) != probability_column_count:
        return list(range(probability_column_count))
    return result


def _fold_detail_row(
    *,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    thresholds: dict[str, float | None],
    test_start,
    test_end,
) -> dict[str, object]:
    row: dict[str, object] = {
        "test_start_date": _date_text(test_start),
        "test_end_date": _date_text(test_end),
        "train_start_date": _date_text(train_data["date"].min()),
        "train_end_date": _date_text(train_data["date"].max()),
        "valid_start_date": _date_text(valid_data["date"].min()),
        "valid_end_date": _date_text(valid_data["date"].max()),
        "train_rows": int(len(train_data)),
        "valid_rows": int(len(valid_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(feature_columns)),
        "label_thresholds": json.dumps(thresholds, ensure_ascii=False),
    }
    train_classes = set(train_data[TARGET_COLUMN].astype(int).unique())
    valid_classes = set(valid_data[TARGET_COLUMN].astype(int).unique())
    row["valid_only_classes"] = ",".join(
        CLASS_NAMES[class_id] for class_id in sorted(valid_classes - train_classes)
    )
    for class_id, class_name in CLASS_NAMES.items():
        row[f"train_{class_name}_count"] = int((train_data[TARGET_COLUMN].astype(int) == class_id).sum())
        row[f"test_{class_name}_count"] = int((test_data[TARGET_COLUMN].astype(int) == class_id).sum())
    return row


def _summary_row(
    frame: pd.DataFrame,
    *,
    source: str,
    period: str,
    min_sample_warning: int,
) -> dict[str, object]:
    rows = _confirmed_multiclass_rows(frame)
    empty = {column: None for column in _summary_columns()}
    empty.update(
        {
            "source": source,
            "period": period,
            "sample_count": 0,
            "warning": "no confirmed rows",
        }
    )
    if rows.empty:
        return empty

    actual = rows["actual_class"].map(safe_int).astype(int)
    predicted = rows["predicted_class"].map(safe_int).astype(int)
    start, end = _period(rows)
    metrics = _multiclass_metric_values(actual, predicted)
    predicted_flat = predicted == 2
    actual_flat = actual == 2
    actual_directional = actual.isin([0, 1, 3, 4])
    actual_tail = actual.isin([0, 4])
    actual_return = pd.to_numeric(rows.get("actual_return"), errors="coerce")
    confidence = _confidence(rows).dropna()
    max_probability = _max_probability(rows).dropna()
    high_confidence = confidence >= HIGH_CONFIDENCE_THRESHOLD
    notes, caveats = _summary_notes(
        sample_count=len(rows),
        predicted_flat_ratio=_safe_divide(int(predicted_flat.sum()), len(rows)),
        actual_flat_ratio=_safe_divide(int(actual_flat.sum()), len(rows)),
        surge_support=int((actual == 4).sum()),
        crash_support=int((actual == 0).sum()),
        confidence_median=float(confidence.median()) if not confidence.empty else None,
        min_sample_warning=min_sample_warning,
    )
    row = {
        "source": source,
        "period": period,
        "sample_count": int(len(rows)),
        "warning": "low sample" if len(rows) < min_sample_warning else "",
        "evaluated_start": start,
        "evaluated_end": end,
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "predicted_flat_ratio": _safe_divide(int(predicted_flat.sum()), len(rows)),
        "actual_flat_ratio": _safe_divide(int(actual_flat.sum()), len(rows)),
        "predicted_flat_actual_return_mean": _mean(actual_return[predicted_flat]),
        "predicted_flat_actual_abs_return_mean": _mean(actual_return[predicted_flat].abs()),
        "actual_up_down_predicted_flat_count": int((actual_directional & predicted_flat).sum()),
        "actual_tail_predicted_flat_count": int((actual_tail & predicted_flat).sum()),
        "surge_support": int((actual == 4).sum()),
        "crash_support": int((actual == 0).sum()),
        "predicted_surge_count": int((predicted == 4).sum()),
        "predicted_crash_count": int((predicted == 0).sum()),
        "surge_recall": metrics["surge_recall"],
        "crash_recall": metrics["crash_recall"],
        "confidence_mean": float(confidence.mean()) if not confidence.empty else None,
        "confidence_median": float(confidence.median()) if not confidence.empty else None,
        "confidence_max": float(confidence.max()) if not confidence.empty else None,
        "max_probability_mean": float(max_probability.mean()) if not max_probability.empty else None,
        "max_probability_median": float(max_probability.median()) if not max_probability.empty else None,
        "high_confidence_count": int(high_confidence.sum()) if not confidence.empty else 0,
        "high_confidence_rate": _safe_divide(int(high_confidence.sum()), len(confidence))
        if len(confidence)
        else None,
        "notes": "; ".join(notes),
        "caveats": "; ".join(caveats),
    }
    if row["warning"] and row["warning"] not in row["caveats"]:
        row["caveats"] = "; ".join([value for value in (row["warning"], row["caveats"]) if value])
    return row


def _multiclass_metric_values(actual: pd.Series, predicted: pd.Series) -> dict[str, float | None]:
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    labels = list(CLASS_NAMES)
    if len(actual) == 0:
        return {
            "accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
            "weighted_f1": None,
            "surge_recall": None,
            "crash_recall": None,
        }
    precision, recall, _f1_scores, support = precision_recall_fscore_support(
        actual,
        predicted,
        labels=labels,
        zero_division=0,
    )
    support_sum = support.sum()
    return {
        "accuracy": float((actual == predicted).mean()),
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)),
        "surge_recall": float(recall[labels.index(4)]),
        "crash_recall": float(recall[labels.index(0)]),
        "support_sum": int(support_sum),
    }


def _confirmed_multiclass_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    rows = frame.copy()
    if "status" in rows.columns:
        rows = rows[rows["status"].astype(str).eq("confirmed")].copy()
    actual = rows.get("actual_class", pd.Series(dtype=object)).map(safe_int)
    predicted = rows.get("predicted_class", pd.Series(dtype=object)).map(safe_int)
    valid = actual.isin(CLASS_NAMES) & predicted.isin(CLASS_NAMES)
    rows = rows[valid].copy()
    rows["actual_class"] = actual[valid].astype(int)
    rows["predicted_class"] = predicted[valid].astype(int)
    if "actual_return" not in rows.columns and "next_close_return" in rows.columns:
        rows["actual_return"] = rows["next_close_return"]
    if "confidence" not in rows.columns or rows["confidence"].isna().all():
        rows["confidence"] = _max_probability(rows)
    return rows


def _summary_columns() -> list[str]:
    return [
        "source",
        "period",
        "sample_count",
        "warning",
        "evaluated_start",
        "evaluated_end",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "predicted_flat_ratio",
        "actual_flat_ratio",
        "predicted_flat_actual_return_mean",
        "predicted_flat_actual_abs_return_mean",
        "actual_up_down_predicted_flat_count",
        "actual_tail_predicted_flat_count",
        "surge_support",
        "crash_support",
        "predicted_surge_count",
        "predicted_crash_count",
        "surge_recall",
        "crash_recall",
        "confidence_mean",
        "confidence_median",
        "confidence_max",
        "max_probability_mean",
        "max_probability_median",
        "high_confidence_count",
        "high_confidence_rate",
        "notes",
        "caveats",
    ]


def _summary_notes(
    *,
    sample_count: int,
    predicted_flat_ratio: float | None,
    actual_flat_ratio: float | None,
    surge_support: int,
    crash_support: int,
    confidence_median: float | None,
    min_sample_warning: int,
) -> tuple[list[str], list[str]]:
    notes: list[str] = []
    caveats: list[str] = []
    if sample_count < min_sample_warning:
        caveats.append("low sample count")
    if surge_support < TAIL_SAMPLE_WARNING:
        caveats.append("surge support is too sparse")
    if crash_support < TAIL_SAMPLE_WARNING:
        caveats.append("crash support is too sparse")
    if predicted_flat_ratio is not None and actual_flat_ratio is not None:
        if predicted_flat_ratio >= actual_flat_ratio + 0.15:
            notes.append("predicted flat ratio is materially above actual flat ratio")
    if confidence_median is not None and confidence_median < 0.45:
        notes.append("multiclass probabilities look compressed")
    return notes, caveats


def _confidence(rows: pd.DataFrame) -> pd.Series:
    if "confidence" in rows.columns:
        confidence = pd.to_numeric(rows["confidence"], errors="coerce")
        if confidence.notna().any():
            return confidence
    return _max_probability(rows)


def _max_probability(rows: pd.DataFrame) -> pd.Series:
    probability_columns = [column for column in PROBABILITY_COLUMNS.values() if column in rows.columns]
    if not probability_columns:
        return pd.Series(np.nan, index=rows.index, dtype=float)
    return rows[probability_columns].apply(pd.to_numeric, errors="coerce").max(axis=1)


def _distribution_rows(values, dataset: str) -> list[dict[str, object]]:
    labels = pd.Series(values if values is not None else [], dtype=object).map(safe_int)
    labels = labels[labels.isin(CLASS_NAMES)].astype(int)
    total = int(len(labels))
    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        count = int((labels == class_id).sum())
        rows.append(
            {
                "dataset": dataset,
                "class_id": class_id,
                "class": class_name,
                "count": count,
                "share": _safe_divide(count, total) if total else 0.0,
            }
        )
    return rows


def _fold_train_distribution_rows(fold_details: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    counts = {}
    for class_id, class_name in CLASS_NAMES.items():
        column = f"train_{class_name}_count"
        counts[class_id] = int(pd.to_numeric(fold_details.get(column), errors="coerce").fillna(0).sum())
    total = sum(counts.values())
    for class_id, class_name in CLASS_NAMES.items():
        count = counts[class_id]
        rows.append(
            {
                "dataset": "historical_train_folds",
                "class_id": class_id,
                "class": class_name,
                "count": count,
                "share": _safe_divide(count, total) if total else 0.0,
            }
        )
    return rows


def _multiclass_direction(value) -> str | None:
    class_id = safe_int(value)
    if class_id in (3, 4):
        return "up"
    if class_id in (0, 1):
        return "down"
    if class_id == 2:
        return "flat"
    return None


def _summary_match(summary: pd.DataFrame, *, source: str, period: str) -> pd.Series | None:
    if summary.empty:
        return None
    rows = summary[
        summary.get("source", pd.Series(dtype=str)).astype(str).eq(source)
        & summary.get("period", pd.Series(dtype=str)).astype(str).eq(period)
    ]
    if rows.empty:
        return None
    return rows.iloc[0]


def _source_period_label(source: str) -> str:
    labels = {
        "historical": "Historical all",
        "live": "Live only",
        "historical + live": "Historical + Live",
    }
    return labels.get(source, source)


def _accuracy(frame: pd.DataFrame) -> float | None:
    rows = _confirmed_multiclass_rows(frame)
    if rows.empty:
        return None
    return float((rows["actual_class"].astype(int) == rows["predicted_class"].astype(int)).mean())


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _period(rows: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(rows.get("date", pd.Series(dtype=object)), errors="coerce").dropna()
    if dates.empty:
        return None, None
    return _date_text(dates.min()), _date_text(dates.max())


def _mean(values) -> float | None:
    if values is None:
        return None
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return None
    return float(numbers.mean())


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _date_text(value) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _anomaly_level(value) -> str:
    score = safe_float(value) or 0.0
    if score >= 60:
        return "high"
    if score >= 30:
        return "medium"
    return "low"


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    return json.loads(normalized.where(pd.notna(normalized), None).to_json(orient="records"))
