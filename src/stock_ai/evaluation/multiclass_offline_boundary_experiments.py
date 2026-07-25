from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.evaluation import multiclass_walk_forward_diagnostics as diagnostics
from stock_ai.labels import distribution as label_distribution
from stock_ai.labels.targets import CLASS_NAMES, classify_return
from stock_ai.models.baseline import select_feature_columns
from stock_ai.utils.safe_cast import safe_float, safe_int


TARGET_COLUMN = "target_offline_multiclass"
PROBABILITY_COLUMNS = diagnostics.PROBABILITY_COLUMNS
CONFIDENCE_THRESHOLDS = diagnostics.CONFIDENCE_THRESHOLDS
PERIOD_ORDER = (
    "Historical all",
    "2023-",
    "2024-",
    "2025-",
    "2026-",
    "Last 6 months",
    "Last 3 months",
)


@dataclass(frozen=True)
class BoundaryCandidate:
    name: str
    boundary_mode: str
    weight_mode: str = label_distribution.DEFAULT_WEIGHT_MODE
    crash_threshold: float | None = None
    down_threshold: float | None = None
    up_threshold: float | None = None
    surge_threshold: float | None = None
    flat_threshold: float | None = None
    class_count: int = 5
    note: str = ""


@dataclass(frozen=True)
class BoundaryExperimentConfig:
    ticker: str = "7203.T"
    train_years: int = 3
    test_step_months: int = 1
    start_date: str | None = None
    end_date: str | None = None
    min_train_rows: int = 252
    valid_fraction: float = 0.2
    min_sample_warning: int = 100


@dataclass
class BoundaryExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    class_metrics: pd.DataFrame
    confusion_matrix: pd.DataFrame
    class_distribution: pd.DataFrame
    confidence_buckets: pd.DataFrame
    probability_summary: pd.DataFrame
    direction_comparison: pd.DataFrame
    context_summary: pd.DataFrame
    fold_details: pd.DataFrame
    candidate_definitions: pd.DataFrame
    conclusions: list[str] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)


def default_boundary_candidates() -> list[BoundaryCandidate]:
    return [
        BoundaryCandidate(
            name="current_multiclass",
            boundary_mode="fixed",
            crash_threshold=-0.05,
            down_threshold=-0.01,
            up_threshold=0.01,
            surge_threshold=0.05,
            note="Production-style fixed +/-5% tails and +/-1% flat band.",
        ),
        BoundaryCandidate(
            name="class_weight_balanced",
            boundary_mode="fixed",
            weight_mode="balanced",
            crash_threshold=-0.05,
            down_threshold=-0.01,
            up_threshold=0.01,
            surge_threshold=0.05,
            note="Current boundaries with LightGBM class_weight='balanced'.",
        ),
        BoundaryCandidate(
            name="class_weight_tail_boost",
            boundary_mode="fixed",
            weight_mode="manual_tail_boost",
            crash_threshold=-0.05,
            down_threshold=-0.01,
            up_threshold=0.01,
            surge_threshold=0.05,
            note="Current boundaries with manual crash/surge tail weights.",
        ),
        BoundaryCandidate(
            name="practical_3_0_5",
            boundary_mode="fixed",
            crash_threshold=-0.03,
            down_threshold=-0.005,
            up_threshold=0.005,
            surge_threshold=0.03,
            note="Fixed tails at +/-3% and flat band at +/-0.5%.",
        ),
        BoundaryCandidate(
            name="practical_2_0_5",
            boundary_mode="fixed",
            crash_threshold=-0.02,
            down_threshold=-0.005,
            up_threshold=0.005,
            surge_threshold=0.02,
            note="Fixed tails at +/-2% and flat band at +/-0.5%.",
        ),
        BoundaryCandidate(
            name="practical_3_0_3",
            boundary_mode="fixed",
            crash_threshold=-0.03,
            down_threshold=-0.003,
            up_threshold=0.003,
            surge_threshold=0.03,
            note="Fixed tails at +/-3% and narrower flat band at +/-0.3%.",
        ),
        BoundaryCandidate(
            name="quantile_5class",
            boundary_mode="quantile",
            note="Fold-train quantile thresholds: 5/30/70/95 percentiles.",
        ),
        BoundaryCandidate(
            name="quantile_tail_fixed_flat",
            boundary_mode="quantile_tail_fixed_flat",
            down_threshold=-0.005,
            up_threshold=0.005,
            note="Fixed flat band at +/-0.5%; crash/surge tails from fold-train 5/95 percentiles.",
        ),
        BoundaryCandidate(
            name="3class_down_flat_up",
            boundary_mode="three_class",
            flat_threshold=0.005,
            class_count=3,
            note="Fold crash/down into down and surge/up into up with a +/-0.5% flat band.",
        ),
    ]


def run_multiclass_offline_boundary_experiments(
    data: pd.DataFrame,
    config: BoundaryExperimentConfig | None = None,
    *,
    candidates: list[BoundaryCandidate] | None = None,
    live_frame: pd.DataFrame | None = None,
    binary_reference: pd.DataFrame | None = None,
    model_factory=None,
) -> BoundaryExperimentResult:
    config = config or BoundaryExperimentConfig()
    candidates = candidates or default_boundary_candidates()
    frame = diagnostics._prepare_diagnostic_frame(  # noqa: SLF001
        data,
        diagnostics.MulticlassWalkForwardConfig(
            ticker=config.ticker,
            train_years=config.train_years,
            test_step_months=config.test_step_months,
            start_date=config.start_date,
            end_date=config.end_date,
            min_train_rows=config.min_train_rows,
            valid_fraction=config.valid_fraction,
        ),
    )
    if frame.empty:
        return _empty_result(candidates)

    prediction_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        predictions, details = _run_candidate(
            frame,
            config,
            candidate,
            model_factory=model_factory,
        )
        if not predictions.empty:
            prediction_frames.append(predictions)
        if not details.empty:
            detail_frames.append(details)

    predictions = _concat_frames(prediction_frames)
    fold_details = _concat_frames(detail_frames)
    result = _build_result(
        predictions,
        fold_details,
        candidates,
        config,
        live_frame=live_frame,
        binary_reference=binary_reference,
    )
    result.conclusions = classify_boundary_experiment_conclusions(result.summary)
    return result


def write_boundary_experiment_reports(
    result: BoundaryExperimentResult,
    output_dir: str | Path,
    *,
    run_date: str | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = run_date or date.today().isoformat()
    csv_path = output / f"multiclass_offline_boundary_experiments_{stamp}.csv"
    json_path = output / f"multiclass_offline_boundary_experiments_{stamp}.json"

    result.summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "conclusions": result.conclusions,
        "candidate_definitions": _json_records(result.candidate_definitions),
        "summary": _json_records(result.summary),
        "class_metrics": _json_records(result.class_metrics),
        "confusion_matrix": _json_records(result.confusion_matrix),
        "class_distribution": _json_records(result.class_distribution),
        "confidence_buckets": _json_records(result.confidence_buckets),
        "probability_summary": _json_records(result.probability_summary),
        "direction_comparison": _json_records(result.direction_comparison),
        "context_summary": _json_records(result.context_summary),
        "fold_details": _json_records(result.fold_details),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    result.report_paths = {"csv": str(csv_path), "json": str(json_path)}
    return result.report_paths


def classify_boundary_experiment_conclusions(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["production_change_not_ready"]

    rows = summary[summary["period"].eq("Historical all")].copy()
    for column in (
        "macro_f1",
        "macro_recall",
        "predicted_flat_ratio",
        "direction_accuracy_flat_excluded",
        "direction_coverage",
        "up_recall",
        "down_recall",
        "surge_recall",
        "crash_recall",
        "tail_precision",
    ):
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    current = _candidate_row(rows, "current_multiclass")
    conclusions: list[str] = []

    if current is not None and safe_float(current.get("predicted_flat_ratio")) is not None:
        if float(current.get("predicted_flat_ratio")) >= 0.90:
            conclusions.append("current_boundary_too_flat_biased")

    best_non_current = rows[~rows["candidate_name"].eq("current_multiclass")].copy()
    current_flat = safe_float(current.get("predicted_flat_ratio")) if current is not None else None
    current_macro_f1 = safe_float(current.get("macro_f1")) if current is not None else None
    if current_flat is not None and not best_non_current.empty:
        fixed = best_non_current[best_non_current["boundary_mode"].eq("fixed")]
        if not fixed.empty and float(fixed["predicted_flat_ratio"].min()) <= current_flat - 0.10:
            conclusions.append("fixed_practical_boundary_improves_flat_bias")
        quantile = best_non_current[best_non_current["boundary_mode"].astype(str).str.contains("quantile")]
        if (
            current_macro_f1 is not None
            and not quantile.empty
            and float(quantile["macro_f1"].max()) >= current_macro_f1 + 0.01
        ):
            conclusions.append("quantile_boundary_improves_macro_f1")

    weighted = rows[rows["weight_mode"].isin(["balanced", "manual_tail_boost"])]
    if not weighted.empty:
        tail_recall = weighted[["surge_recall", "crash_recall"]].max(axis=1)
        if pd.to_numeric(tail_recall, errors="coerce").fillna(0).max() > 0:
            conclusions.append("class_weight_improves_tail_recall")
        tail_precision = pd.to_numeric(weighted.get("tail_precision"), errors="coerce")
        if tail_precision.notna().any() and float(tail_precision.max()) < 0.20:
            conclusions.append("class_weight_hurts_precision")

    three_class = _candidate_row(rows, "3class_down_flat_up")
    if current is not None and three_class is not None:
        three_direction = safe_float(three_class.get("direction_accuracy_flat_excluded"))
        current_direction = safe_float(current.get("direction_accuracy_flat_excluded"))
        if three_direction is not None and current_direction is not None and three_direction > current_direction:
            conclusions.append("3class_more_practical_than_5class")

    binary_refs = summary[summary["candidate_name"].eq("binary_walk_forward_reference")]
    if not binary_refs.empty:
        binary_accuracy = safe_float(binary_refs.iloc[0].get("direction_accuracy_flat_excluded"))
        candidate_best = pd.to_numeric(
            rows.loc[rows["candidate_name"] != "binary_walk_forward_reference", "direction_accuracy_flat_excluded"],
            errors="coerce",
        ).max()
        if binary_accuracy is not None and (pd.isna(candidate_best) or float(candidate_best) <= binary_accuracy + 0.005):
            conclusions.append("multiclass_still_not_better_than_binary")

    if (
        pd.to_numeric(rows.get("surge_support"), errors="coerce").min() < 30
        or pd.to_numeric(rows.get("crash_support"), errors="coerce").min() < 30
    ):
        conclusions.append("tail_classes_not_enough_support")

    promising = best_non_current[
        (pd.to_numeric(best_non_current.get("macro_f1"), errors="coerce") >= (current_macro_f1 or 0) + 0.01)
        & (pd.to_numeric(best_non_current.get("predicted_flat_ratio"), errors="coerce") <= 0.85)
    ]
    if not promising.empty:
        conclusions.append("candidate_promising_for_future")

    conclusions.append("production_change_not_ready")
    return list(dict.fromkeys(conclusions))


def _run_candidate(
    frame: pd.DataFrame,
    config: BoundaryExperimentConfig,
    candidate: BoundaryCandidate,
    *,
    model_factory=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_predictions: list[pd.DataFrame] = []
    fold_details: list[dict[str, object]] = []
    for test_start, test_end in diagnostics._test_windows(  # noqa: SLF001
        frame,
        diagnostics.MulticlassWalkForwardConfig(
            ticker=config.ticker,
            train_years=config.train_years,
            test_step_months=config.test_step_months,
            start_date=config.start_date,
            end_date=config.end_date,
            min_train_rows=config.min_train_rows,
            valid_fraction=config.valid_fraction,
        ),
    ):
        raw_train = diagnostics._training_window(  # noqa: SLF001
            frame,
            test_start,
            diagnostics.MulticlassWalkForwardConfig(
                ticker=config.ticker,
                train_years=config.train_years,
                test_step_months=config.test_step_months,
            ),
        )
        raw_test = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        if len(raw_train) < config.min_train_rows or raw_test.empty:
            continue

        thresholds = _candidate_thresholds(raw_train, candidate)
        train_window = _add_candidate_target(raw_train, candidate, thresholds)
        test_window = _add_candidate_target(raw_test, candidate, thresholds)
        test_window = test_window[test_window[TARGET_COLUMN] >= 0].copy()
        if test_window.empty:
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        feature_columns = [column for column in feature_columns if column != TARGET_COLUMN]
        feature_columns = diagnostics._available_feature_columns(  # noqa: SLF001
            train_window,
            test_window,
            feature_columns,
        )
        if not feature_columns:
            continue

        train_window = train_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        test_window = test_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        train_window = train_window[train_window[TARGET_COLUMN] >= 0].copy()
        test_window = test_window[test_window[TARGET_COLUMN] >= 0].copy()
        if (
            len(train_window) < config.min_train_rows
            or test_window.empty
            or train_window[TARGET_COLUMN].nunique() < 2
        ):
            continue

        train_data, valid_data = diagnostics._train_valid_split(  # noqa: SLF001
            train_window,
            config.valid_fraction,
        )
        if train_data.empty or valid_data.empty or train_data[TARGET_COLUMN].nunique() < 2:
            continue

        model = (
            model_factory(train_data, valid_data, feature_columns, TARGET_COLUMN, candidate)
            if model_factory is not None
            else _fit_lightgbm_candidate(train_data, valid_data, feature_columns, candidate)
        )
        fold_predictions.append(
            _predict_candidate_fold(
                model=model,
                test_data=test_window,
                feature_columns=feature_columns,
                candidate=candidate,
                thresholds=thresholds,
                train_data=train_data,
                valid_data=valid_data,
                test_start=test_start,
                test_end=test_end,
            )
        )
        fold_details.append(
            _fold_detail_row(
                train_data=train_data,
                valid_data=valid_data,
                test_data=test_window,
                feature_columns=feature_columns,
                candidate=candidate,
                thresholds=thresholds,
                test_start=test_start,
                test_end=test_end,
            )
        )

    predictions = _concat_frames(fold_predictions)
    if not predictions.empty:
        predictions = predictions.sort_values(["candidate_name", "date"]).reset_index(drop=True)
    return predictions, pd.DataFrame(fold_details)


def _candidate_thresholds(
    train_window: pd.DataFrame,
    candidate: BoundaryCandidate,
) -> dict[str, float | None]:
    if candidate.boundary_mode == "quantile":
        scheme = label_distribution.quantile_label_scheme(
            key="quantile_5class",
            name="quantile_5class",
        )
        return label_distribution.resolve_label_thresholds(train_window, scheme)
    if candidate.boundary_mode == "quantile_tail_fixed_flat":
        returns = pd.to_numeric(train_window.get("next_close_return"), errors="coerce").dropna()
        q05 = safe_float(returns.quantile(0.05)) if not returns.empty else None
        q95 = safe_float(returns.quantile(0.95)) if not returns.empty else None
        down = candidate.down_threshold if candidate.down_threshold is not None else -0.005
        up = candidate.up_threshold if candidate.up_threshold is not None else 0.005
        return {
            "crash_threshold": min(q05, down) if q05 is not None else down,
            "down_threshold": down,
            "up_threshold": up,
            "surge_threshold": max(q95, up) if q95 is not None else up,
        }
    if candidate.boundary_mode == "three_class":
        flat = candidate.flat_threshold if candidate.flat_threshold is not None else 0.005
        return {
            "crash_threshold": None,
            "down_threshold": -flat,
            "up_threshold": flat,
            "surge_threshold": None,
        }
    return {
        "crash_threshold": candidate.crash_threshold,
        "down_threshold": candidate.down_threshold,
        "up_threshold": candidate.up_threshold,
        "surge_threshold": candidate.surge_threshold,
    }


def _add_candidate_target(
    data: pd.DataFrame,
    candidate: BoundaryCandidate,
    thresholds: dict[str, float | None],
) -> pd.DataFrame:
    rows = data.copy()
    returns = pd.to_numeric(rows.get("next_close_return"), errors="coerce")
    if candidate.boundary_mode == "three_class":
        down = safe_float(thresholds.get("down_threshold"))
        up = safe_float(thresholds.get("up_threshold"))
        rows[TARGET_COLUMN] = returns.map(
            lambda value: _three_class_label(value, down_threshold=down, up_threshold=up)
        )
        return rows

    rows[TARGET_COLUMN] = returns.map(
        lambda value: classify_return(
            value,
            surge_threshold=float(thresholds["surge_threshold"]),
            up_threshold=float(thresholds["up_threshold"]),
            down_threshold=float(thresholds["down_threshold"]),
            crash_threshold=float(thresholds["crash_threshold"]),
        )
        if _thresholds_complete(thresholds)
        else -1
    )
    return rows


def _three_class_label(
    value,
    *,
    down_threshold: float | None,
    up_threshold: float | None,
) -> int:
    if pd.isna(value) or down_threshold is None or up_threshold is None:
        return -1
    if float(value) <= down_threshold:
        return 1
    if float(value) >= up_threshold:
        return 3
    return 2


def _fit_lightgbm_candidate(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    candidate: BoundaryCandidate,
):
    import lightgbm as lgb

    weight_mode = label_distribution.normalize_weight_mode(candidate.weight_mode)
    params: dict[str, object] = {
        "objective": "multiclass",
        "num_class": candidate.class_count,
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbose": -1,
    }
    train_target = train[TARGET_COLUMN].astype(int)
    if candidate.class_count == 5:
        class_weight = label_distribution.multiclass_class_weight(weight_mode)
        if isinstance(class_weight, dict):
            present_classes = set(train_target.unique())
            class_weight = {key: value for key, value in class_weight.items() if key in present_classes}
        params["class_weight"] = class_weight
    elif weight_mode == "balanced":
        params["class_weight"] = "balanced"

    model = lgb.LGBMClassifier(**params)
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


def _predict_candidate_fold(
    *,
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    candidate: BoundaryCandidate,
    thresholds: dict[str, float | None],
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_start,
    test_end,
) -> pd.DataFrame:
    probabilities = np.asarray(model.predict_proba(test_data[feature_columns]), dtype=float)
    class_ids = _model_class_ids(model, probabilities.shape[1])
    predicted_positions = probabilities.argmax(axis=1)
    predicted_classes = np.asarray([class_ids[position] for position in predicted_positions], dtype=int)

    rows = test_data.copy()
    for column in PROBABILITY_COLUMNS.values():
        rows[column] = 0.0
    for position, class_id in enumerate(class_ids):
        column = PROBABILITY_COLUMNS.get(class_id)
        if column is not None:
            rows[column] = probabilities[:, position]
    rows["predicted_class"] = predicted_classes
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = probabilities.max(axis=1)
    rows["candidate_name"] = candidate.name
    rows["boundary_mode"] = candidate.boundary_mode
    rows["weight_mode"] = candidate.weight_mode
    rows["class_count"] = candidate.class_count
    rows["target"] = "next_close_multiclass_offline"
    rows["model_name"] = candidate.name
    rows["prediction_type"] = f"offline_boundary_{candidate.name}"
    rows["status"] = "confirmed"
    rows["is_correct"] = rows["predicted_class"] == rows["actual_class"]
    rows["actual_return"] = pd.to_numeric(rows["next_close_return"], errors="coerce")
    rows["train_start_date"] = pd.to_datetime(train_data["date"].min())
    rows["train_end_date"] = pd.to_datetime(train_data["date"].max())
    rows["valid_start_date"] = pd.to_datetime(valid_data["date"].min())
    rows["valid_end_date"] = pd.to_datetime(valid_data["date"].max())
    rows["test_start_date"] = pd.to_datetime(test_start)
    rows["test_end_date"] = pd.to_datetime(test_end)
    for key, value in thresholds.items():
        rows[key] = value

    keep = [
        "date",
        "ticker",
        "target_close_date",
        "target",
        "model_name",
        "prediction_type",
        "candidate_name",
        "boundary_mode",
        "weight_mode",
        "class_count",
        "predicted_class",
        "actual_class",
        "confidence",
        *PROBABILITY_COLUMNS.values(),
        "actual_return",
        "status",
        "is_correct",
        "regime",
        "anomaly_level",
        "anomaly_score",
        "train_start_date",
        "train_end_date",
        "valid_start_date",
        "valid_end_date",
        "test_start_date",
        "test_end_date",
        "crash_threshold",
        "down_threshold",
        "up_threshold",
        "surge_threshold",
    ]
    return rows[[column for column in keep if column in rows.columns]]


def _build_result(
    predictions: pd.DataFrame,
    fold_details: pd.DataFrame,
    candidates: list[BoundaryCandidate],
    config: BoundaryExperimentConfig,
    *,
    live_frame: pd.DataFrame | None,
    binary_reference: pd.DataFrame | None,
) -> BoundaryExperimentResult:
    summary_frames: list[pd.DataFrame] = []
    class_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    distribution_frames: list[pd.DataFrame] = []
    confidence_frames: list[pd.DataFrame] = []
    probability_frames: list[pd.DataFrame] = []
    direction_frames: list[pd.DataFrame] = []
    context_frames: list[pd.DataFrame] = []

    for candidate in candidates:
        frame = predictions[predictions.get("candidate_name", pd.Series(dtype=str)).eq(candidate.name)].copy()
        if frame.empty:
            summary_frames.append(_no_prediction_summary(candidate))
            continue
        candidate_summary = _candidate_period_summary(frame, candidate, config)
        summary_frames.append(candidate_summary)
        for period in PERIOD_ORDER:
            subset = _period_subset(frame, period)
            if subset.empty:
                continue
            class_frames.append(
                diagnostics.multiclass_class_metrics_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
            confusion_frames.append(
                diagnostics.multiclass_confusion_matrix_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
            confidence_frames.append(
                diagnostics.confidence_bucket_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
            probability_frames.append(
                diagnostics.probability_summary_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
            direction_frames.append(
                diagnostics.direction_collapsed_comparison_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
            context_frames.append(
                diagnostics.context_summary_frame(
                    subset,
                    source=candidate.name,
                    period=period,
                ).assign(candidate_name=candidate.name)
            )
        distribution_frames.append(
            diagnostics.class_distribution_frame(historical_frame=frame, fold_details=None).assign(
                candidate_name=candidate.name
            )
        )

    if live_frame is not None and not live_frame.empty:
        live_summary = _live_summary(live_frame, config)
        summary_frames.append(live_summary)
        direction_frames.append(
            diagnostics.direction_collapsed_comparison_frame(
                live_frame,
                source="current_live_saved",
                period="Live only",
            ).assign(candidate_name="current_live_saved")
        )

    if binary_reference is not None and not binary_reference.empty:
        binary_summary = _binary_reference_summary(binary_reference)
        summary_frames.append(binary_summary)
        direction_frames.append(
            diagnostics.binary_direction_comparison_frame(
                binary_reference,
                source="binary_walk_forward_reference",
                period="Historical all",
            ).assign(candidate_name="binary_walk_forward_reference")
        )

    summary = _concat_frames(summary_frames)
    direction = _concat_frames(direction_frames)
    summary = _attach_direction_metrics(summary, direction)

    return BoundaryExperimentResult(
        summary=summary.reset_index(drop=True),
        predictions=predictions,
        class_metrics=_concat_frames(class_frames).reset_index(drop=True),
        confusion_matrix=_concat_frames(confusion_frames).reset_index(drop=True),
        class_distribution=_concat_frames(distribution_frames).reset_index(drop=True),
        confidence_buckets=_concat_frames(confidence_frames).reset_index(drop=True),
        probability_summary=_concat_frames(probability_frames).reset_index(drop=True),
        direction_comparison=direction.reset_index(drop=True),
        context_summary=_concat_frames(context_frames).reset_index(drop=True),
        fold_details=fold_details.reset_index(drop=True),
        candidate_definitions=_candidate_definitions_frame(candidates),
    )


def _candidate_period_summary(
    frame: pd.DataFrame,
    candidate: BoundaryCandidate,
    config: BoundaryExperimentConfig,
) -> pd.DataFrame:
    summary = diagnostics.multiclass_period_summary_frame(
        frame,
        source=candidate.name,
        min_sample_warning=config.min_sample_warning,
    )
    if summary.empty:
        return _no_prediction_summary(candidate)
    rows = summary.copy()
    rows["candidate_name"] = candidate.name
    rows["boundary_mode"] = candidate.boundary_mode
    rows["weight_mode"] = candidate.weight_mode
    rows["class_count"] = candidate.class_count
    rows["notes"] = rows["notes"].fillna("")
    rows["caveats"] = rows["caveats"].fillna("")
    for index, row in rows.iterrows():
        period_frame = _period_subset(frame, str(row["period"]))
        recalls = _class_recalls(period_frame)
        for key, value in recalls.items():
            rows.loc[index, key] = value
        rows.loc[index, "tail_precision"] = _tail_precision(period_frame)
        rows.loc[index, "predicted_surge_count"] = int(
            (period_frame.get("predicted_class", pd.Series(dtype=object)).map(safe_int) == 4).sum()
        )
        rows.loc[index, "predicted_crash_count"] = int(
            (period_frame.get("predicted_class", pd.Series(dtype=object)).map(safe_int) == 0).sum()
        )
        if candidate.class_count == 3:
            rows.loc[index, "caveats"] = _append_text(
                str(rows.loc[index, "caveats"] or ""),
                "3-class candidate folds crash/surge into down/up; tail recall is not directly comparable",
            )
    for column in (
        "direction_accuracy_flat_excluded",
        "direction_coverage",
        "direction_accuracy_flat_neutral",
        "direction_actual_only_accuracy",
    ):
        if column not in rows.columns:
            rows[column] = None
    return rows[_summary_columns()]


def _live_summary(live_frame: pd.DataFrame, config: BoundaryExperimentConfig) -> pd.DataFrame:
    rows = diagnostics.multiclass_source_summary_frame(
        {"current_live_saved": live_frame},
        min_sample_warning=config.min_sample_warning,
    )
    if rows.empty:
        return pd.DataFrame(columns=_summary_columns())
    rows["candidate_name"] = "current_live_saved"
    rows["period"] = "Live only"
    rows["boundary_mode"] = "saved_live"
    rows["weight_mode"] = label_distribution.DEFAULT_WEIGHT_MODE
    rows["class_count"] = 5
    recalls = _class_recalls(live_frame)
    for key, value in recalls.items():
        rows[key] = value
    rows["tail_precision"] = _tail_precision(live_frame)
    for column in (
        "direction_accuracy_flat_excluded",
        "direction_coverage",
        "direction_accuracy_flat_neutral",
        "direction_actual_only_accuracy",
    ):
        if column not in rows.columns:
            rows[column] = None
    return rows[_summary_columns()]


def _binary_reference_summary(binary_frame: pd.DataFrame) -> pd.DataFrame:
    direction = diagnostics.binary_direction_comparison_frame(
        binary_frame,
        source="binary_walk_forward_reference",
        period="Historical all",
    )
    row = {column: None for column in _summary_columns()}
    row.update(
        {
            "candidate_name": "binary_walk_forward_reference",
            "source": "binary_walk_forward_reference",
            "period": "Historical all",
            "boundary_mode": "binary_reference",
            "weight_mode": "",
            "class_count": 2,
            "sample_count": int(direction["sample_count"].iloc[0]) if not direction.empty else 0,
            "direction_accuracy_flat_excluded": (
                safe_float(direction["direction_accuracy"].iloc[0]) if not direction.empty else None
            ),
            "direction_coverage": safe_float(direction["coverage"].iloc[0]) if not direction.empty else None,
            "notes": "Binary walk-forward reference for direction comparison.",
            "caveats": "Not a multiclass candidate.",
        }
    )
    return pd.DataFrame([row], columns=_summary_columns())


def _attach_direction_metrics(summary: pd.DataFrame, direction: pd.DataFrame) -> pd.DataFrame:
    rows = summary.copy()
    for column in (
        "direction_accuracy_flat_excluded",
        "direction_coverage",
        "direction_accuracy_flat_neutral",
        "direction_actual_only_accuracy",
    ):
        if column not in rows.columns:
            rows[column] = None
    if direction.empty:
        return rows[_summary_columns()]

    for index, row in rows.iterrows():
        candidate = row.get("candidate_name")
        period = row.get("period")
        match = direction[
            direction.get("candidate_name", pd.Series(dtype=str)).astype(str).eq(str(candidate))
            & direction.get("period", pd.Series(dtype=str)).astype(str).eq(str(period))
        ]
        if match.empty:
            continue
        flat_excluded = match[match["mode"].eq("flat_excluded")]
        if not flat_excluded.empty:
            rows.loc[index, "direction_accuracy_flat_excluded"] = flat_excluded.iloc[0].get(
                "direction_accuracy"
            )
            rows.loc[index, "direction_coverage"] = flat_excluded.iloc[0].get("coverage")
        flat_neutral = match[match["mode"].eq("flat_as_neutral")]
        if not flat_neutral.empty:
            rows.loc[index, "direction_accuracy_flat_neutral"] = flat_neutral.iloc[0].get(
                "direction_accuracy"
            )
        actual_only = match[match["mode"].eq("actual_direction_only")]
        if not actual_only.empty:
            rows.loc[index, "direction_actual_only_accuracy"] = actual_only.iloc[0].get(
                "direction_accuracy"
            )
        binary = match[match["mode"].eq("binary_up_down")]
        if not binary.empty:
            rows.loc[index, "direction_accuracy_flat_excluded"] = binary.iloc[0].get(
                "direction_accuracy"
            )
            rows.loc[index, "direction_coverage"] = binary.iloc[0].get("coverage")
    return rows[_summary_columns()]


def _period_subset(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    rows = diagnostics._confirmed_multiclass_rows(frame)  # noqa: SLF001
    if rows.empty or period == "Historical all":
        return rows
    dates = pd.to_datetime(rows.get("date"), errors="coerce")
    latest = pd.Timestamp(dates.max()).normalize()
    masks = {
        "2023-": dates >= pd.Timestamp("2023-01-01"),
        "2024-": dates >= pd.Timestamp("2024-01-01"),
        "2025-": dates >= pd.Timestamp("2025-01-01"),
        "2026-": dates >= pd.Timestamp("2026-01-01"),
        "Last 6 months": dates >= latest - pd.DateOffset(months=6),
        "Last 3 months": dates >= latest - pd.DateOffset(months=3),
        "Live only": pd.Series(True, index=rows.index),
    }
    mask = masks.get(period)
    if mask is None:
        return pd.DataFrame()
    return rows[mask].copy()


def _class_recalls(frame: pd.DataFrame) -> dict[str, float | None]:
    rows = diagnostics._confirmed_multiclass_rows(frame)  # noqa: SLF001
    result = {
        "crash_recall": None,
        "down_recall": None,
        "flat_recall": None,
        "up_recall": None,
        "surge_recall": None,
    }
    if rows.empty:
        return result
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    for class_id, class_name in CLASS_NAMES.items():
        mask = actual == class_id
        result[f"{class_name}_recall"] = diagnostics._safe_divide(  # noqa: SLF001
            int((predicted[mask] == class_id).sum()),
            int(mask.sum()),
        )
    return result


def _tail_precision(frame: pd.DataFrame) -> float | None:
    rows = diagnostics._confirmed_multiclass_rows(frame)  # noqa: SLF001
    if rows.empty:
        return None
    actual = rows["actual_class"].map(safe_int)
    predicted = rows["predicted_class"].map(safe_int)
    predicted_tail = predicted.isin([0, 4])
    return diagnostics._safe_divide(  # noqa: SLF001
        int((actual[predicted_tail] == predicted[predicted_tail]).sum()),
        int(predicted_tail.sum()),
    )


def _fold_detail_row(
    *,
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    candidate: BoundaryCandidate,
    thresholds: dict[str, float | None],
    test_start,
    test_end,
) -> dict[str, object]:
    row: dict[str, object] = {
        "candidate_name": candidate.name,
        "boundary_mode": candidate.boundary_mode,
        "weight_mode": candidate.weight_mode,
        "class_count": candidate.class_count,
        "test_start_date": diagnostics._date_text(test_start),  # noqa: SLF001
        "test_end_date": diagnostics._date_text(test_end),  # noqa: SLF001
        "train_start_date": diagnostics._date_text(train_data["date"].min()),  # noqa: SLF001
        "train_end_date": diagnostics._date_text(train_data["date"].max()),  # noqa: SLF001
        "valid_start_date": diagnostics._date_text(valid_data["date"].min()),  # noqa: SLF001
        "valid_end_date": diagnostics._date_text(valid_data["date"].max()),  # noqa: SLF001
        "train_rows": int(len(train_data)),
        "valid_rows": int(len(valid_data)),
        "test_rows": int(len(test_data)),
        "feature_count": int(len(feature_columns)),
        "label_thresholds": json.dumps(thresholds, ensure_ascii=False),
    }
    train_classes = set(train_data[TARGET_COLUMN].astype(int).unique())
    valid_classes = set(valid_data[TARGET_COLUMN].astype(int).unique())
    row["valid_only_classes"] = ",".join(
        CLASS_NAMES.get(class_id, str(class_id)) for class_id in sorted(valid_classes - train_classes)
    )
    for class_id, class_name in CLASS_NAMES.items():
        row[f"train_{class_name}_count"] = int((train_data[TARGET_COLUMN].astype(int) == class_id).sum())
        row[f"test_{class_name}_count"] = int((test_data[TARGET_COLUMN].astype(int) == class_id).sum())
    return row


def _candidate_definitions_frame(candidates: list[BoundaryCandidate]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": candidate.name,
                "boundary_mode": candidate.boundary_mode,
                "weight_mode": candidate.weight_mode,
                "class_count": candidate.class_count,
                "crash_threshold": candidate.crash_threshold,
                "down_threshold": candidate.down_threshold,
                "up_threshold": candidate.up_threshold,
                "surge_threshold": candidate.surge_threshold,
                "flat_threshold": candidate.flat_threshold,
                "note": candidate.note,
            }
            for candidate in candidates
        ]
    )


def _summary_columns() -> list[str]:
    return [
        "candidate_name",
        "source",
        "period",
        "boundary_mode",
        "weight_mode",
        "class_count",
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
        "up_recall",
        "down_recall",
        "flat_recall",
        "surge_recall",
        "crash_recall",
        "direction_accuracy_flat_excluded",
        "direction_coverage",
        "direction_accuracy_flat_neutral",
        "direction_actual_only_accuracy",
        "predicted_flat_actual_return_mean",
        "predicted_flat_actual_abs_return_mean",
        "actual_up_down_predicted_flat_count",
        "actual_tail_predicted_flat_count",
        "surge_support",
        "crash_support",
        "predicted_surge_count",
        "predicted_crash_count",
        "tail_precision",
        "confidence_mean",
        "confidence_median",
        "confidence_max",
        "high_confidence_count",
        "high_confidence_rate",
        "notes",
        "caveats",
    ]


def _no_prediction_summary(candidate: BoundaryCandidate) -> pd.DataFrame:
    row = {column: None for column in _summary_columns()}
    row.update(
        {
            "candidate_name": candidate.name,
            "source": candidate.name,
            "period": "Historical all",
            "boundary_mode": candidate.boundary_mode,
            "weight_mode": candidate.weight_mode,
            "class_count": candidate.class_count,
            "sample_count": 0,
            "warning": "no predictions",
            "notes": candidate.note,
            "caveats": "no predictions",
        }
    )
    return pd.DataFrame([row], columns=_summary_columns())


def _empty_result(candidates: list[BoundaryCandidate]) -> BoundaryExperimentResult:
    summary = pd.concat([_no_prediction_summary(candidate) for candidate in candidates], ignore_index=True)
    return BoundaryExperimentResult(
        summary=summary,
        predictions=pd.DataFrame(),
        class_metrics=pd.DataFrame(),
        confusion_matrix=pd.DataFrame(),
        class_distribution=pd.DataFrame(),
        confidence_buckets=pd.DataFrame(),
        probability_summary=pd.DataFrame(),
        direction_comparison=pd.DataFrame(),
        context_summary=pd.DataFrame(),
        fold_details=pd.DataFrame(),
        candidate_definitions=_candidate_definitions_frame(candidates),
        conclusions=["production_change_not_ready"],
    )


def _thresholds_complete(thresholds: dict[str, float | None]) -> bool:
    return all(
        thresholds.get(key) is not None
        for key in ("crash_threshold", "down_threshold", "up_threshold", "surge_threshold")
    )


def _model_class_ids(model, probability_column_count: int) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return list(range(probability_column_count))
    result: list[int] = []
    for value in classes:
        if hasattr(value, "item"):
            value = value.item()
        result.append(int(value))
    if len(result) != probability_column_count:
        return list(range(probability_column_count))
    return result


def _candidate_row(rows: pd.DataFrame, name: str) -> pd.Series | None:
    match = rows[rows["candidate_name"].astype(str).eq(name)]
    if match.empty:
        return None
    return match.iloc[0]


def _append_text(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"


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
