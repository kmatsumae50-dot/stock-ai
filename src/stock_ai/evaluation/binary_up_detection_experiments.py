from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stock_ai.backtest.walk_forward import TARGET_COLUMN
from stock_ai.evaluation import accuracy_diagnostics
from stock_ai.evaluation import offline_model_experiments as offline
from stock_ai.models.baseline import select_feature_columns
from stock_ai.utils.safe_cast import safe_float, safe_int


PERIOD_ORDER = (
    "Historical all",
    "2023-",
    "2024-",
    "2025-",
    "2026-",
    "Last 6 months",
    "Last 3 months",
    "Live only",
)
CONFIDENCE_THRESHOLDS = (0.55, 0.60, 0.65)
WEAK_MARGIN = 0.06
MIN_REGIME_ROWS = 30


@dataclass(frozen=True)
class BinaryUpCandidate:
    name: str
    feature_group: str = "full_features"
    weight_mode: str = "none"
    threshold_mode: str = "fixed_0_50"
    hybrid_mode: str = "none"
    note: str = ""


@dataclass(frozen=True)
class BinaryUpExperimentConfig:
    ticker: str = "7203.T"
    train_years: int = 3
    test_step_months: int = 1
    start_date: str | None = None
    end_date: str | None = None
    min_train_rows: int = 252
    valid_fraction: float = 0.2
    min_sample_warning: int = 100


@dataclass
class BinaryUpDetectionResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_details: pd.DataFrame
    confidence_buckets: pd.DataFrame
    failure_patterns: pd.DataFrame
    failure_common_features: pd.DataFrame
    candidate_definitions: pd.DataFrame
    conclusions: list[str] = field(default_factory=list)
    report_paths: dict[str, str] = field(default_factory=dict)


def default_up_detection_candidates() -> list[BinaryUpCandidate]:
    return [
        BinaryUpCandidate(
            name="current_model",
            note="Production-style binary LightGBM settings with 0.50 threshold.",
        ),
        BinaryUpCandidate(
            name="class_weight_balanced",
            weight_mode="class_weight_balanced",
            note="LightGBM class_weight='balanced' inside each fold from train data only.",
        ),
        BinaryUpCandidate(
            name="class_weight_up_boost_mild",
            weight_mode="class_weight_up_boost_mild",
            note="Manual mild up-class weight. Exploratory only.",
        ),
        BinaryUpCandidate(
            name="class_weight_up_boost_medium",
            weight_mode="class_weight_up_boost_medium",
            note="Manual medium up-class weight. Exploratory only.",
        ),
        BinaryUpCandidate(
            name="scale_pos_weight_mild",
            weight_mode="scale_pos_weight_mild",
            note="Train-fold class ratio with mild positive multiplier.",
        ),
        BinaryUpCandidate(
            name="scale_pos_weight_medium",
            weight_mode="scale_pos_weight_medium",
            note="Train-fold class ratio with medium positive multiplier.",
        ),
        BinaryUpCandidate(
            name="balanced_accuracy_optimized_threshold",
            threshold_mode="balanced_accuracy",
            note="Threshold selected on validation rows only for balanced accuracy.",
        ),
        BinaryUpCandidate(
            name="up_recall_mild_threshold",
            threshold_mode="up_recall_mild",
            note="Validation-only threshold preferring up recall while keeping down recall >=45% when possible.",
        ),
        BinaryUpCandidate(
            name="up_recall_medium_threshold",
            threshold_mode="up_recall_medium",
            note="Validation-only threshold preferring up recall with looser down recall guard.",
        ),
        BinaryUpCandidate(
            name="precision_guarded_up_threshold",
            threshold_mode="precision_guarded",
            note="Validation-only threshold requiring reasonable up precision before increasing up calls.",
        ),
        BinaryUpCandidate(
            name="previous_day_direction_hybrid",
            hybrid_mode="previous_day_when_weak",
            note="If model probability margin is weak, use previous-day direction from feature-date data.",
        ),
        BinaryUpCandidate(
            name="moving_average_trend_hybrid",
            hybrid_mode="ma_when_weak",
            note="If model probability margin is weak, use MA trend from feature-date data.",
        ),
        BinaryUpCandidate(
            name="model_ma_agreement_hybrid",
            hybrid_mode="model_ma_agreement",
            note="Treat model-up / MA-down disagreement conservatively as down.",
        ),
        BinaryUpCandidate(
            name="regime_aware_threshold",
            threshold_mode="regime_aware",
            note="Regime thresholds selected on validation rows only when regime sample count is sufficient.",
        ),
        BinaryUpCandidate(
            name="bear_reversal_candidate_threshold",
            hybrid_mode="bear_reversal",
            note="Exploratory rule: allow mild-up calls in weak bearish/oversold contexts. No target data is used.",
        ),
        BinaryUpCandidate(
            name="external_features_excluded",
            feature_group="no_external_sector_features",
            note="Drops external market and auto-sector features during training only.",
        ),
        BinaryUpCandidate(
            name="technical_only",
            feature_group="technical_price_only",
            note="Uses price/technical features only; feature generation is unchanged.",
        ),
        BinaryUpCandidate(
            name="market_external_only",
            feature_group="external_market_only",
            note="Uses only external market and auto-sector numeric features.",
        ),
        BinaryUpCandidate(
            name="news_excluded",
            feature_group="no_news_features",
            note="Drops news and narrative-derived features during training only.",
        ),
    ]


def run_binary_up_detection_experiments(
    data: pd.DataFrame,
    config: BinaryUpExperimentConfig | None = None,
    *,
    candidates: list[BinaryUpCandidate] | None = None,
    live_frame: pd.DataFrame | None = None,
    model_factory=None,
) -> BinaryUpDetectionResult:
    config = config or BinaryUpExperimentConfig()
    candidates = candidates or default_up_detection_candidates()
    frame = offline._prepare_experiment_frame(  # noqa: SLF001
        data,
        offline.OfflineExperimentConfig(
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
        predictions, details = _run_candidate(frame, config, candidate, model_factory=model_factory)
        if not predictions.empty:
            prediction_frames.append(predictions)
        if not details.empty:
            detail_frames.append(details)

    predictions = _concat_frames(prediction_frames)
    if not predictions.empty:
        predictions = predictions.sort_values(["candidate_name", "date"]).reset_index(drop=True)
    fold_details = _concat_frames(detail_frames)
    result = _build_result(
        predictions=predictions,
        fold_details=fold_details,
        candidates=candidates,
        config=config,
        live_frame=live_frame,
    )
    result.conclusions = classify_binary_up_conclusions(result.summary, result.confidence_buckets)
    return result


def write_binary_up_detection_reports(
    result: BinaryUpDetectionResult,
    output_dir: str | Path,
    *,
    run_date: str | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = run_date or date.today().isoformat()
    csv_path = output / f"binary_up_detection_experiments_{stamp}.csv"
    json_path = output / f"binary_up_detection_experiments_{stamp}.json"
    md_path = output / f"binary_up_detection_experiments_{stamp}.md"

    result.summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "conclusions": result.conclusions,
        "candidate_definitions": _json_records(result.candidate_definitions),
        "summary": _json_records(result.summary),
        "confidence_buckets": _json_records(result.confidence_buckets),
        "failure_patterns": _json_records(result.failure_patterns),
        "failure_common_features": _json_records(result.failure_common_features),
        "fold_details": _json_records(result.fold_details),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(_markdown_report(result), encoding="utf-8")
    result.report_paths = {"csv": str(csv_path), "json": str(json_path), "md": str(md_path)}
    return result.report_paths


def classify_binary_up_conclusions(summary: pd.DataFrame, confidence_buckets: pd.DataFrame | None = None) -> list[str]:
    if summary.empty:
        return ["no_candidate_ready_for_production", "production_change_not_ready"]

    historical = summary[summary["period"].eq("Historical all")].copy()
    for column in (
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
    ):
        historical[column] = pd.to_numeric(historical.get(column), errors="coerce")
    current = _candidate_row(historical, "current_model")
    current_up = safe_float(current.get("up_recall")) if current is not None else None
    current_balanced = safe_float(current.get("balanced_accuracy")) if current is not None else None
    conclusions: list[str] = []

    non_current = historical[~historical["candidate_name"].isin(["current_model", "current_live_saved"])].copy()
    if not non_current.empty and current_up is not None:
        weight_rows = non_current[non_current["candidate_name"].astype(str).str.contains("weight|scale_pos")]
        if pd.to_numeric(weight_rows.get("up_recall"), errors="coerce").max() >= current_up + 0.05:
            conclusions.append("mild_up_weight_promising")
        threshold_rows = non_current[non_current["candidate_name"].astype(str).str.contains("threshold")]
        if pd.to_numeric(threshold_rows.get("up_recall"), errors="coerce").max() >= current_up + 0.05:
            conclusions.append("balanced_threshold_promising")
        hybrid_rows = non_current[non_current["candidate_name"].astype(str).str.contains("hybrid")]
        if pd.to_numeric(hybrid_rows.get("balanced_accuracy"), errors="coerce").max() >= (
            (current_balanced or 0.0) + 0.01
        ):
            conclusions.append("hybrid_baseline_promising")

    if not non_current.empty:
        candidate_up = pd.to_numeric(non_current.get("up_recall"), errors="coerce")
        candidate_precision = pd.to_numeric(non_current.get("up_precision"), errors="coerce")
        if candidate_up.max() >= (current_up or 0.0) + 0.08 and candidate_precision.max() < 0.45:
            conclusions.append("up_recall_improves_but_precision_collapses")
        down_damaged = non_current[
            (pd.to_numeric(non_current.get("up_recall"), errors="coerce") >= (current_up or 0.0) + 0.05)
            & (pd.to_numeric(non_current.get("down_recall"), errors="coerce") < 0.45)
        ]
        if not down_damaged.empty:
            conclusions.append("down_recall_too_damaged")
        baseline_cols = [
            "model_minus_always_down",
            "model_minus_previous_day_direction",
            "model_minus_moving_average_trend_only",
        ]
        baseline_deltas = non_current[baseline_cols].apply(pd.to_numeric, errors="coerce")
        if baseline_deltas.empty or float(baseline_deltas.max().max()) <= 0.0:
            conclusions.append("baseline_still_stronger")

    if confidence_buckets is not None and not confidence_buckets.empty:
        current_conf = confidence_buckets[
            confidence_buckets["candidate_name"].eq("current_model")
            & confidence_buckets["period"].eq("Historical all")
        ].copy()
        hit_rates = pd.to_numeric(current_conf.get("hit_rate"), errors="coerce").dropna()
        if len(hit_rates) >= 2 and hit_rates.is_monotonic_decreasing:
            conclusions.append("confidence_reverse_signal_risk")
        if hit_rates.empty or hit_rates.max() < 0.55:
            conclusions.append("confidence_unreliable")

    current_best = pd.to_numeric(non_current.get("balanced_accuracy"), errors="coerce").max()
    if pd.isna(current_best) or current_best < max((current_balanced or 0.0) + 0.015, 0.53):
        conclusions.append("no_candidate_ready_for_production")
    conclusions.append("production_change_not_ready")
    conclusions.append("next_experiment_needed")
    return list(dict.fromkeys(conclusions))


def _run_candidate(
    frame: pd.DataFrame,
    config: BinaryUpExperimentConfig,
    candidate: BinaryUpCandidate,
    *,
    model_factory=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_predictions: list[pd.DataFrame] = []
    fold_details: list[dict[str, object]] = []
    windows = offline._test_windows(  # noqa: SLF001
        frame,
        offline.OfflineExperimentConfig(
            ticker=config.ticker,
            train_years=config.train_years,
            test_step_months=config.test_step_months,
            start_date=config.start_date,
            end_date=config.end_date,
            min_train_rows=config.min_train_rows,
            valid_fraction=config.valid_fraction,
        ),
    )
    for test_start, test_end in windows:
        train_window = offline._training_window(  # noqa: SLF001
            frame,
            test_start,
            offline.OfflineExperimentConfig(
                ticker=config.ticker,
                train_years=config.train_years,
                min_train_rows=config.min_train_rows,
                valid_fraction=config.valid_fraction,
            ),
        )
        test_window = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        test_window = test_window[test_window[TARGET_COLUMN].isin([0, 1])]
        if len(train_window) < config.min_train_rows or test_window.empty:
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        feature_columns = offline._filter_feature_group(feature_columns, candidate.feature_group)  # noqa: SLF001
        feature_columns = offline._available_feature_columns(  # noqa: SLF001
            train_window,
            test_window,
            feature_columns,
        )
        if not feature_columns:
            continue

        train_window = train_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        test_window = test_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        train_window = train_window[train_window[TARGET_COLUMN].isin([0, 1])]
        if len(train_window) < config.min_train_rows or test_window.empty or train_window[TARGET_COLUMN].nunique() < 2:
            continue

        train_data, valid_data = offline._train_valid_split(train_window, config.valid_fraction)  # noqa: SLF001
        if train_data.empty or valid_data.empty or train_data[TARGET_COLUMN].nunique() < 2:
            continue

        model = (
            model_factory(train_data, valid_data, feature_columns, TARGET_COLUMN, candidate)
            if model_factory is not None
            else _fit_lightgbm_binary(train_data, valid_data, feature_columns, candidate)
        )
        valid_prob = offline._predict_up_probability(model, valid_data[feature_columns])  # noqa: SLF001
        threshold_info = _select_thresholds(
            candidate,
            valid_data,
            valid_data[TARGET_COLUMN].astype(int),
            valid_prob,
        )
        predictions = _predict_candidate_fold(
            model=model,
            test_data=test_window,
            feature_columns=feature_columns,
            candidate=candidate,
            threshold_info=threshold_info,
            train_data=train_data,
            valid_data=valid_data,
            test_start=test_start,
            test_end=test_end,
        )
        fold_predictions.append(predictions)
        fold_details.append(
            {
                "candidate_name": candidate.name,
                "test_start_date": offline._date_text(test_start),  # noqa: SLF001
                "test_end_date": offline._date_text(test_end),  # noqa: SLF001
                "train_rows": int(len(train_data)),
                "valid_rows": int(len(valid_data)),
                "test_rows": int(len(test_window)),
                "feature_count": int(len(feature_columns)),
                "threshold": threshold_info.get("global_threshold"),
                "threshold_mode": candidate.threshold_mode,
                "hybrid_mode": candidate.hybrid_mode,
                "threshold_source": "validation_only",
                "regime_threshold_count": len(threshold_info.get("regime_thresholds", {})),
                "regime_thresholds": json.dumps(threshold_info.get("regime_thresholds", {}), ensure_ascii=False),
            }
        )

    return _concat_frames(fold_predictions), pd.DataFrame(fold_details)


def _fit_lightgbm_binary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    candidate: BinaryUpCandidate,
):
    import lightgbm as lgb

    params: dict[str, object] = {
        "objective": "binary",
        "n_estimators": 150,
        "learning_rate": 0.04,
        "num_leaves": 31,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbose": -1,
    }
    y_train = train[TARGET_COLUMN].astype(int)
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    if candidate.weight_mode == "class_weight_balanced":
        params["class_weight"] = "balanced"
    elif candidate.weight_mode == "class_weight_up_boost_mild":
        params["class_weight"] = {0: 1.0, 1: 1.20}
    elif candidate.weight_mode == "class_weight_up_boost_medium":
        params["class_weight"] = {0: 1.0, 1: 1.45}
    elif candidate.weight_mode == "scale_pos_weight_mild" and positives > 0:
        params["scale_pos_weight"] = (negatives / positives) * 1.15
    elif candidate.weight_mode == "scale_pos_weight_medium" and positives > 0:
        params["scale_pos_weight"] = (negatives / positives) * 1.35

    model = lgb.LGBMClassifier(**params)
    model.fit(
        train[feature_columns],
        train[TARGET_COLUMN].astype(int),
        eval_set=[(valid[feature_columns], valid[TARGET_COLUMN].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _select_thresholds(
    candidate: BinaryUpCandidate,
    valid_data: pd.DataFrame,
    valid_actual: pd.Series,
    valid_probability: pd.Series,
) -> dict[str, object]:
    mode = candidate.threshold_mode
    if mode == "balanced_accuracy":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="balanced_accuracy")}
    if mode == "up_recall_mild":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="up_recall_mild")}
    if mode == "up_recall_medium":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="up_recall_medium")}
    if mode == "precision_guarded":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="precision_guarded")}
    if mode == "regime_aware":
        global_threshold = _best_threshold(valid_actual, valid_probability, metric="balanced_accuracy")
        thresholds: dict[str, float] = {}
        if "regime" in valid_data.columns:
            rows = pd.DataFrame(
                {
                    "regime": valid_data["regime"].fillna("Unknown").astype(str),
                    "actual": valid_actual.astype(int),
                    "probability": valid_probability,
                }
            )
            for regime, group in rows.groupby("regime"):
                if len(group) < MIN_REGIME_ROWS or group["actual"].nunique() < 2:
                    continue
                thresholds[str(regime)] = _best_threshold(
                    group["actual"],
                    group["probability"],
                    metric="balanced_accuracy",
                )
        return {"global_threshold": global_threshold, "regime_thresholds": thresholds}
    return {"global_threshold": 0.5}


def _best_threshold(actual: pd.Series, probability: pd.Series, *, metric: str) -> float:
    rows = pd.DataFrame({"actual": actual, "probability": probability}).dropna()
    if rows.empty or rows["actual"].nunique() < 2:
        return 0.5
    best_threshold = 0.5
    best_score = -1.0
    best_down = -1.0
    for threshold in offline.THRESHOLD_GRID:
        predicted = (rows["probability"] >= threshold).astype(int)
        metrics = offline._binary_metrics(rows["actual"].astype(int), predicted)  # noqa: SLF001
        up_recall = safe_float(metrics.get("up_recall")) or 0.0
        down_recall = safe_float(metrics.get("down_recall")) or 0.0
        up_precision = safe_float(metrics.get("up_precision")) or 0.0
        balanced = safe_float(metrics.get("balanced_accuracy")) or 0.0
        accuracy = safe_float(metrics.get("accuracy")) or 0.0
        if metric == "up_recall_mild":
            score = up_recall + 0.20 * up_precision + 0.10 * accuracy
            if down_recall < 0.45:
                score *= 0.60
        elif metric == "up_recall_medium":
            score = up_recall + 0.10 * up_precision
            if down_recall < 0.35:
                score *= 0.65
        elif metric == "precision_guarded":
            score = up_recall + 0.25 * up_precision + 0.15 * down_recall
            if up_precision < 0.45 or down_recall < 0.45:
                score *= 0.55
        else:
            score = balanced
        if score > best_score or (score == best_score and down_recall > best_down):
            best_threshold = float(threshold)
            best_score = score
            best_down = down_recall
    return best_threshold


def _predict_candidate_fold(
    *,
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    candidate: BinaryUpCandidate,
    threshold_info: dict[str, object],
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_start,
    test_end,
) -> pd.DataFrame:
    probabilities = offline._predict_up_probability(model, test_data[feature_columns])  # noqa: SLF001
    rows = test_data.copy()
    rows["probability_up"] = probabilities
    rows["probability_down"] = 1.0 - rows["probability_up"]
    rows["decision_threshold"] = _threshold_for_rows(rows, threshold_info)
    rows["raw_predicted_class"] = (rows["probability_up"] >= rows["decision_threshold"]).astype(int)
    rows = _attach_baseline_columns(rows)
    rows["predicted_class"] = _apply_hybrid_prediction(rows, candidate).astype(int)
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = rows[["probability_up", "probability_down"]].max(axis=1)
    rows["prediction_direction"] = rows["predicted_class"].map(lambda value: "up" if value == 1 else "down")
    rows["actual_direction"] = rows["actual_class"].map(lambda value: "up" if value == 1 else "down")
    rows["candidate_name"] = candidate.name
    rows["feature_group"] = candidate.feature_group
    rows["weight_mode"] = candidate.weight_mode
    rows["threshold_mode"] = candidate.threshold_mode
    rows["hybrid_mode"] = candidate.hybrid_mode
    rows["model_name"] = candidate.name
    rows["target"] = "next_close_binary"
    rows["prediction_type"] = f"offline_up_detection_{candidate.name}"
    rows["train_start_date"] = pd.to_datetime(train_data["date"].min())
    rows["train_end_date"] = pd.to_datetime(train_data["date"].max())
    rows["valid_start_date"] = pd.to_datetime(valid_data["date"].min())
    rows["valid_end_date"] = pd.to_datetime(valid_data["date"].max())
    rows["test_start_date"] = pd.to_datetime(test_start)
    rows["test_end_date"] = pd.to_datetime(test_end)
    rows["feature_count"] = len(feature_columns)
    rows["hybrid_changed_prediction"] = rows["raw_predicted_class"].astype(int) != rows["predicted_class"].astype(int)

    keep = [
        "date",
        "ticker",
        "candidate_name",
        "model_name",
        "target",
        "prediction_type",
        "predicted_class",
        "raw_predicted_class",
        "actual_class",
        "prediction_direction",
        "actual_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "decision_threshold",
        "next_close_return",
        "regime",
        "anomaly_score",
        "anomaly_level",
        "rsi_14",
        "ma_5",
        "ma_25",
        "return_1d",
        "baseline_previous_day_direction",
        "baseline_moving_average_trend",
        "hybrid_changed_prediction",
        "feature_group",
        "weight_mode",
        "threshold_mode",
        "hybrid_mode",
        "feature_count",
        "train_start_date",
        "train_end_date",
        "valid_start_date",
        "valid_end_date",
        "test_start_date",
        "test_end_date",
    ]
    return rows[[column for column in keep if column in rows.columns]]


def _attach_baseline_columns(rows: pd.DataFrame) -> pd.DataFrame:
    result = rows.copy()
    if "return_1d" in result.columns:
        result["baseline_previous_day_direction"] = (
            pd.to_numeric(result["return_1d"], errors="coerce") > 0
        ).astype(int)
    else:
        result["baseline_previous_day_direction"] = 0
    if {"ma_5", "ma_25"}.issubset(result.columns):
        result["baseline_moving_average_trend"] = (
            pd.to_numeric(result["ma_5"], errors="coerce")
            >= pd.to_numeric(result["ma_25"], errors="coerce")
        ).astype(int)
    else:
        result["baseline_moving_average_trend"] = result["baseline_previous_day_direction"]
    if "regime" not in result.columns:
        result["regime"] = "Unknown"
    if "anomaly_level" not in result.columns:
        result["anomaly_level"] = "Unknown"
    return result


def _apply_hybrid_prediction(rows: pd.DataFrame, candidate: BinaryUpCandidate) -> pd.Series:
    predicted = rows["raw_predicted_class"].astype(int).copy()
    margin = (pd.to_numeric(rows["probability_up"], errors="coerce") - 0.5).abs().fillna(1.0)
    weak = margin <= WEAK_MARGIN
    if candidate.hybrid_mode == "previous_day_when_weak":
        baseline = rows["baseline_previous_day_direction"].map(safe_int).fillna(predicted).astype(int)
        predicted.loc[weak] = baseline.loc[weak]
    elif candidate.hybrid_mode == "ma_when_weak":
        baseline = rows["baseline_moving_average_trend"].map(safe_int).fillna(predicted).astype(int)
        predicted.loc[weak] = baseline.loc[weak]
    elif candidate.hybrid_mode == "model_ma_agreement":
        ma_down = rows["baseline_moving_average_trend"].map(safe_int).fillna(0).astype(int).eq(0)
        predicted.loc[predicted.eq(1) & ma_down] = 0
    elif candidate.hybrid_mode == "bear_reversal":
        rsi = pd.to_numeric(rows.get("rsi_14"), errors="coerce")
        prev_down = rows["baseline_previous_day_direction"].map(safe_int).fillna(0).astype(int).eq(0)
        ma_down = rows["baseline_moving_average_trend"].map(safe_int).fillna(0).astype(int).eq(0)
        mild_prob = pd.to_numeric(rows["probability_up"], errors="coerce").fillna(0.0) >= 0.46
        oversold = rsi.notna() & (rsi <= 45)
        predicted.loc[mild_prob & prev_down & ma_down & oversold] = 1
    return predicted


def _threshold_for_rows(rows: pd.DataFrame, threshold_info: dict[str, object]) -> pd.Series:
    global_threshold = safe_float(threshold_info.get("global_threshold")) or 0.5
    thresholds = threshold_info.get("regime_thresholds") or {}
    if not thresholds or "regime" not in rows.columns:
        return pd.Series(global_threshold, index=rows.index, dtype=float)
    return rows["regime"].fillna("Unknown").astype(str).map(thresholds).fillna(global_threshold).astype(float)


def _build_result(
    *,
    predictions: pd.DataFrame,
    fold_details: pd.DataFrame,
    candidates: list[BinaryUpCandidate],
    config: BinaryUpExperimentConfig,
    live_frame: pd.DataFrame | None,
) -> BinaryUpDetectionResult:
    summary_frames: list[pd.DataFrame] = []
    confidence_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        frame = predictions[predictions.get("candidate_name", pd.Series(dtype=str)).eq(candidate.name)].copy()
        if frame.empty:
            summary_frames.append(_no_prediction_summary(candidate))
            continue
        summary_frames.append(_candidate_period_summary(frame, candidate, config))
        confidence_frames.append(_candidate_confidence_buckets(frame, candidate.name))

    live_failures = pd.DataFrame()
    live_common = pd.DataFrame()
    if live_frame is not None and not live_frame.empty:
        live_rows = _confirmed(live_frame)
        if not live_rows.empty:
            summary_frames.append(_live_summary(live_rows, config))
            confidence_frames.append(_candidate_confidence_buckets(live_rows, "current_live_saved", period="Live only"))
            live_failures = _failure_patterns(live_rows)
            live_common = _failure_common_features(live_failures)

    summary = _concat_frames(summary_frames)
    confidence = _concat_frames(confidence_frames)
    summary = _attach_confidence_summary(summary, confidence)
    summary = _attach_constraints(summary)

    return BinaryUpDetectionResult(
        summary=summary.reset_index(drop=True),
        predictions=predictions.reset_index(drop=True) if not predictions.empty else predictions,
        fold_details=fold_details.reset_index(drop=True) if not fold_details.empty else fold_details,
        confidence_buckets=confidence.reset_index(drop=True) if not confidence.empty else confidence,
        failure_patterns=live_failures.reset_index(drop=True) if not live_failures.empty else live_failures,
        failure_common_features=live_common.reset_index(drop=True) if not live_common.empty else live_common,
        candidate_definitions=_candidate_definitions_frame(candidates),
    )


def _candidate_period_summary(
    frame: pd.DataFrame,
    candidate: BinaryUpCandidate,
    config: BinaryUpExperimentConfig,
) -> pd.DataFrame:
    rows = _confirmed(frame)
    result = []
    for period in PERIOD_ORDER[:-1]:
        subset = _period_subset(rows, period)
        result.append(_summary_row(subset, candidate.name, period, config))
    summary = pd.DataFrame(result, columns=_summary_columns())
    summary["feature_group"] = candidate.feature_group
    summary["weight_mode"] = candidate.weight_mode
    summary["threshold_mode"] = candidate.threshold_mode
    summary["hybrid_mode"] = candidate.hybrid_mode
    summary["notes"] = candidate.note
    return summary


def _live_summary(frame: pd.DataFrame, config: BinaryUpExperimentConfig) -> pd.DataFrame:
    row = _summary_row(frame, "current_live_saved", "Live only", config)
    row.update(
        {
            "feature_group": "production_saved",
            "weight_mode": "production_saved",
            "threshold_mode": "production_saved",
            "hybrid_mode": "none",
            "notes": "Saved live binary predictions from SQLite. This is observed operation, not an offline candidate.",
            "caveats": _append_text(str(row.get("caveats") or ""), "low sample and operational-history only"),
        }
    )
    return pd.DataFrame([row], columns=_summary_columns())


def _summary_row(
    frame: pd.DataFrame,
    candidate_name: str,
    period: str,
    config: BinaryUpExperimentConfig,
) -> dict[str, object]:
    row = {column: None for column in _summary_columns()}
    row.update({"candidate_name": candidate_name, "period": period})
    rows = _confirmed(frame)
    if rows.empty:
        row.update({"sample_count": 0, "warning": "no confirmed rows", "caveats": "no confirmed rows"})
        return row

    actual = rows["actual_class"].map(safe_int).astype(int)
    predicted = rows["predicted_class"].map(safe_int).astype(int)
    metrics = offline._binary_metrics(actual, predicted)  # noqa: SLF001
    baselines = _baseline_metrics(rows)
    confidence = pd.to_numeric(rows.get("confidence"), errors="coerce").dropna()
    start, end = _date_range(rows)
    sample_count = int(len(rows))
    predicted_up_count = int((predicted == 1).sum())
    predicted_down_count = int((predicted == 0).sum())
    actual_up_count = int((actual == 1).sum())
    actual_down_count = int((actual == 0).sum())
    caveats = []
    if sample_count < config.min_sample_warning:
        caveats.append("low sample count")
    if predicted_up_count == 0:
        caveats.append("no predicted up rows")
    if metrics.get("down_recall") is not None and float(metrics["down_recall"]) < 0.45:
        caveats.append("down recall below 45% guard")
    row.update(
        {
            "sample_count": sample_count,
            "warning": "low sample" if sample_count < config.min_sample_warning else "",
            "evaluated_start": start,
            "evaluated_end": end,
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "up_recall": metrics["up_recall"],
            "down_recall": metrics["down_recall"],
            "up_precision": metrics["up_precision"],
            "down_precision": metrics["down_precision"],
            "predicted_up_ratio": _safe_divide(predicted_up_count, sample_count),
            "predicted_down_ratio": _safe_divide(predicted_down_count, sample_count),
            "predicted_up_count": predicted_up_count,
            "predicted_down_count": predicted_down_count,
            "actual_up_count": actual_up_count,
            "actual_down_count": actual_down_count,
            "confidence_mean": float(confidence.mean()) if not confidence.empty else None,
            "confidence_median": float(confidence.median()) if not confidence.empty else None,
            "confidence_max": float(confidence.max()) if not confidence.empty else None,
            "always_up": baselines.get("always_up"),
            "always_down": baselines.get("always_down"),
            "previous_day_direction": baselines.get("previous_day_direction"),
            "moving_average_trend_only": baselines.get("moving_average_trend_only"),
            "model_minus_always_up": _delta(metrics["accuracy"], baselines.get("always_up")),
            "model_minus_always_down": _delta(metrics["accuracy"], baselines.get("always_down")),
            "model_minus_previous_day_direction": _delta(metrics["accuracy"], baselines.get("previous_day_direction")),
            "model_minus_moving_average_trend_only": _delta(metrics["accuracy"], baselines.get("moving_average_trend_only")),
            "caveats": "; ".join(caveats),
        }
    )
    return row


def _baseline_metrics(rows: pd.DataFrame) -> dict[str, float | None]:
    actual = rows["actual_class"].map(safe_int)
    values = {
        "always_up": pd.Series(1, index=rows.index),
        "always_down": pd.Series(0, index=rows.index),
        "previous_day_direction": rows.get("baseline_previous_day_direction", pd.Series(index=rows.index)).map(safe_int),
        "moving_average_trend_only": rows.get("baseline_moving_average_trend", pd.Series(index=rows.index)).map(safe_int),
    }
    result: dict[str, float | None] = {}
    for name, prediction in values.items():
        prediction = pd.Series(prediction, index=rows.index).map(safe_int)
        valid = actual.notna() & prediction.notna()
        result[name] = _safe_divide(int((actual[valid].astype(int) == prediction[valid].astype(int)).sum()), int(valid.sum()))
    return result


def _candidate_confidence_buckets(
    frame: pd.DataFrame,
    candidate_name: str,
    *,
    period: str | None = None,
) -> pd.DataFrame:
    rows = _confirmed(frame)
    result = []
    periods = [period] if period else list(PERIOD_ORDER[:-1])
    for period_name in periods:
        subset = rows if period_name == "Live only" else _period_subset(rows, period_name)
        if subset.empty:
            for threshold in CONFIDENCE_THRESHOLDS:
                result.append(_confidence_row(candidate_name, period_name, threshold, subset))
            continue
        for threshold in CONFIDENCE_THRESHOLDS:
            result.append(_confidence_row(candidate_name, period_name, threshold, subset))
    return pd.DataFrame(result)


def _confidence_row(candidate_name: str, period: str, threshold: float, rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        return {
            "candidate_name": candidate_name,
            "period": period,
            "threshold": threshold,
            "sample_count": 0,
            "hit_count": 0,
            "miss_count": 0,
            "hit_rate": None,
            "coverage": None,
            "average_confidence": None,
        }
    confidence = pd.to_numeric(rows.get("confidence"), errors="coerce")
    subset = rows[confidence >= threshold].copy()
    if "is_correct" in subset.columns:
        correct = subset["is_correct"].fillna(False).astype(bool)
    elif {"actual_class", "predicted_class"}.issubset(subset.columns):
        correct = subset["actual_class"].map(safe_int).eq(subset["predicted_class"].map(safe_int))
    else:
        correct = pd.Series(False, index=subset.index)
    hit_count = int(correct.sum())
    sample_count = int(len(subset))
    return {
        "candidate_name": candidate_name,
        "period": period,
        "threshold": threshold,
        "sample_count": sample_count,
        "hit_count": hit_count,
        "miss_count": sample_count - hit_count,
        "hit_rate": _safe_divide(hit_count, sample_count),
        "coverage": _safe_divide(sample_count, len(rows)),
        "average_confidence": _mean(subset.get("confidence")),
    }


def _attach_confidence_summary(summary: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    rows = summary.copy()
    rows["confidence_bucket_result"] = ""
    if confidence.empty or summary.empty:
        return rows
    for index, row in rows.iterrows():
        match = confidence[
            confidence["candidate_name"].astype(str).eq(str(row.get("candidate_name")))
            & confidence["period"].astype(str).eq(str(row.get("period")))
        ]
        if match.empty:
            continue
        payload = [
            {
                "threshold": safe_float(item.get("threshold")),
                "sample_count": safe_int(item.get("sample_count")) or 0,
                "hit_rate": safe_float(item.get("hit_rate")),
            }
            for item in match.to_dict(orient="records")
        ]
        rows.loc[index, "confidence_bucket_result"] = json.dumps(payload, ensure_ascii=False)
    return rows


def _attach_constraints(summary: pd.DataFrame) -> pd.DataFrame:
    rows = summary.copy()
    rows["constraint_passed"] = False
    rows["constraint_notes"] = ""
    if rows.empty:
        return rows
    for period, group in rows.groupby("period", dropna=False):
        current = _candidate_row(group, "current_model")
        if current is None:
            continue
        current_accuracy = safe_float(current.get("accuracy"))
        current_balanced = safe_float(current.get("balanced_accuracy"))
        current_up = safe_float(current.get("up_recall"))
        current_precision = safe_float(current.get("up_precision")) or 0.0
        for index, row in group.iterrows():
            if row.get("candidate_name") in {"current_model", "current_live_saved"}:
                rows.loc[index, "constraint_notes"] = "reference row"
                continue
            checks = {
                "up_recall_improved": _gt(row.get("up_recall"), current_up),
                "down_recall_ge_45": (safe_float(row.get("down_recall")) or 0.0) >= 0.45,
                "accuracy_within_3pt": (
                    current_accuracy is not None
                    and safe_float(row.get("accuracy")) is not None
                    and float(row.get("accuracy")) >= current_accuracy - 0.03
                ),
                "balanced_improved": _gt(row.get("balanced_accuracy"), current_balanced),
                "beats_always_down": (safe_float(row.get("model_minus_always_down")) or -1.0) > 0,
                "beats_previous_day": (safe_float(row.get("model_minus_previous_day_direction")) or -1.0) > 0,
                "beats_ma_trend": (safe_float(row.get("model_minus_moving_average_trend_only")) or -1.0) > 0,
                "up_precision_not_collapsed": (safe_float(row.get("up_precision")) or 0.0)
                >= max(0.25, current_precision - 0.10),
            }
            rows.loc[index, "constraint_passed"] = all(
                checks[key]
                for key in (
                    "up_recall_improved",
                    "down_recall_ge_45",
                    "accuracy_within_3pt",
                    "balanced_improved",
                    "up_precision_not_collapsed",
                )
            )
            rows.loc[index, "constraint_notes"] = "; ".join(
                key for key, passed in checks.items() if passed
            ) or "no constraints passed"
    return rows


def _failure_patterns(live_frame: pd.DataFrame) -> pd.DataFrame:
    rows = _confirmed(live_frame)
    if rows.empty:
        return pd.DataFrame()
    result = rows.copy()
    actual = result["actual_class"].map(safe_int)
    predicted = result["predicted_class"].map(safe_int)
    result["failure_type"] = np.select(
        [
            actual.eq(1) & predicted.eq(0),
            actual.eq(0) & predicted.eq(1),
            actual.eq(1) & predicted.eq(1),
            actual.eq(0) & predicted.eq(0),
        ],
        [
            "actual_up_predicted_down",
            "predicted_up_actual_down",
            "actual_up_predicted_up",
            "actual_down_predicted_down",
        ],
        default="other",
    )
    result = _attach_baseline_columns(result)
    keep = [
        "failure_type",
        "date",
        "ticker",
        "target_close_date",
        "predicted_class",
        "actual_class",
        "actual_return",
        "confidence",
        "probability_up",
        "probability_down",
        "regime",
        "anomaly_score",
        "anomaly_level",
        "baseline_previous_day_direction",
        "baseline_moving_average_trend",
        "top_features",
    ]
    return result[[column for column in keep if column in result.columns]].sort_values("date", ascending=False)


def _failure_common_features(failures: pd.DataFrame) -> pd.DataFrame:
    if failures.empty:
        return pd.DataFrame()
    rows = []
    for failure_type, group in failures.groupby("failure_type"):
        rows.append(
            {
                "failure_type": failure_type,
                "sample_count": int(len(group)),
                "average_confidence": _mean(group.get("confidence")),
                "average_actual_return": _mean(group.get("actual_return")),
                "regime_top": _top_value(group.get("regime")),
                "anomaly_level_top": _top_value(group.get("anomaly_level")),
                "previous_day_up_rate": _mean(group.get("baseline_previous_day_direction")),
                "ma_trend_up_rate": _mean(group.get("baseline_moving_average_trend")),
                "top_features_sample": _top_feature_sample(group.get("top_features")),
            }
        )
    return pd.DataFrame(rows)


def _period_subset(rows: pd.DataFrame, period: str) -> pd.DataFrame:
    confirmed = _confirmed(rows)
    if confirmed.empty or period == "Historical all":
        return confirmed
    dates = pd.to_datetime(confirmed.get("date"), errors="coerce")
    latest = pd.Timestamp(dates.max()).normalize()
    masks = {
        "2023-": dates >= pd.Timestamp("2023-01-01"),
        "2024-": dates >= pd.Timestamp("2024-01-01"),
        "2025-": dates >= pd.Timestamp("2025-01-01"),
        "2026-": dates >= pd.Timestamp("2026-01-01"),
        "Last 6 months": dates >= latest - pd.DateOffset(months=6),
        "Last 3 months": dates >= latest - pd.DateOffset(months=3),
    }
    mask = masks.get(period)
    if mask is None:
        return pd.DataFrame()
    return confirmed[mask].copy()


def _confirmed(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "status" in frame.columns:
        return frame[frame["status"].eq("confirmed")].copy()
    return frame.copy()


def _candidate_definitions_frame(candidates: list[BinaryUpCandidate]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_name": candidate.name,
                "feature_group": candidate.feature_group,
                "weight_mode": candidate.weight_mode,
                "threshold_mode": candidate.threshold_mode,
                "hybrid_mode": candidate.hybrid_mode,
                "note": candidate.note,
            }
            for candidate in candidates
        ]
    )


def _no_prediction_summary(candidate: BinaryUpCandidate) -> pd.DataFrame:
    row = {column: None for column in _summary_columns()}
    row.update(
        {
            "candidate_name": candidate.name,
            "period": "Historical all",
            "sample_count": 0,
            "feature_group": candidate.feature_group,
            "weight_mode": candidate.weight_mode,
            "threshold_mode": candidate.threshold_mode,
            "hybrid_mode": candidate.hybrid_mode,
            "notes": candidate.note,
            "caveats": "no predictions",
        }
    )
    return pd.DataFrame([row], columns=_summary_columns())


def _empty_result(candidates: list[BinaryUpCandidate]) -> BinaryUpDetectionResult:
    summary = pd.concat([_no_prediction_summary(candidate) for candidate in candidates], ignore_index=True)
    return BinaryUpDetectionResult(
        summary=summary,
        predictions=pd.DataFrame(),
        fold_details=pd.DataFrame(),
        confidence_buckets=pd.DataFrame(),
        failure_patterns=pd.DataFrame(),
        failure_common_features=pd.DataFrame(),
        candidate_definitions=_candidate_definitions_frame(candidates),
        conclusions=["no_candidate_ready_for_production", "production_change_not_ready"],
    )


def _summary_columns() -> list[str]:
    return [
        "candidate_name",
        "period",
        "sample_count",
        "warning",
        "evaluated_start",
        "evaluated_end",
        "feature_group",
        "weight_mode",
        "threshold_mode",
        "hybrid_mode",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "predicted_down_ratio",
        "predicted_up_count",
        "predicted_down_count",
        "actual_up_count",
        "actual_down_count",
        "confidence_mean",
        "confidence_median",
        "confidence_max",
        "always_up",
        "always_down",
        "previous_day_direction",
        "moving_average_trend_only",
        "model_minus_always_up",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
        "confidence_bucket_result",
        "constraint_passed",
        "constraint_notes",
        "notes",
        "caveats",
    ]


def _markdown_report(result: BinaryUpDetectionResult) -> str:
    historical = result.summary[result.summary.get("period", pd.Series(dtype=str)).eq("Historical all")].copy()
    keep = [
        "candidate_name",
        "sample_count",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "constraint_passed",
        "caveats",
    ]
    table = historical[[column for column in keep if column in historical.columns]].head(30)
    lines = [
        "# Binary Up Detection Experiments",
        "",
        "This is an offline historical walk-forward report. It does not change production models, daily_predict, SQLite prediction history, or feature generation.",
        "",
        "## Conclusions",
        *[f"- {item}" for item in result.conclusions],
        "",
        "## Historical all summary",
        _markdown_table(table) if not table.empty else "No historical rows.",
        "",
        "## Live failure pattern summary",
    ]
    if result.failure_common_features.empty:
        lines.append("No live failure pattern rows were available.")
    else:
        lines.append(_markdown_table(result.failure_common_features))
    lines.extend(
        [
            "",
            "## Caveat",
            "Candidates that look promising here still require a separate production-change review. Thresholds and weights are selected from train/validation rows only, then applied to test rows.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    normalized = frame.where(pd.notna(frame), "")
    headers = [str(column) for column in normalized.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in normalized.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _markdown_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    return json.loads(normalized.where(pd.notna(normalized), None).to_json(orient="records"))


def _concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [frame for frame in frames if frame is not None and not frame.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _candidate_row(rows: pd.DataFrame, name: str) -> pd.Series | None:
    match = rows[rows.get("candidate_name", pd.Series(dtype=str)).astype(str).eq(name)]
    if match.empty:
        return None
    return match.iloc[0]


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _delta(value, baseline) -> float | None:
    left = safe_float(value)
    right = safe_float(baseline)
    if left is None or right is None:
        return None
    return left - right


def _gt(value, reference) -> bool:
    left = safe_float(value)
    right = safe_float(reference)
    return left is not None and right is not None and left > right


def _mean(values) -> float | None:
    if values is None:
        return None
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return None
    return float(numbers.mean())


def _date_range(rows: pd.DataFrame) -> tuple[str | None, str | None]:
    dates = pd.to_datetime(rows.get("date", pd.Series(dtype=object)), errors="coerce").dropna()
    if dates.empty:
        return None, None
    return pd.Timestamp(dates.min()).strftime("%Y-%m-%d"), pd.Timestamp(dates.max()).strftime("%Y-%m-%d")


def _top_value(values) -> str:
    if values is None:
        return ""
    series = pd.Series(values).dropna().astype(str)
    if series.empty:
        return ""
    return str(series.value_counts().index[0])


def _top_feature_sample(values) -> str:
    if values is None:
        return ""
    for value in pd.Series(values).dropna():
        if not str(value).strip():
            continue
        return accuracy_diagnostics._top_feature_text(value)  # noqa: SLF001
    return ""


def _append_text(existing: str, addition: str) -> str:
    if not existing:
        return addition
    if addition in existing:
        return existing
    return f"{existing}; {addition}"
