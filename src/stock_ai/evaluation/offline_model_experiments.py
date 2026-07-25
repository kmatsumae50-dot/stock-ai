from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
import numpy as np
import pandas as pd

from stock_ai.backtest.walk_forward import TARGET_COLUMN, WalkForwardConfig
from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.models.baseline import (
    EVENT_FEATURE_COLUMNS,
    EVENT_FEATURE_PREFIXES,
    EXTERNAL_FEATURE_PREFIXES,
    NARRATIVE_FEATURE_PREFIXES,
    NEWS_FEATURE_COLUMNS,
    NEWS_FEATURE_PREFIXES,
    select_feature_columns,
)
from stock_ai.evaluation import accuracy_diagnostics
from stock_ai.utils.safe_cast import safe_float, safe_int


THRESHOLD_GRID = tuple(float(value) for value in np.round(np.arange(0.35, 0.651, 0.025), 3))
MIN_REGIME_THRESHOLD_ROWS = 30


@dataclass(frozen=True)
class OfflineExperimentConfig:
    ticker: str = "7203.T"
    train_years: int = 3
    test_step_months: int = 1
    start_date: str | None = None
    end_date: str | None = None
    min_train_rows: int = 252
    valid_fraction: float = 0.2
    min_sample_warning: int = 100


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    feature_group: str = "full_features"
    weight_mode: str = "none"
    threshold_mode: str = "fixed_0_50"
    note: str = ""


@dataclass(frozen=True)
class OfflineExperimentResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    fold_details: pd.DataFrame
    conclusions: list[str]
    report_paths: dict[str, str]


def default_candidate_configs() -> list[CandidateConfig]:
    return [
        CandidateConfig("current_model", note="Current LightGBM settings with full features and 0.50 threshold."),
        CandidateConfig(
            "class_weight_balanced",
            weight_mode="class_weight_balanced",
            note="LightGBM class_weight='balanced' set inside each fold from train data only.",
        ),
        CandidateConfig(
            "scale_pos_weight_train_ratio",
            weight_mode="scale_pos_weight",
            note="scale_pos_weight is computed from each fold's train rows only.",
        ),
        CandidateConfig(
            "balanced_accuracy_optimized_threshold",
            threshold_mode="balanced_accuracy",
            note="Threshold selected on validation rows only to maximize balanced accuracy.",
        ),
        CandidateConfig(
            "up_recall_priority_threshold",
            threshold_mode="up_recall_priority",
            note="Threshold selected on validation rows only to improve up recall while retaining some down recall.",
        ),
        CandidateConfig(
            "regime_specific_threshold",
            threshold_mode="regime_specific",
            note="Regime thresholds are selected on validation rows only when regime sample count is sufficient.",
        ),
        CandidateConfig(
            "no_external_sector_features",
            feature_group="no_external_sector_features",
            note="Drops external market and auto-sector features during training only.",
        ),
        CandidateConfig(
            "no_news_features",
            feature_group="no_news_features",
            note="Drops news and narrative-derived features during training only.",
        ),
        CandidateConfig(
            "technical_price_only",
            feature_group="technical_price_only",
            note="Uses price/technical numeric features while excluding external, event, news, and narrative groups.",
        ),
        CandidateConfig(
            "external_market_only",
            feature_group="external_market_only",
            note="Uses only external market and auto-sector numeric features.",
        ),
    ]


def run_offline_model_experiments(
    data: pd.DataFrame,
    config: OfflineExperimentConfig | None = None,
    candidates: list[CandidateConfig] | None = None,
) -> OfflineExperimentResult:
    config = config or OfflineExperimentConfig()
    candidates = candidates or default_candidate_configs()
    frame = _prepare_experiment_frame(data, config)
    if frame.empty:
        summary = _empty_summary_frame()
        return OfflineExperimentResult(summary, pd.DataFrame(), pd.DataFrame(), ["no_rows"], {})

    prediction_frames: list[pd.DataFrame] = []
    detail_frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for candidate in candidates:
        predictions, details = _run_candidate(frame, config, candidate)
        if predictions.empty:
            summary_rows.append(_empty_candidate_summary(candidate, "no_predictions"))
        else:
            diagnostic = accuracy_diagnostics.build_walk_forward_binary_diagnostic_frame(
                predictions,
                frame,
                prediction_type=f"offline_{candidate.name}",
            )
            summary_rows.append(_candidate_summary(candidate, diagnostic, details, config))
            prediction_frames.append(predictions)
        if not details.empty:
            detail_frames.append(details)

    summary = pd.DataFrame(summary_rows, columns=_summary_columns())
    conclusions = classify_experiment_conclusions(summary)
    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    fold_details = (
        pd.concat(detail_frames, ignore_index=True)
        if detail_frames
        else pd.DataFrame()
    )
    return OfflineExperimentResult(summary, predictions, fold_details, conclusions, {})


def write_offline_experiment_reports(
    result: OfflineExperimentResult,
    output_dir: str | Path,
    *,
    run_date: str | None = None,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = run_date or date.today().isoformat()
    csv_path = output / f"offline_model_experiments_{stamp}.csv"
    json_path = output / f"offline_model_experiments_{stamp}.json"
    result.summary.to_csv(csv_path, index=False, encoding="utf-8-sig")
    payload = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "conclusions": result.conclusions,
        "summary": _json_records(result.summary),
        "fold_details": _json_records(result.fold_details),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"csv": str(csv_path), "json": str(json_path)}


def classify_experiment_conclusions(summary: pd.DataFrame) -> list[str]:
    if summary.empty:
        return ["no_candidate_beats_baseline", "production_change_not_ready"]

    rows = summary.copy()
    for column in ("balanced_accuracy", "up_recall", "model_minus_always_up", "model_minus_always_down"):
        rows[column] = pd.to_numeric(rows.get(column), errors="coerce")
    current = _candidate_row(rows, "current_model")
    current_balanced = safe_float(current.get("balanced_accuracy")) if current is not None else None
    current_up_recall = safe_float(current.get("up_recall")) if current is not None else None
    best_balanced = safe_float(rows["balanced_accuracy"].max())
    conclusions: list[str] = []

    baseline_delta_columns = [
        "model_minus_always_up",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
    ]
    baseline_deltas = rows[baseline_delta_columns].apply(pd.to_numeric, errors="coerce")
    best_baseline_delta = baseline_deltas.stack().max() if not baseline_deltas.empty else np.nan
    if pd.isna(best_baseline_delta) or float(best_baseline_delta) <= 0.01:
        conclusions.append("no_candidate_beats_baseline")

    threshold_rows = rows[rows["threshold_mode"].astype(str).str.contains("balanced|up_recall", na=False)]
    threshold_up = safe_float(threshold_rows["up_recall"].max()) if not threshold_rows.empty else None
    if current_up_recall is not None and threshold_up is not None and threshold_up >= current_up_recall + 0.05:
        conclusions.append("balanced_threshold_improves_up_recall")

    balanced = _candidate_row(rows, "class_weight_balanced")
    if (
        current_balanced is not None
        and balanced is not None
        and safe_float(balanced.get("balanced_accuracy")) is not None
        and safe_float(balanced.get("balanced_accuracy")) >= current_balanced + 0.01
    ):
        conclusions.append("class_weight_improves_balanced_accuracy")

    no_external = _candidate_row(rows, "no_external_sector_features")
    if (
        current_balanced is not None
        and no_external is not None
        and safe_float(no_external.get("balanced_accuracy")) is not None
        and safe_float(no_external.get("balanced_accuracy")) >= current_balanced + 0.01
    ):
        conclusions.append("external_features_hurt_short_term")

    no_news = _candidate_row(rows, "no_news_features")
    if current_balanced is not None and no_news is not None and safe_float(no_news.get("balanced_accuracy")) is not None:
        if abs(float(no_news.get("balanced_accuracy")) - current_balanced) <= 0.005:
            conclusions.append("news_features_no_clear_signal")

    regime = _candidate_row(rows, "regime_specific_threshold")
    if (
        current_balanced is not None
        and regime is not None
        and safe_float(regime.get("balanced_accuracy")) is not None
        and safe_float(regime.get("balanced_accuracy")) >= current_balanced + 0.01
    ):
        conclusions.append("regime_threshold_promising")

    confidence_mean = pd.to_numeric(rows.get("confidence_mean"), errors="coerce").mean()
    if pd.notna(confidence_mean) and float(confidence_mean) < 0.56:
        conclusions.append("probability_calibration_needed")

    if best_balanced is None or best_balanced < 0.54:
        conclusions.append("production_change_not_ready")
    if not conclusions:
        conclusions.append("production_change_not_ready")
    return list(dict.fromkeys(conclusions))


def _run_candidate(
    frame: pd.DataFrame,
    config: OfflineExperimentConfig,
    candidate: CandidateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_predictions: list[pd.DataFrame] = []
    fold_details: list[dict[str, object]] = []
    windows = _test_windows(frame, config)
    for test_start, test_end in windows:
        train_window = _training_window(frame, test_start, config)
        test_window = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        test_window = test_window[test_window[TARGET_COLUMN].isin([0, 1])]
        if len(train_window) < config.min_train_rows or test_window.empty:
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        feature_columns = _filter_feature_group(feature_columns, candidate.feature_group)
        feature_columns = _available_feature_columns(train_window, test_window, feature_columns)
        if not feature_columns:
            continue

        train_window = train_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        test_window = test_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        train_window = train_window[train_window[TARGET_COLUMN].isin([0, 1])]
        if (
            len(train_window) < config.min_train_rows
            or test_window.empty
            or train_window[TARGET_COLUMN].nunique() < 2
        ):
            continue

        train_data, valid_data = _train_valid_split(train_window, config.valid_fraction)
        if train_data.empty or valid_data.empty or train_data[TARGET_COLUMN].nunique() < 2:
            continue

        model = _fit_lightgbm_binary(train_data, valid_data, feature_columns, candidate)
        valid_prob = _predict_up_probability(model, valid_data[feature_columns])
        valid_actual = valid_data[TARGET_COLUMN].astype(int)
        threshold_info = _select_thresholds(
            candidate,
            valid_data,
            valid_actual,
            valid_prob,
        )
        predictions = _predict_candidate_fold(
            model,
            test_window,
            feature_columns,
            candidate,
            threshold_info,
            train_data,
            valid_data,
            test_start,
            test_end,
        )
        fold_predictions.append(predictions)
        fold_details.append(
            {
                "candidate_name": candidate.name,
                "test_start_date": pd.Timestamp(test_start).strftime("%Y-%m-%d"),
                "test_end_date": pd.Timestamp(test_end).strftime("%Y-%m-%d"),
                "train_rows": int(len(train_data)),
                "valid_rows": int(len(valid_data)),
                "test_rows": int(len(test_window)),
                "feature_count": int(len(feature_columns)),
                "threshold": threshold_info.get("global_threshold"),
                "threshold_mode": candidate.threshold_mode,
                "regime_threshold_count": len(threshold_info.get("regime_thresholds", {})),
                "regime_thresholds": json.dumps(threshold_info.get("regime_thresholds", {}), ensure_ascii=False),
            }
        )
    predictions = (
        pd.concat(fold_predictions, ignore_index=True).sort_values("date").reset_index(drop=True)
        if fold_predictions
        else pd.DataFrame()
    )
    details = pd.DataFrame(fold_details)
    return predictions, details


def _prepare_experiment_frame(data: pd.DataFrame, config: OfflineExperimentConfig) -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame()
    frame = data.copy()
    if TARGET_COLUMN not in frame.columns:
        frame = add_next_close_binary_target(frame)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if config.ticker and "ticker" in frame.columns:
        frame = frame[frame["ticker"].astype(str) == config.ticker].copy()
    if frame.empty:
        return pd.DataFrame()
    if "ticker" not in frame.columns:
        frame["ticker"] = config.ticker
    if "next_close_return" not in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["next_close_return"] = close.shift(-1) / close - 1
    if "regime" not in frame.columns:
        frame["regime"] = "Unknown"
    if config.end_date:
        frame = frame[frame["date"] <= pd.to_datetime(config.end_date)]
    return frame.reset_index(drop=True)


def _filter_feature_group(feature_columns: list[str], feature_group: str) -> list[str]:
    if feature_group == "full_features":
        return feature_columns
    if feature_group == "no_external_sector_features":
        return [column for column in feature_columns if not column.startswith(EXTERNAL_FEATURE_PREFIXES)]
    if feature_group == "no_news_features":
        return [
            column
            for column in feature_columns
            if column not in NEWS_FEATURE_COLUMNS
            and not column.startswith(NEWS_FEATURE_PREFIXES)
            and not column.startswith(NARRATIVE_FEATURE_PREFIXES)
        ]
    if feature_group == "technical_price_only":
        return [
            column
            for column in feature_columns
            if not column.startswith(EXTERNAL_FEATURE_PREFIXES)
            and column not in NEWS_FEATURE_COLUMNS
            and not column.startswith(NEWS_FEATURE_PREFIXES)
            and column not in EVENT_FEATURE_COLUMNS
            and not column.startswith(EVENT_FEATURE_PREFIXES)
            and not column.startswith(NARRATIVE_FEATURE_PREFIXES)
        ]
    if feature_group == "external_market_only":
        return [column for column in feature_columns if column.startswith(EXTERNAL_FEATURE_PREFIXES)]
    return feature_columns


def _fit_lightgbm_binary(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
    candidate: CandidateConfig,
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
    if candidate.weight_mode == "class_weight_balanced":
        params["class_weight"] = "balanced"
    elif candidate.weight_mode == "scale_pos_weight":
        y_train = train[TARGET_COLUMN].astype(int)
        positives = int((y_train == 1).sum())
        negatives = int((y_train == 0).sum())
        if positives > 0:
            params["scale_pos_weight"] = negatives / positives

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
    candidate: CandidateConfig,
    valid_data: pd.DataFrame,
    valid_actual: pd.Series,
    valid_probability: pd.Series,
) -> dict[str, object]:
    if candidate.threshold_mode == "balanced_accuracy":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="balanced_accuracy")}
    if candidate.threshold_mode == "up_recall_priority":
        return {"global_threshold": _best_threshold(valid_actual, valid_probability, metric="up_recall_priority")}
    if candidate.threshold_mode == "regime_specific":
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
                if len(group) < MIN_REGIME_THRESHOLD_ROWS or group["actual"].nunique() < 2:
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
    best_down_recall = -1.0
    for threshold in THRESHOLD_GRID:
        predicted = (rows["probability"] >= threshold).astype(int)
        metrics = _binary_metrics(rows["actual"].astype(int), predicted)
        if metric == "up_recall_priority":
            down_recall = safe_float(metrics["down_recall"]) or 0.0
            up_recall = safe_float(metrics["up_recall"]) or 0.0
            score = up_recall if down_recall >= 0.35 else up_recall * 0.75
        else:
            score = safe_float(metrics["balanced_accuracy"]) or -1.0
            down_recall = safe_float(metrics["down_recall"]) or 0.0
        if score > best_score or (score == best_score and down_recall > best_down_recall):
            best_score = score
            best_down_recall = down_recall
            best_threshold = threshold
    return float(best_threshold)


def _predict_candidate_fold(
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    candidate: CandidateConfig,
    threshold_info: dict[str, object],
    train_data: pd.DataFrame,
    valid_data: pd.DataFrame,
    test_start,
    test_end,
) -> pd.DataFrame:
    probabilities = _predict_up_probability(model, test_data[feature_columns])
    rows = test_data.copy()
    rows["probability_up"] = probabilities
    rows["probability_down"] = 1.0 - rows["probability_up"]
    rows["decision_threshold"] = _threshold_for_rows(rows, threshold_info)
    rows["predicted_class"] = (rows["probability_up"] >= rows["decision_threshold"]).astype(int)
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = rows[["probability_up", "probability_down"]].max(axis=1)
    rows["prediction_direction"] = rows["predicted_class"].map(lambda value: "up" if value == 1 else "down")
    rows["actual_direction"] = rows["actual_class"].map(lambda value: "up" if value == 1 else "down")
    rows["candidate_name"] = candidate.name
    rows["feature_group"] = candidate.feature_group
    rows["weight_mode"] = candidate.weight_mode
    rows["threshold_mode"] = candidate.threshold_mode
    rows["model_name"] = candidate.name
    rows["target"] = "next_close_binary"
    rows["prediction_type"] = f"offline_{candidate.name}"
    rows["train_start_date"] = pd.to_datetime(train_data["date"].min())
    rows["train_end_date"] = pd.to_datetime(train_data["date"].max())
    rows["valid_start_date"] = pd.to_datetime(valid_data["date"].min())
    rows["valid_end_date"] = pd.to_datetime(valid_data["date"].max())
    rows["test_start_date"] = pd.to_datetime(test_start)
    rows["test_end_date"] = pd.to_datetime(test_end)
    rows["feature_count"] = len(feature_columns)

    keep = [
        "date",
        "ticker",
        "candidate_name",
        "model_name",
        "target",
        "prediction_type",
        "predicted_class",
        "actual_class",
        "prediction_direction",
        "actual_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "decision_threshold",
        "next_close_return",
        "regime",
        "feature_group",
        "weight_mode",
        "threshold_mode",
        "feature_count",
        "train_start_date",
        "train_end_date",
        "valid_start_date",
        "valid_end_date",
        "test_start_date",
        "test_end_date",
    ]
    return rows[[column for column in keep if column in rows.columns]]


def _threshold_for_rows(rows: pd.DataFrame, threshold_info: dict[str, object]) -> pd.Series:
    global_threshold = safe_float(threshold_info.get("global_threshold")) or 0.5
    thresholds = threshold_info.get("regime_thresholds") or {}
    if not thresholds or "regime" not in rows.columns:
        return pd.Series(global_threshold, index=rows.index, dtype=float)
    return rows["regime"].fillna("Unknown").astype(str).map(thresholds).fillna(global_threshold).astype(float)


def _candidate_summary(
    candidate: CandidateConfig,
    diagnostic: pd.DataFrame,
    details: pd.DataFrame,
    config: OfflineExperimentConfig,
) -> dict[str, object]:
    rows = diagnostic[diagnostic.get("status", pd.Series(dtype=str)).eq("confirmed")].copy()
    if rows.empty:
        return _empty_candidate_summary(candidate, "no_confirmed_rows")
    actual = rows["actual_class"].map(safe_int).astype(int)
    predicted = rows["predicted_class"].map(safe_int).astype(int)
    metrics = _binary_metrics(actual, predicted)
    baselines = accuracy_diagnostics.baseline_accuracy_frame(rows).set_index("baseline")
    confidence = pd.to_numeric(rows.get("confidence"), errors="coerce").dropna()
    thresholds = pd.to_numeric(
        details.get("threshold", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    sample_count = int(len(rows))
    caveats = _candidate_caveats(candidate, sample_count, metrics, config)
    return {
        "candidate_name": candidate.name,
        "sample_count": sample_count,
        "evaluated_start": _date_text(rows["date"].min()),
        "evaluated_end": _date_text(rows["date"].max()),
        "feature_group": candidate.feature_group,
        "weight_mode": candidate.weight_mode,
        "threshold_mode": candidate.threshold_mode,
        "fold_count": int(details["candidate_name"].eq(candidate.name).sum()) if not details.empty else 0,
        "feature_count_median": _median(details.get("feature_count")),
        "threshold_mean": float(thresholds.mean()) if not thresholds.empty else None,
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "up_recall": metrics["up_recall"],
        "down_recall": metrics["down_recall"],
        "up_precision": metrics["up_precision"],
        "down_precision": metrics["down_precision"],
        "predicted_up_ratio": _safe_divide(int((predicted == 1).sum()), len(predicted)),
        "predicted_down_ratio": _safe_divide(int((predicted == 0).sum()), len(predicted)),
        "confidence_mean": float(confidence.mean()) if not confidence.empty else None,
        "confidence_median": float(confidence.median()) if not confidence.empty else None,
        "confidence_min": float(confidence.min()) if not confidence.empty else None,
        "confidence_max": float(confidence.max()) if not confidence.empty else None,
        "model_minus_always_up": _baseline_delta(baselines, "always_up"),
        "model_minus_always_down": _baseline_delta(baselines, "always_down"),
        "model_minus_previous_day_direction": _baseline_delta(baselines, "previous_day_direction"),
        "model_minus_moving_average_trend_only": _baseline_delta(baselines, "moving_average_trend_only"),
        "notes": candidate.note,
        "caveats": "; ".join(caveats),
    }


def _binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | None]:
    valid = actual.notna() & predicted.notna()
    if not bool(valid.any()):
        return _empty_metric_values()
    actual = actual[valid].astype(int)
    predicted = predicted[valid].astype(int)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    up_recall = _safe_divide(tp, tp + fn)
    down_recall = _safe_divide(tn, tn + fp)
    return {
        "accuracy": _safe_divide(tp + tn, len(actual)),
        "balanced_accuracy": (
            (up_recall + down_recall) / 2 if up_recall is not None and down_recall is not None else None
        ),
        "up_recall": up_recall,
        "down_recall": down_recall,
        "up_precision": _safe_divide(tp, tp + fp),
        "down_precision": _safe_divide(tn, tn + fn),
    }


def _empty_candidate_summary(candidate: CandidateConfig, caveat: str) -> dict[str, object]:
    row = {column: None for column in _summary_columns()}
    row.update(
        {
            "candidate_name": candidate.name,
            "sample_count": 0,
            "feature_group": candidate.feature_group,
            "weight_mode": candidate.weight_mode,
            "threshold_mode": candidate.threshold_mode,
            "notes": candidate.note,
            "caveats": caveat,
        }
    )
    return row


def _candidate_caveats(
    candidate: CandidateConfig,
    sample_count: int,
    metrics: dict[str, float | None],
    config: OfflineExperimentConfig,
) -> list[str]:
    caveats: list[str] = []
    if sample_count < config.min_sample_warning:
        caveats.append("low sample count")
    predicted_up_ratio = safe_float(metrics.get("predicted_up_ratio"))
    if predicted_up_ratio is not None and predicted_up_ratio < 0.1:
        caveats.append("rarely predicts up")
    if candidate.threshold_mode == "regime_specific":
        caveats.append("regime thresholds are exploratory and sample-count sensitive")
    if candidate.feature_group != "full_features":
        caveats.append("feature group variant only changes training columns, not feature generation")
    return caveats


def _baseline_delta(baselines: pd.DataFrame, name: str) -> float | None:
    if baselines.empty or name not in baselines.index:
        return None
    return safe_float(baselines.loc[name, "model_minus_baseline"])


def _summary_columns() -> list[str]:
    return [
        "candidate_name",
        "sample_count",
        "evaluated_start",
        "evaluated_end",
        "feature_group",
        "weight_mode",
        "threshold_mode",
        "fold_count",
        "feature_count_median",
        "threshold_mean",
        "accuracy",
        "balanced_accuracy",
        "up_recall",
        "down_recall",
        "up_precision",
        "down_precision",
        "predicted_up_ratio",
        "predicted_down_ratio",
        "confidence_mean",
        "confidence_median",
        "confidence_min",
        "confidence_max",
        "model_minus_always_up",
        "model_minus_always_down",
        "model_minus_previous_day_direction",
        "model_minus_moving_average_trend_only",
        "notes",
        "caveats",
    ]


def _empty_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=_summary_columns())


def _test_windows(frame: pd.DataFrame, config: OfflineExperimentConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    wf_config = WalkForwardConfig(
        ticker=config.ticker,
        train_years=config.train_years,
        test_step_months=config.test_step_months,
        start_date=config.start_date,
        end_date=config.end_date,
        min_train_rows=config.min_train_rows,
        valid_fraction=config.valid_fraction,
    )
    if frame.empty:
        return []
    first_test_date = frame["date"].min() + pd.DateOffset(years=wf_config.train_years)
    if wf_config.start_date:
        first_test_date = max(first_test_date, pd.to_datetime(wf_config.start_date))
    last_date = frame["date"].max()
    if wf_config.end_date:
        last_date = min(last_date, pd.to_datetime(wf_config.end_date))
    candidates = frame[(frame["date"] >= first_test_date) & (frame["date"] <= last_date)].copy()
    if candidates.empty:
        return []
    candidates["period"] = candidates["date"].dt.to_period("M")
    month_starts = candidates.groupby("period")["date"].min().tolist()
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    step = max(1, wf_config.test_step_months)
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
    config: OfflineExperimentConfig,
) -> pd.DataFrame:
    train_start = test_start - pd.DateOffset(years=config.train_years)
    train = frame[(frame["date"] >= train_start) & (frame["date"] < test_start)].copy()
    if len(train) > 1:
        train = train.iloc[:-1].copy()
    return train


def _train_valid_split(train: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(train) * (1 - valid_fraction))
    split_index = max(1, min(split_index, len(train) - 1))
    return train.iloc[:split_index].copy(), train.iloc[split_index:].copy()


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


def _predict_up_probability(model, features: pd.DataFrame) -> pd.Series:
    probabilities = model.predict_proba(features)
    if hasattr(probabilities, "iloc"):
        values = probabilities.iloc[:, -1]
    else:
        values = [row[-1] for row in probabilities]
    return pd.Series(values, index=features.index, dtype=float)


def _safe_divide(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _median(values) -> float | None:
    if values is None:
        return None
    numbers = pd.to_numeric(values, errors="coerce").dropna()
    if numbers.empty:
        return None
    return float(numbers.median())


def _date_text(value) -> str | None:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp).strftime("%Y-%m-%d")


def _json_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    if frame.empty:
        return []
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = normalized[column].dt.strftime("%Y-%m-%d")
    return json.loads(normalized.where(pd.notna(normalized), None).to_json(orient="records"))


def _candidate_row(rows: pd.DataFrame, name: str) -> pd.Series | None:
    match = rows[rows["candidate_name"].astype(str).eq(name)]
    if match.empty:
        return None
    return match.iloc[0]


def _empty_metric_values() -> dict[str, None]:
    return {
        "accuracy": None,
        "balanced_accuracy": None,
        "up_recall": None,
        "down_recall": None,
        "up_precision": None,
        "down_precision": None,
    }
