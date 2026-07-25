from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stock_ai.backtest.walk_forward import summarize_backtest_predictions
from stock_ai.evaluation.performance import (
    build_evaluation_summaries,
    prepare_backtest_evaluation_frame,
    summarize_by_group,
)
from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.models.baseline import select_feature_columns
from stock_ai.validation.time_split import chronological_train_valid_test_split


TARGET_COLUMN = "target_next_close_binary"
MODEL_CHARACTERISTICS = {
    "Dummy Classifier": "Baseline that predicts the most frequent class. It checks whether ML adds value.",
    "Logistic Regression": "Linear baseline. It is easy to interpret but may miss nonlinear market patterns.",
    "Random Forest": "Tree ensemble that captures nonlinear rules, often robust but less time-adaptive.",
    "LightGBM": "Gradient boosting tree model. Strong tabular baseline for mixed technical and context features.",
    "XGBoost": "Gradient boosting tree model. Similar role to LightGBM with different regularization behavior.",
}
DEFAULT_MODEL_ORDER = (
    "Dummy Classifier",
    "Logistic Regression",
    "Random Forest",
    "LightGBM",
    "XGBoost",
)
RANKING_METRICS = (
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


@dataclass(frozen=True)
class ModelComparisonResult:
    predictions: pd.DataFrame
    score_table: pd.DataFrame
    metric_rankings: pd.DataFrame
    skipped_models: pd.DataFrame
    feature_columns: list[str]
    split_summary: dict[str, object]


def run_model_comparison(
    data: pd.DataFrame,
    ticker: str = "7203.T",
    probability_threshold: float = 0.55,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    include_xgboost: bool = True,
    model_names: tuple[str, ...] | None = None,
    feature_columns_override: list[str] | None = None,
) -> ModelComparisonResult:
    """Compare binary classifiers on one chronological train/valid/test split."""
    frame = prepare_model_compare_frame(data, ticker=ticker)
    if frame.empty:
        return _empty_result()

    train, valid, test = chronological_train_valid_test_split(
        frame,
        train_ratio=train_ratio,
        valid_ratio=valid_ratio,
    )
    feature_columns = (
        _valid_feature_columns(train, feature_columns_override)
        if feature_columns_override is not None
        else select_feature_columns(train, target_column=TARGET_COLUMN)
    )
    if not feature_columns:
        return _empty_result()

    train = _usable_rows(train, feature_columns)
    valid = _usable_rows(valid, feature_columns)
    test = _usable_rows(test, feature_columns)
    split_summary = _build_split_summary(train, valid, test)
    if train.empty or valid.empty or test.empty or train[TARGET_COLUMN].nunique() < 2:
        return ModelComparisonResult(
            predictions=pd.DataFrame(),
            score_table=pd.DataFrame(),
            metric_rankings=pd.DataFrame(),
            skipped_models=pd.DataFrame(
                [{"model_name": "All", "reason": "Not enough usable chronological rows."}]
            ),
            feature_columns=feature_columns,
            split_summary=split_summary,
        )

    requested = model_names or DEFAULT_MODEL_ORDER
    if not include_xgboost:
        requested = tuple(name for name in requested if name != "XGBoost")

    prediction_frames: list[pd.DataFrame] = []
    score_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    for model_name in requested:
        model = _build_model(model_name)
        if model is None:
            skipped_rows.append(
                {"model_name": model_name, "reason": "Package is not installed or not available."}
            )
            continue
        try:
            _fit_model(model_name, model, train, valid, feature_columns)
            predictions = _predict_test_rows(
                model=model,
                model_name=model_name,
                test=test,
                feature_columns=feature_columns,
                threshold=probability_threshold,
                split_summary=split_summary,
            )
        except Exception as error:
            skipped_rows.append({"model_name": model_name, "reason": str(error)})
            continue

        metrics = summarize_backtest_predictions(predictions)
        score_rows.append(
            {
                "model_name": model_name,
                "characteristic": MODEL_CHARACTERISTICS.get(model_name, ""),
                **metrics,
            }
        )
        prediction_frames.append(predictions)

    predictions = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    score_table = pd.DataFrame(score_rows)
    return ModelComparisonResult(
        predictions=predictions,
        score_table=score_table,
        metric_rankings=build_metric_rankings(score_table),
        skipped_models=pd.DataFrame(skipped_rows),
        feature_columns=feature_columns,
        split_summary=split_summary,
    )


def prepare_model_compare_frame(data: pd.DataFrame, ticker: str = "7203.T") -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame()

    frame = data.copy()
    if "ticker" not in frame.columns:
        frame["ticker"] = ticker
    frame = frame[frame["ticker"].astype(str) == ticker].copy()
    if frame.empty:
        return pd.DataFrame()

    if TARGET_COLUMN not in frame.columns:
        frame = add_next_close_binary_target(frame)
    if "next_close_return" not in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["next_close_return"] = close.shift(-1) / close - 1

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return _add_missing_context_columns(frame)


def build_metric_rankings(score_table: pd.DataFrame) -> pd.DataFrame:
    if score_table.empty:
        return pd.DataFrame(columns=["metric", "rank", "model_name", "value"])

    rows: list[dict[str, object]] = []
    for metric in RANKING_METRICS:
        if metric not in score_table.columns:
            continue
        ranked = score_table[["model_name", metric]].copy()
        ranked["value"] = pd.to_numeric(ranked[metric], errors="coerce")
        ranked = ranked.dropna(subset=["value"]).sort_values("value", ascending=False)
        for rank, row in enumerate(ranked.itertuples(index=False), start=1):
            rows.append(
                {
                    "metric": metric,
                    "rank": rank,
                    "model_name": row.model_name,
                    "value": float(row.value),
                }
            )
    return pd.DataFrame(rows)


def compare_condition_performance(
    predictions: pd.DataFrame,
    group_column: str,
    metric: str = "accuracy",
    min_count: int = 5,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for model_name, model_rows in predictions.groupby("model_name", dropna=False):
        eval_frame = prepare_backtest_evaluation_frame(model_rows)
        if eval_frame.empty:
            continue
        if group_column == "anomaly_type":
            summary = build_evaluation_summaries(eval_frame).get(group_column, pd.DataFrame())
        elif group_column in eval_frame.columns:
            summary = summarize_by_group(eval_frame, group_column)
        else:
            continue
        if summary.empty or metric not in summary.columns:
            continue
        summary["model_name"] = model_name
        rows.append(summary)
    if not rows:
        return pd.DataFrame()

    result = pd.concat(rows, ignore_index=True)
    result["count"] = pd.to_numeric(result["count"], errors="coerce").fillna(0).astype(int)
    result = result[result["count"] >= min_count].copy()
    if result.empty:
        return result
    result[metric] = pd.to_numeric(result[metric], errors="coerce")
    return result.sort_values(metric, ascending=False).reset_index(drop=True)


def model_characteristics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_name": name, "characteristic": description}
            for name, description in MODEL_CHARACTERISTICS.items()
        ]
    )


def _valid_feature_columns(data: pd.DataFrame, feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if column in data.columns and pd.api.types.is_numeric_dtype(data[column])
    ]


def _build_model(model_name: str):
    if model_name == "Dummy Classifier":
        from sklearn.dummy import DummyClassifier

        return DummyClassifier(strategy="most_frequent")
    if model_name == "Logistic Regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
            ),
        )
    if model_name == "Random Forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "LightGBM":
        try:
            import lightgbm as lgb
        except Exception:
            return None

        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=200,
            learning_rate=0.04,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
        except Exception:
            return None

        return XGBClassifier(
            objective="binary:logistic",
            n_estimators=200,
            learning_rate=0.04,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
            tree_method="hist",
        )
    return None


def _fit_model(
    model_name: str,
    model,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    feature_columns: list[str],
) -> None:
    target = train[TARGET_COLUMN].astype(int)
    if model_name == "LightGBM":
        import lightgbm as lgb

        model.fit(
            train[feature_columns],
            target,
            eval_set=[(valid[feature_columns], valid[TARGET_COLUMN].astype(int))],
            eval_metric="binary_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
        )
        return
    if model_name == "XGBoost":
        model.fit(
            train[feature_columns],
            target,
            eval_set=[(valid[feature_columns], valid[TARGET_COLUMN].astype(int))],
            verbose=False,
        )
        return
    model.fit(train[feature_columns], target)


def _predict_test_rows(
    model,
    model_name: str,
    test: pd.DataFrame,
    feature_columns: list[str],
    threshold: float,
    split_summary: dict[str, object],
) -> pd.DataFrame:
    rows = test.copy()
    rows["model_name"] = model_name
    rows["probability_up"] = _predict_probability_up(model, rows[feature_columns])
    rows["probability_down"] = 1.0 - rows["probability_up"]
    rows["predicted_class"] = (rows["probability_up"] >= 0.5).astype(int)
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = rows[["probability_up", "probability_down"]].max(axis=1)
    rows["prediction_direction"] = rows["predicted_class"].map(
        lambda value: "up" if value == 1 else "down"
    )
    rows["actual_direction"] = rows["actual_class"].map(lambda value: "up" if value == 1 else "down")
    rows["probability_threshold"] = float(threshold)
    rows["train_start_date"] = split_summary.get("train_start_date")
    rows["train_end_date"] = split_summary.get("train_end_date")
    rows["valid_start_date"] = split_summary.get("valid_start_date")
    rows["valid_end_date"] = split_summary.get("valid_end_date")
    rows["test_start_date"] = split_summary.get("test_start_date")
    rows["test_end_date"] = split_summary.get("test_end_date")
    scored = _add_strategy_returns(rows, threshold=threshold)
    return scored[_prediction_columns(scored)]


def _predict_probability_up(model, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if hasattr(probabilities, "iloc"):
            values = probabilities.iloc[:, -1]
        else:
            values = [row[-1] for row in probabilities]
        return pd.Series(values, index=features.index, dtype=float).clip(0.0, 1.0)

    if hasattr(model, "decision_function"):
        scores = pd.Series(model.decision_function(features), index=features.index, dtype=float)
        return (1.0 / (1.0 + (-scores).map(_safe_exp))).clip(0.0, 1.0)

    predictions = model.predict(features)
    return pd.Series(predictions, index=features.index, dtype=float).clip(0.0, 1.0)


def _safe_exp(value: float) -> float:
    import math

    return math.exp(max(min(value, 50.0), -50.0))


def _usable_rows(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = frame.copy()
    rows = rows.dropna(subset=feature_columns + [TARGET_COLUMN])
    rows = rows[rows[TARGET_COLUMN].isin([0, 1])].copy()
    return rows.sort_values("date").reset_index(drop=True)


def _build_split_summary(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    return {
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "test_rows": int(len(test)),
        "train_start_date": _min_date(train),
        "train_end_date": _max_date(train),
        "valid_start_date": _min_date(valid),
        "valid_end_date": _max_date(valid),
        "test_start_date": _min_date(test),
        "test_end_date": _max_date(test),
    }


def _min_date(frame: pd.DataFrame):
    return pd.to_datetime(frame["date"]).min() if not frame.empty and "date" in frame else None


def _max_date(frame: pd.DataFrame):
    return pd.to_datetime(frame["date"]).max() if not frame.empty and "date" in frame else None


def _add_strategy_returns(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = predictions.copy()
    rows["actual_return"] = pd.to_numeric(rows["next_close_return"], errors="coerce")
    rows["trade"] = rows["probability_up"] >= threshold
    rows["strategy_return"] = rows["actual_return"].where(rows["trade"], 0.0).fillna(0.0)
    rows["buy_hold_return"] = rows["actual_return"].fillna(0.0)
    rows["strategy_cumulative_return"] = (1 + rows["strategy_return"]).cumprod() - 1
    rows["buy_hold_cumulative_return"] = (1 + rows["buy_hold_return"]).cumprod() - 1
    return rows


def _add_missing_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    defaults: dict[str, Any] = {
        "regime": "Unknown",
        "surge_risk_level": "Unknown",
        "crash_risk_level": "Unknown",
        "anomaly_types": "",
        "dominant_narrative": "Unknown",
        "earnings_event_state": "Unknown",
        "negative_news_flag": 0,
        "positive_news_flag": 0,
        "importance_weighted_sentiment": 0.0,
    }
    for column, default in defaults.items():
        if column not in rows.columns:
            rows[column] = default
    return rows


def _prediction_columns(rows: pd.DataFrame) -> list[str]:
    keep = [
        "date",
        "ticker",
        "model_name",
        "predicted_class",
        "actual_class",
        "prediction_direction",
        "actual_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "next_close_return",
        "actual_return",
        "trade",
        "strategy_return",
        "buy_hold_return",
        "strategy_cumulative_return",
        "buy_hold_cumulative_return",
        "regime",
        "surge_risk_level",
        "crash_risk_level",
        "anomaly_types",
        "dominant_narrative",
        "earnings_event_state",
        "negative_news_flag",
        "positive_news_flag",
        "importance_weighted_sentiment",
        "probability_threshold",
        "train_start_date",
        "train_end_date",
        "valid_start_date",
        "valid_end_date",
        "test_start_date",
        "test_end_date",
    ]
    return [column for column in keep if column in rows.columns]


def _empty_result() -> ModelComparisonResult:
    return ModelComparisonResult(
        predictions=pd.DataFrame(),
        score_table=pd.DataFrame(),
        metric_rankings=pd.DataFrame(),
        skipped_models=pd.DataFrame(),
        feature_columns=[],
        split_summary={},
    )
