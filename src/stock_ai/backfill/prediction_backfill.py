from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd

from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.models.baseline import select_feature_columns


TARGET_COLUMN = "target_next_close_binary"


@dataclass(frozen=True)
class BackfillConfig:
    ticker: str = "7203.T"
    start_date: str | None = None
    end_date: str | None = None
    model_name: str = "lightgbm_binary"
    min_train_rows: int = 252
    valid_fraction: float = 0.2


@dataclass(frozen=True)
class BackfillResult:
    predictions: pd.DataFrame
    attempted_dates: int
    saved_rows: int
    skipped_rows: pd.DataFrame


def build_binary_prediction_backfill(
    data: pd.DataFrame,
    config: BackfillConfig,
) -> BackfillResult:
    """Build pseudo historical predictions without training on future rows.

    For each prediction date, only rows strictly before that date are used for
    training. This is for history/coverage inspection, not a replacement for
    the true daily live prediction log.
    """
    frame = _prepare_frame(data, config)
    if frame.empty:
        return _empty_result()

    prediction_dates = _prediction_dates(frame, config)
    predictions: list[pd.DataFrame] = []
    skipped_rows: list[dict[str, object]] = []

    for prediction_date in prediction_dates:
        train_window = frame[frame["date"] < prediction_date].copy()
        predict_row = frame[frame["date"] == prediction_date].copy()
        if len(train_window) < config.min_train_rows or predict_row.empty:
            skipped_rows.append(_skip_row(prediction_date, "Not enough prior training rows."))
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        train_window = _usable_training_rows(train_window, feature_columns)
        predict_row = predict_row.dropna(subset=feature_columns)
        if (
            len(train_window) < config.min_train_rows
            or predict_row.empty
            or train_window[TARGET_COLUMN].nunique() < 2
        ):
            skipped_rows.append(_skip_row(prediction_date, "No usable training or prediction rows."))
            continue

        train_data, valid_data = _train_valid_split(train_window, config.valid_fraction)
        if train_data.empty or valid_data.empty or train_data[TARGET_COLUMN].nunique() < 2:
            skipped_rows.append(_skip_row(prediction_date, "No valid chronological split."))
            continue

        try:
            model = _fit_lightgbm_binary(train_data, valid_data, feature_columns)
            predictions.append(
                _predict_row(
                    model,
                    predict_row,
                    feature_columns,
                    config,
                    train_start=train_data["date"].min(),
                    train_end=train_data["date"].max(),
                )
            )
        except Exception as error:
            skipped_rows.append(_skip_row(prediction_date, str(error)))

    result = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    return BackfillResult(
        predictions=result,
        attempted_dates=len(prediction_dates),
        saved_rows=len(result),
        skipped_rows=pd.DataFrame(skipped_rows),
    )


def _prepare_frame(data: pd.DataFrame, config: BackfillConfig) -> pd.DataFrame:
    if data.empty or not {"date", "close"}.issubset(data.columns):
        return pd.DataFrame()

    frame = data.copy()
    if "ticker" not in frame.columns:
        frame["ticker"] = config.ticker
    frame = frame[frame["ticker"].astype(str) == config.ticker].copy()
    if frame.empty:
        return pd.DataFrame()

    if TARGET_COLUMN not in frame.columns:
        frame = add_next_close_binary_target(frame)
    if "next_close_return" not in frame.columns:
        close = pd.to_numeric(frame["close"], errors="coerce")
        frame["next_close_return"] = close.shift(-1) / close - 1

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _prediction_dates(frame: pd.DataFrame, config: BackfillConfig) -> list[pd.Timestamp]:
    dates = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values()
    if config.start_date:
        dates = dates[dates >= pd.to_datetime(config.start_date)]
    if config.end_date:
        dates = dates[dates <= pd.to_datetime(config.end_date)]
    return [pd.Timestamp(date) for date in dates.unique()]


def _usable_training_rows(frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame()
    rows = frame.dropna(subset=feature_columns + [TARGET_COLUMN]).copy()
    rows = rows[rows[TARGET_COLUMN].isin([0, 1])]
    return rows.sort_values("date").reset_index(drop=True)


def _train_valid_split(train: pd.DataFrame, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_index = int(len(train) * (1 - valid_fraction))
    split_index = max(1, min(split_index, len(train) - 1))
    return train.iloc[:split_index].copy(), train.iloc[split_index:].copy()


def _fit_lightgbm_binary(train: pd.DataFrame, valid: pd.DataFrame, feature_columns: list[str]):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        train[feature_columns],
        train[TARGET_COLUMN].astype(int),
        eval_set=[(valid[feature_columns], valid[TARGET_COLUMN].astype(int))],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def _predict_row(
    model,
    row: pd.DataFrame,
    feature_columns: list[str],
    config: BackfillConfig,
    train_start,
    train_end,
) -> pd.DataFrame:
    probabilities = model.predict_proba(row[feature_columns])
    probability_down, probability_up = _probability_columns(probabilities)
    probability_up_series = pd.Series(probability_up, index=row.index, dtype=float)
    probability_down_series = pd.Series(probability_down, index=row.index, dtype=float)
    result = row[["date", "ticker"]].copy()
    result["target"] = "next_close_binary"
    result["model_name"] = config.model_name
    result["predicted_class"] = (probability_up_series >= 0.5).astype(int)
    result["confidence"] = pd.concat(
        [probability_up_series, probability_down_series],
        axis=1,
    ).max(axis=1)
    result["probability_up"] = probability_up_series
    result["probability_down"] = probability_down_series
    result["top_features"] = json.dumps(
        [
            {"feature": feature, "importance": None}
            for feature in feature_columns[:10]
        ],
        ensure_ascii=True,
    )
    result["prediction_type"] = "backfill"
    result["backfill_train_start_date"] = pd.to_datetime(train_start)
    result["backfill_train_end_date"] = pd.to_datetime(train_end)
    return result.reset_index(drop=True)


def _probability_columns(probabilities) -> tuple[list[float], list[float]]:
    if hasattr(probabilities, "iloc"):
        return list(probabilities.iloc[:, 0]), list(probabilities.iloc[:, 1])
    return [float(item[0]) for item in probabilities], [float(item[1]) for item in probabilities]


def _skip_row(prediction_date, reason: str) -> dict[str, object]:
    return {
        "date": pd.Timestamp(prediction_date).strftime("%Y-%m-%d"),
        "reason": reason,
    }


def _empty_result() -> BackfillResult:
    return BackfillResult(
        predictions=pd.DataFrame(),
        attempted_dates=0,
        saved_rows=0,
        skipped_rows=pd.DataFrame(columns=["date", "reason"]),
    )
