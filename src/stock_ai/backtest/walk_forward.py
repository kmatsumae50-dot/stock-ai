from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from stock_ai.labels.targets import add_next_close_binary_target
from stock_ai.models.baseline import select_feature_columns
from stock_ai.utils.safe_cast import safe_float, safe_int


TARGET_COLUMN = "target_next_close_binary"


@dataclass(frozen=True)
class WalkForwardConfig:
    ticker: str = "7203.T"
    train_years: int = 3
    test_step_months: int = 1
    probability_threshold: float = 0.55
    start_date: str | None = None
    end_date: str | None = None
    min_train_rows: int = 252
    valid_fraction: float = 0.2


@dataclass(frozen=True)
class BacktestResult:
    predictions: pd.DataFrame
    metrics: dict[str, object]
    equity_curve: pd.DataFrame
    monthly_performance: pd.DataFrame
    regime_performance: pd.DataFrame
    surge_risk_performance: pd.DataFrame
    crash_risk_performance: pd.DataFrame


def run_walk_forward_backtest(
    data: pd.DataFrame,
    config: WalkForwardConfig | None = None,
    model_factory=None,
) -> BacktestResult:
    """Run a lightweight walk-forward binary backtest.

    Each fold trains only on rows strictly before the test month. The final
    pre-test row is dropped because its next-day label is not known at retrain
    time, which avoids a subtle one-day lookahead leak.
    """
    config = config or WalkForwardConfig()
    frame = _prepare_backtest_frame(data, config)
    if frame.empty:
        return _empty_result()

    fold_rows: list[pd.DataFrame] = []
    for test_start, test_end in _test_windows(frame, config):
        train_window = _training_window(frame, test_start, config)
        test_window = frame[(frame["date"] >= test_start) & (frame["date"] < test_end)].copy()
        test_window = test_window[test_window[TARGET_COLUMN].isin([0, 1])]
        if len(train_window) < config.min_train_rows or test_window.empty:
            continue

        feature_columns = select_feature_columns(train_window, target_column=TARGET_COLUMN)
        feature_columns = _available_feature_columns(train_window, test_window, feature_columns)
        if not feature_columns:
            continue

        train_window = train_window.dropna(subset=feature_columns + [TARGET_COLUMN])
        train_window = train_window[train_window[TARGET_COLUMN].isin([0, 1])]
        test_window = test_window.dropna(subset=feature_columns + [TARGET_COLUMN])
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
            else _fit_lightgbm_binary(train_data, valid_data, feature_columns)
        )
        fold_rows.append(
            _predict_fold(
                model=model,
                test_data=test_window,
                feature_columns=feature_columns,
                config=config,
                train_start=train_data["date"].min(),
                train_end=train_data["date"].max(),
                test_start=test_start,
                test_end=test_end,
            )
        )

    if not fold_rows:
        return _empty_result()

    predictions = pd.concat(fold_rows, ignore_index=True).sort_values("date").reset_index(drop=True)
    predictions = _add_strategy_returns(predictions, threshold=config.probability_threshold)
    metrics = summarize_backtest_predictions(predictions)
    equity_curve = build_equity_curve(predictions)
    monthly = summarize_monthly_performance(predictions)
    regime = summarize_group_performance(predictions, "regime")
    surge = summarize_group_performance(predictions, "surge_risk_level")
    crash = summarize_group_performance(predictions, "crash_risk_level")
    return BacktestResult(
        predictions=predictions,
        metrics=metrics,
        equity_curve=equity_curve,
        monthly_performance=monthly,
        regime_performance=regime,
        surge_risk_performance=surge,
        crash_risk_performance=crash,
    )


def summarize_backtest_predictions(predictions: pd.DataFrame) -> dict[str, object]:
    if predictions.empty:
        return _empty_metrics()

    actual = predictions["actual_class"].map(safe_int)
    predicted = predictions["predicted_class"].map(safe_int)
    probability_up = predictions["probability_up"].map(safe_float)

    valid = pd.DataFrame(
        {
            "actual": actual,
            "predicted": predicted,
            "probability_up": probability_up,
        }
    ).dropna(subset=["actual", "predicted"])
    if valid.empty:
        return _empty_metrics()

    valid["actual"] = valid["actual"].astype(int)
    valid["predicted"] = valid["predicted"].astype(int)
    tn, fp, fn, tp = _confusion_counts(valid["actual"], valid["predicted"])
    total = len(valid)
    accuracy = (tp + tn) / total if total else None
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall) if precision is not None and recall is not None else None

    return {
        "rows": int(total),
        "accuracy": accuracy,
        "f1": f1,
        "auc": _auc(valid["actual"], valid["probability_up"]),
        "precision": precision,
        "recall": recall,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "up_prediction_accuracy": _prediction_direction_accuracy(valid, predicted_class=1),
        "down_prediction_accuracy": _prediction_direction_accuracy(valid, predicted_class=0),
        "trade_count": int(predictions["trade"].sum()) if "trade" in predictions else 0,
        "win_rate": _win_rate(predictions),
        "cumulative_return": _final_value(predictions, "strategy_cumulative_return"),
        "buy_hold_return": _final_value(predictions, "buy_hold_cumulative_return"),
        "max_drawdown": _max_drawdown_from_returns(predictions.get("strategy_return")),
    }


def build_equity_curve(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "strategy_cumulative_return",
                "buy_hold_cumulative_return",
                "drawdown",
            ]
        )
    curve = predictions[
        [
            "date",
            "strategy_return",
            "buy_hold_return",
            "strategy_cumulative_return",
            "buy_hold_cumulative_return",
        ]
    ].copy()
    equity = (1 + pd.to_numeric(curve["strategy_return"], errors="coerce").fillna(0.0)).cumprod()
    curve["drawdown"] = equity / equity.cummax() - 1
    return curve


def summarize_monthly_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return _empty_performance("month")
    rows = predictions.copy()
    rows["month"] = pd.to_datetime(rows["date"]).dt.to_period("M").astype(str)
    return _group_performance(rows, "month")


def summarize_group_performance(predictions: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if predictions.empty or group_column not in predictions.columns:
        return _empty_performance(group_column)
    rows = predictions.copy()
    rows[group_column] = rows[group_column].fillna("Unknown").astype(str)
    return _group_performance(rows, group_column)


def write_backtest_csvs(
    result: BacktestResult,
    output_dir: str | Path,
    ticker: str,
    threshold: float,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("^", "").replace("=", "").replace(".", "_")
    prefix = f"{safe_ticker}_binary_threshold_{threshold:.2f}".replace(".", "_")
    paths = {
        "predictions": output / f"{prefix}_predictions.csv",
        "equity": output / f"{prefix}_equity.csv",
        "monthly": output / f"{prefix}_monthly.csv",
        "regime": output / f"{prefix}_regime.csv",
        "surge_risk": output / f"{prefix}_surge_risk.csv",
        "crash_risk": output / f"{prefix}_crash_risk.csv",
        "metrics": output / f"{prefix}_metrics.json",
    }
    result.predictions.to_csv(paths["predictions"], index=False, encoding="utf-8-sig")
    result.equity_curve.to_csv(paths["equity"], index=False, encoding="utf-8-sig")
    result.monthly_performance.to_csv(paths["monthly"], index=False, encoding="utf-8-sig")
    result.regime_performance.to_csv(paths["regime"], index=False, encoding="utf-8-sig")
    result.surge_risk_performance.to_csv(paths["surge_risk"], index=False, encoding="utf-8-sig")
    result.crash_risk_performance.to_csv(paths["crash_risk"], index=False, encoding="utf-8-sig")
    paths["metrics"].write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    return paths


def _prepare_backtest_frame(data: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
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
    if config.end_date:
        frame = frame[frame["date"] <= pd.to_datetime(config.end_date)]
    return _add_risk_context_columns(frame.reset_index(drop=True))


def _test_windows(frame: pd.DataFrame, config: WalkForwardConfig) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
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


def _training_window(frame: pd.DataFrame, test_start: pd.Timestamp, config: WalkForwardConfig) -> pd.DataFrame:
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


def _fit_lightgbm_binary(train: pd.DataFrame, valid: pd.DataFrame, feature_columns: list[str]):
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=150,
        learning_rate=0.04,
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


def _predict_fold(
    model,
    test_data: pd.DataFrame,
    feature_columns: list[str],
    config: WalkForwardConfig,
    train_start,
    train_end,
    test_start,
    test_end,
) -> pd.DataFrame:
    probabilities = _predict_up_probability(model, test_data[feature_columns])
    rows = test_data.copy()
    rows["probability_up"] = probabilities
    rows["probability_down"] = 1.0 - rows["probability_up"]
    rows["predicted_class"] = (rows["probability_up"] >= 0.5).astype(int)
    rows["actual_class"] = rows[TARGET_COLUMN].astype(int)
    rows["confidence"] = rows[["probability_up", "probability_down"]].max(axis=1)
    rows["prediction_direction"] = rows["predicted_class"].map(lambda value: "up" if value == 1 else "down")
    rows["actual_direction"] = rows["actual_class"].map(lambda value: "up" if value == 1 else "down")
    rows["train_start_date"] = pd.to_datetime(train_start)
    rows["train_end_date"] = pd.to_datetime(train_end)
    rows["test_start_date"] = pd.to_datetime(test_start)
    rows["test_end_date"] = pd.to_datetime(test_end)
    rows["probability_threshold"] = config.probability_threshold

    keep = [
        "date",
        "ticker",
        "predicted_class",
        "actual_class",
        "prediction_direction",
        "actual_direction",
        "probability_up",
        "probability_down",
        "confidence",
        "next_close_return",
        "regime",
        "surge_risk_level",
        "crash_risk_level",
        "anomaly_types",
        "dominant_narrative",
        "earnings_event_state",
        "negative_news_flag",
        "positive_news_flag",
        "importance_weighted_sentiment",
        "train_start_date",
        "train_end_date",
        "test_start_date",
        "test_end_date",
        "probability_threshold",
    ]
    return rows[[column for column in keep if column in rows.columns]]


def _predict_up_probability(model, features: pd.DataFrame) -> pd.Series:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)
        if hasattr(probabilities, "iloc"):
            values = probabilities.iloc[:, -1]
        else:
            values = [row[-1] for row in probabilities]
        return pd.Series(values, index=features.index, dtype=float)
    predictions = model.predict(features)
    return pd.Series(predictions, index=features.index, dtype=float)


def _add_strategy_returns(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    rows = predictions.copy()
    rows["actual_return"] = pd.to_numeric(rows["next_close_return"], errors="coerce")
    rows["trade"] = rows["probability_up"] >= threshold
    rows["strategy_return"] = rows["actual_return"].where(rows["trade"], 0.0).fillna(0.0)
    rows["buy_hold_return"] = rows["actual_return"].fillna(0.0)
    rows["strategy_cumulative_return"] = (1 + rows["strategy_return"]).cumprod() - 1
    rows["buy_hold_cumulative_return"] = (1 + rows["buy_hold_return"]).cumprod() - 1
    return rows


def _add_risk_context_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    if "regime" not in rows.columns:
        rows["regime"] = "Unknown"

    anomaly_raw = rows.get("anomaly_score")
    if anomaly_raw is None:
        anomaly_raw = pd.Series([0.0] * len(rows), index=rows.index)
    anomaly_score = pd.to_numeric(anomaly_raw, errors="coerce").fillna(0.0)
    anomaly_types = rows.get("anomaly_types")
    if anomaly_types is None:
        anomaly_types = pd.Series([""] * len(rows), index=rows.index)
    anomaly_types = anomaly_types.fillna("").astype(str)
    rows["surge_risk_level"] = [
        _risk_level_from_anomaly(score, types, direction="surge")
        for score, types in zip(anomaly_score, anomaly_types, strict=False)
    ]
    rows["crash_risk_level"] = [
        _risk_level_from_anomaly(score, types, direction="crash")
        for score, types in zip(anomaly_score, anomaly_types, strict=False)
    ]
    return rows


def _risk_level_from_anomaly(score: float, types: str, direction: str) -> str:
    if score < 40:
        return "Low"
    type_set = {part.strip() for part in str(types).split("|") if part.strip()}
    if direction == "surge":
        active = bool({"Volume Spike", "Price Shock"} & type_set)
    else:
        active = bool(
            {
                "Price Shock",
                "Volatility Spike",
                "Sector Divergence",
                "Market Divergence",
                "News Shock",
                "Narrative Shift",
            }
            & type_set
        )
    if not active:
        return "Low"
    if score >= 70:
        return "High"
    return "Medium"


def _group_performance(rows: pd.DataFrame, group_column: str) -> pd.DataFrame:
    result_rows: list[dict[str, object]] = []
    for group, group_rows in rows.groupby(group_column, dropna=False):
        strategy = pd.to_numeric(group_rows["strategy_return"], errors="coerce").fillna(0.0)
        buy_hold = pd.to_numeric(group_rows["buy_hold_return"], errors="coerce").fillna(0.0)
        trades = group_rows[group_rows["trade"]].copy()
        result_rows.append(
            {
                group_column: group,
                "rows": int(len(group_rows)),
                "trade_count": int(len(trades)),
                "strategy_return": _compound_return(strategy),
                "buy_hold_return": _compound_return(buy_hold),
                "win_rate": _win_rate(group_rows),
                "average_trade_return": _average_trade_return(trades),
            }
        )
    if not result_rows:
        return _empty_performance(group_column)
    return pd.DataFrame(result_rows).sort_values(group_column).reset_index(drop=True)


def _confusion_counts(actual: pd.Series, predicted: pd.Series) -> tuple[int, int, int, int]:
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    tp = int(((actual == 1) & (predicted == 1)).sum())
    return tn, fp, fn, tp


def _prediction_direction_accuracy(valid: pd.DataFrame, predicted_class: int) -> float | None:
    subset = valid[valid["predicted"] == predicted_class]
    if subset.empty:
        return None
    return float((subset["actual"] == subset["predicted"]).mean())


def _auc(actual: pd.Series, probability: pd.Series) -> float | None:
    rows = pd.DataFrame({"actual": actual, "probability": probability}).dropna()
    if rows.empty or rows["actual"].nunique() < 2:
        return None
    positives = rows[rows["actual"] == 1]
    negatives = rows[rows["actual"] == 0]
    if positives.empty or negatives.empty:
        return None
    ranks = rows["probability"].rank(method="average")
    positive_rank_sum = float(ranks[rows["actual"] == 1].sum())
    n_pos = len(positives)
    n_neg = len(negatives)
    return (positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _win_rate(rows: pd.DataFrame) -> float | None:
    if rows.empty or "trade" not in rows.columns:
        return None
    trades = rows[rows["trade"]]
    if trades.empty:
        return None
    return float((pd.to_numeric(trades["strategy_return"], errors="coerce") > 0).mean())


def _average_trade_return(trades: pd.DataFrame) -> float | None:
    if trades.empty:
        return None
    values = pd.to_numeric(trades["strategy_return"], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.mean())


def _compound_return(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if clean.empty:
        return 0.0
    return float((1 + clean).prod() - 1)


def _max_drawdown_from_returns(values) -> float | None:
    if values is None:
        return None
    returns = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if returns.empty:
        return None
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    return float(drawdown.min())


def _final_value(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    value = safe_float(frame[column].iloc[-1])
    return value


def _empty_result() -> BacktestResult:
    return BacktestResult(
        predictions=pd.DataFrame(),
        metrics=_empty_metrics(),
        equity_curve=pd.DataFrame(),
        monthly_performance=_empty_performance("month"),
        regime_performance=_empty_performance("regime"),
        surge_risk_performance=_empty_performance("surge_risk_level"),
        crash_risk_performance=_empty_performance("crash_risk_level"),
    )


def _empty_metrics() -> dict[str, object]:
    return {
        "rows": 0,
        "accuracy": None,
        "f1": None,
        "auc": None,
        "precision": None,
        "recall": None,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "up_prediction_accuracy": None,
        "down_prediction_accuracy": None,
        "trade_count": 0,
        "win_rate": None,
        "cumulative_return": None,
        "buy_hold_return": None,
        "max_drawdown": None,
    }


def _empty_performance(group_column: str) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            group_column,
            "rows",
            "trade_count",
            "strategy_return",
            "buy_hold_return",
            "win_rate",
            "average_trade_return",
        ]
    )
