from pathlib import Path

import pandas as pd

from stock_ai.backtest.walk_forward import (
    WalkForwardConfig,
    _prepare_backtest_frame,
    _test_windows,
    run_walk_forward_backtest,
    write_backtest_csvs,
)


class SignalModel:
    def predict_proba(self, features):
        probabilities = features["signal"].map(lambda value: 0.8 if value > 0 else 0.2)
        return [[1.0 - value, value] for value in probabilities]


def _model_factory(train, valid, feature_columns, target_column):
    assert train["date"].max() < valid["date"].min()
    assert "signal" in feature_columns
    assert target_column == "target_next_close_binary"
    return SignalModel()


def test_run_walk_forward_backtest_keeps_train_before_test_and_scores_strategy():
    dates = pd.date_range("2018-01-01", periods=900, freq="B")
    signal = [1 if index % 3 else -1 for index in range(len(dates))]
    target = [1 if value > 0 else 0 for value in signal]
    returns = [0.01 if value == 1 else -0.01 for value in target]
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * len(dates),
            "close": [100.0 + index * 0.1 for index in range(len(dates))],
            "signal": signal,
            "target_next_close_binary": target,
            "next_close_return": returns,
            "regime": ["Bull Trend" if value > 0 else "Bear Trend" for value in signal],
        }
    )
    config = WalkForwardConfig(
        ticker="7203.T",
        train_years=1,
        test_step_months=1,
        probability_threshold=0.55,
        min_train_rows=100,
        start_date="2019-06-01",
        end_date="2020-12-31",
    )

    result = run_walk_forward_backtest(data, config=config, model_factory=_model_factory)

    assert not result.predictions.empty
    assert result.metrics["accuracy"] == 1.0
    assert result.metrics["trade_count"] > 0
    assert result.metrics["cumulative_return"] > 0
    assert (result.predictions["train_end_date"] < result.predictions["test_start_date"]).all()
    assert not result.monthly_performance.empty
    assert not result.regime_performance.empty


def test_write_backtest_csvs_saves_expected_artifacts(tmp_path: Path):
    dates = pd.date_range("2018-01-01", periods=700, freq="B")
    signal = [1 if index % 2 else -1 for index in range(len(dates))]
    target = [1 if value > 0 else 0 for value in signal]
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * len(dates),
            "close": [100.0 + index * 0.1 for index in range(len(dates))],
            "signal": signal,
            "target_next_close_binary": target,
            "next_close_return": [0.01 if value == 1 else -0.01 for value in target],
        }
    )
    config = WalkForwardConfig(
        ticker="7203.T",
        train_years=1,
        min_train_rows=100,
        start_date="2019-03-01",
        end_date="2019-12-31",
    )
    result = run_walk_forward_backtest(data, config=config, model_factory=_model_factory)

    paths = write_backtest_csvs(result, tmp_path, ticker="7203.T", threshold=0.55)

    assert set(paths) == {
        "predictions",
        "equity",
        "monthly",
        "regime",
        "surge_risk",
        "crash_risk",
        "metrics",
    }
    assert all(path.exists() for path in paths.values())


def test_walk_forward_ignores_fold_features_that_are_entirely_missing():
    dates = pd.date_range("2018-01-01", periods=700, freq="B")
    signal = [1 if index % 2 else -1 for index in range(len(dates))]
    target = [1 if value > 0 else 0 for value in signal]
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * len(dates),
            "close": [100.0 + index * 0.1 for index in range(len(dates))],
            "signal": signal,
            "all_missing_event_feature": [None] * len(dates),
            "target_next_close_binary": target,
            "next_close_return": [0.01 if value == 1 else -0.01 for value in target],
        }
    )
    config = WalkForwardConfig(
        ticker="7203.T",
        train_years=1,
        min_train_rows=100,
        start_date="2019-03-01",
        end_date="2019-12-31",
    )

    result = run_walk_forward_backtest(data, config=config, model_factory=_model_factory)

    assert not result.predictions.empty
    assert result.metrics["accuracy"] == 1.0


def test_walk_forward_test_windows_do_not_overlap():
    dates = pd.date_range("2020-01-01", periods=900, freq="B")
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * len(dates),
            "close": [100.0 + index * 0.1 for index in range(len(dates))],
            "signal": [1] * len(dates),
            "target_next_close_binary": [1] * len(dates),
        }
    )
    config = WalkForwardConfig(ticker="7203.T", train_years=1, start_date="2021-01-01")
    frame = _prepare_backtest_frame(data, config)

    windows = _test_windows(frame, config)

    assert windows
    assert all(current_end <= next_start for (_, current_end), (next_start, _) in zip(windows, windows[1:], strict=False))
