import pandas as pd

from stock_ai.backfill import prediction_backfill
from stock_ai.backfill.prediction_backfill import (
    BackfillConfig,
    build_binary_prediction_backfill,
)


class SignalModel:
    def predict_proba(self, features):
        return [[0.2, 0.8] if value > 0 else [0.8, 0.2] for value in features["signal"]]


def test_build_binary_prediction_backfill_uses_prior_rows_only(monkeypatch):
    train_windows = []

    def fake_fit(train, valid, feature_columns):
        train_windows.append((train["date"].max(), valid["date"].max(), tuple(feature_columns)))
        return SignalModel()

    monkeypatch.setattr(prediction_backfill, "_fit_lightgbm_binary", fake_fit)

    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    signal = [1 if index % 2 else -1 for index in range(len(dates))]
    data = pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * len(dates),
            "close": [100.0 + index for index in range(len(dates))],
            "volume": [1_000_000 + index for index in range(len(dates))],
            "signal": signal,
            "target_next_close_binary": [1 if value > 0 else 0 for value in signal],
            "next_close_return": [0.01 if value > 0 else -0.01 for value in signal],
        }
    )
    config = BackfillConfig(
        ticker="7203.T",
        start_date=str(dates[25].date()),
        end_date=str(dates[27].date()),
        min_train_rows=20,
    )

    result = build_binary_prediction_backfill(data, config)

    assert len(result.predictions) == 3
    assert set(result.predictions["prediction_type"]) == {"backfill"}
    assert set(result.predictions["model_name"]) == {"lightgbm_binary"}
    prediction_dates = list(pd.to_datetime(result.predictions["date"]))
    assert all(
        train_end < prediction_date
        for (train_end, _valid_end, _features), prediction_date in zip(
            train_windows,
            prediction_dates,
            strict=True,
        )
    )
    assert "signal" in result.predictions["top_features"].iloc[0]
