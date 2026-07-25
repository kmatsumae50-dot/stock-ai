import pandas as pd

from stock_ai.validation.time_split import chronological_train_valid_test_split


def test_chronological_split_keeps_time_order():
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10, freq="D"),
            "ticker": ["7203.T"] * 10,
            "close": range(10),
        }
    )

    train, valid, test = chronological_train_valid_test_split(
        data,
        train_ratio=0.6,
        valid_ratio=0.2,
    )

    assert train["date"].max() < valid["date"].min()
    assert valid["date"].max() < test["date"].min()
    assert len(train) == 6
    assert len(valid) == 2
    assert len(test) == 2
