import pandas as pd

from stock_ai.features.technical import add_technical_features, available_technical_columns


def test_features_do_not_change_when_future_rows_are_added():
    base = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=40, freq="D"),
            "ticker": ["7203.T"] * 40,
            "open": [100 + i for i in range(40)],
            "high": [101 + i for i in range(40)],
            "low": [99 + i for i in range(40)],
            "close": [100 + i for i in range(40)],
            "volume": [1_000_000 + i * 100 for i in range(40)],
        }
    )
    extended = pd.concat(
        [
            base,
            pd.DataFrame(
                {
                    "date": pd.date_range("2024-02-10", periods=5, freq="D"),
                    "ticker": ["7203.T"] * 5,
                    "open": [200 + i for i in range(5)],
                    "high": [201 + i for i in range(5)],
                    "low": [199 + i for i in range(5)],
                    "close": [200 + i for i in range(5)],
                    "volume": [2_000_000 + i * 100 for i in range(5)],
                }
            ),
        ],
        ignore_index=True,
    )

    base_features = add_technical_features(base)
    extended_features = add_technical_features(extended).iloc[: len(base_features)]

    pd.testing.assert_series_equal(
        base_features["ma_25"],
        extended_features["ma_25"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        base_features["rsi_14"],
        extended_features["rsi_14"],
        check_names=False,
    )


def test_available_technical_columns_returns_known_columns_only():
    data = pd.DataFrame({"date": [], "ticker": [], "ma_25": [], "rsi_14": [], "unknown": []})

    columns = available_technical_columns(data)

    assert columns == ["ma_25", "rsi_14"]
