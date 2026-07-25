import pandas as pd

from stock_ai.labels.targets import (
    BINARY_CLASS_NAMES,
    CLASS_NAMES,
    CLASS_LABELS,
    add_next_close_binary_target,
    add_next_close_target,
    classify_return,
)


def test_classify_return_boundaries():
    assert classify_return(0.050) == CLASS_LABELS["surge"]
    assert classify_return(0.010) == CLASS_LABELS["up"]
    assert classify_return(0.000) == CLASS_LABELS["flat"]
    assert classify_return(-0.010) == CLASS_LABELS["down"]
    assert classify_return(-0.050) == CLASS_LABELS["crash"]


def test_next_close_target_uses_next_row_per_ticker():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "close": [100.0, 106.0, 105.0],
        }
    )

    result = add_next_close_target(data)

    assert result.loc[0, "target_next_close_class"] == CLASS_LABELS["surge"]
    assert result.loc[1, "target_next_close_class"] == CLASS_LABELS["flat"]
    assert result.loc[2, "target_next_close_class"] == -1


def test_next_close_binary_target():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "ticker": ["7203.T", "7203.T", "7203.T"],
            "close": [100.0, 101.0, 100.0],
        }
    )

    result = add_next_close_binary_target(data)

    assert result.loc[0, "target_next_close_binary"] == 1
    assert result.loc[1, "target_next_close_binary"] == 0
    assert result.loc[2, "target_next_close_binary"] == -1


def test_binary_class_names_are_lowercase_for_ui_labels():
    assert BINARY_CLASS_NAMES == {0: "down", 1: "up"}


def test_multiclass_class_names_are_stable_for_saved_predictions():
    assert CLASS_NAMES == {
        0: "crash",
        1: "down",
        2: "flat",
        3: "up",
        4: "surge",
    }
