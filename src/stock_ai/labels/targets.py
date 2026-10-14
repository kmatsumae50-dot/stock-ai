from __future__ import annotations

import pandas as pd

CLASS_NAMES = {
    0: "crash",
    1: "down",
    2: "flat",
    3: "up",
    4: "surge",
}

CLASS_LABELS = {name: code for code, name in CLASS_NAMES.items()}

BINARY_CLASS_NAMES = {
    0: "down",
    1: "up",
}


def classify_return(
    value: float,
    surge_threshold: float = 0.05,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
    crash_threshold: float = -0.05,
) -> int:
    if pd.isna(value):
        return -1
    if value >= surge_threshold:
        return CLASS_LABELS["surge"]
    if value >= up_threshold:
        return CLASS_LABELS["up"]
    if value <= crash_threshold:
        return CLASS_LABELS["crash"]
    if value <= down_threshold:
        return CLASS_LABELS["down"]
    return CLASS_LABELS["flat"]


def add_next_close_target(
    data: pd.DataFrame,
    surge_threshold: float = 0.05,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
    crash_threshold: float = -0.05,
) -> pd.DataFrame:
    result = data.sort_values(["ticker", "date"]).copy()
    grouped = result.groupby("ticker", group_keys=False)
    result["next_close"] = grouped["close"].shift(-1)
    result["next_close_return"] = result["next_close"] / result["close"] - 1
    result["target_next_close_class"] = result["next_close_return"].map(
        lambda value: classify_return(
            value,
            surge_threshold=surge_threshold,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            crash_threshold=crash_threshold,
        )
    )
    return result


TARGET_NEXT_CLOSE_BINARY = "target_next_close_binary"
TARGET_NEXT_OPEN_BINARY = "target_next_open_binary"


def add_next_close_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """Add next-day up/down target for the MVP binary classifier.

    Up is 1 when the next close is higher than the current close. Down is 0 when
    the next close is flat or lower. The final row per ticker has no future close,
    so it is marked as -1 and excluded from training.
    """
    result = data.sort_values(["ticker", "date"]).copy()
    if "next_close_return" not in result.columns:
        grouped = result.groupby("ticker", group_keys=False)
        result["next_close"] = grouped["close"].shift(-1)
        result["next_close_return"] = result["next_close"] / result["close"] - 1

    result[TARGET_NEXT_CLOSE_BINARY] = result["next_close_return"].map(
        lambda value: -1 if pd.isna(value) else int(value > 0)
    )
    return result


def add_next_open_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """Add overnight gap up/down target (Close-to-Open).

    Up is 1 when the next open is higher than the current close. Down is 0 when
    the next open is flat or lower. The final row per ticker has no future open,
    so it is marked as -1 and excluded from training.
    """
    result = data.sort_values(["ticker", "date"]).copy()
    if "next_open_return" not in result.columns:
        grouped = result.groupby("ticker", group_keys=False)
        result["next_open"] = grouped["open"].shift(-1)
        result["next_open_return"] = result["next_open"] / result["close"] - 1

    result[TARGET_NEXT_OPEN_BINARY] = result["next_open_return"].map(
        lambda value: -1 if pd.isna(value) else int(value > 0)
    )
    return result


def add_open_to_close_target(
    data: pd.DataFrame,
    surge_threshold: float = 0.05,
    up_threshold: float = 0.01,
    down_threshold: float = -0.01,
    crash_threshold: float = -0.05,
) -> pd.DataFrame:
    result = data.sort_values(["ticker", "date"]).copy()
    result["open_to_close_return"] = result["close"] / result["open"] - 1
    result["target_open_to_close_class"] = result["open_to_close_return"].map(
        lambda value: classify_return(
            value,
            surge_threshold=surge_threshold,
            up_threshold=up_threshold,
            down_threshold=down_threshold,
            crash_threshold=crash_threshold,
        )
    )
    return result
