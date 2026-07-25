from __future__ import annotations

import pandas as pd


def chronological_train_valid_test_split(
    data: pd.DataFrame,
    date_column: str = "date",
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_ratio <= 0 or valid_ratio <= 0 or train_ratio + valid_ratio >= 1:
        raise ValueError("train_ratio and valid_ratio must be positive and sum to less than 1.")

    ordered_dates = pd.Series(pd.to_datetime(data[date_column].unique())).sort_values().reset_index(
        drop=True
    )
    if len(ordered_dates) < 3:
        raise ValueError("At least 3 unique dates are required for train/valid/test split.")

    train_end = max(1, int(len(ordered_dates) * train_ratio))
    valid_end = max(train_end + 1, int(len(ordered_dates) * (train_ratio + valid_ratio)))

    train_dates = set(ordered_dates.iloc[:train_end])
    valid_dates = set(ordered_dates.iloc[train_end:valid_end])
    test_dates = set(ordered_dates.iloc[valid_end:])

    train = data[pd.to_datetime(data[date_column]).isin(train_dates)].copy()
    valid = data[pd.to_datetime(data[date_column]).isin(valid_dates)].copy()
    test = data[pd.to_datetime(data[date_column]).isin(test_dates)].copy()

    return train, valid, test
