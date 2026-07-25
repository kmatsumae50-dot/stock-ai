from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


EXPECTED_MISSING = "expected_missing"
OPTIONAL_MISSING = "optional_missing"
NEEDS_REVIEW = "needs_review"
NOT_APPLICABLE = "not_applicable"

TARGET_COLUMNS = {
    "next_close",
    "next_close_return",
    "target_next_close_class",
    "target_next_close_binary",
    "open_to_close_return",
    "target_open_to_close_class",
}
EVENT_DISTANCE_COLUMNS = {"days_to_earnings", "days_after_earnings"}
EXTERNAL_PREFIXES = (
    "usdjpy_",
    "n225_",
    "topix_",
    "honda_",
    "nissan_",
    "tesla_",
    "byd_",
    "sector_",
)
NEWS_PREFIXES = ("news_",)
EARNINGS_PREFIXES = ("earnings_",)
OPTIONAL_TOKENS = ("reason", "summary", "context", "note", "description")
ROLLING_TOKENS = (
    "ma_",
    "close_vs_ma_",
    "rsi",
    "volatility",
    "macd",
    "bb_",
    "return_1d",
    "volume_change",
)


@dataclass(frozen=True)
class FeatureMissingnessReport:
    rows: pd.DataFrame
    total_feature_columns: int
    columns_with_missing: int
    expected_missing_count: int
    optional_missing_count: int
    needs_review_count: int
    not_applicable_count: int
    latest_row_missing_count: int


def classify_feature_missingness(data: pd.DataFrame) -> FeatureMissingnessReport:
    if data.empty:
        return _empty_report(len(data.columns))

    rows: list[dict[str, object]] = []
    for column in data.columns:
        missing_mask = data[column].isna()
        missing_count = int(missing_mask.sum())
        if missing_count == 0:
            continue

        leading_missing = _leading_missing_count(missing_mask)
        trailing_missing = _trailing_missing_count(missing_mask)
        latest_missing = bool(missing_mask.iloc[-1])
        missing_rate = missing_count / len(data)
        classification, reason = _classify_column(
            column=column,
            series=data[column],
            missing_count=missing_count,
            missing_rate=missing_rate,
            latest_missing=latest_missing,
            leading_missing=leading_missing,
            trailing_missing=trailing_missing,
        )
        rows.append(
            {
                "column": column,
                "classification": classification,
                "reason": reason,
                "missing_count": missing_count,
                "missing_rate": missing_rate,
                "latest_missing": latest_missing,
                "leading_missing": leading_missing,
                "trailing_missing": trailing_missing,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return _empty_report(len(data.columns))
    order = {
        NEEDS_REVIEW: 0,
        NOT_APPLICABLE: 1,
        OPTIONAL_MISSING: 2,
        EXPECTED_MISSING: 3,
    }
    frame["_order"] = frame["classification"].map(order).fillna(9)
    frame = frame.sort_values(
        ["_order", "latest_missing", "missing_rate", "column"],
        ascending=[True, False, False, True],
    ).drop(columns="_order")
    counts = frame["classification"].value_counts()
    return FeatureMissingnessReport(
        rows=frame.reset_index(drop=True),
        total_feature_columns=len(data.columns),
        columns_with_missing=len(frame),
        expected_missing_count=int(counts.get(EXPECTED_MISSING, 0)),
        optional_missing_count=int(counts.get(OPTIONAL_MISSING, 0)),
        needs_review_count=int(counts.get(NEEDS_REVIEW, 0)),
        not_applicable_count=int(counts.get(NOT_APPLICABLE, 0)),
        latest_row_missing_count=int(data.iloc[-1].isna().sum()),
    )


def _classify_column(
    *,
    column: str,
    series: pd.Series,
    missing_count: int,
    missing_rate: float,
    latest_missing: bool,
    leading_missing: int,
    trailing_missing: int,
) -> tuple[str, str]:
    name = column.lower()
    all_missing = missing_count == len(series)

    if column in EVENT_DISTANCE_COLUMNS:
        return NOT_APPLICABLE, "event_date_not_available_for_period"

    if column in TARGET_COLUMNS or name.startswith(("target_", "future_")):
        if trailing_missing > 0:
            return EXPECTED_MISSING, "future_target_boundary_na"

    if any(token in name for token in OPTIONAL_TOKENS):
        return OPTIONAL_MISSING, "optional_context_column"

    if latest_missing:
        if name.startswith(EXTERNAL_PREFIXES):
            return NEEDS_REVIEW, "external_data_gap"
        if name.startswith(NEWS_PREFIXES):
            return NEEDS_REVIEW, "news_feature_gap"
        if name.startswith(EARNINGS_PREFIXES):
            return NEEDS_REVIEW, "earnings_feature_gap"
        if all_missing or pd.api.types.is_numeric_dtype(series):
            return NEEDS_REVIEW, "latest_row_missing"
        return OPTIONAL_MISSING, "optional_context_column"

    if all_missing:
        return NEEDS_REVIEW, "all_values_missing"

    if leading_missing == missing_count:
        if any(token in name for token in ROLLING_TOKENS):
            return EXPECTED_MISSING, "rolling_window_initial_na"
        if name.startswith(EXTERNAL_PREFIXES):
            return EXPECTED_MISSING, "source_history_initial_gap"
        return EXPECTED_MISSING, "initial_history_not_available"

    if missing_rate >= 0.20:
        return NEEDS_REVIEW, "high_missing_rate"

    if name.startswith(EXTERNAL_PREFIXES) and missing_rate >= 0.05:
        return NEEDS_REVIEW, "external_data_gap"
    if name.startswith(NEWS_PREFIXES) and missing_rate >= 0.05:
        return NEEDS_REVIEW, "news_feature_gap"
    if name.startswith(EARNINGS_PREFIXES) and missing_rate >= 0.05:
        return NEEDS_REVIEW, "earnings_feature_gap"

    if missing_rate <= 0.01:
        return EXPECTED_MISSING, "isolated_calculation_na"
    return NEEDS_REVIEW, "unexplained_historical_gap"


def _leading_missing_count(mask: pd.Series) -> int:
    count = 0
    for missing in mask:
        if not bool(missing):
            break
        count += 1
    return count


def _trailing_missing_count(mask: pd.Series) -> int:
    count = 0
    for missing in reversed(mask.tolist()):
        if not bool(missing):
            break
        count += 1
    return count


def _empty_report(total_columns: int) -> FeatureMissingnessReport:
    return FeatureMissingnessReport(
        rows=pd.DataFrame(
            columns=[
                "column",
                "classification",
                "reason",
                "missing_count",
                "missing_rate",
                "latest_missing",
                "leading_missing",
                "trailing_missing",
            ]
        ),
        total_feature_columns=total_columns,
        columns_with_missing=0,
        expected_missing_count=0,
        optional_missing_count=0,
        needs_review_count=0,
        not_applicable_count=0,
        latest_row_missing_count=0,
    )
