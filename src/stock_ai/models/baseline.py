from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ai.labels import distribution as label_distribution
from stock_ai.labels.targets import CLASS_NAMES

TARGET_COLUMNS = {
    "next_close",
    "next_close_return",
    "target_next_close_class",
    "target_next_close_binary",
    "open_to_close_return",
    "target_open_to_close_class",
    "next_open",
    "next_open_return",
    "target_next_open_binary",
}

NON_FEATURE_COLUMNS = {
    "date",
    "ticker",
    "dividends",
    "stock_splits",
    *TARGET_COLUMNS,
}

OPEN_TO_CLOSE_UNAVAILABLE_COLUMNS = {
    "high",
    "low",
    "close",
    "volume",
    "return_1d",
    "volume_change_1d",
    "intraday_return",
    "high_low_range",
    "close_position",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_percent_b",
    "macd",
    "macd_signal",
    "macd_hist",
}

EXTERNAL_FEATURE_PREFIXES = (
    "usdjpy_",
    "n225_",
    "topix_",
    "honda_",
    "nissan_",
    "tesla_",
    "byd_",
    "sector_",
)
EVENT_FEATURE_PREFIXES = ("earnings_",)
EVENT_FEATURE_COLUMNS = {
    "days_to_earnings",
    "days_after_earnings",
    "post_earnings_mode",
}
NEWS_FEATURE_PREFIXES = ("news_",)
NEWS_FEATURE_COLUMNS = {
    "avg_sentiment_1d",
    "avg_sentiment_3d",
    "max_importance_3d",
    "negative_news_flag",
    "positive_news_flag",
    "sentiment_trend_3d",
    "sentiment_trend_7d",
    "importance_weighted_sentiment",
    "negative_news_shock",
    "positive_news_shock",
    "sector_news_strength",
}
NARRATIVE_FEATURE_PREFIXES = ("narrative_", "dominant_narrative_")


def select_feature_columns(
    data: pd.DataFrame,
    target_column: str = "target_next_close_class",
) -> list[str]:
    columns: list[str] = []
    for column in data.columns:
        if column in NON_FEATURE_COLUMNS:
            continue
        if _is_future_context_column(column):
            continue
        if target_column == "target_open_to_close_class" and _is_unavailable_at_open(column):
            continue
        if pd.api.types.is_numeric_dtype(data[column]):
            columns.append(column)
    return columns


def _is_future_context_column(column: str) -> bool:
    return column.startswith("anomaly_return_") and column.endswith("_after")


def _is_unavailable_at_open(column: str) -> bool:
    if column in OPEN_TO_CLOSE_UNAVAILABLE_COLUMNS:
        return True
    if column.startswith(EXTERNAL_FEATURE_PREFIXES):
        return True
    if column in EVENT_FEATURE_COLUMNS or column.startswith(EVENT_FEATURE_PREFIXES):
        return True
    if column in NEWS_FEATURE_COLUMNS or column.startswith(NEWS_FEATURE_PREFIXES):
        return True
    if column.startswith(NARRATIVE_FEATURE_PREFIXES):
        return True
    return column.startswith("ma_") or column.startswith("close_vs_ma_") or column.startswith("rsi_")


def train_lightgbm_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_column: str = "target_next_close_class",
    weight_mode: str = "none",
):
    import lightgbm as lgb

    active_weight_mode = label_distribution.normalize_weight_mode(weight_mode)
    class_weight = label_distribution.multiclass_class_weight(active_weight_mode)
    feature_columns = select_feature_columns(train, target_column=target_column)
    train_data = train.dropna(subset=feature_columns + [target_column])
    valid_data = valid.dropna(subset=feature_columns + [target_column])
    train_data = train_data[train_data[target_column] >= 0]
    valid_data = valid_data[valid_data[target_column] >= 0]

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        class_weight=class_weight,
        random_state=42,
    )
    model.fit(
        train_data[feature_columns],
        train_data[target_column],
        eval_set=[(valid_data[feature_columns], valid_data[target_column])],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return model, feature_columns


def train_lightgbm_binary_classifier(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    target_column: str = "target_next_close_binary",
):
    import lightgbm as lgb

    feature_columns = select_feature_columns(train, target_column=target_column)
    train_data = train.dropna(subset=feature_columns + [target_column])
    valid_data = valid.dropna(subset=feature_columns + [target_column])
    train_data = train_data[train_data[target_column] >= 0]
    valid_data = valid_data[valid_data[target_column] >= 0]

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(
        train_data[feature_columns],
        train_data[target_column],
        eval_set=[(valid_data[feature_columns], valid_data[target_column])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)],
    )
    return model, feature_columns


def evaluate_classifier(
    model,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_next_close_class",
) -> dict[str, object]:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    eval_data = data.dropna(subset=feature_columns + [target_column])
    eval_data = eval_data[eval_data[target_column] >= 0]
    labels = sorted(CLASS_NAMES)
    if eval_data.empty:
        per_class = {
            CLASS_NAMES[label]: {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "support": 0,
            }
            for label in labels
        }
        return {
            "accuracy": None,
            "macro_f1": None,
            "weighted_f1": None,
            "confusion_matrix": [[0 for _ in labels] for _ in labels],
            "per_class": per_class,
            "surge_recall": 0.0,
            "crash_recall": 0.0,
            "surge_precision": 0.0,
            "crash_precision": 0.0,
            "rows": 0,
            "classification_report": {},
        }

    actual = eval_data[target_column].astype(int)
    predictions = model.predict(eval_data[feature_columns]).astype(int)
    precision, recall, f1, support = precision_recall_fscore_support(
        actual,
        predictions,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        CLASS_NAMES[label]: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "macro_f1": float(f1_score(actual, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(actual, predictions, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=labels).tolist(),
        "per_class": per_class,
        "surge_recall": per_class["surge"]["recall"],
        "crash_recall": per_class["crash"]["recall"],
        "surge_precision": per_class["surge"]["precision"],
        "crash_precision": per_class["crash"]["precision"],
        "rows": int(len(eval_data)),
        "classification_report": classification_report(
            actual,
            predictions,
            labels=labels,
            target_names=[CLASS_NAMES[label] for label in labels],
            zero_division=0,
            output_dict=True,
        ),
    }


def evaluate_binary_classifier(
    model,
    data: pd.DataFrame,
    feature_columns: list[str],
    target_column: str = "target_next_close_binary",
) -> dict[str, object]:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    eval_data = data.dropna(subset=feature_columns + [target_column])
    eval_data = eval_data[eval_data[target_column] >= 0]
    actual = eval_data[target_column].astype(int)
    predictions = model.predict(eval_data[feature_columns]).astype(int)
    probabilities = model.predict_proba(eval_data[feature_columns])[:, 1]

    auc = None
    if actual.nunique() == 2:
        auc = float(roc_auc_score(actual, probabilities))

    return {
        "accuracy": float(accuracy_score(actual, predictions)),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "auc": auc,
        "rows": int(len(eval_data)),
    }


def get_feature_importance(
    model,
    feature_columns: list[str],
    top_n: int = 10,
) -> list[dict[str, float]]:
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    ranking = sorted(
        zip(feature_columns, importances, strict=False),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in ranking[:top_n]
    ]


def save_model(
    model,
    feature_columns: list[str],
    path: str | Path,
    metadata: dict[str, object] | None = None,
) -> Path:
    import joblib

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "metadata": metadata or {},
        },
        output_path,
    )
    return output_path
