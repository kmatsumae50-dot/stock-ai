from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd


MULTICLASS_PROBABILITY_COLUMNS = {
    0: "probability_crash",
    1: "probability_down",
    2: "probability_flat",
    3: "probability_up",
    4: "probability_surge",
}


def load_model_bundle(path: str | Path) -> dict[str, object]:
    import joblib

    return joblib.load(Path(path))


def predict_latest_rows(
    model,
    feature_columns: list[str],
    data: pd.DataFrame,
    target: str = "next_close",
    model_name: str = "lightgbm",
) -> pd.DataFrame:
    latest_rows = data.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest_rows = latest_rows.dropna(subset=feature_columns)

    classes = model.predict(latest_rows[feature_columns])
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(latest_rows[feature_columns]).max(axis=1)

    result = latest_rows[["date", "ticker"]].copy()
    result["target"] = target
    result["model_name"] = model_name
    result["predicted_class"] = classes.astype(int)
    result["confidence"] = probabilities
    return result.reset_index(drop=True)


def predict_latest_multiclass_rows(
    model,
    feature_columns: list[str],
    data: pd.DataFrame,
    top_features: list[dict[str, float]] | None = None,
    target: str = "next_close_multiclass",
    model_name: str = "lightgbm_multiclass",
) -> pd.DataFrame:
    latest_rows = data.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest_rows = latest_rows.dropna(subset=feature_columns)
    if latest_rows.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "target",
                "model_name",
                "predicted_class",
                "confidence",
                *MULTICLASS_PROBABILITY_COLUMNS.values(),
                "top_features",
            ]
        )

    probabilities = np.asarray(model.predict_proba(latest_rows[feature_columns]), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError("Multiclass model predict_proba must return a 2D array.")

    class_ids = _model_class_ids(model, probabilities.shape[1])
    predicted_positions = probabilities.argmax(axis=1)
    predicted_classes = np.asarray([class_ids[position] for position in predicted_positions], dtype=int)

    result = latest_rows[["date", "ticker"]].copy()
    result["target"] = target
    result["model_name"] = model_name
    result["predicted_class"] = predicted_classes
    result["confidence"] = probabilities.max(axis=1)
    for column in MULTICLASS_PROBABILITY_COLUMNS.values():
        result[column] = None
    for position, class_id in enumerate(class_ids):
        column = MULTICLASS_PROBABILITY_COLUMNS.get(class_id)
        if column is not None:
            result[column] = probabilities[:, position]
    result["top_features"] = json.dumps(top_features or [], ensure_ascii=True)
    return result.reset_index(drop=True)


def predict_latest_binary_rows(
    model,
    feature_columns: list[str],
    data: pd.DataFrame,
    top_features: list[dict[str, float]] | None = None,
    target: str = "next_close_binary",
    model_name: str = "lightgbm_binary",
) -> pd.DataFrame:
    latest_rows = data.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    latest_rows = latest_rows.dropna(subset=feature_columns)

    probabilities = model.predict_proba(latest_rows[feature_columns])
    probability_down = probabilities[:, 0]
    probability_up = probabilities[:, 1]
    predicted_class = (probability_up >= 0.5).astype(int)
    shap_contributions = get_binary_shap_contributions(
        model,
        feature_columns,
        latest_rows[feature_columns],
        top_n=10,
    )

    result = latest_rows[["date", "ticker"]].copy()
    result["target"] = target
    result["model_name"] = model_name
    result["predicted_class"] = predicted_class
    result["confidence"] = probabilities.max(axis=1)
    result["probability_up"] = probability_up
    result["probability_down"] = probability_down
    result["top_features"] = [
        json.dumps(features or top_features or [], ensure_ascii=True)
        for features in shap_contributions
    ]
    return result.reset_index(drop=True)


def _model_class_ids(model, probability_column_count: int) -> list[int]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return list(range(probability_column_count))
    result: list[int] = []
    for value in classes:
        if hasattr(value, "item"):
            value = value.item()
        result.append(int(value))
    if len(result) != probability_column_count:
        return list(range(probability_column_count))
    return result


def get_binary_shap_contributions(
    model,
    feature_columns: list[str],
    feature_frame: pd.DataFrame,
    top_n: int = 10,
) -> list[list[dict[str, object]]]:
    """Return per-row SHAP contributions for the positive/up class.

    Positive contribution means the feature pushes the binary model toward "up".
    Negative contribution means it pushes the model toward "down".
    """
    if feature_frame.empty:
        return []

    try:
        import numpy as np
        import shap
    except ModuleNotFoundError:
        return [[] for _ in range(len(feature_frame))]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(feature_frame)
        values = _select_up_class_shap_values(shap_values, feature_count=len(feature_columns))
        values = np.asarray(values)
    except Exception:
        return [[] for _ in range(len(feature_frame))]

    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        return [[] for _ in range(len(feature_frame))]

    results: list[list[dict[str, object]]] = []
    for row_position in range(min(len(feature_frame), values.shape[0])):
        row_values = values[row_position]
        feature_values = feature_frame.iloc[row_position]
        contributions = [
            _shap_contribution_item(feature, contribution, feature_values.get(feature))
            for feature, contribution in zip(feature_columns, row_values, strict=False)
        ]
        up_drivers = [item for item in contributions if item["contribution"] > 0]
        down_drivers = [item for item in contributions if item["contribution"] < 0]
        up_drivers.sort(key=lambda item: item["contribution"], reverse=True)
        down_drivers.sort(key=lambda item: item["contribution"])
        if up_drivers or down_drivers:
            results.append(up_drivers[:top_n] + down_drivers[:top_n])
        else:
            contributions.sort(key=lambda item: item["abs_contribution"], reverse=True)
            results.append(contributions[:top_n])

    while len(results) < len(feature_frame):
        results.append([])
    return results


def _shap_contribution_item(feature: str, contribution, value) -> dict[str, object]:
    contribution_value = float(contribution)
    return {
        "feature": feature,
        "value": _to_json_float(value),
        "shap_value": contribution_value,
        "contribution": contribution_value,
        "abs_contribution": abs(contribution_value),
        "direction": "up" if contribution_value >= 0 else "down",
    }


def _select_up_class_shap_values(shap_values, feature_count: int):
    import numpy as np

    if isinstance(shap_values, list):
        if len(shap_values) >= 2:
            return shap_values[1]
        if len(shap_values) == 1:
            return shap_values[0]

    values = np.asarray(shap_values)
    if values.ndim == 3:
        if values.shape[2] == 2:
            return values[:, :, 1]
        if values.shape[1] == 2:
            return values[:, 1, :]
        if values.shape[2] > 1 and values.shape[1] == feature_count:
            return values[:, :, -1]
    return values


def _to_json_float(value) -> float | None:
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None
