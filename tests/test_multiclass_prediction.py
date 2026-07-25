import numpy as np
import pandas as pd

from stock_ai.models.baseline import evaluate_classifier
from stock_ai.models.predict import predict_latest_multiclass_rows


class FakeMulticlassModel:
    classes_ = np.array([0, 1, 2, 3, 4])

    def predict_proba(self, frame):
        assert list(frame.columns) == ["feature_a", "feature_b"]
        return np.array([[0.04, 0.10, 0.21, 0.55, 0.10]])


class SparseClassMetadataModel:
    classes_ = np.array([1, 2, 3])

    def predict_proba(self, frame):
        return np.array([[0.30, 0.05, 0.10, 0.20, 0.35]])


class FixedPredictionModel:
    def __init__(self, predictions):
        self.predictions = np.array(predictions)

    def predict(self, frame):
        return self.predictions[: len(frame)]


def test_predict_latest_multiclass_rows_stores_all_class_probabilities():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["7203.T", "7203.T"],
            "feature_a": [1.0, 2.0],
            "feature_b": [3.0, 4.0],
        }
    )

    result = predict_latest_multiclass_rows(
        FakeMulticlassModel(),
        ["feature_a", "feature_b"],
        data,
        top_features=[{"feature": "feature_a", "importance": 10.0}],
    )

    assert len(result) == 1
    assert result.loc[0, "target"] == "next_close_multiclass"
    assert result.loc[0, "model_name"] == "lightgbm_multiclass"
    assert result.loc[0, "predicted_class"] == 3
    assert result.loc[0, "confidence"] == 0.55
    assert result.loc[0, "probability_crash"] == 0.04
    assert result.loc[0, "probability_down"] == 0.10
    assert result.loc[0, "probability_flat"] == 0.21
    assert result.loc[0, "probability_up"] == 0.55
    assert result.loc[0, "probability_surge"] == 0.10
    assert "feature_a" in result.loc[0, "top_features"]


def test_predict_latest_multiclass_rows_falls_back_when_class_metadata_is_shorter():
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "ticker": ["7203.T"],
            "feature_a": [2.0],
            "feature_b": [4.0],
        }
    )

    result = predict_latest_multiclass_rows(
        SparseClassMetadataModel(),
        ["feature_a", "feature_b"],
        data,
    )

    assert result.loc[0, "predicted_class"] == 4
    assert result.loc[0, "probability_surge"] == 0.35


def test_evaluate_classifier_exposes_tail_precision_and_recall():
    data = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "target_next_close_class": [0, 0, 4, 4, 2],
        }
    )
    model = FixedPredictionModel([0, 2, 4, 2, 2])

    metrics = evaluate_classifier(
        model,
        data,
        ["feature_a"],
        target_column="target_next_close_class",
    )

    assert metrics["crash_precision"] == 1.0
    assert metrics["crash_recall"] == 0.5
    assert metrics["surge_precision"] == 1.0
    assert metrics["surge_recall"] == 0.5
    assert metrics["per_class"]["flat"]["support"] == 1
