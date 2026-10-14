from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stock_ai.backtest.prediction_history import evaluate_prediction_history
from stock_ai.labels.targets import (
    TARGET_NEXT_OPEN_BINARY,
    add_next_open_binary_target,
)
from stock_ai.models.overnight import (
    OVERNIGHT_MODEL_NAME,
    OVERNIGHT_TARGET,
    classify_overnight_quantile,
    predict_latest_overnight_rows,
    train_lightgbm_overnight_classifier,
)
from stock_ai.storage.sqlite_store import (
    init_database,
    read_latest_predictions,
    read_prediction_history,
    upsert_predictions,
)


def test_add_next_open_binary_target_rules_and_boundary_conditions():
    """Verify add_next_open_binary_target:
    - next_open > current_close -> 1 (Up)
    - next_open <= current_close -> 0 (Down/Flat)
    - final row or missing next_open -> -1
    - no data leak across different tickers
    """
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
            "ticker": ["7203.T", "7203.T", "7203.T", "7203.T"],
            "open": [100.0, 105.0, 95.0, 102.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        }
    )

    result = add_next_open_binary_target(df)

    assert TARGET_NEXT_OPEN_BINARY in result.columns
    # Row 0: next_open is 105.0 > close 100.0 -> 1
    assert result.loc[0, TARGET_NEXT_OPEN_BINARY] == 1
    # Row 1: next_open is 95.0 < close 100.0 -> 0
    assert result.loc[1, TARGET_NEXT_OPEN_BINARY] == 0
    # Row 2: next_open is 102.0 > close 100.0 -> 1
    assert result.loc[2, TARGET_NEXT_OPEN_BINARY] == 1
    # Row 3 (last row): next_open is NaN -> -1
    assert result.loc[3, TARGET_NEXT_OPEN_BINARY] == -1


def test_add_next_open_binary_target_flat_returns_zero():
    """Flat overnight return (next_open == current_close) should be 0 (Down/Flat)."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": ["7203.T", "7203.T"],
            "open": [100.0, 100.0],
            "close": [100.0, 100.0],
        }
    )

    result = add_next_open_binary_target(df)
    assert result.loc[0, TARGET_NEXT_OPEN_BINARY] == 0
    assert result.loc[1, TARGET_NEXT_OPEN_BINARY] == -1


def test_add_next_open_binary_target_ticker_grouping_prevents_leak():
    """Test that shifting next_open does not leak across different tickers."""
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "ticker": ["1111.T", "1111.T", "2222.T", "2222.T"],
            "open": [100.0, 110.0, 50.0, 40.0],
            "close": [100.0, 105.0, 50.0, 45.0],
        }
    )

    result = add_next_open_binary_target(df)

    t1 = result[result["ticker"] == "1111.T"].reset_index(drop=True)
    assert t1.loc[0, TARGET_NEXT_OPEN_BINARY] == 1  # 110.0 > 100.0
    assert t1.loc[1, TARGET_NEXT_OPEN_BINARY] == -1  # end of 1111.T

    t2 = result[result["ticker"] == "2222.T"].reset_index(drop=True)
    assert t2.loc[0, TARGET_NEXT_OPEN_BINARY] == 0  # 40.0 < 50.0
    assert t2.loc[1, TARGET_NEXT_OPEN_BINARY] == -1  # end of 2222.T


def test_classify_overnight_quantile_bands():
    """Verify that empirical probability thresholds match the Clean Test calibration."""
    q5 = classify_overnight_quantile(0.65)
    assert q5["quantile"] == "Q5"
    assert q5["direction"] == "up"
    assert "強気ギャップアップ" in q5["label"]
    assert q5["historical_win_rate"] == 0.6446

    q4 = classify_overnight_quantile(0.55)
    assert q4["quantile"] == "Q4"
    assert q4["direction"] == "up"

    q3 = classify_overnight_quantile(0.50)
    assert q3["quantile"] == "Q3"
    assert q3["direction"] == "neutral"

    q2 = classify_overnight_quantile(0.40)
    assert q2["quantile"] == "Q2"
    assert q2["direction"] == "down"

    q1 = classify_overnight_quantile(0.35)
    assert q1["quantile"] == "Q1"
    assert q1["direction"] == "down"
    assert "警戒" in q1["label"]
    assert q1["historical_win_rate"] == 0.4132


def test_train_and_predict_overnight_format():
    """Train a small overnight classifier and check prediction output structure."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "ticker": "7203.T",
            "open": np.random.uniform(90, 110, size=100),
            "close": np.random.uniform(90, 110, size=100),
            "feat_1": np.random.normal(size=100),
            "feat_2": np.random.normal(size=100),
        }
    )
    df = add_next_open_binary_target(df)

    train_data = df.iloc[:70]
    valid_data = df.iloc[70:90]
    model, features = train_lightgbm_overnight_classifier(
        train_data,
        valid_data,
        feature_columns=["feat_1", "feat_2"],
    )

    assert features == ["feat_1", "feat_2"]

    # Predict latest rows
    preds = predict_latest_overnight_rows(
        model,
        features,
        df,
        target=OVERNIGHT_TARGET,
        model_name=OVERNIGHT_MODEL_NAME,
    )

    assert len(preds) == 1
    assert preds.iloc[0]["target"] == OVERNIGHT_TARGET
    assert preds.iloc[0]["model_name"] == OVERNIGHT_MODEL_NAME
    assert 0.0 <= preds.iloc[0]["probability_up"] <= 1.0
    assert 0.0 <= preds.iloc[0]["probability_down"] <= 1.0
    assert preds.iloc[0]["predicted_class"] in (0, 1)

    # Check notes JSON structure
    notes = json.loads(preds.iloc[0]["notes"])
    assert "quantile" in notes
    assert "label" in notes
    assert "historical_win_rate" in notes


def test_sqlite_parallel_storage_and_evaluation_without_collision(tmp_path: Path):
    """Verify that Close-to-Close and Close-to-Open coexist in SQLite without corrupting each other."""
    db_path = tmp_path / "test_stock_ai.db"
    init_database(db_path)

    date_str = "2024-01-01"
    ticker = "7203.T"

    # 1. Save Close-to-Close prediction
    c2c_df = pd.DataFrame(
        [
            {
                "date": date_str,
                "ticker": ticker,
                "target": "next_close_binary",
                "model_name": "lightgbm_binary",
                "predicted_class": 1,
                "confidence": 0.65,
                "probability_up": 0.65,
                "probability_down": 0.35,
                "notes": "c2c_test",
            }
        ]
    )
    upsert_predictions(c2c_df, db_path)

    # 2. Save Close-to-Open (Overnight) prediction
    c2o_df = pd.DataFrame(
        [
            {
                "date": date_str,
                "ticker": ticker,
                "target": OVERNIGHT_TARGET,
                "model_name": OVERNIGHT_MODEL_NAME,
                "predicted_class": 1,
                "confidence": 0.72,
                "probability_up": 0.72,
                "probability_down": 0.28,
                "notes": json.dumps({"quantile": "Q5", "label": "強気ギャップアップ帯"}),
            }
        ]
    )
    upsert_predictions(c2o_df, db_path)

    # 3. Read both back from SQLite
    all_latest = read_latest_predictions(db_path)
    assert len(all_latest) == 2

    c2c_read = read_prediction_history(
        db_path,
        ticker=ticker,
        target="next_close_binary",
        model_name="lightgbm_binary",
    )
    assert len(c2c_read) == 1
    assert c2c_read.iloc[0]["target"] == "next_close_binary"
    assert c2c_read.iloc[0]["probability_up"] == 0.65

    c2o_read = read_prediction_history(
        db_path,
        ticker=ticker,
        target=OVERNIGHT_TARGET,
        model_name=OVERNIGHT_MODEL_NAME,
    )
    assert len(c2o_read) == 1
    assert c2o_read.iloc[0]["target"] == OVERNIGHT_TARGET
    assert c2o_read.iloc[0]["probability_up"] == 0.72

    # Verify that quantile classification operates seamlessly on the stored probability_up
    q_info = classify_overnight_quantile(float(c2o_read.iloc[0]["probability_up"]))
    assert q_info["quantile"] == "Q5"
    assert "強気ギャップアップ" in q_info["label"]

    # 4. Evaluation test: Overnight settles against next_open, C2C settles against next_close
    # Day 0: close=100.0, open=98.0
    # Day 1: open=105.0 (overnight return +5%), close=95.0 (close-to-close return -5%)
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "ticker": [ticker, ticker],
            "open": [98.0, 105.0],
            "close": [100.0, 95.0],
        }
    )

    # C2C: predicted class 1 (Up), but actual next close is 95.0 (< 100.0) -> incorrect (0)
    c2c_history, c2c_metrics = evaluate_prediction_history(
        c2c_read,
        prices,
        target="next_close_binary",
    )
    assert c2c_metrics["settled_predictions"] == 1
    assert c2c_metrics["correct_predictions"] == 0
    assert c2c_history.iloc[0]["actual_class"] == 0

    # C2O: predicted class 1 (Up), actual next open is 105.0 (> 100.0) -> correct (1)
    c2o_history, c2o_metrics = evaluate_prediction_history(
        c2o_read,
        prices,
        target=OVERNIGHT_TARGET,
    )
    assert c2o_metrics["settled_predictions"] == 1
    assert c2o_metrics["correct_predictions"] == 1
    assert c2o_history.iloc[0]["actual_class"] == 1
    assert pytest.approx(c2o_history.iloc[0]["actual_return"], 1e-4) == 0.05
