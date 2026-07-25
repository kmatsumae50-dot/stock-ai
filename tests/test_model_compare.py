import pandas as pd

from stock_ai.model_compare.compare import (
    compare_condition_performance,
    model_characteristics_frame,
    run_model_comparison,
)


def _sample_data(rows: int = 520) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=rows, freq="B")
    signal = [1 if index % 4 in (1, 2) else -1 for index in range(rows)]
    target = [1 if value > 0 else 0 for value in signal]
    returns = [0.012 if value == 1 else -0.011 for value in target]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "close": [100.0 + index * 0.1 for index in range(rows)],
            "volume": [1_000_000 + index * 100 for index in range(rows)],
            "signal": signal,
            "rsi_14": [55.0 if value > 0 else 42.0 for value in signal],
            "macd": [0.5 if value > 0 else -0.5 for value in signal],
            "volatility_20": [0.02 + (index % 5) * 0.001 for index in range(rows)],
            "target_next_close_binary": target,
            "next_close_return": returns,
            "regime": ["Bull Trend" if value > 0 else "Bear Trend" for value in signal],
            "dominant_narrative": ["EV" if value > 0 else "Rates" for value in signal],
            "earnings_event_state": ["normal"] * rows,
        }
    )


def test_run_model_comparison_uses_same_chronological_split_and_scores_models():
    result = run_model_comparison(
        _sample_data(),
        ticker="7203.T",
        probability_threshold=0.55,
        model_names=("Dummy Classifier", "Logistic Regression", "Random Forest"),
    )

    assert not result.predictions.empty
    assert set(result.score_table["model_name"]) == {
        "Dummy Classifier",
        "Logistic Regression",
        "Random Forest",
    }
    assert result.split_summary["train_end_date"] < result.split_summary["valid_start_date"]
    assert result.split_summary["valid_end_date"] < result.split_summary["test_start_date"]
    assert result.predictions.groupby("model_name").size().nunique() == 1
    assert result.score_table["trade_count"].notna().all()
    assert not result.metric_rankings.empty
    assert "Dummy Classifier" in set(result.metric_rankings["model_name"])


def test_compare_condition_performance_and_characteristics_are_available():
    result = run_model_comparison(
        _sample_data(),
        ticker="7203.T",
        model_names=("Dummy Classifier", "Logistic Regression"),
    )

    by_regime = compare_condition_performance(
        result.predictions,
        group_column="regime",
        metric="accuracy",
        min_count=1,
    )
    characteristics = model_characteristics_frame()

    assert not by_regime.empty
    assert {"model_name", "regime", "accuracy"}.issubset(by_regime.columns)
    assert "Dummy Classifier" in set(characteristics["model_name"])
