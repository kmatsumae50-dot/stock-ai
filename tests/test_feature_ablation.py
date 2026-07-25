from types import SimpleNamespace

import pandas as pd

from stock_ai.ablation import feature_ablation


def _sample_data(rows: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=rows, freq="B")
    target = [1 if index % 2 else 0 for index in range(rows)]
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["7203.T"] * rows,
            "close": [100.0 + index * 0.1 for index in range(rows)],
            "volume": [1_000_000 + index * 100 for index in range(rows)],
            "return_1d": [0.01 if value else -0.01 for value in target],
            "rsi_14": [55.0 if value else 45.0 for value in target],
            "macd": [0.3 if value else -0.3 for value in target],
            "usdjpy_return_1d": [0.002] * rows,
            "n225_return_1d": [0.001] * rows,
            "sector_strength": [0.4] * rows,
            "tesla_return_1d": [0.003] * rows,
            "earnings_day_flag": [1 if index % 40 == 0 else 0 for index in range(rows)],
            "news_count_1d": [1 if index % 5 == 0 else 0 for index in range(rows)],
            "avg_sentiment_3d": [0.2 if value else -0.2 for value in target],
            "narrative_bullish_pressure": [0.6 if value else 0.1 for value in target],
            "anomaly_score": [70 if index % 13 == 0 else 10 for index in range(rows)],
            "regime": ["Bull Trend" if value else "Bear Trend" for value in target],
            "target_next_close_binary": target,
            "next_close_return": [0.01 if value else -0.01 for value in target],
        }
    )


def test_build_feature_groups_detects_context_groups():
    frame = feature_ablation.prepare_ablation_frame(_sample_data())
    groups = feature_ablation.build_feature_groups(frame)

    assert "rsi_14" in groups["technical"]
    assert "usdjpy_return_1d" in groups["external_market"]
    assert "sector_strength" in groups["auto_sector"]
    assert "earnings_day_flag" in groups["earnings_event"]
    assert "news_count_1d" in groups["news"]
    assert "narrative_bullish_pressure" in groups["narrative"]
    assert "anomaly_score" in groups["anomaly"]
    assert "regime_bull_trend" in groups["regime"]


def test_run_feature_ablation_builds_deltas_with_same_model(monkeypatch):
    def fake_run_model_comparison(
        data,
        ticker,
        probability_threshold,
        train_ratio,
        valid_ratio,
        include_xgboost,
        model_names,
        feature_columns_override,
    ):
        feature_count = len(feature_columns_override or [])
        f1 = min(0.95, 0.45 + feature_count * 0.01)
        dates = pd.date_range("2022-01-01", periods=6, freq="B")
        predictions = pd.DataFrame(
            {
                "date": dates,
                "ticker": [ticker] * len(dates),
                "model_name": [model_names[0]] * len(dates),
                "predicted_class": [1, 0, 1, 0, 1, 0],
                "actual_class": [1, 0, 0, 0, 1, 1],
                "probability_up": [0.7, 0.3, 0.6, 0.4, 0.8, 0.2],
                "probability_down": [0.3, 0.7, 0.4, 0.6, 0.2, 0.8],
                "actual_return": [0.01, -0.01, -0.01, -0.02, 0.02, 0.01],
                "trade": [True, False, True, False, True, False],
                "strategy_return": [0.01, 0.0, -0.01, 0.0, 0.02, 0.0],
                "buy_hold_return": [0.01, -0.01, -0.01, -0.02, 0.02, 0.01],
                "strategy_cumulative_return": [0.01, 0.01, -0.0001, -0.0001, 0.0199, 0.0199],
                "buy_hold_cumulative_return": [0.01, -0.0001, -0.0101, -0.0299, -0.0105, -0.0006],
            }
        )
        return SimpleNamespace(
            predictions=predictions,
            score_table=pd.DataFrame(
                [
                    {
                        "model_name": model_names[0],
                        "rows": 6,
                        "accuracy": f1,
                        "f1": f1,
                        "precision": f1,
                        "recall": f1,
                        "auc": f1,
                        "up_prediction_accuracy": f1,
                        "down_prediction_accuracy": f1,
                        "cumulative_return": feature_count * 0.001,
                        "buy_hold_return": 0.01,
                        "max_drawdown": -0.01,
                        "win_rate": 0.6,
                        "trade_count": 3,
                        "confusion_matrix": {"tn": 2, "fp": 1, "fn": 1, "tp": 2},
                    }
                ]
            ),
            skipped_models=pd.DataFrame(),
            split_summary={
                "train_rows": 40,
                "valid_rows": 20,
                "test_rows": 20,
                "train_start_date": pd.Timestamp("2021-01-01"),
                "train_end_date": pd.Timestamp("2021-03-01"),
                "valid_start_date": pd.Timestamp("2021-03-02"),
                "valid_end_date": pd.Timestamp("2021-04-01"),
                "test_start_date": pd.Timestamp("2021-04-02"),
                "test_end_date": pd.Timestamp("2021-05-01"),
            },
        )

    monkeypatch.setattr(
        feature_ablation.model_compare,
        "run_model_comparison",
        fake_run_model_comparison,
    )

    result = feature_ablation.run_feature_ablation(
        _sample_data(),
        ticker="7203.T",
        model_name="LightGBM",
    )

    assert not result.score_table.empty
    assert "all_features" in set(result.score_table["config_name"])
    assert "technical_only" in set(result.score_table["config_name"])
    assert "without_news" in set(result.score_table["config_name"])
    assert "f1_delta_vs_full" in result.score_table.columns
    assert not result.group_impact.empty
    assert result.summary_messages
