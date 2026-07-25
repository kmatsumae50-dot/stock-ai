import pandas as pd

from stock_ai.lead_lag.analysis import (
    LeadLagSpec,
    analyze_lead_lag,
    build_lead_lag_reason,
    lead_lag_direction,
    rank_lead_lag_signals,
)


def _close_from_returns(returns: list[float]) -> list[float]:
    close = [100.0]
    for value in returns[1:]:
        close.append(close[-1] * (1 + value))
    return close


def test_analyze_lead_lag_finds_peer_relationship():
    dates = pd.date_range("2024-01-01", periods=160, freq="D")
    peer_returns = [0.0] * 160
    for index in range(10, 145, 8):
        peer_returns[index] = 0.025 if (index // 8) % 2 == 0 else -0.025

    toyota_returns = [0.0] * 160
    for index, value in enumerate(peer_returns[:-3]):
        toyota_returns[index + 3] = value

    target = pd.DataFrame(
        {
            "date": dates,
            "close": _close_from_returns(toyota_returns),
        }
    )
    peer = pd.DataFrame(
        {
            "date": dates,
            "candidate_return": peer_returns,
        }
    )
    spec = LeadLagSpec("TSLA", "Tesla", "tesla", "tesla_return_1d")

    metrics = analyze_lead_lag(
        target_data=target,
        candidate_frames={"TSLA": peer},
        lags=(1, 3, 5),
        specs=(spec,),
    )
    ranking = rank_lead_lag_signals(metrics)

    assert not metrics.empty
    assert ranking.loc[0, "name"] == "Tesla"
    assert int(ranking.loc[0, "lag"]) in {1, 3, 5}
    assert ranking.loc[0, "score"] > 0
    assert lead_lag_direction(ranking) in {"up", "down", "mixed", "unknown"}
    assert "Tesla" in build_lead_lag_reason(metrics, ranking)


def test_analyze_lead_lag_uses_joined_external_feature_columns():
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    honda_returns = [0.01 if index % 5 == 0 else -0.002 for index in range(80)]
    toyota_returns = [0.0] + honda_returns[:-1]
    target = pd.DataFrame(
        {
            "date": dates,
            "close": _close_from_returns(toyota_returns),
            "honda_return_1d": honda_returns,
        }
    )
    spec = LeadLagSpec("7267.T", "Honda", "honda", "honda_return_1d")

    metrics = analyze_lead_lag(target, candidate_frames={}, lags=(1,), specs=(spec,))

    assert not metrics.empty
    assert metrics.loc[0, "symbol"] == "7267.T"
    assert metrics.loc[0, "sample_size"] > 20
