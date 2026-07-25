from __future__ import annotations

import pandas as pd


def build_basic_reasons(row: pd.Series) -> list[str]:
    reasons: list[str] = []

    volume_change = row.get("volume_change_1d")
    if pd.notna(volume_change) and volume_change >= 0.5:
        reasons.append("volume spike")

    rsi = row.get("rsi_14")
    if pd.notna(rsi) and rsi >= 70:
        reasons.append("RSI overbought")
    elif pd.notna(rsi) and rsi <= 30:
        reasons.append("RSI oversold")

    close_vs_ma = row.get("close_vs_ma_25")
    if pd.notna(close_vs_ma) and close_vs_ma >= 0.05:
        reasons.append("price far above MA25")
    elif pd.notna(close_vs_ma) and close_vs_ma <= -0.05:
        reasons.append("price far below MA25")

    volatility = row.get("volatility_20")
    if pd.notna(volatility) and volatility >= 0.04:
        reasons.append("high volatility")

    return reasons or ["no strong rule-based reason"]
