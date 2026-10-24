"""src/stock_ai/features/adaptive_online_learning.py

Online Adaptive Learning & Closed-Loop Self-Evolution Engine.

Theoretic Foundations:
1. Nicolò Cesa-Bianchi & Gábor Lugosi: 'Prediction, Learning, and Games' (2006)
   - Exponentially Weighted Forecaster / Hedge Algorithm for waveform and regime experts.
   - Dynamic weight updates: w_{m, t+1} \\propto w_{m, t} * exp(-eta * loss_{m, t}), clipped to [0.60, 1.40].
2. Richard S. Sutton & Andrew G. Barto: 'Reinforcement Learning: An Introduction' (2018)
   - Rescorla-Wagner / Temporal-Difference (TD) prediction error tracker.
   - delta_{i, t} = R_{actual, i, t} - R_{expected, i, t}
   - Tracked via EMA (alpha = 0.20 for ticker, alpha = 0.15 for sector) to quantify credibility.
3. Contextual Bandit Memory:
   - Failure archetype profile memory (stopped out or yoriten-fade trades).
   - Penalizes candidate stocks matching prior failure profiles (gap_to_atr, hurst, elasticity, clv).

Zero-Leakage Assurance:
- Updates strictly occur post-market (15:35 JST) from settled scorecards.
- Day t predictions strictly utilize state updated up to day t-1.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants
DEFAULT_STATE_FILE = Path("data/adaptive_memory/online_adaptation_state.json")

DEFAULT_WAVEFORM_KEYS = [
    "gu_rebound_high",
    "gd_v_shape",
    "trend_breakout",
    "yoriten_fade",
    "gd_downtrend",
    "box_range",
]

STYLE_MOMENTUM_WAVEFORMS = {"gu_rebound_high", "trend_breakout"}
STYLE_MEAN_REVERSION_WAVEFORMS = {"gd_v_shape", "box_range"}
STYLE_DOWNSIDE_WAVEFORMS = {"yoriten_fade", "gd_downtrend"}

# Clipping and hyperparameters
WEIGHT_CLIP_MIN = 0.60
WEIGHT_CLIP_MAX = 1.40
HEDGE_ETA = 0.20
HEDGE_INACTIVE_DECAY = 0.96

TD_LEARNING_RATE_TICKER = 0.20
TD_LEARNING_RATE_SECTOR = 0.15
TD_V_CLIP_TICKER = 0.08
TD_V_CLIP_SECTOR = 0.06

MAX_FAILURE_ARCHETYPES = 25
FAILURE_SIMILARITY_THRESHOLD = 0.72


@dataclass
class FailureArchetype:
    date: str
    ticker: str
    reason: str  # "STOPPED_OUT" or "YORITEN_FADE"
    gap_to_atr: float
    chan_hurst_20d: float
    micro_upward_elasticity_10d: float
    prior_clv: float


def initialize_fresh_state() -> dict[str, Any]:
    """Create a pristine, un-biased online adaptation state."""
    expert_weights = {wf: 1.0 for wf in DEFAULT_WAVEFORM_KEYS}
    expert_weights["style_momentum"] = 1.0
    expert_weights["style_mean_reversion"] = 1.0

    return {
        "version": "7.0.0",
        "last_updated_date": None,
        "total_updates": 0,
        "regime_expert_weights": expert_weights,
        "ticker_td_state": {},
        "sector_td_state": {},
        "failure_archetypes": [],
        "history_log": [],
    }


def load_adaptation_state(state_file: Path | str | None = None) -> dict[str, Any]:
    """Load persistent online adaptation state JSON, or initialize default if not found."""
    p = Path(state_file) if state_file is not None else DEFAULT_STATE_FILE
    if p.exists():
        try:
            content = p.read_text(encoding="utf-8")
            data = json.loads(content)
            if "regime_expert_weights" in data and "ticker_td_state" in data:
                return data
        except Exception as exc:
            logger.warning("Failed to parse online adaptation state from %s: %s", p, exc)

    fresh = initialize_fresh_state()
    return fresh


def save_adaptation_state(state: dict[str, Any], state_file: Path | str | None = None) -> None:
    """Persist online adaptation state JSON with parent directory creation."""
    p = Path(state_file) if state_file is not None else DEFAULT_STATE_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        p.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.error("Failed to save online adaptation state to %s: %s", p, exc)


# ===========================================================================
# 1. Hedge Algorithm: Waveform & Regime Expert Weight Updates
# ===========================================================================

def update_regime_expert_weights(
    current_weights: Mapping[str, float],
    trades: Sequence[dict[str, Any]],
    eta: float = HEDGE_ETA,
) -> dict[str, float]:
    """Update waveform and trading style expert weights using Hedge Algorithm.

    From Cesa-Bianchi & Lugosi (2006):
        w_{m, t+1} = w_{m, t} * exp(-eta * (loss_{m, t} - baseline_loss))
        Clipped to [0.60, 1.40].

    Loss formulation:
        For trade under waveform m:
        - WIN / target hit: low loss (0.05 ~ 0.20)
        - LOSS / stopped out: high loss (0.80 ~ 1.00)
        - Net return normalization: loss = clip(0.50 - 0.50 * (net_return_pct / 2.0), 0.0, 1.0)
    """
    updated: dict[str, float] = dict(current_weights)

    # Ensure all defaults exist
    for k in DEFAULT_WAVEFORM_KEYS:
        if k not in updated:
            updated[k] = 1.0
    if "style_momentum" not in updated:
        updated["style_momentum"] = 1.0
    if "style_mean_reversion" not in updated:
        updated["style_mean_reversion"] = 1.0

    # Group trades by waveform key
    wf_trades: dict[str, list[dict[str, Any]]] = {k: [] for k in DEFAULT_WAVEFORM_KEYS}
    for t in trades:
        # Match predicted waveform to key
        wf_title = t.get("predicted_waveform", "")
        wf_key = t.get("waveform_key")
        if not wf_key:
            # Map by title keyword
            if "GU" in wf_title or "じり高" in wf_title:
                wf_key = "gu_rebound_high"
            elif "GD" in wf_title and "V字" in wf_title:
                wf_key = "gd_v_shape"
            elif "独歩高" in wf_title or "トレンド" in wf_title or "突破" in wf_title:
                wf_key = "trend_breakout"
            elif "寄り天" in wf_title or "失速" in wf_title:
                wf_key = "yoriten_fade"
            elif "一段安" in wf_title or "下落" in wf_title:
                wf_key = "gd_downtrend"
            else:
                wf_key = "box_range"

        if wf_key in wf_trades:
            wf_trades[wf_key].append(t)

    momentum_losses: list[float] = []
    reversion_losses: list[float] = []

    for wf_key in DEFAULT_WAVEFORM_KEYS:
        recs = wf_trades[wf_key]
        if recs:
            losses: list[float] = []
            for r in recs:
                net_ret = float(r.get("net_return_pct", 0.0))
                status = str(r.get("result_status", "NEUTRAL")).upper()
                norm_ret = float(np.clip(net_ret / 2.0, -1.0, 1.0))
                # Base loss from normalized return
                l_val = float(np.clip(0.50 - 0.50 * norm_ret, 0.0, 1.0))
                if status == "WIN":
                    l_val = min(l_val, 0.25)
                elif status == "LOSS":
                    l_val = max(l_val, 0.75)
                losses.append(l_val)

            avg_loss = float(np.mean(losses))
            # Hedge exponential update centered around neutral loss 0.50
            loss_diff = avg_loss - 0.50
            new_w = updated[wf_key] * np.exp(-eta * loss_diff)
            updated[wf_key] = float(np.clip(new_w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX))

            if wf_key in STYLE_MOMENTUM_WAVEFORMS:
                momentum_losses.extend(losses)
            elif wf_key in STYLE_MEAN_REVERSION_WAVEFORMS:
                reversion_losses.extend(losses)
        else:
            # Inactive decay towards 1.0
            cur = updated[wf_key]
            updated[wf_key] = float(HEDGE_INACTIVE_DECAY * cur + (1.0 - HEDGE_INACTIVE_DECAY) * 1.0)

    # Update style experts
    if momentum_losses:
        avg_m_loss = float(np.mean(momentum_losses))
        new_m_w = updated["style_momentum"] * np.exp(-eta * (avg_m_loss - 0.50))
        updated["style_momentum"] = float(np.clip(new_m_w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX))
    else:
        updated["style_momentum"] = float(0.98 * updated["style_momentum"] + 0.02 * 1.0)

    if reversion_losses:
        avg_r_loss = float(np.mean(reversion_losses))
        new_r_w = updated["style_mean_reversion"] * np.exp(-eta * (avg_r_loss - 0.50))
        updated["style_mean_reversion"] = float(np.clip(new_r_w, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX))
    else:
        updated["style_mean_reversion"] = float(0.98 * updated["style_mean_reversion"] + 0.02 * 1.0)

    return {k: round(v, 4) for k, v in updated.items()}


# ===========================================================================
# 2. Rescorla-Wagner TD Error Credibility Tracker
# ===========================================================================

def update_ticker_td_credibility(
    ticker_state: Mapping[str, Any],
    sector_state: Mapping[str, Any],
    trades: Sequence[dict[str, Any]],
    alpha_ticker: float = TD_LEARNING_RATE_TICKER,
    alpha_sector: float = TD_LEARNING_RATE_SECTOR,
    ticker_sector_map: Mapping[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Update ticker and sector credibility using TD prediction errors.

    From Sutton & Barto (2018):
        delta_{i, t} = R_{actual, i, t} - R_{expected, i, t}
        V_{i, t+1} = V_{i, t} + alpha * delta_{i, t}
    """
    new_ticker_state: dict[str, Any] = {k: dict(v) for k, v in ticker_state.items()}
    new_sector_state: dict[str, Any] = {k: dict(v) for k, v in sector_state.items()}

    sec_errors: dict[str, list[float]] = {}

    for t in trades:
        ticker = t["ticker"]
        # Expected net return
        prob_up = float(t.get("probability_up", 55.0))
        # Expected return estimation from probability or explicitly provided expected return
        exp_return = float(t.get("expected_return_bps", (prob_up - 50.0) * 1.0)) / 10000.0
        # Actual net return after 20bps
        act_net_return = float(t.get("net_return_pct", 0.0)) / 100.0

        # TD prediction error
        td_error = act_net_return - exp_return

        t_node = new_ticker_state.get(ticker, {"v": 0.0, "updates_count": 0, "cumulative_td": 0.0})
        old_v = float(t_node.get("v", 0.0))
        new_v = old_v + alpha_ticker * td_error
        new_v = float(np.clip(new_v, -TD_V_CLIP_TICKER, TD_V_CLIP_TICKER))

        new_ticker_state[ticker] = {
            "v": round(new_v, 5),
            "updates_count": int(t_node.get("updates_count", 0)) + 1,
            "cumulative_td": round(float(t_node.get("cumulative_td", 0.0)) + td_error, 5),
            "last_td_error": round(td_error, 5),
        }

        # Resolve sector
        sector = t.get("sector")
        if not sector and ticker_sector_map:
            sector = ticker_sector_map.get(ticker)
        if sector:
            sec_errors.setdefault(sector, []).append(td_error)

    # Sector-level aggregation
    for sec, errs in sec_errors.items():
        avg_sec_err = float(np.mean(errs))
        s_node = new_sector_state.get(sec, {"v": 0.0, "updates_count": 0})
        old_sec_v = float(s_node.get("v", 0.0))
        new_sec_v = old_sec_v + alpha_sector * avg_sec_err
        new_sec_v = float(np.clip(new_sec_v, -TD_V_CLIP_SECTOR, TD_V_CLIP_SECTOR))

        new_sector_state[sec] = {
            "v": round(new_sec_v, 5),
            "updates_count": int(s_node.get("updates_count", 0)) + len(errs),
            "last_td_error": round(avg_sec_err, 5),
        }

    return new_ticker_state, new_sector_state


def compute_adaptive_credibility_multiplier(
    ticker: str,
    sector: str | None,
    ticker_state: Mapping[str, Any],
    sector_state: Mapping[str, Any],
) -> float:
    """Compute combined credibility multiplier for ticker and sector.

    Returns:
        multiplier in range [0.60, 1.40], default 1.0 for untracked assets.
    """
    v_tick = 0.0
    if ticker in ticker_state:
        v_tick = float(ticker_state[ticker].get("v", 0.0))

    v_sec = 0.0
    if sector and sector in sector_state:
        v_sec = float(sector_state[sector].get("v", 0.0))

    # Scale factor: V in [-0.08, +0.08] maps to [-0.32, +0.32] boost/penalty
    raw_mult = 1.0 + (v_tick * 4.0) + (v_sec * 2.5)
    return float(np.clip(raw_mult, WEIGHT_CLIP_MIN, WEIGHT_CLIP_MAX))


# ===========================================================================
# 3. Contextual Bandit Memory: Failure Archetype Profiler
# ===========================================================================

def extract_failure_profile_from_trade(
    trade: dict[str, Any],
    features_df: pd.DataFrame | None = None,
) -> FailureArchetype | None:
    """Extract contextual feature profile if the trade resulted in a critical failure.

    Failure criteria:
    - STOPPED_OUT (hit intraday stop loss barrier)
    - YORITEN_FADE (morning surge followed by net loss / severe pullback)
    """
    res_code = str(trade.get("result_code", "")).upper()
    res_status = str(trade.get("result_status", "")).upper()
    act_open = float(trade.get("actual_open", 1000.0))
    act_high = float(trade.get("actual_high", act_open))
    net_ret = float(trade.get("net_return_pct", 0.0))

    is_stopped_out = res_code == "STOPPED_OUT"
    is_yoriten = (res_status == "LOSS" and act_high >= act_open * 1.008) or (net_ret < -0.8 and act_high >= act_open * 1.005)

    if not (is_stopped_out or is_yoriten):
        return None

    reason = "STOPPED_OUT" if is_stopped_out else "YORITEN_FADE"

    # Default profile values
    gap_to_atr = 0.50
    chan_hurst = 0.50
    micro_upward = 1.00
    prior_clv = 0.50

    # Extract from trade or features_df if available
    if "gap_to_atr" in trade:
        gap_to_atr = float(trade["gap_to_atr"])
    elif features_df is not None and not features_df.empty:
        last = features_df.iloc[-1]
        vol = float(last.get("volatility_20", 0.02))
        prev_c = float(last.get("close", act_open))
        gap = abs(act_open - prev_c) / prev_c if prev_c > 0 else 0.0
        gap_to_atr = float(np.clip(gap / max(1e-4, vol), 0.0, 3.0))

    if "chan_hurst" in trade and trade["chan_hurst"] is not None:
        chan_hurst = float(trade["chan_hurst"])
    elif features_df is not None and not features_df.empty:
        for c in ["prior_chan_hurst_20d", "chan_hurst_20d"]:
            if c in features_df.columns:
                h = float(features_df[c].iloc[-1])
                if not np.isnan(h):
                    chan_hurst = h
                    break

    if "micro_upward_elasticity" in trade and trade["micro_upward_elasticity"] is not None:
        micro_upward = float(trade["micro_upward_elasticity"])
    elif features_df is not None and not features_df.empty:
        for c in ["prior_micro_upward_elasticity_10d", "micro_upward_elasticity_10d"]:
            if c in features_df.columns:
                u = float(features_df[c].iloc[-1])
                if not np.isnan(u):
                    micro_upward = u
                    break

    if "prior_clv" in trade:
        prior_clv = float(trade["prior_clv"])
    elif features_df is not None and not features_df.empty:
        last = features_df.iloc[-1]
        c = float(last.get("close", 1000.0))
        h = float(last.get("high", c))
        l_p = float(last.get("low", c))
        rng = max(1e-6, h - l_p)
        prior_clv = float(np.clip((c - l_p) / rng, 0.0, 1.0))

    return FailureArchetype(
        date=trade.get("date", datetime.now().strftime("%Y-%m-%d")),
        ticker=trade["ticker"],
        reason=reason,
        gap_to_atr=round(gap_to_atr, 3),
        chan_hurst_20d=round(chan_hurst, 3),
        micro_upward_elasticity_10d=round(micro_upward, 3),
        prior_clv=round(prior_clv, 3),
    )


def update_failure_archetypes(
    current_archetypes: Sequence[dict[str, Any]],
    trades: Sequence[dict[str, Any]],
    features_loader_func: Any | None = None,
    max_archetypes: int = MAX_FAILURE_ARCHETYPES,
) -> list[dict[str, Any]]:
    """Append new failure archetype profiles and maintain FIFO capacity."""
    archetypes = [dict(a) for a in current_archetypes]

    for t in trades:
        f_df = None
        if features_loader_func is not None:
            try:
                f_df = features_loader_func(t["ticker"])
            except Exception:
                pass
        arch = extract_failure_profile_from_trade(t, features_df=f_df)
        if arch is not None:
            archetypes.append(asdict(arch))

    # Keep most recent
    if len(archetypes) > max_archetypes:
        archetypes = archetypes[-max_archetypes:]

    return archetypes


def compute_failure_similarity_penalty(
    profile: Mapping[str, float],
    failure_archetypes: Sequence[dict[str, Any]],
    similarity_threshold: float = FAILURE_SIMILARITY_THRESHOLD,
) -> tuple[float, float, str | None]:
    """Calculate RBF kernel similarity against remembered failure archetypes.

    Returns:
        tuple (prob_penalty, carver_multiplier, risk_flag_message):
            - prob_penalty: negative deduction for probability (e.g. -0.04)
            - carver_multiplier: position sizing dampener (e.g. 0.80)
            - risk_flag_message: human-readable warning badge or None
    """
    if not failure_archetypes:
        return 0.0, 1.0, None

    x_gap = float(profile.get("gap_to_atr", 0.50))
    x_hurst = float(profile.get("chan_hurst_20d", 0.50))
    x_elast = float(profile.get("micro_upward_elasticity_10d", 1.00))
    x_clv = float(profile.get("prior_clv", 0.50))

    # Standard scales for normalization
    s_gap = 0.80
    s_hurst = 0.15
    s_elast = 0.40
    s_clv = 0.30

    max_sim = 0.0
    matched_reason = "寄り天・損切り"

    for a in failure_archetypes:
        d2 = (
            ((x_gap - float(a.get("gap_to_atr", 0.50))) / s_gap) ** 2
            + ((x_hurst - float(a.get("chan_hurst_20d", 0.50))) / s_hurst) ** 2
            + ((x_elast - float(a.get("micro_upward_elasticity_10d", 1.00))) / s_elast) ** 2
            + ((x_clv - float(a.get("prior_clv", 0.50))) / s_clv) ** 2
        )
        sim = float(np.exp(-0.5 * d2))
        if sim > max_sim:
            max_sim = sim
            matched_reason = "損切り抵触" if a.get("reason") == "STOPPED_OUT" else "寄り天失速"

    if max_sim >= similarity_threshold:
        # Scale penalty with similarity
        penalty_factor = (max_sim - similarity_threshold) / (1.0 - similarity_threshold + 1e-6)
        prob_penalty = -float(np.clip(0.02 + 0.04 * penalty_factor, 0.02, 0.06))
        carver_mult = float(np.clip(0.85 - 0.15 * penalty_factor, 0.70, 0.85))
        sim_pct = int(round(max_sim * 100))
        risk_flag = f"🧠 過去失敗類似パターン検知（類似度 {sim_pct}% / {matched_reason}地雷警戒ブレーキ）"
        return prob_penalty, carver_mult, risk_flag

    return 0.0, 1.0, None


# ===========================================================================
# 4. Self-Reflection Summary Generator & Closed-Loop Integration
# ===========================================================================

def generate_self_reflection_summary(
    scorecard: Mapping[str, Any],
    updated_state: Mapping[str, Any],
) -> list[str]:
    """Generate 3-line actionable self-reflection summary from scorecard and updated state."""
    trades = scorecard.get("trades", [])
    weights = updated_state.get("regime_expert_weights", {})
    ticker_td = updated_state.get("ticker_td_state", {})
    sec_td = updated_state.get("sector_td_state", {})
    archetypes = updated_state.get("failure_archetypes", [])

    # Line 1: Hedge Waveform Adaptations
    # Find top boosted and top penalized waveforms
    wf_diffs = []
    for k in DEFAULT_WAVEFORM_KEYS:
        w = float(weights.get(k, 1.0))
        wf_diffs.append((k, w - 1.0))
    wf_diffs.sort(key=lambda x: x[1], reverse=True)
    best_wf, best_diff = wf_diffs[0]
    worst_wf, worst_diff = wf_diffs[-1]

    from stock_ai.models.intraday import WAVEFORM_PATTERNS
    best_name = WAVEFORM_PATTERNS.get(best_wf, {}).get("title", best_wf)
    worst_name = WAVEFORM_PATTERNS.get(worst_wf, {}).get("title", worst_wf)

    line1 = (
        f"①【波形・地合い適応 (Hedge)】: 直近好調の「{best_name}」を {best_diff*100:+.1f}% 重み引き上げ、"
        f"騙し警戒の「{worst_name}」を {worst_diff*100:+.1f}% 抑制。"
    )

    # Line 2: Ticker / Sector TD Credibility
    # Find most credible and most penalized tickers
    t_list = [(t, v.get("v", 0.0)) for t, v in ticker_td.items()]
    t_list.sort(key=lambda x: x[1], reverse=True)
    if t_list:
        top_t = t_list[0][0]
        top_v = t_list[0][1]
        line2 = (
            f"②【銘柄・セクター信用度 (TD学習)】: 期待通り素直に伸長した主力 `{top_t}` への信頼度を強化 (Credibility {top_v*100:+.1f}%)。"
        )
    else:
        line2 = "②【銘柄・セクター信用度 (TD学習)】: 往復20bps控除後の予測誤差を均等蓄積中（正常推移）。"

    # Line 3: Contextual Bandit Archetype
    stopped_count = sum(1 for t in trades if t.get("result_status") == "LOSS")
    arch_count = len(archetypes)
    line3 = (
        f"③【地雷コンテキスト (Bandit Memory)】: 本日確定の損切り・失速パターン {stopped_count}件を失敗プロファイル（全{arch_count}件）に記録、"
        f"明日類似形状銘柄のロットを自動減衰。"
    )

    return [line1, line2, line3]


def update_online_adaptation_from_scorecard(
    scorecard_data: dict[str, Any],
    state_file: Path | str | None = None,
    features_loader_func: Any | None = None,
    ticker_sector_map: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Execute complete closed-loop self-evolution step from post-market scorecard.

    Steps:
    1. Load current state (or initialize)
    2. Update Hedge expert weights from today's trades
    3. Update TD credibility tracker for tickers & sectors
    4. Store new failure archetype profiles
    5. Append reflection log
    6. Persist state to disk
    7. Return reflection summary
    """
    state = load_adaptation_state(state_file=state_file)
    trades = scorecard_data.get("trades", [])
    date_str = scorecard_data.get("date", datetime.now().strftime("%Y-%m-%d"))

    # 1. Update Hedge weights
    updated_weights = update_regime_expert_weights(
        current_weights=state.get("regime_expert_weights", {}),
        trades=trades,
        eta=HEDGE_ETA,
    )
    state["regime_expert_weights"] = updated_weights

    # 2. Update TD credibility
    new_ticker_td, new_sec_td = update_ticker_td_credibility(
        ticker_state=state.get("ticker_td_state", {}),
        sector_state=state.get("sector_td_state", {}),
        trades=trades,
        alpha_ticker=TD_LEARNING_RATE_TICKER,
        alpha_sector=TD_LEARNING_RATE_SECTOR,
        ticker_sector_map=ticker_sector_map,
    )
    state["ticker_td_state"] = new_ticker_td
    state["sector_td_state"] = new_sec_td

    # 3. Update Failure Archetypes
    new_archetypes = update_failure_archetypes(
        current_archetypes=state.get("failure_archetypes", []),
        trades=trades,
        features_loader_func=features_loader_func,
        max_archetypes=MAX_FAILURE_ARCHETYPES,
    )
    state["failure_archetypes"] = new_archetypes

    # 4. Generate Reflection Summary
    reflection = generate_self_reflection_summary(scorecard_data, state)

    # 5. Record History
    hist_entry = {
        "date": date_str,
        "trades_count": len(trades),
        "wins": scorecard_data.get("wins", 0),
        "losses": scorecard_data.get("losses", 0),
        "win_rate_pct": scorecard_data.get("win_rate_pct", 0.0),
        "total_virtual_pnl_yen": scorecard_data.get("total_virtual_pnl_yen", 0),
        "reflection_summary": reflection,
    }
    history = state.get("history_log", [])
    history.append(hist_entry)
    if len(history) > 30:
        history = history[-30:]
    state["history_log"] = history

    state["last_updated_date"] = date_str
    state["total_updates"] = int(state.get("total_updates", 0)) + 1

    save_adaptation_state(state, state_file=state_file)
    logger.info("Online adaptation updated successfully for date %s (total updates: %d)", date_str, state["total_updates"])

    return {
        "updated_state": state,
        "reflection_summary": reflection,
    }


# ===========================================================================
# 5. Rolling Online Adaptor for Zero-Leakage Backtesting
# ===========================================================================

class RollingOnlineAdaptor:
    """Manages sequential online adaptation during backtests with strict zero-lookahead causality."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        self.state = initial_state or initialize_fresh_state()
        self.last_settled_date: str | None = None

    def get_current_state(self) -> dict[str, Any]:
        """Return state snapshot strictly prior to current trade execution."""
        return self.state

    def update_from_settled_trades(
        self,
        date_str: str,
        trades: Sequence[dict[str, Any]],
        ticker_sector_map: Mapping[str, str] | None = None,
    ) -> None:
        """Update state using strictly settled trades from previous day."""
        if not trades:
            return

        mock_scorecard = {
            "date": date_str,
            "trades": list(trades),
            "wins": sum(1 for t in trades if t.get("result_status") == "WIN"),
            "losses": sum(1 for t in trades if t.get("result_status") == "LOSS"),
            "win_rate_pct": float(np.mean([t.get("result_status") == "WIN" for t in trades]) * 100.0) if trades else 0.0,
            "total_virtual_pnl_yen": sum(t.get("virtual_pnl_100_shares", 0) for t in trades),
        }

        # Step 1: Hedge weights
        self.state["regime_expert_weights"] = update_regime_expert_weights(
            current_weights=self.state["regime_expert_weights"],
            trades=trades,
        )

        # Step 2: TD credibility
        self.state["ticker_td_state"], self.state["sector_td_state"] = update_ticker_td_credibility(
            ticker_state=self.state["ticker_td_state"],
            sector_state=self.state["sector_td_state"],
            trades=trades,
            ticker_sector_map=ticker_sector_map,
        )

        # Step 3: Failure Archetypes
        self.state["failure_archetypes"] = update_failure_archetypes(
            current_archetypes=self.state["failure_archetypes"],
            trades=trades,
        )

        self.last_settled_date = date_str
        self.state["total_updates"] = int(self.state.get("total_updates", 0)) + 1
