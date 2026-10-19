"""Interactive Discord Bot for on-demand stock analysis and multi-agent dialogue."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from stock_ai.cloud.discord_provisioner import DiscordProvisioner
from stock_ai.cloud.notifications import (
    get_configured_channel_webhooks,
    get_current_pipeline_summary,
    normalize_webhook_url,
)
from stock_ai.config import PROJECT_ROOT
from stock_ai.data.universe import load_turnaround_universe, load_universe
from stock_ai.events.earnings_guard import check_earnings_risk

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


# ---------------------------------------------------------------------------
# 1. Ticker & Japanese Company Name Resolver
# ---------------------------------------------------------------------------

PRIMARY_ALIASES: dict[str, tuple[str, str]] = {
    # Automotive
    "トヨタ": ("7203.T", "トヨタ自動車"),
    "トヨタ自動車": ("7203.T", "トヨタ自動車"),
    "ホンダ": ("7267.T", "本田技研工業"),
    "本田技研": ("7267.T", "本田技研工業"),
    "日産": ("7201.T", "日産自動車"),
    "日産自動車": ("7201.T", "日産自動車"),
    "ニッサン": ("7201.T", "日産自動車"),
    "デンソー": ("6902.T", "デンソー"),
    "スバル": ("7270.T", "SUBARU"),
    "SUBARU": ("7270.T", "SUBARU"),
    # Trading Companies
    "三菱商事": ("8058.T", "三菱商事"),
    "三井物産": ("8031.T", "三井物産"),
    "伊藤忠": ("8001.T", "伊藤忠商事"),
    "伊藤忠商事": ("8001.T", "伊藤忠商事"),
    "丸紅": ("8002.T", "丸紅"),
    "住友商事": ("8053.T", "住友商事"),
    "住商": ("8053.T", "住友商事"),
    # Financials
    "三菱UFJ": ("8306.T", "三菱UFJ FG"),
    "MUFG": ("8306.T", "三菱UFJ FG"),
    "三菱UFJ FG": ("8306.T", "三菱UFJ FG"),
    "三井住友": ("8316.T", "三井住友FG"),
    "SMBC": ("8316.T", "三井住友FG"),
    "三井住友FG": ("8316.T", "三井住友FG"),
    "みずほ": ("8411.T", "みずほFG"),
    "みずほFG": ("8411.T", "みずほFG"),
    "ゆうちょ": ("7182.T", "ゆうちょ銀行"),
    "ゆうちょ銀行": ("7182.T", "ゆうちょ銀行"),
    "オリックス": ("8591.T", "オリックス"),
    "東京海上": ("8766.T", "東京海上HD"),
    "東京海上HD": ("8766.T", "東京海上HD"),
    # Heavy Industries, Machinery, Defense
    "三菱重工": ("7011.T", "三菱重工業"),
    "三菱重工業": ("7011.T", "三菱重工業"),
    "MHI": ("7011.T", "三菱重工業"),
    "川崎重工": ("7012.T", "川崎重工業"),
    "川重": ("7012.T", "川崎重工業"),
    "IHI": ("7013.T", "IHI"),
    "コマツ": ("6301.T", "小松製作所"),
    "小松製作所": ("6301.T", "小松製作所"),
    "ダイキン": ("6367.T", "ダイキン工業"),
    # Semiconductors & Tech
    "東エレク": ("8035.T", "東京エレクトロン"),
    "東京エレクトロン": ("8035.T", "東京エレクトロン"),
    "TEL": ("8035.T", "東京エレクトロン"),
    "アドバンテスト": ("6857.T", "アドバンテスト"),
    "アドバン": ("6857.T", "アドバンテスト"),
    "ルネサス": ("6723.T", "ルネサスエレクトロニクス"),
    "ディスコ": ("6146.T", "ディスコ"),
    "レーザーテック": ("6920.T", "レーザーテック"),
    "レーザー": ("6920.T", "レーザーテック"),
    # Electronics
    "ソニー": ("6758.T", "ソニーグループ"),
    "ソニーグループ": ("6758.T", "ソニーグループ"),
    "SONY": ("6758.T", "ソニーグループ"),
    "日立": ("6501.T", "日立製作所"),
    "日立製作所": ("6501.T", "日立製作所"),
    "キーエンス": ("6861.T", "キーエンス"),
    "KEYENCE": ("6861.T", "キーエンス"),
    "パナソニック": ("6752.T", "パナソニックHD"),
    "キヤノン": ("7751.T", "キヤノン"),
    "村田製作所": ("6981.T", "村田製作所"),
    # Telecom & IT
    "ソフトバンクグループ": ("9984.T", "ソフトバンクグループ"),
    "ソフトバンクG": ("9984.T", "ソフトバンクグループ"),
    "SBG": ("9984.T", "ソフトバンクグループ"),
    "ソフトバンク": ("9434.T", "ソフトバンク"),
    "ソフバン": ("9434.T", "ソフトバンク"),
    "KDDI": ("9433.T", "KDDI"),
    "au": ("9433.T", "KDDI"),
    "NTT": ("9432.T", "日本電信電話"),
    "日本電信電話": ("9432.T", "日本電信電話"),
    # Retail & Entertainment
    "任天堂": ("7974.T", "任天堂"),
    "ニンテンドー": ("7974.T", "任天堂"),
    "ファストリ": ("9983.T", "ファーストリテイリング"),
    "ファーストリテイリング": ("9983.T", "ファーストリテイリング"),
    "ユニクロ": ("9983.T", "ファーストリテイリング"),
    "オリエンタルランド": ("4661.T", "オリエンタルランド"),
    "OLC": ("4661.T", "オリエンタルランド"),
    "ディズニー": ("4661.T", "オリエンタルランド"),
    "イオン": ("8267.T", "イオン"),
    # Chemicals & Pharma
    "信越化": ("4063.T", "信越化学工業"),
    "信越化学": ("4063.T", "信越化学工業"),
    "三菱ケミカル": ("4188.T", "三菱ケミカルグループ"),
    "武田": ("4502.T", "武田薬品工業"),
    "武田薬品": ("4502.T", "武田薬品工業"),
    "第一三共": ("4568.T", "第一三共"),
    "アステラス": ("4503.T", "アステラス製薬"),
    # Metals, Shipping, Real Estate, Energy
    "日本製鉄": ("5401.T", "日本製鉄"),
    "日鉄": ("5401.T", "日本製鉄"),
    "日本郵船": ("9101.T", "日本郵船"),
    "商船三井": ("9104.T", "商船三井"),
    "川崎汽船": ("9107.T", "川崎汽船"),
    "三井不動産": ("8801.T", "三井不動産"),
    "三菱地所": ("8802.T", "三菱地所"),
    "住友不動産": ("8830.T", "住友不動産"),
    "INPEX": ("1605.T", "INPEX"),
    "インペックス": ("1605.T", "INPEX"),
}


def _get_universe_name_map() -> dict[str, str]:
    """Build a code-to-name lookup dictionary from loaded universes."""
    name_map: dict[str, str] = {}
    try:
        for t in load_turnaround_universe():
            if t.code and t.name:
                name_map[t.code] = t.name
    except Exception:
        pass
    try:
        for t in load_universe():
            if t.code and t.name:
                name_map[t.code] = t.name
    except Exception:
        pass
    return name_map


def resolve_ticker(query: str) -> tuple[str, str] | None:
    """Resolve user query string into a canonical ticker (e.g. '7203.T') and company name.

    Examples:
        '7203' -> ('7203.T', 'トヨタ自動車')
        'トヨタ' -> ('7203.T', 'トヨタ自動車')
        '任天堂を分析して' -> ('7974.T', '任天堂')
        '6758.Tの予想' -> ('6758.T', 'ソニーグループ')
    """
    clean_text = query.strip()
    if not clean_text:
        return None

    # 1. Normalize full-width digits to half-width
    clean_text = clean_text.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    name_map = _get_universe_name_map()

    # 2. Check 4-digit ticker code pattern
    match = re.search(r"(?<!\d)(\d{4})(?:\.T)?(?!\d)", clean_text, re.IGNORECASE)
    if match:
        code_num = match.group(1)
        ticker = f"{code_num}.T"
        name = name_map.get(ticker)
        if not name:
            for k, (t_code, t_name) in PRIMARY_ALIASES.items():
                if t_code == ticker:
                    name = t_name
                    break
        return ticker, name or ticker

    # 3. Check alias map with longest match first
    sorted_aliases = sorted(PRIMARY_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias.lower() in clean_text.lower():
            ticker, name = PRIMARY_ALIASES[alias]
            return ticker, name

    # 4. Check universe names directly
    for code, name in name_map.items():
        if name and name.lower() in clean_text.lower():
            return code, name

    return None


def extract_all_tickers(query: str) -> list[tuple[str, str]]:
    """Extract all distinct ticker symbols and company names mentioned in the query.

    Preserves order of appearance in text and deduplicates.

    Examples:
        'トヨタと日産どっちがいい？' -> [('7203.T', 'トヨタ自動車'), ('7201.T', '日産自動車')]
        'ソニーと任天堂を比較して' -> [('6758.T', 'ソニーグループ'), ('7974.T', '任天堂')]
        '8058と8031' -> [('8058.T', '三菱商事'), ('8031.T', '三井物産')]
    """
    clean_text = query.strip().translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    name_map = _get_universe_name_map()
    results: list[tuple[str, str]] = []
    seen_tickers: set[str] = set()

    # 1. 4-digit code matches (preserving character position)
    for m in re.finditer(r"(?<!\d)(\d{4})(?:\.T)?(?!\d)", clean_text, re.IGNORECASE):
        code_num = m.group(1)
        ticker = f"{code_num}.T"
        if ticker not in seen_tickers:
            name = name_map.get(ticker)
            if not name:
                for t_code, t_name in PRIMARY_ALIASES.values():
                    if t_code == ticker:
                        name = t_name
                        break
            results.append((ticker, name or ticker))
            seen_tickers.add(ticker)

    # 2. Check aliases by longest match first
    sorted_aliases = sorted(PRIMARY_ALIASES.keys(), key=len, reverse=True)
    for alias in sorted_aliases:
        if alias.lower() in clean_text.lower():
            ticker, name = PRIMARY_ALIASES[alias]
            if ticker not in seen_tickers:
                results.append((ticker, name))
                seen_tickers.add(ticker)

    # 3. Check universe names
    for code, name in name_map.items():
        if name and name.lower() in clean_text.lower():
            if code not in seen_tickers:
                results.append((code, name))
                seen_tickers.add(code)

    return results


# ---------------------------------------------------------------------------
# 2. On-demand Single Ticker Analysis Engine
# ---------------------------------------------------------------------------


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Calculate RSI for a price series."""
    if len(series) < period + 1:
        return 50.0
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(period, min_periods=period // 2).mean()
    roll_down = down.rolling(period, min_periods=period // 2).mean()
    rs = roll_up / (roll_down + 1e-8)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50.0


def analyze_single_ticker(query: str) -> dict[str, Any] | None:
    """Perform real-time multi-angle analysis on a requested stock.

    Uses snapshot cache if available, falling back to yfinance live data.
    """
    resolved = resolve_ticker(query)
    if not resolved:
        return None

    ticker, name = resolved
    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1. Try to load cached analysis from autonomous snapshots
    cached_overnight: dict[str, Any] | None = None
    cached_turnaround: dict[str, Any] | None = None

    snapshot_path = PROJECT_ROOT / "reports/autonomous/latest_overnight_scanner_snapshot.json"
    if snapshot_path.exists():
        try:
            s_data = json.loads(snapshot_path.read_text(encoding="utf-8"))
            for r in s_data.get("rankings", []):
                if r.get("ticker") == ticker:
                    cached_overnight = r
                    break
        except Exception:
            pass

    turnaround_path = PROJECT_ROOT / "reports/autonomous/value_turnaround_scanner_results.json"
    if turnaround_path.exists():
        try:
            t_data = json.loads(turnaround_path.read_text(encoding="utf-8"))
            for r in t_data.get("rankings", []):
                if r.get("ticker") == ticker:
                    cached_turnaround = r
                    break
        except Exception:
            pass

    # 2. Retrieve live/historical market data (fallback or complement)
    close_val: float | None = None
    prev_close_val: float | None = None
    pct_change: float = 0.0
    high_52w: float | None = None
    drawdown_52w: float = 0.0
    rsi_14: float = 50.0
    pbr_val: float | None = None
    sector_str: str = ""

    if cached_overnight and cached_overnight.get("close"):
        close_val = float(cached_overnight["close"])
        sector_str = cached_overnight.get("sector", "")
    if cached_turnaround:
        sector_str = sector_str or cached_turnaround.get("sector", "")
        pbr_val = cached_turnaround.get("pbr")

    # Fetch live via yfinance if needed or to get fresh day change / RSI
    try:
        import yfinance as yf

        t_obj = yf.Ticker(ticker)
        hist = t_obj.history(period="1y")
        if not hist.empty and len(hist) >= 2:
            close_val = float(hist["Close"].iloc[-1])
            prev_close_val = float(hist["Close"].iloc[-2])
            pct_change = (close_val - prev_close_val) / prev_close_val
            high_52w = float(hist["High"].max())
            drawdown_52w = (close_val - high_52w) / high_52w if high_52w > 0 else 0.0
            rsi_14 = _compute_rsi(hist["Close"], period=14)

        # Fundamentals fallback
        if pbr_val is None:
            fast_info = getattr(t_obj, "fast_info", None)
            if fast_info and hasattr(fast_info, "price_to_book"):
                pbr_val = getattr(fast_info, "price_to_book", None)
    except Exception as exc:
        logger.debug("Failed fetching live yfinance data for %s: %s", ticker, exc)

    if close_val is None:
        close_val = 1000.0

    # 3. Overnight Sniper Analytics
    if cached_overnight:
        p_up = float(cached_overnight.get("probability_up", 0.50))
        q_band = cached_overnight.get("q_band") or "Q3"
        q_label = cached_overnight.get("label") or "中立帯"
        earnings_risk = bool(cached_overnight.get("earnings_risk", False))
        earnings_date = cached_overnight.get("earnings_date")
    else:
        # Technical Overnight probability estimation
        # Baseline ~ 50%, boosted by oversold RSI rebound or strong momentum
        score_shift = 0.0
        if rsi_14 < 35.0:
            score_shift += 0.04  # oversold bounce
        elif rsi_14 > 70.0:
            score_shift -= 0.03  # overbought stall
        if pct_change > 0.01:
            score_shift += 0.02
        elif pct_change < -0.02:
            score_shift -= 0.02

        p_up = float(np.clip(0.505 + score_shift, 0.42, 0.62))
        if p_up >= 0.55:
            q_band = "Q5"
            q_label = "強気ギャップアップ帯"
        elif p_up >= 0.52:
            q_band = "Q4"
            q_label = "強気先行帯"
        elif p_up >= 0.49:
            q_band = "Q3"
            q_label = "中立帯"
        elif p_up >= 0.46:
            q_band = "Q2"
            q_label = "軟調警戒帯"
        else:
            q_band = "Q1"
            q_label = "弱気警戒帯"

        earnings_risk, earnings_date = check_earnings_risk(ticker)

    # 4. Value Hunter Analytics
    if cached_turnaround:
        bargain_score = float(cached_turnaround.get("bargain_score", 50.0))
        turnaround_status = cached_turnaround.get("status", "⚖️ 中立調整圏")
        if not pbr_val and cached_turnaround.get("pbr"):
            pbr_val = float(cached_turnaround["pbr"])
        drawdown_52w = float(cached_turnaround.get("drawdown_252d", drawdown_52w))
    else:
        # Estimate Turnaround Score
        # Bargain score depends on 52w drawdown, RSI, and PBR
        dd_factor = min(abs(drawdown_52w), 0.50) / 0.50 * 50.0
        pbr_factor = 25.0 if (pbr_val and pbr_val < 0.8) else (15.0 if (pbr_val and pbr_val < 1.2) else 10.0)
        rsi_factor = max(0.0, (50.0 - rsi_14) / 50.0 * 25.0)
        bargain_score = round(min(100.0, dd_factor + pbr_factor + rsi_factor), 1)

        if bargain_score >= 70.0:
            turnaround_status = "💎 超割安・底打ち反転圏"
        elif bargain_score >= 55.0:
            turnaround_status = "🚀 始動サイン"
        elif bargain_score >= 40.0:
            turnaround_status = "👀 調整監視圏"
        else:
            turnaround_status = "📈 高値巡航圏"

    # 5. Formulate AI Analyst Dialogue Statements
    # --- Persona 2: Sniper ---
    e_flag_str = "⚠️ 決算発表直前警戒（決算跨ぎリスクあり）" if earnings_risk else "✅ 通常（決算地雷なし）"
    if p_up >= 0.52:
        sniper_comment = (
            f"明朝寄り付き上昇確率 **{p_up:.1%}**（{q_band}）。買い優位のシグナルが点灯しています！\n"
            f"決算チェック: **{e_flag_str}**。"
        )
    else:
        sniper_comment = (
            f"明朝寄り付き上昇確率 **{p_up:.1%}**（{q_band}）。寄り付きギャップの上値追いは慎重に。\n"
            f"決算チェック: **{e_flag_str}**。"
        )

    # --- Persona 3: Value Hunter ---
    pbr_disp = f"{pbr_val:.2f}倍" if pbr_val is not None else "N/A"
    if bargain_score >= 70.0:
        hunter_comment = (
            f"割安スコア **{bargain_score:.1f}点**（52週高値比: **{drawdown_52w:+.1%}**, PBR: **{pbr_disp}**）。\n"
            f"判定: **{turnaround_status}**！下値余地が限定的で、中長期の反転リバウンドを狙う絶好の仕込み水準です💎"
        )
    elif bargain_score >= 50.0:
        hunter_comment = (
            f"割安スコア **{bargain_score:.1f}点**（52週高値比: **{drawdown_52w:+.1%}**, PBR: **{pbr_disp}**）。\n"
            f"判定: **{turnaround_status}**。押し目形成中で反転サインの確認を推奨します。"
        )
    else:
        hunter_comment = (
            f"割安スコア **{bargain_score:.1f}点**（52週高値比: **{drawdown_52w:+.1%}**, PBR: **{pbr_disp}**）。\n"
            f"判定: **{turnaround_status}**。過度な割安感はなく、高値圏のモメンタム追随モードです。"
        )

    # --- Persona 1: Chief FM Overall Conclusion ---
    if earnings_risk:
        chief_conclusion = (
            "🚨 **【チーフ判断: 見送り推奨】**\n"
            "直近に決算発表が控えており、ボラティリティ急変によるギャンブル跨ぎは厳禁です。"
            "発表後の市場反応を確認してからポジションを構築しましょう。"
        )
    elif p_up >= 0.52 and bargain_score >= 65.0:
        chief_conclusion = (
            "🎯 **【チーフ判断: 強気ダブルシグナル点灯】**\n"
            "短期Overnightの上昇期待と中長期の割安反転余地が合致した好機です。"
            "資金管理を徹底した上で、積極的なエントリーを検討できます！"
        )
    elif p_up >= 0.52:
        chief_conclusion = (
            "⚡ **【チーフ判断: 短期デイトレ・スイング狙い】**\n"
            "短期的な寄り付きギャップアップ確率に優位性があります。"
            "中長期でのホールドではなく、短期利確を徹底する作戦が有効です。"
        )
    elif bargain_score >= 65.0:
        chief_conclusion = (
            "💎 **【チーフ判断: 中長期バリュー分散仕込み】**\n"
            "明朝の寄り付きは無理に追わず、安値をコツコツ拾う中長期視点での種まきに適した水準です。"
        )
    else:
        chief_conclusion = (
            "🛡️ **【チーフ判断: 様子見・キャッシュ待機】**\n"
            "明確なクオンツ優位性が出るまで無理なエントリーは控え、次の好機まで監視を継続します。"
        )

    return {
        "ticker": ticker,
        "name": name,
        "sector": sector_str or "東証プライム",
        "date": date_str,
        "close": close_val,
        "pct_change": pct_change,
        "high_52w": high_52w,
        "drawdown_52w": drawdown_52w,
        "rsi_14": rsi_14,
        "pbr": pbr_val,
        "p_up": p_up,
        "q_band": q_band,
        "q_label": q_label,
        "earnings_risk": earnings_risk,
        "earnings_date": earnings_date,
        "bargain_score": bargain_score,
        "turnaround_status": turnaround_status,
        "chief_conclusion": chief_conclusion,
        "sniper_comment": sniper_comment,
        "hunter_comment": hunter_comment,
    }


# ---------------------------------------------------------------------------
# 3. Multi-Persona Dialogue Payload Builders & Consultation Engines
# ---------------------------------------------------------------------------


def _get_macro_context() -> dict[str, Any]:
    """Retrieve macro index metrics (Nikkei 225, USD/JPY) and paper portfolio status."""
    macro: dict[str, Any] = {
        "nikkei_close": 39500.0,
        "nikkei_pct": 0.002,
        "usdjpy": 154.50,
        "usdjpy_pct": 0.001,
        "portfolio_status": "キャッシュ温存・好機待機中",
        "equity": 1000000.0,
        "win_rate": 0.65,
    }
    try:
        summary = get_current_pipeline_summary()
        pm = summary.get("paper_metrics", {})
        if pm:
            macro["portfolio_status"] = pm.get("position_label", "キャッシュ温存・待機中")
            macro["equity"] = pm.get("equity", 1000000.0)
            macro["win_rate"] = pm.get("win_rate", 0.65)
    except Exception:
        pass

    try:
        import yfinance as yf

        data = yf.download(["^N225", "USDJPY=X"], period="5d", progress=False)
        if not data.empty and "Close" in data:
            c = data["Close"]
            if "^N225" in c:
                s_nk = c["^N225"].dropna()
                if len(s_nk) >= 2:
                    macro["nikkei_close"] = float(s_nk.iloc[-1])
                    macro["nikkei_pct"] = float((s_nk.iloc[-1] / s_nk.iloc[-2]) - 1.0)
            if "USDJPY=X" in c:
                s_fx = c["USDJPY=X"].dropna()
                if len(s_fx) >= 2:
                    macro["usdjpy"] = float(s_fx.iloc[-1])
                    macro["usdjpy_pct"] = float((s_fx.iloc[-1] / s_fx.iloc[-2]) - 1.0)
    except Exception:
        pass

    return macro


def _try_llm_dialogue(system_prompt: str, user_prompt: str) -> str | None:
    """Optional LLM dialogue synthesis if GEMINI_API_KEY or OPENAI_API_KEY is available."""
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if gemini_key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
            payload = {
                "contents": [
                    {"role": "user", "parts": [{"text": f"{system_prompt}\n\nユーザーの質問: {user_prompt}"}]}
                ],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 600},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                candidates = res_data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return str(parts[0]["text"])
        except Exception as exc:
            logger.debug("Gemini LLM optional call skipped or failed: %s", exc)

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        try:
            url = "https://api.openai.com/v1/chat/completions"
            payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 600,
                "temperature": 0.7,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {openai_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                res_data = json.loads(resp.read().decode("utf-8"))
                choices = res_data.get("choices", [])
                if choices:
                    return str(choices[0].get("message", {}).get("content", ""))
        except Exception as exc:
            logger.debug("OpenAI LLM optional call skipped or failed: %s", exc)

    return None


def build_single_ticker_dialogue_payloads(
    analysis: dict[str, Any],
    query: str = "",
) -> list[dict[str, Any]]:
    """Build a rich 3-persona conversational response for a single stock inquiry."""
    ticker = analysis["ticker"]
    name = analysis["name"]
    sector = analysis["sector"]
    close = analysis["close"]
    pct = analysis["pct_change"]
    p_up = analysis["p_up"]
    q_band = analysis["q_band"]
    score = analysis["bargain_score"]
    status = analysis["turnaround_status"]
    rsi = analysis["rsi_14"]
    pbr_s = f"{analysis['pbr']:.2f}倍" if analysis.get("pbr") is not None else "-"
    date_str = analysis.get("date", datetime.now().strftime("%Y-%m-%d"))

    # Intent-aware intro prefix
    intent_intro = ""
    if any(k in query for k in ["買い時", "買うべき", "買ってもいい", "仕込み"]):
        if p_up >= 0.52 or score >= 65.0:
            intent_intro = "💡 **【チーフの買い時判定: エントリー検討圏】**\nクオンツ指標上で買い優位のシグナルが検出されています。\n\n"
        else:
            intent_intro = "💡 **【チーフの買い時判定: 様子見・押し目待ち推奨】**\n無理に飛びつかず、次のシグナル合致を待つのが賢明です。\n\n"
    elif any(k in query for k in ["なぜ安い", "どうして", "理由", "暴落"]):
        intent_intro = f"💡 **【チーフの調整要因解説】**\n52週高値から {analysis['drawdown_52w']:+.1%} 調整中ですが、指標面での底打ちシグナルを精査しました。\n\n"

    # 1. Chief Fund Manager Message
    chief_text = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「`{ticker}` **{name}** ({sector}) のオンデマンド精密分析を完了しました。\n\n"
        f"{intent_intro}"
        f"現在株価: **{close:,.1f} 円** ({pct:+.2%}) | RSI(14): **{rsi:.1f}**\n\n"
        f"{analysis['chief_conclusion']}\n\n"
        f"スナイパー、ハンター、各専門視点の評価詳細を提示してください。」"
    )
    embed_chief = {
        "title": f"📊 【銘柄分析】{name} ({ticker}) - {date_str}",
        "description": f"セクター: {sector} | 終値: **{close:,.1f}円** ({pct:+.2%})",
        "color": 3066993,  # Emerald
        "fields": [
            {"name": "⚡ 短期Overnight確率", "value": f"**{p_up:.1%}** ({q_band})", "inline": True},
            {"name": "💎 割安反転スコア", "value": f"**{score:.1f}点** ({status})", "inline": True},
            {"name": "📈 RSI(14) / PBR", "value": f"**{rsi:.1f}** / **{pbr_s}**", "inline": True},
            {"name": "⚖️ チーフ総合結論", "value": analysis["chief_conclusion"], "inline": False},
        ],
        "footer": {"text": "Stock-AI Real-Time Interactive Analyst Lounge"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_chief = {
        "username": "🎩 チーフ・ファンドマネージャー AI",
        "content": chief_text,
        "text": chief_text,
        "embeds": [embed_chief],
    }

    # 2. Short-Term Sniper Message
    e_flag = "⚠️ 決算発表警戒" if analysis["earnings_risk"] else "✅ 通常"
    e_date_info = f" (発表予定: {analysis['earnings_date']})" if analysis.get("earnings_date") else ""
    sniper_text = (
        f"⚡ **短期スナイパー AI (Overnight担当)**\n"
        f"「ボス、短期テクニカル＆翌朝寄り付きスキャン結果です🎯\n\n"
        f"・翌朝寄り付き上昇確率: **{p_up:.1%}** ({q_band})\n"
        f"・決算発表リスク判定: **{e_flag}**{e_date_info}\n\n"
        f"{analysis['sniper_comment']}」"
    )
    embed_sniper = {
        "title": f"⚡ 短期Overnight判定: {name} ({ticker})",
        "description": "翌朝寄り付き（Close-to-Open）の上昇ギャップ狙い目度",
        "color": 15844367,  # Gold
        "fields": [
            {"name": "確率バンド", "value": f"**{p_up:.1%}** ({analysis['q_label']})", "inline": True},
            {"name": "決算地雷リスク", "value": f"**{e_flag}**{e_date_info}", "inline": True},
            {"name": "スナイパー見立て", "value": analysis["sniper_comment"], "inline": False},
        ],
        "footer": {"text": "Stock-AI Short-Term Sniper Engine"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_sniper = {
        "username": "⚡ 短期スナイパー AI (Overnight担当)",
        "content": sniper_text,
        "text": sniper_text,
        "embeds": [embed_sniper],
    }

    # 3. Value Hunter Message
    hunter_text = (
        f"💎 **バリューハンター AI (中長期・激安株担当)**\n"
        f"「中長期バリュー＆リバウンド視点からの精査結果です💎\n\n"
        f"・割安回復スコア: **{score:.1f}点** (52週高値比: **{analysis['drawdown_52w']:+.1%}**)\n"
        f"・バリュエーション: PBR **{pbr_s}** | 判定: **{status}**\n\n"
        f"{analysis['hunter_comment']}」"
    )
    embed_hunter = {
        "title": f"💎 Value & Turnaround判定: {name} ({ticker})",
        "description": "東証プライム基準における底打ち・下値反転ポテンシャル",
        "color": 10181046,  # Purple
        "fields": [
            {"name": "割安スコア", "value": f"**{score:.1f} / 100点**", "inline": True},
            {"name": "52週高値比下落率", "value": f"**{analysis['drawdown_52w']:+.1%}**", "inline": True},
            {"name": "バリューハンター見立て", "value": analysis["hunter_comment"], "inline": False},
        ],
        "footer": {"text": "Stock-AI Value Hunter Engine"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_hunter = {
        "username": "💎 バリューハンター AI (中長期・激安株担当)",
        "content": hunter_text,
        "text": hunter_text,
        "embeds": [embed_hunter],
    }

    return [p_chief, p_sniper, p_hunter]


def build_comparison_dialogue_payloads(
    tickers: list[tuple[str, str]],
    query: str = "",
) -> list[dict[str, Any]]:
    """Build a rich 3-persona comparative dialogue for 2-3 stocks."""
    analyses: list[dict[str, Any]] = []
    for ticker, _ in tickers[:3]:
        an = analyze_single_ticker(ticker)
        if an:
            analyses.append(an)

    if not analyses:
        return build_ranking_payloads()

    if len(analyses) == 1:
        return build_single_ticker_dialogue_payloads(analyses[0], query=query)

    best_sniper = max(analyses, key=lambda a: a["p_up"])
    best_hunter = max(analyses, key=lambda a: a["bargain_score"])

    t_names = " vs ".join([f"`{a['name']}`" for a in analyses])
    date_str = analyses[0].get("date", datetime.now().strftime("%Y-%m-%d"))

    chief_decision = (
        f"短期のモメンタムと翌朝ギャップ期待なら **{best_sniper['name']}**（上昇確率 {best_sniper['p_up']:.1%}）がリード。\n"
        f"中長期の割安安全域と下値抵抗力なら **{best_hunter['name']}**（割安スコア {best_hunter['bargain_score']:.1f}点）に妙味があります。\n"
    )
    if best_sniper["ticker"] == best_hunter["ticker"]:
        chief_decision += f"総合判定として、攻守両面で優位性が揃っている **{best_sniper['name']}** への優先エントリーを推奨します！"
    else:
        chief_decision += (
            f"総合判定として、デイトレ・スイング狙いなら **{best_sniper['name']}**、"
            f"NISA・じっくり中長期仕込みなら **{best_hunter['name']}** をポートフォリオに選定すべきです。"
        )

    chief_text = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「{t_names} の直接比較リクエストですね。クオンツデータを対比精査しました。\n\n"
        f"【チーフ総合ジャッジ】:\n{chief_decision}\n\n"
        f"スナイパー、ハンター、各サイドの勝敗判定と根拠を述べてください。」"
    )

    fields: list[dict[str, Any]] = []
    for a in analyses:
        e_info = "⚠️ 決算直前" if a["earnings_risk"] else "✅ 通常"
        pbr_s = f"{a['pbr']:.2f}倍" if a.get("pbr") is not None else "-"
        fields.append({
            "name": f"📊 {a['name']} ({a['ticker']})",
            "value": (
                f"終値: **{a['close']:,.1f}円** ({a['pct_change']:+.2%})\n"
                f"⚡ Overnight確率: **{a['p_up']:.1%}** ({a['q_band']})\n"
                f"💎 割安スコア: **{a['bargain_score']:.1f}点** (高値比 {a['drawdown_52w']:+.1%})\n"
                f"PBR: **{pbr_s}** | RSI(14): **{a['rsi_14']:.1f}**\n"
                f"決算リスク: **{e_info}**"
            ),
            "inline": True,
        })

    fields.append({
        "name": "⚖️ チーフ最終ジャッジ",
        "value": chief_decision,
        "inline": False,
    })

    embed_chief = {
        "title": f"⚔️ 【銘柄直接比較】{t_names} ({date_str})",
        "description": "3人のクオンツAIによる多角対決分析",
        "color": 3066993,
        "fields": fields,
        "footer": {"text": "Stock-AI Comparative Intelligence Lounge"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_chief = {
        "username": "🎩 チーフ・ファンドマネージャー AI",
        "content": chief_text,
        "text": chief_text,
        "embeds": [embed_chief],
    }

    sniper_lines = []
    for a in analyses:
        sniper_lines.append(f"・{a['name']}: 翌朝上昇確率 **{a['p_up']:.1%}** ({a['q_band']})")
    sniper_text = (
        "⚡ **短期スナイパー AI (Overnight担当)**\n"
        "「短期勝負なら白黒ハッキリつけようぜ！🎯\n\n"
        + "\n".join(sniper_lines) + "\n\n"
        f"俺の判定は **【{best_sniper['name']}】** の圧勝だ！\n"
        f"明朝の寄り付きギャップアップを狙うなら、確率 {best_sniper['p_up']:.1%} で優位に立っているぜ！」"
    )
    p_sniper = {
        "username": "⚡ 短期スナイパー AI (Overnight担当)",
        "content": sniper_text,
        "text": sniper_text,
    }

    hunter_lines = []
    for a in analyses:
        pbr_s = f"{a['pbr']:.2f}倍" if a.get("pbr") is not None else "-"
        hunter_lines.append(f"・{a['name']}: 割安度 **{a['bargain_score']:.1f}点** (下落率 {a['drawdown_52w']:+.1%}, PBR {pbr_s})")
    hunter_text = (
        "💎 **バリューハンター AI (中長期・激安株担当)**\n"
        "「慌てて短期に飛びつくのは危険です。中長期の安全域を見ましょう💎\n\n"
        + "\n".join(hunter_lines) + "\n\n"
        f"私の判定は **【{best_hunter['name']}】** の勝ちです。\n"
        f"下値リスクが限定的で、売られすぎからの底打ちリバウンド余地が最も大きいと評価します。」"
    )
    p_hunter = {
        "username": "💎 バリューハンター AI (中長期・激安株担当)",
        "content": hunter_text,
        "text": hunter_text,
    }

    return [p_chief, p_sniper, p_hunter]


def build_recommendation_dialogue_payloads(query: str = "") -> list[dict[str, Any]]:
    """Build intelligent recommendations based on user budget and investment horizon."""
    summary = get_current_pipeline_summary()
    top_o = summary.get("top_overnight", [])
    top_t = summary.get("top_turnaround", [])
    date_str = summary.get("date", datetime.now().strftime("%Y-%m-%d"))

    budget: float | None = None
    m_man = re.search(r"(\d+(?:\.\d+)?)\s*万(?:円)?", query)
    if m_man:
        budget = float(m_man.group(1)) * 10000.0
    else:
        m_yen = re.search(r"(\d{5,8})\s*円", query)
        if m_yen:
            budget = float(m_yen.group(1))

    budget_label = f"予算 {budget / 10000:,.0f}万円" if budget else "おすすめ・資金配分相談"

    def get_candidate_cost(item: dict[str, Any]) -> tuple[float, float]:
        close = float(item.get("close") or 0.0)
        if close <= 0:
            ticker = item.get("ticker", "")
            an = analyze_single_ticker(ticker)
            if an and an.get("close"):
                close = float(an["close"])
        if close <= 0:
            close = 2500.0
        return close, close * 100.0

    picked_sniper: dict[str, Any] | None = None
    sniper_cost = 0.0
    for cand in top_o:
        _, cost = get_candidate_cost(cand)
        if budget is None or cost <= budget * 1.05:
            picked_sniper = cand
            sniper_cost = cost
            break
    if not picked_sniper and top_o:
        picked_sniper = top_o[0]
        _, sniper_cost = get_candidate_cost(picked_sniper)

    picked_hunter: dict[str, Any] | None = None
    hunter_cost = 0.0
    for cand in top_t:
        _, cost = get_candidate_cost(cand)
        if budget is None or cost <= budget * 1.05:
            picked_hunter = cand
            hunter_cost = cost
            break
    if not picked_hunter and top_t:
        picked_hunter = top_t[0]
        _, hunter_cost = get_candidate_cost(picked_hunter)

    chief_text = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「【{budget_label}】について、当ファンドの最新クオンツスクリーニング結果を提示します。\n\n"
        f"株式投資（1単元＝100株）において、リスクを抑えつつ資金効率を最大化する2つのポートフォリオ候補を選定しました。\n\n"
        f"短期で資金を回転させるならスナイパー推奨株、中長期で配当や底値反転を狙うならハンター推奨株が最適です。」"
    )

    fields: list[dict[str, Any]] = []
    if picked_sniper:
        p_up = picked_sniper.get("probability_up", 0.55)
        q_b = picked_sniper.get("q_band", "Q5")
        fields.append({
            "name": f"⚡ 短期Overnight推奨: {picked_sniper.get('name')} (`{picked_sniper.get('ticker')}`)",
            "value": (
                f"想定必要資金 (100株): **約 {sniper_cost:,.0f}円**\n"
                f"翌朝上昇確率: **{p_up:.1%}** ({q_b})\n"
                f"特徴: 寄り付きギャップアップ期待値が高く、即効性の高い短期資金運用向き"
            ),
            "inline": False,
        })
    if picked_hunter:
        score = picked_hunter.get("bargain_score", 70.0)
        pbr_s = f"{picked_hunter['pbr']:.2f}倍" if picked_hunter.get("pbr") is not None else "-"
        fields.append({
            "name": f"💎 中長期Turnaround推奨: {picked_hunter.get('name')} (`{picked_hunter.get('ticker')}`)",
            "value": (
                f"想定必要資金 (100株): **約 {hunter_cost:,.0f}円**\n"
                f"割安スコア: **{score:.1f}点** (PBR {pbr_s})\n"
                f"特徴: 52週高値から大きく売り込まれた底値圏。下値不安が極めて少なくNISA・放置運用向き"
            ),
            "inline": False,
        })

    fields.append({
        "name": "💡 資金配分アドバイス",
        "value": (
            "全資金を一度に1銘柄へ集中させるのではなく、まず余力を50%残してエントリーするか、"
            "短期枠と中長期枠に半々で分散するのが機関投資家のリスク管理鉄則です。"
        ),
        "inline": False,
    })

    embed_chief = {
        "title": f"🎯 【銘柄選定・資金配分プラン】{budget_label} ({date_str})",
        "description": "最新クオンツモデルが導き出した最適ポートフォリオ候補",
        "color": 3066993,
        "fields": fields,
        "footer": {"text": "Stock-AI Autonomous Portfolio Advisory"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_chief = {
        "username": "🎩 チーフ・ファンドマネージャー AI",
        "content": chief_text,
        "text": chief_text,
        "embeds": [embed_chief],
    }

    s_name = picked_sniper.get("name") if picked_sniper else "注目銘柄"
    s_p = picked_sniper.get("probability_up", 0.55) if picked_sniper else 0.55
    sniper_text = (
        f"⚡ **短期スナイパー AI (Overnight担当)**\n"
        f"「短期で確実にリターンを抜きたいなら、俺は迷わず **{s_name}** を推すぜ！🎯\n\n"
        f"100株で約 **{sniper_cost:,.0f}円** で仕込めて、明朝の寄り付き上昇確率は **{s_p:.1%}** だ。\n"
        f"翌朝寄り付きで+1.5%〜2%取れたらサクッと利確して資金を回収するのが勝ち組のムーブだぜ！」"
    )
    p_sniper = {
        "username": "⚡ 短期スナイパー AI (Overnight担当)",
        "content": sniper_text,
        "text": sniper_text,
    }

    h_name = picked_hunter.get("name") if picked_hunter else "割安銘柄"
    hunter_text = (
        f"💎 **バリューハンター AI (中長期・激安株担当)**\n"
        f"「夜安心して眠りたいなら、**{h_name}** のような激安反転株が最適です💎\n\n"
        f"100株で約 **{hunter_cost:,.0f}円**。東証プライムの割安底打ちラインに到達しています。\n"
        f"相場の短期的なノイズに惑わされず、中長期で20%〜30%のリバウンドを悠々と待ちましょう。」"
    )
    p_hunter = {
        "username": "💎 バリューハンター AI (中長期・激安株担当)",
        "content": hunter_text,
        "text": hunter_text,
    }

    return [p_chief, p_sniper, p_hunter]


def build_market_outlook_dialogue_payloads(query: str = "") -> list[dict[str, Any]]:
    """Build market outlook, trading discipline, and risk management consultation dialogue."""
    macro = _get_macro_context()
    date_str = datetime.now().strftime("%Y-%m-%d")
    nk_close = macro["nikkei_close"]
    nk_pct = macro["nikkei_pct"]
    usdjpy = macro["usdjpy"]
    portfolio_status = macro["portfolio_status"]
    clean_q = query.strip()

    is_discipline = any(k in clean_q for k in ["利確", "損切り", "基準", "ルール", "失敗"])
    is_timing = any(k in clean_q for k in ["買い時", "今買っていい", "エントリー", "いつ買えば"])

    chief_text = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「相場環境および運用方針に関するご質問ですね。\n\n"
        f"【最新マクロ環境 ({date_str})】\n"
        f"・日経平均株価: **{nk_close:,.0f} 円** ({nk_pct:+.2%})\n"
        f"・ドル円為替レート: **{usdjpy:.2f} 円**\n"
        f"・当ファンド運用ステータス: **【{portfolio_status}】**\n\n"
        f"感情に任せたトレードは機関投資家のカモになります。クオンツシグナルの点灯と資金管理の規律を最優先してください。」"
    )

    fields = [
        {
            "name": "🌐 市場地合いとファンドスタンス",
            "value": (
                f"日経平均: **{nk_close:,.0f}円** ({nk_pct:+.2%}) | ドル円: **{usdjpy:.2f}円**\n"
                f"ファンド状況: **{portfolio_status}**\n"
                f"シグナル確信度が高い局面以外は無駄なトレードをせず、現金を温存するのも立派な投資行動です。"
            ),
            "inline": False,
        },
        {
            "name": "⚡ 短期売買の鉄則 (スナイパー流)",
            "value": (
                "・**利確目標**: 寄り付き +1.5% 〜 +2.0% で欲張らず指値利確\n"
                "・**損切り基準**: -1.5% で躊躇なく機械的ロスカット\n"
                "・**決算跨ぎ**: 発表前日はノーポジ撤退（ギャンブル禁止）"
            ),
            "inline": False,
        },
        {
            "name": "💎 中長期投資の鉄則 (バリューハンター流)",
            "value": (
                "・**仕込み時**: 日経平均急落・市場総悲観（RSI 30割れ）の時こそ好機\n"
                "・**資金配分**: 一括買いせず3回に分けて時間分散\n"
                "・**銘柄選定**: PBR 1倍割れ＆自己資本比率の高い割安優良株に限定"
            ),
            "inline": False,
        },
    ]

    embed_chief = {
        "title": f"🧭 【相場観・投資戦略ブリーフィング】({date_str})",
        "description": "プロのファンドマネジメント規律に基づく相場診断",
        "color": 3066993,
        "fields": fields,
        "footer": {"text": "Stock-AI Market Intelligence Lounge"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    p_chief = {
        "username": "🎩 チーフ・ファンドマネージャー AI",
        "content": chief_text,
        "text": chief_text,
        "embeds": [embed_chief],
    }

    if is_discipline:
        s_msg = (
            "「利確と損切りの質問だな！耳をかっぽじって聞いてくれ🎯\n\n"
            "勝てるトレーダーと負ける奴の唯一の違いは『損切りができるかどうか』だ！\n"
            "-1.5%まで逆行したら『戻るかも…』なんて希望的観測は捨てて即刻切る！\n"
            "利確は寄り付き+1.5%〜2%で欲張らず確実に利益確定。これを徹底するだけで資産曲線は劇的に変わるぜ！」"
        )
    elif is_timing:
        s_msg = (
            "「『今が買い時か？』って聞かれたら、答えは『銘柄のシグナル次第』だ🎯\n\n"
            "日経平均が上がっていようが下がっていようが、Overnight上昇確率が55%（Q5）を超えている銘柄だけをピンポイントで狙い撃つのが俺の流儀だ。\n"
            "シグナルが出ていない時は無理に撃たずに引き金を引く指を休めるのもスナイパーの仕事だぜ！」"
        )
    else:
        s_msg = (
            "「今の相場で気をつけるべきは『相場環境の急変とボラティリティ』だ🎯\n\n"
            "ドル円の乱高下や海外市場の先物急変に巻き込まれないよう、ポジション保有時間は最短に抑えるのが吉だ。\n"
            "明朝の寄り付きでスパッと決済するOvernightスイングなら、日中の乱高下ストレスもゼロだぜ！」"
        )

    p_sniper = {
        "username": "⚡ 短期スナイパー AI (Overnight担当)",
        "content": f"⚡ **短期スナイパー AI (Overnight担当)**\n{s_msg}",
        "text": f"⚡ **短期スナイパー AI (Overnight担当)**\n{s_msg}",
    }

    if is_timing:
        h_msg = (
            "「買い時についてのバリュー視点からのアドバイスです💎\n\n"
            "個人投資家が一番やってはいけないのは、連日高騰している話題の株に高値掴みで飛びつくことです。\n"
            "本当の買い時は、業績堅調なのに市場全体の連れ安で放置され、PBR1倍を割り込んでいるような『総悲観の底値』です。\n"
            "焦らず、バーゲンセールが訪れるのを待ち伏せしましょう。」"
        )
    else:
        h_msg = (
            "「中長期で資産を守り増やすためのアドバイスです💎\n\n"
            "株価が下がった時に『恐怖で投げ売り』してしまう方は、資金管理に問題があります。\n"
            "生活資金には一切手をつけず、余剰資金の3割程度から小さくスタートすること。\n"
            "下げ相場こそが、将来の富を築く最高の仕込み場になります。どっしり構えましょう。」"
        )

    p_hunter = {
        "username": "💎 バリューハンター AI (中長期・激安株担当)",
        "content": f"💎 **バリューハンター AI (中長期・激安株担当)**\n{h_msg}",
        "text": f"💎 **バリューハンター AI (中長期・激安株担当)**\n{h_msg}",
    }

    return [p_chief, p_sniper, p_hunter]


def build_help_payload() -> dict[str, Any]:
    """Build guidance message explaining how to interact with the bot."""
    help_text = (
        "☕ **【Stock-AI AI投資相談室＆対話型アナリストBot 利用ガイド】**\n\n"
        "チャットに気になる**銘柄コード**、**企業名**、または**自由な投資相談**を直接書き込むだけで、3人の専門AIアナリストがリアルタイム回答します！\n\n"
        "💡 **利用例**:\n"
        "1. **複数銘柄の比較相談**:\n"
        "   ・`トヨタと日産どっちがいい？`\n"
        "   ・`ソニーと任天堂を比較して`\n"
        "   ・`8058と8031`\n"
        "2. **おすすめ・銘柄選定・資金相談**:\n"
        "   ・`今おすすめの株は？`\n"
        "   ・`30万円で仕込むなら何がいい？`\n"
        "   ・`短期で勝てるやつ教えて`\n"
        "3. **相場観・売買タイミング相談**:\n"
        "   ・`今の相場どう思う？`\n"
        "   ・`利確や損切りの基準は？`\n"
        "   ・`日経平均どうなりそう？`\n"
        "4. **個別銘柄の深掘り**:\n"
        "   ・`7203` または `トヨタ`\n"
        "   ・`日産は今買い時？`\n\n"
        "・`ランキング` → 本日の注目銘柄ランキング速報"
    )
    embed = {
        "title": "📖 Stock-AI 対話・投資相談コマンドガイド",
        "description": "自然言語での質問や銘柄コードをそのままチャットに入力してください。",
        "color": 3447003,
        "fields": [
            {"name": "⚔️ 銘柄比較相談", "value": "`トヨタと日産どっち？`, `ソニー 任天堂 比較`", "inline": True},
            {"name": "🎯 おすすめ・資金相談", "value": "`30万円で買うなら？`, `今おすすめの株`", "inline": True},
            {"name": "🧭 相場観・売買規律", "value": "`今の相場どう？`, `損切りの基準は？`", "inline": True},
            {"name": "🔍 単一銘柄分析", "value": "`7203`, `トヨタ`, `任天堂買い時？`", "inline": True},
            {"name": "🏆 ランキング速報", "value": "`ランキング`, `top`, `順位`", "inline": True},
            {"name": "ℹ️ ヘルプ", "value": "`ヘルプ`, `help`, `使い方`", "inline": True},
        ],
        "footer": {"text": "Stock-AI Autonomous Investment Consultation Room"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "username": "☕ Stock-AI 投資相談室コンシェルジュ",
        "content": help_text,
        "text": help_text,
        "embeds": [embed],
    }


def build_ranking_payloads() -> list[dict[str, Any]]:
    """Build top rankings briefing using latest snapshot."""
    summary = get_current_pipeline_summary()
    top_o = summary.get("top_overnight", [])
    top_t = summary.get("top_turnaround", [])
    date_str = summary.get("date", datetime.now().strftime("%Y-%m-%d"))

    lines_o = []
    if top_o:
        for idx, item in enumerate(top_o[:3], 1):
            lines_o.append(
                f"{idx}. `{item.get('ticker')}` **{item.get('name')}**: 上昇確率 **{item.get('probability_up', 0.0):.1%}** ({item.get('q_band', 'Q5')})"
            )
    else:
        lines_o.append("該当なし")

    lines_t = []
    if top_t:
        for idx, item in enumerate(top_t[:3], 1):
            pbr_s = f"{item['pbr']:.2f}倍" if item.get("pbr") is not None else "-"
            lines_t.append(
                f"{idx}. `{item.get('ticker')}` **{item.get('name')}**: スコア **{item.get('bargain_score', 0.0):.1f}点** (高値比 {item.get('drawdown_252d', 0.0):+.1%}, PBR {pbr_s})"
            )
    else:
        lines_t.append("該当なし")

    content = (
        f"🏆 **【Stock-AI 本日の注目ランキング速報 ({date_str})】**\n\n"
        f"⚡ **Overnight Q5 (明朝寄り付きギャップアップ期待 Top 3)**:\n"
        + "\n".join(lines_o)
        + "\n\n"
        "💎 **Value & Turnaround (激安回復お宝株 Top 3)**:\n"
        + "\n".join(lines_t)
    )

    embed = {
        "title": f"🏆 Stock-AI デイリーランキング速報 ({date_str})",
        "description": "最新のOvernight予測および激安ターンアラウンドスキャナーの上位銘柄",
        "color": 15844367,  # Gold
        "fields": [
            {"name": "⚡ Overnight 翌朝ギャップ期待 Top 3", "value": "\n".join(lines_o), "inline": False},
            {"name": "💎 激安回復お宝株 Top 3", "value": "\n".join(lines_t), "inline": False},
        ],
        "footer": {"text": "Stock-AI Quantitative Ranking"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return [{
        "username": "🏆 Stock-AI ランキング速報 AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }]


# ---------------------------------------------------------------------------
# 4. Discord REST API Polling Listener Bot
# ---------------------------------------------------------------------------


class DiscordPollingBot:
    """Lightweight REST-based Discord Polling Listener and Multi-Persona Responder."""

    def __init__(
        self,
        bot_token: str | None = None,
        poll_interval: float = 3.0,
    ) -> None:
        self.bot_token = (
            bot_token
            or os.environ.get("DISCORD_BOT_TOKEN")
            or DiscordProvisioner._load_token_from_env_file()
        )
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is not configured in .env or environment")

        self.poll_interval = poll_interval
        self.webhooks: dict[str, str] = get_configured_channel_webhooks()
        self.channel_to_webhook: dict[str, str] = {}  # channel_id -> webhook_url
        self.last_seen_message_id: dict[str, str] = {}
        self.monitored_channels: list[dict[str, Any]] = []

    def _api_request(
        self,
        endpoint: str,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        retries: int = 3,
    ) -> Any:
        url = f"{DISCORD_API_BASE}{endpoint}"
        headers = {
            "Authorization": f"Bot {self.bot_token}",
            "User-Agent": "StockAI-PollingBot/1.0",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=12) as resp:
                    content = resp.read().decode("utf-8")
                    return json.loads(content) if content else {}
            except urllib.error.HTTPError as err:
                err_body = err.read().decode("utf-8", errors="replace")
                if err.code == 429 and attempt < retries - 1:
                    time.sleep(2.0)
                    continue
                logger.warning("Discord API %s %s error %s: %s", method, endpoint, err.code, err_body)
                return {}
            except Exception as exc:
                logger.warning("Discord API network error on %s %s: %s", method, endpoint, exc)
                if attempt < retries - 1:
                    time.sleep(1.0)
                    continue
                return {}
        return {}

    def _get_or_create_channel_webhook(self, channel_id: str) -> str | None:
        """Retrieve existing webhook or auto-create one for this channel."""
        if channel_id in self.channel_to_webhook:
            return self.channel_to_webhook[channel_id]

        # 1. Query existing webhooks for this channel
        whs = self._api_request(f"/channels/{channel_id}/webhooks")
        if isinstance(whs, list) and whs:
            for w in whs:
                if w.get("url"):
                    url = normalize_webhook_url(w["url"])
                    self.channel_to_webhook[channel_id] = url
                    return url

        # 2. Create a new dedicated webhook
        created = self._api_request(
            f"/channels/{channel_id}/webhooks",
            method="POST",
            payload={"name": "Stock-AI Reporter"},
        )
        if isinstance(created, dict) and created.get("url"):
            url = normalize_webhook_url(created["url"])
            self.channel_to_webhook[channel_id] = url
            return url

        return None

    def init_channels(self) -> None:
        """Discover all server text channels, bind webhooks, and record current latest message IDs."""
        guilds = self._api_request("/users/@me/guilds")
        if not guilds:
            logger.warning("Bot is not joined to any guild.")
            return

        guild_id = str(guilds[0]["id"])
        channels = self._api_request(f"/guilds/{guild_id}/channels")
        if not isinstance(channels, list):
            logger.warning("Could not list channels for guild %s", guild_id)
            return

        # Target channels: ALL text channels (type 0) in the server
        self.monitored_channels = [ch for ch in channels if ch.get("type") == 0]
        logger.info("Discovered %d monitored text channels across server.", len(self.monitored_channels))

        # Map channel_id to Webhook URL
        for ch in self.monitored_channels:
            ch_id = str(ch["id"])
            ch_name = ch.get("name", "")

            # 1. Match from predefined environment webhooks by channel keyword
            if "作戦" in ch_name or "main" in ch_name:
                wh = self.webhooks.get("main")
            elif "ファンド" in ch_name or "fund" in ch_name:
                wh = self.webhooks.get("fund")
            elif "激安" in ch_name or "turnaround" in ch_name:
                wh = self.webhooks.get("turnaround")
            elif "アラート" in ch_name or "alert" in ch_name:
                wh = self.webhooks.get("alert")
            elif "相談" in ch_name or "consultation" in ch_name:
                wh = self.webhooks.get("consultation")
            else:
                wh = None

            if wh:
                self.channel_to_webhook[ch_id] = normalize_webhook_url(wh)
            else:
                # 2. Or dynamically resolve/create webhook for this channel
                self._get_or_create_channel_webhook(ch_id)

            # 3. Snapshot current latest message ID so bot ignores historical messages
            msgs = self._api_request(f"/channels/{ch_id}/messages?limit=1")
            if isinstance(msgs, list) and msgs:
                self.last_seen_message_id[ch_id] = str(msgs[0]["id"])
            else:
                self.last_seen_message_id[ch_id] = "0"

    def send_reply(self, channel_id: str, payloads: list[dict[str, Any]]) -> None:
        """Send reply payloads using Webhook (preferred for persona avatars) or direct Bot API."""
        wh_url = self._get_or_create_channel_webhook(channel_id)
        for idx, payload in enumerate(payloads):
            if idx > 0:
                time.sleep(0.5)

            if wh_url:
                try:
                    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                    req = urllib.request.Request(
                        wh_url,
                        data=data,
                        headers={"Content-Type": "application/json", "User-Agent": "StockAI-Bot/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=10):
                        pass
                    continue
                except Exception as exc:
                    logger.debug("Failed sending via Webhook %s: %s", wh_url, exc)

            # Fallback to direct bot message
            post_payload: dict[str, Any] = {}
            if payload.get("content"):
                post_payload["content"] = payload["content"]
            if payload.get("embeds"):
                post_payload["embeds"] = payload["embeds"]
            self._api_request(f"/channels/{channel_id}/messages", method="POST", payload=post_payload)

    def send_announcement(self, channel_keyword: str, message: str, title: str | None = None) -> bool:
        """Post an announcement message to a channel matching keyword."""
        target_ch_id: str | None = None
        for ch in self.monitored_channels:
            if channel_keyword in ch.get("name", "") or channel_keyword in str(ch.get("id")):
                target_ch_id = str(ch["id"])
                break

        if not target_ch_id and self.monitored_channels:
            target_ch_id = str(self.monitored_channels[0]["id"])

        if not target_ch_id:
            return False

        payload = {
            "username": "🤖 Stock-AI アナリストBot",
            "content": message,
            "embeds": [
                {
                    "title": title or "🚀 対話型AIアナリストBot 起動案内",
                    "description": message,
                    "color": 3447003,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ] if title else [],
        }
        self.send_reply(target_ch_id, [payload])
        return True

    def poll_once(self) -> int:
        """Check all monitored channels for new incoming user messages and respond."""
        processed_count = 0
        for ch in self.monitored_channels:
            ch_id = str(ch["id"])
            last_id = self.last_seen_message_id.get(ch_id, "0")

            endpoint = f"/channels/{ch_id}/messages?limit=5"
            if last_id != "0":
                endpoint += f"&after={last_id}"

            msgs = self._api_request(endpoint)
            if not isinstance(msgs, list) or not msgs:
                continue

            # Process in chronological order (oldest first)
            msgs.sort(key=lambda m: int(m["id"]))

            for msg in msgs:
                msg_id = str(msg["id"])
                # Update seen message tracker
                if int(msg_id) > int(self.last_seen_message_id.get(ch_id, "0")):
                    self.last_seen_message_id[ch_id] = msg_id

                # Ignore bot messages (including own messages and webhooks)
                author = msg.get("author", {})
                if author.get("bot") or msg.get("webhook_id"):
                    continue

                raw_content = msg.get("content") or ""
                logger.info("Processing message in channel %s: '%s' from %s", ch.get("name"), raw_content, author.get("username"))
                reply_payloads = self._handle_user_message(raw_content, msg=msg)
                if reply_payloads:
                    self.send_reply(ch_id, reply_payloads)
                    processed_count += 1

        return processed_count

    def _handle_user_message(
        self,
        content: str,
        msg: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Route user input to multi-stock comparison, recommendation, market outlook, or single stock analysis."""
        # Strip mention tags (e.g., <@123456789>)
        cleaned = re.sub(r"<@!?[0-9]+>", "", content).strip()
        clean_lower = cleaned.lower()

        # 1. If content is empty (Message Content Intent is OFF on Discord Developer Portal)
        if not cleaned:
            notice_text = (
                "🔔 **【Stock-AI アナリストBot 受信通知】**\n\n"
                "メッセージを受信しましたが、Discord の **`Message Content Intent`** が未有効化のため、テキスト本文が空文字 `\"\"` として届いております。\n\n"
                "💡 **解決方法**:\n"
                "1. **Botをメンションして発言** (設定変更なしで即動作):\n"
                "   `@Stock-AI トヨタ` や `@Stock-AI トヨタと日産どっち？` のようにメンション付きで書き込むと、即座に回答されます！\n"
                "2. **Discord Developer Portal で有効化** (メンションなしで発言したい場合):\n"
                "   [Discord Developer Portal](https://discord.com/developers/applications) → Bot設定 → `Privileged Gateway Intents` の **`Message Content Intent` を ON** に設定してください。\n\n"
                "--- \n"
                "🏆 **本日の最新Overnight＆激安株ランキング速報をお届けします！** 👇"
            )
            notice_payload = {
                "username": "☕ Stock-AI 投資相談室コンシェルジュ",
                "content": notice_text,
                "text": notice_text,
            }
            return [notice_payload, *build_ranking_payloads()]

        # 2. Help inquiry
        if clean_lower in ("ヘルプ", "help", "使い方", "?", "コマンド", "help me"):
            return [build_help_payload()]

        # 3. Multi-stock comparison inquiry (2 or more distinct tickers found)
        tickers = extract_all_tickers(cleaned)
        if len(tickers) >= 2:
            return build_comparison_dialogue_payloads(tickers, query=cleaned)

        # 4. Recommendation / Budget / Stock Selection inquiry
        recommend_keywords = [
            "おすすめ", "何買えば", "何かいい銘柄", "どれがいい", "仕込むなら",
            "買えばいい", "勝てるやつ", "予算", "nisa", "放置", "万", "30万",
            "50万", "100万", "どの株", "買いたい", "銘柄教えて",
        ]
        if any(k in clean_lower for k in recommend_keywords) and len(tickers) == 0:
            return build_recommendation_dialogue_payloads(query=cleaned)

        # 5. Market outlook / Strategy / Timing inquiry
        market_keywords = [
            "相場", "地合い", "日経", "円安", "円高", "ドル円", "買い時", "売り時",
            "利確", "損切り", "基準", "気をつける", "どう思う", "市場", "暴落",
            "調整", "ポジション", "ルール", "注意点", "見通し",
        ]
        if any(k in clean_lower for k in market_keywords) and len(tickers) == 0:
            return build_market_outlook_dialogue_payloads(query=cleaned)

        # 6. Single stock deep dive inquiry (1 ticker found or direct code/name)
        if len(tickers) == 1:
            analysis = analyze_single_ticker(tickers[0][0])
            if analysis:
                return build_single_ticker_dialogue_payloads(analysis, query=cleaned)

        # Fallback single ticker analysis
        analysis = analyze_single_ticker(cleaned)
        if analysis:
            return build_single_ticker_dialogue_payloads(analysis, query=cleaned)

        # 7. Daily Rankings inquiry
        if clean_lower in ("ランキング", "top", "順位", "注目", "ranking"):
            return build_ranking_payloads()

        # 8. Mentions or explicit request without resolved ticker
        if "分析" in cleaned or "株価" in cleaned or "予想" in cleaned:
            guidance = (
                f"申し訳ありません、「**{cleaned[:30]}**」から日本株の銘柄コードまたは企業名を特定できませんでした。\n\n"
                "例: `7203`, `トヨタ`, `6758`, `ソニー`, `トヨタと日産どっち？`, `30万円でおすすめ` のように入力してください。"
            )
            return [{
                "username": "☕ Stock-AI 投資相談室コンシェルジュ",
                "content": guidance,
                "text": guidance,
            }]

        # 9. General investment question fallback
        if any(k in clean_lower for k in ["株", "投資", "買い", "売り", "儲かる", "教えて", "どう", "アドバイス"]):
            return build_market_outlook_dialogue_payloads(query=cleaned)

        return None

    def run(
        self,
        max_iterations: int | None = None,
        stop_event: Any = None,
    ) -> None:
        """Main listening loop."""
        self.init_channels()
        logger.info("Stock-AI Polling Bot started. Listening on %d channels...", len(self.monitored_channels))

        iterations = 0
        while True:
            if stop_event and getattr(stop_event, "is_set", lambda: False)():
                logger.info("Stop event received. Exiting polling loop.")
                break

            try:
                self.poll_once()
            except Exception as exc:
                logger.warning("Error during polling iteration: %s", exc)

            iterations += 1
            if max_iterations is not None and iterations >= max_iterations:
                break

            time.sleep(self.poll_interval)


def main() -> None:
    """CLI entrypoint to run the polling bot."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=================================================================")
    print("🤖 [Stock-AI] Interactive Discord Analyst Polling Bot Starting...")
    print("=================================================================")
    bot = DiscordPollingBot(poll_interval=3.0)
    bot.run()


if __name__ == "__main__":
    main()
