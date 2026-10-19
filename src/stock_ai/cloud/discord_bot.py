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
    match = re.search(r"\b(\d{4})(?:\.T)?\b", clean_text, re.IGNORECASE)
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
# 3. Multi-Persona Dialogue Payload Builders
# ---------------------------------------------------------------------------


def build_single_ticker_dialogue_payloads(analysis: dict[str, Any]) -> list[dict[str, Any]]:
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

    # 1. Chief Fund Manager Message
    chief_text = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「`{ticker}` **{name}** ({sector}) のオンデマンド精密分析を完了しました。\n\n"
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


def build_help_payload() -> dict[str, Any]:
    """Build guidance message explaining how to interact with the bot."""
    help_text = (
        "🤖 **【Stock-AI 対話型アナリストBot 利用ガイド】**\n\n"
        "チャットに気になる**銘柄コード**または**企業名**を直接書き込むだけで、3人の専門AIアナリストがリアルタイム分析を行います！\n\n"
        "💡 **利用例**:\n"
        "・`7203` または `トヨタ` → トヨタ自動車の最新株価・Overnight確率・割安スコアを返信\n"
        "・`任天堂`, `ソニー`, `6758`, `IHI`, `三菱商事` など主要日本株に対応\n"
        "・`ランキング` または `top` → 本日のOvernight Q5有望株 Top 3 & 激安株 Top 3 を一括返信\n"
        "・`ヘルプ` → この案内を表示"
    )
    embed = {
        "title": "📖 Stock-AI 対話コマンド一覧",
        "description": "銘柄コード（4桁）または日本語企業名をそのままチャットに入力してください。",
        "color": 3447003,
        "fields": [
            {"name": "🔍 銘柄分析", "value": "`7203`, `トヨタ`, `6758.T`, `任天堂` など", "inline": True},
            {"name": "🏆 ランキング速報", "value": "`ランキング`, `top`, `順位`", "inline": True},
            {"name": "ℹ️ ガイド", "value": "`ヘルプ`, `help`, `使い方`", "inline": True},
        ],
        "footer": {"text": "Stock-AI Interactive Assistant"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "username": "🤖 Stock-AI コンシェルジュ",
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

    def init_channels(self) -> None:
        """Discover server channels, match webhooks, and record current latest message IDs."""
        guilds = self._api_request("/users/@me/guilds")
        if not guilds:
            logger.warning("Bot is not joined to any guild.")
            return

        guild_id = str(guilds[0]["id"])
        channels = self._api_request(f"/guilds/{guild_id}/channels")
        if not isinstance(channels, list):
            logger.warning("Could not list channels for guild %s", guild_id)
            return

        # Target channels: text channels (type 0)
        self.monitored_channels = [
            ch for ch in channels if ch.get("type") == 0 and "ai" in ch.get("name", "").lower() or "相場" in ch.get("name", "") or "ファンド" in ch.get("name", "") or "激安" in ch.get("name", "") or "作戦" in ch.get("name", "")
        ]

        logger.info("Discovered %d monitored text channels.", len(self.monitored_channels))

        # Map channel_id to Webhook URL
        for ch in self.monitored_channels:
            ch_id = str(ch["id"])
            ch_name = ch.get("name", "")

            # 1. Match from configured webhooks by key
            if "作戦" in ch_name or "main" in ch_name:
                wh = self.webhooks.get("main")
            elif "ファンド" in ch_name or "fund" in ch_name:
                wh = self.webhooks.get("fund")
            elif "激安" in ch_name or "turnaround" in ch_name:
                wh = self.webhooks.get("turnaround")
            elif "アラート" in ch_name or "alert" in ch_name:
                wh = self.webhooks.get("alert")
            else:
                wh = None

            # 2. Or query channel webhooks via API
            if not wh:
                ch_whs = self._api_request(f"/channels/{ch_id}/webhooks")
                if isinstance(ch_whs, list) and ch_whs:
                    wh = ch_whs[0].get("url")

            if wh:
                self.channel_to_webhook[ch_id] = normalize_webhook_url(wh)

            # 3. Snapshot current latest message ID so bot ignores historical messages
            msgs = self._api_request(f"/channels/{ch_id}/messages?limit=1")
            if isinstance(msgs, list) and msgs:
                self.last_seen_message_id[ch_id] = str(msgs[0]["id"])
            else:
                self.last_seen_message_id[ch_id] = "0"

    def send_reply(self, channel_id: str, payloads: list[dict[str, Any]]) -> None:
        """Send reply payloads using Webhook (preferred for persona avatars) or direct Bot API."""
        wh_url = self.channel_to_webhook.get(channel_id)
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

                # Ignore bot messages (including own messages)
                author = msg.get("author", {})
                if author.get("bot"):
                    continue

                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                logger.info("Processing message in channel %s: '%s'", ch.get("name"), content)
                reply_payloads = self._handle_user_message(content)
                if reply_payloads:
                    self.send_reply(ch_id, reply_payloads)
                    processed_count += 1

        return processed_count

    def _handle_user_message(self, content: str) -> list[dict[str, Any]] | None:
        """Route user input to help, ranking, or stock analysis."""
        clean = content.lower().strip()

        # 1. Help inquiry
        if clean in ("ヘルプ", "help", "使い方", "?", "コマンド", "help me"):
            return [build_help_payload()]

        # 2. Daily Rankings inquiry
        if clean in ("ランキング", "top", "順位", "注目", "おすすめ", "ranking"):
            return build_ranking_payloads()

        # 3. Single Ticker Analysis inquiry
        analysis = analyze_single_ticker(content)
        if analysis:
            return build_single_ticker_dialogue_payloads(analysis)

        # 4. Mentions or explicit request without resolved ticker
        if "分析" in content or "株価" in content or "どう" in content or "予想" in content:
            guidance = (
                f"申し訳ありません、「**{content[:30]}**」から日本株の銘柄コードまたは企業名を特定できませんでした。\n\n"
                "例: `7203`, `トヨタ`, `6758`, `ソニー`, `任天堂`, `8058` のように銘柄コード（4桁）または企業名を入力してください。"
            )
            return [{
                "username": "🤖 Stock-AI コンシェルジュ",
                "content": guidance,
                "text": guidance,
            }]

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
