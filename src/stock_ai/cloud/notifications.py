from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_webhook_payload(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Build a rich notification payload compatible with both Discord and Slack."""
    date_str = summary_data.get("date") or datetime.now().strftime("%Y-%m-%d")
    top_overnight = summary_data.get("top_overnight", [])
    paper_metrics = summary_data.get("paper_trader", {})
    turnaround_picks = summary_data.get("top_turnaround", [])

    lines = [
        f"📊 **Stock AI Daily Cloud Pipeline ({date_str})**",
        "",
        "🏆 **Overnight Q5 (明朝ギャップアップ期待 Top 3)**:",
    ]
    if top_overnight:
        for idx, item in enumerate(top_overnight[:3], 1):
            p_up = item.get("probability_up", 0.0)
            q_info = item.get("quantile") or item.get("label") or "Q5"
            e_flag = "⚠️決算警戒" if item.get("earnings_risk") else "✅通常"
            ticker = item.get("ticker", "")
            name = item.get("name") or ticker
            sector = item.get("sector", "-")
            lines.append(
                f"  {idx}. `{ticker}` {name} ({sector}): "
                f"**{p_up:.1%}** ({q_info}) | {e_flag}"
            )
    else:
        lines.append("  (該当なし)")

    eq = paper_metrics.get("final_equity", 1_000_000.0)
    ret = paper_metrics.get("cumulative_return", 0.0)
    win = paper_metrics.get("win_rate", 0.0)
    sh = paper_metrics.get("sharpe_ratio", 0.0)
    max_dd = paper_metrics.get("max_drawdown", 0.0)
    pos = paper_metrics.get("next_positions") or "本日エントリーなし (キャッシュ待機判定)"

    lines.extend([
        "",
        "💰 **仮想100万円ファンド (Overnight Paper Trader)**:",
        f"  ・総資産: **{eq:,.0f} 円** ({ret:+.1%})",
        f"  ・勝率: **{win:.1%}** | Sharpe: **{sh:.2f}** | MaxDD: **{max_dd:.1%}**",
        f"  ・本日エントリー判定: {pos}",
        "",
        "💎 **Value & Turnaround (将来回復しそうな激安株 Top 3)**:",
    ])
    if turnaround_picks:
        for idx, item in enumerate(turnaround_picks[:3], 1):
            pbr_s = f"{item['pbr']:.2f}倍" if item.get("pbr") is not None else "-"
            score = item.get("bargain_score", 0.0)
            dd = item.get("drawdown_252d", 0.0)
            status = item.get("status", "")
            ticker = item.get("ticker", "")
            name = item.get("name") or ticker
            sector = item.get("sector", "-")
            lines.append(
                f"  {idx}. `{ticker}` {name} ({sector}): "
                f"スコア **{score:.1f}点** (52週高値比: {dd:+.1%}, PBR: {pbr_s}) — {status}"
            )
    else:
        lines.append("  (該当なし)")

    content_text = "\n".join(lines)

    overnight_embed_items = []
    if top_overnight:
        for i in top_overnight[:3]:
            e_flag = "⚠️決算警戒" if i.get("earnings_risk") else "✅通常"
            ticker = i.get("ticker", "")
            name = i.get("name") or ticker
            p_up = i.get("probability_up", 0.0)
            q_band = i.get("q_band") or i.get("quantile") or "Q5"
            overnight_embed_items.append(
                f"・`{ticker}` {name}: **{p_up:.1%}** ({q_band}) | {e_flag}"
            )

    turnaround_embed_items = []
    if turnaround_picks:
        for idx, i in enumerate(turnaround_picks[:3], 1):
            pbr_s = f"{i['pbr']:.2f}倍" if i.get("pbr") is not None else "-"
            status = i.get("status", "")
            ticker = i.get("ticker", "")
            name = i.get("name") or ticker
            score = i.get("bargain_score", 0.0)
            dd = i.get("drawdown_252d", 0.0)
            turnaround_embed_items.append(
                f"{idx}. `{ticker}` {name}: **{score:.1f}点** ({dd:+.1%}, PBR: {pbr_s})\n   └ {status}"
            )

    embed = {
        "title": f"🚀 Stock-AI クラウド自動更新完了 ({date_str})",
        "description": "PC電源オフ環境での日次自動予測パイプラインが正常に完走しました。",
        "color": 3447003,
        "fields": [
            {
                "name": "🏆 Overnight Q5 注目銘柄",
                "value": "\n".join(overnight_embed_items) if overnight_embed_items else "なし",
                "inline": False,
            },
            {
                "name": "💰 仮想100万円ファンド成績",
                "value": f"資産: **{eq:,.0f}円** ({ret:+.1%})\n勝率: **{win:.1%}** | Sharpe: **{sh:.2f}** | MaxDD: **{max_dd:.1%}**\n本日判定: **{pos}**",
                "inline": False,
            },
            {
                "name": "💎 激安回復株 (Turnaround Top 3)",
                "value": "\n".join(turnaround_embed_items) if turnaround_embed_items else "なし",
                "inline": False,
            },
        ],
        "footer": {"text": "Stock AI Cloud Automation Pipeline"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "content": content_text,
        "text": content_text,
        "embeds": [embed],
    }


def get_current_pipeline_summary(
    snapshot_path: Path | str = "reports/autonomous/latest_overnight_scanner_snapshot.json",
    paper_path: Path | str = "reports/autonomous/overnight_portfolio_simulation_results.json",
    turnaround_path: Path | str = "reports/autonomous/value_turnaround_scanner_results.json",
) -> dict[str, Any]:
    """Load latest results from autonomous report files to create a pipeline summary."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    top_overnight: list[dict[str, Any]] = []
    paper_metrics: dict[str, Any] = {}
    top_turnaround: list[dict[str, Any]] = []

    s_file = Path(snapshot_path)
    if s_file.exists():
        try:
            s_data = json.loads(s_file.read_text(encoding="utf-8"))
            date_str = s_data.get("date") or date_str
            top_overnight = s_data.get("rankings", [])[:5]
        except Exception:
            pass

    p_file = Path(paper_path)
    if p_file.exists():
        try:
            p_data = json.loads(p_file.read_text(encoding="utf-8"))
            best_s = p_data.get("results_by_cost", {}).get("5bps", {}).get("elite_top2_long", {})
            m = best_s.get("metrics", {})
            pending_b = p_data.get("latest_pending", {}).get("strategies", {}).get("elite_top2_long", {})
            picks = pending_b.get("long_picks", [])
            paper_metrics = {
                "final_equity": m.get("final_equity", 1_000_000.0),
                "cumulative_return": m.get("cumulative_return", 0.0),
                "win_rate": m.get("win_rate", 0.0),
                "sharpe_ratio": m.get("sharpe_ratio", 0.0),
                "max_drawdown": m.get("max_drawdown", 0.0),
                "next_positions": ", ".join(picks) if picks else "本日エントリーなし (キャッシュ待機判定)",
            }
        except Exception:
            pass

    t_file = Path(turnaround_path)
    if t_file.exists():
        try:
            t_data = json.loads(t_file.read_text(encoding="utf-8"))
            top_turnaround = t_data.get("rankings", [])[:3]
        except Exception:
            pass

    return {
        "date": date_str,
        "top_overnight": top_overnight,
        "paper_trader": paper_metrics,
        "top_turnaround": top_turnaround,
    }


def normalize_webhook_url(raw_url: str) -> str:
    """Normalize Discord and other webhook URLs (e.g. discordapp.com -> discord.com)."""
    url = raw_url.strip().strip('"').strip("'")
    if "discordapp.com" in url:
        url = url.replace("discordapp.com", "discord.com")
    return url


def _load_webhook_url_from_env_file() -> str | None:
    """Search for STOCK_AI_WEBHOOK_URL in local .env files if available."""
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[3] / ".env",
    ]
    for env_path in candidates:
        if env_path.exists():
            try:
                for line in env_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("#") or not line or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "STOCK_AI_WEBHOOK_URL":
                        val = v.strip().strip('"').strip("'")
                        if val:
                            return val
            except Exception:
                pass
    return None


def build_multi_agent_webhook_payloads(summary_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build a 3-agent briefing dialogue payload series for Discord Webhook.

    Personas:
    1. 🎩 チーフ・ファンドマネージャー AI (Chief Fund Manager)
    2. ⚡ 短期スナイパー AI (Overnight担当) (Short-Term Overnight Sniper)
    3. 💎 バリューハンター AI (中長期・激安株担当) (Long-Term Value Hunter)
    """
    date_str = summary_data.get("date") or datetime.now().strftime("%Y-%m-%d")
    top_overnight = summary_data.get("top_overnight", [])
    paper_metrics = summary_data.get("paper_trader", {})
    turnaround_picks = summary_data.get("top_turnaround", [])

    eq = paper_metrics.get("final_equity", 1_000_000.0)
    ret = paper_metrics.get("cumulative_return", 0.0)
    win = paper_metrics.get("win_rate", 0.0)
    sh = paper_metrics.get("sharpe_ratio", 0.0)
    max_dd = paper_metrics.get("max_drawdown", 0.0)
    pos = paper_metrics.get("next_positions") or "本日エントリーなし (キャッシュ待機判定)"

    # --- Persona 1: Chief Fund Manager ---
    is_cash_wait = (
        "本日エントリーなし" in pos
        or "キャッシュ待機" in pos
        or not pos
        or pos.strip() == "なし"
    )
    if is_cash_wait:
        entry_decision = (
            "🛡️ **【本日の売買判定: キャッシュ待機】**\n"
            "本日は優位性の高いQ5シグナル条件を満たす銘柄がないため、**キャッシュ待機（ノーポジ）** で資産を守ります。"
            "無理なエントリーは避け、勝率の高い好機を待ちましょう。"
        )
    else:
        entry_decision = (
            f"🎯 **【本日の売買判定: 実戦エントリー】**\n"
            f"シグナル点灯！明朝の寄り付きに向けて **【{pos}】** で勝負します！"
            "リスク管理を徹底しつつ、規律あるトレードで利益を狙いに行きます。"
        )

    chief_speech = (
        f"🎩 **チーフ・ファンドマネージャー AI**\n"
        f"「皆さん、本日もお疲れ様です。{date_str} の作戦ブリーフィングを開始します。\n\n"
        f"当ファンド（Overnight Paper Trader）の現在の総資産は **{eq:,.0f} 円**（通算リターン: **{ret:+.1%}** / 勝率: **{win:.1%}** / Sharpe: **{sh:.2f}** / MaxDD: **{max_dd:.1%}**）で推移しています。\n\n"
        f"{entry_decision}\n\n"
        f"それではスナイパー、短期Overnightスキャンの精査結果を報告してください。」"
    )

    embed_chief = {
        "title": f"💼 仮想100万円ファンド 本日の運用状況 ({date_str})",
        "description": "規律あるリスク管理に基づく Overnight Paper Trader の運用推移",
        "color": 3066993,  # 0x2ecc71 (Emerald Green)
        "fields": [
            {
                "name": "💵 総資産 / 累積リターン",
                "value": f"**{eq:,.0f} 円** ({ret:+.1%})",
                "inline": True,
            },
            {
                "name": "🎯 勝率 / Sharpe",
                "value": f"**{win:.1%}** / **{sh:.2f}**",
                "inline": True,
            },
            {
                "name": "🛡️ 最大ドローダウン",
                "value": f"**{max_dd:.1%}**",
                "inline": True,
            },
            {
                "name": "⚖️ 本日の売買判定",
                "value": f"**{pos}**",
                "inline": False,
            },
        ],
        "footer": {"text": "Stock AI Chief Fund Manager | 規律あるポートフォリオ運営"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload_chief = {
        "username": "🎩 チーフ・ファンドマネージャー AI",
        "content": chief_speech,
        "text": chief_speech,
        "embeds": [embed_chief],
    }

    # --- Persona 2: Short-Term Sniper ---
    has_earnings_risk = any(item.get("earnings_risk") for item in top_overnight[:3])
    overnight_fields = []

    if top_overnight:
        for idx, i in enumerate(top_overnight[:3], 1):
            ticker = i.get("ticker", "")
            name = i.get("name") or ticker
            sector = i.get("sector", "-")
            p_up = i.get("probability_up", 0.0)
            q_band = i.get("q_band") or i.get("quantile") or "Q5"
            e_flag = "⚠️決算警戒" if i.get("earnings_risk") else "✅通常"
            overnight_fields.append({
                "name": f"{idx}. `{ticker}` {name} ({sector})",
                "value": f"上昇確率: **{p_up:.1%}** ({q_band}) | 決算リスク: **{e_flag}**",
                "inline": False,
            })

        if has_earnings_risk:
            risk_comment = (
                "⚠️ **注意喚起**: ピックアップ銘柄の中に直近の決算発表警戒フラグ（`⚠️決算警戒`）が出ている銘柄があります！"
                "決算前のボラティリティ急変を回避するため、慎重なロット配分または見送りを推奨します。"
            )
        else:
            risk_comment = (
                "✅ 直近の決算発表リスクはクリアしています！"
                "クリーンなテクニカル・クオンツ優位性に基づき、明朝の寄り付きギャップアップを狙える布陣です。"
            )

        sniper_speech = (
            f"⚡ **短期スナイパー AI (Overnight担当)**\n"
            f"「了解です、ボス！短期Overnightスキャンの結果を報告します🎯\n\n"
            f"日経225ユニバースの中から、明朝の寄り付き上昇確率が高い上位3銘柄をロックオンしました。\n\n"
            f"{risk_comment}\n\n"
            f"短期チームの報告は以上です。バリューハンター、中長期の割安銘柄の動向はどうですか？」"
        )
    else:
        overnight_fields.append({
            "name": "該当銘柄なし",
            "value": "本日のスクリーニング基準を満たす銘柄はありませんでした。",
            "inline": False,
        })
        sniper_speech = (
            "⚡ **短期スナイパー AI (Overnight担当)**\n"
            "「了解です、ボス！本日のスクリーニングでは基準を突破する確度の高い強気銘柄は見つかりませんでした。\n"
            "無駄撃ちは避け、次の寄り付き好機まで照準を研ぎ澄ませて待機します！\n\n"
            "バリューハンター、中長期のお宝株の状況を教えてください！」"
        )

    embed_sniper = {
        "title": f"🏆 Overnight Q5 翌朝ギャップアップ期待 Top 3 ({date_str})",
        "description": "東証大引け後〜翌朝寄り付き（Close-to-Open）の高確率上昇候補",
        "color": 15844367,  # 0xf1c40f (Gold)
        "fields": overnight_fields,
        "footer": {"text": "Stock AI Short-Term Sniper | Overnight Quantitative Model"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload_sniper = {
        "username": "⚡ 短期スナイパー AI (Overnight担当)",
        "content": sniper_speech,
        "text": sniper_speech,
        "embeds": [embed_sniper],
    }

    # --- Persona 3: Value Hunter ---
    turnaround_fields = []
    if turnaround_picks:
        for idx, i in enumerate(turnaround_picks[:3], 1):
            ticker = i.get("ticker", "")
            name = i.get("name") or ticker
            sector = i.get("sector", "-")
            pbr_s = f"{i['pbr']:.2f}倍" if i.get("pbr") is not None else "-"
            status = i.get("status", "")
            score = i.get("bargain_score", 0.0)
            dd = i.get("drawdown_252d", 0.0)
            turnaround_fields.append({
                "name": f"{idx}. `{ticker}` {name} ({sector})",
                "value": f"割安スコア: **{score:.1f}点** (52週高値比: {dd:+.1%}, PBR: {pbr_s})\nステータス: **{status}**",
                "inline": False,
            })

        high_score_picks = [p for p in turnaround_picks[:3] if p.get("bargain_score", 0.0) >= 70.0]
        if high_score_picks:
            first_high = high_score_picks[0]
            h_name = first_high.get("name") or first_high.get("ticker", "")
            score_comment = (
                f"💎 **お宝発見**: 激安スコア70点超えの超割安銘柄（**{h_name}**: スコア {first_high.get('bargain_score', 0.0):.1f}点）が点灯しています！\n"
                f"52週高値から大幅に売り叩かれ、PBR面でも下値余地が限定的な極上リバウンド候補です。"
            )
        else:
            top1 = turnaround_picks[0]
            t1_name = top1.get("name") or top1.get("ticker", "")
            score_comment = (
                f"東証プライム主要銘柄の中から、売られすぎ水準からの底打ち回復が期待できる銘柄（**{t1_name}** など）をピックアップしました！\n"
                f"中長期の種まきとして魅力的な水準に位置しています。"
            )

        hunter_speech = (
            f"💎 **バリューハンター AI (中長期・激安株担当)**\n"
            f"「スナイパーさん、お疲れ様です！短期が様子見でも動意づいていても、中長期の『お宝激安株』の仕込みチャンスは見逃せませんよ💎✨\n\n"
            f"{score_comment}\n\n"
            f"短期の機動力と中長期のバリュー投資、両輪を意識して明日も賢く相場を攻略していきましょう！」"
        )
    else:
        turnaround_fields.append({
            "name": "該当銘柄なし",
            "value": "現在基準を満たす割安回復銘柄はありません。",
            "inline": False,
        })
        hunter_speech = (
            "💎 **バリューハンター AI (中長期・激安株担当)**\n"
            "「スナイパーさん、お疲れ様です！本日は特筆すべき極端な割安反転シグナルは一服しています。\n"
            "焦って飛びつかず、優良銘柄が大きく下押しした瞬間を狙い撃ちできるよう監視を継続します！\n\n"
            "短期・中長期ともに油断なく、明日も万全の体制で臨みましょう！」"
        )

    embed_hunter = {
        "title": f"💎 Value & Turnaround 激安回復お宝株 Top 3 ({date_str})",
        "description": "東証プライム主要銘柄から下落率・PBR・反転スコアで厳選",
        "color": 10181046,  # 0x9b59b6 (Amethyst Purple)
        "fields": turnaround_fields,
        "footer": {"text": "Stock AI Value Hunter | Bottom-Fishing & Rebound Model"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    payload_hunter = {
        "username": "💎 バリューハンター AI (中長期・激安株担当)",
        "content": hunter_speech,
        "text": hunter_speech,
        "embeds": [embed_hunter],
    }

    return [payload_chief, payload_sniper, payload_hunter]


def send_daily_summary_webhook(
    summary_data: dict[str, Any],
    webhook_url: str | None = None,
    *,
    sleep_interval: float = 0.6,
) -> dict[str, Any]:
    """Send formatted multi-agent briefing to Discord or Slack webhook if URL is provided."""
    raw_url = (
        webhook_url
        if webhook_url is not None
        else (os.environ.get("STOCK_AI_WEBHOOK_URL") or _load_webhook_url_from_env_file())
    )
    if not raw_url:
        return {"status": "SKIPPED", "reason": "STOCK_AI_WEBHOOK_URL is not set"}

    url = normalize_webhook_url(raw_url)
    if not url:
        return {"status": "SKIPPED", "reason": "STOCK_AI_WEBHOOK_URL is not set"}

    payloads = build_multi_agent_webhook_payloads(summary_data)
    last_code = 204
    sent_count = 0

    for idx, payload in enumerate(payloads):
        if idx > 0 and sleep_interval > 0:
            time.sleep(sleep_interval)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "StockAI-CloudRunner/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                code = resp.getcode()
                last_code = code
                sent_count += 1
        except Exception as exc:
            logger.warning(
                f"Webhook delivery failed for persona '{payload.get('username')}': {exc}"
            )
            return {"status": "FAILED", "error": str(exc), "sent_count": sent_count}

    return {"status": "SUCCESS", "status_code": last_code, "messages_sent": sent_count}


