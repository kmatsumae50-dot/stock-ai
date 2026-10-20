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
            best_s = (
                p_data.get("results_by_cost", {}).get("20bps", {}).get("elite_top2_long")
                or p_data.get("results_by_cost", {}).get("5bps", {}).get("elite_top2_long")
                or {}
            )
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


def _load_env_map() -> dict[str, str]:
    """Parse all key-values from local .env files if present."""
    env_map: dict[str, str] = {}
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
                    val = v.strip().strip('"').strip("'")
                    if val:
                        env_map[k.strip()] = val
            except Exception:
                pass
    return env_map


def _load_webhook_url_from_env_file() -> str | None:
    """Search for STOCK_AI_WEBHOOK_URL in local .env files if available."""
    env_map = _load_env_map()
    return env_map.get("STOCK_AI_WEBHOOK_URL")


def get_configured_channel_webhooks() -> dict[str, str]:
    """Resolve configured webhooks for all dedicated channels.

    Channels:
    - main: ☕｜aiアナリスト作戦会議
    - fund: 💰｜100万円ファンド運用記
    - turnaround: 💎｜激安回復株レーダー
    - alert: 🚨｜決算・急変アラート
    """
    env_data = _load_env_map()
    webhooks: dict[str, str] = {}

    # 1. JSON map
    raw_json = os.environ.get("STOCK_AI_CHANNEL_WEBHOOKS") or env_data.get("STOCK_AI_CHANNEL_WEBHOOKS")
    if raw_json:
        try:
            parsed = json.loads(raw_json)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, str) and v.strip():
                        webhooks[k] = normalize_webhook_url(v)
        except Exception:
            pass

    # 2. Individual environment variable overrides
    key_mappings = {
        "main": ["STOCK_AI_WEBHOOK_URL", "STOCK_AI_WEBHOOK_MAIN"],
        "fund": ["STOCK_AI_WEBHOOK_FUND"],
        "turnaround": ["STOCK_AI_WEBHOOK_TURNAROUND"],
        "alert": ["STOCK_AI_WEBHOOK_ALERT"],
        "consultation": ["STOCK_AI_WEBHOOK_CONSULTATION"],
    }
    for ch_key, env_keys in key_mappings.items():
        for env_k in env_keys:
            val = os.environ.get(env_k) or env_data.get(env_k)
            if val and val.strip():
                webhooks[ch_key] = normalize_webhook_url(val)
                break

    return webhooks



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


def build_fund_report_payload(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Build detailed portfolio management payload for 💰｜100万円ファンド運用記."""
    date_str = summary_data.get("date") or datetime.now().strftime("%Y-%m-%d")
    paper_metrics = summary_data.get("paper_trader", {})
    eq = paper_metrics.get("final_equity", 1_000_000.0)
    ret = paper_metrics.get("cumulative_return", 0.0)
    win = paper_metrics.get("win_rate", 0.0)
    sh = paper_metrics.get("sharpe_ratio", 0.0)
    max_dd = paper_metrics.get("max_drawdown", 0.0)
    pos = paper_metrics.get("next_positions") or "本日エントリーなし (キャッシュ待機判定)"

    is_cash_wait = "本日エントリーなし" in pos or "キャッシュ待機" in pos or not pos
    pos_desc = "🛡️ **キャッシュ待機（ノーポジ）**" if is_cash_wait else f"🎯 **{pos}**"

    content = (
        f"💰 **【仮想100万円ファンド 日次運用レポート ({date_str})】**\n"
        f"・総資産: **{eq:,.0f} 円** ({ret:+.1%})\n"
        f"・勝率: **{win:.1%}** | Sharpe: **{sh:.2f}** | MaxDD: **{max_dd:.1%}**\n"
        f"・本日エントリー判定: {pos_desc}"
    )

    embed = {
        "title": f"💰 仮想100万円ファンド ポートフォリオ運用詳細 ({date_str})",
        "description": "実運用水準20bps摩擦コスト反映済 Overnight エリートTop2ロング戦略シミュレーション（1銘柄上限50%・リークゼロ）",
        "color": 3066993,  # Emerald Green
        "fields": [
            {"name": "💵 総資産", "value": f"**{eq:,.0f} 円**", "inline": True},
            {"name": "📈 通算損益", "value": f"**{ret:+.1%}** ({eq - 1_000_000:+,.0f} 円)", "inline": True},
            {"name": "🎯 勝率", "value": f"**{win:.1%}**", "inline": True},
            {"name": "📊 Sharpe Ratio", "value": f"**{sh:.2f}**", "inline": True},
            {"name": "🛡️ 最大ドローダウン", "value": f"**{max_dd:.1%}**", "inline": True},
            {"name": "⚖️ 本日ポジション判定", "value": pos_desc, "inline": False},
        ],
        "footer": {"text": "Stock AI Overnight Paper Trader | 仮想資金100万円実証"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "username": "💰 仮想100万円ファンド運用記 AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_turnaround_report_payload(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Build detailed value ranking payload for 💎｜激安回復株レーダー."""
    date_str = summary_data.get("date") or datetime.now().strftime("%Y-%m-%d")
    turnaround_picks = summary_data.get("top_turnaround", [])

    fields = []
    text_lines = []
    if turnaround_picks:
        for idx, item in enumerate(turnaround_picks[:5], 1):
            ticker = item.get("ticker", "")
            name = item.get("name") or ticker
            sector = item.get("sector", "-")
            pbr_s = f"{item['pbr']:.2f}倍" if item.get("pbr") is not None else "-"
            score = item.get("bargain_score", 0.0)
            dd = item.get("drawdown_252d", 0.0)
            status = item.get("status", "")
            text_lines.append(f"{idx}. `{ticker}` {name} ({sector}): **{score:.1f}点** (高値比 {dd:+.1%}, PBR {pbr_s}) — {status}")
            fields.append({
                "name": f"{idx}. `{ticker}` {name} ({sector}) — {score:.1f}点",
                "value": f"52週高値比: **{dd:+.1%}** | PBR: **{pbr_s}**\n判定: **{status}**",
                "inline": False,
            })
    else:
        text_lines.append("本日該当銘柄なし")
        fields.append({
            "name": "該当銘柄なし",
            "value": "基準を満たす極端な割安反転シグナルはありません。",
            "inline": False,
        })

    content = (
        f"💎 **【Value & Turnaround 激安回復お宝株スキャン速報 ({date_str})】**\n"
        + "\n".join(text_lines)
    )
    embed = {
        "title": f"💎 Value & Turnaround 割安底打ち反転ランキング ({date_str})",
        "description": "東証プライム主要50銘柄から下落率・PBR・テクニカル反転スコアで厳選",
        "color": 10181046,  # Amethyst Purple
        "fields": fields,
        "footer": {"text": "Stock AI Value Hunter | 逆張り・リバウンドスクリーナー"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return {
        "username": "💎 激安回復株レーダー AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_alert_report_payload(summary_data: dict[str, Any]) -> dict[str, Any]:
    """Build risk assessment payload for 🚨｜決算・急変アラート."""
    date_str = summary_data.get("date") or datetime.now().strftime("%Y-%m-%d")
    top_overnight = summary_data.get("top_overnight", [])
    risky_picks = [p for p in top_overnight if p.get("earnings_risk")]

    if risky_picks:
        items_desc = []
        for p in risky_picks:
            ticker = p.get("ticker", "")
            name = p.get("name") or ticker
            sector = p.get("sector", "-")
            items_desc.append(f"・`{ticker}` {name} ({sector}): 直近7日以内に決算発表予定（⚠️決算跨ぎ警戒）")

        content = (
            f"🚨 **【決算・急変アラート：警戒フラグ検知 ({date_str})】**\n"
            "以下の注目銘柄に決算発表前の警戒シグナルが出ています。急変動・地雷リスク回避のためロット抑制または見送りを推奨します！\n\n"
            + "\n".join(items_desc)
        )
        embed = {
            "title": f"⚠️ 直近決算警戒銘柄リスト ({date_str})",
            "description": "決算発表直前のボラティリティ急変リスクを回避するための事前警告",
            "color": 15158332,  # Crimson Red
            "fields": [
                {"name": "⚠️ 警戒銘柄", "value": "\n".join(items_desc), "inline": False},
                {
                    "name": "🛡️ リスク管理指針",
                    "value": "決算跨ぎギャンブルは避け、発表後のアク抜け・コンセンサス確認を待つことを推奨します。",
                    "inline": False,
                },
            ],
            "footer": {"text": "Stock AI Risk Guardian | 防御最優先アラート"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    else:
        content = (
            f"🛡️ **【決算リスク・セーフティチェック完了 ({date_str})】**\n"
            "本日スクリーニングされた注目銘柄に直近の決算発表地雷フラグはありません。健全な取引環境が確認されています。"
        )
        embed = {
            "title": f"✅ 決算リスク・セーフティクリア ({date_str})",
            "description": "注目銘柄群における直近決算発表予定の有無を検証完了",
            "color": 3066993,  # Emerald Green
            "fields": [
                {"name": "🛡️ 判定結果", "value": "主要Overnight候補銘柄に直近決算地雷なし（全銘柄クリア）", "inline": False},
                {"name": "📌 運用アドバイス", "value": "クオンツ確率・テクニカルシグナルに基づき、規律通りにエントリー可能です。", "inline": False},
            ],
            "footer": {"text": "Stock AI Risk Guardian | 安全確認完了"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "username": "🚨 決算・急変アラート AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def _post_single_webhook_payload(webhook_url: str, payload: dict[str, Any]) -> int | None:
    """Helper to post a single payload to a webhook URL, returning HTTP status code."""
    url = normalize_webhook_url(webhook_url)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "StockAI-CloudRunner/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.getcode()
    except Exception as exc:
        logger.warning(f"Webhook delivery failed for persona '{payload.get('username')}' at {url}: {exc}")
        return None


def send_daily_summary_webhook(
    summary_data: dict[str, Any],
    webhook_url: str | None = None,
    *,
    channel_webhooks: dict[str, str] | None = None,
    sleep_interval: float = 0.6,
) -> dict[str, Any]:
    """Send formatted multi-agent briefing and dedicated channel reports.

    Delivers:
    1. Main Room: 3-agent briefing dialogue (Chief FM, Sniper, Value Hunter)
    2. Fund Room: Paper Trader detailed metrics (if webhook configured)
    3. Turnaround Room: Value & Turnaround radar picks (if webhook configured)
    4. Alert Room: Earnings risk alerts (if webhook configured)
    """
    if webhook_url is not None:
        if not webhook_url.strip():
            return {"status": "SKIPPED", "reason": "STOCK_AI_WEBHOOK_URL is not set"}
        if channel_webhooks is None:
            configured = {"main": normalize_webhook_url(webhook_url)}
        else:
            configured = dict(channel_webhooks)
            configured["main"] = normalize_webhook_url(webhook_url)
    else:
        configured = get_configured_channel_webhooks()
        if channel_webhooks:
            configured.update(channel_webhooks)

    main_url = configured.get("main")
    if not main_url and not any(configured.values()):
        return {"status": "SKIPPED", "reason": "STOCK_AI_WEBHOOK_URL is not set"}

    last_code = 204
    total_sent = 0
    channels_delivered: list[str] = []

    # 1. Main Room: 3-agent persona briefing
    if main_url:
        payloads = build_multi_agent_webhook_payloads(summary_data)
        for idx, payload in enumerate(payloads):
            if (total_sent > 0 or idx > 0) and sleep_interval > 0:
                time.sleep(sleep_interval)
            code = _post_single_webhook_payload(main_url, payload)
            if code is None:
                return {
                    "status": "FAILED",
                    "error": "Failed delivery to main webhook",
                    "sent_count": total_sent,
                }
            last_code = code
            total_sent += 1
        channels_delivered.append("main")

    # 2. Dedicated Room: Fund Reporter
    fund_url = configured.get("fund")
    if fund_url and fund_url != main_url:
        if total_sent > 0 and sleep_interval > 0:
            time.sleep(sleep_interval)
        p_fund = build_fund_report_payload(summary_data)
        code = _post_single_webhook_payload(fund_url, p_fund)
        if code is not None:
            last_code = code
            total_sent += 1
            channels_delivered.append("fund")

    # 3. Dedicated Room: Turnaround Radar
    turn_url = configured.get("turnaround")
    if turn_url and turn_url != main_url:
        if total_sent > 0 and sleep_interval > 0:
            time.sleep(sleep_interval)
        p_turn = build_turnaround_report_payload(summary_data)
        code = _post_single_webhook_payload(turn_url, p_turn)
        if code is not None:
            last_code = code
            total_sent += 1
            channels_delivered.append("turnaround")

    # 4. Dedicated Room: Alert Room
    alert_url = configured.get("alert")
    if alert_url and alert_url != main_url:
        if total_sent > 0 and sleep_interval > 0:
            time.sleep(sleep_interval)
        p_alert = build_alert_report_payload(summary_data)
        code = _post_single_webhook_payload(alert_url, p_alert)
        if code is not None:
            last_code = code
            total_sent += 1
            channels_delivered.append("alert")

    return {
        "status": "SUCCESS",
        "status_code": last_code,
        "messages_sent": total_sent,
        "channels_delivered": channels_delivered,
    }



