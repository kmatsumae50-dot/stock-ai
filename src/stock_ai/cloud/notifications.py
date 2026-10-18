from __future__ import annotations

import json
import logging
import os
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
            lines.append(
                f"  {idx}. `{item['ticker']}` {item['name']} ({item['sector']}): "
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
            lines.append(
                f"  {idx}. `{item['ticker']}` {item['name']} ({item['sector']}): "
                f"スコア **{score:.1f}点** (52週高値比: {dd:+.1%}, PBR: {pbr_s}) — {status}"
            )
    else:
        lines.append("  (該当なし)")

    content_text = "\n".join(lines)

    overnight_embed_items = []
    if top_overnight:
        for i in top_overnight[:3]:
            e_flag = "⚠️決算警戒" if i.get("earnings_risk") else "✅通常"
            overnight_embed_items.append(
                f"・`{i['ticker']}` {i['name']}: **{i['probability_up']:.1%}** ({i.get('q_band', 'Q5')}) | {e_flag}"
            )

    turnaround_embed_items = []
    if turnaround_picks:
        for idx, i in enumerate(turnaround_picks[:3], 1):
            pbr_s = f"{i['pbr']:.2f}倍" if i.get("pbr") is not None else "-"
            status = i.get("status", "")
            turnaround_embed_items.append(
                f"{idx}. `{i['ticker']}` {i['name']}: **{i['bargain_score']:.1f}点** ({i['drawdown_252d']:+.1%}, PBR: {pbr_s})\n   └ {status}"
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


def send_daily_summary_webhook(
    summary_data: dict[str, Any],
    webhook_url: str | None = None,
) -> dict[str, Any]:
    """Send formatted notification to Discord or Slack webhook if URL is provided."""
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

    payload = build_webhook_payload(summary_data)
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "User-Agent": "StockAI-CloudRunner/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            code = resp.getcode()
            return {"status": "SUCCESS", "status_code": code}
    except Exception as exc:
        logger.warning(f"Webhook delivery failed: {exc}")
        return {"status": "FAILED", "error": str(exc)}

