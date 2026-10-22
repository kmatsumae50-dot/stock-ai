"""Discord server and channel provisioner for Stock-AI multi-room broadcasting."""

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

from stock_ai.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"


class DiscordProvisioner:
    """Provisions categories, channels, and webhooks on a Discord guild."""

    def __init__(self, bot_token: str | None = None) -> None:
        self.bot_token = (
            bot_token
            or os.environ.get("DISCORD_BOT_TOKEN")
            or self._load_token_from_env_file()
        )
        if not self.bot_token:
            raise ValueError("DISCORD_BOT_TOKEN is not provided or configured in .env")

    @staticmethod
    def _load_token_from_env_file() -> str | None:
        candidates = [
            Path.cwd() / ".env",
            Path(__file__).resolve().parents[3] / ".env",
        ]
        for env_path in candidates:
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        if k.strip() == "DISCORD_BOT_TOKEN":
                            val = v.strip().strip('"').strip("'")
                            if val:
                                return val
                except Exception:
                    pass
        return None

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
            "User-Agent": "StockAI-Provisioner/1.0",
            "Content-Type": "application/json",
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(req, timeout=15) as resp:
                    content = resp.read().decode("utf-8")
                    return json.loads(content) if content else {}
            except urllib.error.HTTPError as err:
                err_body = err.read().decode("utf-8", errors="replace")
                if err.code == 429 and attempt < retries - 1:
                    try:
                        err_json = json.loads(err_body)
                        retry_after = float(err_json.get("retry_after", 1.0))
                    except Exception:
                        retry_after = 2.0
                    logger.warning(f"Rate limited on {endpoint}. Sleeping for {retry_after}s...")
                    time.sleep(retry_after + 0.5)
                    continue
                logger.error(f"Discord API error {err.code} on {method} {endpoint}: {err_body}")
                raise RuntimeError(f"Discord API {method} {endpoint} failed ({err.code}): {err_body}") from err

    def get_target_guild_id(self) -> str:
        """Find the guild ID where the bot is installed, prioritizing 'AI株'."""
        guilds = self._api_request("/users/@me/guilds")
        if not guilds:
            # Fallback check existing webhook URL if set
            existing_wh = os.environ.get("STOCK_AI_WEBHOOK_URL")
            if existing_wh:
                try:
                    wh_req = urllib.request.Request(
                        existing_wh, headers={"User-Agent": "StockAI/1.0"}
                    )
                    with urllib.request.urlopen(wh_req, timeout=10) as r:
                        wh_data = json.loads(r.read().decode("utf-8"))
                        if wh_data.get("guild_id"):
                            return str(wh_data["guild_id"])
                except Exception:
                    pass
            # Try to get client_id and show invite URL
            app_info = self._api_request("/oauth2/applications/@me")
            client_id = app_info.get("id", "")
            invite_url = (
                f"https://discord.com/api/oauth2/authorize?client_id={client_id}&permissions=8&scope=bot"
            )
            print(f"\n[Discord Bot Invitation Required]\nPlease invite the bot via:\n{invite_url}\n")
            raise RuntimeError(f"Bot is not in any guild. Please invite via: {invite_url}")

        for g in guilds:
            g_name = g.get("name", "")
            if "AI" in g_name or "株" in g_name:
                return str(g["id"])
        return str(guilds[0]["id"])

    def get_existing_channels(self, guild_id: str) -> list[dict[str, Any]]:
        return self._api_request(f"/guilds/{guild_id}/channels")

    def get_or_create_category(self, guild_id: str, name: str) -> dict[str, Any]:
        channels = self.get_existing_channels(guild_id)
        for ch in channels:
            if ch.get("type") == 4 and ch.get("name") == name:
                logger.info(f"Reusing existing category: {name} ({ch['id']})")
                return ch

        logger.info(f"Creating category: {name}")
        return self._api_request(
            f"/guilds/{guild_id}/channels",
            method="POST",
            payload={"name": name, "type": 4},
        )

    def get_or_create_text_channel(
        self,
        guild_id: str,
        name: str,
        parent_id: str | None = None,
    ) -> dict[str, Any]:
        channels = self.get_existing_channels(guild_id)
        for ch in channels:
            if ch.get("type") == 0 and ch.get("name") == name:
                logger.info(f"Reusing existing text channel: {name} ({ch['id']})")
                if parent_id and ch.get("parent_id") != parent_id:
                    try:
                        self._api_request(
                            f"/channels/{ch['id']}",
                            method="PATCH",
                            payload={"parent_id": parent_id},
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to move channel {name} to category {parent_id}: {exc}")
                return ch

        logger.info(f"Creating text channel: {name} under category {parent_id}")
        payload: dict[str, Any] = {"name": name, "type": 0}
        if parent_id:
            payload["parent_id"] = parent_id
        return self._api_request(
            f"/guilds/{guild_id}/channels",
            method="POST",
            payload=payload,
        )

    def get_or_create_webhook(self, channel_id: str, webhook_name: str = "Stock-AI Reporter") -> str:
        webhooks = self._api_request(f"/channels/{channel_id}/webhooks")
        for wh in webhooks:
            if wh.get("url"):
                logger.info(f"Reusing existing webhook for channel {channel_id}: {wh['url']}")
                return str(wh["url"])

        logger.info(f"Creating webhook '{webhook_name}' for channel {channel_id}")
        res = self._api_request(
            f"/channels/{channel_id}/webhooks",
            method="POST",
            payload={"name": webhook_name},
        )
        return str(res["url"])

    def provision_all(self) -> dict[str, Any]:
        """Provisions all designated categories, channels, and webhooks."""
        guild_id = self.get_target_guild_id()
        logger.info(f"Target Guild ID: {guild_id}")

        # Category 1: 📡 AI自動配信ルーム
        cat1 = self.get_or_create_category(guild_id, "📡 AI自動配信ルーム")
        cat1_id = cat1["id"]

        ch_main = self.get_or_create_text_channel(guild_id, "☕｜aiアナリスト作戦会議", cat1_id)
        ch_fund = self.get_or_create_text_channel(guild_id, "💰｜100万円ファンド運用記", cat1_id)
        ch_turn = self.get_or_create_text_channel(guild_id, "💎｜激安回復株レーダー", cat1_id)
        ch_alert = self.get_or_create_text_channel(guild_id, "🚨｜決算・急変アラート", cat1_id)

        # Category 2: 🧠 戦略・ラボ
        cat2 = self.get_or_create_category(guild_id, "🧠 戦略・ラボ")
        cat2_id = cat2["id"]
        ch_consult = self.get_or_create_text_channel(guild_id, "💬｜ai投資相談室", cat2_id)
        ch_daytrade = self.get_or_create_text_channel(guild_id, "⚡｜当日デイトレ特化予想", cat2_id)
        ch_portfolio = self.get_or_create_text_channel(guild_id, "📊｜マイ取引ノート＆資産管理", cat2_id)
        ch_notes = self.get_or_create_text_channel(guild_id, "📝｜相場メモ・独り言", cat2_id)

        # Webhooks
        wh_main = self.get_or_create_webhook(ch_main["id"], "Stock-AI Analyst Lounge")
        wh_fund = self.get_or_create_webhook(ch_fund["id"], "Stock-AI Fund Reporter")
        wh_turn = self.get_or_create_webhook(ch_turn["id"], "Stock-AI Value Hunter")
        wh_alert = self.get_or_create_webhook(ch_alert["id"], "Stock-AI Risk Guardian")
        wh_consult = self.get_or_create_webhook(ch_consult["id"], "Stock-AI Consultation Room")
        wh_daytrade = self.get_or_create_webhook(ch_daytrade["id"], "Stock-AI Daytrade Sniper")
        wh_portfolio = self.get_or_create_webhook(ch_portfolio["id"], "Stock-AI Portfolio Ledger")
        wh_notes = self.get_or_create_webhook(ch_notes["id"], "Stock-AI Notes Reporter")

        return {
            "guild_id": guild_id,
            "categories": {
                "broadcasting": cat1_id,
                "lab": cat2_id,
            },
            "channels": {
                "main": ch_main["id"],
                "fund": ch_fund["id"],
                "turnaround": ch_turn["id"],
                "alert": ch_alert["id"],
                "consultation": ch_consult["id"],
                "daytrade": ch_daytrade["id"],
                "portfolio": ch_portfolio["id"],
                "notes": ch_notes["id"],
            },
            "webhooks": {
                "main": wh_main,
                "fund": wh_fund,
                "turnaround": wh_turn,
                "alert": wh_alert,
                "consultation": wh_consult,
                "daytrade": wh_daytrade,
                "portfolio": wh_portfolio,
                "notes": wh_notes,
            },
        }

    def send_welcome_messages(self, webhooks: dict[str, str], *, sleep_interval: float = 0.6) -> None:
        """Post welcome messages to each dedicated channel from corresponding persona."""
        now_ts = datetime.now(timezone.utc).isoformat()

        # 1. Main Analyst Lounge
        if "main" in webhooks:
            payload = {
                "username": "🎩 チーフ・ファンドマネージャー AI",
                "content": (
                    "🎩 **チーフ・ファンドマネージャー AI**\n"
                    "「全メンバー、注目！当サーバーのメイン作戦司令室（`☕｜aiアナリスト作戦会議`）の開通を宣言します。\n\n"
                    "ここでは、毎営業日の東証大引け後（16:00 JST）および寄り付き前（07:30 JST）に、私を含めた3名の専門AIアナリストが自動で集結し、当日の相場作戦会議（マルチペルソナ・ブリーフィング）をリアルタイム実況します。\n\n"
                    "各専門チャンネル（💰ファンド運用記 / 💎激安回復株 / 🚨決算急変）とも連携し、完全自律で日々のトレードを強力にサポートします。明日からの配信にご期待ください！」"
                ),
                "embeds": [
                    {
                        "title": "🚀 Stock-AI クラウド自動配信司令室 開通",
                        "description": "3名のAIアナリストが集結するメイン・ブリーフィングルーム",
                        "color": 3066993,  # Emerald Green
                        "fields": [
                            {
                                "name": "☕ 作戦会議 (メイン)",
                                "value": "毎朝・毎夕の3人掛け合いトーク実況",
                                "inline": True,
                            },
                            {
                                "name": "💰 ファンド運用記",
                                "value": "仮想100万円ファンドの資産推移・約定ログ",
                                "inline": True,
                            },
                            {
                                "name": "💎 激安回復株",
                                "value": "東証プライムから発掘する70点超えお宝株",
                                "inline": True,
                            },
                            {
                                "name": "🚨 決算・急変アラート",
                                "value": "決算跨ぎの地雷リスク事前警告",
                                "inline": True,
                            },
                        ],
                        "footer": {"text": "Stock AI Multi-Room Broadcasting System"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["main"], payload)
            time.sleep(sleep_interval)

        # 2. Fund Room
        if "fund" in webhooks:
            payload = {
                "username": "🎩 チーフ・ファンドマネージャー AI",
                "content": (
                    "🎩 **チーフ・ファンドマネージャー AI**\n"
                    "「このチャンネルは『仮想100万円ファンド（Overnight Paper Trader）』の専用運用記録ルームです。\n\n"
                    "元本100万円からスタートし、寄り付きギャップアップ狙いのクオンツ戦略に基づいて、毎朝のエントリー銘柄・保有ポジション・約定結果・資産残高の推移を詳細に記録します。\n"
                    "感情を排した規律あるクオンツ運用で、どこまで資産を拡大できるかを実証していきます！」"
                ),
                "embeds": [
                    {
                        "title": "💰 仮想100万円ファンド 運用記チャンネル始動",
                        "description": "Overnight戦略に基づくリアルトレードシミュレーション記録",
                        "color": 3066993,  # Emerald Green
                        "fields": [
                            {"name": "🎯 運用戦略", "value": "Overnight Q5 (寄り付きギャップアップ狙い)", "inline": True},
                            {"name": "💵 初期元本", "value": "1,000,000 円", "inline": True},
                            {"name": "📊 配信内容", "value": "毎日の総資産、累積リターン、勝率、エントリー判定", "inline": False},
                        ],
                        "footer": {"text": "Stock AI Paper Trader Log Room"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["fund"], payload)
            time.sleep(sleep_interval)

        # 3. Turnaround Room
        if "turnaround" in webhooks:
            payload = {
                "username": "💎 バリューハンター AI (中長期・激安株担当)",
                "content": (
                    "💎 **バリューハンター AI (中長期・激安株担当)**\n"
                    "「バリューハンターの特設ラボへようこそ！✨\n\n"
                    "短期的な市場の波乱に惑わされず、東証プライム主要銘柄の中から『売り叩かれすぎて割安になった極上のお宝銘柄』を徹底追跡する専用ルームです。\n"
                    "52週高値からの下落率、PBR水準、そしてテクニカル反転シグナルを掛け合わせた【割安スコア】を毎日算出し、特に70点以上の超割安・底打ち銘柄を速報します！」"
                ),
                "embeds": [
                    {
                        "title": "💎 Value & Turnaround 激安回復株レーダー始動",
                        "description": "東証プライム主要銘柄から下落率・PBR・反転スコアで厳選",
                        "color": 10181046,  # Amethyst Purple
                        "fields": [
                            {"name": "🔍 監視対象", "value": "東証プライム主要50銘柄", "inline": True},
                            {"name": "⚡ 注目シグナル", "value": "割安スコア70点以上の超割安・底打ち反転圏", "inline": True},
                            {"name": "📈 配信内容", "value": "割安ランキングTop 5、PBR、下落率、反転ステータス", "inline": False},
                        ],
                        "footer": {"text": "Stock AI Value & Turnaround Radar"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["turnaround"], payload)
            time.sleep(sleep_interval)

        # 4. Alert Room
        if "alert" in webhooks:
            payload = {
                "username": "🛡️ リスク管理AI (ガーディアン)",
                "content": (
                    "🛡️ **リスク管理AI (ガーディアン)**\n"
                    "「リスク管理担当のガーディアンです。\n"
                    "相場で最も恐ろしいのは、予期せぬ『決算発表跨ぎの急落地雷』を踏むことです。\n\n"
                    "どれほどクオンツ確率が高くても、決算直前のエントリーはギャンブルになります。この部屋では、注目銘柄の直近決算予定や突発的な地雷フラグ（`⚠️決算警戒`）をいち早く検知し、皆様の大切な資産を徹底防衛します！」"
                ),
                "embeds": [
                    {
                        "title": "🚨 決算・急変アラート ルーム稼働",
                        "description": "決算跨ぎ急落リスクの事前回避と注意喚起",
                        "color": 15158332,  # Crimson Red
                        "fields": [
                            {"name": "🛡️ 役割", "value": "決算発表前の急変動リスク検知 & 事前警告", "inline": True},
                            {"name": "⚠️ 判定基準", "value": "直近7日以内の決算予定銘柄を自動検知", "inline": True},
                            {"name": "📢 配信内容", "value": "決算警戒リスト、地雷回避ガイダンス", "inline": False},
                        ],
                        "footer": {"text": "Stock AI Risk Guardian | 防御最優先"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["alert"], payload)
            time.sleep(sleep_interval)

        # 5. Consultation Room
        if "consultation" in webhooks:
            payload = {
                "username": "☕ AI投資相談室 コンシェルジュ",
                "content": (
                    "☕ **【AI投資相談室へようこそ！】**\n\n"
                    "ここでは、3人のAIアナリスト（🎩チーフFM、⚡短期スナイパー、💎バリューハンター）に対して、投資に関する疑問や戦略を自由に相談できます！\n\n"
                    "💡 **こんな質問にお答えします**:\n"
                    "・**複数銘柄の比較**: `トヨタと日産どっちがいい？`, `ソニーと任天堂を比較して`\n"
                    "・**おすすめ・資金相談**: `今おすすめの株は？`, `30万円で仕込むなら？`, `短期で勝てるやつ教えて`\n"
                    "・**相場観・売買タイミング**: `今の相場どう思う？`, `買い時？`, `今の相場で気をつけることは？`\n"
                    "・**個別銘柄の深掘り**: `トヨタって今買い時かな？`, `日産はなぜ安いの？`\n\n"
                    "気になることを何でも気軽に話しかけてみてください！"
                ),
                "embeds": [
                    {
                        "title": "💬 AI投資相談室 (Interactive Consultation Room) 開通",
                        "description": "3人の専門AIアナリストがデータと戦略に基づいて親身にお答えします。",
                        "color": 3447003,
                        "fields": [
                            {"name": "⚖️ 銘柄比較", "value": "短期上昇確率 × 中長期割安度の徹底比較", "inline": True},
                            {"name": "💰 資金配分", "value": "予算（30万、50万等）に合わせた単元株提案", "inline": True},
                            {"name": "📊 相場診断", "value": "日経平均・為替・ファンド動向から地合い診断", "inline": True},
                        ],
                        "footer": {"text": "Stock AI Real-Time Consultation Engine"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["consultation"], payload)
            time.sleep(sleep_interval)

        # 6. Daytrade Sniper Room
        if "daytrade" in webhooks:
            payload = {
                "username": "⚡ デイトレ特化スナイパー AI",
                "content": (
                    "⚡ **【当日デイトレ特化予想チャンネル 開通！】**\n\n"
                    "「お待たせしました！中長期・バリューの話題を一切排除した **【当日デイトレ特化予想チャンネル（`⚡｜当日デイトレ特化予想`）】** を開通します🎯\n\n"
                    "ここでは、東証プライム主力50銘柄を中心に **『その日（09:00寄付き〜15:30大引け）上がるか・下がるか』だけに100%特化** してリアルタイムに診断します。\n\n"
                    "💡 **使い方・質問例**:\n"
                    "・**個別銘柄の当日診断**: `フジクラ`, `7012`, `キオクシア`, `レーザーテック`, `トヨタどう？`\n"
                    "  → **当日上昇確率 %**、**明確な白黒判定（🟢今日上がる / 🔴今日下がる / ⚪レンジ様子見）**、**本日想定レンジ（上値ターゲット・損切り価格）**、**場中デイトレ戦術** を即答！\n"
                    "・**今日上がる株・下がる株ランキング**: `今日なにが上がる？`, `ランキング`, `上がる株`, `下がる株`\n"
                    "  → 50銘柄から **本日上昇確率 TOP 5** ＆ **下落警戒 TOP 5** を即座にスキャンして返答！\n\n"
                    "今日勝つためのシャープな武器としてガンガン使い倒してください！」"
                ),
                "embeds": [
                    {
                        "title": "⚡ 当日デイトレ特化予想 (1-Day Intraday Sniper) 開通",
                        "description": "09:00寄付き〜15:30大引けの『今日上がるか下がるか』だけに100%特化した最速レスポンス",
                        "color": 15844367,  # Gold
                        "fields": [
                            {"name": "🎯 対象時間軸", "value": "当日 09:00 寄付き 〜 15:30 大引け（翌日持ち越しゼロ）", "inline": True},
                            {"name": "📈 提示データ", "value": "当日上昇確率 % ＋ 🟢今日上がる / 🔴今日下がる", "inline": True},
                            {"name": "🛡️ 執行規律", "value": "本日想定レンジ（上値目標 / 損切りライン）を完全数値化", "inline": True},
                            {"name": "💬 入力例", "value": "`7012`, `フジクラ`, `今日なにが上がる？`, `ランキング`", "inline": False},
                        ],
                        "footer": {"text": "Stock AI 1-Day Intraday Sniper Engine"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["daytrade"], payload)
            time.sleep(sleep_interval)

        # 7. Portfolio Ledger & Asset Manager Room (📊｜マイ取引ノート＆資産管理)
        if "portfolio" in webhooks:
            payload = {
                "username": "📊 マイ資産管理コンシェルジュ AI",
                "content": (
                    "📊 **【マイ取引ノート＆個人資産管理チャンネル 開通！】**\n\n"
                    "「ご自身の実際の日本株トレードを、メモ感覚で自然に記録・管理できる専用チャンネルを開通しました📝✨\n\n"
                    "難しいコマンドや形式は一切不要！『川重 100株 5000円で買った』『トヨタ 100株 2850円で売った』とつぶやくだけで、自動で取得単価・保有株数・実現損益を計算して台帳に記帳します。\n\n"
                    "さらに、記帳した直後に **最新AIクオンツ予測（明朝Overnight上昇確率・日中デイトレ判定・想定上値ターゲット・推奨損切りライン）** をその場で即答フィードバック！\n"
                    "保有中の銘柄に暴落や決算の危険が迫った時は **『⚠️AIガードマン急落アラート』** で即座に警告を発令します🛡️\n\n"
                    "💡 **入力例（コピペですぐ試せます）**:\n"
                    "・**買い記帳**: `川重 100株 5000円で買った`, `トヨタ 200株 2750円 買い`, `フジクラ 100株 @4800 購入`\n"
                    "・**売り記帳**: `トヨタ 100株 2850円で売った`, `7012 100株 5500円 利確`, `川重 全株 5100円で損切り`\n"
                    "・**収支・資産確認**: `収支`, `ポートフォリオ`, `保有株`, `履歴`\n"
                    "・**保有株の危険度診断**: `今日大丈夫？`, `診断`, `リスク`\n"
                    "・**書き間違いの取消**: `最後の取引を取り消し`, `取り消し`\n\n"
                    "今日からのリアルトレードの相棒として、ぜひお気軽にご活用ください！」"
                ),
                "embeds": [
                    {
                        "title": "📊 マイ取引ノート＆個人資産管理 使い方ガイド",
                        "description": "自然な日本語で話しかけるだけで、全自動で台帳記帳・損益計算・AIリスク診断を実施",
                        "color": 3066993,  # Emerald Green
                        "fields": [
                            {"name": "🛒 買い記帳", "value": "`川重 100株 5000円で買った`\n（即座にAI予測＆ターゲット価格を返信）", "inline": True},
                            {"name": "💰 売り記帳", "value": "`トヨタ 100株 2850円 利確`\n（今回の確定損益と累計確定損益を即答）", "inline": True},
                            {"name": "📈 総合収支照会", "value": "`収支` または `保有株`\n（現在の評価額・含み損益・通算勝率を表示）", "inline": True},
                            {"name": "🛡️ 暴落・急落診断", "value": "`今日大丈夫？` または `診断`\n（保有株の下落確率・決算地雷・過熱度を一括診断）", "inline": True},
                            {"name": "↩️ 直前取消", "value": "`最後の取引を取り消し`\n（直前の入力を安全にロールバック）", "inline": True},
                            {"name": "🔒 データの安全性", "value": "専用ローカルDB（SQLite）に永続保存され、外部漏洩なし", "inline": True},
                        ],
                        "footer": {"text": "Stock AI Personal Portfolio & Risk Guardian Engine"},
                        "timestamp": now_ts,
                    }
                ],
            }
            self._post_webhook(webhooks["portfolio"], payload)

    @staticmethod
    def save_webhooks_to_env_file(webhooks: dict[str, str]) -> None:
        """Update local .env file with newly provisioned webhooks."""
        env_path = PROJECT_ROOT / ".env"
        if not env_path.exists():
            return
        lines = env_path.read_text(encoding="utf-8").splitlines()
        new_lines: list[str] = []
        has_daytrade = False
        has_portfolio = False
        has_json = False
        daytrade_url = webhooks.get("daytrade")
        portfolio_url = webhooks.get("portfolio")
        json_str = json.dumps(webhooks, ensure_ascii=False)

        for line in lines:
            if line.startswith("STOCK_AI_WEBHOOK_DAYTRADE="):
                if daytrade_url:
                    new_lines.append(f"STOCK_AI_WEBHOOK_DAYTRADE={daytrade_url}")
                    has_daytrade = True
            elif line.startswith("STOCK_AI_WEBHOOK_PORTFOLIO="):
                if portfolio_url:
                    new_lines.append(f"STOCK_AI_WEBHOOK_PORTFOLIO={portfolio_url}")
                    has_portfolio = True
            elif line.startswith("STOCK_AI_CHANNEL_WEBHOOKS="):
                new_lines.append(f"STOCK_AI_CHANNEL_WEBHOOKS={json_str}")
                has_json = True
            else:
                new_lines.append(line)

        if not has_daytrade and daytrade_url:
            new_lines.append(f"STOCK_AI_WEBHOOK_DAYTRADE={daytrade_url}")
        if not has_portfolio and portfolio_url:
            new_lines.append(f"STOCK_AI_WEBHOOK_PORTFOLIO={portfolio_url}")
        if not has_json:
            new_lines.append(f"STOCK_AI_CHANNEL_WEBHOOKS={json_str}")

        env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        logger.info("Updated .env file with daytrade & portfolio webhooks.")


    @staticmethod
    def _post_webhook(webhook_url: str, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            webhook_url,
            data=data,
            headers={"Content-Type": "application/json", "User-Agent": "StockAI-Provisioner/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                if resp.getcode() not in (200, 204):
                    logger.warning(f"Webhook return code {resp.getcode()} for {webhook_url}")
        except Exception as exc:
            logger.error(f"Failed to post welcome message to {webhook_url}: {exc}")
