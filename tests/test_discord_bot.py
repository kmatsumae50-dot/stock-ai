"""Tests for Discord interactive polling bot and single ticker analysis."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from stock_ai.cloud.discord_bot import (
    DiscordPollingBot,
    analyze_single_ticker,
    build_help_payload,
    build_ranking_payloads,
    build_single_ticker_dialogue_payloads,
    resolve_ticker,
)


def test_resolve_ticker_numeric_code() -> None:
    # Half-width and full-width 4-digit codes
    assert resolve_ticker("7203") == ("7203.T", "トヨタ自動車")
    assert resolve_ticker("7203.T") == ("7203.T", "トヨタ自動車")
    assert resolve_ticker("７２０３") == ("7203.T", "トヨタ自動車")
    assert resolve_ticker("7974") == ("7974.T", "任天堂")
    assert resolve_ticker("6758.Tの予想教えて") == ("6758.T", "ソニーグループ")


def test_resolve_ticker_company_names() -> None:
    # Major Japanese company names & aliases
    assert resolve_ticker("トヨタ") == ("7203.T", "トヨタ自動車")
    assert resolve_ticker("任天堂について") == ("7974.T", "任天堂")
    assert resolve_ticker("日産の株価分析") == ("7201.T", "日産自動車")
    assert resolve_ticker("IHI") == ("7013.T", "IHI")
    assert resolve_ticker("ソフトバンクグループ") == ("9984.T", "ソフトバンクグループ")
    assert resolve_ticker("三菱商事") == ("8058.T", "三菱商事")


def test_resolve_ticker_unknown() -> None:
    assert resolve_ticker("こんにちは") is None
    assert resolve_ticker("明日の天気") is None
    assert resolve_ticker("") is None


def test_analyze_single_ticker_cached() -> None:
    # 7201.T (Nissan) exists in recent autonomous snapshot
    result = analyze_single_ticker("7201")
    assert result is not None
    assert result["ticker"] == "7201.T"
    assert "日産" in result["name"]
    assert "close" in result
    assert "p_up" in result
    assert "bargain_score" in result
    assert "chief_conclusion" in result
    assert "sniper_comment" in result
    assert "hunter_comment" in result
    assert 0.0 <= result["p_up"] <= 1.0
    assert 0.0 <= result["bargain_score"] <= 100.0


def test_build_single_ticker_dialogue_payloads() -> None:
    sample_analysis = {
        "ticker": "7203.T",
        "name": "トヨタ自動車",
        "sector": "自動車",
        "close": 3050.0,
        "pct_change": 0.015,
        "p_up": 0.54,
        "q_band": "Q4",
        "q_label": "強気先行帯",
        "earnings_risk": False,
        "earnings_date": "2026-11-06",
        "bargain_score": 62.5,
        "turnaround_status": "🚀 始動サイン",
        "drawdown_52w": -0.15,
        "rsi_14": 48.0,
        "pbr": 0.95,
        "chief_conclusion": "🎯 買い優位のシグナル点灯",
        "sniper_comment": "明朝の寄り付き上昇確率は高めです。",
        "hunter_comment": "割安水準で反転が期待できます。",
    }

    payloads = build_single_ticker_dialogue_payloads(sample_analysis)
    assert len(payloads) == 3

    # Chief FM
    assert "チーフ・ファンドマネージャー" in payloads[0]["username"]
    assert "トヨタ自動車" in payloads[0]["content"]
    assert len(payloads[0]["embeds"]) == 1

    # Sniper
    assert "短期スナイパー" in payloads[1]["username"]
    assert "Overnight" in payloads[1]["embeds"][0]["title"]

    # Value Hunter
    assert "バリューハンター" in payloads[2]["username"]
    assert "Turnaround" in payloads[2]["embeds"][0]["title"]


def test_build_help_and_ranking_payloads() -> None:
    help_p = build_help_payload()
    assert "利用ガイド" in help_p["content"]
    assert len(help_p["embeds"]) == 1

    ranking_p = build_ranking_payloads()
    assert len(ranking_p) >= 1
    assert "ランキング速報" in ranking_p[0]["content"]


def test_polling_bot_message_routing() -> None:
    with patch.object(DiscordPollingBot, "__init__", lambda self: None):
        bot = DiscordPollingBot()

        # Help
        res_help = bot._handle_user_message("ヘルプ")
        assert res_help is not None
        assert "利用ガイド" in res_help[0]["content"]

        # Ranking
        res_rank = bot._handle_user_message("ランキング")
        assert res_rank is not None
        assert "ランキング速報" in res_rank[0]["content"]

        # Ticker query
        res_stock = bot._handle_user_message("7203")
        assert res_stock is not None
        assert len(res_stock) == 3
        assert "トヨタ自動車" in res_stock[0]["content"]

        # Mention stripped query (e.g. <@1561416539687751771> トヨタ)
        res_mention = bot._handle_user_message("<@1561416539687751771> トヨタ")
        assert res_mention is not None
        assert len(res_mention) == 3
        assert "トヨタ自動車" in res_mention[0]["content"]

        # Empty content (Message Content Intent is OFF) -> returns intent guide & ranking fallback
        res_empty = bot._handle_user_message("")
        assert res_empty is not None
        assert "Message Content Intent" in res_empty[0]["content"]
        assert len(res_empty) >= 2  # notice + ranking payloads

        # Unmatched query with mention keyword
        res_unknown = bot._handle_user_message("この銘柄の株価分析はどう？")
        assert res_unknown is not None
        assert "特定できませんでした" in res_unknown[0]["content"]

        # Completely unrelated text (no response needed)
        assert bot._handle_user_message("おはようございます") is None


def test_polling_bot_poll_once_mock() -> None:
    with patch.object(DiscordPollingBot, "__init__", lambda self: None):
        bot = DiscordPollingBot()
        bot.monitored_channels = [{"id": "12345", "name": "ai-test"}]
        bot.last_seen_message_id = {"12345": "100"}
        bot.channel_to_webhook = {}

        # Mock API requests:
        # Message 1: bot's own message (should be ignored)
        # Message 2: user asking for 'トヨタ'
        mock_messages = [
            {"id": "101", "content": "bot message", "author": {"id": "bot_id", "bot": True}},
            {"id": "102", "content": "トヨタ", "author": {"id": "user_id", "bot": False}},
        ]

        bot._api_request = MagicMock(return_value=mock_messages)
        bot.send_reply = MagicMock()

        processed = bot.poll_once()
        assert processed == 1
        assert bot.last_seen_message_id["12345"] == "102"
        bot.send_reply.assert_called_once()
