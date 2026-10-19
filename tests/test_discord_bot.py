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
        assert "トヨタ" in res_stock[0]["content"]

        # Mention stripped query (e.g. <@1561416539687751771> トヨタ)
        res_mention = bot._handle_user_message("<@1561416539687751771> トヨタ")
        assert res_mention is not None
        assert len(res_mention) == 3
        assert "トヨタ" in res_mention[0]["content"]

        # Empty content (Message Content Intent is OFF) -> returns intent guide & ranking fallback
        res_empty = bot._handle_user_message("")
        assert res_empty is not None
        assert "Message Content Intent" in res_empty[0]["content"]
        assert len(res_empty) >= 2  # notice + ranking payloads

        # Unmatched query with mention keyword
        res_unknown = bot._handle_user_message("この銘柄の株価分析はどう？")
        assert res_unknown is not None
        assert any(k in res_unknown[0]["content"] for k in ["特定できませんでした", "銘柄の指定", "銘柄コード", "名称", "分析"])

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


def test_extract_all_tickers() -> None:
    from stock_ai.cloud.discord_bot import extract_all_tickers

    # Multi-ticker extraction
    res1 = extract_all_tickers("トヨタと日産どっちがいい？")
    assert len(res1) == 2
    assert res1[0][0] == "7203.T"
    assert res1[1][0] == "7201.T"

    res2 = extract_all_tickers("ソニーと任天堂を比較して")
    assert len(res2) == 2
    assert res2[0][0] == "6758.T"
    assert res2[1][0] == "7974.T"

    res3 = extract_all_tickers("8058と8031")
    assert len(res3) == 2
    assert res3[0][0] == "8058.T"
    assert res3[1][0] == "8031.T"


def test_build_comparison_dialogue_payloads() -> None:
    from stock_ai.cloud.discord_bot import build_comparison_dialogue_payloads

    tickers = [("7203.T", "トヨタ自動車"), ("7201.T", "日産自動車")]
    payloads = build_comparison_dialogue_payloads(tickers, query="トヨタと日産どっちがいい？")

    assert len(payloads) == 3
    # Chief FM with comparison embed
    assert "チーフ・ファンドマネージャー" in payloads[0]["username"]
    assert len(payloads[0]["embeds"]) == 1
    assert "銘柄直接比較" in payloads[0]["embeds"][0]["title"]
    # Sniper & Hunter
    assert "短期スナイパー" in payloads[1]["username"]
    assert "バリューハンター" in payloads[2]["username"]


def test_build_recommendation_dialogue_payloads() -> None:
    from stock_ai.cloud.discord_bot import build_recommendation_dialogue_payloads

    payloads = build_recommendation_dialogue_payloads("今から30万円で仕込むなら何がいい？")
    assert len(payloads) == 3
    assert "チーフ・ファンドマネージャー" in payloads[0]["username"]
    assert len(payloads[0]["embeds"]) == 1
    assert "銘柄選定・資金配分プラン" in payloads[0]["embeds"][0]["title"]
    assert "予算 30万円" in payloads[0]["embeds"][0]["title"]


def test_build_market_outlook_dialogue_payloads() -> None:
    from stock_ai.cloud.discord_bot import build_market_outlook_dialogue_payloads

    payloads = build_market_outlook_dialogue_payloads("今の相場で気をつけることは？")
    assert len(payloads) == 3
    assert "チーフ・ファンドマネージャー" in payloads[0]["username"]
    assert len(payloads[0]["embeds"]) == 1
    assert "相場観・投資戦略ブリーフィング" in payloads[0]["embeds"][0]["title"]
    assert "日経平均" in payloads[0]["content"]

    payloads_timing = build_market_outlook_dialogue_payloads("利確や損切りの基準は？")
    assert "損切り" in payloads_timing[1]["content"]


def test_polling_bot_consultation_routing() -> None:
    with patch.object(DiscordPollingBot, "__init__", lambda self: None):
        bot = DiscordPollingBot()

        # 1. Comparison consultation
        res_comp = bot._handle_user_message("トヨタと日産どっちがいい？")
        assert res_comp is not None
        assert len(res_comp) == 3
        title_comp = res_comp[0]["embeds"][0]["title"]
        assert any(k in title_comp for k in ["比較", "対決", "トヨタ", "日産", "相談", "分析", "銘柄"])

        # 2. Recommendation / Budget consultation
        res_rec = bot._handle_user_message("30万円で仕込むなら何がいい？")
        assert res_rec is not None
        assert len(res_rec) == 3
        title_rec = res_rec[0]["embeds"][0]["title"]
        assert any(k in title_rec for k in ["選定", "資金", "プラン", "おすすめ", "相談", "ポートフォリオ", "投資", "戦略", "仕込", "30万", "銘柄"])

        # 3. Market outlook consultation
        res_market = bot._handle_user_message("今の相場で気をつけることは？")
        assert res_market is not None
        assert len(res_market) == 3
        title_market = res_market[0]["embeds"][0]["title"]
        assert any(k in title_market for k in ["相場", "戦略", "ブリーフィング", "相談", "市場", "注意", "リスク"])

        # 4. Single ticker deep dive with intent
        res_deep = bot._handle_user_message("日産は今買い時かな？")
        assert res_deep is not None
        assert len(res_deep) == 3
        assert "日産自動車" in res_deep[0]["content"]

        # 5. Holding / Exit strategy consultation (The exact user inquiry!)
        res_exit1 = bot._handle_user_message("今持っている株だけど評価して")
        assert res_exit1 is not None
        assert len(res_exit1) == 3
        # Either LLM synthesis or Method B exit strategy sheet
        has_exit_content = (
            "売却・ホールド判定チェックシート" in str(res_exit1[0])
            or "売買判断" in str(res_exit1[0])
            or "出口戦略" in str(res_exit1[0])
            or "損切り" in str(res_exit1[0])
        )
        assert has_exit_content

        res_exit2 = bot._handle_user_message("もっと正確にこの株は売った方がいいとかこれはまだとっておくべきとか")
        assert res_exit2 is not None
        assert len(res_exit2) == 3


def test_search_ticker_dynamically_and_expanded_aliases() -> None:
    from stock_ai.cloud.discord_bot import extract_all_tickers, search_ticker_dynamically

    # Test newly added aliases
    assert search_ticker_dynamically("マツダ") == ("7261.T", "マツダ")
    assert search_ticker_dynamically("楽天") == ("4755.T", "楽天グループ")
    assert search_ticker_dynamically("カプコン") == ("9697.T", "カプコン")
    assert search_ticker_dynamically("サンリオ") == ("8136.T", "サンリオ")
    assert search_ticker_dynamically("イオン") == ("8267.T", "イオン")
    assert search_ticker_dynamically("レーザーテック") == ("6920.T", "レーザーテック")
    assert search_ticker_dynamically("ホンダ") == ("7267.T", "本田技研工業")
    assert search_ticker_dynamically("三菱重工") == ("7011.T", "三菱重工業")

    # Test extract_all_tickers with particle tokenization
    extracted = extract_all_tickers("マツダとカプコンどっちがいい？")
    assert len(extracted) == 2
    tickers = [t[0] for t in extracted]
    assert "7261.T" in tickers
    assert "9697.T" in tickers


def test_build_holding_exit_strategy_dialogue_payloads() -> None:
    from stock_ai.cloud.discord_bot import build_holding_exit_strategy_dialogue_payloads

    payloads = build_holding_exit_strategy_dialogue_payloads("もっと正確にこの株は売った方がいいとかこれはまだとっておくべきとか")
    assert len(payloads) == 3
    assert "チーフ・ファンドマネージャー" in payloads[0]["username"]
    assert len(payloads[0]["embeds"]) == 1
    assert "損切り" in payloads[1]["content"] or "ロスカット" in payloads[1]["content"]
    assert "保有" in payloads[2]["content"] or "ホールド" in payloads[2]["content"]

