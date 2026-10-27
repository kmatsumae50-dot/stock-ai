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
    # New code system (2024+, 3 digits + 1 letter like 285A)
    assert resolve_ticker("285A") == ("285A.T", "キオクシアホールディングス")
    assert resolve_ticker("285A.T") == ("285A.T", "キオクシアホールディングス")
    assert resolve_ticker("２８５Ａ") == ("285A.T", "キオクシアホールディングス")


def test_resolve_ticker_company_names() -> None:
    # Major Japanese company names & aliases
    assert resolve_ticker("トヨタ") == ("7203.T", "トヨタ自動車")
    assert resolve_ticker("任天堂について") == ("7974.T", "任天堂")
    assert resolve_ticker("日産の株価分析") == ("7201.T", "日産自動車")
    assert resolve_ticker("IHI") == ("7013.T", "IHI")
    assert resolve_ticker("ソフトバンクグループ") == ("9984.T", "ソフトバンクグループ")
    assert resolve_ticker("三菱商事") == ("8058.T", "三菱商事")
    assert resolve_ticker("キオクシア") == ("285A.T", "キオクシアホールディングス")
    assert resolve_ticker("東京メトロ") == ("9023.T", "東京地下鉄")
    assert resolve_ticker("ソシオネクスト") == ("6526.T", "ソシオネクスト")
    assert resolve_ticker("さくらインターネット") == ("3778.T", "さくらインターネット")
    assert resolve_ticker("フジクラ") == ("5803.T", "フジクラ")


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


def test_build_daytrade_sniper_and_ranking_payloads() -> None:
    from stock_ai.cloud.discord_bot import (
        build_daytrade_ranking_payloads,
        build_daytrade_sniper_payloads,
    )

    # 1. Daytrade single sniper
    payloads = build_daytrade_sniper_payloads("7012.T", name="川崎重工業")
    assert len(payloads) == 1
    p = payloads[0]
    assert "デイトレ特化スナイパー" in p["username"]
    assert "川崎重工業" in p["content"]
    assert len(p["embeds"]) == 1
    embed = p["embeds"][0]
    assert "当日デイトレ特化予想" in embed["title"]
    # Check that mid-long term value terms are NOT present
    assert "PBR" not in embed["fields"][0]["value"]
    assert "割安" not in embed["fields"][0]["value"]

    # 2. Daytrade ranking
    rank_payloads = build_daytrade_ranking_payloads()
    assert len(rank_payloads) == 1
    rp = rank_payloads[0]
    assert "デイトレ特化スナイパー" in rp["username"]
    assert len(rp["embeds"]) == 1
    rembed = rp["embeds"][0]
    assert "本日デイトレ特化予想ランキング" in rembed["title"]
    assert len(rembed["fields"]) >= 2
    # Ensure field values never exceed Discord 1024-char limit
    for f in rembed["fields"]:
        assert len(f["value"]) <= 1024


def test_daytrade_channel_and_keyword_routing() -> None:
    from stock_ai.cloud.discord_bot import DiscordPollingBot

    with patch.object(DiscordPollingBot, "__init__", lambda self: None):
        bot = DiscordPollingBot()
        bot.monitored_channels = [
            {"id": "12345", "name": "⚡｜当日デイトレ特化予想"},
            {"id": "67890", "name": "☕｜aiアナリスト作戦会議"},
        ]

        # 1. In daytrade channel, ticker query gets Daytrade Sniper
        res_ch = bot._handle_user_message("7012", msg={"channel_id": "12345"})
        assert res_ch is not None
        assert len(res_ch) == 1
        assert "デイトレ特化スナイパー" in res_ch[0]["username"]

        # 2. In generic channel, "今日" / "デイトレ" keyword triggers Daytrade Sniper
        res_kw = bot._handle_user_message("今日の7012どう？", msg={"channel_id": "67890"})
        assert res_kw is not None
        assert len(res_kw) == 1
        assert "デイトレ特化スナイパー" in res_kw[0]["username"]

        # 3. Ranking query in daytrade channel
        res_rank = bot._handle_user_message("今日なにが上がる？", msg={"channel_id": "12345"})
        assert res_rank is not None
        assert "デイトレ特化スナイパー" in res_rank[0]["username"]
        assert "本日デイトレ特化予想ランキング" in res_rank[0]["embeds"][0]["title"]


def test_daytrade_premarket_forecast_incorporates_yesterday_reflection():
    """Verify build_premarket_daytrade_forecast_payloads injects yesterday's reflection and closed-loop feedback."""
    from stock_ai.cloud.discord_bot import build_premarket_daytrade_forecast_payloads

    mock_prev_scorecard = {
        "date": "2026-10-26",
        "total_trades": 3,
        "wins": 2,
        "flats": 0,
        "losses": 1,
        "win_rate_pct": 66.7,
        "avg_return_pct": 1.42,
        "total_virtual_pnl_yen": 42500,
        "trades": [
            {
                "ticker": "7011.T",
                "name": "三菱重工",
                "predicted_waveform": "🚀GU朝押し型",
                "result_status": "WIN",
                "result_code": "TARGET_HIT",
                "result_badge": "🎯 利確目標到達",
                "net_return_pct": 3.15,
            },
            {
                "ticker": "5803.T",
                "name": "フジクラ",
                "predicted_waveform": "📈窓埋め反発型",
                "result_status": "WIN",
                "result_code": "WIN_CLOSE",
                "result_badge": "🟢 予想通り上昇",
                "net_return_pct": 1.28,
            },
            {
                "ticker": "8035.T",
                "name": "東エレク",
                "predicted_waveform": "🚀ORBブレイク型",
                "result_status": "LOSS",
                "result_code": "LOSS_CLOSE",
                "result_badge": "🔴 寄り天失速",
                "net_return_pct": -0.18,
            },
        ],
    }

    mock_rankings = {
        "top_up": [
            {
                "ticker": "7011.T",
                "name": "三菱重工",
                "probability_up": 65.0,
                "expected_return_bps": 120.0,
                "best_entry_zone": "寄付直後",
                "target_final_price": 2500.0,
                "target_final_pct": 3.5,
            },
            {
                "ticker": "5803.T",
                "name": "フジクラ",
                "probability_up": 60.0,
                "expected_return_bps": 90.0,
                "best_entry_zone": "寄付直後",
                "target_final_price": 4500.0,
                "target_final_pct": 2.8,
            },
            {
                "ticker": "8035.T",
                "name": "東エレク",
                "probability_up": 58.0,
                "expected_return_bps": 80.0,
                "best_entry_zone": "寄付直後",
                "target_final_price": 26000.0,
                "target_final_pct": 2.2,
            },
        ],
        "top_down": [],
        "systemic_stability_badge": "🛡️ 市場システミック安定度: 正常・分散安定",
    }

    with patch("stock_ai.models.intraday.evaluate_previous_daytrade_scorecard", return_value=mock_prev_scorecard), \
         patch("stock_ai.models.intraday.get_daily_daytrade_rankings", return_value=mock_rankings), \
         patch("stock_ai.models.intraday.predict_daytrade_sniper", return_value=None):
        payloads = build_premarket_daytrade_forecast_payloads(summary_data={"date": "2026-10-27"})

    assert len(payloads) >= 1
    overview = payloads[0]["content"]
    assert "☀️ **【東証開場前・本日の1日値動きシミュレーション＆デイトレ予想】(2026-10-27)**" in overview
    assert "📋 **【前日予想の答え合わせ見返し (10/26実績)】**" in overview
    assert "推奨トップ3成績: **2勝1敗** (勝率 **66.7%** ｜ 平均リターン **+1.42%**)" in overview
    assert "1. `7011` **三菱重工**" in overview
    assert "🎯 利確目標到達" in overview
    assert "2. `5803` **フジクラ**" in overview
    assert "🟢 予想通り上昇" in overview
    assert "3. `8035` **東エレク**" in overview
    assert "🔴 寄り天失速" in overview
    assert "🧠 **【前日からのAI学習・本日の補正】**" in overview
    assert "前日の寄り天失速(TD誤差" in overview
    assert "「高Jump Ratio(フェイク急騰)」への警戒ペナルティを強化" in overview


def test_daytrade_scorecard_payloads_structure_and_hedge_weights():
    """Verify scorecard payload includes Hedge weights and 4-stage scoring rules."""
    from stock_ai.cloud.discord_bot import build_daytrade_scorecard_payloads

    mock_scorecard = {
        "date": "2026-10-26",
        "total_trades": 2,
        "wins": 1,
        "flats": 0,
        "losses": 1,
        "win_rate_pct": 50.0,
        "avg_return_pct": 0.85,
        "total_virtual_pnl_yen": 25000,
        "mvp": {
            "ticker": "7011.T",
            "name": "三菱重工業",
            "virtual_pnl_100_shares": 50000,
            "waveform_verdict": "🎯 波形的中",
        },
        "trades": [
            {
                "ticker": "7011.T",
                "name": "三菱重工業",
                "result_badge": "🎯 利確目標到達",
                "waveform_verdict": "🎯 波形的中",
                "virtual_pnl_100_shares": 50000,
                "net_return_pct": 2.50,
                "actual_open": 2000.0,
                "exit_price": 2050.0,
                "actual_high": 2060.0,
                "actual_low": 1990.0,
                "predicted_waveform": "GU朝押し型",
                "actual_waveform": "前場高値引け陽線",
            }
        ],
    }

    payloads = build_daytrade_scorecard_payloads(mock_scorecard)
    assert len(payloads) == 1
    embed = payloads[0]["embeds"][0]
    field_names = [f["name"] for f in embed["fields"]]

    assert any("AI波形パターン信頼度ウェイト一覧 (Hedge Weights)" in name for name in field_names)
    assert any("答え合わせ採点ルール（4段階厳密判定・往復20bps控除後）" in name for name in field_names)


def test_daytrade_scorecard_routing_keywords():
    """Verify '見返し', '答え合わせ', '通信簿' route to daytrade scorecard payload."""
    from stock_ai.cloud.discord_bot import DiscordPollingBot

    with patch.object(DiscordPollingBot, "__init__", lambda self: None):
        bot = DiscordPollingBot()
        bot.monitored_channels = []

        for kw in ["見返し", "答え合わせ", "通信簿", "デイトレ結果", "今日の成績"]:
            with patch("stock_ai.cloud.discord_bot.build_daytrade_scorecard_payloads") as mock_build:
                mock_build.return_value = [{"content": "scorecard_test"}]
                res = bot._handle_user_message(kw)
                assert res == [{"content": "scorecard_test"}]
                mock_build.assert_called_once()



