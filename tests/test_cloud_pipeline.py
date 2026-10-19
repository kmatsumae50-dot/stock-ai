from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from stock_ai.cloud.pipeline import (
    build_alert_report_payload,
    build_fund_report_payload,
    build_multi_agent_webhook_payloads,
    build_turnaround_report_payload,
    build_webhook_payload,
    get_configured_channel_webhooks,
    get_current_pipeline_summary,
    save_overnight_scanner_snapshot,
    send_daily_summary_webhook,
)


def test_build_webhook_payload_structure() -> None:
    sample_summary = {
        "date": "2026-10-14",
        "top_overnight": [
            {
                "ticker": "7974.T",
                "name": "任天堂",
                "sector": "ゲーム・エンタメ",
                "probability_up": 0.6099,
                "quantile": "Q5 (強気ギャップアップ帯)",
                "label": "強気ギャップアップ帯",
                "earnings_risk": True,
            },
            {
                "ticker": "9983.T",
                "name": "ファーストリテイ",
                "sector": "小売・内需",
                "probability_up": 0.5918,
                "quantile": "Q5 (強気ギャップアップ帯)",
                "label": "強気ギャップアップ帯",
                "earnings_risk": False,
            },
        ],
        "paper_trader": {
            "final_equity": 1_328_500.0,
            "cumulative_return": 0.3285,
            "win_rate": 0.612,
            "sharpe_ratio": 2.14,
            "max_drawdown": -0.068,
            "next_positions": "7974.T, 9983.T",
        },
        "top_turnaround": [
            {
                "ticker": "7201.T",
                "name": "日産自動車",
                "sector": "自動車",
                "bargain_score": 72.0,
                "drawdown_252d": -0.391,
                "pbr": 0.20,
                "status": "💎 超割安・底打ち反転圏 ⭐ 始動",
            }
        ],
    }

    payload = build_webhook_payload(sample_summary)

    assert "content" in payload
    assert "text" in payload
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1

    content = payload["content"]
    assert "任天堂" in content
    assert "7974.T" in content
    assert "61.0%" in content
    assert "⚠️決算警戒" in content
    assert "1,328,500" in content
    assert "+32.9%" in content
    assert "日産自動車" in content
    assert "72.0点" in content

    embed = payload["embeds"][0]
    assert "Stock-AI" in embed["title"]
    assert len(embed["fields"]) == 3
    # Check that earnings alert is shown in Overnight embed field
    assert "⚠️決算警戒" in embed["fields"][0]["value"]
    # Check paper trader position in field
    assert "本日判定" in embed["fields"][1]["value"]
    assert "7974.T" in embed["fields"][1]["value"]
    # Check turnaround top pick in field
    assert "日産自動車" in embed["fields"][2]["value"]


def test_save_overnight_scanner_snapshot(tmp_path: Path) -> None:
    snapshot_file = tmp_path / "test_snapshot.json"
    dummy_rankings = [
        {
            "rank": 1,
            "ticker": "7974.T",
            "name": "任天堂",
            "sector": "ゲーム・エンタメ",
            "probability_up": 0.6099,
            "quantile": "Q5 (強気ギャップアップ帯)",
            "direction": "Up ⬆️",
            "date": "2026-10-14",
        },
        {
            "rank": 2,
            "ticker": "8411.T",
            "name": "みずほFG",
            "sector": "銀行・金融",
            "probability_up": 0.4402,
            "quantile": "Q1 (ギャップダウン警戒帯)",
            "direction": "Down ⬇️",
            "date": "2026-10-14",
        },
    ]

    saved_path = save_overnight_scanner_snapshot(
        dummy_rankings,
        output_path=snapshot_file,
        date_str="2026-10-14",
    )

    assert saved_path.exists()
    data = json.loads(saved_path.read_text(encoding="utf-8"))
    assert data["date"] == "2026-10-14"
    assert data["tickers_count"] == 2
    assert data["q5_count"] == 1
    assert len(data["rankings"]) == 2
    assert data["rankings"][0]["ticker"] == "7974.T"


def test_send_daily_summary_webhook_skipped_when_no_url() -> None:
    res = send_daily_summary_webhook({"date": "2026-10-14"}, webhook_url="")
    assert res["status"] == "SKIPPED"
    assert "not set" in res["reason"]


def test_build_multi_agent_webhook_payloads() -> None:
    sample_summary = {
        "date": "2026-10-19",
        "top_overnight": [
            {
                "ticker": "7974.T",
                "name": "任天堂",
                "sector": "ゲーム・エンタメ",
                "probability_up": 0.62,
                "quantile": "Q5 (強気ギャップアップ帯)",
                "earnings_risk": False,
            }
        ],
        "paper_trader": {
            "final_equity": 1050000.0,
            "cumulative_return": 0.05,
            "win_rate": 0.60,
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.03,
            "next_positions": "7974.T (任天堂)",
        },
        "top_turnaround": [
            {
                "ticker": "7201.T",
                "name": "日産自動車",
                "sector": "自動車",
                "bargain_score": 75.0,
                "drawdown_252d": -0.45,
                "pbr": 0.38,
                "status": "💎 超割安・底打ち反転圏",
            }
        ],
    }

    payloads = build_multi_agent_webhook_payloads(sample_summary)
    assert len(payloads) == 3

    # 1. Chief Fund Manager
    p_chief = payloads[0]
    assert "チーフ・ファンドマネージャー" in p_chief["username"]
    assert "任天堂" in p_chief["content"]
    assert p_chief["embeds"][0]["color"] == 3066993
    assert "仮想100万円ファンド" in p_chief["embeds"][0]["title"]

    # 2. Overnight Sniper
    p_sniper = payloads[1]
    assert "短期スナイパー" in p_sniper["username"]
    assert "Overnight" in p_sniper["content"]
    assert p_sniper["embeds"][0]["color"] == 15844367
    assert "Overnight Q5" in p_sniper["embeds"][0]["title"]

    # 3. Value Hunter
    p_hunter = payloads[2]
    assert "バリューハンター" in p_hunter["username"]
    assert "お宝" in p_hunter["content"]
    assert p_hunter["embeds"][0]["color"] == 10181046
    assert "Value & Turnaround" in p_hunter["embeds"][0]["title"]


def test_send_daily_summary_webhook_mock() -> None:
    mock_response = MagicMock()
    mock_response.__enter__.return_value.getcode.return_value = 204

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        res = send_daily_summary_webhook(
            {"date": "2026-10-14"},
            webhook_url="https://discord.com/api/webhooks/dummy/test",
            sleep_interval=0.0,
        )
        assert res["status"] == "SUCCESS"
        assert res["status_code"] == 204
        assert res.get("messages_sent") == 3
        assert mock_urlopen.call_count == 3


def test_send_daily_summary_webhook_discordapp_normalized() -> None:
    from stock_ai.cloud.pipeline import normalize_webhook_url

    raw = "  \"https://discordapp.com/api/webhooks/123/xyz\"  "
    normalized = normalize_webhook_url(raw)
    assert normalized == "https://discord.com/api/webhooks/123/xyz"

    mock_response = MagicMock()
    mock_response.__enter__.return_value.getcode.return_value = 204
    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        res = send_daily_summary_webhook(
            {"date": "2026-10-14"},
            webhook_url=raw,
            sleep_interval=0.0,
        )
        assert res["status"] == "SUCCESS"
        called_req = mock_urlopen.call_args[0][0]
        assert called_req.full_url == "https://discord.com/api/webhooks/123/xyz"


def test_specialized_channel_report_payloads() -> None:
    sample_summary = {
        "date": "2026-10-19",
        "top_overnight": [
            {
                "ticker": "7974.T",
                "name": "任天堂",
                "sector": "ゲーム・エンタメ",
                "probability_up": 0.62,
                "quantile": "Q5 (強気ギャップアップ帯)",
                "earnings_risk": True,
            }
        ],
        "paper_trader": {
            "final_equity": 1050000.0,
            "cumulative_return": 0.05,
            "win_rate": 0.60,
            "sharpe_ratio": 1.85,
            "max_drawdown": -0.03,
            "next_positions": "7974.T (任天堂)",
        },
        "top_turnaround": [
            {
                "ticker": "7201.T",
                "name": "日産自動車",
                "sector": "自動車",
                "bargain_score": 75.0,
                "drawdown_252d": -0.45,
                "pbr": 0.38,
                "status": "💎 超割安・底打ち反転圏",
            }
        ],
    }

    # Fund payload test
    p_fund = build_fund_report_payload(sample_summary)
    assert "仮想100万円ファンド" in p_fund["username"]
    assert "1,050,000" in p_fund["content"]
    assert p_fund["embeds"][0]["color"] == 3066993

    # Turnaround payload test
    p_turn = build_turnaround_report_payload(sample_summary)
    assert "激安回復株レーダー" in p_turn["username"]
    assert "7201.T" in p_turn["content"]
    assert p_turn["embeds"][0]["color"] == 10181046

    # Alert payload test (with risk)
    p_alert = build_alert_report_payload(sample_summary)
    assert "決算・急変アラート" in p_alert["username"]
    assert "7974.T" in p_alert["content"]
    assert p_alert["embeds"][0]["color"] == 15158332

    # Alert payload test (safe / no risk)
    safe_summary = dict(sample_summary)
    safe_summary["top_overnight"] = [
        dict(sample_summary["top_overnight"][0], earnings_risk=False)
    ]
    p_safe = build_alert_report_payload(safe_summary)
    assert "セーフティクリア" in p_safe["embeds"][0]["title"]
    assert p_safe["embeds"][0]["color"] == 3066993


def test_send_daily_summary_webhook_multi_channel_dispatch() -> None:
    mock_response = MagicMock()
    mock_response.__enter__.return_value.getcode.return_value = 204

    channel_webhooks = {
        "main": "https://discord.com/api/webhooks/main/test",
        "fund": "https://discord.com/api/webhooks/fund/test",
        "turnaround": "https://discord.com/api/webhooks/turnaround/test",
        "alert": "https://discord.com/api/webhooks/alert/test",
    }

    with patch("urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
        res = send_daily_summary_webhook(
            {"date": "2026-10-19"},
            channel_webhooks=channel_webhooks,
            sleep_interval=0.0,
        )
        assert res["status"] == "SUCCESS"
        assert res["status_code"] == 204
        assert res.get("messages_sent") == 6  # 3 (main) + 1 (fund) + 1 (turnaround) + 1 (alert)
        assert set(res.get("channels_delivered", [])) == {"main", "fund", "turnaround", "alert"}
        assert mock_urlopen.call_count == 6


def test_get_configured_channel_webhooks() -> None:
    fake_env = {
        "STOCK_AI_WEBHOOK_URL": "https://discord.com/api/webhooks/main/fake",
        "STOCK_AI_WEBHOOK_FUND": "https://discord.com/api/webhooks/fund/fake",
    }
    with patch.dict("os.environ", fake_env, clear=True):
        whs = get_configured_channel_webhooks()
        assert whs.get("main") == "https://discord.com/api/webhooks/main/fake"
        assert whs.get("fund") == "https://discord.com/api/webhooks/fund/fake"


def test_cli_cloud_pipeline_parser_registered() -> None:
    import inspect
    from stock_ai.cli import main

    cli_source = inspect.getsource(main)
    assert "run-cloud-daily-pipeline" in cli_source
    assert "--snapshot-path" in cli_source
    assert "--webhook-url" in cli_source


def test_ui_cloud_sync_header_callable() -> None:
    from stock_ai.app.main import _render_cloud_sync_header

    assert callable(_render_cloud_sync_header)


def test_run_cloud_daily_pipeline_mocked(tmp_path: Path) -> None:
    from stock_ai.cloud.pipeline import run_cloud_daily_pipeline
    from stock_ai.data.universe import TickerInfo
    import pandas as pd

    test_universe = [
        TickerInfo(code="7974.T", name="任天堂", sector="ゲーム・エンタメ"),
    ]
    test_snapshot = tmp_path / "test_snapshot.json"

    dummy_preds = pd.DataFrame([
        {
            "date": "2026-10-14",
            "ticker": "7974.T",
            "probability_up": 0.62,
            "prediction": 1,
            "model_name": "lightgbm_overnight_binary",
            "notes": json.dumps({"quantile": "Q5", "label": "強気ギャップアップ帯", "historical_win_rate": 0.61}),
        }
    ])

    dummy_sim = {
        "start_date": "2024-04-01",
        "end_date": "2026-10-14",
        "total_days": 100,
        "results_by_cost": {
            "5bps": {
                "elite_top2_long": {
                    "metrics": {
                        "final_equity": 1_250_000.0,
                        "cumulative_return": 0.25,
                        "win_rate": 0.60,
                        "sharpe_ratio": 1.95,
                        "max_drawdown": -0.05,
                    }
                }
            }
        },
        "latest_pending": {
            "date": "2026-10-14",
            "strategies": {
                "elite_top2_long": {
                    "long_picks": ["7974.T"],
                    "short_picks": [],
                    "equity": 1_250_000.0,
                }
            }
        },
    }

    dummy_turnaround = {
        "date": "2026-10-14",
        "total_scanned": 1,
        "rankings": [
            {
                "ticker": "7201.T",
                "name": "日産自動車",
                "bargain_score": 75.0,
            }
        ],
    }

    with patch("stock_ai.cloud.pipeline.download_external_market_data_report"), \
         patch("stock_ai.cloud.pipeline.fetch_daily_ohlcv"), \
         patch("stock_ai.cli._build_features_for_ticker"), \
         patch("stock_ai.cli.train_overnight_model_for_ticker"), \
         patch("stock_ai.cli.predict_overnight_for_ticker", return_value=(1, dummy_preds)), \
         patch("stock_ai.backtest.overnight_portfolio.run_full_overnight_portfolio_backtest", return_value=dummy_sim), \
         patch("stock_ai.backtest.overnight_portfolio.export_portfolio_simulation_reports"), \
         patch("stock_ai.cloud.pipeline.scan_turnaround_universe", return_value=dummy_turnaround), \
         patch("stock_ai.cloud.pipeline.export_turnaround_reports"):

        res = run_cloud_daily_pipeline(
            universe=test_universe,
            snapshot_path=test_snapshot,
            webhook_url=None,
        )

        assert res["status"] == "SUCCESS"
        assert res["steps"]["overnight_count"] == 1
        assert res["steps"]["paper_trader"] == "SUCCESS"
        assert res["steps"]["turnaround"] == "SUCCESS"
        assert test_snapshot.exists()


def test_main_ui_renders_3_dashboards_when_parquet_missing() -> None:
    """Verify that main() renders the 3 cloud dashboards even if ticker parquet data is empty."""
    import pandas as pd
    from stock_ai.app.main import main

    with patch("stock_ai.app.main._render_cloud_sync_header") as mock_header, \
         patch("stock_ai.app.main._render_multi_ticker_overnight_scanner") as mock_scanner, \
         patch("stock_ai.app.main._render_overnight_paper_trader_section") as mock_paper, \
         patch("stock_ai.app.main._render_turnaround_scanner_section") as mock_turnaround, \
         patch("stock_ai.app.main._load_dataset", return_value=(pd.DataFrame(), "missing")), \
         patch("streamlit.info") as mock_info, \
         patch("streamlit.tabs") as mock_tabs:

        main()

        # All 3 major dashboards must be rendered prior to the empty data check
        mock_header.assert_called_once()
        mock_scanner.assert_called_once()
        mock_paper.assert_called_once()
        mock_turnaround.assert_called_once()

        # User guidance info must be shown
        mock_info.assert_called_once()
        assert "個別銘柄の詳細ヒストリカル分析" in mock_info.call_args[0][0]

        # Heavy individual analysis tabs must not be created since parquet is missing
        mock_tabs.assert_not_called()


def test_get_current_pipeline_summary(tmp_path: Path) -> None:
    snap_p = tmp_path / "snap.json"
    paper_p = tmp_path / "paper.json"
    turn_p = tmp_path / "turn.json"

    snap_p.write_text(json.dumps({
        "date": "2026-10-15",
        "rankings": [{"ticker": "7203.T", "name": "トヨタ", "probability_up": 0.65, "earnings_risk": True}],
    }), encoding="utf-8")

    paper_p.write_text(json.dumps({
        "results_by_cost": {
            "5bps": {
                "elite_top2_long": {
                    "metrics": {"final_equity": 1_050_000.0, "cumulative_return": 0.05, "win_rate": 0.60, "sharpe_ratio": 2.0, "max_drawdown": -0.02}
                }
            }
        },
        "latest_pending": {"strategies": {"elite_top2_long": {"long_picks": ["8306.T"]}}},
    }), encoding="utf-8")

    turn_p.write_text(json.dumps({
        "rankings": [{"ticker": "7201.T", "name": "日産", "bargain_score": 75.0, "drawdown_252d": -0.35, "pbr": 0.28, "status": "超割安"}],
    }), encoding="utf-8")

    summary = get_current_pipeline_summary(snapshot_path=snap_p, paper_path=paper_p, turnaround_path=turn_p)

    assert summary["date"] == "2026-10-15"
    assert len(summary["top_overnight"]) == 1
    assert summary["top_overnight"][0]["ticker"] == "7203.T"
    assert summary["paper_trader"]["final_equity"] == 1_050_000.0
    assert summary["paper_trader"]["next_positions"] == "8306.T"
    assert len(summary["top_turnaround"]) == 1
    assert summary["top_turnaround"][0]["ticker"] == "7201.T"


def test_render_discord_webhook_panel_success() -> None:
    from stock_ai.app.main import _render_discord_webhook_panel

    with patch("streamlit.sidebar.expander"), \
         patch("streamlit.text_input", return_value="https://discord.com/api/webhooks/dummy/123"), \
         patch("streamlit.button", return_value=True), \
         patch("stock_ai.cloud.pipeline.send_daily_summary_webhook", return_value={"status": "SUCCESS"}) as mock_send, \
         patch("streamlit.success") as mock_success:

        _render_discord_webhook_panel()

        mock_send.assert_called_once()
        mock_success.assert_called_once()
        assert "Discord へリッチ通知を送信しました" in mock_success.call_args[0][0]


def test_render_discord_webhook_panel_main_container() -> None:
    import streamlit as st
    from stock_ai.app.main import _render_discord_webhook_panel

    with patch.object(st, "expander") as mock_expander, \
         patch("streamlit.text_input", return_value="https://discord.com/api/webhooks/main/456"), \
         patch("streamlit.button", return_value=True), \
         patch("stock_ai.cloud.pipeline.send_daily_summary_webhook", return_value={"status": "SUCCESS"}) as mock_send, \
         patch("streamlit.success") as mock_success:

        _render_discord_webhook_panel(container=st, key_prefix="main")

        mock_expander.assert_called_once_with("🔔 Discord 通知設定 & テスト送信", expanded=False)
        mock_send.assert_called_once()
        mock_success.assert_called_once()


def test_discord_provisioner_mock() -> None:
    from stock_ai.cloud.discord_provisioner import DiscordProvisioner

    p = DiscordProvisioner(bot_token="fake_bot_token")
    mock_guilds = [{"id": "123456", "name": "AI株"}]
    mock_channels = [
        {"id": "cat1", "name": "📡 AI自動配信ルーム", "type": 4},
        {"id": "ch1", "name": "☕｜aiアナリスト作戦会議", "type": 0, "parent_id": "cat1"},
        {"id": "ch2", "name": "💰｜100万円ファンド運用記", "type": 0, "parent_id": "cat1"},
        {"id": "ch3", "name": "💎｜激安回復株レーダー", "type": 0, "parent_id": "cat1"},
        {"id": "ch4", "name": "🚨｜決算・急変アラート", "type": 0, "parent_id": "cat1"},
        {"id": "cat2", "name": "🧠 戦略・ラボ", "type": 4},
        {"id": "ch5", "name": "📝｜相場メモ・独り言", "type": 0, "parent_id": "cat2"},
    ]
    mock_webhooks = [{"id": "wh1", "url": "https://discord.com/api/webhooks/123/token"}]

    with patch.object(p, "_api_request") as mock_api:
        def side_effect(endpoint: str, method: str = "GET", payload: Any = None, retries: int = 3) -> Any:
            if endpoint == "/users/@me/guilds":
                return mock_guilds
            if "/channels" in endpoint and method == "GET":
                return mock_channels
            if "/webhooks" in endpoint and method == "GET":
                return mock_webhooks
            return {"id": "new_id", "url": "https://discord.com/api/webhooks/mock/url"}

        mock_api.side_effect = side_effect
        res = p.provision_all()
        assert res["guild_id"] == "123456"
        assert "main" in res["webhooks"]
        assert "fund" in res["webhooks"]
        assert "turnaround" in res["webhooks"]
        assert "alert" in res["webhooks"]


def test_webhook_deduplication_logic(tmp_path: Path) -> None:
    from stock_ai.cloud.pipeline import _record_webhook_delivery, _should_skip_webhook_delivery

    test_state_file = tmp_path / "webhook_delivery_state.json"
    with patch("stock_ai.cloud.pipeline.Path") as mock_path:
        # Point to tmp state file
        mock_path.return_value = test_state_file
        # Initial check (file does not exist)
        assert _should_skip_webhook_delivery("2026-10-19") is False

        # Record delivery
        _record_webhook_delivery("2026-10-19")
        assert test_state_file.exists()

        # Should skip right after recording
        assert _should_skip_webhook_delivery("2026-10-19", min_interval_minutes=15.0) is True
        # Different date should not skip
        assert _should_skip_webhook_delivery("2026-10-20", min_interval_minutes=15.0) is False
