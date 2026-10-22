"""tests/test_user_portfolio.py

Unit tests for user portfolio storage, trading ledger, natural language trade parser,
PnL calculation, and risk diagnostics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from stock_ai.storage.user_portfolio_store import (
    cancel_last_trade,
    diagnose_portfolio_risk,
    get_open_positions,
    get_portfolio_summary,
    get_trade_history,
    handle_user_portfolio_message,
    init_portfolio_db,
    parse_trade_message,
    record_buy,
    record_sell,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_user_portfolio.db"
    init_portfolio_db(db_file)
    return db_file


class TestNaturalLanguageTradeParser:
    """Tests for parsing natural language trade notes."""

    def test_parse_buy_variants(self) -> None:
        p1 = parse_trade_message("川重 100株 5000円で買った")
        assert p1["action"] == "BUY"
        assert p1["ticker"] == "7012.T"
        assert p1["shares"] == 100
        assert p1["price"] == 5000.0

        p2 = parse_trade_message("7203 トヨタ 200株 2750円 買い")
        assert p2["action"] == "BUY"
        assert p2["ticker"] == "7203.T"
        assert p2["shares"] == 200
        assert p2["price"] == 2750.0

        p3 = parse_trade_message("フジクラ 100株 @4800 購入")
        assert p3["action"] == "BUY"
        assert p3["ticker"] == "5803.T"
        assert p3["shares"] == 100
        assert p3["price"] == 4800.0

        p4 = parse_trade_message("7012 300株 5200円 買い増し")
        assert p4["action"] == "BUY"
        assert p4["ticker"] == "7012.T"
        assert p4["shares"] == 300
        assert p4["price"] == 5200.0

    def test_parse_sell_variants(self) -> None:
        p1 = parse_trade_message("トヨタ 100株 2850円で売った")
        assert p1["action"] == "SELL"
        assert p1["ticker"] == "7203.T"
        assert p1["shares"] == 100
        assert p1["price"] == 2850.0

        p2 = parse_trade_message("7012 100株 5500円 利確")
        assert p2["action"] == "SELL"
        assert p2["ticker"] == "7012.T"
        assert p2["shares"] == 100
        assert p2["price"] == 5500.0

        p3 = parse_trade_message("川重 全株 5100円で損切り")
        assert p3["action"] == "SELL"
        assert p3["ticker"] == "7012.T"
        assert p3["shares"] is None  # Full close
        assert p3["price"] == 5100.0

        p4 = parse_trade_message("フジクラ すべて 5200円 売却")
        assert p4["action"] == "SELL"
        assert p4["ticker"] == "5803.T"
        assert p4["shares"] is None

    def test_parse_control_and_inquiry(self) -> None:
        assert parse_trade_message("収支")["action"] == "PORTFOLIO"
        assert parse_trade_message("ポートフォリオ")["action"] == "PORTFOLIO"
        assert parse_trade_message("保有株")["action"] == "PORTFOLIO"
        assert parse_trade_message("履歴")["action"] == "HISTORY"
        assert parse_trade_message("最後の取引を取り消し")["action"] == "CANCEL"
        assert parse_trade_message("取り消し")["action"] == "CANCEL"
        assert parse_trade_message("今日大丈夫？")["action"] == "DIAGNOSE"
        assert parse_trade_message("診断")["action"] == "DIAGNOSE"

    def test_parse_non_trade(self) -> None:
        assert parse_trade_message("こんにちは")["action"] is None


class TestPortfolioLedgerOperations:
    """Tests for record_buy, record_sell, and cancel_last_trade."""

    def test_buy_and_add_position(self, temp_db: Path) -> None:
        # 1. First Buy: 100 shares @ 2,000
        r1 = record_buy("7203.T", "トヨタ自動車", 100, 2000.0, db_path=temp_db)
        assert r1["position_status"] == "NEW"
        assert r1["total_shares"] == 100
        assert r1["avg_price"] == 2000.0

        # 2. Add to Position: 100 shares @ 2,200
        r2 = record_buy("7203.T", "トヨタ自動車", 100, 2200.0, db_path=temp_db)
        assert r2["position_status"] == "ADDED"
        assert r2["total_shares"] == 200
        assert r2["avg_price"] == 2100.0  # Weighted average (200,000 + 220,000) / 200

        # Verify active positions
        positions = get_open_positions(temp_db)
        assert len(positions) == 1
        assert positions[0]["ticker"] == "7203.T"
        assert positions[0]["shares"] == 200
        assert positions[0]["avg_price"] == 2100.0

    def test_partial_sell_and_full_close(self, temp_db: Path) -> None:
        # Buy 200 shares @ 5,000
        record_buy("7012.T", "川崎重工業", 200, 5000.0, db_path=temp_db)

        # Partial sell: 100 shares @ 5,500 (+50,000 PnL)
        s1 = record_sell("7012.T", 100, 5500.0, db_path=temp_db)
        assert s1["sold_shares"] == 100
        assert s1["realized_pnl"] == 50000.0
        assert s1["realized_pnl_pct"] == 0.10
        assert s1["is_full_close"] is False
        assert s1["remaining_shares"] == 100
        assert s1["total_realized_pnl"] == 50000.0

        # Remaining position
        positions = get_open_positions(temp_db)
        assert len(positions) == 1
        assert positions[0]["shares"] == 100
        assert positions[0]["avg_price"] == 5000.0

        # Full sell of remaining 100 shares @ 4,800 (-20,000 PnL)
        s2 = record_sell("7012.T", None, 4800.0, db_path=temp_db)
        assert s2["sold_shares"] == 100
        assert s2["realized_pnl"] == -20000.0
        assert s2["is_full_close"] is True
        assert s2["remaining_shares"] == 0
        assert s2["total_realized_pnl"] == 30000.0  # 50,000 - 20,000

        # Position is now completely closed
        positions_after = get_open_positions(temp_db)
        assert len(positions_after) == 0

    def test_cancel_last_trade(self, temp_db: Path) -> None:
        # Trade 1: Buy 100 @ 1,000
        record_buy("8306.T", "三菱UFJ", 100, 1000.0, db_path=temp_db)
        # Trade 2: Mistake buy 100 @ 2,000
        record_buy("8306.T", "三菱UFJ", 100, 2000.0, db_path=temp_db)
        assert get_open_positions(temp_db)[0]["shares"] == 200

        # Cancel last trade
        c = cancel_last_trade(db_path=temp_db)
        assert c["ticker"] == "8306.T"
        assert c["reconstructed_shares"] == 100
        assert get_open_positions(temp_db)[0]["shares"] == 100
        assert get_open_positions(temp_db)[0]["avg_price"] == 1000.0

    def test_invalid_operations(self, temp_db: Path) -> None:
        with pytest.raises(ValueError):
            record_buy("7203.T", "トヨタ", -10, 2000.0, db_path=temp_db)
        with pytest.raises(ValueError):
            record_buy("7203.T", "トヨタ", 100, -2000.0, db_path=temp_db)
        with pytest.raises(ValueError):
            record_sell("9999.T", 100, 1000.0, db_path=temp_db)


class TestPortfolioSummaryAndRisk:
    """Tests for PnL summary and risk diagnostics."""

    def test_portfolio_summary_calculation(self, temp_db: Path) -> None:
        record_buy("7203.T", "トヨタ自動車", 100, 2500.0, db_path=temp_db)
        record_sell("7203.T", 50, 3000.0, db_path=temp_db)  # +25,000 realized

        summary = get_portfolio_summary(db_path=temp_db)
        assert summary["positions_count"] == 1
        assert summary["total_realized_pnl"] == 25000.0
        assert summary["win_rate"] == 1.0
        assert summary["profit_factor"] == 99.9  # No losses

    def test_diagnose_portfolio_risk(self, temp_db: Path) -> None:
        # Empty portfolio diagnosis
        diag_empty = diagnose_portfolio_risk(db_path=temp_db)
        assert diag_empty["status"] == "EMPTY"

        # Position with dummy rankings
        record_buy("7012.T", "川崎重工業", 100, 5000.0, db_path=temp_db)
        dummy_rankings = [
            {
                "ticker": "7012.T",
                "probability_up": 0.35,  # Red flag < 0.40
                "earnings_risk": True,
                "jumping_catch_risk": False,
            }
        ]
        diag = diagnose_portfolio_risk(db_path=temp_db, summary_data={"all_rankings": dummy_rankings})
        assert diag["status"] == "SUCCESS"
        assert diag["risk_counts"]["red"] == 1
        assert "🔴" in diag["diagnosed_positions"][0]["status_label"]

    def test_handle_user_portfolio_message(self, temp_db: Path) -> None:
        # Buy via message
        res_buy = handle_user_portfolio_message("川重 100株 5000円で買った", db_path=temp_db)
        assert res_buy is not None
        assert "買付記帳完了" in res_buy[0]["content"]

        # Check balance
        res_bal = handle_user_portfolio_message("収支", db_path=temp_db)
        assert res_bal is not None
        assert "総合ポートフォリオ収支" in res_bal[0]["content"]

        # Check diagnosis
        res_diag = handle_user_portfolio_message("今日大丈夫？", db_path=temp_db)
        assert res_diag is not None
        assert "危険度診断" in res_diag[0]["content"]

        # Sell via message
        res_sell = handle_user_portfolio_message("川重 全株 5500円で利確", db_path=temp_db)
        assert res_sell is not None
        assert "決済記帳完了" in res_sell[0]["content"]
        assert "+50,000 円" in res_sell[0]["content"]
