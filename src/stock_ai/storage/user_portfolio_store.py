"""src/stock_ai/storage/user_portfolio_store.py

Personal trading ledger, portfolio position tracker, and PnL calculation engine.
Persists trade history and active holdings in a dedicated SQLite database (data/user_portfolio.db).
"""

from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from stock_ai.config import PROJECT_ROOT
from stock_ai.storage.parquet_store import read_parquet

logger = logging.getLogger(__name__)

DEFAULT_PORTFOLIO_DB_PATH = PROJECT_ROOT / "data" / "user_portfolio.db"

USER_TRADES_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_date TEXT NOT NULL,
    ticker TEXT NOT NULL,
    name TEXT NOT NULL,
    side TEXT NOT NULL,
    shares INTEGER NOT NULL,
    price REAL NOT NULL,
    realized_pnl REAL,
    realized_pnl_pct REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);
"""

USER_POSITIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS user_positions (
    ticker TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    shares INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    total_cost REAL NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _get_db_path(db_path: Path | str | None = None) -> Path:
    if db_path is not None:
        p = Path(db_path)
    else:
        p = DEFAULT_PORTFOLIO_DB_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _connect_db(db_path: Path | str | None = None) -> sqlite3.Connection:
    p = _get_db_path(db_path)
    conn = sqlite3.connect(p)
    conn.row_factory = sqlite3.Row
    return conn


def init_portfolio_db(db_path: Path | str | None = None) -> Path:
    """Initialize SQLite tables for user trading ledger."""
    p = _get_db_path(db_path)
    conn = _connect_db(db_path)
    try:
        conn.execute(USER_TRADES_SCHEMA)
        conn.execute(USER_POSITIONS_SCHEMA)
        conn.commit()
    finally:
        conn.close()
    return p


def record_buy(
    ticker: str,
    name: str,
    shares: int,
    price: float,
    trade_date: str | None = None,
    notes: str = "",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Record a buy trade (new holding or adding to existing position).

    Computes weighted average cost basis automatically.
    """
    if shares <= 0:
        raise ValueError("Shares must be positive.")
    if price <= 0:
        raise ValueError("Price must be positive.")

    init_portfolio_db(db_path)
    p = _get_db_path(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    t_date = trade_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Clean ticker format
    clean_ticker = ticker.strip().upper()
    if clean_ticker.isdigit() and len(clean_ticker) == 4:
        clean_ticker = f"{clean_ticker}.T"
    clean_name = name.strip() or clean_ticker

    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()

        # Check existing position
        cur.execute("SELECT ticker, name, shares, avg_price, total_cost FROM user_positions WHERE ticker = ?", (clean_ticker,))
        row = cur.fetchone()

        if row:
            old_shares = int(row["shares"])
            old_avg = float(row["avg_price"])
            old_name = str(row["name"]) or clean_name
            new_shares = old_shares + shares
            new_total_cost = (old_shares * old_avg) + (shares * price)
            new_avg_price = new_total_cost / new_shares
            final_name = clean_name if clean_name != clean_ticker else old_name

            cur.execute(
                """
                UPDATE user_positions
                SET shares = ?, avg_price = ?, total_cost = ?, name = ?, updated_at = ?
                WHERE ticker = ?
                """,
                (new_shares, new_avg_price, new_total_cost, final_name, now_iso, clean_ticker),
            )
            position_status = "ADDED"
        else:
            new_shares = shares
            new_avg_price = float(price)
            new_total_cost = float(shares * price)
            final_name = clean_name

            cur.execute(
                """
                INSERT INTO user_positions (ticker, name, shares, avg_price, total_cost, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (clean_ticker, final_name, new_shares, new_avg_price, new_total_cost, now_iso),
            )
            position_status = "NEW"

        # Record trade in ledger
        cur.execute(
            """
            INSERT INTO user_trades (trade_date, ticker, name, side, shares, price, realized_pnl, realized_pnl_pct, notes, created_at)
            VALUES (?, ?, ?, 'BUY', ?, ?, 0.0, 0.0, ?, ?)
            """,
            (t_date, clean_ticker, final_name, shares, float(price), notes, now_iso),
        )
        trade_id = cur.lastrowid
        conn.commit()
    finally:
        conn.close()

    return {
        "trade_id": trade_id,
        "ticker": clean_ticker,
        "name": final_name,
        "side": "BUY",
        "shares": shares,
        "price": float(price),
        "trade_value": float(shares * price),
        "trade_date": t_date,
        "position_status": position_status,
        "total_shares": new_shares,
        "avg_price": round(new_avg_price, 2),
        "total_cost": round(new_total_cost, 2),
    }


def record_sell(
    ticker: str,
    shares: int | None,
    price: float,
    trade_date: str | None = None,
    notes: str = "",
    db_path: Path | str | None = None,
) -> dict[str, Any]:
    """Record a sell trade (partial sell or full close).

    Computes realized PnL based on weighted average cost.
    """
    if price <= 0:
        raise ValueError("Price must be positive.")

    init_portfolio_db(db_path)
    now_iso = datetime.now(timezone.utc).isoformat()
    t_date = trade_date or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    clean_ticker = ticker.strip().upper()
    if clean_ticker.isdigit() and len(clean_ticker) == 4:
        clean_ticker = f"{clean_ticker}.T"

    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()

        cur.execute("SELECT ticker, name, shares, avg_price, total_cost FROM user_positions WHERE ticker = ?", (clean_ticker,))
        row = cur.fetchone()

        if not row:
            raise ValueError(f"No active position found for ticker '{clean_ticker}'.")

        pos_name = str(row["name"])
        current_shares = int(row["shares"])
        avg_price = float(row["avg_price"])

        # Determine shares to sell (None means sell ALL)
        if shares is None or shares >= current_shares:
            sold_shares = current_shares
            remaining_shares = 0
            is_full_close = True
        else:
            if shares <= 0:
                raise ValueError("Shares to sell must be positive.")
            sold_shares = shares
            remaining_shares = current_shares - sold_shares
            is_full_close = False

        realized_pnl = float((price - avg_price) * sold_shares)
        realized_pnl_pct = float((price - avg_price) / avg_price) if avg_price > 0 else 0.0
        trade_value = float(sold_shares * price)

        if is_full_close:
            cur.execute("DELETE FROM user_positions WHERE ticker = ?", (clean_ticker,))
        else:
            new_total_cost = remaining_shares * avg_price
            cur.execute(
                """
                UPDATE user_positions
                SET shares = ?, total_cost = ?, updated_at = ?
                WHERE ticker = ?
                """,
                (remaining_shares, new_total_cost, now_iso, clean_ticker),
            )

        # Record trade in ledger
        cur.execute(
            """
            INSERT INTO user_trades (trade_date, ticker, name, side, shares, price, realized_pnl, realized_pnl_pct, notes, created_at)
            VALUES (?, ?, ?, 'SELL', ?, ?, ?, ?, ?, ?)
            """,
            (t_date, clean_ticker, pos_name, sold_shares, float(price), realized_pnl, realized_pnl_pct, notes, now_iso),
        )
        trade_id = cur.lastrowid

        # Calculate all-time cumulative realized PnL
        cur.execute("SELECT SUM(realized_pnl) FROM user_trades WHERE side = 'SELL'")
        total_realized_sum = cur.fetchone()[0] or 0.0

        conn.commit()
    finally:
        conn.close()

    return {
        "trade_id": trade_id,
        "ticker": clean_ticker,
        "name": pos_name,
        "side": "SELL",
        "sold_shares": sold_shares,
        "price": float(price),
        "trade_value": trade_value,
        "avg_price": round(avg_price, 2),
        "realized_pnl": round(realized_pnl, 2),
        "realized_pnl_pct": round(realized_pnl_pct, 4),
        "is_full_close": is_full_close,
        "remaining_shares": remaining_shares,
        "total_realized_pnl": round(float(total_realized_sum), 2),
        "trade_date": t_date,
    }


def cancel_last_trade(db_path: Path | str | None = None) -> dict[str, Any]:
    """Revert the most recent trade from ledger and recalculate positions."""
    init_portfolio_db(db_path)
    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()

        cur.execute("SELECT * FROM user_trades ORDER BY id DESC LIMIT 1")
        last_trade = cur.fetchone()
        if not last_trade:
            raise ValueError("No trades to cancel.")

        trade_id = last_trade["id"]
        ticker = last_trade["ticker"]
        side = last_trade["side"]
        shares = last_trade["shares"]
        price = last_trade["price"]
        name = last_trade["name"]

        # Delete the last trade
        cur.execute("DELETE FROM user_trades WHERE id = ?", (trade_id,))

        # Re-derive position from all remaining trades for this ticker
        cur.execute(
            "SELECT side, shares, price, name, trade_date FROM user_trades WHERE ticker = ? ORDER BY id ASC",
            (ticker,),
        )
        remaining_trades = cur.fetchall()

        cur.execute("DELETE FROM user_positions WHERE ticker = ?", (ticker,))

        cur_shares = 0
        cur_cost = 0.0
        cur_name = name

        for t in remaining_trades:
            cur_name = t["name"] or cur_name
            t_side = t["side"]
            t_shares = t["shares"]
            t_price = t["price"]
            if t_side == "BUY":
                cur_cost += t_shares * t_price
                cur_shares += t_shares
            elif t_side == "SELL":
                if cur_shares > 0:
                    avg_p = cur_cost / cur_shares
                    cur_shares = max(0, cur_shares - t_shares)
                    cur_cost = cur_shares * avg_p

        if cur_shares > 0:
            avg_p = cur_cost / cur_shares
            now_iso = datetime.now(timezone.utc).isoformat()
            cur.execute(
                """
                INSERT INTO user_positions (ticker, name, shares, avg_price, total_cost, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (ticker, cur_name, cur_shares, avg_p, cur_cost, now_iso),
            )

        conn.commit()
    finally:
        conn.close()

    return {
        "cancelled_trade_id": trade_id,
        "ticker": ticker,
        "name": name,
        "side": side,
        "shares": shares,
        "price": price,
        "reconstructed_shares": cur_shares,
    }


def get_open_positions(db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Retrieve all active open positions."""
    init_portfolio_db(db_path)
    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT ticker, name, shares, avg_price, total_cost, updated_at FROM user_positions WHERE shares > 0 ORDER BY total_cost DESC")
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "ticker": r["ticker"],
            "name": r["name"],
            "shares": int(r["shares"]),
            "avg_price": round(float(r["avg_price"]), 2),
            "total_cost": round(float(r["total_cost"]), 2),
            "updated_at": r["updated_at"],
        }
        for r in rows
    ]


def get_trade_history(limit: int = 20, db_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Retrieve recent trade history."""
    init_portfolio_db(db_path)
    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, trade_date, ticker, name, side, shares, price, realized_pnl, realized_pnl_pct, notes
            FROM user_trades
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    return [
        {
            "id": r["id"],
            "trade_date": r["trade_date"],
            "ticker": r["ticker"],
            "name": r["name"],
            "side": r["side"],
            "shares": int(r["shares"]),
            "price": float(r["price"]),
            "realized_pnl": round(float(r["realized_pnl"] or 0.0), 2),
            "realized_pnl_pct": round(float(r["realized_pnl_pct"] or 0.0), 4),
            "notes": r["notes"] or "",
        }
        for r in rows
    ]


def _get_latest_market_price(ticker: str) -> float | None:
    """Fetch latest available market price for a ticker (from local feature cache or yfinance)."""
    # 1. Local feature/raw parquet cache
    features_dir = PROJECT_ROOT / "data" / "features"
    feat_path = features_dir / f"{ticker}.parquet"
    if feat_path.exists():
        try:
            df = read_parquet(feat_path)
            if not df.empty and "close" in df.columns:
                val = float(df["close"].dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception:
            pass

    raw_dir = PROJECT_ROOT / "data" / "raw"
    raw_path = raw_dir / f"{ticker}.parquet"
    if raw_path.exists():
        try:
            df = read_parquet(raw_path)
            if not df.empty and "close" in df.columns:
                val = float(df["close"].dropna().iloc[-1])
                if val > 0:
                    return val
        except Exception:
            pass

    # 2. yfinance provider fallback
    try:
        from stock_ai.data.yfinance_provider import fetch_daily_ohlcv
        df_live = fetch_daily_ohlcv(ticker, start="2024-01-01")
        if not df_live.empty and "close" in df_live.columns:
            val = float(df_live["close"].dropna().iloc[-1])
            if val > 0:
                return val
    except Exception:
        pass

    return None


def get_portfolio_summary(db_path: Path | str | None = None) -> dict[str, Any]:
    """Calculate comprehensive user portfolio summary with real-time unrealized and realized PnL."""
    init_portfolio_db(db_path)
    positions = get_open_positions(db_path=db_path)

    # 1. Realized metrics from history
    conn = _connect_db(db_path)
    try:
        cur = conn.cursor()
        cur.execute("SELECT realized_pnl, realized_pnl_pct FROM user_trades WHERE side = 'SELL'")
        sell_trades = cur.fetchall()
    finally:
        conn.close()

    total_realized_pnl = sum(float(r["realized_pnl"] or 0.0) for r in sell_trades)
    win_trades = [r for r in sell_trades if float(r["realized_pnl"] or 0.0) > 0]
    loss_trades = [r for r in sell_trades if float(r["realized_pnl"] or 0.0) < 0]
    total_trades_count = len(sell_trades)
    win_trades_count = len(win_trades)
    loss_trades_count = len(loss_trades)
    win_rate = win_trades_count / total_trades_count if total_trades_count > 0 else 0.0

    total_gross_profit = sum(float(r["realized_pnl"]) for r in win_trades)
    total_gross_loss = abs(sum(float(r["realized_pnl"]) for r in loss_trades))
    if total_gross_loss > 0:
        profit_factor = total_gross_profit / total_gross_loss
    elif total_gross_profit > 0:
        profit_factor = 99.9  # No losses
    else:
        profit_factor = 1.0

    # 2. Unrealized metrics for current positions
    enriched_positions: list[dict[str, Any]] = []
    total_book_value = 0.0
    total_market_value = 0.0

    for pos in positions:
        ticker = pos["ticker"]
        shares = pos["shares"]
        avg_price = pos["avg_price"]
        book_val = pos["total_cost"]

        current_price = _get_latest_market_price(ticker) or avg_price
        market_val = shares * current_price
        unrealized_pnl = market_val - book_val
        unrealized_pnl_pct = (current_price - avg_price) / avg_price if avg_price > 0 else 0.0

        total_book_value += book_val
        total_market_value += market_val

        enriched_positions.append({
            "ticker": ticker,
            "name": pos["name"],
            "shares": shares,
            "avg_price": avg_price,
            "current_price": round(current_price, 2),
            "book_value": round(book_val, 2),
            "market_value": round(market_val, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "unrealized_pnl_pct": round(unrealized_pnl_pct, 4),
        })

    total_unrealized_pnl = total_market_value - total_book_value
    total_unrealized_pnl_pct = total_unrealized_pnl / total_book_value if total_book_value > 0 else 0.0
    total_combined_pnl = total_realized_pnl + total_unrealized_pnl

    return {
        "positions_count": len(enriched_positions),
        "positions": enriched_positions,
        "total_book_value": round(total_book_value, 2),
        "total_market_value": round(total_market_value, 2),
        "total_unrealized_pnl": round(total_unrealized_pnl, 2),
        "total_unrealized_pnl_pct": round(total_unrealized_pnl_pct, 4),
        "total_realized_pnl": round(total_realized_pnl, 2),
        "total_combined_pnl": round(total_combined_pnl, 2),
        "total_trades_count": total_trades_count,
        "win_trades_count": win_trades_count,
        "loss_trades_count": loss_trades_count,
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 2),
    }


# ---------------------------------------------------------------------------
# 3. Natural Language Trade Parser
# ---------------------------------------------------------------------------

_ZEN_TO_HAN_TABLE = str.maketrans(
    "０１２３４５６７８９ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ＠，",
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@,",
)


def _resolve_stock_symbol(text: str) -> tuple[str, str] | None:
    """Resolve ticker and company name using aliases and universe."""
    try:
        from stock_ai.cloud.discord_bot import resolve_ticker
        res = resolve_ticker(text)
        if res:
            return res
    except Exception:
        pass

    # Direct 4-digit fallback
    m = re.search(r"\b(\d{4}|\d{3}[A-Za-z])(?:\.T)?\b", text, re.IGNORECASE)
    if m:
        c = m.group(1).upper()
        return f"{c}.T", f"{c}.T"
    return None


def parse_trade_message(raw_text: str) -> dict[str, Any]:
    """Parse user natural language input into structured trading commands.

    Supported intents:
    - BUY: 'トヨタ 100株 2750円で買った', '7012 川重 200株 5200円 買い', 'フジクラ 100株 @4800 購入'
    - SELL: 'トヨタ 100株 2850円で売った', '7012 100株 5500円 利確', '川重 全株 5100円で損切り'
    - PORTFOLIO: '収支', 'ポートフォリオ', '保有株', '資産'
    - HISTORY: '履歴', '取引履歴', '売買履歴'
    - CANCEL: '最後の取引を取り消し', '取り消し', 'キャンセル'
    - DIAGNOSE: '今日大丈夫？', '診断', 'リスク', '保有株診断'
    """
    clean = raw_text.strip().translate(_ZEN_TO_HAN_TABLE)
    clean_lower = clean.lower()

    # 1. Cancel / Undo intent
    if re.search(r"^(最後|直前)?の?(取引|売買)?(を)?(取り消し|取消|キャンセル|戻す)$", clean) or clean in ("取り消し", "取消", "キャンセル", "undo"):
        return {"action": "CANCEL"}

    # 2. Portfolio / Holdings query
    if clean in ("収支", "ポートフォリオ", "保有株", "保有銘柄", "資産", "マイノート", "残高", "含み損益") or re.search(r"^(収支|ポートフォリオ|保有株|資産)(見せて|教えて|確認)?$", clean):
        return {"action": "PORTFOLIO"}

    # 3. History query
    if clean in ("履歴", "取引履歴", "売買履歴", "トレード履歴", "約定履歴") or re.search(r"^(履歴|取引履歴|売買履歴)(見せて|教えて)?$", clean):
        return {"action": "HISTORY"}

    # 4. Risk Diagnosis query
    if clean in ("診断", "今日大丈夫？", "今日大丈夫", "大丈夫？", "大丈夫", "リスク", "保有株診断", "危険度") or re.search(r"^(今日|今)(の保有株)?大丈夫", clean):
        return {"action": "DIAGNOSE"}

    # Exclude consultation / inquiry questions (e.g., '日産は今買い時かな？', '30万円で仕込むなら何がいい？', '評価して')
    consultation_indicators = [
        "買い時", "売り時", "どう思う", "どうかな", "どう？", "どうですか",
        "何がいい", "なにがいい", "どれがいい", "おすすめ", "仕込むなら",
        "買うべき", "売るべき", "買えばいい", "売ればいい", "評価して",
        "見通し", "教えて", "相談", "比較", "どっち",
    ]
    if any(ci in clean for ci in consultation_indicators) or clean.endswith("？") or clean.endswith("?"):
        return {"action": None, "reason": "Consultation or opinion inquiry"}

    # 5. Determine Buy vs Sell Side
    is_sell = False
    is_buy = False

    # Explicit past/completed or declarative trade words
    sell_keywords = ["売った", "売りました", "利確", "損切り", "ロスカット", "手仕舞い", "手放した", "売却", "決済"]
    buy_keywords = ["買った", "買いました", "買い増し", "ナンピン", "購入", "買付", "新規買"]

    for kw in sell_keywords:
        if kw in clean:
            is_sell = True
            break
    if not is_sell:
        for kw in buy_keywords:
            if kw in clean:
                is_buy = True
                break

    # Also detect bare "買い" or "売り" if accompanied by numeric quantity/price (e.g. '7012 100株 5000円 買い')
    has_quantity_or_price = bool(re.search(r"(\d+\s*株|@\s*\d+|\d+\s*円|全株|すべて)", clean))
    if not is_sell and not is_buy and has_quantity_or_price:
        if "売り" in clean or "売" in clean:
            is_sell = True
        elif "買い" in clean or "買" in clean:
            is_buy = True

    if not is_sell and not is_buy:
        return {"action": None, "reason": "No trading intent recognized"}

    # Require at least shares or price to be explicitly stated for recording
    if not has_quantity_or_price:
        return {"action": None, "reason": "Trade record must include shares or price"}

    action = "SELL" if is_sell else "BUY"

    # Extract Shares
    shares: int | None = None
    if re.search(r"(全株|すべて|全部|全売却|全決済)", clean):
        shares = None  # Full close
    else:
        m_shares = re.search(r"(\d+)\s*株", clean)
        if m_shares:
            shares = int(m_shares.group(1))

    # Extract Price
    price: float | None = None
    # Pattern A: @2750 or @ 2750
    m_at_price = re.search(r"@\s*(\d+(?:\.\d+)?)", clean)
    if m_at_price:
        price = float(m_at_price.group(1))
    else:
        # Pattern B: 2750円, 2750.5円
        m_yen_price = re.search(r"(\d+(?:\.\d+)?)\s*円", clean)
        if m_yen_price:
            price = float(m_yen_price.group(1))
        else:
            # Pattern C: 5000で or 5000にて
            m_de_price = re.search(r"(\d{3,7}(?:\.\d+)?)\s*(?:で|にて)", clean)
            if m_de_price:
                price = float(m_de_price.group(1))

    # Extract Stock / Ticker
    # Remove shares and price from text to isolate ticker/stock name
    isolated_text = clean
    if m_at_price:
        isolated_text = isolated_text.replace(m_at_price.group(0), " ")
    if "円" in isolated_text:
        isolated_text = re.sub(r"\d+(?:\.\d+)?\s*円(?:で)?", " ", isolated_text)
    isolated_text = re.sub(r"\d+\s*株", " ", isolated_text)
    for kw in sell_keywords + buy_keywords + ["全株", "すべて", "全部", "で", "を", "にて", "購入", "売却"]:
        isolated_text = isolated_text.replace(kw, " ")

    resolved = _resolve_stock_symbol(isolated_text.strip())
    if not resolved:
        # Fallback search in entire original clean text
        resolved = _resolve_stock_symbol(clean)

    if not resolved:
        return {"action": None, "reason": "Could not identify stock code or name"}

    ticker, name = resolved

    # Fallback price to market price if not specified
    if price is None:
        market_p = _get_latest_market_price(ticker)
        if market_p:
            price = market_p

    # Fallback shares for buy (default 100 shares if unspecified)
    if action == "BUY" and shares is None:
        shares = 100

    return {
        "action": action,
        "ticker": ticker,
        "name": name,
        "shares": shares,
        "price": price,
    }


# ---------------------------------------------------------------------------
# 4. AI Risk Diagnostic Engine for Holdings (AI Guardman)
# ---------------------------------------------------------------------------

def diagnose_portfolio_risk(
    db_path: Path | str | None = None,
    summary_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Diagnose all open user holdings against AI models, earnings calendar, and technical overheating.

    Produces rating:
    - 🔴 急落警戒・撤退推奨: Downward probability high (p_up < 0.40), upcoming earnings mine, or severe loss
    - 🟡 利確検討: High profit buffer (>= +10%), jumping catch exhaustion, or neutral/stagnant
    - 🟢 継続保有OK: Sound quant probability (p_up >= 0.50), no earnings risk
    """
    init_portfolio_db(db_path)
    summary = get_portfolio_summary(db_path=db_path)
    positions = summary.get("positions", [])

    if not positions:
        return {
            "status": "EMPTY",
            "message": "現在保有中の銘柄はありません。キャッシュ100%待機中です。",
            "positions": [],
            "risk_counts": {"red": 0, "yellow": 0, "green": 0},
        }

    # Fetch daily ranking pool if available
    rankings_map: dict[str, dict[str, Any]] = {}
    if summary_data and "all_rankings" in summary_data:
        for r in summary_data["all_rankings"]:
            if "ticker" in r:
                rankings_map[r["ticker"]] = r
    else:
        try:
            from stock_ai.cloud.notifications import get_current_pipeline_summary
            p_sum = get_current_pipeline_summary()
            for r in p_sum.get("all_rankings", []):
                if "ticker" in r:
                    rankings_map[r["ticker"]] = r
        except Exception:
            pass

    diagnosed_positions: list[dict[str, Any]] = []
    red_count = 0
    yellow_count = 0
    green_count = 0

    for pos in positions:
        ticker = pos["ticker"]
        name = pos["name"]
        shares = pos["shares"]
        avg_price = pos["avg_price"]
        current_price = pos["current_price"]
        unrealized_pnl = pos["unrealized_pnl"]
        unrealized_pnl_pct = pos["unrealized_pnl_pct"]

        # Check ranking metrics
        r_info = rankings_map.get(ticker, {})
        p_up = r_info.get("probability_up")
        pred_intra = r_info.get("pred_intraday_return")
        earnings_risk = bool(r_info.get("earnings_risk"))
        jc_risk = bool(r_info.get("jumping_catch_risk"))

        # If not in rankings pool, query modules directly
        if p_up is None:
            try:
                from stock_ai.cloud.discord_bot import analyze_single_ticker
                single = analyze_single_ticker(ticker)
                if single:
                    p_up = single.get("p_overnight", 0.50)
                    pred_intra = single.get("pred_intra_ret", 0.0)
                    earnings_risk = single.get("earnings_flag", False)
                    jc_risk = single.get("jc_flag", False)
            except Exception:
                p_up = 0.50

        p_up_val = float(p_up) if p_up is not None else 0.50

        # Diagnosis logic
        reasons: list[str] = []
        action_advice: str = ""

        if earnings_risk:
            reasons.append("⚠️ 直前決算発表予定あり（決算跨ぎギャンブル急落リスク）")
        if jc_risk:
            reasons.append("⚠️ 高値掴み・ジャンピングキャッチ過熱シグナル検知")
        if p_up_val < 0.40:
            reasons.append(f"📉 AI明朝下落警戒 (上昇確率 {p_up_val:.1%} で下落優位)")
        if unrealized_pnl_pct <= -0.07:
            reasons.append(f"🛑 含み損拡大 ({unrealized_pnl_pct:.1%}) により損切り規律ライン接近")

        # Grade determination
        if earnings_risk or p_up_val < 0.40 or unrealized_pnl_pct <= -0.07:
            grade = "RED"
            status_label = "🔴 急落警戒・撤退推奨"
            action_advice = "早めの利益確定または逆指値（ロスカット）設定を推奨します。"
            red_count += 1
        elif jc_risk or unrealized_pnl_pct >= 0.10 or p_up_val < 0.48:
            grade = "YELLOW"
            status_label = "🟡 利確検討・様子見"
            if unrealized_pnl_pct >= 0.10:
                reasons.append(f"🎉 含み益 {unrealized_pnl_pct:+.1%} 到達（利確検討ゾーン）")
            action_advice = "部分利確で確定益を確保するか、ストップ高値引き上げを推奨します。"
            yellow_count += 1
        else:
            grade = "GREEN"
            status_label = "🟢 継続保有OK"
            reasons.append(f"安定推移（明朝上昇確率: {p_up_val:.1%} / 重大リスクなし）")
            action_advice = "クオンツモデル上は健全です。ターゲット価格までホールド継続。"
            green_count += 1

        diagnosed_positions.append({
            "ticker": ticker,
            "name": name,
            "shares": shares,
            "avg_price": avg_price,
            "current_price": current_price,
            "unrealized_pnl": unrealized_pnl,
            "unrealized_pnl_pct": unrealized_pnl_pct,
            "p_up": p_up_val,
            "pred_intra": pred_intra,
            "earnings_risk": earnings_risk,
            "jc_risk": jc_risk,
            "grade": grade,
            "status_label": status_label,
            "reasons": reasons,
            "action_advice": action_advice,
        })

    return {
        "status": "SUCCESS",
        "positions_count": len(positions),
        "diagnosed_positions": diagnosed_positions,
        "risk_counts": {
            "red": red_count,
            "yellow": yellow_count,
            "green": green_count,
        },
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# 5. Discord Rich Response Payload Builders
# ---------------------------------------------------------------------------

def build_buy_feedback_payload(buy_res: dict[str, Any]) -> dict[str, Any]:
    """Build response payload immediately after recording a buy trade."""
    now_ts = datetime.now(timezone.utc).isoformat()
    ticker = buy_res["ticker"]
    name = buy_res["name"]
    shares = buy_res["shares"]
    price = buy_res["price"]
    total_shares = buy_res["total_shares"]
    avg_price = buy_res["avg_price"]
    status = buy_res["position_status"]

    # Fetch AI prediction feedback
    p_up_str = "52.0%"
    target_price = round(price * 1.035, 1)
    stop_loss = round(price * 0.965, 1)
    intra_decision = "⚪ レンジ様子見"
    try:
        from stock_ai.cloud.discord_bot import analyze_single_ticker
        analysis = analyze_single_ticker(ticker)
        if analysis:
            p_val = analysis.get("p_overnight", 0.52)
            p_up_str = f"{p_val:.1%}"
            if p_val >= 0.55:
                intra_decision = "🟢 寄付き買いエントリー優位"
            elif p_val <= 0.45:
                intra_decision = "🔴 売り・下落警戒"
            if analysis.get("upper_target"):
                target_price = analysis["upper_target"]
            if analysis.get("stop_loss"):
                stop_loss = analysis["stop_loss"]
    except Exception:
        pass

    status_tag = "新規" if status == "NEW" else "買い増し"
    content = (
        f"✅ **【買付記帳完了】`{ticker}` {name}**\n"
        f"🛒 約定: **`{shares:,}株 @{price:,.1f}円`** ({status_tag}) ➔ 保有計 **`{total_shares:,}株`**（平均 **`{avg_price:,.1f}円`**）\n\n"
        f"> 🌙 Overnight確率: **`{p_up_str}`**\n"
        f"> ⚡ デイトレ判定: **`{intra_decision}`**\n"
        f"> 🎯 上値ターゲット: **`{target_price:,.1f}円`** (`+3.5%`)\n"
        f"> 🛡️ 推奨損切ライン: **`{stop_loss:,.1f}円`** (`-3.5%`)"
    )

    embed = {
        "title": f"🛒 買付記帳完了: {name} (`{ticker}`)",
        "description": f"約定: **`{shares:,}株 @{price:,.1f}円`** | 保有計: **`{total_shares:,}株`**",
        "color": 3066993,  # Emerald Green
        "fields": [
            {
                "name": "📦 更新後ポジション",
                "value": f"> 保有株数: **`{total_shares:,}株`**\n> 平均取得単価: **`{avg_price:,.1f}円`**\n> 投下元本: **`{total_shares * avg_price:,.0f}円`**",
                "inline": False,
            },
            {
                "name": "🤖 最新AIクオンツ見立て",
                "value": (
                    f"> 🌙 Overnight確率: **`{p_up_str}`**\n"
                    f"> ⚡ デイトレ判定: **`{intra_decision}`**\n"
                    f"> 🎯 上値ターゲット: **`{target_price:,.1f}円`** (`+3.5%`)\n"
                    f"> 🛡️ 推奨損切りライン: **`{stop_loss:,.1f}円`** (`-3.5%`)"
                ),
                "inline": False,
            },
        ],
        "footer": {"text": "Stock AI Personal Portfolio Ledger | モバイル最適化"},
        "timestamp": now_ts,
    }

    return {
        "username": "📊 マイ資産管理コンシェルジュ AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_sell_feedback_payload(sell_res: dict[str, Any]) -> dict[str, Any]:
    """Build response payload immediately after recording a sell trade."""
    now_ts = datetime.now(timezone.utc).isoformat()
    ticker = sell_res["ticker"]
    name = sell_res["name"]
    sold_shares = sell_res["sold_shares"]
    price = sell_res["price"]
    realized_pnl = sell_res["realized_pnl"]
    realized_pnl_pct = sell_res["realized_pnl_pct"]
    is_full_close = sell_res["is_full_close"]
    remaining_shares = sell_res["remaining_shares"]
    total_realized = sell_res["total_realized_pnl"]

    pnl_sign = "+" if realized_pnl >= 0 else ""
    pnl_icon = "🎉" if realized_pnl >= 0 else "🛑"
    color = 15844367 if realized_pnl >= 0 else 15158332

    close_tag = "全株決済完了" if is_full_close else f"一部売却（残: {remaining_shares:,}株）"

    content = (
        f"💰 **【決済記帳完了】`{ticker}` {name}**\n"
        f"📉 売却: **`{sold_shares:,}株 @{price:,.1f}円`** ({close_tag})\n\n"
        f"> ⚖️ 今回確定損益: **{pnl_icon} `{pnl_sign}{realized_pnl:,.0f} 円` (`{realized_pnl_pct:+.1%}`)**\n"
        f"> 🏆 累計確定収支: **`{total_realized:+,.0f} 円`**"
    )

    embed = {
        "title": f"💰 決済記帳完了: {name} (`{ticker}`)",
        "description": f"売却: **`{sold_shares:,}株 @{price:,.1f}円`** | 状態: **{close_tag}**",
        "color": color,
        "fields": [
            {
                "name": "⚖️ 今回の確定損益",
                "value": f"> 実現損益: **`{pnl_sign}{realized_pnl:,.0f} 円`** (`{realized_pnl_pct:+.2%}`)\n> 受渡代金: **`{sold_shares * price:,.0f}円`**",
                "inline": False,
            },
            {
                "name": "🏆 通算累計実現損益",
                "value": f"> 通算確定収支: **`{total_realized:+,.0f} 円`**",
                "inline": False,
            },
        ],
        "footer": {"text": "Stock AI Personal Portfolio Ledger | モバイル最適化"},
        "timestamp": now_ts,
    }

    return {
        "username": "📊 マイ資産管理コンシェルジュ AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_portfolio_overview_payload(db_path: Path | str | None = None) -> dict[str, Any]:
    """Build comprehensive portfolio overview card."""
    now_ts = datetime.now(timezone.utc).isoformat()
    summary = get_portfolio_summary(db_path=db_path)

    pos_count = summary["positions_count"]
    total_book = summary["total_book_value"]
    total_market = summary["total_market_value"]
    total_unrealized = summary["total_unrealized_pnl"]
    total_unrealized_pct = summary["total_unrealized_pnl_pct"]
    total_realized = summary["total_realized_pnl"]
    total_comb = summary["total_combined_pnl"]
    win_rate = summary["win_rate"]
    pf = summary["profit_factor"]
    positions = summary["positions"]

    u_sign = "+" if total_unrealized >= 0 else ""
    c_sign = "+" if total_comb >= 0 else ""

    # Mobile-first list
    if pos_count == 0:
        pos_desc = "現在保有中の銘柄はありません。（全額キャッシュ待機中）"
        holdings_lines = ["（保有ポジションなし・キャッシュ100%待機）"]
    else:
        holdings_lines = []
        embed_lines = []
        for p in positions:
            p_sign = "+" if p["unrealized_pnl"] >= 0 else ""
            h_line = (
                f"・`{p['ticker']}` **{p['name']}** ({p['shares']:,}株)\n"
                f"  取得 `{p['avg_price']:,.0f}円` ➔ 現在 `{p['current_price']:,.0f}円` (**`{p_sign}{p['unrealized_pnl']:,.0f}円` / `{p['unrealized_pnl_pct']:+.1%}`**)"
            )
            holdings_lines.append(h_line)
            embed_lines.append(
                f"> **`{p['ticker']}` {p['name']}** ({p['shares']:,}株)\n"
                f"> 取得 `{p['avg_price']:,.0f}円` ➔ 現在 `{p['current_price']:,.0f}円` (**`{p_sign}{p['unrealized_pnl']:,.0f}円` / `{p['unrealized_pnl_pct']:+.1%}`**)"
            )
        pos_desc = "\n\n".join(embed_lines)

    content = (
        f"💼 **総合ポートフォリオ収支**: **`{c_sign}{total_comb:,.0f}円`**\n"
        f"（確定: `{total_realized:+,.0f}円` ／ 含み: `{u_sign}{total_unrealized:,.0f}円`）\n"
        f"📊 勝率: `{win_rate:.1%}` ({summary['win_trades_count']}勝{summary['loss_trades_count']}敗) ｜ PF: `{pf:.2f}`\n\n"
        f"**【保有ポジション】** ({pos_count}銘柄)\n"
        + "\n".join(holdings_lines)
    )

    embed = {
        "title": "📊 マイポートフォリオ 総合収支レポート",
        "description": f"トータル総合収支: **`{c_sign}{total_comb:,.0f} 円`**",
        "color": 3447003,  # Blue
        "fields": [
            {
                "name": "💼 収支サマリー",
                "value": (
                    f"> 総合収支: **`{c_sign}{total_comb:,.0f} 円`**\n"
                    f"> 含み損益: **`{u_sign}{total_unrealized:,.0f} 円`** (`{total_unrealized_pct:+.1%}`)\n"
                    f"> 確定損益: **`{total_realized:+,.0f} 円`**\n"
                    f"> 評価額計: **`{total_market:,.0f} 円`** (投下: `{total_book:,.0f} 円`)\n"
                    f"> 通算勝率: **`{win_rate:.1%}`** | PF: **`{pf:.2f}`**"
                ),
                "inline": False,
            },
            {
                "name": f"📦 現在の保有銘柄 ({pos_count}銘柄)",
                "value": pos_desc[:1020],
                "inline": False,
            },
        ],
        "footer": {"text": "Stock AI Personal Portfolio Manager | モバイル最適化"},
        "timestamp": now_ts,
    }

    return {
        "username": "📊 マイ資産管理コンシェルジュ AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_risk_diagnosis_payload(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Build risk assessment card for holdings."""
    now_ts = datetime.now(timezone.utc).isoformat()
    if diagnosis.get("status") == "EMPTY":
        content = "🛡️ **【保有株リスク診断】** 現在保有中の銘柄はありません。全額キャッシュ待機でリスクゼロです。"
        embed = {
            "title": "🛡️ 保有株リスク診断 (AIガードマン)",
            "description": "現在保有中の銘柄はありません。キャッシュ100%待機中です。",
            "color": 3066993,
            "footer": {"text": "Stock AI Risk Guardian"},
            "timestamp": now_ts,
        }
        return {"username": "🚨 AIリスクガードマン", "content": content, "text": content, "embeds": [embed]}

    counts = diagnosis["risk_counts"]
    reds = counts["red"]
    yellows = counts["yellow"]
    greens = counts["green"]

    fields = []
    for item in diagnosis["diagnosed_positions"]:
        t = item["ticker"]
        n = item["name"]
        lbl = item["status_label"]
        pnl = item["unrealized_pnl"]
        pnl_pct = item["unrealized_pnl_pct"]
        pnl_str = f"{pnl:+,.0f}円 / {pnl_pct:+.1%}"
        reasons_str = "、".join(item["reasons"]) if item.get("reasons") else "特記事項なし"
        adv = item["action_advice"]
        shares = item.get("shares", 100)
        curr = item.get("current_price", 0.0)
        avg = item.get("avg_price", curr)

        icon = "🔴" if "急落" in lbl or "警戒" in lbl else ("🟡" if "注意" in lbl or "利確" in lbl else "🟢")

        fields.append({
            "name": f"{icon} **{n}** (`{t}`) — {shares:,}株",
            "value": (
                f"> 💰 取得 `{avg:,.0f}円` ➔ 現在 `{curr:,.0f}円` (**`{pnl_str}`**)\n"
                f"> 🤖 判定: **{lbl}** ({reasons_str})\n"
                f"> 💡 **推奨**: {adv}"
            ),
            "inline": False,
        })

    color = 15158332 if reds > 0 else (15844367 if yellows > 0 else 3066993)
    header_status = f"🔴 警戒 {reds} ｜ 🟡 注意 {yellows} ｜ 🟢 順調 {greens}"

    content = (
        f"🚨 **【保有株リアルタイム危険度診断】**\n"
        f"📊 **状況**: **{header_status}**\n"
        f"💡 {'⚠️ 急落警戒または決算注意の保有銘柄があります。早めの手仕舞いを検討してください。' if reds > 0 else '健全なポートフォリオ状況が維持されています。'}"
    )

    embed = {
        "title": "🛡️ 保有ポジション＆AIリスク診断",
        "description": "翌朝AI Overnight予測、直前決算、テクニカル過熱天井を一括スクリーニング",
        "color": color,
        "fields": fields,
        "footer": {"text": "Stock AI Risk Guardian | 保有銘柄セーフティガード"},
        "timestamp": now_ts,
    }

    return {
        "username": "🚨 AIリスクガードマン",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def build_trade_history_payload(db_path: Path | str | None = None) -> dict[str, Any]:
    """Build recent trade history card."""
    now_ts = datetime.now(timezone.utc).isoformat()
    trades = get_trade_history(limit=10, db_path=db_path)
    if not trades:
        content = "📝 **【取引履歴】** まだ記帳された取引履歴はありません。"
        embed = {
            "title": "📝 取引履歴",
            "description": "『川重 100株 5000円で買った』のように書き込むと履歴が記録されます。",
            "color": 3447003,
            "footer": {"text": "Stock AI Trade Ledger"},
            "timestamp": now_ts,
        }
        return {"username": "📊 マイ資産管理コンシェルジュ AI", "content": content, "text": content, "embeds": [embed]}

    lines = []
    for t in trades:
        side_tag = "🛒 買" if t["side"] == "BUY" else "💰 売"
        pnl_str = ""
        if t["side"] == "SELL":
            pnl_str = f" | 損益: **{t['realized_pnl']:+,.0f}円** ({t['realized_pnl_pct']:+.1%})"
        lines.append(f"・`{t['trade_date'][:16]}` {side_tag} `{t['ticker']}` **{t['name']}**: {t['shares']:,}株 @{t['price']:,.1f}円{pnl_str}")

    content = f"📝 **【直近の取引履歴 (最新10件)】**\n" + "\n".join(lines)
    embed = {
        "title": "📝 直近の売買取引履歴 (最新10件)",
        "description": "台帳に記録された直近の約定履歴です。",
        "color": 3447003,
        "fields": [
            {"name": "売買レコード", "value": "\n".join(lines), "inline": False},
        ],
        "footer": {"text": "Stock AI Personal Portfolio Ledger"},
        "timestamp": now_ts,
    }
    return {"username": "📊 マイ資産管理コンシェルジュ AI", "content": content, "text": content, "embeds": [embed]}


def build_cancel_feedback_payload(cancel_res: dict[str, Any]) -> dict[str, Any]:
    """Build response payload when a trade is undone."""
    now_ts = datetime.now(timezone.utc).isoformat()
    ticker = cancel_res["ticker"]
    name = cancel_res["name"]
    side = cancel_res["side"]
    shares = cancel_res["shares"]
    price = cancel_res["price"]
    recon = cancel_res["reconstructed_shares"]

    content = (
        f"↩️ **【取引取消完了】** 直前の取引を取り消しました。\n"
        f"・取消対象: `{ticker}` {name} ({side} {shares:,}株 @{price:,.1f}円)\n"
        f"・現在の再計算保有株数: **{recon:,}株**"
    )

    embed = {
        "title": "↩️ 直前取引の取り消し・ポジション再構成完了",
        "description": "直前の取引レコードを削除し、保有ポジションと平均取得単価を自動再計算しました。",
        "color": 15844367,  # Gold
        "fields": [
            {"name": "取り消した取引", "value": f"`{ticker}` {name}\n{side} **{shares:,}株** @{price:,.1f}円", "inline": True},
            {"name": "現在の保有株数", "value": f"**{recon:,}株**", "inline": True},
        ],
        "footer": {"text": "Stock AI Personal Portfolio Ledger | 取引取消"},
        "timestamp": now_ts,
    }

    return {
        "username": "📊 マイ資産管理コンシェルジュ AI",
        "content": content,
        "text": content,
        "embeds": [embed],
    }


def handle_user_portfolio_message(
    text: str,
    db_path: Path | str | None = None,
) -> list[dict[str, Any]] | None:
    """High-level entrypoint for Discord bot to parse and process portfolio/trade messages."""
    parsed = parse_trade_message(text)
    action = parsed.get("action")

    if action is None:
        return None

    if action == "BUY":
        try:
            buy_res = record_buy(
                ticker=parsed["ticker"],
                name=parsed["name"],
                shares=parsed["shares"],
                price=parsed["price"],
                db_path=db_path,
            )
            return [build_buy_feedback_payload(buy_res)]
        except Exception as exc:
            err_msg = f"⚠️ 買付記帳エラー: {exc}"
            return [{"username": "📊 マイ資産管理コンシェルジュ AI", "content": err_msg, "text": err_msg}]

    elif action == "SELL":
        try:
            sell_res = record_sell(
                ticker=parsed["ticker"],
                shares=parsed.get("shares"),
                price=parsed["price"],
                db_path=db_path,
            )
            return [build_sell_feedback_payload(sell_res)]
        except Exception as exc:
            err_msg = f"⚠️ 売却記帳エラー: {exc}"
            return [{"username": "📊 マイ資産管理コンシェルジュ AI", "content": err_msg, "text": err_msg}]

    elif action == "PORTFOLIO":
        return [build_portfolio_overview_payload(db_path=db_path)]

    elif action == "HISTORY":
        return [build_trade_history_payload(db_path=db_path)]

    elif action == "DIAGNOSE":
        diag = diagnose_portfolio_risk(db_path=db_path)
        return [build_risk_diagnosis_payload(diag)]

    elif action == "CANCEL":
        try:
            cancel_res = cancel_last_trade(db_path=db_path)
            return [build_cancel_feedback_payload(cancel_res)]
        except Exception as exc:
            err_msg = f"⚠️ 取消エラー: {exc}"
            return [{"username": "📊 マイ資産管理コンシェルジュ AI", "content": err_msg, "text": err_msg}]

    return None

