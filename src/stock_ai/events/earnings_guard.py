from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EARNINGS_CACHE_PATH = Path("data/earnings_calendar_cache.json")


def _parse_date_to_str(val: Any) -> str | None:
    """Safely convert various date representations to YYYY-MM-DD string."""
    if val is None or pd.isna(val):
        return None
    if isinstance(val, (datetime, date)):
        return val.strftime("%Y-%m-%d")
    try:
        ts = pd.to_datetime(val)
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def fetch_next_earnings_date(
    ticker: str,
    cache_path: Path | str | None = DEFAULT_EARNINGS_CACHE_PATH,
    force_refresh: bool = False,
    timeout: float = 8.0,
) -> str | None:
    """Safely fetch upcoming earnings date from yfinance with cache fallback.

    Never raises exceptions; safely returns None on network/rate-limit/parser errors.
    """
    cache_file = Path(cache_path) if cache_path else None
    cache: dict[str, dict[str, Any]] = {}

    if cache_file and cache_file.exists():
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    if not force_refresh and ticker in cache:
        cached_entry = cache[ticker]
        cached_date = cached_entry.get("earnings_date")
        if cached_date:
            return str(cached_date)

    # Fetch from yfinance safely
    earnings_date_str: str | None = None
    try:
        import yfinance as yf

        t = yf.Ticker(ticker)
        # Try calendar property first (fast, dictionary-based)
        cal = getattr(t, "calendar", None)
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and len(ed) > 0:
                earnings_date_str = _parse_date_to_str(ed[0])
            elif ed:
                earnings_date_str = _parse_date_to_str(ed)
        elif isinstance(cal, pd.DataFrame) and not cal.empty:
            if "Earnings Date" in cal.index:
                val = cal.loc["Earnings Date"].iloc[0]
                earnings_date_str = _parse_date_to_str(val)

        # Fallback to get_earnings_dates if lxml is available and calendar was empty
        if not earnings_date_str:
            try:
                ed_df = t.get_earnings_dates(limit=2)
                if isinstance(ed_df, pd.DataFrame) and not ed_df.empty:
                    first_idx = ed_df.index[0]
                    earnings_date_str = _parse_date_to_str(first_idx)
            except Exception:
                pass
    except Exception as exc:
        logger.debug("Failed to fetch earnings calendar for %s: %s", ticker, exc)

    if earnings_date_str is None and ticker in cache:
        return cache[ticker].get("earnings_date")

    if cache_file and earnings_date_str:
        try:
            cache[ticker] = {
                "earnings_date": earnings_date_str,
                "updated_at": datetime.now().isoformat(),
            }
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            logger.debug("Failed writing earnings cache: %s", exc)

    return earnings_date_str


def check_earnings_risk(
    ticker: str,
    ref_date: str | date | datetime | pd.Timestamp | None = None,
    earnings_date: str | None = None,
    cache_path: Path | str | None = DEFAULT_EARNINGS_CACHE_PATH,
) -> tuple[bool, str | None]:
    """Determine whether holding an overnight position carries earnings announcement gap risk.

    Returns:
        (earnings_risk: bool, earnings_date: str | None)
        earnings_risk is True if earnings date is today or the immediate next trading day.
    """
    if not earnings_date:
        earnings_date = fetch_next_earnings_date(ticker, cache_path=cache_path)

    if not earnings_date:
        return False, None

    try:
        e_dt = datetime.strptime(earnings_date, "%Y-%m-%d").date()
    except Exception:
        return False, earnings_date

    if ref_date is None:
        r_dt = date.today()
    elif isinstance(ref_date, (datetime, pd.Timestamp)):
        r_dt = ref_date.date()
    elif isinstance(ref_date, date):
        r_dt = ref_date
    else:
        try:
            r_dt = datetime.strptime(str(ref_date)[:10], "%Y-%m-%d").date()
        except Exception:
            r_dt = date.today()

    diff_days = (e_dt - r_dt).days

    # Same day earnings announcement (typically after 15:00 close in Japan):
    if diff_days == 0:
        return True, earnings_date

    # Next trading day:
    # If ref_date is Friday (weekday == 4), Saturday (5), or Sunday (6), next business day spans to Monday
    is_friday = (r_dt.weekday() == 4)
    next_biz_threshold = 3 if is_friday else 1

    if 0 < diff_days <= next_biz_threshold:
        return True, earnings_date

    return False, earnings_date


def batch_check_earnings_guard(
    tickers: list[str],
    ref_date: str | date | datetime | pd.Timestamp | None = None,
    cache_path: Path | str | None = DEFAULT_EARNINGS_CACHE_PATH,
) -> dict[str, dict[str, Any]]:
    """Batch evaluate earnings guard for a list of tickers."""
    results: dict[str, dict[str, Any]] = {}
    for code in tickers:
        risk, e_date = check_earnings_risk(code, ref_date=ref_date, cache_path=cache_path)
        results[code] = {
            "earnings_risk": risk,
            "earnings_date": e_date,
            "status": "⚠️ 決算警戒" if risk else "✅ 通常",
        }
    return results
