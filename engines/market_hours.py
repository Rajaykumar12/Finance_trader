"""Market session utilities — US Eastern Time awareness.

Provides helpers for:
  - Checking if US equity market is open (9:30–16:00 ET, Mon–Fri)
  - Detecting pre-market hours (4:00–9:30 ET)
  - Determining when to flatten positions (15:45 ET = 15 min before close)
  - Computing minutes to open/close
  - Checking if today is a trading day (weekday only — does not check holidays)

All times are US Eastern Time (America/New_York), which automatically
handles EST/EDT daylight saving transitions via the standard library
zoneinfo module (Python 3.9+, no external dependencies).
"""

from __future__ import annotations

from datetime import datetime, time, date, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")

# ── Session times (US Eastern) ─────────────────────────────────────────
PREMARKET_START   = time(4, 0)
MARKET_OPEN       = time(9, 30)
EOD_FLAT_TRIGGER  = time(15, 45)   # start closing positions 15 min before close
MARKET_CLOSE      = time(16, 0)
AFTERHOURS_END    = time(20, 0)

# Pre-market scan runs at this time to build the day's watchlist
PREMARKET_SCAN_TIME = time(9, 15)


def now_et() -> datetime:
    """Current wall-clock time in US Eastern."""
    return datetime.now(ET)


def market_date() -> date:
    """Today's date in US Eastern (relevant when it's past midnight UTC but still 'today' ET)."""
    return now_et().date()


def is_weekend() -> bool:
    return now_et().weekday() >= 5  # Saturday=5, Sunday=6


def is_trading_day() -> bool:
    """True if today is a weekday. Does NOT account for US market holidays."""
    return not is_weekend()


def is_premarket() -> bool:
    """True during pre-market hours: 4:00–9:30 ET on a trading day."""
    if not is_trading_day():
        return False
    t = now_et().time()
    return PREMARKET_START <= t < MARKET_OPEN


def is_market_open() -> bool:
    """True during regular trading hours: 9:30–16:00 ET on a trading day."""
    if not is_trading_day():
        return False
    t = now_et().time()
    return MARKET_OPEN <= t < MARKET_CLOSE


def is_afterhours() -> bool:
    """True during after-hours: 16:00–20:00 ET on a trading day."""
    if not is_trading_day():
        return False
    t = now_et().time()
    return MARKET_CLOSE <= t < AFTERHOURS_END


def should_flatten_eod() -> bool:
    """True when it's time to close all positions for EOD safety.

    Triggers at 15:45 ET — 15 minutes before the close — giving
    enough time for market orders to fill before 16:00.
    """
    if not is_trading_day():
        return False
    t = now_et().time()
    return EOD_FLAT_TRIGGER <= t < MARKET_CLOSE


def minutes_to_open() -> float:
    """Minutes until next market open (9:30 ET). Negative if market is open."""
    now = now_et()
    today_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= today_open:
        # Market already opened today — next open is next trading day
        next_open = today_open + timedelta(days=1)
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)
        return (next_open - now).total_seconds() / 60.0
    return (today_open - now).total_seconds() / 60.0


def minutes_to_close() -> float:
    """Minutes until market close (16:00 ET). Negative if market is closed."""
    now = now_et()
    today_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return (today_close - now).total_seconds() / 60.0


def is_premarket_scan_window() -> bool:
    """True during the pre-market scan window (9:10–9:25 ET).

    Used to trigger the watchlist preparation scan before open.
    """
    if not is_trading_day():
        return False
    t = now_et().time()
    return time(9, 10) <= t <= time(9, 25)


def session_label() -> str:
    """Human-readable label for the current session."""
    if is_weekend():
        return "WEEKEND"
    if not is_trading_day():
        return "CLOSED"
    t = now_et().time()
    if t < PREMARKET_START:
        return "OVERNIGHT"
    if t < MARKET_OPEN:
        return "PRE-MARKET"
    if t < EOD_FLAT_TRIGGER:
        return "REGULAR"
    if t < MARKET_CLOSE:
        return "EOD-FLATTEN"
    if t < AFTERHOURS_END:
        return "AFTER-HOURS"
    return "CLOSED"


def session_summary() -> dict:
    """Return a dict describing the current market session state."""
    return {
        "session":           session_label(),
        "is_market_open":    is_market_open(),
        "is_premarket":      is_premarket(),
        "is_afterhours":     is_afterhours(),
        "should_flatten_eod": should_flatten_eod(),
        "minutes_to_open":   round(minutes_to_open(), 1) if not is_market_open() else None,
        "minutes_to_close":  round(minutes_to_close(), 1) if is_market_open() else None,
        "et_time":           now_et().strftime("%H:%M:%S ET"),
        "date":              market_date().isoformat(),
    }
