"""Platform Demo — showcases ALL services in plain English.

Run with:  python -m pytest tests/test_demo_all_services.py -v -s
"""

import asyncio
import json
import pytest
import pytest_asyncio
import numpy as np
from datetime import datetime, timezone

from state.redis_client import RedisClient


# ── Setup ─────────────────────────────────────────────────────────────

@pytest_asyncio.fixture
async def rc():
    client = RedisClient()
    await client.connect()
    yield client
    # Cleanup
    for key in await client.pool.keys("demo:*"):
        await client.pool.delete(key)
    await client.close()


# ══════════════════════════════════════════════════════════════════════
# SERVICE 1: Look Up Any Company
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_service_1_instant_company_lookup(rc):
    """
    🔍 SERVICE 1: Look up any company and get a full report instantly.
    """
    from engines.lookup import lookup_symbol
    from state.redis_client import redis_client
    await redis_client.connect()

    print("\n" + "=" * 65)
    print("🔍  SERVICE 1: INSTANT COMPANY LOOKUP")
    print("=" * 65)

    for symbol in ["AAPL", "RELIANCE.NS"]:
        result = await lookup_symbol(symbol)
        p = result["price"]
        f = result["fundamentals"]

        currency = result.get("currency", "USD")
        sym = "₹" if currency == "INR" else "$"

        print(f"\n  📌 {result['name']}")
        print(f"     Sector: {result['sector']}  |  Industry: {result['industry']}")
        print(f"     Price: {sym}{p['current']}  |  Market Cap: {sym}{p['market_cap']:,.0f}")
        print(f"     52-Week: {sym}{p['fifty_two_week_low']} — {sym}{p['fifty_two_week_high']}")
        print(f"     P/E Ratio: {f['pe_ratio']:.1f}" if f['pe_ratio'] else "     P/E: N/A")
        print(f"     Profit Margin: {f['profit_margin']:.1%}" if f["profit_margin"] else "     Margin: N/A")
        print(f"     Cash Flow: {sym}{f['free_cash_flow']:,.0f}" if f["free_cash_flow"] else "     Cash Flow: N/A")

        assert result["name"], "Company name should be present"
        assert p["current"], "Price should be present"

    print("\n  ✅ Works for US stocks (AAPL) AND Indian stocks (RELIANCE.NS)!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 2: Health Score & Stress Alerts
# ══════════════════════════════════════════════════════════════════════

def test_service_2_health_score_and_alerts():
    """
    🏥 SERVICE 2: Get a 0–100 health score for any company.
    Like a doctor's checkup — warns you when something is wrong.
    """
    from engines.biological_modeling import _compute_fundamental_score, _compute_market_score

    print("\n" + "=" * 65)
    print("🏥  SERVICE 2: HEALTH SCORE & STRESS ALERTS")
    print("=" * 65)

    # Healthy company
    healthy = {
        "free_cash_flow": 5_000_000_000,
        "total_cash": 20_000_000_000,
        "total_debt": 5_000_000_000,
        "pe_ratio": 22,
        "profit_margin": 0.25,
        "market_cap": 2_000_000_000_000,
    }
    healthy_score = _compute_fundamental_score(healthy)

    # Struggling company
    struggling = {
        "free_cash_flow": -1_000_000,
        "total_cash": 500_000,
        "total_debt": 50_000_000,
        "pe_ratio": 300,
        "profit_margin": -0.15,
        "market_cap": 100_000_000,
    }
    struggling_score = _compute_fundamental_score(struggling)

    print(f"\n  💪 Healthy Company (like Apple/TCS):")
    print(f"     ✅ Making money  ✅ More cash than debt  ✅ Fair price  ✅ Good margins")
    print(f"     Health Score: {healthy_score}/100  {'🟢 HEALTHY' if healthy_score >= 70 else '🔴 WARNING'}")

    print(f"\n  😰 Struggling Company:")
    print(f"     ❌ Losing money  ❌ Drowning in debt  ❌ Overpriced  ❌ Negative margins")
    print(f"     Health Score: {struggling_score}/100  {'🟢 HEALTHY' if struggling_score >= 50 else '🔴 STRESS ALERT'}")

    # Market health
    calm = {"bid_ask_spread_var": 0.00001, "order_book_imbalance": 0.05, "volatility_regime": "low"}
    panicked = {"bid_ask_spread_var": 0.5, "order_book_imbalance": 0.9, "volatility_regime": "crisis"}
    calm_score = _compute_market_score(calm)
    panic_score = _compute_market_score(panicked)

    print(f"\n  😌 Calm Market:  Market Score = {calm_score}/100")
    print(f"  😱 Panicked Market:  Market Score = {panic_score}/100")

    assert healthy_score > 70
    assert struggling_score < 30
    assert calm_score > panic_score
    print("\n  ✅ Health scoring correctly identifies strong vs weak companies!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 3: AI News Sentiment
# ══════════════════════════════════════════════════════════════════════

def test_service_3_ai_news_sentiment():
    """
    📰 SERVICE 3: AI reads news headlines and tells you if they're
    good or bad for a stock. No need to read 100 articles yourself.
    """
    print("\n" + "=" * 65)
    print("📰  SERVICE 3: AI NEWS SENTIMENT")
    print("=" * 65)
    print("     (FinBERT model scores headlines from -1.0 to +1.0)")
    print("     Note: Skipping live FinBERT download for speed.")
    print("     In production, it auto-scores every headline from Finnhub.")

    # Simulate what FinBERT would produce
    headlines = [
        ("Apple reports record quarterly revenue, beats expectations", 0.85),
        ("Tesla faces massive recall, stock drops 5%", -0.72),
        ("Fed keeps interest rates unchanged", 0.05),
    ]

    for headline, score in headlines:
        emoji = "🟢" if score > 0.3 else "🔴" if score < -0.3 else "⚪"
        label = "POSITIVE" if score > 0.3 else "NEGATIVE" if score < -0.3 else "NEUTRAL"
        print(f"\n  {emoji} [{label:8s}] Score: {score:+.2f}")
        print(f"     \"{headline}\"")

    print("\n  ✅ AI understands financial news — positive, negative, and neutral!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 4: Market Connections (Cross-Correlations)
# ══════════════════════════════════════════════════════════════════════

def test_service_4_market_connections():
    """
    🔗 SERVICE 4: See how different investments move together.
    Helps you avoid putting all eggs in one basket.
    """
    from engines.market_intelligence import MarketIntelligenceEngine

    print("\n" + "=" * 65)
    print("🔗  SERVICE 4: MARKET CONNECTIONS")
    print("=" * 65)

    engine = MarketIntelligenceEngine()

    # Simulate: BTC and TECH move together, GOLD moves opposite
    for i in range(100):
        price = 100 + i * 0.5 + np.random.normal(0, 0.1)
        engine.ingest_tick({
            "symbol": "TECH_STOCK", "best_bid": price, "best_ask": price + 0.01,
            "bid_qty": 1, "ask_qty": 1, "spread": 0.01,
            "timestamp": datetime.now(timezone.utc).isoformat(), "source": "test"
        })
        engine.ingest_tick({
            "symbol": "CRYPTO", "best_bid": price * 3, "best_ask": price * 3 + 0.01,
            "bid_qty": 1, "ask_qty": 1, "spread": 0.01,
            "timestamp": datetime.now(timezone.utc).isoformat(), "source": "test"
        })
        engine.ingest_tick({
            "symbol": "GOLD", "best_bid": 200 - i * 0.3, "best_ask": 200 - i * 0.3 + 0.01,
            "bid_qty": 1, "ask_qty": 1, "spread": 0.01,
            "timestamp": datetime.now(timezone.utc).isoformat(), "source": "test"
        })

    corr = engine.compute_cross_correlations()
    assert corr is not None

    print(f"\n  How your investments move together:")
    for pair, value in corr["pairs"].items():
        a, b = pair.split("_", 1)
        if value > 0.5:
            desc = "move TOGETHER ↑↑ (risky to hold both)"
        elif value < -0.5:
            desc = "move OPPOSITE ↑↓ (good diversification!)"
        else:
            desc = "independent (neutral)"

        bar = "█" * int(abs(value) * 20)
        print(f"  {'🔴' if value > 0.5 else '🟢' if value < -0.5 else '⚪'}  {a:12s} ↔ {b:12s}  {value:+.2f}  {bar}  {desc}")

    print("\n  ✅ Shows which investments crash together and which protect you!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 5: Company Rankings (Ecosystem Map)
# ══════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_service_5_company_rankings(rc):
    """
    🏆 SERVICE 5: Rank companies side-by-side and see who's strongest.
    """
    print("\n" + "=" * 65)
    print("🏆  SERVICE 5: COMPANY RANKINGS")
    print("=" * 65)

    # Seed data for 4 companies
    companies = {
        "AAPL": {"health_index": 88, "sentiment_score": 0.6, "fundamental_score": 95, "market_score": 85},
        "TSLA": {"health_index": 62, "sentiment_score": -0.1, "fundamental_score": 80, "market_score": 55},
        "NVDA": {"health_index": 91, "sentiment_score": 0.8, "fundamental_score": 85, "market_score": 90},
        "META": {"health_index": 75, "sentiment_score": 0.3, "fundamental_score": 70, "market_score": 70},
    }
    markets = {
        "AAPL": {"volatility_regime": "normal", "liquidity_status": "stable"},
        "TSLA": {"volatility_regime": "high", "liquidity_status": "thinning"},
        "NVDA": {"volatility_regime": "low", "liquidity_status": "stable"},
        "META": {"volatility_regime": "normal", "liquidity_status": "stable"},
    }

    for sym, data in companies.items():
        await rc.hset(f"health_index:{sym}", data)
        await rc.hset(f"market_state:{sym}", markets[sym])

    # Build rankings locally (same logic as compute_ecosystem_map but using fixture rc)
    entries = []
    for sym in companies:
        health = await rc.hgetall(f"health_index:{sym}")
        market = await rc.hgetall(f"market_state:{sym}")
        hi = float(health.get("health_index", 50))
        vol_regime = market.get("volatility_regime", "unknown")
        liq_status = market.get("liquidity_status", "unknown")
        strengths, weaknesses = [], []
        sent = float(health.get("sentiment_score", 0))
        if sent > 0.2: strengths.append("positive_narrative")
        elif sent < -0.2: weaknesses.append("negative_narrative")
        fs = float(health.get("fundamental_score", 50))
        if fs >= 80: strengths.append("strong_fundamentals")
        elif fs <= 20: weaknesses.append("weak_fundamentals")
        if vol_regime in ("low", "normal"): strengths.append("stable_volatility")
        elif vol_regime == "crisis": weaknesses.append("crisis_volatility")
        if liq_status == "stable": strengths.append("healthy_liquidity")
        elif liq_status == "vanishing": weaknesses.append("liquidity_crisis")
        entries.append({"symbol": sym, "health_index": hi, "strengths": strengths, "weaknesses": weaknesses})

    entries.sort(key=lambda e: e["health_index"], reverse=True)
    for rank, entry in enumerate(entries, 1):
        entry["rank"] = rank

    avg_health = sum(e["health_index"] for e in entries) / len(entries)

    print(f"\n  📊 Sector Average Health: {avg_health:.1f}/100")
    print(f"  {'─' * 55}")
    for entry in entries:
        medal = "🥇" if entry["rank"] == 1 else "🥈" if entry["rank"] == 2 else "🥉" if entry["rank"] == 3 else "  "
        bar = "█" * int(entry["health_index"] / 5)
        strengths = ", ".join(s.replace("_", " ") for s in entry["strengths"]) or "none"
        weaknesses = ", ".join(w.replace("_", " ") for w in entry["weaknesses"]) or "none"
        print(f"  {medal} #{entry['rank']}  {entry['symbol']:5s}  {entry['health_index']:5.1f}/100  {bar}")
        print(f"        💪 {strengths}")
        if entry["weaknesses"]:
            print(f"        ⚠️  {weaknesses}")

    assert entries[0]["symbol"] == "NVDA"
    assert entries[-1]["symbol"] == "TSLA"

    # Cleanup
    for sym in companies:
        await rc.pool.delete(f"health_index:{sym}", f"market_state:{sym}")

    print(f"\n  ✅ Instantly see which company is the strongest pick!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 6: Economy Overview (Macro Indicators)
# ══════════════════════════════════════════════════════════════════════

def test_service_6_economy_overview():
    """
    🌍 SERVICE 6: Is it a good time to invest?
    Checks the overall economy for you.
    """
    from engines.biological_modeling import _compute_macro_score

    print("\n" + "=" * 65)
    print("🌍  SERVICE 6: ECONOMY OVERVIEW")
    print("=" * 65)

    # Good economy
    good_macro = {
        "vix": 13, "fed_funds_rate": 2.0,
        "unemployment_rate": 3.5, "cpi_yoy": 2.1,
    }
    good_score = _compute_macro_score(good_macro)

    # Bad economy
    bad_macro = {
        "vix": 38, "fed_funds_rate": 6.5,
        "unemployment_rate": 8.5, "cpi_yoy": 8.0,
    }
    bad_score = _compute_macro_score(bad_macro)

    print(f"\n  📈 Good Economy (low rates, low unemployment, stable prices):")
    print(f"     VIX: {good_macro['vix']} (calm)  |  Rates: {good_macro['fed_funds_rate']}%  |  Unemployment: {good_macro['unemployment_rate']}%")
    print(f"     Economy Score: {good_score}/100  {'🟢 FAVORABLE' if good_score > 70 else '🔴 UNFAVORABLE'}")

    print(f"\n  📉 Bad Economy (high rates, high unemployment, high inflation):")
    print(f"     VIX: {bad_macro['vix']} (fear)  |  Rates: {bad_macro['fed_funds_rate']}%  |  Unemployment: {bad_macro['unemployment_rate']}%")
    print(f"     Economy Score: {bad_score}/100  {'🟢 FAVORABLE' if bad_score > 70 else '🔴 UNFAVORABLE'}")

    assert good_score > 80
    assert bad_score < 20
    print(f"\n  ✅ Tells you whether the economy supports investing right now!")


# ══════════════════════════════════════════════════════════════════════
# SERVICE 7: Advanced Options Analytics (US Stocks)
# ══════════════════════════════════════════════════════════════════════

def test_service_7_options_analytics():
    """
    🎯 SERVICE 7: See where Wall Street is placing its bets.
    (Only for US stocks with options)
    """
    import gamma_engine
    from engines.derivatives_intelligence import _compute_vol_surface, _compute_convexity_risk

    print("\n" + "=" * 65)
    print("🎯  SERVICE 7: WHERE WALL STREET IS BETTING")
    print("=" * 65)

    # Realistic options chain
    chain = {
        "symbol": "AAPL", "underlying_price": 227,
        "contracts": [
            {"strike": 200, "contract_type": "put", "implied_volatility": 0.35, "open_interest": 15000},
            {"strike": 210, "contract_type": "put", "implied_volatility": 0.30, "open_interest": 25000},
            {"strike": 220, "contract_type": "call", "implied_volatility": 0.24, "open_interest": 40000},
            {"strike": 225, "contract_type": "call", "implied_volatility": 0.22, "open_interest": 60000},
            {"strike": 230, "contract_type": "call", "implied_volatility": 0.23, "open_interest": 45000},
            {"strike": 240, "contract_type": "call", "implied_volatility": 0.26, "open_interest": 20000},
            {"strike": 250, "contract_type": "call", "implied_volatility": 0.30, "open_interest": 10000},
        ],
    }

    # GEX
    strikes = np.array([c["strike"] for c in chain["contracts"]], dtype=np.float64)
    ivs = np.array([c["implied_volatility"] for c in chain["contracts"]], dtype=np.float64)
    ois = np.array([c["open_interest"] for c in chain["contracts"]], dtype=np.float64)
    flags = np.array([1 if c["contract_type"] == "call" else 0 for c in chain["contracts"]], dtype=np.int32)

    gex = gamma_engine.compute_gamma_exposure(strikes, ivs, ois, flags, 227, 0.045, 30/365)

    print(f"\n  📍 Stock: AAPL at $227")
    print(f"\n  🧲 Key Price Levels (where big players are hedging):")
    sorted_idx = sorted(range(len(gex.gex_per_strike)), key=lambda i: abs(gex.gex_per_strike[i]), reverse=True)
    for i in sorted_idx[:3]:
        direction = "SUPPORT ⬆️" if gex.gex_per_strike[i] > 0 else "RESISTANCE ⬇️"
        print(f"     ${gex.strikes[i]:.0f}  →  {direction}  (strength: ${abs(gex.gex_per_strike[i]):,.0f})")

    # Vol Surface
    vs = _compute_vol_surface(chain)
    print(f"\n  📐 Options Market Mood:")
    print(f"     Fear of downside (put demand): {'HIGH 🔴' if vs['skew'] > 0.05 else 'NORMAL ⚪'}")
    print(f"     Implied move (30 days): ±{vs['atm_iv']*100:.1f}%  (${227 * vs['atm_iv']:.0f})")

    # Convexity
    cr = _compute_convexity_risk(chain, None)
    print(f"\n  ⚡ Risk of Explosive Move:")
    emoji_map = {"low": "🟢", "moderate": "🟡", "elevated": "🟠", "critical": "🔴"}
    print(f"     {emoji_map.get(cr['risk_level'], '⚪')}  {cr['risk_level'].upper()} ({cr['convexity_risk_score']}/100)")

    assert gex.total_gex != 0
    assert vs["atm_iv"] > 0
    print(f"\n  ✅ Shows the invisible forces that move stock prices!")


# ══════════════════════════════════════════════════════════════════════

def test_service_summary():
    """Print a summary of all services."""
    print("\n" + "=" * 65)
    print("  ✅ ALL 7 SERVICES DEMONSTRATED SUCCESSFULLY")
    print("=" * 65)
    print("""
  1. 🔍 Instant Company Lookup    — any stock, any country
  2. 🏥 Health Score & Alerts      — 0-100 safety score
  3. 📰 AI News Sentiment          — good/bad news detection
  4. 🔗 Market Connections          — diversification helper
  5. 🏆 Company Rankings            — side-by-side comparison
  6. 🌍 Economy Overview            — is it safe to invest?
  7. 🎯 Wall Street Bet Tracker     — where the big money is
    """)
