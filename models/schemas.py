"""Pydantic models shared across the entire platform."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# ────────────────────────────────────────────────────────────────────
# Ingestion Schemas
# ────────────────────────────────────────────────────────────────────

class NormalizedTick(BaseModel):
    """A single normalised market data tick."""
    symbol: str
    timestamp: datetime
    best_bid: float
    best_ask: float
    bid_qty: float
    ask_qty: float
    spread: float = Field(description="ask - bid")
    source: str = Field(description="binance | alpaca")


class NewsHeadline(BaseModel):
    """A single financial news headline from Finnhub."""
    headline: str
    source: str
    url: str
    timestamp: datetime
    symbol: Optional[str] = None
    sentiment: Optional[float] = Field(
        default=None, description="FinBERT sentiment score (-1 to +1)"
    )


class OptionsContract(BaseModel):
    """A single options contract from an options chain."""
    strike: float
    expiry: str
    contract_type: str  # "call" | "put"
    last_price: float
    bid: float
    ask: float
    volume: int
    open_interest: int
    implied_volatility: float


class OptionsChainData(BaseModel):
    """Full options chain for a symbol."""
    symbol: str
    timestamp: datetime
    contracts: list[OptionsContract]
    underlying_price: float


class FundamentalsData(BaseModel):
    """Company fundamentals snapshot (from yfinance)."""
    symbol: str
    timestamp: datetime
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    total_debt: Optional[float] = None
    total_cash: Optional[float] = None
    free_cash_flow: Optional[float] = None
    revenue: Optional[float] = None
    profit_margin: Optional[float] = None


class MacroIndicators(BaseModel):
    """Macroeconomic indicators snapshot."""
    timestamp: datetime
    vix: Optional[float] = None
    treasury_10y: Optional[float] = None
    dollar_index: Optional[float] = None
    sp500: Optional[float] = None
    gold: Optional[float] = None
    fed_funds_rate: Optional[float] = None


# ────────────────────────────────────────────────────────────────────
# Intelligence Engine Schemas
# ────────────────────────────────────────────────────────────────────

class MarketStateVector(BaseModel):
    """Output of Module A — real-time market snapshot."""
    symbol: str
    timestamp: datetime
    rolling_volatility: float
    bid_ask_spread_mean: float
    bid_ask_spread_var: float
    order_book_imbalance: float
    tick_count: int
    # New fields
    volatility_regime: str = Field(description="low | normal | high | crisis")
    liquidity_shift: bool = False
    shift_magnitude: float = 1.0
    liquidity_status: str = Field(default="stable", description="stable | thinning | vanishing")


class CrossCorrelations(BaseModel):
    """Cross-market pairwise correlation matrix."""
    timestamp: datetime
    pairs: dict[str, float]
    symbols: list[str]
    window_size: int


class AssetHealthScore(BaseModel):
    """Output of Module B — unified biological health index."""
    symbol: str
    timestamp: datetime
    health_index: float = Field(ge=0, le=100)
    sentiment_score: float = Field(ge=-1, le=1)
    fundamental_score: float = Field(ge=0, le=100)
    market_score: float = Field(ge=0, le=100)
    macro_score: float = Field(default=50.0, ge=0, le=100)
    metabolic_stress_alert: bool = False


class EcosystemEntry(BaseModel):
    """A single asset in the ecosystem competitive map."""
    symbol: str
    health_index: float
    market_cap: float
    volatility_regime: str
    liquidity_status: str
    strengths: list[str]
    weaknesses: list[str]
    rank: int


class EcosystemMap(BaseModel):
    """Sector-level competitive positioning map."""
    timestamp: datetime
    sector_average_health: float
    total_assets: int
    rankings: list[EcosystemEntry]


class GammaExposureResult(BaseModel):
    """Output of Module C — Gamma Exposure per strike."""
    symbol: str
    timestamp: datetime
    total_gex: float
    hedging_pressure_zones: list[dict]
    flip_point: Optional[float] = Field(
        default=None, description="Strike where dealer gamma flips sign"
    )


class VolSurface(BaseModel):
    """Implied volatility surface for a symbol."""
    symbol: str
    timestamp: datetime
    atm_iv: float
    otm_put_iv: float
    otm_call_iv: float
    skew: float = Field(description="OTM Put IV - OTM Call IV")
    smile: float = Field(description="avg OTM IV - ATM IV")
    skew_status: str = Field(description="normal | moderate_skew | severe_skew")
    smile_status: str = Field(description="flat | mild_smile | pronounced_smile")
    total_contracts: int


class ConvexityRisk(BaseModel):
    """Convexity risk assessment for a symbol."""
    symbol: str
    timestamp: datetime
    convexity_risk_score: float = Field(ge=0, le=100)
    risk_level: str = Field(description="low | moderate | elevated | critical")
    total_dollar_gamma: float
    negative_gamma_exposure: float
    total_dollar_vega: float
    gamma_concentration: float


class VolRegimeIndicator(BaseModel):
    """Overall derivatives-implied volatility regime."""
    symbol: str
    timestamp: datetime
    iv_regime: str = Field(description="compressed | normal | elevated | extreme")
    overall_regime: str = Field(description="calm | normal | stressed | crisis")
    regime_score: float = Field(ge=0, le=100)
    atm_iv: float
    skew_magnitude: float
    convexity_risk: float
