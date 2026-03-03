"""Platform configuration — reads from .env via Pydantic Settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Trading Intelligence Platform."""

    # ── Redis ──────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ── Alpaca (Paper Trading) ─────────────────────────────────────────
    alpaca_api_key: str = Field(default="")
    alpaca_api_secret: str = Field(default="")
    alpaca_base_url: str = Field(default="https://paper-api.alpaca.markets")

    # ── Finnhub ────────────────────────────────────────────────────────
    finnhub_api_key: str = Field(default="")

    # ── Tracked symbols ────────────────────────────────────────────────
    equity_symbols: str = Field(default="AAPL,NVDA")
    crypto_symbols: str = Field(default="BTC")

    # ── Polling intervals (seconds) ────────────────────────────────────
    finnhub_poll_interval: int = Field(default=15)
    yahoo_poll_interval: int = Field(default=600)

    # ── Server ─────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ── Helpers ────────────────────────────────────────────────────────
    @property
    def equity_symbol_list(self) -> list[str]:
        return [s.strip() for s in self.equity_symbols.split(",") if s.strip()]

    @property
    def crypto_symbol_list(self) -> list[str]:
        return [s.strip() for s in self.crypto_symbols.split(",") if s.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import this everywhere
settings = Settings()
