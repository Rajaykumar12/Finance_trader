"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from state.redis_client import redis_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Redis on startup, close on shutdown."""
    await redis_client.connect()
    yield
    await redis_client.close()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trading Intelligence Platform",
        description="Universal AI-Powered Prediction & Trading Intelligence API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router, prefix="/api/v1")
    return app


app = create_app()
