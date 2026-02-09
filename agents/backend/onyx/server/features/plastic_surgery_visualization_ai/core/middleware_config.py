"""Middleware configuration."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.settings import settings
from middleware.rate_limit import RateLimitMiddleware
from middleware.security import SecurityHeadersMiddleware
from middleware.logging_middleware import LoggingMiddleware
from core.constants import DEFAULT_RATE_LIMIT_PER_MINUTE
from utils.logger import get_logger

logger = get_logger(__name__)


def setup_middleware(app: FastAPI) -> None:
    """Configure middleware for the application."""
    logger.info("Setting up middleware...")
    
    # Logging (first to capture all requests)
    app.add_middleware(LoggingMiddleware)
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)
    
    # Rate limiting
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=DEFAULT_RATE_LIMIT_PER_MINUTE
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    logger.info("Middleware setup complete")

