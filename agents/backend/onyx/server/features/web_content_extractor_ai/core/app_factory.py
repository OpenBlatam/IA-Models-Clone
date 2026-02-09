"""
Factory para crear la aplicación FastAPI
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from infrastructure.middleware.rate_limiter import RateLimitMiddleware, RateLimiter
from infrastructure.utils.logger_config import setup_logging
from config import settings


def create_app() -> FastAPI:
    """Crear y configurar la aplicación FastAPI"""
    
    setup_logging(level=settings.log_level)
    
    app = FastAPI(
        title="Web Content Extractor AI",
        description="Extrae información completa de páginas web usando OpenRouter",
        version="1.0.0"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting
    rate_limiter = RateLimiter(
        max_requests=settings.rate_limit_max_requests,
        window_seconds=settings.rate_limit_window_seconds
    )
    app.add_middleware(RateLimitMiddleware, rate_limiter=rate_limiter)
    
    return app








