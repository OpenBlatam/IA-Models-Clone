"""
Main FastAPI Application
========================

Main application setup and configuration.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .config import settings
from .routers import (
    health_router, models_router, batch_router, utils_router,
    advanced_router, version_router, analysis_router, search_router
)
from .exceptions import TruthGPTAPIException
from .middleware import LoggingMiddleware, SecurityHeadersMiddleware, RateLimitMiddleware

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Starting TruthGPT API Server...")
    logger.info(f"Version: {settings.app_version}")
    yield
    logger.info("Shutting down TruthGPT API Server...")


app = FastAPI(
    title=settings.app_name,
    description=settings.app_description,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "models",
            "description": "Model management operations. Create, compile, train, evaluate, and manage neural network models."
        },
        {
            "name": "batch",
            "description": "Batch operations for managing multiple models at once. Delete, predict, and get statistics."
        },
        {
            "name": "health",
            "description": "Health check and monitoring endpoints. Get system status, metrics, and cache statistics."
        },
        {
            "name": "utils",
            "description": "Utility endpoints. List available layers, optimizers, losses, and validate configurations."
        },
        {
            "name": "advanced",
            "description": "Advanced operations. Model comparison, history, export, and detailed statistics."
        },
        {
            "name": "version",
            "description": "API versioning and information endpoints."
        },
        {
            "name": "analysis",
            "description": "Data analysis and validation endpoints. Analyze, validate, normalize, and split data."
        },
        {
            "name": "search",
            "description": "Advanced search endpoints. Search models by various criteria and layer types."
        }
    ]
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


@app.exception_handler(TruthGPTAPIException)
async def truthgpt_exception_handler(request: Request, exc: TruthGPTAPIException):
    """Handle custom TruthGPT API exceptions."""
    logger.warning(f"TruthGPT API Exception: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.__class__.__name__,
            "detail": exc.detail,
            "path": request.url.path
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "path": request.url.path
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "detail": "An internal server error occurred",
            "path": request.url.path
        }
    )


app.include_router(health_router)
app.include_router(models_router)
app.include_router(batch_router)
app.include_router(utils_router)
app.include_router(advanced_router)
app.include_router(version_router)
app.include_router(analysis_router)
app.include_router(search_router)

