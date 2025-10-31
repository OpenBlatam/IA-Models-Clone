from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

from .models import (
    demonstrate_ultra_advanced_libraries,
    demonstrate_ultra_advanced_libraries_async,
    batch_demonstrate_libraries,
    generate_analytics_report,
    create_monitoring_dashboard,
    run_enterprise_demonstration,
)


pdf_variantes_bp = Blueprint("pdf_variantes", __name__, url_prefix="/pdf-variantes")


def _json_error(message: str, status: int = 400) -> Any:
    return jsonify({"error": message}), status


@pdf_variantes_bp.route("/demo", methods=["POST"])
def demo() -> Any:
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    verbose: bool = bool(body.get("verbose", False))
    top_n: int = int(body.get("top_n", 10))
    compatibility_pairs: Optional[List[List[str]]] = body.get("compatibility_pairs")
    enable_caching: bool = bool(body.get("enable_caching", True))
    timeout_seconds: int = int(body.get("timeout_seconds", 30))

    # Basic input validation
    if top_n < 1:
        return _json_error("top_n must be >= 1", 422)

    # Convert nested lists to tuples if provided
    pairs_typed: Optional[List[tuple]] = (
        [tuple(p) for p in compatibility_pairs] if compatibility_pairs else None
    )

    result = demonstrate_ultra_advanced_libraries(
        verbose=verbose,
        top_n=top_n,
        compatibility_pairs=pairs_typed,
        enable_caching=enable_caching,
        timeout_seconds=timeout_seconds,
    )
    return jsonify(result)


@pdf_variantes_bp.route("/demo-async", methods=["POST"])
async def demo_async() -> Any:
    body: Dict[str, Any] = await request.get_json(silent=True) or {}
    verbose: bool = bool(body.get("verbose", False))
    top_n: int = int(body.get("top_n", 10))
    compatibility_pairs: Optional[List[List[str]]] = body.get("compatibility_pairs")
    enable_caching: bool = bool(body.get("enable_caching", True))
    timeout_seconds: int = int(body.get("timeout_seconds", 30))

    if top_n < 1:
        return _json_error("top_n must be >= 1", 422)

    pairs_typed: Optional[List[tuple]] = (
        [tuple(p) for p in compatibility_pairs] if compatibility_pairs else None
    )

    result = await demonstrate_ultra_advanced_libraries_async(
        verbose=verbose,
        top_n=top_n,
        compatibility_pairs=pairs_typed,
        enable_caching=enable_caching,
        timeout_seconds=timeout_seconds,
    )
    return jsonify(result)


@pdf_variantes_bp.route("/batch", methods=["POST"])
def demo_batch() -> Any:
    body: Dict[str, Any] = request.get_json(silent=True) or {}
    configs: List[Dict[str, Any]] = body.get("configs", [])
    verbose: bool = bool(body.get("verbose", False))
    max_concurrent: int = int(body.get("max_concurrent", 3))

    if not isinstance(configs, list) or not configs:
        return _json_error("configs must be a non-empty list", 422)

    results = batch_demonstrate_libraries(
        configs=configs,
        verbose=verbose,
        max_concurrent=max_concurrent,
    )
    return jsonify({"results": results, "count": len(results)})


@pdf_variantes_bp.route("/analytics", methods=["GET"])
def analytics() -> Any:
    time_range: str = request.args.get("time_range", "24h")
    include_trends: bool = request.args.get("include_trends", "true").lower() == "true"
    include_predictions: bool = (
        request.args.get("include_predictions", "false").lower() == "true"
    )

    report = generate_analytics_report(
        time_range=time_range,
        include_trends=include_trends,
        include_predictions=include_predictions,
    )
    return jsonify(report)


@pdf_variantes_bp.route("/monitoring", methods=["GET"])
def monitoring() -> Any:
    dashboard = create_monitoring_dashboard()
    return jsonify(dashboard)


@pdf_variantes_bp.route("/enterprise-demo", methods=["POST"])
def enterprise_demo() -> Any:
    result = run_enterprise_demonstration()
    return jsonify(result)

"""
PDF Variantes API
================

Main FastAPI application for PDF variantes features.
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import Dict, Any
import logging
from contextlib import asynccontextmanager

from .config import ConfigManager, PDFVariantesConfig
from .middleware import setup_middleware
from .routers import pdf_router, analytics_router, collaboration_router
from .exceptions import PDFVariantesError, get_http_status_code
from .dependencies import get_config

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting PDF Variantes API")
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Store config in app state
    app.state.config = config
    
    logger.info(f"Configuration loaded for {config.environment.value} environment")
    
    yield
    
    # Shutdown
    logger.info("Shutting down PDF Variantes API")


def create_app() -> FastAPI:
    """Create FastAPI application with all configurations."""
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Create FastAPI app
    app = FastAPI(
        title="PDF Variantes API",
        description="Advanced PDF processing system with AI capabilities",
        version="2.0.0",
        docs_url="/docs" if config.debug else None,
        redoc_url="/redoc" if config.debug else None,
        openapi_url="/openapi.json" if config.debug else None,
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app, config)
    
    # Include routers
    app.include_router(pdf_router)
    app.include_router(analytics_router)
    app.include_router(collaboration_router)
    
    # Global exception handler
    @app.exception_handler(PDFVariantesError)
    async def pdf_variantes_exception_handler(request, exc: PDFVariantesError):
        """Handle PDF variantes specific exceptions."""
        status_code = get_http_status_code(exc.error_code)
        
        return JSONResponse(
            status_code=status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": getattr(request.state, "request_id", "")
            }
        )
    
    # Health check endpoint
    @app.get("/health", tags=["System"])
    async def health_check(config: PDFVariantesConfig = Depends(get_config)):
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "pdf-variantes",
            "version": "2.0.0",
            "environment": config.environment.value,
            "features_enabled": len([f for f in config.features if f.enabled]),
            "timestamp": "2024-01-01T00:00:00Z"  # TODO: Use actual timestamp
        }
    
    # Root endpoint
    @app.get("/", tags=["System"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": "PDF Variantes API",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


# Create the app instance
app = create_app()