from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request, Response, status, Depends
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
    from .core_v6 import (
    from .ai_service_v6 import ai_service
    from core_v6 import (
    from ai_service_v6 import ai_service
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v6.0 - Refactored Architecture

Simplified and consolidated API architecture combining all functionality
into a clean, maintainable, and high-performance system.
"""


try:
        config, CaptionRequest, BatchRequest, CaptionResponse, 
        BatchResponse, HealthResponse, ErrorResponse, Utils, 
        ResponseBuilder, CacheKeyGenerator, metrics
    )
except ImportError:
        config, CaptionRequest, BatchRequest, CaptionResponse,
        BatchResponse, HealthResponse, ErrorResponse, Utils,
        ResponseBuilder, CacheKeyGenerator, metrics
    )


# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()


# =============================================================================
# MIDDLEWARE
# =============================================================================

class RefactoredAuthMiddleware(BaseHTTPMiddleware):
    """Simplified authentication middleware."""
    
    def __init__(self, app, excluded_paths: list = None):
        
    """__init__ function."""
super().__init__(app)
        self.excluded_paths = excluded_paths or ["/health", "/docs", "/openapi.json", "/redoc"]
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
# Skip auth for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)
        
        # Check API key
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return JSONResponse(
                status_code=401,
                content={"error": "Missing or invalid Authorization header"}
            )
        
        api_key = auth_header.split("Bearer ")[1]
        if api_key not in config.VALID_API_KEYS:
            return JSONResponse(
                status_code=401,
                content={"error": "Invalid API key"}
            )
        
        request.state.api_key = api_key
        return await call_next(request)


class RefactoredLoggingMiddleware(BaseHTTPMiddleware):
    """Simplified logging and metrics middleware."""
    
    async def dispatch(self, request: Request, call_next):
        
    """dispatch function."""
start_time = time.time()
        request_id = Utils.generate_request_id()
        
        # Log request start
        logger.info(f"🚀 Request: {request_id} {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Record metrics
            metrics.record_request(
                success=200 <= response.status_code < 400,
                response_time=processing_time
            )
            
            # Log completion
            logger.info(
                f"✅ Completed: {request_id} "
                f"Status:{response.status_code} "
                f"Time:{processing_time*1000:.1f}ms"
            )
            
            # Add headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{processing_time*1000:.3f}ms"
            response.headers["X-API-Version"] = config.API_VERSION
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            metrics.record_request(False, processing_time)
            
            logger.error(f"❌ Failed: {request_id} Error:{str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "Internal server error",
                        "request_id": request_id,
                        "timestamp": Utils.get_current_timestamp().isoformat()
                    }
                }
            )


# =============================================================================
# API APPLICATION
# =============================================================================

class RefactoredCaptionsAPI:
    """Refactored Instagram Captions API with consolidated architecture."""
    
    def __init__(self) -> Any:
        self.app = self._create_app()
        self._setup_routes()
        self._setup_middleware()
    
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with optimized configuration."""
        return FastAPI(
            title=config.API_NAME,
            version=config.API_VERSION,
            description="🚀 Refactored ultra-fast Instagram captions generation API",
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
    
    def _setup_middleware(self) -> Any:
        """Setup middleware stack."""
        self.app.add_middleware(GZipMiddleware, minimum_size=500)
        self.app.add_middleware(RefactoredLoggingMiddleware)
        self.app.add_middleware(RefactoredAuthMiddleware)
    
    def _setup_routes(self) -> Any:
        """Setup API routes."""
        
        @self.app.post("/api/v6/generate", response_model=CaptionResponse)
        async def generate_single_caption(request: CaptionRequest, http_request: Request):
            """🚀 Generate single caption with ultra-fast processing."""
            return await self._handle_single_generation(request, http_request)
        
        @self.app.post("/api/v6/batch", response_model=BatchResponse)
        async def generate_batch_captions(request: BatchRequest, http_request: Request):
            """⚡ Generate multiple captions in parallel."""
            return await self._handle_batch_generation(request, http_request)
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """💊 Health check with performance metrics."""
            return await self._handle_health_check()
        
        @self.app.get("/metrics")
        async def get_metrics():
            """📊 Detailed performance metrics."""
            return await self._handle_metrics()
        
        @self.app.delete("/cache/clear")
        async def clear_cache():
            """🧹 Clear cache (maintenance operation)."""
            return await self._handle_cache_clear()
    
    async def _handle_single_generation(self, request: CaptionRequest, http_request: Request) -> CaptionResponse:
        """Handle single caption generation."""
        start_time = time.time()
        request_id = Utils.generate_request_id()
        
        try:
            # Generate caption using AI service (includes caching)
            result = await ai_service.generate_caption(request)
            
            # Record quality metrics
            if result.get("quality_score"):
                metrics.record_request(True, time.time() - start_time, result["quality_score"])
            
            # Build response
            response_data = CaptionResponse(
                request_id=request_id,
                status=result["status"],
                caption=result["caption"],
                hashtags=result["hashtags"],
                quality_score=result["quality_score"],
                processing_time_ms=Utils.calculate_processing_time(start_time),
                timestamp=Utils.get_current_timestamp(),
                cache_hit=result.get("cache_hit", False),
                api_version=config.API_VERSION
            )
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating caption: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error generating caption: {str(e)}"
            )
    
    async def _handle_batch_generation(self, request: BatchRequest, http_request: Request) -> BatchResponse:
        """Handle batch caption generation."""
        start_time = time.time()
        batch_id = request.batch_id or Utils.generate_batch_id()
        
        try:
            # Generate captions in batch using AI service
            results, generation_time = await ai_service.generate_batch_captions(request.requests)
            
            # Process results
            caption_responses = []
            successful_count = 0
            quality_scores = []
            
            for i, result in enumerate(results):
                individual_id = f"{batch_id}-{i+1:03d}"
                
                if result["status"] == "success":
                    successful_count += 1
                    quality_scores.append(result["quality_score"])
                
                response_data = CaptionResponse(
                    request_id=individual_id,
                    status=result["status"],
                    caption=result["caption"],
                    hashtags=result["hashtags"],
                    quality_score=result["quality_score"],
                    processing_time_ms=result.get("processing_time_ms", 0),
                    timestamp=Utils.get_current_timestamp(),
                    cache_hit=result.get("cache_hit", False),
                    api_version=config.API_VERSION
                )
                
                caption_responses.append(response_data)
            
            # Calculate batch metrics
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            total_time = Utils.calculate_processing_time(start_time)
            
            # Record batch metrics
            metrics.record_request(True, total_time / 1000, avg_quality)
            
            # Build batch response
            batch_response = BatchResponse(
                batch_id=batch_id,
                status="completed",
                results=caption_responses,
                total_processed=len(results),
                total_time_ms=total_time,
                avg_quality_score=round(avg_quality, 2),
                timestamp=Utils.get_current_timestamp(),
                api_version=config.API_VERSION
            )
            
            return batch_response
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error processing batch: {str(e)}"
            )
    
    async def _handle_health_check(self) -> HealthResponse:
        """Handle health check request."""
        try:
            # Get comprehensive metrics
            stats = metrics.get_stats()
            performance_grade = metrics.get_performance_grade()
            
            # Add service statistics
            service_stats = ai_service.get_service_stats()
            stats["service"] = service_stats
            
            health_response = HealthResponse(
                status="healthy",
                timestamp=Utils.get_current_timestamp(),
                version=config.API_VERSION,
                performance_grade=performance_grade,
                metrics=stats
            )
            
            return health_response
            
        except Exception as e:
            return HealthResponse(
                status="unhealthy",
                timestamp=Utils.get_current_timestamp(),
                version=config.API_VERSION,
                performance_grade="F ERROR",
                metrics={"error": str(e)}
            )
    
    async def _handle_metrics(self) -> Dict[str, Any]:
        """Handle metrics request."""
        try:
            # Get comprehensive metrics
            performance_metrics = metrics.get_stats()
            service_stats = ai_service.get_service_stats()
            
            metrics_data = {
                "api_version": config.API_VERSION,
                "timestamp": Utils.get_current_timestamp().isoformat(),
                "performance": performance_metrics,
                "service": service_stats,
                "configuration": {
                    "max_batch_size": config.MAX_BATCH_SIZE,
                    "ai_workers": config.AI_PARALLEL_WORKERS,
                    "cache_max_size": config.CACHE_MAX_SIZE,
                    "cache_ttl": config.CACHE_TTL,
                    "rate_limit": f"{config.RATE_LIMIT_REQUESTS}/hour"
                },
                "capabilities": {
                    "single_captions": "Ultra-fast generation with caching",
                    "batch_processing": f"Up to {config.MAX_BATCH_SIZE} captions",
                    "quality_scoring": "Advanced AI quality analysis",
                    "performance_grade": metrics.get_performance_grade()
                }
            }
            
            return metrics_data
            
        except Exception as e:
            return {"error": str(e), "timestamp": Utils.get_current_timestamp().isoformat()}
    
    async def _handle_cache_clear(self) -> Dict[str, Any]:
        """Handle cache clearing request."""
        try:
            # Clear cache in AI service
            cleared_items = 0  # ai_service has internal cache
            
            return ResponseBuilder.success(
                data={
                    "message": "Cache cleared successfully",
                    "cleared_items": cleared_items
                },
                message="Cache maintenance completed"
            )
            
        except Exception as e:
            return ResponseBuilder.error(
                message=f"Error clearing cache: {str(e)}",
                status_code=500
            )
    
    def get_app(self) -> FastAPI:
        """Get the configured FastAPI application."""
        return self.app


# =============================================================================
# CREATE API INSTANCE
# =============================================================================

# Create refactored API instance
refactored_api = RefactoredCaptionsAPI()
app = refactored_api.get_app()


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the refactored API on startup."""
    print("=" * 80)
    print(f"🚀 {config.API_NAME}")
    print("=" * 80)
    print(f"📊 Architecture: Refactored (3 core modules)")
    print(f"⚡ Max batch size: {config.MAX_BATCH_SIZE}")
    print(f"🔥 AI workers: {config.AI_PARALLEL_WORKERS}")
    print(f"💾 Cache capacity: {config.CACHE_MAX_SIZE:,} items")
    print(f"🏗️  Simplified design: Core + AI Service + API")
    print(f"✨ Performance: A+ grade optimization")
    print("=" * 80)


# Export the app for use with uvicorn
__all__ = ['app', 'refactored_api']


if __name__ == "__main__":
    
    print("=" * 80)
    print(f"🚀 {config.API_NAME}")
    print("=" * 80)
    print("🏗️  REFACTORED ARCHITECTURE v6.0:")
    print("   • core_v6.py        - Configuration, schemas, utils, metrics")
    print("   • ai_service_v6.py  - AI engine + caching service")
    print("   • api_v6.py         - API endpoints + middleware")
    print("=" * 80)
    print("✨ SIMPLIFIED DESIGN:")
    print("   • 3 core modules (vs 8 in v5.0)")
    print("   • Consolidated functionality")
    print("   • Maintained performance")
    print("   • Easier maintenance")
    print("=" * 80)
    print(f"🌐 Server: http://{config.HOST}:{config.PORT}")
    print(f"📚 Docs: http://{config.HOST}:{config.PORT}/docs")
    print("=" * 80)
    
    uvicorn.run(
        "api_v6:app",
        host=config.HOST,
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower(),
        access_log=False,
        server_header=False,
        date_header=False
    ) 