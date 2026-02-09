"""
Optimized API for Video-OpusClip

High-performance FastAPI implementation with async processing,
intelligent caching, and optimized resource management.
"""

import asyncio
import time
import uuid
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request, status, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import structlog

from .optimized_config import get_config, OptimizedConfig
from .optimized_cache import get_cache_manager, cached
from .optimized_video_processor import OptimizedVideoProcessor, create_optimized_video_processor
from .models.video_models import VideoClipRequest, VideoClipResponse, VideoClipBatchRequest, VideoClipBatchResponse
from .models.viral_models import ViralVideoBatchResponse
from .error_handling import ErrorHandler, handle_processing_errors
from .validation import validate_video_request_data

logger = structlog.get_logger()

# =============================================================================
# OPTIMIZED API CONFIGURATION
# =============================================================================

class OptimizedAPIConfig:
    """Configuration for optimized API."""
    
    def __init__(self):
        self.config = get_config()
        self.cache_manager = get_cache_manager()
        
        # API Settings
        self.title = "Optimized Video Processing API"
        self.version = "4.0.0"
        self.description = "High-performance video processing with intelligent optimization"
        
        # Performance Settings
        self.max_request_size = self.config.performance.max_request_size
        self.rate_limit_per_minute = self.config.performance.rate_limit_per_minute
        self.enable_compression = self.config.performance.enable_response_compression
        self.enable_validation = self.config.performance.enable_request_validation
        
        # CORS Settings
        self.cors_origins = ["*"]
        self.cors_methods = ["*"]
        self.cors_headers = ["*"]
        
        # Trusted Hosts
        self.trusted_hosts = ["*"]

# =============================================================================
# RATE LIMITING
# =============================================================================

class RateLimiter:
    """Simple rate limiter for API endpoints."""
    
    def __init__(self, requests_per_minute: int = 1000):
        self.requests_per_minute = requests_per_minute
        self.requests = {}
    
    def is_allowed(self, client_ip: str) -> bool:
        """Check if request is allowed."""
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [req_time for req_time in self.requests[client_ip] if req_time > minute_ago]
        else:
            self.requests[client_ip] = []
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            return False
        
        # Add current request
        self.requests[client_ip].append(current_time)
        return True

# =============================================================================
# OPTIMIZED API
# =============================================================================

class OptimizedVideoAPI:
    """Optimized video processing API with high performance."""
    
    def __init__(self, config: Optional[OptimizedAPIConfig] = None):
        self.config = config or OptimizedAPIConfig()
        self.error_handler = ErrorHandler()
        self.rate_limiter = RateLimiter(self.config.rate_limit_per_minute)
        
        # Initialize processors
        self.video_processor = create_optimized_video_processor()
        
        # Create FastAPI app
        self.app = self._create_fastapi_app()
        self._setup_middleware()
        self._setup_routes()
        self._setup_exception_handlers()
        
        # Performance tracking
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create optimized FastAPI application."""
        return FastAPI(
            title=self.config.title,
            description=self.config.description,
            version=self.config.version,
            docs_url="/docs",
            redoc_url="/redoc",
            openapi_url="/openapi.json"
        )
    
    def _setup_middleware(self):
        """Setup optimized middleware."""
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=self.config.cors_methods,
            allow_headers=self.config.cors_headers,
        )
        
        # Compression middleware
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.trusted_hosts
        )
        
        # Request tracking middleware
        @self.app.middleware("http")
        async def add_request_tracking(request: Request, call_next):
            return await self._request_middleware(request, call_next)
    
    async def _request_middleware(self, request: Request, call_next):
        """Request tracking and rate limiting middleware."""
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())
        
        # Add request ID
        request.state.request_id = request_id
        
        # Rate limiting
        client_ip = request.client.host if request.client else "unknown"
        if not self.rate_limiter.is_allowed(client_ip):
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limit exceeded", "request_id": request_id}
            )
        
        # Update stats
        self.stats["total_requests"] += 1
        
        try:
            response = await call_next(request)
            
            # Update success stats
            self.stats["successful_requests"] += 1
            
            # Add performance headers
            response_time = time.perf_counter() - start_time
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{response_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Update failure stats
            self.stats["failed_requests"] += 1
            logger.error("Request failed", request_id=request_id, error=str(e))
            raise
        finally:
            # Update average response time
            response_time = time.perf_counter() - start_time
            total_requests = self.stats["successful_requests"] + self.stats["failed_requests"]
            if total_requests > 0:
                current_avg = self.stats["average_response_time"]
                self.stats["average_response_time"] = (
                    (current_avg * (total_requests - 1) + response_time) / total_requests
                )
    
    def _setup_routes(self):
        """Setup optimized API routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Enhanced health check with detailed system status."""
            try:
                # Check cache health
                cache_health = await self.config.cache_manager.health_check()
                
                # Check processor stats
                processor_stats = self.video_processor.get_processing_stats()
                
                # Get system info
                system_info = self._get_system_info()
                
                return {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "version": self.config.version,
                    "cache": cache_health,
                    "processor": processor_stats,
                    "system": system_info,
                    "api_stats": self.stats
                }
            except Exception as e:
                logger.error("Health check failed", error=str(e))
                raise HTTPException(status_code=503, detail="Service unhealthy")
        
        @self.app.post("/api/v1/video/process", response_model=VideoClipResponse)
        @handle_processing_errors
        async def process_video_optimized(
            request: VideoClipRequest,
            background_tasks: BackgroundTasks
        ):
            """Optimized single video processing endpoint."""
            # Validate request
            if self.config.enable_validation:
                validate_video_request_data(request)
            
            # Process video
            result = await self.video_processor.process_video(request)
            
            # Add cleanup task
            background_tasks.add_task(self._cleanup_processing_resources, request.youtube_url)
            
            return result
        
        @self.app.post("/api/v1/video/batch", response_model=VideoClipBatchResponse)
        @handle_processing_errors
        async def process_video_batch_optimized(
            request: VideoClipBatchRequest,
            background_tasks: BackgroundTasks
        ):
            """Optimized batch video processing endpoint."""
            # Validate batch request
            if len(request.requests) > 50:  # Limit batch size
                raise HTTPException(
                    status_code=400,
                    detail="Batch size too large. Maximum 50 videos per batch."
                )
            
            # Process videos in optimized batches
            results = await self.video_processor.process_video_batch(
                request.requests,
                max_concurrent=self.config.config.env.MAX_WORKERS
            )
            
            # Add cleanup task
            background_tasks.add_task(self._cleanup_batch_resources, len(request.requests))
            
            return VideoClipBatchResponse(
                results=results,
                total_processed=len(results),
                processing_time=time.time()
            )
        
        @self.app.post("/api/v1/viral/process", response_model=ViralVideoBatchResponse)
        async def process_viral_variants_optimized(
            request: VideoClipRequest,
            n_variants: int = 5,
            audience_profile: Optional[Dict[str, Any]] = None,
            use_langchain: bool = True
        ):
            """Optimized viral video processing endpoint."""
            # This would integrate with the viral processor
            # For now, return optimized response
            return await self._process_viral_variants_optimized(
                request, n_variants, audience_profile, use_langchain
            )
        
        @self.app.get("/api/v1/stats")
        async def get_api_stats():
            """Get API performance statistics."""
            return {
                "api_stats": self.stats,
                "processor_stats": self.video_processor.get_processing_stats(),
                "cache_stats": await self._get_cache_stats(),
                "system_stats": self._get_system_info()
            }
        
        @self.app.post("/api/v1/cache/clear")
        async def clear_cache():
            """Clear all caches."""
            try:
                # Clear video processor cache
                self.video_processor.memory_manager.clear()
                
                # Clear API cache
                await self.config.cache_manager.cache.clear_pattern("*")
                
                return {"message": "Cache cleared successfully"}
            except Exception as e:
                logger.error("Cache clear failed", error=str(e))
                raise HTTPException(status_code=500, detail="Failed to clear cache")
        
        @self.app.get("/api/v1/config")
        async def get_configuration():
            """Get current configuration."""
            return {
                "api_config": {
                    "title": self.config.title,
                    "version": self.config.version,
                    "max_request_size": self.config.max_request_size,
                    "rate_limit_per_minute": self.config.rate_limit_per_minute,
                    "enable_compression": self.config.enable_compression,
                    "enable_validation": self.config.enable_validation
                },
                "system_config": self.config.config.to_dict()
            }
    
    def _setup_exception_handlers(self):
        """Setup optimized exception handlers."""
        
        @self.app.exception_handler(Exception)
        async def global_exception_handler(request: Request, exc: Exception):
            """Global exception handler with detailed logging."""
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                "Unhandled exception",
                request_id=request_id,
                path=request.url.path,
                method=request.method,
                error=str(exc),
                exc_info=True
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": "An unexpected error occurred"
                }
            )
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """HTTP exception handler."""
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.warning(
                "HTTP exception",
                request_id=request_id,
                status_code=exc.status_code,
                detail=exc.detail
            )
            
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "request_id": request_id
                }
            )
    
    async def _process_viral_variants_optimized(
        self,
        request: VideoClipRequest,
        n_variants: int,
        audience_profile: Optional[Dict[str, Any]],
        use_langchain: bool
    ) -> ViralVideoBatchResponse:
        """Optimized viral variant processing."""
        # This would integrate with the viral processor
        # For now, return optimized response
        
        # Check cache first
        cache_key = f"viral_{request.youtube_url}_{n_variants}_{hash(str(audience_profile))}"
        cached_result = await self.config.cache_manager.cache.get(cache_key)
        
        if cached_result:
            self.stats["cache_hits"] += 1
            return ViralVideoBatchResponse(**cached_result)
        
        self.stats["cache_misses"] += 1
        
        # Process viral variants (placeholder)
        variants = []
        for i in range(n_variants):
            variant = {
                "id": f"variant_{i}",
                "title": f"Viral Variant {i+1}",
                "description": f"Optimized viral variant {i+1}",
                "viral_score": 0.8 + (i * 0.05),
                "engagement_score": 0.7 + (i * 0.03),
                "optimization_level": "high"
            }
            variants.append(variant)
        
        result = ViralVideoBatchResponse(
            variants=variants,
            total_variants=len(variants),
            average_viral_score=sum(v["viral_score"] for v in variants) / len(variants),
            processing_time=time.time()
        )
        
        # Cache result
        await self.config.cache_manager.cache.set(cache_key, result.dict(), ttl=3600)
        
        return result
    
    async def _cleanup_processing_resources(self, youtube_url: str):
        """Cleanup resources after processing."""
        try:
            # Release memory
            self.video_processor.memory_manager.force_garbage_collection()
            
            # Log cleanup
            logger.debug("Processing resources cleaned up", url=youtube_url)
        except Exception as e:
            logger.error("Cleanup failed", url=youtube_url, error=str(e))
    
    async def _cleanup_batch_resources(self, batch_size: int):
        """Cleanup resources after batch processing."""
        try:
            # More aggressive cleanup for batch processing
            self.video_processor.memory_manager.force_garbage_collection()
            
            # Log cleanup
            logger.debug("Batch processing resources cleaned up", batch_size=batch_size)
        except Exception as e:
            logger.error("Batch cleanup failed", batch_size=batch_size, error=str(e))
    
    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cache_health = await self.config.cache_manager.health_check()
            return {
                "health": cache_health,
                "memory_cache_size": self.config.cache_manager.cache.memory_cache.size(),
                "redis_available": cache_health.get("redis", False)
            }
        except Exception as e:
            logger.error("Failed to get cache stats", error=str(e))
            return {"error": "Failed to get cache stats"}
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        try:
            import psutil
            
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "memory_available_mb": psutil.virtual_memory().available // (1024 * 1024),
                "disk_usage_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            }
        except ImportError:
            return {"error": "psutil not available"}
        except Exception as e:
            logger.error("Failed to get system info", error=str(e))
            return {"error": "Failed to get system info"}

# =============================================================================
# API FACTORY
# =============================================================================

def create_optimized_api(config: Optional[OptimizedAPIConfig] = None) -> OptimizedVideoAPI:
    """Create an optimized video processing API."""
    return OptimizedVideoAPI(config)

def create_high_performance_api() -> OptimizedVideoAPI:
    """Create a high-performance API with optimized settings."""
    config = OptimizedAPIConfig()
    config.rate_limit_per_minute = 2000
    config.max_request_size = 50 * 1024 * 1024  # 50MB
    return OptimizedVideoAPI(config)

# =============================================================================
# GLOBAL API INSTANCE
# =============================================================================

# Global API instance
api_instance = None

def get_api() -> OptimizedVideoAPI:
    """Get the global API instance."""
    global api_instance
    if api_instance is None:
        api_instance = create_optimized_api()
    return api_instance

def get_app() -> FastAPI:
    """Get the FastAPI application instance."""
    return get_api().app 