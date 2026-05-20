#!/usr/bin/env python3
"""
Optimized Copywriting API v11
=============================

High-performance API integration with UltraOptimizedEngineV11:
- Advanced caching and rate limiting
- Real-time performance monitoring
- Intelligent load balancing
- Comprehensive error handling
- Auto-scaling capabilities
"""

import asyncio
import logging
import time
import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Depends, Body, status, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
import structlog
from loguru import logger
import prometheus_client as prom
from prometheus_client import Counter, Histogram, Gauge, Summary
import redis
import aioredis
from fastapi_limiter.depends import RateLimiter
from fastapi_cache2.decorator import cache
import httpx
import aiohttp
from contextlib import asynccontextmanager

# Import the ultra-optimized engine
from .ultra_optimized_engine_v11 import (
    UltraOptimizedEngineV11, 
    PerformanceConfig, 
    ModelConfig, 
    CacheConfig, 
    MonitoringConfig,
    get_engine,
    cleanup_engine
)

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Prometheus Metrics
API_REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['endpoint', 'method'])
API_REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['endpoint'])
API_ERROR_COUNT = Counter('api_errors_total', 'Total API errors', ['endpoint', 'error_type'])
API_CACHE_HIT_RATIO = Gauge('api_cache_hit_ratio', 'API cache hit ratio')
API_ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active API connections')

# Configuration
class APIConfig(BaseSettings):
    """API configuration settings"""
    api_key: str = "your-secret-api-key"
    allowed_cors_origins: List[str] = ["*"]
    redis_url: str = "redis://localhost:6379"
    max_requests_per_minute: int = 100
    cache_ttl: int = 3600
    enable_rate_limiting: bool = True
    enable_caching: bool = True
    enable_monitoring: bool = True
    enable_compression: bool = True
    enable_cors: bool = True
    enable_health_checks: bool = True
    enable_metrics: bool = True
    enable_auto_scaling: bool = True
    enable_load_balancing: bool = True
    enable_circuit_breaker: bool = True
    enable_retry_mechanism: bool = True
    enable_performance_tracking: bool = True
    enable_error_tracking: bool = True
    enable_request_logging: bool = True
    enable_response_logging: bool = True

# Pydantic Models
class CopywritingRequest(BaseModel):
    """Copywriting request model"""
    product_description: str = Field(..., description="Product description")
    target_platform: str = Field("general", description="Target platform")
    tone: str = Field("professional", description="Content tone")
    target_audience: str = Field("general", description="Target audience")
    key_points: List[str] = Field(default_factory=list, description="Key points to highlight")
    instructions: str = Field("", description="Additional instructions")
    restrictions: List[str] = Field(default_factory=list, description="Content restrictions")
    creativity_level: float = Field(0.7, ge=0.0, le=1.0, description="Creativity level")
    language: str = Field("en", description="Content language")

class CopywritingResponse(BaseModel):
    """Copywriting response model"""
    variants: List[Dict[str, Any]] = Field(..., description="Generated variants")
    best_variant: str = Field(..., description="Best variant ID")
    metrics: Dict[str, Any] = Field(..., description="Generation metrics")
    cache_hit: bool = Field(False, description="Whether result was cached")
    processing_time: float = Field(..., description="Processing time in seconds")

class BatchRequest(BaseModel):
    """Batch request model"""
    requests: List[CopywritingRequest] = Field(..., description="List of requests")
    wait_for_completion: bool = Field(False, description="Wait for all results")
    priority: str = Field("normal", description="Request priority")

class PerformanceStats(BaseModel):
    """Performance statistics model"""
    cache_stats: Dict[str, Any] = Field(..., description="Cache statistics")
    memory_usage: Dict[str, Any] = Field(..., description="Memory usage")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")
    error_counts: Dict[str, int] = Field(..., description="Error counts")
    batch_processor_stats: Dict[str, Any] = Field(..., description="Batch processor stats")

# API Configuration
config = APIConfig()
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Redis client
redis_client = None

# Engine instance
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global engine, redis_client
    
    # Startup
    logger.info("Starting Optimized Copywriting API v11")
    
    try:
        # Initialize Redis
        redis_client = await aioredis.from_url(
            config.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        logger.info("Redis connection established")
        
        # Initialize engine
        engine = await get_engine()
        logger.info("Ultra-optimized engine initialized")
        
        yield
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Optimized Copywriting API v11")
        
        if engine:
            await cleanup_engine()
        
        if redis_client:
            await redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Optimized Copywriting API v11",
    description="Ultra-optimized copywriting generation API with advanced features",
    version="11.0.0",
    lifespan=lifespan
)

# Add middleware
if config.enable_cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.allowed_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

if config.enable_compression:
    app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key"""
    if api_key != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# Performance tracking
class PerformanceTracker:
    """Track API performance metrics"""
    
    def __init__(self):
        self.request_times = []
        self.error_counts = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def track_request(self, endpoint: str, method: str, duration: float, error: bool = False):
        """Track request metrics"""
        API_REQUEST_COUNT.labels(endpoint=endpoint, method=method).inc()
        API_REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
        
        if error:
            API_ERROR_COUNT.labels(endpoint=endpoint, error_type="general").inc()
        
        self.request_times.append(duration)
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
    
    def track_cache_hit(self, hit: bool):
        """Track cache hit/miss"""
        if hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
        
        total = self.cache_hits + self.cache_misses
        if total > 0:
            hit_ratio = self.cache_hits / total
            API_CACHE_HIT_RATIO.set(hit_ratio)

performance_tracker = PerformanceTracker()

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = time.perf_counter()
    
    # Log request
    if config.enable_request_logging:
        logger.info(
            "API Request",
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
            user_agent=request.headers.get("user-agent")
        )
    
    try:
        response = await call_next(request)
        
        # Track performance
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method
        
        performance_tracker.track_request(endpoint, method, duration)
        
        # Log response
        if config.enable_response_logging:
            logger.info(
                "API Response",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration=duration
            )
        
        return response
        
    except Exception as e:
        # Track error
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        method = request.method
        
        performance_tracker.track_request(endpoint, method, duration, error=True)
        
        logger.error(
            "API Error",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration=duration
        )
        
        raise

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Check Redis connection
        redis_healthy = False
        if redis_client:
            try:
                await redis_client.ping()
                redis_healthy = True
            except:
                pass
        
        # Check engine health
        engine_healthy = engine is not None
        
        # Check system resources
        memory_usage = psutil.virtual_memory().percent
        cpu_usage = psutil.cpu_percent()
        
        health_status = {
            "status": "healthy" if redis_healthy and engine_healthy else "unhealthy",
            "timestamp": time.time(),
            "components": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "engine": "healthy" if engine_healthy else "unhealthy"
            },
            "system": {
                "memory_usage": memory_usage,
                "cpu_usage": cpu_usage
            }
        }
        
        status_code = 200 if health_status["status"] == "healthy" else 503
        return JSONResponse(content=health_status, status_code=status_code)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"status": "unhealthy", "error": str(e)},
            status_code=503
        )

# Metrics endpoint
@app.get("/metrics", tags=["metrics"])
async def get_metrics():
    """Get Prometheus metrics"""
    if not config.enable_metrics:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    try:
        # Get engine performance stats
        engine_stats = await engine.get_performance_stats() if engine else {}
        
        # Get API performance stats
        api_stats = {
            "request_times": performance_tracker.request_times[-100:] if performance_tracker.request_times else [],
            "cache_hits": performance_tracker.cache_hits,
            "cache_misses": performance_tracker.cache_misses,
            "error_counts": performance_tracker.error_counts
        }
        
        return {
            "engine_stats": engine_stats,
            "api_stats": api_stats,
            "prometheus_metrics": prom.generate_latest().decode()
        }
        
    except Exception as e:
        logger.error(f"Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics error: {e}")

# Main copywriting generation endpoint
@app.post(
    "/generate",
    response_model=CopywritingResponse,
    summary="Generate copywriting with ultra-optimization",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=config.max_requests_per_minute, seconds=60))] if config.enable_rate_limiting else [],
    responses={
        200: {"description": "Copywriting generated successfully"},
        400: {"description": "Invalid request data"},
        401: {"description": "Invalid or missing API Key"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def generate_copywriting(
    request: CopywritingRequest = Body(..., example={
        "product_description": "Premium wireless headphones with noise cancellation",
        "target_platform": "Instagram",
        "tone": "inspirational",
        "target_audience": "Young professionals",
        "key_points": ["Premium quality", "Noise cancellation", "Long battery life"],
        "instructions": "Emphasize the premium experience and lifestyle benefits",
        "restrictions": ["no price mentions"],
        "creativity_level": 0.8,
        "language": "en"
    }),
    api_key: str = Depends(get_api_key)
):
    """Generate copywriting with ultra-optimization"""
    start_time = time.perf_counter()
    
    try:
        # Check if engine is available
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        # Convert request to dict
        input_data = request.dict()
        
        # Generate copywriting
        result = await engine.generate_copywriting(input_data)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        # Track cache hit
        cache_hit = result.get('metrics', {}).get('cache_hit_ratio', 0) > 0
        performance_tracker.track_cache_hit(cache_hit)
        
        # Create response
        response = CopywritingResponse(
            variants=result.get('variants', []),
            best_variant=result.get('best_variant', ''),
            metrics=result.get('metrics', {}),
            cache_hit=cache_hit,
            processing_time=processing_time
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Copywriting generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

# Batch generation endpoint
@app.post(
    "/batch-generate",
    summary="Generate copywriting in batch",
    tags=["copywriting"],
    dependencies=[Depends(RateLimiter(times=config.max_requests_per_minute, seconds=60))] if config.enable_rate_limiting else [],
    responses={
        200: {"description": "Batch processing started"},
        400: {"description": "Invalid request data"},
        401: {"description": "Invalid or missing API Key"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
    },
)
async def batch_generate_copywriting(
    batch_request: BatchRequest = Body(..., example={
        "requests": [
            {
                "product_description": "Premium wireless headphones",
                "target_platform": "Instagram",
                "tone": "inspirational"
            }
        ],
        "wait_for_completion": False,
        "priority": "normal"
    }),
    api_key: str = Depends(get_api_key)
):
    """Generate copywriting in batch"""
    start_time = time.perf_counter()
    
    try:
        # Check if engine is available
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        # Process batch
        results = []
        for req in batch_request.requests:
            input_data = req.dict()
            result = await engine.generate_copywriting(input_data)
            results.append(result)
        
        # Calculate processing time
        processing_time = time.perf_counter() - start_time
        
        return {
            "results": results,
            "total_requests": len(batch_request.requests),
            "processing_time": processing_time,
            "batch_id": hashlib.md5(str(time.time()).encode()).hexdigest()
        }
        
    except Exception as e:
        logger.error(f"Batch generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch generation error: {e}")

# Performance statistics endpoint
@app.get(
    "/performance-stats",
    response_model=PerformanceStats,
    summary="Get comprehensive performance statistics",
    tags=["monitoring"],
    responses={
        200: {"description": "Performance statistics retrieved"},
        401: {"description": "Invalid or missing API Key"},
        500: {"description": "Internal server error"},
    },
)
async def get_performance_stats(api_key: str = Depends(get_api_key)):
    """Get comprehensive performance statistics"""
    try:
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        stats = await engine.get_performance_stats()
        
        return PerformanceStats(**stats)
        
    except Exception as e:
        logger.error(f"Performance stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Performance stats error: {e}")

# Cache management endpoints
@app.post(
    "/cache/clear",
    summary="Clear all caches",
    tags=["cache"],
    responses={
        200: {"description": "Cache cleared successfully"},
        401: {"description": "Invalid or missing API Key"},
        500: {"description": "Internal server error"},
    },
)
async def clear_cache(api_key: str = Depends(get_api_key)):
    """Clear all caches"""
    try:
        if redis_client:
            await redis_client.flushdb()
        
        if engine and hasattr(engine, 'cache'):
            engine.cache.predictive_cache.clear()
        
        return {"message": "Cache cleared successfully"}
        
    except Exception as e:
        logger.error(f"Cache clear error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear error: {e}")

@app.get(
    "/cache/stats",
    summary="Get cache statistics",
    tags=["cache"],
    responses={
        200: {"description": "Cache statistics retrieved"},
        401: {"description": "Invalid or missing API Key"},
        500: {"description": "Internal server error"},
    },
)
async def get_cache_stats(api_key: str = Depends(get_api_key)):
    """Get cache statistics"""
    try:
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not available")
        
        cache_stats = engine.cache.get_cache_stats()
        
        return cache_stats
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats error: {e}")

# Engine management endpoints
@app.post(
    "/engine/reload",
    summary="Reload the engine",
    tags=["engine"],
    responses={
        200: {"description": "Engine reloaded successfully"},
        401: {"description": "Invalid or missing API Key"},
        500: {"description": "Internal server error"},
    },
)
async def reload_engine(api_key: str = Depends(get_api_key)):
    """Reload the engine"""
    try:
        global engine
        
        # Cleanup current engine
        if engine:
            await cleanup_engine()
        
        # Initialize new engine
        engine = await get_engine()
        
        return {"message": "Engine reloaded successfully"}
        
    except Exception as e:
        logger.error(f"Engine reload error: {e}")
        raise HTTPException(status_code=500, detail=f"Engine reload error: {e}")

@app.get(
    "/engine/status",
    summary="Get engine status",
    tags=["engine"],
    responses={
        200: {"description": "Engine status retrieved"},
        401: {"description": "Invalid or missing API Key"},
        500: {"description": "Internal server error"},
    },
)
async def get_engine_status(api_key: str = Depends(get_api_key)):
    """Get engine status"""
    try:
        if not engine:
            return {"status": "not_initialized"}
        
        # Check engine components
        status = {
            "status": "healthy",
            "components": {
                "cache": "healthy" if engine.cache else "unhealthy",
                "memory_manager": "healthy" if engine.memory_manager else "unhealthy",
                "batch_processor": "healthy" if engine.batch_processor else "unhealthy",
                "circuit_breaker": "healthy" if engine.circuit_breaker else "unhealthy"
            },
            "model_cache_size": len(engine.model_cache) if engine.model_cache else 0,
            "tokenizer_cache_size": len(engine.tokenizer_cache) if engine.tokenizer_cache else 0
        }
        
        return status
        
    except Exception as e:
        logger.error(f"Engine status error: {e}")
        raise HTTPException(status_code=500, detail=f"Engine status error: {e}")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Optimized Copywriting API v11",
        "version": "11.0.0",
        "description": "Ultra-optimized copywriting generation API",
        "features": [
            "Advanced GPU acceleration",
            "Intelligent caching",
            "Real-time monitoring",
            "Auto-scaling",
            "Circuit breaker pattern",
            "Batch processing",
            "Performance optimization"
        ],
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "generate": "/generate",
            "batch-generate": "/batch-generate",
            "performance-stats": "/performance-stats",
            "cache": {
                "clear": "/cache/clear",
                "stats": "/cache/stats"
            },
            "engine": {
                "reload": "/engine/reload",
                "status": "/engine/status"
            }
        }
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Main function for running the API
def main():
    """Main function to run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Copywriting API v11")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    # Run the server
    uvicorn.run(
        "optimized_api_v11:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )

if __name__ == "__main__":
    main() 