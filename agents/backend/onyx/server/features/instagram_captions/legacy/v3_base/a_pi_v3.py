from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, List, Optional
from functools import wraps, lru_cache
from datetime import datetime
import logging
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from schemas import (
from dependencies import get_captions_engine, get_gmt_system, get_redis_client
from core import InstagramCaptionsEngine
from gmt_system import SimplifiedGMTSystem
from typing import Any, List, Dict, Optional
"""
Instagram Captions API v3.0 - Refactored & Optimized

Clean, simple, and ultra-fast architecture:
- Single API with smart optimizations
- Consolidated caching strategy  
- Simplified dependency management
- Clean error handling
- Performance monitoring
"""



    CaptionGenerationRequest,
    CaptionGenerationResponse,
    QualityAnalysisRequest,
    QualityMetricsResponse,
    CaptionOptimizationRequest,
    CaptionOptimizationResponse,
    BatchOptimizationRequest,
    HealthCheckResponse
)

logger = logging.getLogger(__name__)

# Simple in-memory cache for ultra-fast responses
_cache: Dict[str, Any] = {}
_cache_times: Dict[str, float] = {}
_metrics = {"requests": 0, "cache_hits": 0, "avg_time": 0.0}

# Initialize router
router = APIRouter(
    prefix="/api/v3/instagram-captions",
    tags=["Instagram Captions v3 - Refactored"],
    responses={
        200: {"description": "Success"},
        400: {"description": "Bad Request"}, 
        500: {"description": "Internal Server Error"}
    }
)


def smart_cache(ttl: int = 300):
    """Smart caching decorator with automatic cleanup."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Generate simple cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # Check cache
            current_time = time.time()
            if (cache_key in _cache and 
                cache_key in _cache_times and
                current_time - _cache_times[cache_key] < ttl):
                
                _metrics["cache_hits"] += 1
                return _cache[cache_key]
            
            # Execute function
            start_time = time.perf_counter()
            result = await func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            # Update cache and metrics
            _cache[cache_key] = result
            _cache_times[cache_key] = current_time
            _metrics["requests"] += 1
            _metrics["avg_time"] = (
                (_metrics["avg_time"] * (_metrics["requests"] - 1) + execution_time) /
                _metrics["requests"]
            )
            
            # Simple cleanup - keep only 50 most recent items
            if len(_cache) > 50:
                oldest_key = min(_cache_times.keys(), key=_cache_times.get)
                _cache.pop(oldest_key, None)
                _cache_times.pop(oldest_key, None)
            
            return result
        return wrapper
    return decorator


def handle_errors(func) -> Any:
    """Simple error handling decorator."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
    return wrapper


@router.get("/")
@smart_cache(ttl=3600)
async async def get_api_info() -> Dict[str, Any]:
    """Get API information."""
    return {
        "name": "Instagram Captions API v3.0",
        "version": "3.0.0",
        "description": "Refactored & optimized Instagram caption generation",
        "features": [
            "Smart caching with automatic cleanup",
            "Simplified architecture",
            "Ultra-fast responses",
            "Clean error handling",
            "Performance monitoring"
        ],
        "performance": {
            "cache_enabled": True,
            "parallel_processing": True,
            "streaming_responses": True
        }
    }


@router.post("/generate", response_model=CaptionGenerationResponse)
@handle_errors
@smart_cache(ttl=1800)
async def generate_caption(
    request: CaptionGenerationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    gmt_system: SimplifiedGMTSystem = Depends(get_gmt_system)
) -> CaptionGenerationResponse:
    """Generate optimized Instagram captions."""
    
    if not request.content_description.strip():
        raise ValueError("Content description cannot be empty")
    
    start_time = time.perf_counter()
    
    # Generate captions efficiently
    captions_result = await engine.generate_captions_async(
        content_description=request.content_description,
        style=request.style,
        audience=request.audience,
        include_hashtags=request.include_hashtags,
        hashtag_strategy=request.hashtag_strategy,
        hashtag_count=request.hashtag_count,
        brand_context=request.brand_context
    )
    
    # Get timezone insights if needed (non-blocking)
    timezone_insights = None
    if request.timezone and request.timezone != "UTC":
        try:
            timezone_insights = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(gmt_system.get_timezone_insights, request.timezone)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                ),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            pass  # Continue without timezone insights
    
    processing_time = time.perf_counter() - start_time
    
    return CaptionGenerationResponse(
        status="success",
        variations=captions_result.variations,
        best_variation_id=captions_result.best_variation_id,
        timezone_insights=timezone_insights,
        engagement_recommendations=captions_result.engagement_recommendations,
        generation_metadata={
            **captions_result.metadata,
            "processing_time_ms": round(processing_time * 1000, 2),
            "api_version": "3.0.0"
        }
    )


@router.post("/analyze-quality", response_model=QualityMetricsResponse)
@handle_errors
@smart_cache(ttl=3600)
async def analyze_quality(
    request: QualityAnalysisRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine)
) -> QualityMetricsResponse:
    """Analyze caption quality with smart caching."""
    
    if not request.caption.strip():
        raise ValueError("Caption cannot be empty")
    
    # Use thread pool for CPU-intensive analysis
    quality_metrics = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        engine.analyze_quality,
        caption=request.caption,
        style=request.style,
        audience=request.audience
    )
    
    return QualityMetricsResponse(**quality_metrics.model_dump())


@router.post("/optimize", response_model=CaptionOptimizationResponse)
@handle_errors
@smart_cache(ttl=1800)
async def optimize_caption(
    request: CaptionOptimizationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine)
) -> CaptionOptimizationResponse:
    """Optimize caption with parallel analysis."""
    
    if not request.caption.strip():
        raise ValueError("Caption cannot be empty")
    
    # Parallel execution for maximum speed
    original_task = asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        engine.analyze_quality,
        caption=request.caption,
        style=request.style,
        audience=request.audience
    )
    
    optimization_task = engine.optimize_content(
        caption=request.caption,
        style=request.style,
        audience=request.audience,
        enhancement_level=request.enhancement_level,
        preserve_meaning=request.preserve_meaning
    )
    
    original_quality, (optimized_caption, optimized_quality) = await asyncio.gather(
        original_task,
        optimization_task
    )
    
    improvement_score = 0.0
    if original_quality.overall_score > 0:
        improvement_score = (
            (optimized_quality.overall_score - original_quality.overall_score) /
            original_quality.overall_score * 100
        )
    
    return CaptionOptimizationResponse(
        status="success",
        original_caption=request.caption,
        optimized_caption=optimized_caption,
        original_quality=QualityMetricsResponse(**original_quality.model_dump()),
        optimized_quality=QualityMetricsResponse(**optimized_quality.model_dump()),
        improvements_applied=optimized_quality.suggestions,
        improvement_score=round(improvement_score, 2)
    )


@router.post("/batch-optimize")
@handle_errors
async def batch_optimize(
    request: BatchOptimizationRequest,
    engine: InstagramCaptionsEngine = Depends(get_captions_engine)
) -> StreamingResponse:
    """Stream batch optimization results for immediate feedback."""
    
    if not request.captions:
        raise ValueError("At least one caption is required")
    
    async def process_streaming():
        
    """process_streaming function."""
yield f'{{"status": "processing", "total": {len(request.captions)}, "results": ['
        
        first = True
        
        async def optimize_single(caption: str, index: int):
            
    """optimize_single function."""
try:
                # Quick parallel processing
                original_task = asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    engine.analyze_quality, caption, request.style, request.audience
                )
                optimization_task = engine.optimize_content(
                    caption, request.style, request.audience, request.enhancement_level
                )
                
                original, (optimized, optimized_quality) = await asyncio.gather(
                    original_task, optimization_task
                )
                
                improvement = 0.0
                if original.overall_score > 0:
                    improvement = (
                        (optimized_quality.overall_score - original.overall_score) /
                        original.overall_score * 100
                    )
                
                return {
                    "index": index,
                    "status": "success",
                    "original": caption,
                    "optimized": optimized,
                    "improvement": round(improvement, 2)
                }
            except Exception as e:
                return {
                    "index": index,
                    "status": "error",
                    "original": caption,
                    "error": str(e)
                }
        
        # Process in parallel batches of 10
        for i in range(0, len(request.captions), 10):
            batch = request.captions[i:i + 10]
            tasks = [optimize_single(caption, i + j) for j, caption in enumerate(batch)]
            
            for completed in asyncio.as_completed(tasks):
                result = await completed
                
                if not first:
                    yield ","
                yield json.dumps(result)
                first = False
        
        yield '], "completed": true}'
    
    return StreamingResponse(
        process_streaming(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )


@router.get("/health", response_model=HealthCheckResponse)
@handle_errors
async def health_check(
    engine: InstagramCaptionsEngine = Depends(get_captions_engine),
    gmt_system: SimplifiedGMTSystem = Depends(get_gmt_system),
    redis_client = Depends(get_redis_client)
) -> HealthCheckResponse:
    """Quick health check with parallel testing."""
    
    async def check_engine():
        
    """check_engine function."""
try:
            test_quality = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                engine.analyze_quality, "Test caption"
            )
            return {"status": "healthy", "test_score": test_quality.overall_score}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_gmt():
        
    """check_gmt function."""
try:
            insights = await asyncio.to_thread(
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                gmt_system.get_timezone_insights, "UTC"
            )
            return {"status": "healthy", "timezones": len(insights.optimal_posting_times)}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def check_redis():
        
    """check_redis function."""
if not redis_client:
            return {"status": "disabled"}
        try:
            await redis_client.ping()
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    # Parallel health checks
    engine_health, gmt_health, redis_health = await asyncio.gather(
        check_engine(), check_gmt(), check_redis(), return_exceptions=True
    )
    
    all_healthy = all(
        isinstance(h, dict) and h.get("status") == "healthy"
        for h in [engine_health, gmt_health]
    )
    
    return HealthCheckResponse(
        status="healthy" if all_healthy else "degraded",
        components={
            "captions_engine": engine_health,
            "gmt_system": gmt_health,
            "redis_cache": redis_health
        },
        performance_metrics=get_performance_metrics()
    )


@router.get("/metrics")
@smart_cache(ttl=60)
async def get_metrics() -> Dict[str, Any]:
    """Get performance metrics."""
    return get_performance_metrics()


def get_performance_metrics() -> Dict[str, Any]:
    """Get current performance statistics."""
    total_requests = _metrics["requests"]
    cache_hits = _metrics["cache_hits"]
    hit_rate = (cache_hits / total_requests * 100) if total_requests > 0 else 0
    
    return {
        "api_version": "3.0.0",
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(hit_rate, 2),
        "avg_response_time": round(_metrics["avg_time"], 3),
        "cache_size": len(_cache),
        "performance_tier": "ultra_fast" if hit_rate > 80 else "optimized"
    }


@router.delete("/cache")
async def clear_cache() -> Dict[str, Any]:
    """Clear cache for testing."""
    global _cache, _cache_times, _metrics
    
    cache_size = len(_cache)
    _cache.clear()
    _cache_times.clear()
    _metrics = {"requests": 0, "cache_hits": 0, "avg_time": 0.0}
    
    return {"status": "cache_cleared", "items_cleared": cache_size}


# Startup and shutdown events
async def startup():
    """Initialize optimizations on startup."""
    logger.info("Instagram Captions API v3.0 starting up...")
    # Pre-warm cache with common requests
    common_requests = [
        ("Test content", "casual", "general"),
        ("Business announcement", "professional", "business"),
        ("Creative project", "creative", "millennials")
    ]
    
    for content, style, audience in common_requests:
        cache_key = f"analyze_quality:{hash(content + style + audience)}"
        _cache[cache_key] = {"warmed": True}
        _cache_times[cache_key] = time.time()
    
    logger.info("Cache pre-warmed with common requests")


async def shutdown():
    """Cleanup on shutdown."""
    global _cache, _cache_times
    _cache.clear()
    _cache_times.clear()
    logger.info("Instagram Captions API v3.0 shutdown complete") 