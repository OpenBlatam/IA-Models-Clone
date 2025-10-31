from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from fastapi import (
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
    import orjson
    import json as orjson
    import redis.asyncio as aioredis
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from prometheus_client import Counter, Histogram, Gauge, generate_latest
    from prometheus_fastapi_instrumentator import Instrumentator
import structlog
from .models import (
from .optimized_service import get_optimized_service
    from fastapi.responses import PlainTextResponse
        from .models import CopyVariant
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized Production Copywriting API.

High-performance FastAPI with advanced features:
- Language support, tone, voice, variants, creativity
- Translation capabilities, use cases, website info
- Redis caching, rate limiting, monitoring
- Prometheus metrics, structured logging
"""


# FastAPI imports
    APIRouter, HTTPException, Query, Depends, Body, 
    status, Request, Security, BackgroundTasks
)

# High-performance imports
try:
    JSON_AVAILABLE = True
except ImportError:
    JSON_AVAILABLE = False

try:
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Rate limiting
try:
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

# Monitoring
try:
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Logging

# Import models and service
    CopywritingInput, CopywritingOutput, Language, CopyTone, 
    UseCase, CreativityLevel, WebsiteInfo, BrandVoice,
    TranslationSettings, VariantSettings
)

logger = structlog.get_logger(__name__)

# === CONFIGURATION ===
API_KEY = "ultra-optimized-copywriting-2024"
REDIS_URL = "redis://localhost:6379/1"

# === PROMETHEUS METRICS ===
if PROMETHEUS_AVAILABLE:
    API_REQUESTS = Counter('copywriting_api_requests_total', 'Total API requests', ['endpoint', 'method'])
    API_DURATION = Histogram('copywriting_api_duration_seconds', 'API request duration', ['endpoint'])
    API_ERRORS = Counter('copywriting_api_errors_total', 'API errors', ['endpoint', 'error_type'])
    ACTIVE_CONNECTIONS = Gauge('copywriting_active_connections', 'Active connections')
    CACHE_OPERATIONS = Counter('copywriting_cache_operations_total', 'Cache operations', ['operation', 'result'])

# === RATE LIMITING ===
if RATE_LIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# === SECURITY ===
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    """Validate API key."""
    if api_key != API_KEY:
        if PROMETHEUS_AVAILABLE:
            API_ERRORS.labels(endpoint="auth", error_type="invalid_key").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

# === REDIS CLIENT ===
redis_client: Optional[aioredis.Redis] = None

async def get_redis():
    """Get Redis client."""
    global redis_client
    if redis_client is None and REDIS_AVAILABLE:
        try:
            redis_client = await aioredis.from_url(
                REDIS_URL,
                max_connections=20,
                encoding="utf-8",
                decode_responses=True
            )
            await redis_client.ping()
            logger.info("Redis connected for API")
        except Exception as e:
            logger.warning("Redis connection failed", error=str(e))
    return redis_client

# === ROUTER SETUP ===
router = APIRouter(
    prefix="/copywriting/v2",
    tags=["copywriting-optimized"],
    responses={
        401: {"description": "Invalid API Key"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Add rate limiting to router
if RATE_LIMIT_AVAILABLE and limiter:
    router.state.limiter = limiter
    router.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# === PERFORMANCE MONITORING ===
@router.middleware("http")
async def monitor_performance(request: Request, call_next):
    """Monitor API performance."""
    start_time = time.perf_counter()
    
    if PROMETHEUS_AVAILABLE:
        ACTIVE_CONNECTIONS.inc()
    
    try:
        response = await call_next(request)
        
        # Log performance
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        
        if PROMETHEUS_AVAILABLE:
            API_REQUESTS.labels(endpoint=endpoint, method=request.method).inc()
            API_DURATION.labels(endpoint=endpoint).observe(duration)
        
        logger.info("API request completed",
                   endpoint=endpoint,
                   method=request.method,
                   duration_ms=duration * 1000,
                   status_code=response.status_code)
        
        return response
        
    except Exception as e:
        duration = time.perf_counter() - start_time
        endpoint = request.url.path
        
        if PROMETHEUS_AVAILABLE:
            API_ERRORS.labels(endpoint=endpoint, error_type=type(e).__name__).inc()
        
        logger.error("API request failed",
                    endpoint=endpoint,
                    method=request.method,
                    duration_ms=duration * 1000,
                    error=str(e))
        raise
    
    finally:
        if PROMETHEUS_AVAILABLE:
            ACTIVE_CONNECTIONS.dec()

# === API ENDPOINTS ===

@router.get("/health", summary="Health check endpoint")
async def health_check():
    """Health check with system status."""
    redis = await get_redis()
    service = await get_optimized_service()
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "services": {
            "redis": redis is not None,
            "copywriting_service": service is not None,
            "rate_limiter": RATE_LIMIT_AVAILABLE,
            "metrics": PROMETHEUS_AVAILABLE
        },
        "optimization_libraries": {
            "orjson": JSON_AVAILABLE,
            "redis": REDIS_AVAILABLE,
            "prometheus": PROMETHEUS_AVAILABLE,
            "rate_limiting": RATE_LIMIT_AVAILABLE
        }
    }
    
    return health_status

@router.get("/metrics", include_in_schema=False)
async def get_metrics():
    """Prometheus metrics endpoint."""
    if not PROMETHEUS_AVAILABLE:
        raise HTTPException(
            status_code=404,
            detail="Metrics not available. Install prometheus-client."
        )
    
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )

@router.get("/capabilities", summary="Get service capabilities")
async def get_capabilities(api_key: str = Depends(get_api_key)):
    """Get detailed service capabilities and supported features."""
    service = await get_optimized_service()
    stats = await service.get_performance_stats()
    
    capabilities = {
        "languages": [lang.value for lang in Language],
        "tones": [tone.value for tone in CopyTone],
        "use_cases": [case.value for case in UseCase],
        "creativity_levels": [level.value for level in CreativityLevel],
        "max_variants": 20,
        "translation_support": True,
        "website_info_support": True,
        "brand_voice_support": True,
        "performance_stats": stats,
        "features": {
            "parallel_generation": True,
            "caching": True,
            "rate_limiting": RATE_LIMIT_AVAILABLE,
            "metrics": PROMETHEUS_AVAILABLE,
            "background_tasks": True,
            "batch_processing": True
        }
    }
    
    return capabilities

@router.post(
    "/generate",
    response_model=CopywritingOutput,
    summary="Generate optimized copywriting content",
    description="Generate high-quality copywriting with advanced AI and optimization features"
)
async def generate_copy(
    request_data: CopywritingInput = Body(
        ...,
        example={
            "product_description": "Plataforma de marketing digital con IA",
            "target_platform": "instagram",
            "content_type": "social_post",
            "tone": "professional",
            "use_case": "brand_awareness",
            "language": "es",
            "creativity_level": "creative",
            "website_info": {
                "website_name": "MarketingAI Pro",
                "about": "Automatizamos el marketing digital con inteligencia artificial",
                "features": ["Automatización", "Analytics", "Personalización"]
            },
            "brand_voice": {
                "tone": "professional",
                "voice_style": "tech",
                "personality_traits": ["innovador", "confiable", "experto"]
            },
            "variant_settings": {
                "max_variants": 5,
                "variant_diversity": 0.8
            }
        }
    ),
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Generate optimized copywriting content with advanced features."""
    
    try:
        # Get service
        service = await get_optimized_service()
        
        # Generate content
        result = await service.generate_copy(request_data)
        
        # Background task for analytics
        background_tasks.add_task(
            log_generation_analytics,
            request_data.tracking_id,
            len(result.variants),
            result.generation_time
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Content generation failed"
        )

@router.post(
    "/generate-batch",
    summary="Batch generate copywriting content",
    description="Generate multiple copywriting requests in parallel"
)
async def generate_batch(
    requests: List[CopywritingInput] = Body(..., max_items=10),
    wait_for_completion: bool = Query(False, description="Wait for all generations to complete"),
    api_key: str = Depends(get_api_key)
):
    """Generate multiple copywriting requests in batch."""
    
    if len(requests) > 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 10 requests per batch"
        )
    
    try:
        service = await get_optimized_service()
        
        if wait_for_completion:
            # Generate all in parallel
            tasks = [service.generate_copy(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            successful_results = []
            errors = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append({
                        "request_index": i,
                        "error": str(result)
                    })
                else:
                    successful_results.append({
                        "request_index": i,
                        "result": result
                    })
            
            return {
                "batch_id": str(uuid.uuid4()),
                "total_requests": len(requests),
                "successful": len(successful_results),
                "failed": len(errors),
                "results": successful_results,
                "errors": errors
            }
        
        else:
            # Return task IDs for async processing
            batch_id = str(uuid.uuid4())
            task_ids = []
            
            for i, req in enumerate(requests):
                task_id = f"{batch_id}_{i}"
                task_ids.append(task_id)
                # In production, you'd queue these with Celery or similar
            
            return {
                "batch_id": batch_id,
                "task_ids": task_ids,
                "status": "queued",
                "total_requests": len(requests)
            }
            
    except Exception as e:
        logger.error("Batch generation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch generation failed"
        )

@router.post(
    "/translate",
    summary="Translate existing copywriting content",
    description="Translate copywriting variants to multiple languages"
)
async def translate_content(
    variants: List[Dict[str, Any]] = Body(..., description="Variants to translate"),
    translation_settings: TranslationSettings = Body(..., description="Translation configuration"),
    api_key: str = Depends(get_api_key)
):
    """Translate existing copywriting content."""
    
    try:
        service = await get_optimized_service()
        
        # Convert dict variants to CopyVariant objects (simplified)
        copy_variants = []
        
        for variant_data in variants:
            variant = CopyVariant(
                variant_id=variant_data.get("variant_id", str(uuid.uuid4())),
                headline=variant_data.get("headline", ""),
                primary_text=variant_data.get("primary_text", ""),
                call_to_action=variant_data.get("call_to_action"),
                hashtags=variant_data.get("hashtags", []),
                character_count=variant_data.get("character_count", 0),
                word_count=variant_data.get("word_count", 0)
            )
            copy_variants.append(variant)
        
        # Apply translations
        translated_variants = await service._apply_translations(copy_variants, translation_settings)
        
        return {
            "original_variants": len(copy_variants),
            "translated_variants": len(translated_variants) - len(copy_variants),
            "total_variants": len(translated_variants),
            "target_languages": [lang.value for lang in translation_settings.target_languages],
            "variants": [variant.model_dump() for variant in translated_variants]
        }
        
    except Exception as e:
        logger.error("Translation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Translation failed"
        )

@router.get(
    "/templates",
    summary="Get available templates",
    description="Get optimized templates for different use cases and tones"
)
async def get_templates(
    use_case: Optional[UseCase] = Query(None, description="Filter by use case"),
    tone: Optional[CopyTone] = Query(None, description="Filter by tone"),
    language: Optional[Language] = Query(Language.es, description="Template language"),
    api_key: str = Depends(get_api_key)
):
    """Get available copywriting templates."""
    
    templates = {
        "product_launch": {
            "urgent": [
                {
                    "name": "Launch Alert",
                    "headline": "🚀 ¡{product} Ya Está Aquí!",
                    "text": "El momento que esperabas ha llegado. {product} revoluciona {benefit}.",
                    "cta": "¡Consíguelo Ahora!",
                    "use_cases": ["product_launch", "brand_awareness"]
                },
                {
                    "name": "Lightning Launch",
                    "headline": "⚡ Lanzamiento: {product}",
                    "text": "Por fin disponible. {product} cambia todo lo que conocías sobre {benefit}.",
                    "cta": "¡Pruébalo Ya!",
                    "use_cases": ["product_launch"]
                }
            ],
            "professional": [
                {
                    "name": "Professional Launch",
                    "headline": "Presentamos {product}",
                    "text": "Una nueva solución profesional para {benefit}. Diseñado para empresas que buscan excelencia.",
                    "cta": "Solicitar Demo",
                    "use_cases": ["product_launch", "lead_generation"]
                }
            ]
        },
        "brand_awareness": {
            "friendly": [
                {
                    "name": "Friendly Introduction",
                    "headline": "¡Hola! Somos {brand} 👋",
                    "text": "Nos dedicamos a hacer que {benefit} sea más fácil para ti.",
                    "cta": "¡Conócenos!",
                    "use_cases": ["brand_awareness", "social_media"]
                }
            ]
        }
    }
    
    # Filter templates
    filtered_templates = templates
    
    if use_case:
        filtered_templates = {
            k: v for k, v in templates.items() 
            if k == use_case.value
        }
    
    if tone:
        for use_case_key in filtered_templates:
            filtered_templates[use_case_key] = {
                k: v for k, v in filtered_templates[use_case_key].items()
                if k == tone.value
            }
    
    return {
        "language": language.value,
        "total_templates": sum(
            len(tones) for tones in filtered_templates.values()
        ),
        "templates": filtered_templates
    }

@router.get(
    "/analytics/{tracking_id}",
    summary="Get generation analytics",
    description="Get detailed analytics for a specific generation"
)
async def get_analytics(
    tracking_id: str,
    api_key: str = Depends(get_api_key)
):
    """Get analytics for a specific generation."""
    
    redis = await get_redis()
    
    if redis:
        try:
            # Get analytics from Redis
            analytics_key = f"analytics:{tracking_id}"
            analytics_data = await redis.hgetall(analytics_key)
            
            if analytics_data:
                return {
                    "tracking_id": tracking_id,
                    "analytics": analytics_data,
                    "retrieved_at": datetime.now().isoformat()
                }
        except Exception as e:
            logger.warning("Failed to retrieve analytics", error=str(e))
    
    # Return default analytics
    return {
        "tracking_id": tracking_id,
        "analytics": {
            "generations": 1,
            "avg_generation_time": "0.5s",
            "most_used_tone": "professional",
            "best_performing_variant": "variant_0"
        },
        "note": "Analytics data not available"
    }

@router.delete(
    "/cache",
    summary="Clear service cache",
    description="Clear all cached data (admin only)"
)
async def clear_cache(
    confirm: bool = Query(False, description="Confirm cache clearing"),
    api_key: str = Depends(get_api_key)
):
    """Clear service cache."""
    
    if not confirm:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cache clearing requires confirmation"
        )
    
    try:
        redis = await get_redis()
        service = await get_optimized_service()
        
        cleared_items = 0
        
        # Clear Redis cache
        if redis:
            keys = await redis.keys("copy:*")
            if keys:
                cleared_items += await redis.delete(*keys)
        
        # Clear service caches
        if hasattr(service, 'template_cache'):
            cleared_items += len(service.template_cache)
            service.template_cache.clear()
        
        if hasattr(service, 'metrics_cache'):
            cleared_items += len(service.metrics_cache)
            service.metrics_cache.clear()
        
        return {
            "status": "success",
            "cleared_items": cleared_items,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Cache clearing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache clearing failed"
        )

# === BACKGROUND TASKS ===

async def log_generation_analytics(tracking_id: str, variant_count: int, generation_time: float):
    """Log generation analytics to Redis."""
    redis = await get_redis()
    
    if redis:
        try:
            analytics_key = f"analytics:{tracking_id}"
            await redis.hset(analytics_key, mapping={
                "variant_count": variant_count,
                "generation_time": generation_time,
                "timestamp": datetime.now().isoformat(),
                "service_version": "2.0.0"
            })
            await redis.expire(analytics_key, 86400)  # 24 hours
            
            if PROMETHEUS_AVAILABLE:
                CACHE_OPERATIONS.labels(operation="analytics_set", result="success").inc()
                
        except Exception as e:
            logger.warning("Failed to log analytics", error=str(e))
            if PROMETHEUS_AVAILABLE:
                CACHE_OPERATIONS.labels(operation="analytics_set", result="error").inc()

# === STARTUP/SHUTDOWN ===

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting optimized copywriting API v2.0")
    
    # Initialize Redis
    await get_redis()
    
    # Initialize service
    await get_optimized_service()
    
    # Setup Prometheus if available
    if PROMETHEUS_AVAILABLE:
        instrumentator = Instrumentator()
        # instrumentator.instrument(app).expose(app, endpoint="/metrics")
    
    logger.info("Optimized copywriting API v2.0 started successfully")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down optimized copywriting API v2.0")
    
    # Cleanup service
    service = await get_optimized_service()
    await service.cleanup()
    
    # Close Redis
    if redis_client:
        await redis_client.close()
    
    logger.info("Optimized copywriting API v2.0 shutdown complete")

# Export router
__all__ = ["router"] 