"""
FastAPI router for multi-model API
Optimized with async operations, caching, and health monitoring
"""

import asyncio
import logging
import time
import uuid
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Request, Header
from fastapi.responses import JSONResponse, ORJSONResponse, StreamingResponse, Response
from pydantic import BaseModel, Field

try:
    import sentry_sdk
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False

try:
    import orjson
    ORJSON_AVAILABLE = True
except ImportError:
    ORJSON_AVAILABLE = False

from ..core.performance import fast_json_dumps, parallel_map
from ..core.response_optimizer import ResponseOptimizer

from ..core.models import ModelRegistry, get_registry
from ..core.cache import EnhancedCache, get_cache
from ..core.rate_limiter import get_rate_limiter, RateLimitInfo
from ..core.consensus import apply_consensus
from ..integrations.openrouter import get_openrouter_client
from ..api.schemas import (
    MultiModelRequest,
    MultiModelResponse,
    ModelResponse,
    ModelConfig,
    ModelType,
    ModelsListResponse,
    ModelStatus,
    BatchMultiModelRequest,
    BatchMultiModelResponse
)
from .helpers import (
    build_model_kwargs,
    validate_rate_limit,
    validate_enabled_models,
    validate_responses,
    build_response_data,
    get_model_types_str,
    get_weights_map
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/multi-model", tags=["Multi-Model API"])


def get_model_registry() -> ModelRegistry:
    """Dependency to get model registry"""
    return get_registry()


def get_cache_instance() -> EnhancedCache:
    """Dependency to get cache instance"""
    return get_cache()


def get_client_identifier(request: Request, x_api_key: Optional[str] = Header(None)) -> str:
    """Get client identifier for rate limiting"""
    if x_api_key:
        return f"api_key:{x_api_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"


async def check_rate_limit(
    identifier: str,
    endpoint: str = "execute"
) -> RateLimitInfo:
    """Check rate limit for a request"""
    rate_limiter = get_rate_limiter()
    return await rate_limiter.is_allowed(identifier, endpoint)


@router.post("/execute", response_model=MultiModelResponse)
async def execute_multi_model(
    request: MultiModelRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance),
    background_tasks: BackgroundTasks = None
):
    """
    Execute multiple AI models with optimized parallel processing
    
    Features:
    - Rate limiting protection
    - Multi-tier caching
    - Parallel, sequential, or consensus strategies
    - Partial success handling
    - Configurable timeouts
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    client_id = get_client_identifier(http_request)
    rate_limit_info = await check_rate_limit(client_id, "execute")
    validate_rate_limit(rate_limit_info)
    
    enabled_models = [m for m in request.models if m.is_enabled]
    validate_enabled_models(enabled_models)
    
    model_types_str = get_model_types_str(enabled_models)
    cache_key = cache._generate_key(
        "multi_model",
        request.prompt,
        model_types_str,
        request.strategy,
        request.consensus_method or "majority"
    )
    
    if request.cache_enabled:
        cached_response = await cache.get(cache_key)
        if cached_response:
            logger.info(f"Cache hit for request {request_id}")
            return MultiModelResponse(
                **cached_response,
                cache_hit=True
            )
    
    try:
        timeout = request.timeout or 30.0
        
        if request.strategy == "parallel":
            responses = await asyncio.wait_for(
                _execute_parallel(
                    enabled_models,
                    request.prompt,
                    registry,
                    timeout=timeout
                ),
                timeout=timeout
            )
        elif request.strategy == "sequential":
            responses = await asyncio.wait_for(
                _execute_sequential(
                    enabled_models,
                    request.prompt,
                    registry,
                    timeout=timeout
                ),
                timeout=timeout
            )
        elif request.strategy == "consensus":
            responses = await asyncio.wait_for(
                _execute_consensus(
                    enabled_models,
                    request.prompt,
                    registry,
                    consensus_method=request.consensus_method or "majority",
                    timeout=timeout
                ),
                timeout=timeout
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")
        
        validate_responses(responses, request)
        
        weights_map = get_weights_map(enabled_models)
        consensus_method = request.consensus_method or "majority"
        aggregated_response = _aggregate_responses(
            responses,
            request.strategy,
            consensus_method,
            weights_map
        )
        
        response_data = build_response_data(
            request_id,
            request,
            responses,
            aggregated_response,
            start_time,
            enabled_models
        )
        
        if request.cache_enabled:
            cache_ttl = request.cache_ttl or 3600
            background_tasks.add_task(
                cache.set,
                cache_key,
                response_data,
                ttl=cache_ttl
            )
        
        if ORJSON_AVAILABLE:
            response_size = len(fast_json_dumps(response_data))
            if response_size > 1024:
                compressed, is_compressed = ResponseOptimizer.compress_response(response_data, threshold=1024)
                if is_compressed:
                    return Response(
                        content=compressed,
                        media_type="application/json",
                        headers={"Content-Encoding": "gzip", "Content-Length": str(len(compressed))}
                    )
            return ORJSONResponse(content=response_data)
        
        return MultiModelResponse(**response_data)
    
    except asyncio.TimeoutError:
        logger.error(f"Request {request_id} timed out after {timeout}s")
        if SENTRY_AVAILABLE:
            sentry_sdk.capture_message(
                f"Request timeout: {request_id}",
                level="warning",
                extra={"request_id": request_id, "timeout": timeout}
            )
        raise HTTPException(status_code=504, detail=f"Request timed out after {timeout} seconds")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing multi-model request {request_id}: {e}", exc_info=True)
        if SENTRY_AVAILABLE:
            sentry_sdk.capture_exception(
                e,
                extra={
                    "request_id": request_id,
                    "prompt_length": len(request.prompt),
                    "strategy": request.strategy,
                    "models_count": len(enabled_models)
                }
            )
        raise HTTPException(status_code=500, detail=str(e))


async def _execute_parallel(
    models: List[ModelConfig],
    prompt: str,
    registry: ModelRegistry,
    timeout: Optional[float] = None
) -> List[ModelResponse]:
    """Execute models in parallel with optimized error handling"""
    if not models:
        return []
    
    tasks = [
        registry.execute_model(
            model.model_type,
            prompt,
            **build_model_kwargs(model, timeout)
        )
        for model in models
    ]
    
    if timeout:
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=timeout
        )
    else:
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Optimized: list comprehension with early error handling
    responses = [
        result if not isinstance(result, Exception) else ModelResponse(
            model_type=model.model_type,
            response="",
            success=False,
            error=str(result)
        )
        for model, result in zip(models, results)
    ]
    
    # Log errors separately to avoid overhead in hot path
    for model, result in zip(models, results):
        if isinstance(result, Exception):
            logger.error(f"Model {model.model_type} failed: {result}")
    
    return responses


async def _execute_sequential(
    models: List[ModelConfig],
    prompt: str,
    registry: ModelRegistry,
    timeout: Optional[float] = None
) -> List[ModelResponse]:
    """Execute models sequentially with error handling"""
    responses = []
    for model in models:
        try:
            response = await asyncio.wait_for(
                registry.execute_model(
                    model.model_type,
                    prompt,
                    **build_model_kwargs(model, timeout)
                ),
                timeout=timeout
            )
            responses.append(response)
        except asyncio.TimeoutError:
            logger.error(f"Model {model.model_type} timed out")
            responses.append(ModelResponse(
                model_type=model.model_type,
                response="",
                success=False,
                error=f"Timeout after {timeout}s"
            ))
        except Exception as e:
            logger.error(f"Model {model.model_type} failed: {e}")
            responses.append(ModelResponse(
                model_type=model.model_type,
                response="",
                success=False,
                error=str(e)
            ))
    return responses


async def _execute_consensus(
    models: List[ModelConfig],
    prompt: str,
    registry: ModelRegistry,
    consensus_method: str = "majority",
    timeout: Optional[float] = None
) -> List[ModelResponse]:
    """Execute models and use consensus/voting"""
    responses = await _execute_parallel(models, prompt, registry, timeout=timeout)
    return responses


def _aggregate_responses(
    responses: List[ModelResponse],
    strategy: str,
    consensus_method: str = "majority",
    weights: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """Aggregate multiple model responses - optimized"""
    # Optimized: single pass filter with direct attribute access
    successful_responses = [r for r in responses if r.success and r.response]
    
    if not successful_responses:
        return None
    
    if strategy == "consensus":
        return apply_consensus(successful_responses, consensus_method, weights)
    
    if len(successful_responses) == 1:
        return successful_responses[0].response
    
    # Optimized: list comprehension with cached latency formatting
    parts = [
        f"**{r.model_type.value}** (latency: {r.latency_ms:.2f}ms):\n{r.response}"
        if r.latency_ms is not None
        else f"**{r.model_type.value}** (latency: N/A):\n{r.response}"
        for r in successful_responses
    ]
    
    return "\n\n---\n\n".join(parts)


@router.get("/models", response_model=ModelsListResponse)
async def list_models(
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    List all available models with status
    
    Returns:
    - List of all registered models
    - Availability status
    - Performance metrics (success rate, latency)
    - Last used timestamp
    """
    available_models = registry.get_available_models()
    
    model_statuses = []
    for model_meta in registry.models.values():
        model_statuses.append(ModelStatus(
            model_type=model_meta.model_type,
            is_available=model_meta.is_available,
            is_enabled=model_meta.is_available,
            multiplier=1,
            last_used=model_meta.last_used.isoformat() if model_meta.last_used else None,
            success_rate=round(model_meta.success_rate, 2),
            avg_latency_ms=round(model_meta.avg_latency_ms, 2)
        ))
    
    return ModelsListResponse(
        models=model_statuses,
        total_available=len(available_models),
        total_enabled=len([m for m in model_statuses if m.is_enabled])
    )


@router.get("/models/{model_type}/health")
async def get_model_health(
    model_type: ModelType = Path(..., description="Model type identifier"),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    Get health metrics for a specific model
    
    Returns detailed health information including:
    - Availability status
    - Success rate and error count
    - Latency metrics (average, p95)
    - Circuit breaker state
    - Last used timestamp
    """
    health = registry.get_model_health(model_type)
    if "error" in health:
        raise HTTPException(status_code=404, detail=health["error"])
    return health


@router.get("/health")
async def health_check(
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Comprehensive health check
    
    Returns system health status including:
    - Overall system status (healthy/degraded/unhealthy)
    - Cache health
    - Model availability
    - Circuit breaker states
    """
    cache_stats = await cache.get_stats()
    available_models = len(registry.get_available_models())
    total_models = len(registry.models)
    
    circuit_breakers_open = sum(
        1 for m in registry.models.values()
        if m.circuit_breaker.state == "open"
    )
    
    error_rate = 0.0
    if total_models > 0:
        total_calls = sum(m.call_count for m in registry.models.values())
        total_errors = sum(m.error_count for m in registry.models.values())
        if total_calls > 0:
            error_rate = (total_errors / total_calls) * 100
    
    is_healthy = (
        available_models > 0 and
        error_rate < 10.0 and
        circuit_breakers_open < total_models / 2
    )
    
    status = "healthy" if is_healthy else "degraded"
    if available_models == 0:
        status = "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "cache": {
            "enabled": True,
            "hit_rate": round(cache_stats.get("hit_rate", 0), 2),
            "l1_size": cache_stats.get("l1_size", 0),
            "l1_max_size": cache_stats.get("l1_max_size", 0),
            "l1_utilization": round(cache_stats.get("l1_utilization", 0), 2),
            "l2_enabled": cache_stats.get("l2_enabled", False),
            "l3_enabled": cache_stats.get("l3_enabled", False)
        },
        "models": {
            "total_available": available_models,
            "total_registered": total_models,
            "circuit_breakers_open": circuit_breakers_open,
            "error_rate": round(error_rate, 2)
        },
        "system": {
            "uptime_seconds": time.time(),
            "version": "2.6.0"
        }
    }


@router.get("/stats")
async def get_stats(
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Get comprehensive statistics
    
    Returns detailed metrics including:
    - Cache performance (hit rate, size, latency)
    - Model performance (success rates, latencies, circuit breaker states)
    - System health metrics
    - Rate limiting configuration
    """
    cache_stats = await cache.get_stats()
    
    model_stats = []
    total_calls = 0
    total_success = 0
    total_errors = 0
    total_latency = 0.0
    circuit_breakers_open = 0
    
    for model_type, model_meta in registry.models.items():
        model_stats.append({
            "model_type": model_type.value,
            "name": model_meta.name,
            "call_count": model_meta.call_count,
            "success_count": model_meta.success_count,
            "error_count": model_meta.error_count,
            "success_rate": round(model_meta.success_rate, 2),
            "avg_latency_ms": round(model_meta.avg_latency_ms, 2),
            "p95_latency_ms": round(model_meta.p95_latency_ms, 2),
            "circuit_breaker_state": model_meta.circuit_breaker.state,
            "circuit_breaker_failures": model_meta.circuit_breaker.failures,
            "is_available": model_meta.is_available,
            "last_used": model_meta.last_used.isoformat() if model_meta.last_used else None
        })
        
        total_calls += model_meta.call_count
        total_success += model_meta.success_count
        total_errors += model_meta.error_count
        total_latency += model_meta.total_latency_ms
        if model_meta.circuit_breaker.state == "open":
            circuit_breakers_open += 1
    
    overall_success_rate = (total_success / total_calls * 100) if total_calls > 0 else 0.0
    overall_avg_latency = (total_latency / total_calls) if total_calls > 0 else 0.0
    
    rate_limiter = get_rate_limiter()
    
    return {
        "cache": cache_stats,
        "models": model_stats,
        "summary": {
            "total_calls": total_calls,
            "total_success": total_success,
            "total_errors": total_errors,
            "overall_success_rate": round(overall_success_rate, 2),
            "overall_avg_latency_ms": round(overall_avg_latency, 2),
            "circuit_breakers_open": circuit_breakers_open,
            "total_models": len(registry.models),
            "available_models": len(registry.get_available_models())
        },
        "rate_limiting": {
            "default_limit": rate_limiter.default_limit,
            "default_window": rate_limiter.default_window,
            "burst_limit": rate_limiter.burst_limit
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/execute/batch", response_model=BatchMultiModelResponse)
async def execute_batch_multi_model(
    batch_request: BatchMultiModelRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance),
    background_tasks: BackgroundTasks = None
):
    """
    Execute multiple multi-model requests in batch
    
    Processes up to 10 requests in parallel with rate limiting protection
    """
    batch_id = str(uuid.uuid4())
    start_time = time.time()
    
    client_id = get_client_identifier(http_request)
    rate_limit_info = await check_rate_limit(client_id, "batch")
    validate_rate_limit(rate_limit_info)
    
    responses = []
    successful_count = 0
    failed_count = 0
    
    async def process_single_request(req: MultiModelRequest) -> Optional[MultiModelResponse]:
        """Process a single request"""
        try:
            request_id = str(uuid.uuid4())
            start_time = time.time()
            
            enabled_models = [m for m in req.models if m.is_enabled]
            validate_enabled_models(enabled_models)
            
            model_types_str = get_model_types_str(enabled_models)
            cache_key = cache._generate_key(
                "multi_model",
                req.prompt,
                model_types_str,
                req.strategy,
                req.consensus_method or "majority"
            )
            
            if req.cache_enabled:
                cached_response = await cache.get(cache_key)
                if cached_response:
                    return MultiModelResponse(**cached_response, cache_hit=True)
            
            timeout = req.timeout or 30.0
            
            if req.strategy == "parallel":
                responses = await asyncio.wait_for(
                    _execute_parallel(enabled_models, req.prompt, registry, timeout=timeout),
                    timeout=timeout
                )
            elif req.strategy == "sequential":
                responses = await asyncio.wait_for(
                    _execute_sequential(enabled_models, req.prompt, registry, timeout=timeout),
                    timeout=timeout
                )
            elif req.strategy == "consensus":
                responses = await asyncio.wait_for(
                    _execute_consensus(
                        enabled_models,
                        req.prompt,
                        registry,
                        consensus_method=req.consensus_method or "majority",
                        timeout=timeout
                    ),
                    timeout=timeout
                )
            else:
                raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")
            
            validate_responses(responses, req)
            
            weights_map = get_weights_map(enabled_models)
            consensus_method = req.consensus_method or "majority"
            aggregated_response = _aggregate_responses(
                responses,
                req.strategy,
                consensus_method,
                weights_map
            )
            
            response_data = build_response_data(
                request_id,
                req,
                responses,
                aggregated_response,
                start_time,
                enabled_models
            )
            
            if req.cache_enabled:
                cache_ttl = req.cache_ttl or 3600
                background_tasks.add_task(cache.set, cache_key, response_data, ttl=cache_ttl)
            
            return MultiModelResponse(**response_data)
        except Exception as e:
            logger.error(f"Batch request failed: {e}", exc_info=True)
            if batch_request.stop_on_first_error:
                raise
            return None
    
    tasks = [process_single_request(req) for req in batch_request.requests]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for result in results:
        if isinstance(result, Exception):
            failed_count += 1
            if batch_request.stop_on_first_error:
                raise HTTPException(
                    status_code=500,
                    detail=f"Batch processing stopped due to error: {str(result)}"
                )
        elif result is not None:
            responses.append(result)
            successful_count += 1
        else:
            failed_count += 1
    
    total_latency_ms = (time.time() - start_time) * 1000
    
    return BatchMultiModelResponse(
        request_id=batch_id,
        total_requests=len(batch_request.requests),
        successful_requests=successful_count,
        failed_requests=failed_count,
        responses=responses,
        timestamp=datetime.now().isoformat()
    )


@router.delete("/cache")
async def clear_cache(
    level: Optional[str] = Query(None, description="Cache level: l1, l2, or both"),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """Clear cache"""
    success = await cache.clear(level=level)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to clear cache")
    return {"message": "Cache cleared successfully", "level": level or "both"}


@router.get("/rate-limit/info")
async def get_rate_limit_info(
    http_request: Request,
    endpoint: str = Query("execute", description="Endpoint name")
):
    """Get current rate limit information"""
    client_id = get_client_identifier(http_request)
    rate_limiter = get_rate_limiter()
    info = await rate_limiter.get_rate_limit_info(client_id, endpoint)
    return info


@router.get("/metrics")
async def get_prometheus_metrics():
    """
    Prometheus metrics endpoint
    
    Returns metrics in Prometheus format for scraping
    """
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Prometheus client not available"
        )


@router.get("/openrouter/models")
async def list_openrouter_models(
    provider: Optional[str] = Query(None, description="Filter by provider (e.g., 'openai', 'anthropic')"),
    search: Optional[str] = Query(None, description="Search models by name")
):
    """
    List all available OpenRouter models
    
    Returns:
    - List of all available models from OpenRouter
    - Model details including pricing, context length, etc.
    - Filtered by provider or search term if provided
    """
    try:
        client = get_openrouter_client()
        models = await client.list_models()
        
        if not models:
            return {
                "models": [],
                "total": 0,
                "message": "No models available. Check OPENROUTER_API_KEY configuration."
            }
        
        filtered_models = models
        
        if provider:
            filtered_models = [
                m for m in filtered_models 
                if m.get("id", "").startswith(f"{provider}/")
            ]
        
        if search:
            search_lower = search.lower()
            filtered_models = [
                m for m in filtered_models
                if search_lower in m.get("id", "").lower() or 
                   search_lower in m.get("name", "").lower()
            ]
        
        formatted_models = []
        for model in filtered_models[:100]:
            formatted_models.append({
                "id": model.get("id"),
                "name": model.get("name"),
                "description": model.get("description"),
                "context_length": model.get("context_length"),
                "pricing": model.get("pricing", {}),
                "architecture": model.get("architecture", {}),
                "top_provider": model.get("top_provider", {}),
                "permission": model.get("permission")
            })
        
        return {
            "models": formatted_models,
            "total": len(formatted_models),
            "filtered": len(formatted_models) < len(models),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing OpenRouter models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list OpenRouter models: {str(e)}"
        )


@router.post("/execute/stream")
async def execute_multi_model_stream(
    request: MultiModelRequest,
    http_request: Request,
    registry: ModelRegistry = Depends(get_model_registry),
    cache: EnhancedCache = Depends(get_cache_instance)
):
    """
    Execute multiple AI models with streaming responses (SSE)
    
    Returns Server-Sent Events stream with:
    - Model responses as they complete
    - Progress updates
    - Final aggregated response
    """
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    client_id = get_client_identifier(http_request)
    rate_limit_info = await check_rate_limit(client_id, "execute")
    validate_rate_limit(rate_limit_info)
    
    enabled_models = [m for m in request.models if m.is_enabled]
    validate_enabled_models(enabled_models)
    
    async def generate_stream():
        """Generate SSE stream"""
        try:
            yield f"data: {fast_json_dumps({'type': 'start', 'request_id': request_id, 'models_count': len(enabled_models)})}\n\n"
            
            timeout = request.timeout or 30.0
            responses = []
            
            if request.strategy == "parallel":
                # Optimized: create tasks and process as they complete
                tasks = {
                    asyncio.create_task(registry.execute_model(
                        model.model_type,
                        request.prompt,
                        **build_model_kwargs(model, timeout)
                    )): model
                    for model in enabled_models
                }
                
                # Process tasks as they complete (faster than sequential)
                try:
                    for task in asyncio.as_completed(tasks.keys()):
                        model = tasks[task]
                        try:
                            response = await asyncio.wait_for(task, timeout=timeout)
                            responses.append(response)
                            # Optimized: build response dict once
                            response_data = {
                                'type': 'model_response',
                                'model_type': model.model_type.value,
                                'response': response.response if response.success else None,
                                'success': response.success,
                                'error': response.error,
                                'latency_ms': response.latency_ms,
                                'tokens_used': response.tokens_used
                            }
                            yield f"data: {fast_json_dumps(response_data)}\n\n"
                        except Exception as e:
                            error_response = ModelResponse(
                                model_type=model.model_type,
                                response="",
                                success=False,
                                error=str(e)
                            )
                            responses.append(error_response)
                            yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': str(e)})}\n\n"
                except asyncio.TimeoutError:
                    # Handle remaining tasks
                    for task, model in tasks.items():
                        if not task.done():
                            error_response = ModelResponse(
                                model_type=model.model_type,
                                response="",
                                success=False,
                                error=f"Timeout after {timeout}s"
                            )
                            responses.append(error_response)
                            yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': f'Timeout after {timeout}s'})}\n\n"
            
            elif request.strategy == "sequential":
                for model in enabled_models:
                    try:
                        response = await asyncio.wait_for(
                            registry.execute_model(
                                model.model_type,
                                request.prompt,
                                **build_model_kwargs(model, timeout)
                            ),
                            timeout=timeout
                        )
                        responses.append(response)
                        # Optimized: build response dict once
                        response_data = {
                            'type': 'model_response',
                            'model_type': model.model_type.value,
                            'response': response.response if response.success else None,
                            'success': response.success,
                            'error': response.error,
                            'latency_ms': response.latency_ms,
                            'tokens_used': response.tokens_used
                        }
                        yield f"data: {fast_json_dumps(response_data)}\n\n"
                    except Exception as e:
                        error_response = ModelResponse(
                            model_type=model.model_type,
                            response="",
                            success=False,
                            error=str(e)
                        )
                        responses.append(error_response)
                        yield f"data: {fast_json_dumps({'type': 'model_error', 'model_type': model.model_type.value, 'error': str(e)})}\n\n"
            
            success_count = sum(1 for r in responses if r.success)
            failure_count = len(responses) - success_count
            
            if request.strategy == "consensus":
                successful_responses = [r for r in responses if r.success and r.response]
                if successful_responses:
                    weights_map = get_weights_map(enabled_models)
                    consensus_method = request.consensus_method or "majority"
                    consensus_result = apply_consensus(
                        successful_responses,
                        consensus_method,
                        weights_map
                    )
                    yield f"data: {fast_json_dumps({'type': 'consensus', 'result': consensus_result, 'method': consensus_method})}\n\n"
            
            weights_map = get_weights_map(enabled_models)
            consensus_method = request.consensus_method or "majority"
            aggregated_response = _aggregate_responses(
                responses,
                request.strategy,
                consensus_method,
                weights_map
            )
            
            response_data = build_response_data(
                request_id,
                request,
                responses,
                aggregated_response,
                start_time,
                enabled_models
            )
            
            yield f"data: {fast_json_dumps({'type': 'complete', 'request_id': request_id, 'aggregated_response': aggregated_response, 'total_tokens': response_data['total_tokens'], 'total_latency_ms': response_data['total_latency_ms'], 'success_count': response_data['success_count'], 'failure_count': response_data['failure_count'], 'timestamp': response_data['timestamp']})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error for request {request_id}: {e}", exc_info=True)
            yield f"data: {fast_json_dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

