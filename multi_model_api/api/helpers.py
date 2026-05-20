"""
Helper functions for multi-model API router
Refactored to eliminate code duplication
"""

from typing import List, Optional, Dict, Any
from fastapi import HTTPException, Request, Header
from datetime import datetime
import time

from ..api.schemas import ModelConfig, ModelResponse, MultiModelRequest
from ..core.rate_limiter import RateLimitInfo


def build_model_kwargs(
    model: ModelConfig,
    timeout: Optional[float] = None
) -> Dict[str, Any]:
    """Build kwargs for model execution - optimized"""
    # Direct attribute access (faster than property lookups)
    kwargs: Dict[str, Any] = {}
    
    if model.temperature is not None:
        kwargs["temperature"] = model.temperature
    if model.max_tokens is not None:
        kwargs["max_tokens"] = model.max_tokens
    if timeout is not None:
        kwargs["timeout"] = timeout
    
    # Batch update for custom params (most efficient)
    if model.custom_params:
        kwargs.update(model.custom_params)
    
    if model.openrouter_model:
        kwargs["openrouter_model"] = model.openrouter_model
    
    return kwargs


def validate_rate_limit(rate_limit_info: RateLimitInfo) -> None:
    """Validate rate limit and raise exception if exceeded"""
    if not rate_limit_info.allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {rate_limit_info.retry_after} seconds",
            headers={
                "X-RateLimit-Limit": str(rate_limit_info.limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(rate_limit_info.reset_at)),
                "Retry-After": str(rate_limit_info.retry_after or 60)
            }
        )


def validate_enabled_models(enabled_models: List[ModelConfig]) -> None:
    """Validate that at least one model is enabled"""
    if not enabled_models:
        raise HTTPException(
            status_code=400,
            detail="At least one model must be enabled"
        )


def validate_responses(
    responses: List[ModelResponse],
    request: MultiModelRequest
) -> None:
    """Validate responses against request requirements"""
    success_count, failure_count = calculate_response_stats(responses)
    
    if request.min_successful_models and success_count < request.min_successful_models:
        if not request.allow_partial_success:
            raise HTTPException(
                status_code=500,
                detail=f"Only {success_count} models succeeded, minimum {request.min_successful_models} required"
            )
    
    if not request.allow_partial_success and failure_count > 0:
        raise HTTPException(
            status_code=500,
            detail=f"{failure_count} model(s) failed and allow_partial_success is False"
        )


def calculate_latency_ms(start_time: float) -> float:
    """Calculate latency in milliseconds from start time"""
    return (time.time() - start_time) * 1000


def get_enabled_models(models: List[ModelConfig]) -> List[ModelConfig]:
    """Get list of enabled models - optimized helper"""
    return [m for m in models if m.is_enabled]


def calculate_response_stats(responses: List[ModelResponse]) -> tuple[int, int]:
    """
    Calculate success and failure counts from responses - optimized single pass
    
    Returns:
        Tuple of (success_count, failure_count)
    """
    success_count = sum(1 for r in responses if r.success)
    return success_count, len(responses) - success_count


def build_response_data(
    request_id: str,
    request: MultiModelRequest,
    responses: List[ModelResponse],
    aggregated_response: Optional[str],
    start_time: float,
    enabled_models: List[ModelConfig]
) -> Dict[str, Any]:
    """Build response data dictionary - optimized"""
    total_latency_ms = calculate_latency_ms(start_time)
    
    # Optimized: single pass to calculate tokens, success and failure counts
    total_tokens = 0
    success_count = 0
    for r in responses:
        if r.success:
            success_count += 1
            if r.tokens_used is not None:
                total_tokens += r.tokens_used
    
    failure_count = len(responses) - success_count
    partial_success = success_count > 0 and failure_count > 0
    
    # Optimize: use model_dump() if available (Pydantic v2), fallback to dict()
    _dict_method = getattr(responses[0], 'model_dump', None) if responses else None
    if _dict_method:
        responses_data = [r.model_dump() for r in responses]
    else:
        responses_data = [r.dict() for r in responses]
    
    return {
        "request_id": request_id,
        "prompt": request.prompt,
        "strategy": request.strategy,
        "responses": responses_data,
        "aggregated_response": aggregated_response,
        "total_tokens": total_tokens,
        "total_latency_ms": round(total_latency_ms, 2),
        "cache_hit": False,
        "timestamp": datetime.now().isoformat(),
        "success_count": success_count,
        "failure_count": failure_count,
        "partial_success": partial_success
    }


def get_model_types_str(enabled_models: List[ModelConfig]) -> str:
    """Get sorted model types string for cache key - optimized"""
    # Optimized: extract values first, then sort (faster than sorting objects)
    values = [m.model_type.value for m in enabled_models]
    values.sort()  # In-place sort is faster
    return ",".join(values)


def get_weights_map(enabled_models: List[ModelConfig]) -> Dict[str, float]:
    """
    Get weights map from enabled models
    
    Args:
        enabled_models: List of enabled model configurations
        
    Returns:
        Dictionary mapping model_type to multiplier (weight)
    """
    return {m.model_type.value: float(m.multiplier) for m in enabled_models}


def get_client_identifier(request: Request, x_api_key: Optional[str] = Header(None)) -> str:
    """Get client identifier for rate limiting"""
    if x_api_key:
        return f"api_key:{x_api_key}"
    client_host = request.client.host if request.client else "unknown"
    return f"ip:{client_host}"

