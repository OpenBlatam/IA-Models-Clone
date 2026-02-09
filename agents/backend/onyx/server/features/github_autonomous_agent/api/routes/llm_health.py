"""
Health Check Routes para LLM Service.

Endpoints para verificar el estado y salud del servicio LLM.
"""

from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, Optional
from datetime import datetime

from api.utils import handle_api_errors
from config.di_setup import get_service
from config.logging_config import get_logger
from core.services.llm_service import LLMService
from core.services.llm import (
    get_ab_testing_framework,
    get_webhook_service,
    get_prompt_versioning,
    get_llm_testing_framework,
    get_semantic_cache,
    get_advanced_rate_limiter
)

logger = get_logger(__name__)
router = APIRouter(
    prefix="/llm",
    tags=["LLM Health"],
    responses={
        503: {"description": "Service unavailable"}
    }
)


def get_llm_service() -> Optional[LLMService]:
    """Obtener servicio LLM del DI container."""
    try:
        return get_service("llm_service")
    except (ValueError, Exception):
        return None


@router.get("/health")
@handle_api_errors
async def health_check() -> Dict[str, Any]:
    """
    Health check básico del servicio LLM.
    
    Returns:
        Estado de salud del servicio
    """
    llm_service = get_llm_service()
    
    return {
        "status": "healthy" if llm_service else "unavailable",
        "service": "llm",
        "timestamp": datetime.now().isoformat(),
        "available": llm_service is not None
    }


@router.get("/health/detailed")
@handle_api_errors
async def detailed_health_check() -> Dict[str, Any]:
    """
    Health check detallado del servicio LLM.
    
    Verifica:
    - Disponibilidad del servicio
    - Estado de componentes
    - Conectividad con OpenRouter
    - Estado de cache y rate limiting
    """
    llm_service = get_llm_service()
    
    health_status = {
        "status": "healthy",
        "service": "llm",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }
    
    # Verificar servicio principal
    if not llm_service:
        health_status["status"] = "unhealthy"
        health_status["components"]["llm_service"] = {
            "status": "unavailable",
            "error": "Service not initialized"
        }
        return health_status
    
    health_status["components"]["llm_service"] = {
        "status": "available",
        "circuit_breaker": llm_service.circuit_breaker.state,
        "cache_enabled": llm_service.enable_cache,
        "default_models": llm_service.default_models
    }
    
    # Verificar componentes modulares
    try:
        ab_framework = get_ab_testing_framework()
        health_status["components"]["ab_testing"] = {
            "status": "available",
            "tests_count": len(ab_framework.list_tests())
        }
    except Exception as e:
        health_status["components"]["ab_testing"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        webhook_service = get_webhook_service()
        webhooks = webhook_service.list_webhooks()
        health_status["components"]["webhooks"] = {
            "status": "available",
            "registered_count": len(webhooks)
        }
    except Exception as e:
        health_status["components"]["webhooks"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        prompt_versioning = get_prompt_versioning()
        prompts = prompt_versioning.list_prompts()
        health_status["components"]["prompt_versioning"] = {
            "status": "available",
            "prompts_count": len(prompts)
        }
    except Exception as e:
        health_status["components"]["prompt_versioning"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        testing_framework = get_llm_testing_framework()
        suites = testing_framework.list_test_suites()
        health_status["components"]["testing_framework"] = {
            "status": "available",
            "suites_count": len(suites)
        }
    except Exception as e:
        health_status["components"]["testing_framework"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        semantic_cache = get_semantic_cache()
        cache_stats = semantic_cache.get_stats()
        health_status["components"]["semantic_cache"] = {
            "status": "available",
            "items": cache_stats.get("total_items", 0),
            "usage_percent": cache_stats.get("usage_percent", 0)
        }
    except Exception as e:
        health_status["components"]["semantic_cache"] = {
            "status": "error",
            "error": str(e)
        }
    
    try:
        rate_limiter = get_advanced_rate_limiter()
        rate_stats = rate_limiter.get_stats()
        health_status["components"]["rate_limiter"] = {
            "status": "available",
            "configured_limits": len(rate_stats)
        }
    except Exception as e:
        health_status["components"]["rate_limiter"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Verificar conectividad con OpenRouter
    try:
        models = await llm_service.get_available_models()
        health_status["components"]["openrouter"] = {
            "status": "connected",
            "models_available": len(models) if models else 0
        }
    except Exception as e:
        health_status["components"]["openrouter"] = {
            "status": "disconnected",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Determinar estado general
    component_statuses = [
        comp.get("status") for comp in health_status["components"].values()
    ]
    if "error" in component_statuses:
        health_status["status"] = "degraded"
    elif "unavailable" in component_statuses:
        health_status["status"] = "unhealthy"
    
    return health_status


@router.get("/health/readiness")
@handle_api_errors
async def readiness_check() -> Dict[str, Any]:
    """
    Readiness check - verifica si el servicio está listo para recibir tráfico.
    
    Returns:
        Estado de readiness
    """
    llm_service = get_llm_service()
    
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service no está listo"
        )
    
    # Verificar circuit breaker
    if llm_service.circuit_breaker.state == "open":
        raise HTTPException(
            status_code=503,
            detail="Circuit breaker abierto"
        )
    
    return {
        "status": "ready",
        "service": "llm",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/health/liveness")
@handle_api_errors
async def liveness_check() -> Dict[str, Any]:
    """
    Liveness check - verifica si el servicio está vivo.
    
    Returns:
        Estado de liveness
    """
    return {
        "status": "alive",
        "service": "llm",
        "timestamp": datetime.now().isoformat()
    }

