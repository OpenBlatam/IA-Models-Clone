"""
LLM Routes - Rutas para interactuar con modelos de IA vía OpenRouter.
"""

import json
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from api.utils import handle_api_errors
from config.di_setup import get_service
from config.logging_config import get_logger
from core.services.llm_service import LLMService, LLMResponse
from core.constants import ErrorMessages

logger = get_logger(__name__)
from core.services.llm import (
    get_ab_testing_framework,
    get_webhook_service,
    get_prompt_versioning,
    get_llm_testing_framework,
    get_semantic_cache,
    get_advanced_rate_limiter,
    get_model_validator,
    ModelCapability,
    WebhookEvent
)

router = APIRouter(
    prefix="/llm",
    tags=["LLM Service"],
    responses={
        404: {"description": "Not found"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "Service unavailable"}
    }
)


class LLMGenerateRequest(BaseModel):
    """Request para generar respuesta de un modelo."""
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Prompt del usuario"
    )
    model: Optional[str] = Field(
        None,
        min_length=1,
        max_length=200,
        description="Modelo a usar (opcional, usa default si no se proporciona)"
    )
    system_prompt: Optional[str] = Field(
        None,
        max_length=10000,
        description="Prompt del sistema (opcional)"
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Temperatura para sampling (0.0-2.0)"
    )
    max_tokens: Optional[int] = Field(
        None,
        ge=1,
        le=8000,
        description="Máximo de tokens a generar (1-8000)"
    )
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "prompt": "Explica qué es Python",
                "model": "openai/gpt-4",
                "system_prompt": "Eres un asistente útil",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }


class LLMGenerateParallelRequest(BaseModel):
    """Request para generar respuestas de múltiples modelos en paralelo."""
    prompt: str = Field(..., description="Prompt del usuario")
    models: Optional[List[str]] = Field(None, description="Lista de modelos a usar (opcional, usa defaults si no se proporciona)")
    system_prompt: Optional[str] = Field(None, description="Prompt del sistema (opcional)")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Temperatura para sampling")
    max_tokens: Optional[int] = Field(None, ge=1, le=8000, description="Máximo de tokens a generar")


class LLMResponseModel(BaseModel):
    """Modelo de respuesta LLM."""
    model: str
    content: str
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    latency_ms: Optional[float] = None
    cached: bool = False
    retry_count: int = 0


class LLMGenerateResponse(BaseModel):
    """Respuesta de generación LLM."""
    success: bool
    response: Optional[LLMResponseModel] = None
    error: Optional[str] = None


class LLMGenerateParallelResponse(BaseModel):
    """Respuesta de generación paralela LLM."""
    success: bool
    responses: Dict[str, LLMResponseModel] = Field(default_factory=dict)
    total_models: int = 0
    successful_models: int = 0
    error: Optional[str] = None


def get_llm_service() -> Optional[LLMService]:
    """
    Obtener servicio LLM del DI container con mejor manejo de errores.
    
    Returns:
        Instancia de LLMService o None si no está disponible
    """
    try:
        service = get_service("llm_service")
        if service is None:
            logger.debug("LLM service está registrado pero retorna None")
        return service
    except ValueError as e:
        logger.debug(f"LLM service no encontrado en DI container: {e}")
        return None
    except Exception as e:
        logger.warning(f"Error al obtener LLM service: {e}", exc_info=True)
        return None


@router.post("/generate", response_model=LLMGenerateResponse)
@handle_api_errors
async def generate_llm_response(
    request: LLMGenerateRequest,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Generar respuesta de un modelo LLM con validaciones mejoradas.
    
    Usa OpenRouter para acceder a múltiples modelos de IA.
    
    Args:
        request: Request con prompt y parámetros de generación
        llm_service: Servicio LLM (inyectado)
        
    Returns:
        LLMGenerateResponse: Respuesta del modelo
        
    Raises:
        HTTPException: Si el servicio no está disponible o hay un error
    """
    if not llm_service:
        logger.warning("Intento de usar LLM service cuando no está disponible")
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    # Validación de prompt
    prompt_clean = request.prompt.strip() if request.prompt else ""
    if not prompt_clean:
        logger.warning("Intento de generar con prompt vacío")
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.LLM_PROMPT_EMPTY
        )
    
    # Validación de longitud de prompt
    if len(prompt_clean) > 50000:
        logger.warning(f"Prompt demasiado largo: {len(prompt_clean)} caracteres")
        raise HTTPException(
            status_code=400,
            detail="El prompt excede el límite máximo de 50000 caracteres"
        )
    
    # Validar modelo si se proporciona
    if request.model:
        validator = get_model_validator()
        if not validator.available_models:
            await validator.fetch_available_models(llm_service)
        
        is_valid, error_msg = validator.validate_model(request.model)
        if not is_valid:
            logger.warning(f"Modelo inválido: {request.model} - {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=error_msg or ErrorMessages.LLM_MODEL_NOT_FOUND
            )
    
    logger.info(f"Generando respuesta LLM con modelo: {request.model or 'default'}")
    
    try:
        # Validar rate limiting
        rate_limiter = get_advanced_rate_limiter()
        rate_limit_key = f"llm:{request.model or 'default'}"
        rate_info = rate_limiter.is_allowed(rate_limit_key, tokens=1)
        
        if not rate_info.allowed:
            logger.warning(
                f"Rate limit excedido para {rate_limit_key}. "
                f"Retry después de {rate_info.retry_after:.1f}s"
            )
            raise HTTPException(
                status_code=429,
                detail=ErrorMessages.LLM_RATE_LIMIT_EXCEEDED,
                headers={"Retry-After": str(int(rate_info.retry_after))}
            )
        
        response = await llm_service.generate(
            prompt=prompt_clean,
            model=request.model,
            system_prompt=request.system_prompt.strip() if request.system_prompt else None,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        if response.error:
            logger.warning(f"Error en respuesta LLM: {response.error}")
            return LLMGenerateResponse(
                success=False,
                error=response.error
            )
        
        logger.info(
            f"Respuesta LLM generada exitosamente: "
            f"model={response.model}, "
            f"latency={response.latency_ms:.1f}ms, "
            f"cached={response.cached}, "
            f"tokens={response.usage.get('total_tokens', 0) if response.usage else 0}"
        )
        
        return LLMGenerateResponse(
            success=True,
            response=LLMResponseModel(
                model=response.model,
                content=response.content,
                usage=response.usage,
                finish_reason=response.finish_reason,
                latency_ms=response.latency_ms,
                cached=response.cached,
                retry_count=response.retry_count
            )
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar respuesta LLM: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.LLM_GENERATION_FAILED
        )


@router.post("/generate-parallel", response_model=LLMGenerateParallelResponse)
@handle_api_errors
async def generate_llm_responses_parallel(
    request: LLMGenerateParallelRequest,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Generar respuestas de múltiples modelos LLM en paralelo.
    
    Ejecuta el mismo prompt en varios modelos simultáneamente y retorna todas las respuestas.
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    # Validación de prompt
    prompt_clean = request.prompt.strip() if request.prompt else ""
    if not prompt_clean:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.LLM_PROMPT_EMPTY
        )
    
    # Validar número de modelos
    if request.models and len(request.models) > 10:
        raise HTTPException(
            status_code=400,
            detail="Máximo 10 modelos permitidos para generación paralela"
        )
    
    try:
        responses = await llm_service.generate_parallel(
            prompt=prompt_clean,
            models=request.models,
            system_prompt=request.system_prompt.strip() if request.system_prompt else None,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )
        
        # Convertir respuestas a modelo
        response_models = {}
        successful = 0
        
        for model, response in responses.items():
            if not response.error:
                successful += 1
            
            response_models[model] = LLMResponseModel(
                model=response.model,
                content=response.content,
                usage=response.usage,
                finish_reason=response.finish_reason,
                error=response.error,
                latency_ms=response.latency_ms,
                cached=response.cached,
                retry_count=response.retry_count
            )
        
        return LLMGenerateParallelResponse(
            success=True,
            responses=response_models,
            total_models=len(responses),
            successful_models=successful
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error al generar respuestas paralelas: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=ErrorMessages.LLM_GENERATION_FAILED
        )


@router.get("/models")
@handle_api_errors
async def get_available_models(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener lista de modelos disponibles en OpenRouter.
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service no disponible. Verifica que OPENROUTER_API_KEY esté configurada."
        )
    
    try:
        models = await llm_service.get_available_models()
        return {
            "success": True,
            "models": models,
            "total": len(models)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al obtener modelos: {str(e)}"
        )


@router.post("/generate-stream")
@handle_api_errors
async def generate_llm_stream(
    request: LLMGenerateRequest,
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Generar respuesta en modo streaming.
    
    Retorna chunks de texto conforme se generan.
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail=ErrorMessages.LLM_SERVICE_UNAVAILABLE
        )
    
    # Validación de prompt
    prompt_clean = request.prompt.strip() if request.prompt else ""
    if not prompt_clean:
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.LLM_PROMPT_EMPTY
        )
    
    async def generate():
        try:
            async for chunk in llm_service.generate_stream(
                prompt=prompt_clean,
                model=request.model,
                system_prompt=request.system_prompt.strip() if request.system_prompt else None,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error durante streaming: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': ErrorMessages.LLM_STREAMING_ERROR})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/stats")
@handle_api_errors
async def get_llm_stats(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener estadísticas del servicio LLM.
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service no disponible"
        )
    
    return llm_service.get_stats()


@router.post("/stats/reset")
@handle_api_errors
async def reset_llm_stats(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Resetear estadísticas del servicio LLM.
    """
    if not llm_service:
        raise HTTPException(
            status_code=503,
            detail="LLM service no disponible"
        )
    
    llm_service.reset_stats()
    return {"message": "Estadísticas reseteadas correctamente"}


@router.get("/status")
@handle_api_errors
async def get_llm_status(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener estado del servicio LLM.
    """
    if not llm_service:
        return {
            "enabled": False,
            "message": "LLM service no disponible"
        }
    
    stats = llm_service.get_stats()
    
    return {
        "enabled": True,
        "default_models": llm_service.default_models,
        "max_parallel_requests": llm_service.max_parallel_requests,
        "timeout": llm_service.timeout,
        "cache_enabled": llm_service.enable_cache,
        "circuit_breaker_state": llm_service.circuit_breaker.state,
        "stats_summary": {
            "total_requests": stats.get("total_requests", 0),
            "successful_requests": stats.get("successful_requests", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0),
            "average_latency_ms": stats.get("average_latency_ms", 0.0)
        }
    }


# ==================== Dashboard & Analytics ====================

@router.get("/dashboard/stats")
@handle_api_errors
async def get_dashboard_stats(
    llm_service: Optional[LLMService] = Depends(get_llm_service)
):
    """
    Obtener estadísticas completas para el dashboard.
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service no disponible")
    
    stats = llm_service.get_stats()
    cost_stats = llm_service.get_cost_stats()
    performance = llm_service.get_performance_report()
    
    return {
        "llm_stats": stats,
        "cost_stats": cost_stats,
        "performance": performance,
        "cache_stats": get_semantic_cache().get_stats() if hasattr(get_semantic_cache, "_instances") else {},
        "rate_limit_stats": get_advanced_rate_limiter().get_stats()
    }


@router.get("/dashboard/analytics")
@handle_api_errors
async def get_analytics(
    llm_service: Optional[LLMService] = Depends(get_llm_service),
    days: int = 7
):
    """
    Obtener analytics detallados.
    
    Args:
        days: Número de días de datos a incluir
    """
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service no disponible")
    
    # Aquí se podrían agregar queries a una base de datos de analytics
    # Por ahora, retornamos estadísticas actuales
    stats = llm_service.get_stats()
    
    return {
        "period_days": days,
        "summary": {
            "total_requests": stats.get("total_requests", 0),
            "success_rate": stats.get("success_rate", 0.0),
            "avg_latency_ms": stats.get("average_latency_ms", 0.0),
            "total_tokens": stats.get("total_tokens_used", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0.0)
        },
        "by_model": stats.get("requests_by_model", {}),
        "errors": stats.get("errors_by_type", {})
    }


# ==================== A/B Testing ====================

@router.post("/ab-test/create")
@handle_api_errors
async def create_ab_test(request: Dict[str, Any]):
    """
    Crear un nuevo A/B test.
    
    Args:
        request: Dict con name, description, variants, prompt, system_prompt (opcional),
                 evaluation_criteria (opcional), min_samples, max_samples
                 
    Returns:
        Dict con test_id y mensaje de éxito
    """
    framework = get_ab_testing_framework()
    
    # Validaciones
    if not request.get("name"):
        raise HTTPException(status_code=400, detail="Nombre es requerido")
    
    if not request.get("prompt"):
        raise HTTPException(status_code=400, detail=ErrorMessages.LLM_PROMPT_EMPTY)
    
    variants_data = request.get("variants", [])
    if len(variants_data) < 2:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 2 variantes para un A/B test"
        )
    
    try:
        from core.services.llm import Variant, VariantType
        
        variants = []
        for v in variants_data:
            if not v.get("name"):
                raise HTTPException(
                    status_code=400,
                    detail="Cada variante debe tener un nombre"
                )
            try:
                variants.append(
                    Variant(
                        name=v["name"],
                        variant_type=VariantType(v["variant_type"]),
                        config=v["config"],
                        weight=v.get("weight", 1.0)
                    )
                )
            except (ValueError, KeyError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Variante inválida: {str(e)}"
                )
        
        test_id = framework.create_test(
            name=request["name"],
            description=request.get("description", ""),
            variants=variants,
            prompt=request["prompt"],
            system_prompt=request.get("system_prompt"),
            evaluation_criteria=request.get("evaluation_criteria", []),
            min_samples=request.get("min_samples", 10),
            max_samples=request.get("max_samples", 100)
        )
        
        logger.info(f"A/B test creado exitosamente: {test_id}")
        return {"test_id": test_id, "message": "A/B test creado exitosamente"}
    except ValueError as e:
        logger.warning(f"Error al crear A/B test: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/ab-test/{test_id}")
@handle_api_errors
async def get_ab_test(test_id: str):
    """Obtener información de un A/B test."""
    framework = get_ab_testing_framework()
    test = framework.get_test(test_id)
    
    if not test:
        logger.warning(f"A/B test no encontrado: {test_id}")
        raise HTTPException(
            status_code=404,
            detail=ErrorMessages.LLM_TEST_NOT_FOUND
        )
    
    summary = framework.get_summary(test_id)
    
    return {
        "test": test.to_dict(),
        "summary": summary.to_dict() if summary else None
    }


@router.get("/ab-test")
@handle_api_errors
async def list_ab_tests(status: Optional[str] = None):
    """Listar todos los A/B tests."""
    framework = get_ab_testing_framework()
    tests = framework.list_tests(status=status)
    
    return {
        "tests": [t.to_dict() for t in tests],
        "total": len(tests)
    }


# ==================== Webhooks ====================

@router.post("/webhooks/register")
@handle_api_errors
async def register_webhook(request: Dict[str, Any]):
    """
    Registrar un nuevo webhook.
    
    Args:
        request: Dict con url, events, secret (opcional), timeout, retry_count, headers
        
    Returns:
        Dict con webhook_id y mensaje de éxito
    """
    service = get_webhook_service()
    
    # Validaciones
    if not request.get("url"):
        raise HTTPException(
            status_code=400,
            detail="URL es requerida"
        )
    
    if not request.get("events"):
        raise HTTPException(
            status_code=400,
            detail="Al menos un evento es requerido"
        )
    
    try:
        events = [WebhookEvent(e) for e in request.get("events", [])]
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Evento inválido: {str(e)}"
        )
    
    try:
        webhook_id = service.register_webhook(
            url=request["url"],
            events=events,
            secret=request.get("secret"),
            timeout=request.get("timeout", 5),
            retry_count=request.get("retry_count", 3),
            headers=request.get("headers")
        )
        
        logger.info(f"Webhook registrado exitosamente: {webhook_id}")
        return {"webhook_id": webhook_id, "message": "Webhook registrado exitosamente"}
    except ValueError as e:
        logger.warning(f"Error al registrar webhook: {e}")
        raise HTTPException(
            status_code=400,
            detail=ErrorMessages.LLM_INVALID_WEBHOOK_URL
        )


@router.get("/webhooks")
@handle_api_errors
async def list_webhooks(event: Optional[str] = None):
    """Listar webhooks registrados."""
    service = get_webhook_service()
    
    event_enum = WebhookEvent(event) if event else None
    webhooks = service.list_webhooks(event=event_enum)
    
    return {
        "webhooks": [w.to_dict() for w in webhooks],
        "total": len(webhooks)
    }


@router.post("/webhooks/{webhook_id}/test")
@handle_api_errors
async def test_webhook(webhook_id: str):
    """Probar un webhook."""
    service = get_webhook_service()
    result = await service.test_webhook(webhook_id)
    
    return result


# ==================== Prompt Versioning ====================

@router.post("/prompts/create")
@handle_api_errors
async def create_prompt(request: Dict[str, Any]):
    """Crear un nuevo prompt con versionado."""
    versioning = get_prompt_versioning()
    
    prompt_id = versioning.create_prompt(
        name=request["name"],
        prompt=request["prompt"],
        system_prompt=request.get("system_prompt"),
        description=request.get("description"),
        version=request.get("version", "1.0.0"),
        metadata=request.get("metadata"),
        created_by=request.get("created_by"),
        tags=request.get("tags", [])
    )
    
    return {"prompt_id": prompt_id, "message": "Prompt creado exitosamente"}


@router.get("/prompts/{prompt_id}")
@handle_api_errors
async def get_prompt(prompt_id: str, version: Optional[str] = None):
    """Obtener un prompt (versión específica o actual)."""
    versioning = get_prompt_versioning()
    prompt = versioning.get_prompt(prompt_id, version=version)
    
    if not prompt:
        logger.warning(f"Prompt no encontrado: {prompt_id} (versión: {version or 'actual'})")
        raise HTTPException(
            status_code=404,
            detail=ErrorMessages.LLM_PROMPT_NOT_FOUND
        )
    
    return prompt.to_dict()


@router.get("/prompts")
@handle_api_errors
async def list_prompts(tags: Optional[str] = None, status: Optional[str] = None):
    """Listar prompts."""
    versioning = get_prompt_versioning()
    
    tags_list = tags.split(",") if tags else None
    status_enum = None
    if status:
        from core.services.llm import PromptStatus
        status_enum = PromptStatus(status)
    
    prompts = versioning.list_prompts(tags=tags_list, status=status_enum)
    
    return {
        "prompts": [p.to_dict() for p in prompts],
        "total": len(prompts)
    }


# ==================== Testing Framework ====================

@router.post("/tests/create-suite")
@handle_api_errors
async def create_test_suite(request: Dict[str, Any]):
    """Crear una nueva suite de tests."""
    framework = get_llm_testing_framework()
    
    from core.services.llm import TestCase, TestAssertion, AssertionType, TestType
    
    test_cases = []
    for tc_data in request.get("test_cases", []):
        assertions = [
            TestAssertion(
                name=a["name"],
                assertion_type=AssertionType(a["assertion_type"]),
                expected=a["expected"],
                custom_function=a.get("custom_function")
            )
            for a in tc_data.get("assertions", [])
        ]
        
        test_cases.append(TestCase(
            case_id=tc_data.get("case_id", ""),
            name=tc_data["name"],
            prompt=tc_data["prompt"],
            system_prompt=tc_data.get("system_prompt"),
            assertions=assertions,
            expected_output=tc_data.get("expected_output"),
            metadata=tc_data.get("metadata", {})
        ))
    
    suite_id = framework.create_test_suite(
        name=request["name"],
        description=request.get("description", ""),
        test_type=TestType(request.get("test_type", "functional")),
        test_cases=test_cases,
        model=request.get("model"),
        metadata=request.get("metadata", {})
    )
    
    return {"suite_id": suite_id, "message": "Test suite creada exitosamente"}


@router.post("/tests/{suite_id}/run")
@handle_api_errors
async def run_test_suite(
    suite_id: str,
    llm_service: Optional[LLMService] = Depends(get_llm_service),
    model: Optional[str] = None
):
    """Ejecutar una suite de tests."""
    if not llm_service:
        raise HTTPException(status_code=503, detail="LLM service no disponible")
    
    framework = get_llm_testing_framework()
    result = await framework.run_test_suite(suite_id, llm_service, model=model)
    
    if not result:
        logger.warning(f"Test suite no encontrada: {suite_id}")
        raise HTTPException(
            status_code=404,
            detail=ErrorMessages.LLM_TEST_SUITE_NOT_FOUND
        )
    
    return result.to_dict()


@router.get("/tests/{suite_id}/results")
@handle_api_errors
async def get_test_results(suite_id: str, limit: Optional[int] = None):
    """Obtener resultados de una suite de tests."""
    framework = get_llm_testing_framework()
    results = framework.get_test_results(suite_id, limit=limit)
    
    return {
        "results": [r.to_dict() for r in results],
        "total": len(results)
    }

