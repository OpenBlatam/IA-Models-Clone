"""
Burnout Prevention API Routes
=============================
"""

from fastapi import APIRouter, HTTPException, Depends, status
from typing import TypeVar

T = TypeVar('T')

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

from ...schemas import (
    BurnoutAssessmentRequest,
    BurnoutAssessmentResponse,
    WellnessCheckRequest,
    WellnessCheckResponse,
    CopingStrategyRequest,
    CopingStrategyResponse,
    ChatRequest,
    ChatResponse,
    ProgressTrackingRequest,
    ProgressTrackingResponse,
    TrendAnalysisRequest,
    TrendAnalysisResponse,
    ResourceRequest,
    ResourceResponse,
    PersonalizedPlanRequest,
    PersonalizedPlanResponse,
    ContinuousProcessorStatusResponse,
    ContinuousProcessorControlRequest,
    ContinuousProcessorControlResponse
)
from ...services import BurnoutPreventionService
from ...services.continuous_burnout_service import get_continuous_service
from ...infrastructure.openrouter import OpenRouterClient
from ...core.constants import MIN_INTERVAL_SECONDS, MAX_ERROR_MESSAGE_LENGTH

router = APIRouter(prefix="/api/v1", tags=["burnout"])


# Singleton para cliente OpenRouter (reutiliza conexiones HTTP)
_openrouter_client_instance: Optional[OpenRouterClient] = None


def get_openrouter_client() -> OpenRouterClient:
    """
    Dependency para obtener cliente OpenRouter (singleton).
    
    Reutiliza la misma instancia para mejor gestión de conexiones HTTP.
    """
    global _openrouter_client_instance
    if _openrouter_client_instance is None:
        _openrouter_client_instance = OpenRouterClient()
    return _openrouter_client_instance


def cleanup_openrouter_client() -> None:
    """
    Cleanup function for OpenRouter client.
    
    Can be called during shutdown to properly close HTTP connections.
    """
    global _openrouter_client_instance
    _openrouter_client_instance = None


def get_burnout_service(
    client: OpenRouterClient = Depends(get_openrouter_client)
) -> BurnoutPreventionService:
    """
    Dependency para obtener servicio de burnout.
    
    Crea una nueva instancia del servicio por request para thread-safety.
    """
    return BurnoutPreventionService(client)


async def handle_service_error(
    coro: Coroutine[Any, Any, T],
    error_message: str
) -> T:
    """
    Handle service errors consistently (optimized).
    
    Args:
        coro: Coroutine to execute
        error_message: Base error message for unexpected errors
        
    Returns:
        Result from coroutine
        
    Raises:
        HTTPException for various error types
    """
    from ...core.exceptions import ValidationError, APIError, BurnoutPreventionError
    from ...core.logging_helpers import log_error, log_warning, truncate_error_message
    
    try:
        return await coro
    except ValidationError as e:
        error_msg = truncate_error_message(e)
        log_warning("Validation error", context={"error": error_msg})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except ValueError as e:
        error_msg = truncate_error_message(e)
        log_warning("Validation error", context={"error": error_msg})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=error_msg)
    except APIError as e:
        error_msg = truncate_error_message(e)
        log_error("API error", e, context={"error_message": error_message})
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External API error: {error_msg}"
        )
    except HTTPException:
        raise
    except BurnoutPreventionError as e:
        error_msg = truncate_error_message(e)
        log_error("Service error", e, context={"error_message": error_message})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{error_message}: {error_msg}"
        )
    except Exception as e:
        error_msg = truncate_error_message(e)
        log_error(
            "Unexpected error",
            e,
            context={
                "error_message": error_message,
                "error_type": type(e).__name__
            }
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{error_message}: {error_msg}"
        )


@router.post("/assess", response_model=BurnoutAssessmentResponse)
async def assess_burnout(
    request: BurnoutAssessmentRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Evaluar riesgo de burnout basado en múltiples factores.
    
    Analiza el riesgo de burnout considerando:
    - Horas trabajadas por semana
    - Nivel de estrés (1-10)
    - Horas de sueño
    - Satisfacción laboral
    - Síntomas físicos y emocionales
    - Ambiente laboral
    
    Returns:
        BurnoutAssessmentResponse con nivel de riesgo, score, factores de riesgo,
        recomendaciones, acciones inmediatas y estrategias a largo plazo
    """
    return await handle_service_error(
        service.assess_burnout(request),
        "Error assessing burnout"
    )


@router.post("/wellness-check", response_model=WellnessCheckResponse)
async def wellness_check(
    request: WellnessCheckRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Realizar chequeo de bienestar general.
    
    Evalúa el estado de bienestar considerando:
    - Estado de ánimo actual
    - Nivel de energía
    - Desafíos recientes
    - Sistema de apoyo disponible
    
    Returns:
        WellnessCheckResponse con score de bienestar, análisis de ánimo,
        recomendaciones de apoyo y sugerencias de autocuidado
    """
    return await handle_service_error(
        service.wellness_check(request),
        "Error in wellness check"
    )


@router.post("/coping-strategies", response_model=CopingStrategyResponse)
async def get_coping_strategies(
    request: CopingStrategyRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Obtener estrategias de afrontamiento personalizadas.
    
    Proporciona estrategias adaptadas a:
    - Tipo de estresor
    - Métodos actuales de afrontamiento
    - Tiempo disponible
    - Preferencias personales
    
    Returns:
        CopingStrategyResponse con estrategias, plan de implementación y recursos
    """
    return await handle_service_error(
        service.get_coping_strategies(request),
        "Error getting coping strategies"
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Chat conversacional con el asistente de burnout.
    
    Args:
        request: Chat request con mensaje y historial opcional
        
    Returns:
        ChatResponse con respuesta del asistente
    """
    # Message length is already validated by Pydantic schema
    # No need for additional validation here
    
    return await handle_service_error(
        service.chat(request),
        "Error in chat"
    )


@router.post("/progress", response_model=ProgressTrackingResponse)
async def track_progress(
    request: ProgressTrackingRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Seguimiento de progreso en prevención de burnout.
    
    Analiza el progreso basándose en:
    - Historial de evaluaciones
    - Metas establecidas
    - Estado actual
    
    Returns:
        ProgressTrackingResponse con score de progreso, tendencia, logros,
        próximos pasos e insights detallados
    """
    return await handle_service_error(
        service.track_progress(request),
        "Error tracking progress"
    )


@router.post("/trends", response_model=TrendAnalysisResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Analizar tendencias en evaluaciones de burnout.
    
    Identifica patrones y predicciones basadas en:
    - Múltiples evaluaciones históricas
    - Período de tiempo analizado
    
    Returns:
        TrendAnalysisResponse con tendencia general, métricas clave, patrones,
        predicciones futuras y recomendaciones basadas en tendencias
    """
    return await handle_service_error(
        service.analyze_trends(request),
        "Error analyzing trends"
    )


@router.post("/resources", response_model=ResourceResponse)
async def get_resources(
    request: ResourceRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Obtener recursos educativos personalizados sobre burnout.
    
    Proporciona recursos adaptados a:
    - Tema específico de interés
    - Nivel de conocimiento
    - Formato preferido (artículo, video, podcast, libro)
    
    Returns:
        ResourceResponse con recursos, ruta de aprendizaje, conceptos clave
        y elementos de acción práctica
    """
    return await handle_service_error(
        service.get_resources(request),
        "Error getting resources"
    )


@router.post("/personalized-plan", response_model=PersonalizedPlanResponse)
async def create_personalized_plan(
    request: PersonalizedPlanRequest,
    service: BurnoutPreventionService = Depends(get_burnout_service)
):
    """
    Crear plan personalizado de prevención de burnout.
    
    Genera un plan estructurado considerando:
    - Situación actual del usuario
    - Metas específicas
    - Restricciones y limitaciones
    - Preferencias personales
    
    Returns:
        PersonalizedPlanResponse con nombre del plan, duración, metas semanales,
        acciones diarias, hitos importantes y recursos necesarios
    """
    return await handle_service_error(
        service.create_personalized_plan(request),
        "Error creating personalized plan"
    )


@router.get("/health")
async def health_check(
    client: OpenRouterClient = Depends(get_openrouter_client)
) -> Dict[str, Any]:
    """
    Health check endpoint with detailed status.
    
    Verifica el estado de la API y la conectividad con OpenRouter.
    Incluye estadísticas de cache y estado del servicio.
    
    Returns:
        Dict con status, openrouter health, cache stats y timestamp
    """
    from ...config.app_config import get_config
    from ...core.cache import get_cache_stats
    from ...core.datetime_utils import get_utc_iso_timestamp
    
    config = get_config()
    health_data = {
        "status": "healthy",
        "service": "burnout_prevention_ai",
        "version": config.app_version,
        "timestamp": get_utc_iso_timestamp(),
        "checks": {},
        "cache": get_cache_stats()
    }
    
    try:
        openrouter_health = await client.health_check()
        is_healthy = openrouter_health.get("healthy", False)
        health_data["openrouter"] = openrouter_health
        health_data["checks"]["openrouter"] = is_healthy
        
        if not is_healthy:
            health_data["status"] = "degraded"
    except Exception as e:
        from ...core.logging_helpers import log_error
        error_msg = str(e)[:MAX_ERROR_MESSAGE_LENGTH]
        log_error("Health check failed", e, context={"endpoint": "health"})
        health_data["status"] = "unhealthy"
        health_data["openrouter"] = {"error": error_msg}
        health_data["checks"]["openrouter"] = False
    
    # Add continuous service status if available
    try:
        from ...services.continuous_burnout_service import get_continuous_service
        continuous_service = get_continuous_service()
        service_status = continuous_service.get_status()
        health_data["continuous_service"] = {
            "is_active": service_status.get("is_active", False),
            "pending_assessments": service_status.get("pending_assessments", 0),
            "processed_count": service_status.get("processed_count", 0),
            "status": service_status.get("status", "unknown")
        }
        health_data["checks"]["continuous_service"] = True
    except Exception as e:
        from ...core.logging_helpers import log_warning
        log_warning("Could not get continuous service status", context={"error": str(e)[:100]})
        health_data["checks"]["continuous_service"] = False
    
    return health_data


@router.get("/continuous/status", response_model=ContinuousProcessorStatusResponse)
async def get_continuous_status():
    """
    Obtener estado del procesador continuo.
    
    Retorna información sobre el procesador que ejecuta evaluaciones
    de forma continua en segundo plano.
    
    Returns:
        ContinuousProcessorStatusResponse con estado, estadísticas y métricas
    """
    continuous_service = get_continuous_service()
    status = continuous_service.get_status()
    
    return ContinuousProcessorStatusResponse(
        is_active=status["is_active"],
        status=status["status"],
        interval_seconds=status["interval_seconds"],
        execution_count=status["execution_count"],
        error_count=status["error_count"],
        last_execution=status["last_execution"],
        last_error=status["last_error"],
        uptime_seconds=status["uptime_seconds"],
        start_time=status["start_time"],
    )


@router.post("/continuous/control", response_model=ContinuousProcessorControlResponse)
async def control_continuous_processor(
    request: ContinuousProcessorControlRequest
):
    """
    Controlar el procesador continuo (start/stop/restart).
    
    Permite iniciar, detener o reiniciar el procesador que ejecuta
    evaluaciones de burnout de forma continua.
    
    Args:
        request: ControlRequest con acción (start/stop/restart) y opcionalmente
                 intervalo en segundos (solo para start/restart)
    
    Returns:
        ContinuousProcessorControlResponse con éxito de la operación, mensaje
        y estado actual del procesador
    """
    continuous_service = get_continuous_service()
    
    # Validate action
    valid_actions = {"start", "stop", "restart"}
    if request.action not in valid_actions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid action. Must be one of: {', '.join(valid_actions)}"
        )
    
    # Validate interval if provided
    if request.interval_seconds is not None and request.interval_seconds < MIN_INTERVAL_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interval must be at least {MIN_INTERVAL_SECONDS} seconds"
        )
    
    try:
        if request.action == "start":
            await continuous_service.start(interval_seconds=request.interval_seconds)
            message = "Continuous processing started"
        elif request.action == "stop":
            await continuous_service.stop()
            message = "Continuous processing stopped"
        else:  # restart
            await continuous_service.restart(interval_seconds=request.interval_seconds)
            message = "Continuous processing restarted"
        
        service_status = continuous_service.get_status()
        
        return ContinuousProcessorControlResponse(
            success=True,
            message=message,
            status=service_status
        )
    except Exception as e:
        logger.error("Error controlling continuous processor", error=str(e), exc_info=True)
        service_status = continuous_service.get_status()
        return ContinuousProcessorControlResponse(
            success=False,
            message=f"Error: {str(e)}",
            status=service_status
        )

