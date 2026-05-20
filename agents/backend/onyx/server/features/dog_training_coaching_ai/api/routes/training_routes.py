"""
Dog Training Coaching API Routes
=================================
"""

from fastapi import APIRouter, Depends, Request
from typing import Dict, Any
from ...services.coaching_service import DogTrainingCoach
from ...utils.rate_limiter import limiter
from ...api.helpers import handle_api_errors, validate_request_fields
from ...schemas import (
    CoachingRequest,
    CoachingResponse,
    TrainingPlanRequest,
    TrainingPlanResponse,
    BehaviorAnalysisRequest,
    BehaviorAnalysisResponse,
    ChatRequest,
    ChatResponse,
    TrainingProgressRequest,
    TrainingProgressResponse,
    TrainingAssessmentRequest,
    TrainingAssessmentResponse,
    TrainingResourceRequest,
    TrainingResourceResponse,
    TrainingTrendAnalysisRequest,
    TrainingTrendAnalysisResponse
)
from ...api.dependencies import get_coaching_service

router = APIRouter(prefix="/api/v1", tags=["dog-training"])


@router.post("/coach", response_model=CoachingResponse)
@limiter.limit("10/minute")
@handle_api_errors
async def get_coaching(
    request: Request,
    coaching_request: CoachingRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> CoachingResponse:
    """
    Obtener consejos personalizados de adiestramiento.
    
    Proporciona consejos expertos sobre preguntas de adiestramiento usando IA.
    """
    validate_request_fields(
        dog_breed=coaching_request.dog_breed,
        dog_age=coaching_request.dog_age,
        dog_size=coaching_request.dog_size,
        experience_level=coaching_request.experience_level
    )
    
    result = await service.get_coaching_advice(
        question=coaching_request.question,
        dog_breed=coaching_request.dog_breed,
        dog_age=coaching_request.dog_age,
        dog_size=coaching_request.dog_size,
        training_goal=coaching_request.training_goal,
        experience_level=coaching_request.experience_level,
        previous_context=coaching_request.previous_context,
        specific_issues=coaching_request.specific_issues
    )
    return CoachingResponse(**result)


@router.post("/training-plan", response_model=TrainingPlanResponse)
@limiter.limit("5/minute")
@handle_api_errors
async def create_training_plan(
    request: Request,
    plan_request: TrainingPlanRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> TrainingPlanResponse:
    """
    Crear un plan de entrenamiento personalizado.
    
    Genera un plan completo basado en raza, edad y objetivos del perro.
    """
    validate_request_fields(
        dog_breed=plan_request.dog_breed,
        dog_age=plan_request.dog_age,
        dog_size=plan_request.dog_size,
        training_goals=plan_request.training_goals,
        experience_level=plan_request.experience_level
    )
    
    result = await service.create_training_plan(
        dog_breed=plan_request.dog_breed,
        dog_age=plan_request.dog_age,
        dog_size=plan_request.dog_size,
        training_goals=plan_request.training_goals,
        time_available=plan_request.time_available,
        experience_level=plan_request.experience_level,
        current_issues=plan_request.current_issues,
        preferred_methods=plan_request.preferred_methods
    )
    return TrainingPlanResponse(**result)


@router.post("/analyze-behavior", response_model=BehaviorAnalysisResponse)
@limiter.limit("10/minute")
@handle_api_errors
async def analyze_behavior(
    request: Request,
    behavior_request: BehaviorAnalysisRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> BehaviorAnalysisResponse:
    """
    Analizar comportamiento del perro.
    
    Proporciona análisis experto sobre comportamientos específicos y recomendaciones.
    """
    validate_request_fields(
        dog_breed=behavior_request.dog_breed,
        dog_age=behavior_request.dog_age
    )
    
    result = await service.analyze_behavior(
        behavior_description=behavior_request.behavior_description,
        dog_breed=behavior_request.dog_breed,
        dog_age=behavior_request.dog_age,
        frequency=behavior_request.frequency,
        triggers=behavior_request.triggers,
        context=behavior_request.context
    )
    return BehaviorAnalysisResponse(**result)


@router.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
@handle_api_errors
async def chat(
    request: Request,
    chat_request: ChatRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> ChatResponse:
    """
    Chat conversacional con el asistente de adiestramiento.
    
    Permite conversaciones naturales sobre adiestramiento de perros.
    """
    result = await service.chat(
        message=chat_request.message,
        conversation_history=chat_request.conversation_history,
        dog_info=chat_request.dog_info
    )
    return ChatResponse(**result)


@router.post("/training-progress", response_model=TrainingProgressResponse)
@handle_api_errors
async def track_training_progress(
    request: TrainingProgressRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> TrainingProgressResponse:
    """
    Seguimiento del progreso de entrenamiento.
    
    Analiza el progreso del perro basado en sesiones de entrenamiento.
    """
    result = await service.track_training_progress(
        dog_id=request.dog_id,
        training_sessions=request.training_sessions,
        current_skills=request.current_skills,
        training_goals=request.training_goals,
        challenges_faced=request.challenges_faced,
        time_period_days=request.time_period_days
    )
    return TrainingProgressResponse(**result)


@router.post("/training-assessment", response_model=TrainingAssessmentResponse)
@handle_api_errors
async def assess_training(
    request: TrainingAssessmentRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> TrainingAssessmentResponse:
    """
    Evaluación completa del entrenamiento.
    
    Evalúa las habilidades actuales y proporciona recomendaciones.
    """
    result = await service.assess_training(
        dog_breed=request.dog_breed,
        dog_age=request.dog_age,
        current_skills=request.current_skills,
        training_goals=request.training_goals,
        training_duration_weeks=request.training_duration_weeks,
        behavior_issues=request.behavior_issues,
        owner_experience=request.owner_experience
    )
    return TrainingAssessmentResponse(**result)


@router.post("/training-resources", response_model=TrainingResourceResponse)
@handle_api_errors
async def get_training_resources(
    request: TrainingResourceRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
):
    """
    Obtener recursos educativos de entrenamiento.
    
    Proporciona recursos, ejercicios y guías personalizadas.
    """
    result = await service.get_training_resources(
        topic=request.topic,
        level=request.level,
        format_preference=request.format_preference,
        dog_breed=request.dog_breed,
        specific_need=request.specific_need
    )
    return TrainingResourceResponse(**result)


@router.post("/training-trends", response_model=TrainingTrendAnalysisResponse)
@handle_api_errors
async def analyze_training_trends(
    request: TrainingTrendAnalysisRequest,
    service: DogTrainingCoach = Depends(get_coaching_service)
) -> TrainingTrendAnalysisResponse:
    """
    Análisis de tendencias de entrenamiento.
    
    Analiza patrones y tendencias en el progreso del entrenamiento.
    """
    result = await service.analyze_training_trends(
        training_sessions=request.training_sessions,
        time_period_days=request.time_period_days,
        metrics_to_analyze=request.metrics_to_analyze
    )
    return TrainingTrendAnalysisResponse(**result)


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    from ...config.app_config import get_config
    from ...infrastructure.openrouter import OpenRouterClient
    
    config = get_config()
    
    # Check OpenRouter health
    openrouter_health = {"status": "unknown"}
    if config.openrouter_api_key:
        try:
            client = OpenRouterClient()
            openrouter_health = await client.health_check()
        except Exception as e:
            openrouter_health = {"status": "unhealthy", "error": str(e)}
    
    return {
        "status": "healthy",
        "service": config.app_name,
        "version": config.app_version,
        "openrouter": openrouter_health,
        "openrouter_configured": bool(config.openrouter_api_key)
    }

