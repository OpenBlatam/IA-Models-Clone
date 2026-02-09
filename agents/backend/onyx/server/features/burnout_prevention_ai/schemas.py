"""
Schemas para Burnout Prevention AI
===================================
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from .core.constants import (
    MAX_FIELD_LENGTH_SHORT,
    MAX_FIELD_LENGTH_MEDIUM,
    MAX_FIELD_LENGTH_LONG,
    MAX_FIELD_LENGTH_EXTRA_LONG,
    MAX_LIST_ITEMS,
    MAX_GOAL_LENGTH
)


class BurnoutAssessmentRequest(BaseModel):
    """Request para evaluación de burnout."""
    
    work_hours_per_week: int = Field(..., ge=0, le=168, description="Horas trabajadas por semana")
    stress_level: int = Field(..., ge=1, le=10, description="Nivel de estrés (1-10)")
    sleep_hours_per_night: float = Field(..., ge=0, le=24, description="Horas de sueño por noche")
    work_satisfaction: int = Field(..., ge=1, le=10, description="Satisfacción laboral (1-10)")
    physical_symptoms: List[str] = Field(default_factory=list, description="Síntomas físicos", max_length=20)
    emotional_symptoms: List[str] = Field(default_factory=list, description="Síntomas emocionales", max_length=20)
    work_environment: Optional[str] = Field(None, description="Descripción del ambiente laboral", max_length=500)
    additional_context: Optional[str] = Field(None, description="Contexto adicional", max_length=1000)
    
    @field_validator('physical_symptoms', 'emotional_symptoms')
    @classmethod
    def validate_symptoms(cls, v: List[str]) -> List[str]:
        """Validate and sanitize symptoms list."""
        if not isinstance(v, list):
            return []
        # Limit to MAX_LIST_ITEMS and ensure all are strings
        return [str(item).strip()[:MAX_FIELD_LENGTH_SHORT] for item in v[:MAX_LIST_ITEMS] if item]
    
    @field_validator('work_environment', 'additional_context', mode='before')
    @classmethod
    def validate_string_fields(cls, v: Any) -> Optional[str]:
        """Validate and sanitize string fields."""
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        return v.strip()[:MAX_FIELD_LENGTH_LONG] if v else None


class BurnoutAssessmentResponse(BaseModel):
    """Response de evaluación de burnout."""
    
    burnout_risk_level: str = Field(..., description="Nivel de riesgo: low, medium, high, critical")
    burnout_score: float = Field(..., ge=0, le=100, description="Puntuación de burnout (0-100)")
    risk_factors: List[str] = Field(default_factory=list, description="Factores de riesgo identificados")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones personalizadas")
    immediate_actions: List[str] = Field(default_factory=list, description="Acciones inmediatas")
    long_term_strategies: List[str] = Field(default_factory=list, description="Estrategias a largo plazo")
    assessment_date: datetime = Field(default_factory=datetime.now)


class WellnessCheckRequest(BaseModel):
    """Request para chequeo de bienestar."""
    
    current_mood: str = Field(..., description="Estado de ánimo actual", max_length=200)
    energy_level: int = Field(..., ge=1, le=10, description="Nivel de energía (1-10)")
    recent_challenges: Optional[str] = Field(None, description="Desafíos recientes", max_length=1000)
    support_system: Optional[str] = Field(None, description="Sistema de apoyo disponible", max_length=500)
    
    @field_validator('current_mood')
    @classmethod
    def validate_mood(cls, v: str) -> str:
        """Validate and sanitize mood."""
        if not v or not v.strip():
            raise ValueError("Current mood cannot be empty")
        return v.strip()[:MAX_FIELD_LENGTH_SHORT]


class WellnessCheckResponse(BaseModel):
    """Response de chequeo de bienestar."""
    
    wellness_score: float = Field(..., ge=0, le=100, description="Puntuación de bienestar")
    mood_analysis: str = Field(..., description="Análisis del estado de ánimo")
    support_recommendations: List[str] = Field(default_factory=list, description="Recomendaciones de apoyo")
    self_care_suggestions: List[str] = Field(default_factory=list, description="Sugerencias de autocuidado")
    check_date: datetime = Field(default_factory=datetime.now)


class CopingStrategyRequest(BaseModel):
    """Request para estrategias de afrontamiento."""
    
    stressor_type: str = Field(..., description="Tipo de estresor", max_length=200)
    current_coping_methods: Optional[List[str]] = Field(None, description="Métodos actuales de afrontamiento", max_length=20)
    available_time: Optional[str] = Field(None, description="Tiempo disponible para implementar estrategias", max_length=200)
    preferences: Optional[List[str]] = Field(None, description="Preferencias personales", max_length=20)
    
    @field_validator('stressor_type')
    @classmethod
    def validate_stressor_type(cls, v: str) -> str:
        """Validate and sanitize stressor type."""
        if not v or not v.strip():
            raise ValueError("Stressor type cannot be empty")
        return v.strip()[:MAX_FIELD_LENGTH_SHORT]


class CopingStrategyResponse(BaseModel):
    """Response de estrategias de afrontamiento."""
    
    strategies: List[Dict[str, Any]] = Field(default_factory=list, description="Estrategias recomendadas")
    implementation_plan: List[str] = Field(default_factory=list, description="Plan de implementación")
    resources: List[str] = Field(default_factory=list, description="Recursos adicionales")


class ChatRequest(BaseModel):
    """Request para chat con el asistente."""
    
    message: str = Field(..., description="Mensaje del usuario", max_length=10000)
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Historial de conversación", max_length=50)
    
    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        """Validate and sanitize message."""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()[:MAX_FIELD_LENGTH_EXTRA_LONG]
    
    @field_validator('conversation_history')
    @classmethod
    def validate_conversation_history(cls, v: Optional[List[Dict[str, str]]]) -> Optional[List[Dict[str, str]]]:
        """Validate conversation history structure."""
        if v is None:
            return None
        if not isinstance(v, list):
            raise ValueError("Conversation history must be a list")
        if len(v) > 50:
            raise ValueError("Conversation history too long. Maximum 50 messages.")
        
        valid_roles = {"system", "user", "assistant"}
        for i, msg in enumerate(v):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} must be a dictionary")
            role = msg.get("role")
            content = msg.get("content")
            if not role or not content:
                raise ValueError(f"Message {i} must have 'role' and 'content' fields")
            if role not in valid_roles:
                raise ValueError(f"Message {i} has invalid role: {role}")
            if len(str(content)) > 10000:
                raise ValueError(f"Message {i} content too long. Maximum 10000 characters.")
        
        return v


class ChatResponse(BaseModel):
    """Response del chat."""
    
    response: str = Field(..., description="Respuesta del asistente")
    suggestions: Optional[List[str]] = Field(None, description="Sugerencias de seguimiento")
    resources: Optional[List[str]] = Field(None, description="Recursos relevantes")


class ProgressTrackingRequest(BaseModel):
    """Request para seguimiento de progreso."""
    
    user_id: Optional[str] = Field(None, max_length=200, description="ID del usuario")
    assessment_history: List[Dict[str, Any]] = Field(default_factory=list, description="Historial de evaluaciones")
    goals: Optional[List[str]] = Field(None, description="Metas establecidas")
    current_status: Optional[Dict[str, Any]] = Field(None, description="Estado actual")
    
    @field_validator('assessment_history')
    @classmethod
    def validate_assessment_history(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate assessment history is not empty."""
        if not v:
            raise ValueError("At least one assessment in history is required for progress tracking")
        # Limit to 1000 assessments
        return v[:MAX_FIELD_LENGTH_LONG]
    
    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: Optional[str]) -> Optional[str]:
        """Validate and sanitize user_id."""
        if v is None:
            return None
        return str(v).strip()[:200] if v else None


class ProgressTrackingResponse(BaseModel):
    """Response de seguimiento de progreso."""
    
    progress_score: float = Field(..., ge=0, le=100, description="Puntuación de progreso")
    trend: str = Field(..., description="Tendencia: improving, stable, declining")
    milestones_achieved: List[str] = Field(default_factory=list, description="Logros alcanzados")
    next_steps: List[str] = Field(default_factory=list, description="Próximos pasos recomendados")
    insights: str = Field(..., description="Insights sobre el progreso")
    progress_date: datetime = Field(default_factory=datetime.now)


class TrendAnalysisRequest(BaseModel):
    """Request para análisis de tendencias."""
    
    assessments: List[Dict[str, Any]] = Field(..., min_length=1, description="Historial de evaluaciones")
    time_period_days: Optional[int] = Field(30, ge=1, description="Período de tiempo en días (mínimo 1)")
    
    @field_validator('assessments')
    @classmethod
    def validate_assessments(cls, v: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate assessments list is not empty."""
        if not v:
            raise ValueError("At least one assessment is required for trend analysis")
        # Limit to 1000 assessments to prevent excessive processing
        return v[:MAX_FIELD_LENGTH_LONG]


class TrendAnalysisResponse(BaseModel):
    """Response de análisis de tendencias."""
    
    overall_trend: str = Field(..., description="Tendencia general")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas clave")
    patterns: List[str] = Field(default_factory=list, description="Patrones identificados")
    predictions: Dict[str, Any] = Field(default_factory=dict, description="Predicciones futuras")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones basadas en tendencias")
    analysis_date: datetime = Field(default_factory=datetime.now)


class ResourceRequest(BaseModel):
    """Request para recursos educativos."""
    
    topic: str = Field(..., description="Tema de interés", max_length=200)
    level: Optional[str] = Field("beginner", description="Nivel: beginner, intermediate, advanced")
    format_preference: Optional[str] = Field(None, description="Formato preferido: article, video, podcast, book", max_length=50)
    
    @field_validator('topic')
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Validate and sanitize topic."""
        if not v or not v.strip():
            raise ValueError("Topic cannot be empty")
        return v.strip()[:MAX_FIELD_LENGTH_SHORT]
    
    @field_validator('level')
    @classmethod
    def validate_level(cls, v: Optional[str]) -> str:
        """Validate level."""
        if v is None:
            return "beginner"
        valid_levels = {"beginner", "intermediate", "advanced"}
        if v.lower() not in valid_levels:
            return "beginner"
        return v.lower()


class ResourceResponse(BaseModel):
    """Response de recursos educativos."""
    
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="Recursos recomendados")
    learning_path: List[str] = Field(default_factory=list, description="Ruta de aprendizaje sugerida")
    key_concepts: List[str] = Field(default_factory=list, description="Conceptos clave")
    action_items: List[str] = Field(default_factory=list, description="Elementos de acción")


class PersonalizedPlanRequest(BaseModel):
    """Request para plan personalizado."""
    
    current_situation: Dict[str, Any] = Field(..., description="Situación actual")
    goals: List[str] = Field(..., min_length=1, description="Metas personales")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Restricciones (tiempo, recursos, etc.)")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Preferencias personales")
    
    @field_validator('goals')
    @classmethod
    def validate_goals(cls, v: List[str]) -> List[str]:
        """Validate and sanitize goals."""
        if not v:
            raise ValueError("At least one goal is required")
        # Filter empty strings and limit length
        filtered = [str(goal).strip() for goal in v if goal and str(goal).strip()]
        if not filtered:
            raise ValueError("At least one non-empty goal is required")
        # Limit to 20 goals and 500 chars each
        return [goal[:MAX_GOAL_LENGTH] for goal in filtered[:MAX_LIST_ITEMS]]
    
    @model_validator(mode='after')
    def validate_current_situation(self):
        """Validate current_situation is not empty."""
        if not self.current_situation:
            raise ValueError("current_situation cannot be empty")
        return self


class PersonalizedPlanResponse(BaseModel):
    """Response de plan personalizado."""
    
    plan_name: str = Field(..., description="Nombre del plan")
    duration_weeks: int = Field(..., description="Duración en semanas")
    weekly_goals: List[Dict[str, Any]] = Field(default_factory=list, description="Metas semanales")
    daily_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Acciones diarias")
    milestones: List[Dict[str, Any]] = Field(default_factory=list, description="Hitos del plan")
    resources: List[str] = Field(default_factory=list, description="Recursos necesarios")
    created_date: datetime = Field(default_factory=datetime.now)


# ContinuousProcessorStatusRequest removed - GET endpoint doesn't need request body


class ContinuousProcessorStatusResponse(BaseModel):
    """Response con estado del procesador continuo."""
    
    is_active: bool = Field(..., description="Si el procesador está activo")
    status: str = Field(..., description="Estado: idle, running, stopping, stopped, error")
    interval_seconds: float = Field(..., description="Intervalo entre ejecuciones en segundos")
    execution_count: int = Field(..., description="Número de ejecuciones completadas")
    error_count: int = Field(..., description="Número de errores")
    last_execution: Optional[str] = Field(None, description="Última ejecución (ISO format)")
    last_error: Optional[str] = Field(None, description="Último error si existe")
    uptime_seconds: Optional[float] = Field(None, description="Tiempo activo en segundos")
    start_time: Optional[str] = Field(None, description="Hora de inicio (ISO format)")


class ContinuousProcessorControlRequest(BaseModel):
    """Request para controlar el procesador continuo."""
    
    action: str = Field(..., description="Acción: start, stop, restart")
    interval_seconds: Optional[float] = Field(
        None,
        ge=0.1,  # MIN_INTERVAL_SECONDS from constants
        description="Nuevo intervalo en segundos (mínimo 0.1, solo para start/restart)"
    )


class ContinuousProcessorControlResponse(BaseModel):
    """Response de control del procesador continuo."""
    
    success: bool = Field(..., description="Si la acción fue exitosa")
    message: str = Field(..., description="Mensaje descriptivo")
    status: Dict[str, Any] = Field(..., description="Estado actual del procesador")
