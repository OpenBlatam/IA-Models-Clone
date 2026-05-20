"""
Schemas para Dog Training Coaching AI
======================================
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class BaseResponse(BaseModel):
    """Base response con campos comunes."""
    success: bool = Field(..., description="Indica si la solicitud fue exitosa")
    error: Optional[str] = Field(None, description="Mensaje de error si hubo algún problema")
    timestamp: Optional[str] = Field(default_factory=lambda: datetime.now().isoformat())
    model: Optional[str] = Field(None, description="Modelo de IA utilizado")


class CoachingRequest(BaseModel):
    """Request para coaching de adiestramiento."""
    
    question: str = Field(..., min_length=5, description="Pregunta sobre adiestramiento de perros")
    dog_breed: Optional[str] = Field(None, description="Raza del perro")
    dog_age: Optional[str] = Field(None, description="Edad del perro")
    dog_size: Optional[str] = Field(None, description="Tamaño del perro (small, medium, large, giant)")
    training_goal: Optional[str] = Field(None, description="Objetivo de entrenamiento")
    experience_level: Optional[str] = Field(None, description="Nivel de experiencia del dueño (beginner, intermediate, advanced)")
    previous_context: Optional[str] = Field(None, description="Contexto previo de la conversación")
    specific_issues: Optional[List[str]] = Field(default_factory=list, description="Problemas específicos a abordar")


class CoachingResponse(BaseResponse):
    """Response de coaching."""
    
    advice: Optional[str] = Field(None, description="Consejo de entrenamiento proporcionado")
    key_points: Optional[List[str]] = Field(default_factory=list, description="Puntos clave del consejo")
    next_steps: Optional[List[str]] = Field(default_factory=list, description="Próximos pasos recomendados")


class TrainingPlanRequest(BaseModel):
    """Request para crear plan de entrenamiento."""
    
    dog_breed: str = Field(..., min_length=2, description="Raza del perro")
    dog_age: str = Field(..., description="Edad del perro")
    dog_size: Optional[str] = Field(None, description="Tamaño del perro")
    training_goals: List[str] = Field(..., min_items=1, max_items=10, description="Objetivos de entrenamiento")
    time_available: Optional[str] = Field(None, description="Tiempo disponible por día/semana")
    experience_level: Optional[str] = Field(None, description="Nivel de experiencia del dueño")
    current_issues: Optional[List[str]] = Field(default_factory=list, description="Problemas actuales a abordar")
    preferred_methods: Optional[List[str]] = Field(default_factory=list, description="Métodos de entrenamiento preferidos")


class TrainingPlanResponse(BaseResponse):
    """Response del plan de entrenamiento."""
    
    plan: Optional[str] = Field(None, description="Plan de entrenamiento completo")
    phases: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Fases del plan estructuradas")
    milestones: Optional[List[str]] = Field(default_factory=list, description="Hitos del entrenamiento")
    estimated_duration: Optional[str] = Field(None, description="Duración estimada del plan")


class BehaviorAnalysisRequest(BaseModel):
    """Request para análisis de comportamiento."""
    
    behavior_description: str = Field(..., min_length=10, description="Descripción del comportamiento")
    dog_breed: Optional[str] = Field(None, description="Raza del perro")
    dog_age: Optional[str] = Field(None, description="Edad del perro")
    frequency: Optional[str] = Field(None, description="Frecuencia del comportamiento")
    triggers: Optional[List[str]] = Field(default_factory=list, description="Desencadenantes del comportamiento")
    context: Optional[str] = Field(None, description="Contexto adicional")


class BehaviorAnalysisResponse(BaseResponse):
    """Response del análisis de comportamiento."""
    
    analysis: Optional[str] = Field(None, description="Análisis del comportamiento")
    possible_causes: Optional[List[str]] = Field(default_factory=list, description="Posibles causas")
    recommendations: Optional[List[str]] = Field(default_factory=list, description="Recomendaciones")
    training_exercises: Optional[List[str]] = Field(default_factory=list, description="Ejercicios de entrenamiento sugeridos")


class ChatRequest(BaseModel):
    """Request para chat con el asistente."""
    
    message: str = Field(..., min_length=1, description="Mensaje del usuario")
    conversation_history: Optional[List[Dict[str, str]]] = Field(None, description="Historial de conversación")
    dog_info: Optional[Dict[str, str]] = Field(None, description="Información del perro")


class ChatResponse(BaseResponse):
    """Response del chat."""
    
    response: str = Field(..., description="Respuesta del asistente")
    suggestions: Optional[List[str]] = Field(default_factory=list, description="Sugerencias de seguimiento")
    resources: Optional[List[str]] = Field(default_factory=list, description="Recursos relevantes")


class TrainingProgressRequest(BaseModel):
    """Request para seguimiento de progreso de entrenamiento."""
    
    dog_id: Optional[str] = Field(None, description="ID del perro")
    training_sessions: List[Dict[str, Any]] = Field(default_factory=list, description="Historial de sesiones de entrenamiento")
    current_skills: Optional[List[str]] = Field(None, description="Habilidades actuales del perro")
    training_goals: Optional[List[str]] = Field(None, description="Objetivos de entrenamiento establecidos")
    challenges_faced: Optional[List[str]] = Field(default_factory=list, description="Desafíos encontrados")
    time_period_days: Optional[int] = Field(30, ge=1, le=365, description="Período de tiempo en días")


class TrainingProgressResponse(BaseResponse):
    """Response de seguimiento de progreso."""
    
    progress_score: float = Field(..., ge=0, le=100, description="Puntuación de progreso (0-100)")
    trend: str = Field(..., description="Tendencia: improving, stable, declining")
    milestones_achieved: List[str] = Field(default_factory=list, description="Logros alcanzados")
    skills_improved: List[str] = Field(default_factory=list, description="Habilidades mejoradas")
    next_steps: List[str] = Field(default_factory=list, description="Próximos pasos recomendados")
    insights: Optional[str] = Field(None, description="Insights sobre el progreso")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones personalizadas")
    progress_date: datetime = Field(default_factory=datetime.now)


class TrainingAssessmentRequest(BaseModel):
    """Request para evaluación de entrenamiento."""
    
    dog_breed: str = Field(..., description="Raza del perro")
    dog_age: str = Field(..., description="Edad del perro")
    current_skills: List[str] = Field(default_factory=list, description="Habilidades actuales")
    training_goals: List[str] = Field(..., min_items=1, description="Objetivos de entrenamiento")
    training_duration_weeks: Optional[int] = Field(None, ge=0, description="Duración del entrenamiento en semanas")
    behavior_issues: Optional[List[str]] = Field(default_factory=list, description="Problemas de comportamiento")
    owner_experience: Optional[str] = Field(None, description="Experiencia del dueño")


class TrainingAssessmentResponse(BaseResponse):
    """Response de evaluación de entrenamiento."""
    
    assessment_score: float = Field(..., ge=0, le=100, description="Puntuación de evaluación")
    skill_level: str = Field(..., description="Nivel de habilidad: beginner, intermediate, advanced")
    strengths: List[str] = Field(default_factory=list, description="Fortalezas identificadas")
    areas_for_improvement: List[str] = Field(default_factory=list, description="Áreas de mejora")
    recommended_focus: List[str] = Field(default_factory=list, description="Áreas de enfoque recomendadas")
    training_readiness: str = Field(..., description="Preparación: ready, needs_preparation, not_ready")
    personalized_recommendations: List[str] = Field(default_factory=list, description="Recomendaciones personalizadas")
    assessment_date: datetime = Field(default_factory=datetime.now)


class TrainingResourceRequest(BaseModel):
    """Request para recursos educativos de entrenamiento."""
    
    topic: str = Field(..., description="Tema de interés (ej: obedience, agility, behavior)")
    level: Optional[str] = Field("beginner", description="Nivel: beginner, intermediate, advanced")
    format_preference: Optional[str] = Field(None, description="Formato: article, video, guide, exercise")
    dog_breed: Optional[str] = Field(None, description="Raza del perro para recursos específicos")
    specific_need: Optional[str] = Field(None, description="Necesidad específica")


class TrainingResourceResponse(BaseResponse):
    """Response de recursos educativos."""
    
    resources: List[Dict[str, Any]] = Field(default_factory=list, description="Recursos recomendados")
    learning_path: List[str] = Field(default_factory=list, description="Ruta de aprendizaje sugerida")
    key_concepts: List[str] = Field(default_factory=list, description="Conceptos clave")
    exercises: List[Dict[str, Any]] = Field(default_factory=list, description="Ejercicios prácticos")
    tools_needed: List[str] = Field(default_factory=list, description="Herramientas necesarias")
    time_estimate: Optional[str] = Field(None, description="Estimación de tiempo")


class TrainingTrendAnalysisRequest(BaseModel):
    """Request para análisis de tendencias de entrenamiento."""
    
    training_sessions: List[Dict[str, Any]] = Field(..., description="Historial de sesiones de entrenamiento")
    time_period_days: Optional[int] = Field(30, ge=1, le=365, description="Período de tiempo en días")
    metrics_to_analyze: Optional[List[str]] = Field(default_factory=list, description="Métricas a analizar")


class TrainingTrendAnalysisResponse(BaseResponse):
    """Response de análisis de tendencias."""
    
    overall_trend: str = Field(..., description="Tendencia general: improving, stable, declining")
    key_metrics: Dict[str, Any] = Field(default_factory=dict, description="Métricas clave")
    patterns: List[str] = Field(default_factory=list, description="Patrones identificados")
    predictions: Dict[str, Any] = Field(default_factory=dict, description="Predicciones futuras")
    recommendations: List[str] = Field(default_factory=list, description="Recomendaciones basadas en tendencias")
    improvement_areas: List[str] = Field(default_factory=list, description="Áreas de mejora identificadas")
    analysis_date: datetime = Field(default_factory=datetime.now)

