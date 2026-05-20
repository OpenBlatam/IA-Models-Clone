"""
Shared Models - Modelos Compartidos
=================================

Modelos de datos compartidos entre todos los microservicios del sistema ultra refactorizado.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
import uuid


class ModelType(str, Enum):
    """Tipos de modelos de IA."""
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_3_HAIKU = "claude-3-haiku"
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"
    CUSTOM = "custom"


class AnalysisStatus(str, Enum):
    """Estados de análisis."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ServiceStatus(str, Enum):
    """Estados de microservicios."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class HistoryEntry(BaseModel):
    """Entrada de historial de IA."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_type: ModelType
    model_version: str
    prompt: str
    response: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Métricas de rendimiento
    response_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    cost_usd: Optional[float] = None
    
    # Métricas de calidad
    coherence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    relevance_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    creativity_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    accuracy_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Análisis adicional
    analysis_data: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    @validator('response')
    def validate_response(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Response cannot be empty')
        return v.strip()
    
    @validator('prompt')
    def validate_prompt(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        return v.strip()


class ComparisonResult(BaseModel):
    """Resultado de comparación entre entradas."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entry_1_id: str
    entry_2_id: str
    
    # Métricas de similitud
    semantic_similarity: float = Field(..., ge=0.0, le=1.0)
    lexical_similarity: float = Field(..., ge=0.0, le=1.0)
    structural_similarity: float = Field(..., ge=0.0, le=1.0)
    overall_similarity: float = Field(..., ge=0.0, le=1.0)
    
    # Diferencias detectadas
    differences: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    
    # Análisis detallado
    analysis_details: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadatos
    comparison_type: str = "comprehensive"
    confidence_score: float = Field(default=0.8, ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QualityReport(BaseModel):
    """Reporte de calidad de una entrada."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entry_id: str
    
    # Puntuaciones de calidad
    overall_quality: float = Field(..., ge=0.0, le=1.0)
    coherence: float = Field(..., ge=0.0, le=1.0)
    relevance: float = Field(..., ge=0.0, le=1.0)
    creativity: float = Field(..., ge=0.0, le=1.0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    clarity: float = Field(..., ge=0.0, le=1.0)
    
    # Recomendaciones
    recommendations: List[str] = Field(default_factory=list)
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    
    # Metadatos
    analysis_method: str = "automated"
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    assessor_version: str = "1.0"
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisJob(BaseModel):
    """Trabajo de análisis."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    job_type: str  # "comparison", "quality_assessment", "trend_analysis", "comprehensive_analysis"
    status: AnalysisStatus = AnalysisStatus.PENDING
    
    # Parámetros del trabajo
    parameters: Dict[str, Any] = Field(default_factory=dict)
    target_entries: List[str] = Field(default_factory=list)
    
    # Resultados
    results: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    # Metadatos
    created_by: Optional[str] = None
    priority: int = Field(default=1, ge=1, le=10)
    estimated_duration: Optional[int] = None  # segundos
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class TrendAnalysis(BaseModel):
    """Análisis de tendencias."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Período de análisis
    start_date: datetime
    end_date: datetime
    
    # Tendencias detectadas
    model_performance_trends: Dict[str, Any] = Field(default_factory=dict)
    quality_trends: Dict[str, Any] = Field(default_factory=dict)
    usage_trends: Dict[str, Any] = Field(default_factory=dict)
    similarity_trends: Dict[str, Any] = Field(default_factory=dict)
    
    # Insights
    key_insights: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    
    # Métricas
    total_entries_analyzed: int = 0
    confidence_level: float = Field(..., ge=0.0, le=1.0)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class SystemMetrics(BaseModel):
    """Métricas del sistema."""
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Métricas de rendimiento
    total_entries: int = 0
    total_comparisons: int = 0
    total_quality_reports: int = 0
    active_jobs: int = 0
    
    # Métricas de calidad promedio
    average_quality_score: float = 0.0
    average_coherence_score: float = 0.0
    average_relevance_score: float = 0.0
    average_creativity_score: float = 0.0
    average_accuracy_score: float = 0.0
    
    # Métricas de rendimiento
    average_response_time_ms: float = 0.0
    total_tokens_processed: int = 0
    total_cost_usd: float = 0.0
    
    # Métricas de uso por modelo
    model_usage_stats: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Métricas de microservicios
    service_health: Dict[str, ServiceStatus] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class EventMessage(BaseModel):
    """Mensaje de evento para el sistema de mensajería."""
    
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_service: str
    target_service: Optional[str] = None
    
    # Datos del evento
    data: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadatos
    priority: int = Field(default=1, ge=1, le=10)
    retry_count: int = 0
    max_retries: int = 3
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class CircuitBreakerState(str, Enum):
    """Estados del circuit breaker."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class HealthCheck(BaseModel):
    """Verificación de salud de un microservicio."""
    
    service_name: str
    status: ServiceStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Detalles de salud
    database_status: Optional[str] = None
    message_broker_status: Optional[str] = None
    external_dependencies: Dict[str, str] = Field(default_factory=dict)
    
    # Métricas de rendimiento
    response_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Información adicional
    version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class APIResponse(BaseModel):
    """Respuesta estándar de la API."""
    
    success: bool
    message: str
    data: Optional[Any] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Metadatos de respuesta
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel):
    """Respuesta paginada."""
    
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    # Metadatos
    has_next: bool
    has_previous: bool
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ErrorResponse(BaseModel):
    """Respuesta de error estándar."""
    
    error: str
    message: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Detalles del error
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    request_id: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




