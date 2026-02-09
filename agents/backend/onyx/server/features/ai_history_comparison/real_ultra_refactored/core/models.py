"""
Core Models - Modelos Principales
================================

Modelos de datos principales del sistema ultra refactorizado real.
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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisJob(BaseModel):
    """Trabajo de análisis."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    job_type: str  # "comparison", "quality_assessment", "trend_analysis"
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
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




