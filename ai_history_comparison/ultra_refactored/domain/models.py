"""
Domain Models - Modelos de Dominio
=================================

Entidades de dominio que representan los conceptos principales
del sistema de comparación de historial de IA.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
import uuid


class ModelType(str, Enum):
    """Tipos de modelos de IA soportados."""
    GPT_4 = "gpt-4"
    GPT_3_5 = "gpt-3.5-turbo"
    CLAUDE_3 = "claude-3"
    CLAUDE_2 = "claude-2"
    GEMINI_PRO = "gemini-pro"
    CUSTOM = "custom"


class QualityLevel(str, Enum):
    """Niveles de calidad del contenido."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    VERY_POOR = "very_poor"


class HistoryEntry(BaseModel):
    """
    Entrada de historial de IA.
    
    Representa una entrada individual en el historial de contenido generado por IA.
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    model_type: ModelType
    content: str = Field(..., min_length=1, max_length=50000)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    @validator('content')
    def validate_content(cls, v):
        """Validar que el contenido no esté vacío."""
        if not v or not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()
    
    @validator('metadata')
    def validate_metadata(cls, v):
        """Validar que los metadatos no excedan el tamaño máximo."""
        if len(str(v)) > 10000:  # Límite de 10KB para metadatos
            raise ValueError('Metadata too large')
        return v
    
    class Config:
        """Configuración del modelo."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComparisonResult(BaseModel):
    """
    Resultado de comparación entre dos entradas de historial.
    
    Contiene métricas de similitud y diferencias entre dos entradas.
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entry_1_id: str
    entry_2_id: str
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    content_similarity: float = Field(..., ge=0.0, le=1.0)
    quality_difference: float = Field(..., ge=-1.0, le=1.0)
    differences: Dict[str, Any] = Field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('entry_1_id', 'entry_2_id')
    def validate_entry_ids(cls, v):
        """Validar que los IDs de entrada no sean iguales."""
        return v
    
    @validator('entry_1_id', 'entry_2_id')
    def validate_entry_ids_different(cls, v, values):
        """Validar que los IDs de entrada sean diferentes."""
        if 'entry_1_id' in values and v == values['entry_1_id']:
            raise ValueError('Entry IDs must be different')
        return v
    
    class Config:
        """Configuración del modelo."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QualityReport(BaseModel):
    """
    Reporte de calidad de una entrada de historial.
    
    Contiene análisis detallado de la calidad del contenido.
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    entry_id: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    quality_level: QualityLevel
    readability_score: float = Field(..., ge=0.0, le=1.0)
    coherence_score: float = Field(..., ge=0.0, le=1.0)
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    recommendations: List[str] = Field(default_factory=list)
    detailed_analysis: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('quality_level')
    def validate_quality_level(cls, v, values):
        """Validar que el nivel de calidad coincida con el score."""
        if 'overall_score' in values:
            score = values['overall_score']
            if score >= 0.9 and v != QualityLevel.EXCELLENT:
                pass  # Permitir discrepancias menores
            elif score >= 0.7 and v not in [QualityLevel.EXCELLENT, QualityLevel.GOOD]:
                pass
            elif score >= 0.5 and v not in [QualityLevel.GOOD, QualityLevel.AVERAGE]:
                pass
            elif score >= 0.3 and v not in [QualityLevel.AVERAGE, QualityLevel.POOR]:
                pass
            elif score < 0.3 and v not in [QualityLevel.POOR, QualityLevel.VERY_POOR]:
                pass
        return v
    
    class Config:
        """Configuración del modelo."""
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisJob(BaseModel):
    """
    Trabajo de análisis en lote.
    
    Representa un trabajo de análisis que puede procesar múltiples entradas.
    """
    id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    job_type: str = Field(..., min_length=1, max_length=100)
    status: str = Field(default="pending", regex="^(pending|running|completed|failed)$")
    total_entries: int = Field(..., ge=1)
    processed_entries: int = Field(default=0, ge=0)
    failed_entries: int = Field(default=0, ge=0)
    results: Dict[str, Any] = Field(default_factory=dict)
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    @validator('processed_entries', 'failed_entries')
    def validate_processed_entries(cls, v, values):
        """Validar que las entradas procesadas no excedan el total."""
        if 'total_entries' in values:
            total = values['total_entries']
            if v > total:
                raise ValueError('Processed entries cannot exceed total entries')
        return v
    
    @property
    def progress_percentage(self) -> float:
        """Calcular el porcentaje de progreso."""
        if self.total_entries == 0:
            return 0.0
        return (self.processed_entries / self.total_entries) * 100
    
    @property
    def is_completed(self) -> bool:
        """Verificar si el trabajo está completado."""
        return self.status == "completed"
    
    @property
    def is_failed(self) -> bool:
        """Verificar si el trabajo falló."""
        return self.status == "failed"
    
    class Config:
        """Configuración del modelo."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




