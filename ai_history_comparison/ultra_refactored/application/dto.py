"""
Data Transfer Objects (DTOs) - Objetos de Transferencia de Datos
==============================================================

DTOs para la comunicación entre capas y con el exterior.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..domain.models import ModelType, QualityLevel


class CreateHistoryEntryRequest(BaseModel):
    """Request para crear una entrada de historial."""
    model_type: ModelType
    content: str = Field(..., min_length=1, max_length=50000)
    metadata: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    assess_quality: bool = Field(default=False)


class UpdateHistoryEntryRequest(BaseModel):
    """Request para actualizar una entrada de historial."""
    content: Optional[str] = Field(None, min_length=1, max_length=50000)
    metadata: Optional[Dict[str, Any]] = None
    assess_quality: bool = Field(default=False)


class CompareEntriesRequest(BaseModel):
    """Request para comparar dos entradas."""
    entry_1_id: str = Field(..., min_length=1)
    entry_2_id: str = Field(..., min_length=1)
    include_differences: bool = Field(default=True)
    
    @validator('entry_1_id', 'entry_2_id')
    def validate_entry_ids_different(cls, v, values):
        """Validar que los IDs sean diferentes."""
        if 'entry_1_id' in values and v == values['entry_1_id']:
            raise ValueError('Entry IDs must be different')
        return v


class QualityAssessmentRequest(BaseModel):
    """Request para evaluación de calidad."""
    entry_id: str = Field(..., min_length=1)
    include_recommendations: bool = Field(default=True)
    detailed_analysis: bool = Field(default=False)


class AnalysisRequest(BaseModel):
    """Request para análisis en lote."""
    job_type: str = Field(..., regex="^(quality_assessment|content_analysis|comparison_analysis)$")
    user_id: Optional[str] = None
    model_type: Optional[ModelType] = None
    limit: Optional[int] = Field(None, ge=1, le=10000)
    filters: Optional[Dict[str, Any]] = None


class HistoryEntryResponse(BaseModel):
    """Response para entrada de historial."""
    id: str
    timestamp: datetime
    model_type: ModelType
    content: str
    metadata: Dict[str, Any]
    quality_score: Optional[float]
    user_id: Optional[str]
    session_id: Optional[str]
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ComparisonResultResponse(BaseModel):
    """Response para resultado de comparación."""
    id: str
    timestamp: datetime
    entry_1_id: str
    entry_2_id: str
    similarity_score: float
    content_similarity: float
    quality_difference: float
    differences: Dict[str, Any]
    analysis_metadata: Dict[str, Any]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QualityReportResponse(BaseModel):
    """Response para reporte de calidad."""
    id: str
    timestamp: datetime
    entry_id: str
    overall_score: float
    quality_level: QualityLevel
    readability_score: float
    coherence_score: float
    relevance_score: float
    sentiment_score: float
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AnalysisJobResponse(BaseModel):
    """Response para trabajo de análisis."""
    id: str
    timestamp: datetime
    job_type: str
    status: str
    total_entries: int
    processed_entries: int
    failed_entries: int
    progress_percentage: float
    results: Dict[str, Any]
    error_message: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class PaginatedResponse(BaseModel):
    """Response paginado genérico."""
    items: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    has_next: bool
    has_previous: bool


class ErrorResponse(BaseModel):
    """Response de error."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheckResponse(BaseModel):
    """Response para health check."""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    database_status: str
    services_status: Dict[str, str]
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class StatisticsResponse(BaseModel):
    """Response para estadísticas del sistema."""
    total_entries: int
    total_comparisons: int
    total_quality_reports: int
    entries_by_model: Dict[str, int]
    average_quality_score: float
    most_common_model: str
    recent_activity: Dict[str, int]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }




