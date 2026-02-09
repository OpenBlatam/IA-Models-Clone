"""
Esquemas Pydantic para Validación Psicológica AI
================================================
"""

from typing import Optional, List, Dict, Any
from uuid import UUID
from datetime import datetime
import structlog
from pydantic import BaseModel, Field, ConfigDict, field_validator
import orjson

from .models import (
    SocialMediaPlatform,
    ConnectionStatus,
    ValidationStatus,
)

logger = structlog.get_logger()


class ORJSONModel(BaseModel):
    """Modelo base con soporte ORJSON"""
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class SocialMediaConnectRequest(ORJSONModel):
    """Solicitud para conectar una red social"""
    platform: SocialMediaPlatform = Field(..., description="Plataforma a conectar")
    access_token: str = Field(..., description="Token de acceso")
    refresh_token: Optional[str] = Field(None, description="Token de refresco")
    expires_in: Optional[int] = Field(None, description="Tiempo de expiración en segundos")

    @field_validator('access_token')
    def validate_token(cls, v: str) -> str:
        """Validar que el token no esté vacío"""
        if not v or not v.strip():
            raise ValueError("Access token cannot be empty")
        return v.strip()


class SocialMediaConnectionResponse(ORJSONModel):
    """Respuesta de conexión a red social"""
    id: UUID
    platform: SocialMediaPlatform
    status: ConnectionStatus
    connected_at: Optional[datetime]
    profile_data: Dict[str, Any] = Field(default_factory=dict)


class ValidationCreate(ORJSONModel):
    """Esquema para crear una nueva validación"""
    platforms: List[SocialMediaPlatform] = Field(
        default_factory=list,
        description="Plataformas a analizar"
    )
    include_historical_data: bool = Field(
        default=True,
        description="Incluir datos históricos en el análisis"
    )
    analysis_depth: str = Field(
        default="standard",
        description="Profundidad del análisis: basic, standard, deep"
    )

    @field_validator('analysis_depth')
    def validate_depth(cls, v: str) -> str:
        """Validar profundidad de análisis"""
        allowed = ["basic", "standard", "deep"]
        if v not in allowed:
            raise ValueError(f"Analysis depth must be one of: {', '.join(allowed)}")
        return v


class ValidationRead(ORJSONModel):
    """Esquema para leer una validación"""
    id: UUID
    user_id: UUID
    status: ValidationStatus
    connected_platforms: List[SocialMediaPlatform]
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime]
    has_profile: bool = Field(default=False, description="Si tiene perfil generado")
    has_report: bool = Field(default=False, description="Si tiene reporte generado")


class PsychologicalProfileResponse(ORJSONModel):
    """Respuesta con perfil psicológico"""
    id: UUID
    user_id: UUID
    personality_traits: Dict[str, float]
    emotional_state: Dict[str, Any]
    behavioral_patterns: List[Dict[str, Any]]
    risk_factors: List[str]
    strengths: List[str]
    recommendations: List[str]
    confidence_score: float
    created_at: datetime
    updated_at: datetime


class ValidationReportResponse(ORJSONModel):
    """Respuesta con reporte de validación"""
    id: UUID
    validation_id: UUID
    summary: str
    detailed_analysis: Dict[str, Any]
    social_media_insights: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    content_analysis: Dict[str, Any]
    interaction_patterns: Dict[str, Any]
    generated_at: datetime


class ValidationDetailResponse(ORJSONModel):
    """Respuesta completa de validación con perfil y reporte"""
    validation: ValidationRead
    profile: Optional[PsychologicalProfileResponse] = None
    report: Optional[ValidationReportResponse] = None
    connections: List[SocialMediaConnectionResponse] = Field(default_factory=list)


class SocialMediaDataRequest(ORJSONModel):
    """Solicitud para obtener datos de redes sociales"""
    platform: SocialMediaPlatform
    data_types: List[str] = Field(
        default_factory=lambda: ["posts", "comments", "likes", "shares"],
        description="Tipos de datos a obtener"
    )
    date_range: Optional[Dict[str, datetime]] = Field(
        None,
        description="Rango de fechas para filtrar datos"
    )


class SocialMediaDataResponse(ORJSONModel):
    """Respuesta con datos de redes sociales"""
    platform: SocialMediaPlatform
    data: Dict[str, Any]
    total_items: int
    date_range: Optional[Dict[str, datetime]] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)




