"""
Modelos de datos para Validación Psicológica AI
================================================
"""

from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime
from enum import Enum
import structlog
import orjson
from pydantic import BaseModel, Field, ConfigDict, field_validator

from onyx.core.models import OnyxBaseModel

logger = structlog.get_logger()


class SocialMediaPlatform(str, Enum):
    """Plataformas de redes sociales soportadas"""
    FACEBOOK = "facebook"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    REDDIT = "reddit"
    DISCORD = "discord"
    TELEGRAM = "telegram"


class ConnectionStatus(str, Enum):
    """Estado de conexión a redes sociales"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    PENDING = "pending"
    ERROR = "error"
    EXPIRED = "expired"


class ValidationStatus(str, Enum):
    """Estado de la validación psicológica"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ORJSONModel(OnyxBaseModel):
    """Modelo base con soporte ORJSON"""
    model_config = ConfigDict(json_loads=orjson.loads, json_dumps=orjson.dumps)


class SocialMediaConnection(ORJSONModel):
    """Conexión a una red social específica"""
    __slots__ = (
        'id', 'user_id', 'platform', 'access_token', 'refresh_token',
        'status', 'connected_at', 'last_sync_at', 'expires_at',
        'profile_data', 'created_at', 'updated_at'
    )
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="ID del usuario")
    platform: SocialMediaPlatform = Field(..., description="Plataforma de red social")
    access_token: Optional[str] = Field(None, description="Token de acceso")
    refresh_token: Optional[str] = Field(None, description="Token de refresco")
    status: ConnectionStatus = Field(default=ConnectionStatus.PENDING)
    connected_at: Optional[datetime] = Field(None, description="Fecha de conexión")
    last_sync_at: Optional[datetime] = Field(None, description="Última sincronización")
    expires_at: Optional[datetime] = Field(None, description="Fecha de expiración del token")
    profile_data: Dict[str, Any] = Field(default_factory=dict, description="Datos del perfil")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    def update_status(self, status: ConnectionStatus) -> None:
        """Actualizar estado de conexión"""
        self.status = status
        self.updated_at = datetime.utcnow()
        logger.info("Connection status updated", connection_id=str(self.id), status=status.value)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return self.model_dump()


class PsychologicalProfile(ORJSONModel):
    """Perfil psicológico del usuario"""
    __slots__ = (
        'id', 'user_id', 'personality_traits', 'emotional_state',
        'behavioral_patterns', 'risk_factors', 'strengths',
        'recommendations', 'confidence_score', 'created_at', 'updated_at'
    )
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="ID del usuario")
    personality_traits: Dict[str, float] = Field(
        default_factory=dict,
        description="Rasgos de personalidad con scores (0-1)"
    )
    emotional_state: Dict[str, Any] = Field(
        default_factory=dict,
        description="Estado emocional actual"
    )
    behavioral_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Patrones de comportamiento identificados"
    )
    risk_factors: List[str] = Field(
        default_factory=list,
        description="Factores de riesgo identificados"
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Fortalezas identificadas"
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recomendaciones basadas en el análisis"
    )
    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza del análisis (0-1)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator('confidence_score')
    def validate_confidence(cls, v: float) -> float:
        """Validar score de confianza"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return self.model_dump()


class ValidationReport(ORJSONModel):
    """Reporte de validación psicológica"""
    __slots__ = (
        'id', 'validation_id', 'summary', 'detailed_analysis',
        'social_media_insights', 'timeline_analysis', 'sentiment_analysis',
        'content_analysis', 'interaction_patterns', 'generated_at'
    )
    
    id: UUID = Field(default_factory=uuid4)
    validation_id: UUID = Field(..., description="ID de la validación asociada")
    summary: str = Field(..., description="Resumen ejecutivo del análisis")
    detailed_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Análisis detallado por categoría"
    )
    social_media_insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Insights por plataforma de red social"
    )
    timeline_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Análisis temporal de comportamiento"
    )
    sentiment_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Análisis de sentimientos"
    )
    content_analysis: Dict[str, Any] = Field(
        default_factory=dict,
        description="Análisis de contenido publicado"
    )
    interaction_patterns: Dict[str, Any] = Field(
        default_factory=dict,
        description="Patrones de interacción social"
    )
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return self.model_dump()


class PsychologicalValidation(ORJSONModel):
    """Validación psicológica completa"""
    __slots__ = (
        'id', 'user_id', 'status', 'connected_platforms',
        'profile', 'report', 'metadata', 'created_at',
        'updated_at', 'completed_at', 'version', 'trace_id'
    )
    
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID = Field(..., description="ID del usuario")
    status: ValidationStatus = Field(default=ValidationStatus.PENDING)
    connected_platforms: List[SocialMediaPlatform] = Field(
        default_factory=list,
        description="Plataformas conectadas para esta validación"
    )
    profile: Optional[PsychologicalProfile] = Field(
        None,
        description="Perfil psicológico generado"
    )
    report: Optional[ValidationReport] = Field(
        None,
        description="Reporte de validación generado"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadatos adicionales"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None, description="Fecha de finalización")
    version: int = Field(default=1, description="Versión de la validación")
    trace_id: Optional[str] = Field(None, description="ID de trazabilidad")

    def update_status(self, status: ValidationStatus) -> None:
        """Actualizar estado de validación"""
        self.status = status
        self.updated_at = datetime.utcnow()
        if status == ValidationStatus.COMPLETED:
            self.completed_at = datetime.utcnow()
        logger.info("Validation status updated", validation_id=str(self.id), status=status.value)

    def add_platform(self, platform: SocialMediaPlatform) -> None:
        """Agregar plataforma conectada"""
        if platform not in self.connected_platforms:
            self.connected_platforms.append(platform)
            self.updated_at = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return self.model_dump()

    def to_json(self) -> str:
        """Convertir a JSON"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "PsychologicalValidation":
        """Crear desde JSON"""
        return cls.model_validate_json(data)




