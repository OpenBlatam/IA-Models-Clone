"""
Schemas para el detector multimodal de IA
Define los modelos de datos para inputs y outputs
"""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import base64


class ContentType(str, Enum):
    """Tipos de contenido soportados"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"


class AIDetectionInput(BaseModel):
    """Input para detección de contenido generado por IA"""
    content: Union[str, bytes] = Field(..., description="Contenido a analizar (texto, imagen base64, audio, video)")
    content_type: ContentType = Field(..., description="Tipo de contenido")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Metadatos adicionales del contenido")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Este es un texto generado por IA...",
                "content_type": "text",
                "metadata": {"source": "web", "language": "es"}
            }
        }


class DetectedModel(BaseModel):
    """Modelo de IA detectado"""
    model_name: str = Field(..., description="Nombre del modelo detectado")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confianza de la detección (0-1)")
    provider: Optional[str] = Field(None, description="Proveedor del modelo (OpenAI, Anthropic, etc.)")
    version: Optional[str] = Field(None, description="Versión del modelo")


class ForensicAnalysis(BaseModel):
    """Análisis forense del posible prompt usado"""
    estimated_prompt: Optional[str] = Field(None, description="Prompt estimado que pudo generar el contenido")
    prompt_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confianza en el prompt estimado")
    prompt_patterns: List[str] = Field(default_factory=list, description="Patrones detectados en el prompt")
    generation_parameters: Optional[Dict[str, Any]] = Field(None, description="Parámetros de generación estimados")
    forensic_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="Evidencia forense encontrada")


class QualityInfo(BaseModel):
    """Información sobre la calidad y confiabilidad de la detección"""
    writing_quality: float = Field(0.0, ge=0.0, le=1.0, description="Calidad de escritura (0-1)")
    paraphrase_likelihood: float = Field(0.0, ge=0.0, le=1.0, description="Probabilidad de parafraseo (0-1)")
    risk_score: float = Field(0.0, ge=0.0, le=1.0, description="Score de riesgo y confiabilidad (0-1)")
    reliability: str = Field("medium", description="Nivel de confiabilidad: high, medium, low")


class Alert(BaseModel):
    """Alerta generada por el sistema de detección"""
    type: str = Field(..., description="Tipo de alerta")
    severity: str = Field(..., description="Severidad: critical, high, medium, low")
    message: str = Field(..., description="Mensaje de la alerta")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confianza asociada (si aplica)")
    model: Optional[str] = Field(None, description="Modelo asociado (si aplica)")
    models: Optional[List[str]] = Field(None, description="Lista de modelos (si aplica)")
    ai_percentage: Optional[float] = Field(None, ge=0.0, le=100.0, description="Porcentaje de IA (si aplica)")


class AIDetectionResult(BaseModel):
    """Resultado de la detección de IA"""
    is_ai_generated: bool = Field(..., description="Si el contenido fue generado por IA")
    ai_percentage: float = Field(..., ge=0.0, le=100.0, description="Porcentaje de contenido generado por IA (0-100)")
    detected_models: List[DetectedModel] = Field(default_factory=list, description="Modelos de IA detectados")
    primary_model: Optional[DetectedModel] = Field(None, description="Modelo principal más probable")
    forensic_analysis: Optional[ForensicAnalysis] = Field(None, description="Análisis forense del prompt")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confianza general de la detección")
    detection_methods: List[str] = Field(default_factory=list, description="Métodos de detección utilizados")
    processing_time: float = Field(..., description="Tiempo de procesamiento en segundos")
    timestamp: float = Field(..., description="Timestamp de la detección")
    quality_info: Optional[QualityInfo] = Field(None, description="Información sobre calidad y confiabilidad")
    alerts: List[Alert] = Field(default_factory=list, description="Alertas generadas por el sistema")
    from_cache: Optional[bool] = Field(False, description="Si el resultado viene del cache")


class BatchDetectionInput(BaseModel):
    """Input para detección en batch"""
    items: List[AIDetectionInput] = Field(..., description="Lista de contenidos a analizar")
    parallel: bool = Field(True, description="Procesar en paralelo")


class BatchDetectionResult(BaseModel):
    """Resultado de detección en batch"""
    results: List[AIDetectionResult] = Field(..., description="Resultados de cada detección")
    total_processed: int = Field(..., description="Total de items procesados")
    successful: int = Field(..., description="Detecciones exitosas")
    failed: int = Field(..., description="Detecciones fallidas")
    processing_time: float = Field(..., description="Tiempo total de procesamiento")


class HealthResponse(BaseModel):
    """Respuesta de health check"""
    status: str = Field(..., description="Estado del servicio")
    version: str = Field(..., description="Versión del servicio")
    models_loaded: int = Field(..., description="Número de modelos cargados")
    uptime: float = Field(..., description="Tiempo de actividad en segundos")

