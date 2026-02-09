"""
Modelos de respuesta estandarizados para la API con validaciones mejoradas.
"""

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, validator
from datetime import datetime
from config.logging_config import get_logger

logger = get_logger(__name__)


class ErrorResponse(BaseModel):
    """
    Modelo de respuesta de error con validaciones.
    
    Attributes:
        error: Siempre True para respuestas de error
        message: Mensaje de error principal
        detail: Detalles adicionales del error (opcional)
        status_code: Código de estado HTTP
        timestamp: Timestamp de cuando ocurrió el error
        error_id: ID único del error para tracking (opcional)
    """
    error: bool = Field(True, description="Indica que es una respuesta de error")
    message: str = Field(..., description="Mensaje de error principal", min_length=1)
    detail: Optional[str] = Field(None, description="Detalles adicionales del error")
    status_code: int = Field(..., description="Código de estado HTTP", ge=400, le=599)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp del error"
    )
    error_id: Optional[str] = Field(None, description="ID único del error para tracking")
    
    @validator('message')
    def validate_message(cls, v):
        """Validar que el mensaje no esté vacío."""
        if not v or not v.strip():
            raise ValueError("El mensaje de error no puede estar vacío")
        return v.strip()
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "error": True,
                "message": "Error al procesar la solicitud",
                "detail": "Detalles específicos del error",
                "status_code": 500,
                "timestamp": "2024-01-01T00:00:00",
                "error_id": "err_123456"
            }
        }


class SuccessResponse(BaseModel):
    """
    Modelo de respuesta de éxito con validaciones.
    
    Attributes:
        success: Siempre True para respuestas de éxito
        message: Mensaje de éxito
        data: Datos de la respuesta (opcional)
        timestamp: Timestamp de cuando se completó la operación
    """
    success: bool = Field(True, description="Indica que es una respuesta de éxito")
    message: str = Field(..., description="Mensaje de éxito", min_length=1)
    data: Optional[Dict[str, Any]] = Field(None, description="Datos de la respuesta")
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp del éxito"
    )
    
    @validator('message')
    def validate_message(cls, v):
        """Validar que el mensaje no esté vacío."""
        if not v or not v.strip():
            raise ValueError("El mensaje de éxito no puede estar vacío")
        return v.strip()
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Operación completada exitosamente",
                "data": {"result": "datos"},
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class HealthResponse(BaseModel):
    """
    Modelo de respuesta de health check con validaciones.
    
    Attributes:
        status: Estado general del sistema
        version: Versión de la aplicación
        timestamp: Timestamp del health check
        services: Estado de servicios individuales (opcional)
        details: Detalles adicionales de servicios (opcional)
    """
    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Estado general del sistema"
    )
    version: str = Field(..., description="Versión de la aplicación", min_length=1)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Timestamp del health check"
    )
    services: Optional[Dict[str, bool]] = Field(
        None,
        description="Estado de servicios individuales"
    )
    details: Optional[Dict[str, Dict[str, str]]] = Field(
        None,
        description="Detalles adicionales de servicios"
    )
    
    @validator('version')
    def validate_version(cls, v):
        """Validar que la versión no esté vacía."""
        if not v or not v.strip():
            raise ValueError("La versión no puede estar vacía")
        return v.strip()
    
    class Config:
        """Configuración del modelo."""
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-01T00:00:00",
                "services": {
                    "database": True,
                    "github_api": True
                },
                "details": {
                    "database": {
                        "status": "ok",
                        "response_time_ms": "10"
                    }
                }
            }
        }


