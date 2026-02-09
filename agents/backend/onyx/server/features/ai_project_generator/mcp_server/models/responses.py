"""
Response models for MCP Server
===============================

Modelos Pydantic para responses del servidor MCP, incluyendo
validaciones y estructura consistente.
"""

from typing import Any, Dict, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field, validator


class MCPResponse(BaseModel):
    """
    Response model para llamadas MCP.
    
    Estructura consistente para todas las respuestas del servidor,
    incluyendo éxito, errores y metadata.
    """
    success: bool = Field(
        ...,
        description="Indica si la operación fue exitosa"
    )
    data: Optional[Any] = Field(
        None,
        description="Datos de respuesta (solo si success=True)"
    )
    error: Optional[str] = Field(
        None,
        description="Mensaje de error (solo si success=False)"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata adicional sobre la operación"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp de la respuesta"
    )
    
    @validator("error")
    def validate_error(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        """Valida que error solo esté presente si success=False"""
        success = values.get("success", True)
        if success and v:
            raise ValueError("error should not be set when success=True")
        if not success and not v:
            # Permitir error vacío pero loguear warning
            pass
        return v
    
    @validator("data")
    def validate_data(cls, v: Optional[Any], values: Dict[str, Any]) -> Optional[Any]:
        """Valida que data solo esté presente si success=True"""
        success = values.get("success", True)
        if not success and v is not None:
            # Permitir data en errores para contexto, pero no es común
            pass
        return v
    
    @classmethod
    def success_response(
        cls,
        data: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MCPResponse":
        """
        Crea una respuesta de éxito.
        
        Args:
            data: Datos de respuesta
            metadata: Metadata adicional (opcional)
            
        Returns:
            MCPResponse con success=True
        """
        return cls(
            success=True,
            data=data,
            metadata=metadata or {},
        )
    
    @classmethod
    def error_response(
        cls,
        error: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "MCPResponse":
        """
        Crea una respuesta de error.
        
        Args:
            error: Mensaje de error
            metadata: Metadata adicional (opcional)
            
        Returns:
            MCPResponse con success=False
        """
        return cls(
            success=False,
            error=error,
            metadata=metadata or {},
        )
    
    class Config:
        """Configuración del modelo Pydantic"""
        json_schema_extra = {
            "example_success": {
                "success": True,
                "data": {"content": "example data"},
                "metadata": {"resource_id": "test", "operation": "read"},
                "timestamp": "2024-01-01T00:00:00"
            },
            "example_error": {
                "success": False,
                "error": "Resource not found",
                "metadata": {"resource_id": "test", "operation": "read"},
                "timestamp": "2024-01-01T00:00:00"
            }
        }

