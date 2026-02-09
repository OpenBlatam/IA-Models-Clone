"""
Request models for MCP Server
==============================

Modelos Pydantic para requests del servidor MCP, incluyendo
validaciones y documentación.
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, validator
from ..contracts import ContextFrame


class MCPRequest(BaseModel):
    """
    Request model para llamadas MCP.
    
    Valida y estructura las peticiones al servidor MCP,
    incluyendo validaciones de formato y contenido.
    """
    resource_id: str = Field(
        ...,
        description="ID del recurso a consultar",
        min_length=1,
        max_length=255,
    )
    operation: str = Field(
        ...,
        description="Operación a realizar (read, write, query, etc.)",
        min_length=1,
        max_length=100,
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parámetros de la operación"
    )
    context: Optional[ContextFrame] = Field(
        None,
        description="Frame de contexto adicional"
    )
    
    @validator("resource_id")
    def validate_resource_id(cls, v: str) -> str:
        """Valida que resource_id sea válido"""
        if not v or not v.strip():
            raise ValueError("resource_id cannot be empty")
        return v.strip()
    
    @validator("operation")
    def validate_operation(cls, v: str) -> str:
        """Valida que operation sea válido"""
        if not v or not v.strip():
            raise ValueError("operation cannot be empty")
        # Normalizar a lowercase para consistencia
        return v.strip().lower()
    
    @validator("parameters")
    def validate_parameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Valida que parameters sea un diccionario válido"""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("parameters must be a dictionary")
        return v
    
    class Config:
        """Configuración del modelo Pydantic"""
        json_schema_extra = {
            "example": {
                "resource_id": "filesystem:/path/to/file",
                "operation": "read",
                "parameters": {"path": "/example.txt"},
                "context": None
            }
        }

