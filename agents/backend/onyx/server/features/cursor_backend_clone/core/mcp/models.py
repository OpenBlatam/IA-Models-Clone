"""
MCP Models - Modelos Pydantic para el servidor MCP
===================================================

Modelos de validación para requests y responses del servidor MCP.
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CommandRequest(BaseModel):
    """Modelo de validación para requests de comandos"""
    command: str = Field(..., min_length=1, max_length=10000, description="Comando a ejecutar")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadatos adicionales")


class BatchCommandRequest(BaseModel):
    """Modelo para requests de comandos en lote"""
    commands: List[str] = Field(..., min_items=1, max_items=100, description="Lista de comandos a ejecutar")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadatos adicionales")
    parallel: bool = Field(default=False, description="Ejecutar comandos en paralelo")


class LoginRequest(BaseModel):
    """Modelo para requests de login"""
    username: str = Field(..., min_length=1, description="Nombre de usuario")
    password: str = Field(..., min_length=1, description="Contraseña")


class JSONRPCRequest(BaseModel):
    """Modelo de validación para mensajes JSON-RPC"""
    jsonrpc: str = Field(default="2.0", description="Versión JSON-RPC")
    method: str = Field(..., description="Método a ejecutar")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parámetros del método")
    id: Optional[str] = Field(default=None, description="ID de la solicitud")




