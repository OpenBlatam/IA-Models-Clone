"""
MCP Errors - Códigos de error específicos para MCP
==================================================

Sistema de códigos de error estandarizados para el servidor MCP.
"""

from enum import Enum
from typing import Optional, Dict, Any
from fastapi import HTTPException


class MCPErrorCode(Enum):
    """Códigos de error específicos de MCP"""
    INVALID_COMMAND = "MCP_001"
    COMMAND_TOO_LONG = "MCP_002"
    COMMAND_EMPTY = "MCP_003"
    RATE_LIMIT_EXCEEDED = "MCP_004"
    CIRCUIT_BREAKER_OPEN = "MCP_005"
    AUTHENTICATION_REQUIRED = "MCP_006"
    AUTHENTICATION_FAILED = "MCP_007"
    INSUFFICIENT_PERMISSIONS = "MCP_008"
    TASK_NOT_FOUND = "MCP_009"
    INVALID_EVENT_TYPE = "MCP_010"
    CACHE_NOT_AVAILABLE = "MCP_011"
    BATCH_OPERATIONS_DISABLED = "MCP_012"
    INVALID_JSON_RPC = "MCP_013"
    WEBSOCKET_TIMEOUT = "MCP_014"
    INTERNAL_ERROR = "MCP_500"


class MCPException(HTTPException):
    """Excepción personalizada para errores MCP"""
    
    def __init__(
        self,
        error_code: MCPErrorCode,
        detail: str,
        status_code: int = 400,
        headers: Optional[Dict[str, Any]] = None
    ):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error_code = error_code
        self.detail = detail
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario"""
        return {
            "error": {
                "code": self.error_code.value,
                "message": self.detail,
                "type": self.error_code.name
            },
            "status_code": self.status_code
        }


def create_mcp_exception(
    error_code: MCPErrorCode,
    detail: str,
    status_code: int = 400
) -> MCPException:
    """Crear excepción MCP"""
    return MCPException(error_code=error_code, detail=detail, status_code=status_code)

