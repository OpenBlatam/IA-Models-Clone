"""
Códigos de Error Estandarizados.
"""

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime


class ErrorCode(str, Enum):
    """Códigos de error estandarizados."""
    
    # Errores generales (1000-1999)
    INTERNAL_ERROR = "ERR_1000"
    INVALID_REQUEST = "ERR_1001"
    VALIDATION_ERROR = "ERR_1002"
    NOT_FOUND = "ERR_1003"
    UNAUTHORIZED = "ERR_1004"
    FORBIDDEN = "ERR_1005"
    RATE_LIMIT_EXCEEDED = "ERR_1006"
    SERVICE_UNAVAILABLE = "ERR_1007"
    TIMEOUT = "ERR_1008"
    
    # Errores de tareas (2000-2999)
    TASK_NOT_FOUND = "ERR_2000"
    TASK_ALREADY_EXISTS = "ERR_2001"
    TASK_INVALID_STATUS = "ERR_2002"
    TASK_PROCESSING_ERROR = "ERR_2003"
    TASK_CREATION_FAILED = "ERR_2004"
    
    # Errores de GitHub (3000-3999)
    GITHUB_AUTH_ERROR = "ERR_3000"
    GITHUB_RATE_LIMIT = "ERR_3001"
    GITHUB_REPO_NOT_FOUND = "ERR_3002"
    GITHUB_PERMISSION_DENIED = "ERR_3003"
    GITHUB_API_ERROR = "ERR_3004"
    
    # Errores de LLM (4000-4999)
    LLM_SERVICE_UNAVAILABLE = "ERR_4000"
    LLM_INVALID_REQUEST = "ERR_4001"
    LLM_RATE_LIMIT = "ERR_4002"
    LLM_TIMEOUT = "ERR_4003"
    LLM_QUOTA_EXCEEDED = "ERR_4004"
    
    # Errores de base de datos (5000-5999)
    DATABASE_ERROR = "ERR_5000"
    DATABASE_CONNECTION_ERROR = "ERR_5001"
    DATABASE_QUERY_ERROR = "ERR_5002"
    DATABASE_TRANSACTION_ERROR = "ERR_5003"
    
    # Errores de autenticación (6000-6999)
    AUTH_INVALID_TOKEN = "ERR_6000"
    AUTH_TOKEN_EXPIRED = "ERR_6001"
    AUTH_INSUFFICIENT_PERMISSIONS = "ERR_6002"
    AUTH_API_KEY_INVALID = "ERR_6003"
    AUTH_API_KEY_EXPIRED = "ERR_6004"
    
    # Errores de configuración (7000-7999)
    CONFIG_MISSING = "ERR_7000"
    CONFIG_INVALID = "ERR_7001"
    CONFIG_VALIDATION_ERROR = "ERR_7002"


class ErrorResponse:
    """Respuesta de error estandarizada."""
    
    def __init__(
        self,
        code: ErrorCode,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        http_status: int = 500
    ):
        """
        Inicializar respuesta de error.
        
        Args:
            code: Código de error
            message: Mensaje de error
            details: Detalles adicionales (opcional)
            http_status: Status HTTP (opcional)
        """
        self.code = code
        self.message = message
        self.details = details or {}
        self.http_status = http_status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "error": True,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "timestamp": datetime.now().isoformat()
        }


# Mapeo de códigos a mensajes
ERROR_MESSAGES: Dict[ErrorCode, str] = {
    ErrorCode.INTERNAL_ERROR: "Error interno del servidor",
    ErrorCode.INVALID_REQUEST: "Request inválido",
    ErrorCode.VALIDATION_ERROR: "Error de validación",
    ErrorCode.NOT_FOUND: "Recurso no encontrado",
    ErrorCode.UNAUTHORIZED: "No autorizado",
    ErrorCode.FORBIDDEN: "Acceso prohibido",
    ErrorCode.RATE_LIMIT_EXCEEDED: "Límite de rate excedido",
    ErrorCode.SERVICE_UNAVAILABLE: "Servicio no disponible",
    ErrorCode.TIMEOUT: "Timeout de operación",
    ErrorCode.TASK_NOT_FOUND: "Tarea no encontrada",
    ErrorCode.TASK_ALREADY_EXISTS: "La tarea ya existe",
    ErrorCode.TASK_INVALID_STATUS: "Estado de tarea inválido",
    ErrorCode.TASK_PROCESSING_ERROR: "Error procesando tarea",
    ErrorCode.TASK_CREATION_FAILED: "Error al crear tarea",
    ErrorCode.GITHUB_AUTH_ERROR: "Error de autenticación con GitHub",
    ErrorCode.GITHUB_RATE_LIMIT: "Rate limit de GitHub excedido",
    ErrorCode.GITHUB_REPO_NOT_FOUND: "Repositorio de GitHub no encontrado",
    ErrorCode.GITHUB_PERMISSION_DENIED: "Permisos insuficientes en GitHub",
    ErrorCode.GITHUB_API_ERROR: "Error en API de GitHub",
    ErrorCode.LLM_SERVICE_UNAVAILABLE: "Servicio LLM no disponible",
    ErrorCode.LLM_INVALID_REQUEST: "Request inválido para LLM",
    ErrorCode.LLM_RATE_LIMIT: "Rate limit de LLM excedido",
    ErrorCode.LLM_TIMEOUT: "Timeout en servicio LLM",
    ErrorCode.LLM_QUOTA_EXCEEDED: "Cuota de LLM excedida",
    ErrorCode.DATABASE_ERROR: "Error de base de datos",
    ErrorCode.DATABASE_CONNECTION_ERROR: "Error de conexión a base de datos",
    ErrorCode.DATABASE_QUERY_ERROR: "Error en query de base de datos",
    ErrorCode.DATABASE_TRANSACTION_ERROR: "Error en transacción de base de datos",
    ErrorCode.AUTH_INVALID_TOKEN: "Token inválido",
    ErrorCode.AUTH_TOKEN_EXPIRED: "Token expirado",
    ErrorCode.AUTH_INSUFFICIENT_PERMISSIONS: "Permisos insuficientes",
    ErrorCode.AUTH_API_KEY_INVALID: "API key inválida",
    ErrorCode.AUTH_API_KEY_EXPIRED: "API key expirada",
    ErrorCode.CONFIG_MISSING: "Configuración faltante",
    ErrorCode.CONFIG_INVALID: "Configuración inválida",
    ErrorCode.CONFIG_VALIDATION_ERROR: "Error de validación de configuración",
}


def get_error_message(code: ErrorCode) -> str:
    """
    Obtener mensaje de error para un código.
    
    Args:
        code: Código de error
        
    Returns:
        Mensaje de error
    """
    return ERROR_MESSAGES.get(code, "Error desconocido")


def create_error_response(
    code: ErrorCode,
    details: Optional[Dict[str, Any]] = None,
    http_status: Optional[int] = None
) -> ErrorResponse:
    """
    Crear respuesta de error.
    
    Args:
        code: Código de error
        details: Detalles adicionales (opcional)
        http_status: Status HTTP (opcional)
        
    Returns:
        ErrorResponse
    """
    message = get_error_message(code)
    
    # Determinar status HTTP si no se proporciona
    if http_status is None:
        if code.value.startswith("ERR_1"):
            http_status = 500
        elif code.value.startswith("ERR_2"):
            http_status = 400
        elif code.value.startswith("ERR_3"):
            http_status = 400
        elif code.value.startswith("ERR_4"):
            http_status = 503
        elif code.value.startswith("ERR_5"):
            http_status = 500
        elif code.value.startswith("ERR_6"):
            http_status = 401 if "EXPIRED" not in code.value else 403
        else:
            http_status = 500
    
    return ErrorResponse(code, message, details, http_status)

