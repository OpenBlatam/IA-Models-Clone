"""
Error Handler - Manejador de Errores
=====================================

Sistema centralizado de manejo de errores.
"""

import logging
from typing import Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCode(Enum):
    """Códigos de error"""
    # Posts
    POST_NOT_FOUND = ("POST_001", "Post no encontrado")
    POST_INVALID_CONTENT = ("POST_002", "Contenido inválido")
    POST_PLATFORM_ERROR = ("POST_003", "Error en plataforma")
    
    # Memes
    MEME_NOT_FOUND = ("MEME_001", "Meme no encontrado")
    MEME_INVALID_IMAGE = ("MEME_002", "Imagen inválida")
    
    # Platforms
    PLATFORM_NOT_CONNECTED = ("PLATFORM_001", "Plataforma no conectada")
    PLATFORM_AUTH_ERROR = ("PLATFORM_002", "Error de autenticación")
    
    # General
    VALIDATION_ERROR = ("GEN_001", "Error de validación")
    INTERNAL_ERROR = ("GEN_002", "Error interno del servidor")
    RATE_LIMIT_EXCEEDED = ("GEN_003", "Límite de tasa excedido")
    
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message


class AppException(Exception):
    """Excepción base de la aplicación"""
    
    def __init__(
        self,
        error_code: ErrorCode,
        details: Optional[str] = None,
        status_code: int = 400
    ):
        self.error_code = error_code
        self.details = details
        self.status_code = status_code
        super().__init__(self.error_code.message)


class ErrorHandler:
    """Manejador de errores"""
    
    @staticmethod
    def handle_error(
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Manejar error y retornar respuesta estructurada
        
        Args:
            error: Excepción
            context: Contexto adicional
            
        Returns:
            Dict con información del error
        """
        if isinstance(error, AppException):
            response = {
                "error": {
                    "code": error.error_code.code,
                    "message": error.error_code.message,
                    "details": error.details,
                    "status_code": error.status_code
                }
            }
            
            if context:
                response["error"]["context"] = context
            
            logger.warning(
                f"Error manejado: {error.error_code.code} - {error.error_code.message}"
            )
            
            return response
        
        # Error no manejado
        logger.error(f"Error no manejado: {error}", exc_info=True)
        
        return {
            "error": {
                "code": ErrorCode.INTERNAL_ERROR.code,
                "message": ErrorCode.INTERNAL_ERROR.message,
                "details": str(error),
                "status_code": 500
            }
        }
    
    @staticmethod
    def raise_validation_error(message: str, details: Optional[str] = None):
        """Lanzar error de validación"""
        raise AppException(
            ErrorCode.VALIDATION_ERROR,
            details=details or message,
            status_code=400
        )
    
    @staticmethod
    def raise_not_found(resource: str, resource_id: str):
        """Lanzar error de recurso no encontrado"""
        if resource == "post":
            raise AppException(
                ErrorCode.POST_NOT_FOUND,
                details=f"Post {resource_id} no encontrado",
                status_code=404
            )
        elif resource == "meme":
            raise AppException(
                ErrorCode.MEME_NOT_FOUND,
                details=f"Meme {resource_id} no encontrado",
                status_code=404
            )
        else:
            raise AppException(
                ErrorCode.INTERNAL_ERROR,
                details=f"Recurso {resource} {resource_id} no encontrado",
                status_code=404
            )
    
    @staticmethod
    def raise_platform_error(platform: str, details: Optional[str] = None):
        """Lanzar error de plataforma"""
        raise AppException(
            ErrorCode.POST_PLATFORM_ERROR,
            details=details or f"Error en plataforma {platform}",
            status_code=500
        )
    
    @staticmethod
    def raise_rate_limit_error(platform: Optional[str] = None):
        """Lanzar error de rate limit"""
        message = "Límite de tasa excedido"
        if platform:
            message += f" para {platform}"
        raise AppException(
            ErrorCode.RATE_LIMIT_EXCEEDED,
            details=message,
            status_code=429
        )
    
    @staticmethod
    def raise_auth_error(platform: str, details: Optional[str] = None):
        """Lanzar error de autenticación"""
        raise AppException(
            ErrorCode.PLATFORM_AUTH_ERROR,
            details=details or f"Error de autenticación en {platform}",
            status_code=401
        )



