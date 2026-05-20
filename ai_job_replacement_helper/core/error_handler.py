"""
Error Handler - Manejo avanzado de errores
===========================================

Sistema centralizado de manejo de errores.
"""

import logging
from typing import Optional, Dict, Any
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from datetime import datetime

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Manejador de errores"""
    
    def __init__(self):
        """Inicializar manejador"""
        self.error_logs: list = []
        logger.info("ErrorHandler initialized")
    
    def handle_exception(self, request: Request, exc: Exception) -> JSONResponse:
        """Manejar excepción"""
        error_id = f"err_{int(datetime.now().timestamp())}"
        
        # Log error
        logger.error(
            f"Error {error_id}: {type(exc).__name__}: {str(exc)} "
            f"at {request.url.path}"
        )
        
        # Guardar en logs
        self.error_logs.append({
            "error_id": error_id,
            "type": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Retornar respuesta de error
        if isinstance(exc, HTTPException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": True,
                    "error_id": error_id,
                    "message": exc.detail,
                    "status_code": exc.status_code,
                }
            )
        
        # Error genérico
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "error_id": error_id,
                "message": "Internal server error",
                "status_code": 500,
            }
        )
    
    def get_error_logs(self, limit: int = 100) -> list:
        """Obtener logs de errores"""
        return self.error_logs[-limit:]


# Instancia global
error_handler = ErrorHandler()




