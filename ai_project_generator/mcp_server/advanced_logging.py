"""
MCP Advanced Logging - Logging avanzado
=========================================
"""

import logging
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Niveles de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredLogger:
    """
    Logger estructurado
    
    Proporciona logging estructurado con metadata.
    """
    
    def __init__(
        self,
        name: str,
        log_file: Optional[str] = None,
        level: LogLevel = LogLevel.INFO,
    ):
        """
        Args:
            name: Nombre del logger
            log_file: Archivo de log (opcional)
            level: Nivel de log
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))
        
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
    
    def log(
        self,
        level: LogLevel,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Registra un log estructurado
        
        Args:
            level: Nivel de log
            message: Mensaje
            metadata: Metadata adicional
            **kwargs: Campos adicionales
        """
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level.value,
            "message": message,
            "logger": self.name,
            **(metadata or {}),
            **kwargs,
        }
        
        log_message = json.dumps(log_data)
        
        log_method = getattr(self.logger, level.value.lower())
        log_method(log_message)
    
    def debug(self, message: str, **kwargs):
        """Log debug"""
        self.log(LogLevel.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info"""
        self.log(LogLevel.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning"""
        self.log(LogLevel.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error"""
        self.log(LogLevel.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical"""
        self.log(LogLevel.CRITICAL, message, **kwargs)


class RequestLogger:
    """
    Logger de requests
    
    Registra requests y responses con metadata completa.
    """
    
    def __init__(self, structured_logger: StructuredLogger):
        """
        Args:
            structured_logger: Logger estructurado
        """
        self.logger = structured_logger
    
    def log_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Any] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        """
        Registra un request
        
        Args:
            method: Método HTTP
            path: Path del request
            headers: Headers (opcional)
            params: Parámetros (opcional)
            body: Body (opcional)
            user_id: ID del usuario (opcional)
            ip_address: Dirección IP (opcional)
        """
        self.logger.info(
            "Request received",
            metadata={
                "type": "request",
                "method": method,
                "path": path,
                "headers": headers,
                "params": params,
                "body": str(body)[:500] if body else None,  # Limitar tamaño
                "user_id": user_id,
                "ip_address": ip_address,
            }
        )
    
    def log_response(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        user_id: Optional[str] = None,
        error: Optional[str] = None,
    ):
        """
        Registra un response
        
        Args:
            method: Método HTTP
            path: Path del request
            status_code: Código de estado
            duration_ms: Duración en milisegundos
            user_id: ID del usuario (opcional)
            error: Error si hubo (opcional)
        """
        level = LogLevel.ERROR if status_code >= 400 else LogLevel.INFO
        
        self.logger.log(
            level,
            "Request completed",
            metadata={
                "type": "response",
                "method": method,
                "path": path,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "user_id": user_id,
                "error": error,
            }
        )

