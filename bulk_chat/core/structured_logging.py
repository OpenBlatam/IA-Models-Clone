"""
Structured Logging - Sistema de Logging Estructurado
====================================================

Sistema avanzado de logging estructurado con contexto.
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """Formateador de logs estructurado."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatear log record como JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Agregar contexto si existe
        if hasattr(record, "context"):
            log_data["context"] = record.context
        
        # Agregar excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data, ensure_ascii=False)


class ContextualLogger:
    """Logger con contexto."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log con contexto."""
        extra = kwargs.get("extra", {})
        extra["context"] = self.context
        kwargs["extra"] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log debug."""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log info."""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log warning."""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log error."""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log critical."""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = True,
):
    """
    Configurar logging estructurado.
    
    Args:
        log_level: Nivel de logging
        log_file: Archivo de log (opcional)
        json_format: Usar formato JSON
    """
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remover handlers existentes
    root_logger.handlers.clear()
    
    # Handler para stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # Handler para archivo si se especifica
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if json_format:
            file_handler.setFormatter(StructuredFormatter())
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            file_handler.setFormatter(formatter)
        
        root_logger.addHandler(file_handler)
    
    return root_logger


def get_contextual_logger(name: str, context: Dict[str, Any]) -> ContextualLogger:
    """Obtener logger con contexto."""
    logger = logging.getLogger(name)
    return ContextualLogger(logger, context)
































