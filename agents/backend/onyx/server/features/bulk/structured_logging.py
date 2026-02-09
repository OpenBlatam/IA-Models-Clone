"""
Structured Logging para BUL API
================================
Logging estructurado en formato JSON
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """Formatter que produce logs en formato JSON."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea un log record como JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if hasattr(record, 'user_id'):
            log_data['user_id'] = record.user_id
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
        if hasattr(record, 'request_id'):
            log_data['request_id'] = record.request_id
        if hasattr(record, 'duration'):
            log_data['duration_ms'] = record.duration * 1000
        if hasattr(record, 'status_code'):
            log_data['status_code'] = record.status_code
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        return json.dumps(log_data, ensure_ascii=False)


def setup_structured_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Configura logging estructurado.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        log_file: Archivo para escribir logs (opcional)
        console_output: Si escribir a consola
        
    Returns:
        Logger configurado
    """
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    logger.handlers.clear()
    
    formatter = JSONFormatter()
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_with_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **kwargs
):
    """
    Log con contexto adicional.
    
    Args:
        logger: Logger a usar
        level: Nivel de log
        message: Mensaje
        **kwargs: Campos adicionales para el log
    """
    extra = kwargs.copy()
    log_method = getattr(logger, level.lower())
    log_method(message, extra=extra)


if __name__ == "__main__":
    logger = setup_structured_logging(
        log_level="INFO",
        log_file="logs/app.json",
        console_output=True
    )
    
    logger.info("Aplicación iniciada")
    logger.warning("Esta es una advertencia")
    
    log_with_context(
        logger,
        "info",
        "Documento generado",
        user_id="user123",
        task_id="task456",
        document_type="strategy",
        duration=2.5
    )
    
    try:
        raise ValueError("Error de ejemplo")
    except Exception as e:
        logger.error("Error procesando documento", exc_info=True, extra={
            'task_id': 'task789',
            'error_type': type(e).__name__
        })
































