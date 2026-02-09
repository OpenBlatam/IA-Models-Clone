"""
Structured Logging
Logging estructurado para mejor análisis y observabilidad
"""

import logging
import json
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Niveles de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StructuredFormatter(logging.Formatter):
    """Formateador de logs estructurados (JSON)"""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea un log record a JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Agregar información de excepción si existe
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Agregar campos extra
        if self.include_extra:
            for key, value in record.__dict__.items():
                if key not in [
                    "name", "msg", "args", "created", "filename",
                    "funcName", "levelname", "levelno", "lineno",
                    "module", "msecs", "message", "pathname", "process",
                    "processName", "relativeCreated", "thread", "threadName",
                    "exc_info", "exc_text", "stack_info"
                ]:
                    log_data[key] = value
        
        return json.dumps(log_data)


class CloudWatchFormatter(logging.Formatter):
    """Formateador optimizado para AWS CloudWatch"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Formatea para CloudWatch Logs"""
        log_data = {
            "timestamp": int(record.created * 1000),  # CloudWatch usa milisegundos
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
        }
        
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_structured_logging(
    level: str = "INFO",
    format_type: str = "json",  # json, cloudwatch, text
    output: str = "stdout"  # stdout, stderr, file
):
    """
    Configura logging estructurado
    
    Args:
        level: Nivel de logging
        format_type: Tipo de formato (json, cloudwatch, text)
        output: Destino de output
    """
    # Configurar root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remover handlers existentes
    root_logger.handlers.clear()
    
    # Configurar handler
    if output == "stdout":
        handler = logging.StreamHandler(sys.stdout)
    elif output == "stderr":
        handler = logging.StreamHandler(sys.stderr)
    else:
        handler = logging.FileHandler(output)
    
    # Configurar formatter
    if format_type == "json":
        formatter = StructuredFormatter()
    elif format_type == "cloudwatch":
        formatter = CloudWatchFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    
    logger.info("Structured logging configured", extra={
        "format_type": format_type,
        "output": output,
        "level": level
    })


def get_logger(name: str) -> logging.Logger:
    """Obtiene un logger con nombre"""
    return logging.getLogger(name)


def log_with_context(
    logger: logging.Logger,
    level: LogLevel,
    message: str,
    **context
):
    """
    Log con contexto adicional
    
    Args:
        logger: Logger a usar
        level: Nivel de log
        message: Mensaje
        **context: Contexto adicional
    """
    extra = {"context": context}
    
    if level == LogLevel.DEBUG:
        logger.debug(message, extra=extra)
    elif level == LogLevel.INFO:
        logger.info(message, extra=extra)
    elif level == LogLevel.WARNING:
        logger.warning(message, extra=extra)
    elif level == LogLevel.ERROR:
        logger.error(message, extra=extra)
    elif level == LogLevel.CRITICAL:
        logger.critical(message, extra=extra)
