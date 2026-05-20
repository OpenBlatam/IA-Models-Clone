"""
Logging Configuration - Configuración de Logging
=================================================

Configuración centralizada de logging estructurado para el proyecto.
"""

import logging
import sys
from typing import Optional, Dict, Any
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    log_file: Optional[str] = None,
    enable_structured: bool = True
) -> logging.Logger:
    """
    Configurar logging estructurado para la aplicación.
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Tipo de formato (json, text, colored)
        log_file: Archivo opcional para escribir logs
        enable_structured: Habilitar logging estructurado con structlog
        
    Returns:
        Logger configurado
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if enable_structured:
        try:
            import structlog
            
            processors = [
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
            ]
            
            if format_type == "json":
                processors.append(structlog.processors.JSONRenderer())
            elif format_type == "colored":
                processors.append(structlog.dev.ConsoleRenderer())
            else:
                processors.append(structlog.dev.ConsoleRenderer())
            
            structlog.configure(
                processors=processors,
                wrapper_class=structlog.stdlib.BoundLogger,
                context_class=dict,
                logger_factory=structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )
            
            logger = structlog.get_logger()
            
        except ImportError:
            enable_structured = False
    
    if not enable_structured:
        formatter = logging.Formatter(
            fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        
        logger = logging.getLogger()
        logger.setLevel(log_level)
        logger.addHandler(handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        if enable_structured:
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Obtener logger para un módulo específico.
    
    Args:
        name: Nombre del módulo
        
    Returns:
        Logger configurado
    """
    try:
        import structlog
        return structlog.get_logger(name)
    except ImportError:
        return logging.getLogger(name)


class LogContext:
    """
    Context manager para agregar contexto a logs.
    """
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_context = {}
    
    def __enter__(self):
        try:
            import structlog
            if isinstance(self.logger, structlog.BoundLogger):
                self.old_context = self.logger._context.copy()
                self.logger = self.logger.bind(**self.context)
        except (ImportError, AttributeError):
            pass
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            import structlog
            if isinstance(self.logger, structlog.BoundLogger):
                self.logger._context.clear()
                self.logger._context.update(self.old_context)
        except (ImportError, AttributeError):
            pass

