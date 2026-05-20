"""
Logging Utilities
=================
"""

import logging
import sys
import structlog
from typing import Optional

from ..config.app_config import get_config

config = get_config()


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configurar logging estructurado.
    
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
    """
    level = log_level or ("DEBUG" if config.debug else "INFO")
    
    # Configurar structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configurar logging estándar
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper())
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Obtener logger estructurado.
    
    Args:
        name: Nombre del logger
        
    Returns:
        Logger estructurado
    """
    return structlog.get_logger(name)

