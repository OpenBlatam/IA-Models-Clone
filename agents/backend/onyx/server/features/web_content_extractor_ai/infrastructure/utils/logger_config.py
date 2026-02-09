"""
Configuración de logging estructurado
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_type: str = "standard") -> None:
    """
    Configurar logging estructurado
    
    Args:
        level: Nivel de logging (DEBUG, INFO, WARNING, ERROR)
        format_type: Tipo de formato (standard, json)
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    if format_type == "json":
        try:
            import json_log_formatter
            formatter = json_log_formatter.JSONFormatter()
        except ImportError:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        )
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(log_level)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Evitar duplicar handlers
    if not root_logger.handlers:
        root_logger.handlers = [handler]
    
    # Configurar loggers específicos para reducir ruido
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

