"""
Base Service
===========

Clase base para servicios del sistema.
"""

import logging
from abc import ABC
from typing import Optional

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Clase base para servicios."""
    
    def __init__(self, logger_name: Optional[str] = None):
        """
        Inicializar servicio base.
        
        Args:
            logger_name: Nombre del logger (opcional)
        """
        self._logger = logging.getLogger(logger_name or self.__class__.__module__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message."""
        self._logger.info(message, extra=kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message."""
        self._logger.error(message, extra=kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message."""
        self._logger.warning(message, extra=kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message."""
        self._logger.debug(message, extra=kwargs)

