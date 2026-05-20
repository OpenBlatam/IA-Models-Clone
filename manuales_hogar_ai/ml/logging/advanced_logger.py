"""
Advanced Logger
==============

Logging avanzado con contexto y estructura.
"""

import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

logger = logging.getLogger(__name__)


class AdvancedLogger:
    """Logger avanzado con contexto."""
    
    def __init__(
        self,
        name: str,
        level: int = logging.INFO,
        include_context: bool = True
    ):
        """
        Inicializar logger avanzado.
        
        Args:
            name: Nombre del logger
            level: Nivel de logging
            include_context: Incluir contexto
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.include_context = include_context
        self.context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs):
        """Establecer contexto."""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Limpiar contexto."""
        self.context.clear()
    
    def _format_message(
        self,
        message: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Formatear mensaje con contexto."""
        formatted = {
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        if self.include_context and self.context:
            formatted["context"] = self.context.copy()
        
        if extra:
            formatted["extra"] = extra
        
        return formatted
    
    def info(self, message: str, **kwargs):
        """Log info."""
        formatted = self._format_message(message, kwargs)
        self.logger.info(json.dumps(formatted))
    
    def error(
        self,
        message: str,
        exception: Optional[Exception] = None,
        **kwargs
    ):
        """Log error."""
        formatted = self._format_message(message, kwargs)
        
        if exception:
            formatted["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        self.logger.error(json.dumps(formatted))
    
    def warning(self, message: str, **kwargs):
        """Log warning."""
        formatted = self._format_message(message, kwargs)
        self.logger.warning(json.dumps(formatted))
    
    def debug(self, message: str, **kwargs):
        """Log debug."""
        formatted = self._format_message(message, kwargs)
        self.logger.debug(json.dumps(formatted))
    
    def performance(
        self,
        operation: str,
        duration: float,
        **kwargs
    ):
        """Log de rendimiento."""
        formatted = self._format_message(
            f"Performance: {operation}",
            {**kwargs, "duration_seconds": duration}
        )
        self.logger.info(json.dumps(formatted))




