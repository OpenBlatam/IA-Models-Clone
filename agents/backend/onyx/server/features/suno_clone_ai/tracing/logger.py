"""
Structured Logger - Logger estructurado
"""

import logging
import json
from typing import Any, Dict


class StructuredLogger:
    """Logger estructurado para logging consistente"""

    def __init__(self, name: str = "suno_clone_ai"):
        """Inicializa el logger estructurado"""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def _log(self, level: int, message: str, **kwargs) -> None:
        """Método interno de logging"""
        extra = {k: v for k, v in kwargs.items()}
        self.logger.log(level, message, extra=extra)

    def info(self, message: str, **kwargs) -> None:
        """Log de información"""
        self._log(logging.INFO, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log de error"""
        self._log(logging.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log de advertencia"""
        self._log(logging.WARNING, message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """Log de debug"""
        self._log(logging.DEBUG, message, **kwargs)

