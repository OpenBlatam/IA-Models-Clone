"""
Base Tracer - Clase base para tracing
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseTracer(ABC):
    """Clase base abstracta para tracing"""

    @abstractmethod
    def trace(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Traza una operación"""
        pass

    @abstractmethod
    def span(self, name: str) -> Any:
        """Crea un span de tracing"""
        pass

