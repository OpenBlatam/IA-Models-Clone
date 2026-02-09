"""
Base Tool - Clase base para herramientas
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTool(ABC):
    """Clase base abstracta para herramientas"""

    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Ejecuta la herramienta"""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Obtiene la descripción de la herramienta"""
        pass

