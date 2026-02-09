"""
Base Agent - Clase base para agentes
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAgent(ABC):
    """Clase base abstracta para agentes de IA"""

    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Ejecuta una tarea"""
        pass

    @abstractmethod
    def get_capabilities(self) -> list:
        """Obtiene las capacidades del agente"""
        pass

