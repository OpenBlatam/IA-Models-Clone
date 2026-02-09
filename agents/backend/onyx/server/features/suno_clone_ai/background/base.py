"""
Base Task - Clase base para tareas
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseTask(ABC):
    """Clase base abstracta para tareas en background"""

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """Ejecuta la tarea"""
        pass

