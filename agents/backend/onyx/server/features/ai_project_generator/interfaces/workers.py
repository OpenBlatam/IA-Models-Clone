"""
Worker Interface - Interfaz para workers
=========================================

Define contrato para servicios de workers asíncronos.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable


class IWorkerService(ABC):
    """Interfaz para servicio de workers"""
    
    @abstractmethod
    def enqueue_task(
        self,
        task_func: Callable,
        *args,
        **kwargs
    ) -> Optional[str]:
        """Encola una tarea para ejecución asíncrona"""
        pass
    
    @abstractmethod
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una tarea"""
        pass















