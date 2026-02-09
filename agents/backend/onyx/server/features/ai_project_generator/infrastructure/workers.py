"""
Worker Service - Servicio de workers asíncronos
================================================

Servicio que abstrae los workers asíncronos (Celery, RQ, ARQ).
"""

import logging
from typing import Optional, Dict, Any, Callable
from functools import wraps

from ..core.async_workers import get_worker_manager
from ..core.microservices_config import get_microservices_config, WorkerBackend

logger = logging.getLogger(__name__)


class WorkerService:
    """
    Servicio de workers asíncronos.
    
    Abstrae el backend de workers utilizado para ejecutar
    tareas en background.
    """
    
    def __init__(self):
        config = get_microservices_config()
        if config.worker_backend.value != "none":
            try:
                self.worker_manager = get_worker_manager()
            except Exception as e:
                logger.warning(f"Worker manager not available: {e}")
                self.worker_manager = None
        else:
            self.worker_manager = None
    
    def enqueue_task(
        self,
        task_func: Callable,
        *args,
        **kwargs
    ) -> Optional[str]:
        """
        Encola una tarea para ejecución asíncrona.
        
        Args:
            task_func: Función a ejecutar
            *args: Argumentos posicionales
            **kwargs: Argumentos keyword
        
        Returns:
            Task ID o None si falla
        """
        if not self.worker_manager:
            logger.warning("Worker manager not available")
            return None
        
        try:
            return self.worker_manager.enqueue_task(task_func, *args, **kwargs)
        except Exception as e:
            logger.error(f"Error enqueueing task: {e}")
            return None
    
    async def enqueue_generation_task(
        self,
        description: str,
        project_name: Optional[str] = None,
        author: str = "Blatam Academy",
        **kwargs
    ) -> Optional[str]:
        """
        Encola una tarea de generación de proyecto.
        
        Args:
            description: Descripción del proyecto
            project_name: Nombre del proyecto
            author: Autor
            **kwargs: Parámetros adicionales
        
        Returns:
            Task ID
        """
        from ..services.generation_service import GenerationService
        
        # Importar función de generación
        def generate_task():
            # Esta función se ejecutará en el worker
            # Necesita acceso a los servicios
            pass
        
        return self.enqueue_task(
            generate_task,
            description=description,
            project_name=project_name,
            author=author,
            **kwargs
        )
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene estado de una tarea.
        
        Args:
            task_id: ID de la tarea
        
        Returns:
            Estado de la tarea o None
        """
        if not self.worker_manager:
            return None
        
        try:
            return self.worker_manager.get_task_result(task_id)
        except Exception as e:
            logger.error(f"Error getting task status: {e}")
            return None
    
    async def get_task_status_async(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Versión async de get_task_status"""
        return self.get_task_status(task_id)


def get_worker_service() -> WorkerService:
    """Obtiene instancia de worker service"""
    return WorkerService()















