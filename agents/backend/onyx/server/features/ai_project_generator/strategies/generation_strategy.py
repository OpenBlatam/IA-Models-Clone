"""
Generation Strategy - Estrategias de generación
===============================================

Implementa el patrón Strategy para diferentes estrategias de generación.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GenerationStrategy(ABC):
    """Interfaz para estrategias de generación"""
    
    @abstractmethod
    async def generate(
        self,
        description: str,
        project_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera un proyecto"""
        pass


class SyncGenerationStrategy(GenerationStrategy):
    """Estrategia de generación síncrona"""
    
    def __init__(self, project_generator):
        self.project_generator = project_generator
    
    async def generate(
        self,
        description: str,
        project_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera proyecto de forma síncrona"""
        if not self.project_generator:
            raise ValueError("Project generator not available")
        
        return await self.project_generator.generate(
            description=description,
            project_name=project_name,
            **kwargs
        )


class AsyncGenerationStrategy(GenerationStrategy):
    """Estrategia de generación asíncrona con workers"""
    
    def __init__(self, worker_service):
        self.worker_service = worker_service
    
    async def generate(
        self,
        description: str,
        project_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera proyecto de forma asíncrona"""
        if not self.worker_service:
            raise ValueError("Worker service not available")
        
        task_id = self.worker_service.enqueue_task(
            self._generate_task,
            description=description,
            project_name=project_name,
            **kwargs
        )
        
        return {
            "task_id": task_id,
            "status": "queued",
            "message": "Generation task queued"
        }
    
    def _generate_task(self, description: str, project_name: Optional[str] = None, **kwargs):
        """Tarea que se ejecuta en el worker"""
        # Esta función se ejecuta en el worker
        # Necesita acceso al project_generator
        pass















