"""
Generation Service - Servicio de generación de proyectos
========================================================

Servicio independiente para generación de proyectos de IA.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from ..core.project_generator import ProjectGenerator
from ..infrastructure.workers import WorkerService
from ..infrastructure.events import EventPublisher
from ..infrastructure.cache import CacheService

logger = logging.getLogger(__name__)


class GenerationService:
    """
    Servicio para generación de proyectos.
    
    Responsabilidades:
    - Generar proyectos de IA
    - Gestionar generación asíncrona
    - Optimizar generación
    """
    
    def __init__(
        self,
        project_generator: Optional[ProjectGenerator] = None,
        worker_service: Optional[WorkerService] = None,
        event_publisher: Optional[EventPublisher] = None,
        cache_service: Optional[CacheService] = None
    ):
        self.project_generator = project_generator
        self.worker_service = worker_service
        self.event_publisher = event_publisher
        self.cache_service = cache_service
    
    async def generate_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        author: str = "Blatam Academy",
        async_generation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Genera un proyecto.
        
        Args:
            description: Descripción del proyecto
            project_name: Nombre del proyecto
            author: Autor
            async_generation: Si usar generación asíncrona
            **kwargs: Parámetros adicionales
        
        Returns:
            Información del proyecto generado
        """
        if async_generation and self.worker_service:
            # Generación asíncrona con workers
            task_id = await self.worker_service.enqueue_generation_task(
                description=description,
                project_name=project_name,
                author=author,
                **kwargs
            )
            
            return {
                "task_id": task_id,
                "status": "queued",
                "message": "Generation task queued"
            }
        else:
            # Generación síncrona
            return await self._generate_sync(
                description, project_name, author, **kwargs
            )
    
    async def _generate_sync(
        self,
        description: str,
        project_name: Optional[str],
        author: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Generación síncrona"""
        if not self.project_generator:
            raise ValueError("Project generator not available")
        
        start_time = datetime.now()
        
        try:
            # Publicar evento de inicio
            if self.event_publisher:
                await self.event_publisher.publish("generation.started", {
                    "description": description,
                    "project_name": project_name
                })
            
            # Generar proyecto
            result = await self.project_generator.generate(
                description=description,
                project_name=project_name,
                author=author,
                **kwargs
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Publicar evento de completado
            if self.event_publisher:
                await self.event_publisher.publish("generation.completed", {
                    "project_id": result.get("project_id"),
                    "duration": duration
                })
            
            # Guardar en cache
            if self.cache_service and result.get("project_id"):
                await self.cache_service.set(
                    f"generation:{result['project_id']}",
                    result,
                    ttl=86400  # 24 horas
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating project: {e}", exc_info=True)
            
            # Publicar evento de error
            if self.event_publisher:
                await self.event_publisher.publish("generation.failed", {
                    "error": str(e),
                    "description": description
                })
            
            raise
    
    async def get_generation_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de una generación"""
        if self.worker_service:
            return await self.worker_service.get_task_status(task_id)
        return None
    
    async def batch_generate(
        self,
        projects: list,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Genera múltiples proyectos en batch.
        
        Args:
            projects: Lista de proyectos a generar
            parallel: Si generar en paralelo
        
        Returns:
            Resultados de generación
        """
        results = []
        errors = []
        
        if parallel and self.worker_service:
            # Generación paralela con workers
            task_ids = []
            for project in projects:
                task_id = await self.worker_service.enqueue_generation_task(**project)
                task_ids.append(task_id)
            
            return {
                "task_ids": task_ids,
                "status": "queued",
                "message": f"{len(task_ids)} tasks queued"
            }
        else:
            # Generación secuencial
            for project in projects:
                try:
                    result = await self._generate_sync(**project)
                    results.append(result)
                except Exception as e:
                    errors.append({
                        "project": project.get("project_name", "unknown"),
                        "error": str(e)
                    })
            
            return {
                "results": results,
                "errors": errors,
                "total": len(projects),
                "successful": len(results),
                "failed": len(errors)
            }















