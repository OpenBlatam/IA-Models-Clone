"""
Project Service - Servicio de gestión de proyectos
==================================================

Servicio independiente para gestión de proyectos siguiendo principios de microservicios.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from ..interfaces.repository import IProjectRepository
from ..interfaces.cache import ICacheService
from ..interfaces.events import IEventPublisher
from ..core.project_generator import ProjectGenerator
from ..core.continuous_generator import ContinuousGenerator

logger = logging.getLogger(__name__)


class ProjectService:
    """
    Servicio para gestión de proyectos.
    
    Responsabilidades:
    - Crear proyectos
    - Obtener información de proyectos
    - Gestionar estado de proyectos
    - Interactuar con generador continuo
    """
    
    def __init__(
        self,
        repository: Optional[IProjectRepository] = None,
        project_generator: Optional[ProjectGenerator] = None,
        continuous_generator: Optional[ContinuousGenerator] = None,
        cache_service: Optional[ICacheService] = None,
        event_publisher: Optional[IEventPublisher] = None
    ):
        self.repository = repository
        self.project_generator = project_generator
        self.continuous_generator = continuous_generator
        self.cache_service = cache_service
        self.event_publisher = event_publisher
    
    async def create_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        author: str = "Blatam Academy",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Crea un nuevo proyecto.
        
        Args:
            description: Descripción del proyecto
            project_name: Nombre del proyecto
            author: Autor del proyecto
            **kwargs: Parámetros adicionales
        
        Returns:
            Información del proyecto creado
        """
        try:
            # Publicar evento
            if self.event_publisher:
                await self.event_publisher.publish("project.creating", {
                    "description": description,
                    "project_name": project_name,
                    "author": author
                })
            
            # Crear proyecto usando repositorio
            if self.repository:
                project_data = {
                    "description": description,
                    "project_name": project_name,
                    "author": author,
                    **kwargs
                }
                project = await self.repository.create(project_data)
                
                # Publicar evento
                if self.event_publisher:
                    await self.event_publisher.publish("project.queued", {
                        "project_id": project["project_id"],
                        "status": "queued"
                    })
                
                return {
                    "project_id": project["project_id"],
                    "status": "queued",
                    "message": "Project added to queue"
                }
            else:
                # Generación inmediata si no hay repositorio
                result = await self._generate_immediately(
                    description, project_name, author, **kwargs
                )
                return result
                
        except Exception as e:
            logger.error(f"Error creating project: {e}", exc_info=True)
            if self.event_publisher:
                await self.event_publisher.publish("project.failed", {
                    "error": str(e)
                })
            raise
    
    async def _generate_immediately(
        self,
        description: str,
        project_name: Optional[str],
        author: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera proyecto inmediatamente"""
        if not self.project_generator:
            raise ValueError("Project generator not available")
        
        result = await self.project_generator.generate(
            description=description,
            project_name=project_name,
            author=author,
            **kwargs
        )
        
        # Publicar evento
        if self.event_publisher:
            await self.event_publisher.publish("project.completed", {
                "project_id": result.get("project_id"),
                "status": "completed"
            })
        
        return result
    
    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene información de un proyecto.
        
        Args:
            project_id: ID del proyecto
        
        Returns:
            Información del proyecto o None si no existe
        """
        # Intentar obtener de cache
        if self.cache_service:
            cached = await self.cache_service.get(f"project:{project_id}")
            if cached:
                return cached
        
        # Obtener del repositorio
        if self.repository:
            project_info = await self.repository.get_by_id(project_id)
            if project_info:
                # Guardar en cache
                if self.cache_service:
                    await self.cache_service.set(
                        f"project:{project_id}",
                        project_info,
                        ttl=3600
                    )
                return project_info
        
        return None
    
    async def list_projects(
        self,
        status: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Lista proyectos con filtros.
        
        Args:
            status: Filtrar por estado
            author: Filtrar por autor
            limit: Límite de resultados
            offset: Offset para paginación
        
        Returns:
            Lista de proyectos
        """
        if self.repository:
            filters = {}
            if status:
                filters["status"] = status
            if author:
                filters["author"] = author
            return await self.repository.list(filters=filters, limit=limit, offset=offset)
        return []
    
    async def delete_project(self, project_id: str) -> bool:
        """
        Elimina un proyecto.
        
        Args:
            project_id: ID del proyecto
        
        Returns:
            True si se eliminó exitosamente
        """
        try:
            if self.repository:
                success = await self.repository.delete(project_id)
                if success:
                    # Eliminar de cache
                    if self.cache_service:
                        await self.cache_service.delete(f"project:{project_id}")
                    
                    # Publicar evento
                    if self.event_publisher:
                        await self.event_publisher.publish("project.deleted", {
                            "project_id": project_id
                        })
                    
                    return True
            return False
        except Exception as e:
            logger.error(f"Error deleting project {project_id}: {e}")
            return False
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Obtiene estado de la cola"""
        if self.continuous_generator:
            return {
                "queue_size": len(self.continuous_generator.queue),
                "processing": self.continuous_generator.is_processing,
                "total_processed": self.continuous_generator.stats.get("total_processed", 0),
                "total_failed": self.continuous_generator.stats.get("total_failed", 0)
            }
        return {"queue_size": 0, "processing": False}

