"""
Refactored Services - Servicios refactorizados
==============================================

Servicios refactorizados usando BaseService y mejores prácticas.
"""

import logging
import time
from typing import Optional, Dict, Any, List

from .base_service import BaseService
from .exceptions import ProjectNotFoundError, ProjectGenerationError
from .utils import generate_id, sanitize_dict
from .validators import ProjectNameValidator, DescriptionValidator
from .decorators import timed, logged, cached
from ..interfaces.repository import IProjectRepository
from ..interfaces.workers import IWorkerService
from ..core.project_generator import ProjectGenerator

logger = logging.getLogger(__name__)


class RefactoredProjectService(BaseService):
    """
    Servicio de proyectos refactorizado.
    
    Usa BaseService para funcionalidad común.
    """
    
    def __init__(
        self,
        repository: IProjectRepository,
        cache_service=None,
        event_publisher=None
    ):
        super().__init__(
            cache_service=cache_service,
            event_publisher=event_publisher,
            service_name="ProjectService"
        )
        self.repository = repository
    
    @timed
    @logged
    async def create_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        author: str = "Blatam Academy",
        **kwargs
    ) -> Dict[str, Any]:
        """Crea proyecto con validación y eventos"""
        start_time = time.time()
        
        try:
            # Validar
            DescriptionValidator.validate(description)
            if project_name:
                project_name = ProjectNameValidator.validate(project_name)
            
            # Publicar evento
            await self._publish_event("project.creating", {
                "description": description,
                "project_name": project_name,
                "author": author
            })
            
            # Crear proyecto
            project_data = sanitize_dict({
                "description": description,
                "project_name": project_name,
                "author": author,
                **kwargs
            })
            
            project = await self.repository.create(project_data)
            
            # Cachear
            if project.get("project_id"):
                await self._set_to_cache(
                    f"project:{project['project_id']}",
                    project,
                    ttl=3600
                )
            
            # Publicar evento
            await self._publish_event("project.created", {
                "project_id": project.get("project_id"),
                "status": "queued"
            })
            
            duration = time.time() - start_time
            self._log_service_call("create_project", duration, True)
            
            return project
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_service_call("create_project", duration, False, error=str(e))
            await self._publish_event("project.failed", {"error": str(e)})
            raise ProjectGenerationError(f"Failed to create project: {e}") from e
    
    @cached(ttl=3600, tags=["projects"])
    @timed
    async def get_project(self, project_id: str) -> Dict[str, Any]:
        """Obtiene proyecto con cache"""
        if not project_id:
            raise ValueError("project_id is required")
        
        # Intentar cache
        cached = await self._get_from_cache(f"project:{project_id}")
        if cached:
            return cached
        
        # Obtener del repositorio
        project = await self.repository.get_by_id(project_id)
        if not project:
            raise ProjectNotFoundError(f"Project {project_id} not found")
        
        # Cachear
        await self._set_to_cache(f"project:{project_id}", project, ttl=3600)
        
        return project
    
    @timed
    async def list_projects(
        self,
        status: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lista proyectos con filtros optimizados"""
        filters = sanitize_dict({
            "status": status,
            "author": author
        })
        
        return await self.repository.list(filters=filters, limit=limit, offset=offset)
    
    @timed
    @logged
    async def delete_project(self, project_id: str) -> bool:
        """Elimina proyecto con eventos"""
        if not project_id:
            return False
        
        success = await self.repository.delete(project_id)
        
        if success:
            # Eliminar de cache
            await self._delete_from_cache(f"project:{project_id}")
            
            # Publicar evento
            await self._publish_event("project.deleted", {
                "project_id": project_id
            })
        
        return success


class RefactoredGenerationService(BaseService):
    """Servicio de generación refactorizado"""
    
    def __init__(
        self,
        project_generator: Optional[ProjectGenerator] = None,
        worker_service=None,
        cache_service=None,
        event_publisher=None
    ):
        super().__init__(
            cache_service=cache_service,
            event_publisher=event_publisher,
            service_name="GenerationService"
        )
        self.project_generator = project_generator
        self.worker_service = worker_service
    
    @timed
    @logged
    async def generate_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        async_generation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera proyecto con validación"""
        # Validar
        DescriptionValidator.validate(description)
        if project_name:
            project_name = ProjectNameValidator.validate(project_name)
        
        # Publicar evento
        await self._publish_event("generation.started", {
            "description": description,
            "project_name": project_name
        })
        
        if async_generation and self.worker_service:
            # Generación asíncrona
            task_id = self.worker_service.enqueue_task(
                self._generate_task,
                description=description,
                project_name=project_name,
                **kwargs
            )
            return {
                "task_id": task_id,
                "status": "queued"
            }
        else:
            # Generación síncrona
            return await self._generate_sync(description, project_name, **kwargs)
    
    async def _generate_sync(
        self,
        description: str,
        project_name: Optional[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Generación síncrona"""
        if not self.project_generator:
            raise ProjectGenerationError("Project generator not available")
        
        start_time = time.time()
        
        try:
            result = await self.project_generator.generate(
                description=description,
                project_name=project_name,
                **kwargs
            )
            
            duration = time.time() - start_time
            self._log_service_call("generate_project", duration, True)
            
            await self._publish_event("generation.completed", {
                "project_id": result.get("project_id"),
                "duration": duration
            })
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            self._log_service_call("generate_project", duration, False, error=str(e))
            await self._publish_event("generation.failed", {"error": str(e)})
            raise ProjectGenerationError(f"Generation failed: {e}") from e
    
    def _generate_task(self, description: str, project_name: Optional[str] = None, **kwargs):
        """Tarea para worker"""
        # Implementación para worker
        pass















