"""
Project Repository - Repositorio de proyectos
============================================

Implementación del repositorio de proyectos usando el generador continuo.
"""

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..interfaces.repository import IProjectRepository
from ..core.base_repository import BaseRepository
from ..core.continuous_generator import ContinuousGenerator
from ..core.exceptions import RepositoryError

logger = logging.getLogger(__name__)


class ProjectRepository(BaseRepository, IProjectRepository):
    """Repositorio de proyectos usando ContinuousGenerator"""
    
    def __init__(self, continuous_generator: Optional[ContinuousGenerator] = None):
        self.continuous_generator = continuous_generator
    
    async def _get_by_id_impl(self, id: str) -> Optional[Dict[str, Any]]:
        """Implementación de get_by_id"""
        if not self.continuous_generator:
            raise RepositoryError("Continuous generator not available")
        
        return self.continuous_generator.get_project_info(id)
    
    async def _list_impl(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Implementación de list"""
        if not self.continuous_generator:
            return []
        
        status = filters.get("status") if filters else None
        author = filters.get("author") if filters else None
        
        return self.continuous_generator.list_projects(
            status=status,
            author=author,
            limit=limit,
            offset=offset
        )
    
    async def _create_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación de create"""
        if not self.continuous_generator:
            raise RepositoryError("Continuous generator not available")
        
        project_id = self.continuous_generator.add_to_queue(**data)
        return {
            "project_id": project_id,
            "status": "queued",
            **data
        }
    
    async def _update_impl(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Implementación de update"""
        # Los proyectos en cola no se actualizan directamente
        project = await self._get_by_id_impl(id)
        if project:
            project.update(data)
            return project
        return None
    
    async def _delete_impl(self, id: str) -> bool:
        """Implementación de delete"""
        if not self.continuous_generator:
            return False
        
        return self.continuous_generator.remove_from_queue(id)
    
    async def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene proyecto por nombre"""
        projects = await self.list(filters={"project_name": name}, limit=1)
        return projects[0] if projects else None
    
    async def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por estado"""
        return await self.list(filters={"status": status})
    
    async def get_by_author(self, author: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por autor"""
        return await self.list(filters={"author": author})

