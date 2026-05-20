"""
Memory Repository - Repositorio en memoria
==========================================

Repositorio simple en memoria para desarrollo y testing.
"""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from ..interfaces.repository import IProjectRepository
from ..core.base_repository import BaseRepository

logger = logging.getLogger(__name__)


class MemoryProjectRepository(BaseRepository, IProjectRepository):
    """Repositorio de proyectos en memoria"""
    
    def __init__(self):
        self._storage: Dict[str, Dict[str, Any]] = {}
    
    async def _get_by_id_impl(self, id: str) -> Optional[Dict[str, Any]]:
        """Implementación de get_by_id"""
        return self._storage.get(id)
    
    async def _list_impl(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Implementación de list"""
        projects = list(self._storage.values())
        
        # Aplicar filtros
        if filters:
            if "status" in filters:
                projects = [p for p in projects if p.get("status") == filters["status"]]
            if "author" in filters:
                projects = [p for p in projects if p.get("author") == filters["author"]]
            if "project_name" in filters:
                projects = [p for p in projects if p.get("project_name") == filters["project_name"]]
        
        # Paginación
        return projects[offset:offset + limit]
    
    async def _create_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación de create"""
        project_id = str(uuid.uuid4())
        project = {
            "project_id": project_id,
            "status": "queued",
            "created_at": datetime.now().isoformat(),
            **data
        }
        self._storage[project_id] = project
        return project
    
    async def _update_impl(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Implementación de update"""
        if id not in self._storage:
            return None
        
        self._storage[id].update(data)
        self._storage[id]["updated_at"] = datetime.now().isoformat()
        return self._storage[id]
    
    async def _delete_impl(self, id: str) -> bool:
        """Implementación de delete"""
        if id in self._storage:
            del self._storage[id]
            return True
        return False
    
    async def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene proyecto por nombre"""
        for project in self._storage.values():
            if project.get("project_name") == name:
                return project
        return None
    
    async def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por estado"""
        return [p for p in self._storage.values() if p.get("status") == status]
    
    async def get_by_author(self, author: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por autor"""
        return [p for p in self._storage.values() if p.get("author") == author]
    
    def clear(self):
        """Limpia el repositorio (útil para testing)"""
        self._storage.clear()

