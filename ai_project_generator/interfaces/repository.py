"""
Repository Interfaces - Interfaces para repositorios
===================================================

Define contratos para repositorios siguiendo el patrón Repository.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class IRepository(ABC):
    """Interfaz base para repositorios"""
    
    @abstractmethod
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Obtiene entidad por ID"""
        pass
    
    @abstractmethod
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lista entidades con filtros"""
        pass
    
    @abstractmethod
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea nueva entidad"""
        pass
    
    @abstractmethod
    async def update(self, id: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Actualiza entidad"""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Elimina entidad"""
        pass


class IProjectRepository(IRepository):
    """Interfaz para repositorio de proyectos"""
    
    @abstractmethod
    async def get_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Obtiene proyecto por nombre"""
        pass
    
    @abstractmethod
    async def get_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por estado"""
        pass
    
    @abstractmethod
    async def get_by_author(self, author: str) -> List[Dict[str, Any]]:
        """Obtiene proyectos por autor"""
        pass















