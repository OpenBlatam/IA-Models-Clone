"""
Service Interfaces - Interfaces para servicios
==============================================

Define contratos para servicios de negocio.
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any


class IService(ABC):
    """Interfaz base para servicios"""
    pass


class IProjectService(IService):
    """Interfaz para servicio de proyectos"""
    
    @abstractmethod
    async def create_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Crea un nuevo proyecto"""
        pass
    
    @abstractmethod
    async def get_project(self, project_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un proyecto"""
        pass
    
    @abstractmethod
    async def list_projects(
        self,
        status: Optional[str] = None,
        author: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lista proyectos con filtros"""
        pass
    
    @abstractmethod
    async def delete_project(self, project_id: str) -> bool:
        """Elimina un proyecto"""
        pass


class IGenerationService(IService):
    """Interfaz para servicio de generación"""
    
    @abstractmethod
    async def generate_project(
        self,
        description: str,
        project_name: Optional[str] = None,
        async_generation: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Genera un proyecto"""
        pass
    
    @abstractmethod
    async def get_generation_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene estado de generación"""
        pass
    
    @abstractmethod
    async def batch_generate(
        self,
        projects: List[Dict[str, Any]],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Genera múltiples proyectos"""
        pass















