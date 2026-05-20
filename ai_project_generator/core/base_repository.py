"""
Base Repository - Repositorio base
==================================

Clase base para repositorios con funcionalidad común.
"""

import logging
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from ..interfaces.repository import IRepository

logger = logging.getLogger(__name__)


class BaseRepository(IRepository, ABC):
    """
    Clase base para repositorios.
    
    Proporciona implementaciones comunes y helpers.
    """
    
    @abstractmethod
    async def _get_by_id_impl(self, id: str) -> Optional[Dict[str, Any]]:
        """Implementación específica de get_by_id"""
        pass
    
    @abstractmethod
    async def _list_impl(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Implementación específica de list"""
        pass
    
    @abstractmethod
    async def _create_impl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Implementación específica de create"""
        pass
    
    @abstractmethod
    async def _update_impl(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Implementación específica de update"""
        pass
    
    @abstractmethod
    async def _delete_impl(self, id: str) -> bool:
        """Implementación específica de delete"""
        pass
    
    async def get_by_id(self, id: str) -> Optional[Dict[str, Any]]:
        """Obtiene entidad por ID con validación"""
        if not id:
            return None
        return await self._get_by_id_impl(id)
    
    async def list(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lista entidades con validación"""
        # Validar y optimizar parámetros
        limit = max(1, min(limit, 1000))
        offset = max(0, offset)
        
        # Optimizar filtros
        if filters:
            filters = {k: v for k, v in filters.items() if v is not None}
        
        return await self._list_impl(filters, limit, offset)
    
    async def create(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Crea entidad con validación"""
        if not data:
            raise ValueError("Data cannot be empty")
        return await self._create_impl(data)
    
    async def update(
        self,
        id: str,
        data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Actualiza entidad con validación"""
        if not id:
            return None
        if not data:
            raise ValueError("Data cannot be empty")
        return await self._update_impl(id, data)
    
    async def delete(self, id: str) -> bool:
        """Elimina entidad con validación"""
        if not id:
            return False
        return await self._delete_impl(id)















