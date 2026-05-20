"""
Cache Interface - Interfaz para servicios de cache
==================================================

Define contrato para servicios de cache.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any


class ICacheService(ABC):
    """Interfaz para servicio de cache"""
    
    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Obtiene valor del cache"""
        pass
    
    @abstractmethod
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """Establece valor en cache"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Elimina clave del cache"""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Verifica si clave existe"""
        pass
    
    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """Elimina claves que coinciden con patrón"""
        pass















