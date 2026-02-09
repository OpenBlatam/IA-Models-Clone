"""
Base Redis - Clase base para Redis
"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseRedis(ABC):
    """Clase base abstracta para operaciones Redis"""

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Obtiene un valor por clave"""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Establece un valor con opcional TTL"""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Elimina una clave"""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Verifica si una clave existe"""
        pass

