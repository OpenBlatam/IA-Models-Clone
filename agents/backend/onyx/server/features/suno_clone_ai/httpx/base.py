"""
Base HTTP Client - Clase base para cliente HTTP
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseHTTPClient(ABC):
    """Clase base abstracta para clientes HTTP"""

    @abstractmethod
    async def get(self, url: str, **kwargs) -> Any:
        """Realiza una petición GET"""
        pass

    @abstractmethod
    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Realiza una petición POST"""
        pass

    @abstractmethod
    async def put(self, url: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Realiza una petición PUT"""
        pass

    @abstractmethod
    async def delete(self, url: str, **kwargs) -> Any:
        """Realiza una petición DELETE"""
        pass

