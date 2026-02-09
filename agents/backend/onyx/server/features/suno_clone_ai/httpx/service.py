"""
HTTP Service - Servicio HTTP
"""

from typing import Any, Dict, Optional
from .client import AsyncHTTPClient
from .retry import RetryHandler


class HTTPService:
    """Servicio para realizar peticiones HTTP"""

    def __init__(self, retry_handler: Optional[RetryHandler] = None):
        """Inicializa el servicio HTTP"""
        self.client = AsyncHTTPClient()
        self.retry_handler = retry_handler or RetryHandler()

    async def get(self, url: str, retry: bool = True, **kwargs) -> Any:
        """Realiza una petición GET con opción de retry"""
        if retry:
            return await self.retry_handler.execute(self.client.get, url, **kwargs)
        return await self.client.get(url, **kwargs)

    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, retry: bool = True, **kwargs) -> Any:
        """Realiza una petición POST con opción de retry"""
        if retry:
            return await self.retry_handler.execute(self.client.post, url, data=data, **kwargs)
        return await self.client.post(url, data=data, **kwargs)

