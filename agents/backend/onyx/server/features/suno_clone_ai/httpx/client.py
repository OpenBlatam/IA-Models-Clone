"""
Async HTTP Client - Cliente HTTP principal
"""

from typing import Any, Dict, Optional
import httpx


class AsyncHTTPClient:
    """Cliente HTTP asíncrono"""

    def __init__(self, timeout: float = 30.0):
        """Inicializa el cliente HTTP"""
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)

    async def get(self, url: str, **kwargs) -> Any:
        """Realiza una petición GET"""
        response = await self.client.get(url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Realiza una petición POST"""
        response = await self.client.post(url, json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    async def put(self, url: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """Realiza una petición PUT"""
        response = await self.client.put(url, json=data, **kwargs)
        response.raise_for_status()
        return response.json()

    async def delete(self, url: str, **kwargs) -> Any:
        """Realiza una petición DELETE"""
        response = await self.client.delete(url, **kwargs)
        response.raise_for_status()
        return response.json()

    async def close(self) -> None:
        """Cierra el cliente"""
        await self.client.aclose()

