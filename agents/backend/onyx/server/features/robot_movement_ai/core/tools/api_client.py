"""
API Client System
=================

Cliente HTTP avanzado para APIs externas.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)


@dataclass
class APIRequest:
    """Solicitud API."""
    method: str
    url: str
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    json: Optional[Dict[str, Any]] = None
    timeout: float = 30.0


@dataclass
class APIResponse:
    """Respuesta API."""
    status_code: int
    headers: Dict[str, str]
    json: Optional[Dict[str, Any]] = None
    text: Optional[str] = None
    request: Optional[APIRequest] = None


class APIClient:
    """
    Cliente HTTP avanzado.
    
    Cliente para hacer solicitudes HTTP con retry, rate limiting, etc.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
        max_retries: int = 3
    ):
        """
        Inicializar cliente API.
        
        Args:
            base_url: URL base
            default_headers: Headers por defecto
            timeout: Timeout por defecto
            max_retries: Máximo de reintentos
        """
        self.base_url = base_url
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.max_retries = max_retries
        self.client: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        """Entrar al contexto."""
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self.default_headers,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Salir del contexto."""
        if self.client:
            await self.client.aclose()
    
    async def request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> APIResponse:
        """
        Hacer solicitud HTTP.
        
        Args:
            method: Método HTTP
            url: URL
            headers: Headers adicionales
            params: Parámetros de query
            json: Datos JSON
            timeout: Timeout
            
        Returns:
            Respuesta API
        """
        if not self.client:
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self.default_headers,
                timeout=timeout or self.timeout
            )
        
        request = APIRequest(
            method=method,
            url=url,
            headers=headers,
            params=params,
            json=json,
            timeout=timeout or self.timeout
        )
        
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                response = await self.client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json,
                    timeout=timeout or self.timeout
                )
                
                # Intentar parsear JSON
                json_data = None
                try:
                    json_data = response.json()
                except Exception:
                    pass
                
                return APIResponse(
                    status_code=response.status_code,
                    headers=dict(response.headers),
                    json=json_data,
                    text=response.text,
                    request=request
                )
            
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    logger.warning(f"Request failed, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise last_exception
    
    async def get(
        self,
        url: str,
        **kwargs
    ) -> APIResponse:
        """GET request."""
        return await self.request("GET", url, **kwargs)
    
    async def post(
        self,
        url: str,
        **kwargs
    ) -> APIResponse:
        """POST request."""
        return await self.request("POST", url, **kwargs)
    
    async def put(
        self,
        url: str,
        **kwargs
    ) -> APIResponse:
        """PUT request."""
        return await self.request("PUT", url, **kwargs)
    
    async def delete(
        self,
        url: str,
        **kwargs
    ) -> APIResponse:
        """DELETE request."""
        return await self.request("DELETE", url, **kwargs)


# Instancia global
_api_client: Optional[APIClient] = None


def get_api_client(
    base_url: Optional[str] = None,
    default_headers: Optional[Dict[str, str]] = None
) -> APIClient:
    """Obtener instancia global del cliente API."""
    global _api_client
    if _api_client is None:
        _api_client = APIClient(base_url=base_url, default_headers=default_headers)
    return _api_client






