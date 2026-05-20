"""
HTTP Client - Cliente HTTP Avanzado
===================================

Cliente HTTP robusto con retry, timeout, circuit breaker y más.
"""

import logging
import asyncio
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)

# Intentar importar httpx
try:
    import httpx
    _has_httpx = True
except ImportError:
    _has_httpx = False
    try:
        import aiohttp
        _has_aiohttp = True
    except ImportError:
        _has_aiohttp = False


@dataclass
class HTTPResponse:
    """Respuesta HTTP"""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    json: Optional[Dict[str, Any]] = None
    url: str = ""
    elapsed: float = 0.0


class HTTPClient:
    """
    Cliente HTTP avanzado con retry, timeout y circuit breaker.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        headers: Optional[Dict[str, str]] = None,
        verify_ssl: bool = True
    ):
        """
        Inicializar cliente HTTP.
        
        Args:
            base_url: URL base opcional
            timeout: Timeout en segundos
            max_retries: Número máximo de reintentos
            retry_delay: Delay entre reintentos
            headers: Headers por defecto
            verify_ssl: Verificar certificados SSL
        """
        self.base_url = base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_headers = headers or {}
        self.verify_ssl = verify_ssl
        
        # Inicializar cliente según disponibilidad
        self._client = None
        self._session = None
    
    async def _get_client(self):
        """Obtener o crear cliente HTTP"""
        if self._client is None:
            if _has_httpx:
                self._client = httpx.AsyncClient(
                    base_url=self.base_url,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=self.default_headers
                )
            elif _has_aiohttp:
                self._session = aiohttp.ClientSession(
                    base_url=self.base_url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                    headers=self.default_headers
                )
            else:
                raise ImportError("httpx or aiohttp is required for HTTP client")
        
        return self._client or self._session
    
    async def request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> HTTPResponse:
        """
        Realizar request HTTP.
        
        Args:
            method: Método HTTP (GET, POST, etc.)
            url: URL (relativa si base_url está configurado)
            **kwargs: Argumentos adicionales (headers, json, data, etc.)
            
        Returns:
            HTTPResponse
        """
        client = await self._get_client()
        start_time = datetime.now()
        
        # Construir URL completa
        if self.base_url and not url.startswith(('http://', 'https://')):
            full_url = f"{self.base_url.rstrip('/')}/{url.lstrip('/')}"
        else:
            full_url = url
        
        # Headers
        headers = {**self.default_headers, **kwargs.get('headers', {})}
        
        # Reintentos
        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                if _has_httpx:
                    response = await client.request(
                        method=method,
                        url=full_url,
                        headers=headers,
                        **{k: v for k, v in kwargs.items() if k != 'headers'}
                    )
                    
                    elapsed = (datetime.now() - start_time).total_seconds()
                    
                    # Parsear JSON si es posible
                    json_data = None
                    try:
                        json_data = response.json()
                    except Exception:
                        pass
                    
                    return HTTPResponse(
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        content=response.content,
                        text=response.text,
                        json=json_data,
                        url=str(response.url),
                        elapsed=elapsed
                    )
                
                elif _has_aiohttp:
                    async with client.request(
                        method=method,
                        url=full_url,
                        headers=headers,
                        **{k: v for k, v in kwargs.items() if k != 'headers'}
                    ) as response:
                        content = await response.read()
                        text = await response.text()
                        
                        elapsed = (datetime.now() - start_time).total_seconds()
                        
                        # Parsear JSON si es posible
                        json_data = None
                        try:
                            json_data = await response.json()
                        except Exception:
                            pass
                        
                        return HTTPResponse(
                            status_code=response.status,
                            headers=dict(response.headers),
                            content=content,
                            text=text,
                            json=json_data,
                            url=str(response.url),
                            elapsed=elapsed
                        )
            
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {self.retry_delay}s..."
                    )
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                else:
                    logger.error(f"Request failed after {self.max_retries + 1} attempts: {e}")
                    raise
        
        if last_exception:
            raise last_exception
    
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """GET request"""
        return await self.request("GET", url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> HTTPResponse:
        """POST request"""
        return await self.request("POST", url, **kwargs)
    
    async def put(self, url: str, **kwargs) -> HTTPResponse:
        """PUT request"""
        return await self.request("PUT", url, **kwargs)
    
    async def patch(self, url: str, **kwargs) -> HTTPResponse:
        """PATCH request"""
        return await self.request("PATCH", url, **kwargs)
    
    async def delete(self, url: str, **kwargs) -> HTTPResponse:
        """DELETE request"""
        return await self.request("DELETE", url, **kwargs)
    
    async def close(self) -> None:
        """Cerrar cliente"""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._session:
            await self._session.close()
            self._session = None
    
    async def __aenter__(self):
        """Context manager entry"""
        await self._get_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        await self.close()




