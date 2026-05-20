"""
HTTP Client Module
Async HTTP client wrapper using httpx with resilience.
"""

import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Optional imports
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not available. Using requests as fallback.")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class HttpResponse:
    """HTTP response wrapper."""
    status_code: int
    content: bytes
    text: str
    json_data: Optional[Dict[str, Any]]
    headers: Dict[str, str]
    elapsed_ms: float
    success: bool


class AsyncHttpClient:
    """
    Async HTTP client with retry and timeout support.
    Uses httpx for async requests, falls back to requests.
    """
    
    def __init__(
        self,
        base_url: str = "",
        timeout: float = 30.0,
        max_retries: int = 3,
        headers: Dict[str, str] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_headers = headers or {}
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"AsyncHttpClient initialized. Base URL: {base_url or 'none'}")
    
    async def _get_client(self) -> "httpx.AsyncClient":
        """Get or create async client."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx required for async HTTP requests")
        
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=self.default_headers
            )
        return self._client
    
    async def close(self) -> None:
        """Close the client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    def _build_url(self, path: str) -> str:
        """Build full URL from path."""
        if path.startswith(('http://', 'https://')):
            return path
        return f"{self.base_url}/{path.lstrip('/')}" if self.base_url else path
    
    async def request(
        self,
        method: str,
        path: str,
        params: Dict[str, Any] = None,
        json: Dict[str, Any] = None,
        data: Any = None,
        headers: Dict[str, str] = None
    ) -> HttpResponse:
        """Make HTTP request with retry logic."""
        url = self._build_url(path)
        merged_headers = {**self.default_headers, **(headers or {})}
        
        start_time = datetime.now()
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if HTTPX_AVAILABLE:
                    client = await self._get_client()
                    response = await client.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json,
                        data=data,
                        headers=merged_headers
                    )
                    
                    elapsed = (datetime.now() - start_time).total_seconds() * 1000
                    
                    try:
                        json_data = response.json()
                    except:
                        json_data = None
                    
                    return HttpResponse(
                        status_code=response.status_code,
                        content=response.content,
                        text=response.text,
                        json_data=json_data,
                        headers=dict(response.headers),
                        elapsed_ms=elapsed,
                        success=200 <= response.status_code < 300
                    )
                
                elif REQUESTS_AVAILABLE:
                    # Sync fallback
                    response = requests.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json,
                        data=data,
                        headers=merged_headers,
                        timeout=self.timeout
                    )
                    
                    elapsed = (datetime.now() - start_time).total_seconds() * 1000
                    
                    try:
                        json_data = response.json()
                    except:
                        json_data = None
                    
                    return HttpResponse(
                        status_code=response.status_code,
                        content=response.content,
                        text=response.text,
                        json_data=json_data,
                        headers=dict(response.headers),
                        elapsed_ms=elapsed,
                        success=200 <= response.status_code < 300
                    )
                else:
                    raise ImportError("No HTTP library available")
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {url}: {e}")
        
        raise last_error or Exception("Request failed")
    
    async def get(self, path: str, **kwargs) -> HttpResponse:
        return await self.request("GET", path, **kwargs)
    
    async def post(self, path: str, **kwargs) -> HttpResponse:
        return await self.request("POST", path, **kwargs)
    
    async def put(self, path: str, **kwargs) -> HttpResponse:
        return await self.request("PUT", path, **kwargs)
    
    async def delete(self, path: str, **kwargs) -> HttpResponse:
        return await self.request("DELETE", path, **kwargs)
    
    async def patch(self, path: str, **kwargs) -> HttpResponse:
        return await self.request("PATCH", path, **kwargs)
