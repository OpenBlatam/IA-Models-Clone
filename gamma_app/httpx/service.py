"""
HTTPX Service Implementation
"""

from typing import Dict, Any, Optional
import logging
import httpx as httpx_lib

from .base import HTTPClientBase, HTTPRequest, HTTPResponse, HTTPMethod

logger = logging.getLogger(__name__)


class HTTPXService(HTTPClientBase):
    """HTTPX service implementation"""
    
    def __init__(self, config_service=None, tracing_service=None):
        """Initialize HTTPX service"""
        self.config_service = config_service
        self.tracing_service = tracing_service
        self._client: Optional[httpx_lib.AsyncClient] = None
    
    async def _get_client(self) -> httpx_lib.AsyncClient:
        """Get or create HTTP client"""
        if self._client is None:
            self._client = httpx_lib.AsyncClient(
                timeout=30,
                follow_redirects=True
            )
        return self._client
    
    async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Make HTTP request"""
        try:
            client = await self._get_client()
            
            response = await client.request(
                method=request.method.value,
                url=request.url,
                headers=request.headers,
                params=request.params,
                json=request.json,
                content=request.data,
                timeout=request.timeout
            )
            
            return HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=response.content,
                json=response.json() if response.headers.get("content-type", "").startswith("application/json") else None
            )
            
        except Exception as e:
            logger.error(f"Error making HTTP request: {e}")
            raise
    
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """GET request"""
        request = HTTPRequest(
            method=HTTPMethod.GET,
            url=url,
            **kwargs
        )
        return await self.request(request)
    
    async def post(self, url: str, **kwargs) -> HTTPResponse:
        """POST request"""
        request = HTTPRequest(
            method=HTTPMethod.POST,
            url=url,
            **kwargs
        )
        return await self.request(request)
    
    async def close(self):
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

