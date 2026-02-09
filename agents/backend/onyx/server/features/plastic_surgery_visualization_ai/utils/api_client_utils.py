"""API client utilities for making HTTP requests."""

from typing import Any, Dict, Optional, List
import asyncio
import aiohttp
from datetime import datetime, timedelta


class APIClient:
    """HTTP API client with retry and error handling."""
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        headers: Optional[Dict[str, str]] = None
    ):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.default_headers = headers or {}
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=self.timeout,
            headers=self.default_headers
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> aiohttp.ClientResponse:
        """Make HTTP request with retry."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.request(method, url, **kwargs) as response:
                    return response
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == self.max_retries - 1:
                    raise
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        
        raise Exception("Max retries exceeded")
    
    async def get(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """GET request."""
        async with self._request('GET', endpoint, **kwargs) as response:
            return await response.json()
    
    async def post(self, endpoint: str, data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """POST request."""
        async with self._request('POST', endpoint, json=data, **kwargs) as response:
            return await response.json()
    
    async def put(self, endpoint: str, data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """PUT request."""
        async with self._request('PUT', endpoint, json=data, **kwargs) as response:
            return await response.json()
    
    async def delete(self, endpoint: str, **kwargs) -> Dict[str, Any]:
        """DELETE request."""
        async with self._request('DELETE', endpoint, **kwargs) as response:
            return await response.json()
    
    async def patch(self, endpoint: str, data: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
        """PATCH request."""
        async with self._request('PATCH', endpoint, json=data, **kwargs) as response:
            return await response.json()


class CachedAPIClient(APIClient):
    """API client with response caching."""
    
    def __init__(self, *args, cache_ttl: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, tuple] = {}
    
    def _get_cache_key(self, method: str, endpoint: str, **kwargs) -> str:
        """Generate cache key."""
        import hashlib
        key_data = f"{method}:{endpoint}:{str(kwargs)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cached(self, key: str) -> bool:
        """Check if response is cached."""
        if key not in self.cache:
            return False
        
        data, timestamp = self.cache[key]
        return (datetime.utcnow() - timestamp).total_seconds() < self.cache_ttl
    
    async def get(self, endpoint: str, use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """GET request with caching."""
        if use_cache:
            cache_key = self._get_cache_key('GET', endpoint, **kwargs)
            if self._is_cached(cache_key):
                return self.cache[cache_key][0]
        
        result = await super().get(endpoint, **kwargs)
        
        if use_cache:
            self.cache[cache_key] = (result, datetime.utcnow())
        
        return result
    
    def clear_cache(self):
        """Clear cache."""
        self.cache.clear()


class BatchAPIClient:
    """Client for batch API requests."""
    
    def __init__(self, client: APIClient, max_concurrent: int = 10):
        self.client = client
        self.max_concurrent = max_concurrent
    
    async def batch_get(
        self,
        endpoints: List[str],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch GET requests."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch(endpoint: str):
            async with semaphore:
                return await self.client.get(endpoint, **kwargs)
        
        tasks = [fetch(endpoint) for endpoint in endpoints]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_post(
        self,
        requests: List[tuple],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Batch POST requests."""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def fetch(endpoint: str, data: Any):
            async with semaphore:
                return await self.client.post(endpoint, data=data, **kwargs)
        
        tasks = [fetch(endpoint, data) for endpoint, data in requests]
        return await asyncio.gather(*tasks, return_exceptions=True)


def create_client(
    base_url: str,
    api_key: Optional[str] = None,
    **kwargs
) -> APIClient:
    """Create API client with authentication."""
    headers = kwargs.pop('headers', {})
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    return APIClient(base_url, headers=headers, **kwargs)


