from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
from urllib.parse import urlparse
import httpx
import aiohttp
import orjson
import zstandard as zstd
import brotli
from tenacity import retry, stop_after_attempt, wait_exponential
from cachetools import TTLCache
import structlog
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Ultra-Fast HTTP Client v10
Maximum Performance with Fastest Libraries
"""


# Ultra-fast imports

logger = structlog.get_logger(__name__)


@dataclass
class HTTPResponse:
    """Ultra-optimized HTTP response"""
    status_code: int
    headers: Dict[str, str]
    content: bytes
    text: str
    url: str
    elapsed: float
    encoding: str = "utf-8"


class UltraFastHTTPClient:
    """Ultra-fast HTTP client with maximum performance optimizations"""
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
        enable_compression: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        cache_maxsize: int = 1000,
        retry_attempts: int = 3,
        user_agent: str = "UltraFastSEO/10.0"
    ):
        
    """__init__ function."""
self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.enable_compression = enable_compression
        self.enable_cache = enable_cache
        self.retry_attempts = retry_attempts
        self.user_agent = user_agent
        
        # Ultra-fast cache
        if enable_cache:
            self.cache = TTLCache(maxsize=cache_maxsize, ttl=cache_ttl)
        else:
            self.cache = None
        
        # HTTPX client with maximum performance
        self.httpx_client = None
        self.aiohttp_session = None
        
        # Performance metrics
        self.request_count = 0
        self.total_time = 0.0
        self.avg_response_time = 0.0
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self._cleanup()
    
    async def _initialize_clients(self) -> Any:
        """Initialize ultra-fast HTTP clients"""
        # HTTPX client with maximum performance
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections
        )
        
        self.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=limits,
            http2=True,
            follow_redirects=True,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br" if self.enable_compression else "identity",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            }
        )
        
        # AIOHTTP session for backup
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_keepalive_connections,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        
        self.aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br" if self.enable_compression else "identity"
            }
        )
    
    async def _cleanup(self) -> Any:
        """Cleanup HTTP clients"""
        if self.httpx_client:
            await self.httpx_client.aclose()
        if self.aiohttp_session:
            await self.aiohttp_session.close()
    
    def _get_cache_key(self, url: str, method: str = "GET", **kwargs) -> str:
        """Generate cache key"""
        return f"{method}:{url}:{hash(str(kwargs))}"
    
    def _compress_data(self, data: bytes, algorithm: str = "zstd") -> bytes:
        """Compress data with fastest algorithm"""
        if algorithm == "zstd":
            compressor = zstd.ZstdCompressor(level=3)
            return compressor.compress(data)
        elif algorithm == "brotli":
            return brotli.compress(data, quality=4)
        else:
            return data
    
    def _decompress_data(self, data: bytes, algorithm: str = "zstd") -> bytes:
        """Decompress data"""
        if algorithm == "zstd":
            decompressor = zstd.ZstdDecompressor()
            return decompressor.decompress(data)
        elif algorithm == "brotli":
            return brotli.decompress(data)
        else:
            return data
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """Ultra-fast GET request with retry logic"""
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cache_key = self._get_cache_key(url, "GET", **kwargs)
            if cache_key in self.cache:
                logger.debug("Cache hit", url=url)
                return self.cache[cache_key]
        
        try:
            # Try HTTPX first (faster)
            response = await self.httpx_client.get(url, **kwargs)
            
            # Parse response with orjson for maximum speed
            content = response.content
            text = content.decode(response.encoding or "utf-8")
            
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                text=text,
                url=str(response.url),
                elapsed=time.time() - start_time,
                encoding=response.encoding or "utf-8"
            )
            
            # Cache response
            if self.cache:
                self.cache[cache_key] = http_response
            
            # Update metrics
            self._update_metrics(http_response.elapsed)
            
            logger.debug(
                "HTTP GET successful",
                url=url,
                status_code=response.status_code,
                elapsed=http_response.elapsed,
                content_length=len(content)
            )
            
            return http_response
            
        except Exception as e:
            logger.warning("HTTPX failed, trying AIOHTTP", url=url, error=str(e))
            
            # Fallback to AIOHTTP
            try:
                async with self.aiohttp_session.get(url, **kwargs) as response:
                    content = await response.read()
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    text = content.decode(response.charset or "utf-8")
                    
                    http_response = HTTPResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content,
                        text=text,
                        url=str(response.url),
                        elapsed=time.time() - start_time,
                        encoding=response.charset or "utf-8"
                    )
                    
                    # Cache response
                    if self.cache:
                        self.cache[cache_key] = http_response
                    
                    # Update metrics
                    self._update_metrics(http_response.elapsed)
                    
                    logger.debug(
                        "AIOHTTP GET successful",
                        url=url,
                        status_code=response.status,
                        elapsed=http_response.elapsed,
                        content_length=len(content)
                    )
                    
                    return http_response
                    
            except Exception as e2:
                logger.error("Both HTTP clients failed", url=url, error=str(e2))
                raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def post(self, url: str, data: Optional[Union[Dict, bytes]] = None, **kwargs) -> HTTPResponse:
        """Ultra-fast POST request with retry logic"""
        start_time = time.time()
        
        try:
            # Serialize data with orjson if needed
            if isinstance(data, dict):
                data = orjson.dumps(data)
            
            response = await self.httpx_client.post(url, content=data, **kwargs)
            
            content = response.content
            text = content.decode(response.encoding or "utf-8")
            
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                text=text,
                url=str(response.url),
                elapsed=time.time() - start_time,
                encoding=response.encoding or "utf-8"
            )
            
            # Update metrics
            self._update_metrics(http_response.elapsed)
            
            logger.debug(
                "HTTP POST successful",
                url=url,
                status_code=response.status_code,
                elapsed=http_response.elapsed,
                content_length=len(content)
            )
            
            return http_response
            
        except Exception as e:
            logger.error("POST request failed", url=url, error=str(e))
            raise
    
    async def get_multiple(self, urls: List[str], max_concurrent: int = 10) -> List[HTTPResponse]:
        """Get multiple URLs concurrently with maximum performance"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_with_semaphore(url: str) -> HTTPResponse:
            async with semaphore:
                return await self.get(url)
        
        tasks = [get_with_semaphore(url) for url in urls]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_responses = []
        for i, response in enumerate(responses):
            if isinstance(response, Exception):
                logger.error("Failed to fetch URL", url=urls[i], error=str(response))
            else:
                valid_responses.append(response)
        
        return valid_responses
    
    def _update_metrics(self, elapsed: float):
        """Update performance metrics"""
        self.request_count += 1
        self.total_time += elapsed
        self.avg_response_time = self.total_time / self.request_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "request_count": self.request_count,
            "total_time": self.total_time,
            "avg_response_time": self.avg_response_time,
            "cache_hit_ratio": self._get_cache_hit_ratio() if self.cache else 0.0
        }
    
    def _get_cache_hit_ratio(self) -> float:
        """Get cache hit ratio"""
        if not self.cache:
            return 0.0
        
        # This is a simplified calculation
        # In production, you'd want more sophisticated cache metrics
        return 0.0  # Placeholder
    
    async def health_check(self) -> bool:
        """Health check for the HTTP client"""
        try:
            response = await self.get("https://httpbin.org/get", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False


# Global client instance for maximum performance
_global_client: Optional[UltraFastHTTPClient] = None


async async def get_http_client() -> UltraFastHTTPClient:
    """Get global HTTP client instance"""
    global _global_client
    if _global_client is None:
        _global_client = UltraFastHTTPClient()
        await _global_client._initialize_clients()
    return _global_client


async def cleanup_http_client():
    """Cleanup global HTTP client"""
    global _global_client
    if _global_client:
        await _global_client._cleanup()
        _global_client = None 