from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
BUFFER_SIZE = 1024

import asyncio
import time
import hashlib
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
        import heapq
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Ultra-Fast HTTP Client v12
Maximum Performance with Latest Optimizations
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
    compressed: bool = False
    compression_ratio: float = 1.0
    protocol: str = "http/1.1"


class UltraFastHTTPClientV12:
    """Ultra-fast HTTP client with maximum performance optimizations v12"""
    
    def __init__(
        self,
        timeout: float = 30.0,
        max_connections: int = 500,
        max_keepalive_connections: int = 100,
        enable_compression: bool = True,
        enable_cache: bool = True,
        cache_ttl: int = 300,
        cache_maxsize: int = 100000,
        retry_attempts: int = 3,
        user_agent: str = "UltraFastSEO/12.0",
        http2_enabled: bool = True,
        connection_pool_size: int = 200,
        enable_stats: bool = True
    ):
        
    """__init__ function."""
self.timeout = timeout
        self.max_connections = max_connections
        self.max_keepalive_connections = max_keepalive_connections
        self.enable_compression = enable_compression
        self.enable_cache = enable_cache
        self.retry_attempts = retry_attempts
        self.user_agent = user_agent
        self.http2_enabled = http2_enabled
        self.connection_pool_size = connection_pool_size
        self.enable_stats = enable_stats
        
        # Ultra-fast cache with increased size
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
        self.cache_hits = 0
        self.cache_misses = 0
        self.http2_requests = 0
        self.http1_requests = 0
        
        # Advanced features
        self._connection_pool = {}
        self._request_stats = {}
    
    async def __aenter__(self) -> Any:
        """Async context manager entry"""
        await self._initialize_clients()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit"""
        await self._cleanup()
    
    async def _initialize_clients(self) -> Any:
        """Initialize ultra-fast HTTP clients with latest optimizations"""
        # HTTPX client with maximum performance
        limits = httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            max_requests=2000
        )
        
        # Advanced HTTPX configuration
        self.httpx_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=3.0,
                read=self.timeout,
                write=5.0,
                pool=30.0
            ),
            limits=limits,
            http2=self.http2_enabled,
            follow_redirects=True,
            max_redirects=10,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br, zstd" if self.enable_compression else "identity",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache"
            },
            transport=httpx.AsyncHTTPTransport(
                retries=1,
                verify=True,
                http2=self.http2_enabled,
                pool_limits=httpx.PoolLimits(
                    max_keepalive_connections=self.max_keepalive_connections,
                    max_connections=self.max_connections
                )
            )
        )
        
        # AIOHTTP session for backup with advanced configuration
        connector = aiohttp.TCPConnector(
            limit=self.max_connections,
            limit_per_host=self.max_keepalive_connections,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
            force_close=False,
            enable_hook_proxy=False,
            use_dns_cache=True,
            ttl_dns_cache=300
        )
        
        timeout_config = aiohttp.ClientTimeout(
            total=self.timeout,
            connect=3.0,
            sock_read=30.0,
            sock_connect=3.0
        )
        
        self.aiohttp_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout_config,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br, zstd" if self.enable_compression else "identity"
            },
            skip_auto_headers=["User-Agent"]
        )
    
    async def _cleanup(self) -> Any:
        """Cleanup HTTP clients"""
        if self.httpx_client:
            await self.httpx_client.aclose()
        if self.aiohttp_session:
            await self.aiohttp_session.close()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key with maximum performance"""
        # Use orjson for fast serialization
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_bytes = orjson.dumps(key_data, sort_keys=True, option=orjson.OPT_SERIALIZE_NUMPY)
        return hashlib.sha256(key_bytes).hexdigest()
    
    def _compress_data(self, data: bytes, algorithm: str = "zstd") -> bytes:
        """Compress data with fastest algorithm and advanced options"""
        if algorithm == "zstd":
            compressor = zstd.ZstdCompressor(
                level=3,
                threads=0,  # Use all available threads
                write_content_size=True,
                write_checksum=True,
                strategy=zstd.STRATEGY_FAST
            )
            return compressor.compress(data)
        elif algorithm == "brotli":
            return brotli.compress(
                data, 
                quality=4, 
                lgwin=22,
                lgblock=24,
                mode=brotli.MODE_GENERIC
            )
        else:
            return data
    
    def _decompress_data(self, data: bytes, algorithm: str = "zstd") -> bytes:
        """Decompress data with error handling"""
        try:
            if algorithm == "zstd":
                decompressor = zstd.ZstdDecompressor()
                return decompressor.decompress(data)
            elif algorithm == "brotli":
                return brotli.decompress(data)
            else:
                return data
        except Exception as e:
            logger.error("Decompression failed", algorithm=algorithm, error=str(e))
            return data
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value with maximum performance"""
        return orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_NAIVE_UTC)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value with maximum performance"""
        return orjson.loads(data)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """Ultra-fast GET request with retry logic and advanced optimizations"""
        start_time = time.time()
        
        # Check cache first
        if self.cache:
            cache_key = self._generate_key(url, "GET", **kwargs)
            if cache_key in self.cache:
                self.cache_hits += 1
                cached_response = self.cache[cache_key]
                self._update_metrics(time.time() - start_time)
                logger.debug("Cache hit", url=url, cache_hits=self.cache_hits)
                return cached_response
        
        try:
            # Try HTTPX first (faster)
            response = await self.httpx_client.get(url, **kwargs)
            
            # Parse response with orjson for maximum speed
            content = response.content
            text = content.decode(response.encoding or "utf-8")
            
            # Calculate compression ratio
            original_size = len(content)
            compressed_size = original_size
            compression_ratio = 1.0
            
            if self.enable_compression and original_size > 1024:
                compressed = self._compress_data(content, "zstd")
                compressed_size = len(compressed)
                compression_ratio = compressed_size / original_size
            
            # Determine protocol
            protocol = "http/2" if hasattr(response, 'http_version') and response.http_version == "HTTP/2" else "http/1.1"
            if protocol == "http/2":
                self.http2_requests += 1
            else:
                self.http1_requests += 1
            
            http_response = HTTPResponse(
                status_code=response.status_code,
                headers=dict(response.headers),
                content=content,
                text=text,
                url=str(response.url),
                elapsed=time.time() - start_time,
                encoding=response.encoding or "utf-8",
                compressed=compression_ratio < 1.0,
                compression_ratio=compression_ratio,
                protocol=protocol
            )
            
            # Cache response
            if self.cache:
                self.cache[cache_key] = http_response
            
            # Update metrics
            self._update_metrics(http_response.elapsed)
            self.cache_misses += 1
            
            logger.debug(
                "HTTP GET successful",
                url=url,
                status_code=response.status_code,
                elapsed=http_response.elapsed,
                content_length=len(content),
                compression_ratio=compression_ratio,
                protocol=protocol
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
                    
                    # Calculate compression ratio
                    original_size = len(content)
                    compressed_size = original_size
                    compression_ratio = 1.0
                    
                    if self.enable_compression and original_size > 1024:
                        compressed = self._compress_data(content, "zstd")
                        compressed_size = len(compressed)
                        compression_ratio = compressed_size / original_size
                    
                    http_response = HTTPResponse(
                        status_code=response.status,
                        headers=dict(response.headers),
                        content=content,
                        text=text,
                        url=str(response.url),
                        elapsed=time.time() - start_time,
                        encoding=response.charset or "utf-8",
                        compressed=compression_ratio < 1.0,
                        compression_ratio=compression_ratio,
                        protocol="http/1.1"
                    )
                    
                    # Cache response
                    if self.cache:
                        self.cache[cache_key] = http_response
                    
                    # Update metrics
                    self._update_metrics(http_response.elapsed)
                    self.cache_misses += 1
                    self.http1_requests += 1
                    
                    logger.debug(
                        "AIOHTTP GET successful",
                        url=url,
                        status_code=response.status,
                        elapsed=http_response.elapsed,
                        content_length=len(content),
                        compression_ratio=compression_ratio
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
        """Ultra-fast POST request with retry logic and advanced optimizations"""
        start_time = time.time()
        
        try:
            # Serialize data with orjson if needed
            if isinstance(data, dict):
                data = orjson.dumps(data, option=orjson.OPT_SERIALIZE_NUMPY)
            
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
    
    async def get_multiple(self, urls: List[str], max_concurrent: int = 50) -> List[HTTPResponse]:
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
    
    async def get_with_priority(self, urls: List[str], priorities: List[int]) -> List[HTTPResponse]:
        """Get URLs with priority-based concurrent processing"""
        if len(urls) != len(priorities):
            raise ValueError("URLs and priorities must have the same length")
        
        # Create priority queue
        queue = [(priority, i, url) for i, (priority, url) in enumerate(zip(priorities, urls))]
        heapq.heapify(queue)
        
        results = [None] * len(urls)
        semaphore = asyncio.Semaphore(20)  # Limit concurrent requests
        
        async def process_url(priority: int, index: int, url: str):
            
    """process_url function."""
async with semaphore:
                try:
                    response = await self.get(url)
                    results[index] = response
                except Exception as e:
                    logger.error("Failed to fetch URL", url=url, error=str(e))
                    results[index] = None
        
        # Process URLs in priority order
        tasks = []
        while queue:
            priority, index, url = heapq.heappop(queue)
            task = asyncio.create_task(process_url(priority, index, url))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        return [r for r in results if r is not None]
    
    def _update_metrics(self, elapsed: float):
        """Update performance metrics"""
        self.request_count += 1
        self.total_time += elapsed
        self.avg_response_time = self.total_time / self.request_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_cache_requests = self.cache_hits + self.cache_misses
        cache_hit_ratio = (self.cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        total_requests = self.http1_requests + self.http2_requests
        http2_ratio = (self.http2_requests / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "request_count": self.request_count,
            "total_time": self.total_time,
            "avg_response_time": self.avg_response_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_ratio": cache_hit_ratio,
            "cache_size": len(self.cache) if self.cache else 0,
            "http1_requests": self.http1_requests,
            "http2_requests": self.http2_requests,
            "http2_ratio": http2_ratio,
            "requests_per_second": self.request_count / self.total_time if self.total_time > 0 else 0
        }
    
    async def health_check(self) -> bool:
        """Health check for the HTTP client"""
        try:
            response = await self.get("https://httpbin.org/get", timeout=5.0)
            return response.status_code == 200
        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return False
    
    async def benchmark(self, urls: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        logger.info("Starting HTTP client benchmark v12", url_count=len(urls), iterations=iterations)
        
        all_times = []
        success_count = 0
        total_requests = 0
        http2_count = 0
        
        for iteration in range(iterations):
            logger.info(f"Benchmark iteration {iteration + 1}/{iterations}")
            
            start_time = time.time()
            responses = await self.get_multiple(urls, max_concurrent=50)
            iteration_time = time.time() - start_time
            
            successful_responses = [r for r in responses if r and r.status_code == 200]
            success_count += len(successful_responses)
            total_requests += len(urls)
            all_times.append(iteration_time)
            
            # Count HTTP/2 responses
            http2_count += len([r for r in successful_responses if r.protocol == "http/2"])
            
            logger.info(
                f"Iteration {iteration + 1} completed",
                successful=len(successful_responses),
                total=len(urls),
                time=iteration_time,
                http2_count=http2_count
            )
        
        avg_time = statistics.mean(all_times)
        requests_per_second = (total_requests / avg_time) if avg_time > 0 else 0
        success_rate = (success_count / total_requests * 100) if total_requests > 0 else 0
        http2_usage = (http2_count / success_count * 100) if success_count > 0 else 0
        
        benchmark_results = {
            "iterations": iterations,
            "total_requests": total_requests,
            "successful_requests": success_count,
            "success_rate": success_rate,
            "avg_time_per_iteration": avg_time,
            "requests_per_second": requests_per_second,
            "min_time": min(all_times),
            "max_time": max(all_times),
            "std_dev": statistics.stdev(all_times) if len(all_times) > 1 else 0,
            "http2_usage": http2_usage,
            "http2_requests": http2_count,
            "cache_metrics": self.get_metrics()
        }
        
        logger.info("Benchmark completed", **benchmark_results)
        return benchmark_results


# Global client instance for maximum performance
_global_client: Optional[UltraFastHTTPClientV12] = None


async async def get_http_client() -> UltraFastHTTPClientV12:
    """Get global HTTP client instance"""
    global _global_client
    if _global_client is None:
        _global_client = UltraFastHTTPClientV12()
        await _global_client._initialize_clients()
    return _global_client


async def cleanup_http_client():
    """Cleanup global HTTP client"""
    global _global_client
    if _global_client:
        await _global_client._cleanup()
        _global_client = None 