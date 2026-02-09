from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import time
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from urllib.parse import urlparse
import httpx
from httpx import AsyncClient, Response, Timeout
from loguru import logger
import orjson
from tenacity import retry, stop_after_attempt, wait_exponential
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Optimized HTTP Client
High-performance HTTP client with connection pooling and advanced features
"""



@dataclass
class HTTPConfig:
    """HTTP client configuration"""
    max_connections: int = 100
    max_keepalive_connections: int = 20
    keepalive_timeout: int = 30
    connect_timeout: float = 10.0
    read_timeout: float = 30.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "Ultra-Optimized-SEO-Client/1.0"
    default_headers: Dict[str, str] = None
    
    def __post_init__(self) -> Any:
        if self.default_headers is None:
            self.default_headers = {
                'User-Agent': self.user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            }


@dataclass
class RequestMetrics:
    """Request performance metrics"""
    url: str
    method: str
    status_code: int
    response_time: float
    content_length: int
    timestamp: float
    success: bool
    error: Optional[str] = None


class ConnectionPool:
    """Advanced connection pool with domain-based routing"""
    
    def __init__(self, config: HTTPConfig):
        
    """__init__ function."""
self.config = config
        self.clients: Dict[str, AsyncClient] = {}
        self.request_counts: Dict[str, int] = {}
        self.error_counts: Dict[str, int] = {}
        self.last_used: Dict[str, float] = {}
        
    async def get_client(self, url: str) -> AsyncClient:
        """Get or create HTTP client for domain"""
        domain = urlparse(url).netloc
        
        if domain not in self.clients:
            # Create new client for domain
            client = AsyncClient(
                timeout=Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.write_timeout,
                    pool=self.config.pool_timeout
                ),
                limits=httpx.Limits(
                    max_connections=self.config.max_connections,
                    max_keepalive_connections=self.config.max_keepalive_connections,
                    keepalive_expiry=self.config.keepalive_timeout
                ),
                headers=self.config.default_headers,
                follow_redirects=True,
                max_redirects=5
            )
            
            self.clients[domain] = client
            self.request_counts[domain] = 0
            self.error_counts[domain] = 0
            self.last_used[domain] = time.time()
            
            logger.debug(f"Created HTTP client for domain: {domain}")
        
        # Update usage metrics
        self.request_counts[domain] += 1
        self.last_used[domain] = time.time()
        
        return self.clients[domain]
    
    async def close_client(self, domain: str):
        """Close specific client"""
        if domain in self.clients:
            await self.clients[domain].aclose()
            del self.clients[domain]
            del self.request_counts[domain]
            del self.error_counts[domain]
            del self.last_used[domain]
            logger.debug(f"Closed HTTP client for domain: {domain}")
    
    async def close_all(self) -> Any:
        """Close all clients"""
        for domain in list(self.clients.keys()):
            await self.close_client(domain)
        logger.info("All HTTP clients closed")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return {
            'active_clients': len(self.clients),
            'total_requests': sum(self.request_counts.values()),
            'total_errors': sum(self.error_counts.values()),
            'domains': {
                domain: {
                    'requests': self.request_counts.get(domain, 0),
                    'errors': self.error_counts.get(domain, 0),
                    'last_used': self.last_used.get(domain, 0)
                }
                for domain in self.clients.keys()
            }
        }


class HTTPClient:
    """Ultra-optimized HTTP client with advanced features"""
    
    def __init__(self, config: Optional[HTTPConfig] = None):
        
    """__init__ function."""
self.config = config or HTTPConfig()
        self.pool = ConnectionPool(self.config)
        self.metrics: List[RequestMetrics] = []
        self.rate_limits: Dict[str, float] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
    async def get(self, url: str, headers: Optional[Dict[str, str]] = None) -> Response:
        """Perform GET request with optimizations"""
        return await self._request('GET', url, headers=headers)
    
    async def post(self, url: str, data: Optional[Any] = None, 
                  headers: Optional[Dict[str, str]] = None) -> Response:
        """Perform POST request with optimizations"""
        return await self._request('POST', url, data=data, headers=headers)
    
    async def head(self, url: str, headers: Optional[Dict[str, str]] = None) -> Response:
        """Perform HEAD request with optimizations"""
        return await self._request('HEAD', url, headers=headers)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async async def _request(self, method: str, url: str, data: Optional[Any] = None,
                      headers: Optional[Dict[str, str]] = None) -> Response:
        """Perform HTTP request with retry logic and metrics"""
        start_time = time.time()
        domain = urlparse(url).netloc
        
        # Check rate limiting
        await self._check_rate_limit(domain)
        
        # Check circuit breaker
        if self._is_circuit_open(domain):
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            raise Exception(f"Circuit breaker open for domain: {domain}")
        
        try:
            # Get client from pool
            client = await self.pool.get_client(url)
            
            # Prepare headers
            request_headers = self.config.default_headers.copy()
            if headers:
                request_headers.update(headers)
            
            # Perform request
            response = await client.request(
                method=method,
                url=url,
                data=data,
                headers=request_headers
            )
            
            # Record metrics
            response_time = time.time() - start_time
            self._record_metrics(url, method, response, response_time, None)
            
            # Update circuit breaker
            self._update_circuit_breaker(domain, True)
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            self._record_metrics(url, method, None, response_time, str(e))
            
            # Update circuit breaker
            self._update_circuit_breaker(domain, False)
            
            # Update error count
            self.pool.error_counts[domain] = self.pool.error_counts.get(domain, 0) + 1
            
            logger.error(f"HTTP request failed: {method} {url} - {e}")
            raise
    
    async def _check_rate_limit(self, domain: str):
        """Check and enforce rate limiting"""
        current_time = time.time()
        last_request = self.rate_limits.get(domain, 0)
        
        # Simple rate limiting: max 10 requests per second per domain
        if current_time - last_request < 0.1:  # 100ms between requests
            await asyncio.sleep(0.1)
        
        self.rate_limits[domain] = current_time
    
    def _is_circuit_open(self, domain: str) -> bool:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
        """Check if circuit breaker is open for domain"""
        if domain not in self.circuit_breakers:
            return False
        
        circuit = self.circuit_breakers[domain]
        if circuit['state'] == 'OPEN':
            # Check if recovery timeout has passed
            if time.time() - circuit['last_failure'] > circuit['recovery_timeout']:
                circuit['state'] = 'HALF_OPEN'
                return False
            return True
        
        return False
    
    def _update_circuit_breaker(self, domain: str, success: bool):
        """Update circuit breaker state"""
        if domain not in self.circuit_breakers:
            self.circuit_breakers[domain] = {
                'state': 'CLOSED',
                'failure_count': 0,
                'last_failure': 0,
                'recovery_timeout': 60
            }
        
        circuit = self.circuit_breakers[domain]
        
        if success:
            if circuit['state'] == 'HALF_OPEN':
                circuit['state'] = 'CLOSED'
                circuit['failure_count'] = 0
        else:
            circuit['failure_count'] += 1
            circuit['last_failure'] = time.time()
            
            if circuit['failure_count'] >= 5:  # Open after 5 failures
                circuit['state'] = 'OPEN'
    
    def _record_metrics(self, url: str, method: str, response: Optional[Response],
                       response_time: float, error: Optional[str]):
        """Record request metrics"""
        metrics = RequestMetrics(
            url=url,
            method=method,
            status_code=response.status_code if response else 0,
            response_time=response_time,
            content_length=len(response.content) if response else 0,
            timestamp=time.time(),
            success=response is not None and response.status_code < 400,
            error=error
        )
        
        self.metrics.append(metrics)
        
        # Keep only last 1000 metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-1000:]
    
    async def batch_get(self, urls: List[str], 
                       max_concurrent: int = 10) -> List[Response]:
        """Perform batch GET requests with concurrency control"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def get_with_semaphore(url: str) -> Response:
            async with semaphore:
                return await self.get(url)
        
        tasks = [get_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        responses = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch request failed for {urls[i]}: {result}")
                # Create error response
                responses.append(None)
            else:
                responses.append(result)
        
        return responses
    
    async def get_with_retry(self, url: str, max_retries: int = 3) -> Response:
        """Get with custom retry logic"""
        for attempt in range(max_retries + 1):
            try:
                return await self.get(url)
            except Exception as e:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def get_json(self, url: str) -> Dict[str, Any]:
        """Get and parse JSON response"""
        response = await self.get(url)
        return orjson.loads(response.content)
    
    async def get_text(self, url: str, encoding: str = 'utf-8') -> str:
        """Get and decode text response"""
        response = await self.get(url)
        return response.content.decode(encoding)
    
    async def get_bytes(self, url: str) -> bytes:
        """Get raw bytes response"""
        response = await self.get(url)
        return response.content
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive client statistics"""
        if not self.metrics:
            return {'error': 'No metrics available'}
        
        total_requests = len(self.metrics)
        successful_requests = sum(1 for m in self.metrics if m.success)
        failed_requests = total_requests - successful_requests
        
        response_times = [m.response_time for m in self.metrics]
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        min_response_time = min(response_times)
        
        # Status code distribution
        status_codes = {}
        for metric in self.metrics:
            status = metric.status_code
            status_codes[status] = status_codes.get(status, 0) + 1
        
        return {
            'requests': {
                'total': total_requests,
                'successful': successful_requests,
                'failed': failed_requests,
                'success_rate': successful_requests / total_requests if total_requests > 0 else 0
            },
            'performance': {
                'avg_response_time': avg_response_time,
                'max_response_time': max_response_time,
                'min_response_time': min_response_time,
                'total_bytes_transferred': sum(m.content_length for m in self.metrics)
            },
            'status_codes': status_codes,
            'connection_pool': self.pool.get_stats(),
            'circuit_breakers': {
                domain: {
                    'state': circuit['state'],
                    'failure_count': circuit['failure_count']
                }
                for domain, circuit in self.circuit_breakers.items()
            }
        }
    
    async def close(self) -> Any:
        """Close all connections"""
        await self.pool.close_all()
        logger.info("HTTP client closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        try:
            # Test with a simple request
            response = await self.get('https://httpbin.org/get')
            return {
                'status': 'healthy',
                'response_time': response.elapsed.total_seconds(),
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            } 