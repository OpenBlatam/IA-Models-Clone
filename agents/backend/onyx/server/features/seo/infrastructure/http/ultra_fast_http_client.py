from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

import asyncio
import time
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import httpx
import asyncio_throttle
import pybreaker
import tenacity
from shared.core.exceptions import HTTPClientError
from shared.core.config import Settings
from shared.core.logging import get_logger
from typing import Any, List, Dict, Optional
import logging
"""
Ultra-Fast HTTP Client
Maximum Performance with Fastest Libraries
"""



logger = get_logger(__name__)


class UltraFastHTTPClient:
    """
    Ultra-fast HTTP client with advanced features
    
    This client provides maximum performance with connection pooling,
    rate limiting, circuit breakers, and retry logic.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize HTTP client
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.client: Optional[httpx.AsyncClient] = None
        self.rate_limiter = asyncio_throttle.Throttler(rate_limit=settings.rate_limit)
        self.circuit_breaker = pybreaker.CircuitBreaker(
            fail_max=settings.circuit_breaker_fail_max,
            reset_timeout=settings.circuit_breaker_reset_timeout
        )
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
    
    async def initialize(self) -> None:
        """Initialize HTTP client with connection pooling"""
        try:
            self.client = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=self.settings.max_connections,
                    max_keepalive_connections=self.settings.max_keepalive_connections,
                    keepalive_expiry=self.settings.keepalive_expiry
                ),
                timeout=httpx.Timeout(self.settings.request_timeout),
                http2=True,
                verify=False,  # For development
                follow_redirects=True,
                max_redirects=5
            )
            logger.info("Ultra-fast HTTP client initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize HTTP client", error=str(e))
            raise HTTPClientError(f"Failed to initialize HTTP client: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup HTTP client resources"""
        if self.client:
            try:
                await self.client.aclose()
                logger.info("HTTP client cleaned up successfully")
            except Exception as e:
                logger.warning("Error during HTTP client cleanup", error=str(e))
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def get(self, url: str, timeout: Optional[float] = None, headers: Optional[Dict[str, str]] = None) -> httpx.Response:
        """
        Get URL with retry logic and circuit breaker
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            headers: Request headers
            
        Returns:
            httpx.Response: HTTP response
            
        Raises:
            HTTPClientError: If request fails
        """
        start_time = time.time()
        
        async with self.rate_limiter:
            try:
                # Use circuit breaker
                response = await self.circuit_breaker.call(
                    self._make_request, url, "GET", timeout, headers
                )
                
                # Update metrics
                self._request_count += 1
                response_time = time.time() - start_time
                self._total_response_time += response_time
                
                logger.debug(
                    "HTTP GET request successful",
                    url=url,
                    status_code=response.status_code,
                    response_time=response_time
                )
                
                return response
                
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "HTTP GET request failed",
                    url=url,
                    error=str(e),
                    response_time=time.time() - start_time
                )
                raise HTTPClientError(f"Failed to fetch URL {url}: {str(e)}")
    
    @tenacity.retry(
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        retry=tenacity.retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException))
    )
    async def post(self, url: str, data: Optional[Dict[str, Any]] = None, 
                   json: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None,
                   headers: Optional[Dict[str, str]] = None) -> httpx.Response:
        """
        Post to URL with retry logic and circuit breaker
        
        Args:
            url: URL to post to
            data: Form data
            json: JSON data
            timeout: Request timeout
            headers: Request headers
            
        Returns:
            httpx.Response: HTTP response
            
        Raises:
            HTTPClientError: If request fails
        """
        start_time = time.time()
        
        async with self.rate_limiter:
            try:
                # Use circuit breaker
                response = await self.circuit_breaker.call(
                    self._make_request, url, "POST", timeout, headers, data, json
                )
                
                # Update metrics
                self._request_count += 1
                response_time = time.time() - start_time
                self._total_response_time += response_time
                
                logger.debug(
                    "HTTP POST request successful",
                    url=url,
                    status_code=response.status_code,
                    response_time=response_time
                )
                
                return response
                
            except Exception as e:
                self._error_count += 1
                logger.error(
                    "HTTP POST request failed",
                    url=url,
                    error=str(e),
                    response_time=time.time() - start_time
                )
                raise HTTPClientError(f"Failed to post to URL {url}: {str(e)}")
    
    async async def _make_request(self, url: str, method: str, timeout: Optional[float] = None,
                           headers: Optional[Dict[str, str]] = None, data: Optional[Dict[str, Any]] = None,
                           json: Optional[Dict[str, Any]] = None) -> httpx.Response:
        """
        Make HTTP request
        
        Args:
            url: Request URL
            method: HTTP method
            timeout: Request timeout
            headers: Request headers
            data: Form data
            json: JSON data
            
        Returns:
            httpx.Response: HTTP response
            
        Raises:
            HTTPClientError: If request fails
        """
        if not self.client:
            raise HTTPClientError("HTTP client not initialized")
        
        try:
            if method.upper() == "GET":
                response = await self.client.get(
                    url,
                    timeout=timeout or self.settings.request_timeout,
                    headers=headers
                )
            elif method.upper() == "POST":
                response = await self.client.post(
                    url,
                    data=data,
                    json=json,
                    timeout=timeout or self.settings.request_timeout,
                    headers=headers
                )
            else:
                raise HTTPClientError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response
            
        except httpx.HTTPStatusError as e:
            raise HTTPClientError(f"HTTP {e.response.status_code} error: {e.response.text}")
        except httpx.RequestError as e:
            raise HTTPClientError(f"Request error: {str(e)}")
        except Exception as e:
            raise HTTPClientError(f"Unexpected error: {str(e)}")
    
    @asynccontextmanager
    async def session(self) -> Any:
        """
        Context manager for HTTP session
        
        Yields:
            UltraFastHTTPClient: HTTP client instance
        """
        try:
            yield self
        finally:
            await self.cleanup()
    
    async def get_text(self, url: str, timeout: Optional[float] = None, 
                      headers: Optional[Dict[str, str]] = None) -> str:
        """
        Get URL content as text
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            headers: Request headers
            
        Returns:
            str: Response text
            
        Raises:
            HTTPClientError: If request fails
        """
        response = await self.get(url, timeout, headers)
        return response.text
    
    async def get_json(self, url: str, timeout: Optional[float] = None,
                      headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Get URL content as JSON
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            headers: Request headers
            
        Returns:
            Dict[str, Any]: Response JSON
            
        Raises:
            HTTPClientError: If request fails
        """
        response = await self.get(url, timeout, headers)
        return response.json()
    
    async def get_bytes(self, url: str, timeout: Optional[float] = None,
                       headers: Optional[Dict[str, str]] = None) -> bytes:
        """
        Get URL content as bytes
        
        Args:
            url: URL to fetch
            timeout: Request timeout
            headers: Request headers
            
        Returns:
            bytes: Response bytes
            
        Raises:
            HTTPClientError: If request fails
        """
        response = await self.get(url, timeout, headers)
        return response.content
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get client metrics
        
        Returns:
            Dict[str, Any]: Client metrics
        """
        avg_response_time = (
            self._total_response_time / self._request_count 
            if self._request_count > 0 else 0.0
        )
        
        error_rate = (
            self._error_count / self._request_count * 100 
            if self._request_count > 0 else 0.0
        )
        
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "success_rate": 100.0 - error_rate,
            "error_rate": error_rate,
            "average_response_time": avg_response_time,
            "total_response_time": self._total_response_time,
            "circuit_breaker_state": self.circuit_breaker.current_state,
            "circuit_breaker_fail_count": self.circuit_breaker.fail_counter
        }
    
    def reset_metrics(self) -> None:
        """Reset client metrics"""
        self._request_count = 0
        self._error_count = 0
        self._total_response_time = 0.0
        logger.info("HTTP client metrics reset")
    
    def is_healthy(self) -> bool:
        """
        Check if client is healthy
        
        Returns:
            bool: True if healthy
        """
        return (
            self.client is not None and
            self.circuit_breaker.current_state == "closed"
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Dict[str, Any]: Health check results
        """
        try:
            # Test with a simple request
            response = await self.get("https://httpbin.org/status/200", timeout=5.0)
            return {
                "status": "healthy",
                "client_initialized": self.client is not None,
                "circuit_breaker_state": self.circuit_breaker.current_state,
                "test_request_successful": response.status_code == 200
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "client_initialized": self.client is not None,
                "circuit_breaker_state": self.circuit_breaker.current_state,
                "error": str(e)
            } 