from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncGenerator
from urllib.parse import urljoin, urlparse
import hashlib
import hmac
import base64
    import requests
    from requests import Session, Response
    from requests.adapters import HTTPAdapter
    from requests.packages.urllib3.util.retry import Retry
    from requests.auth import HTTPBasicAuth, HTTPDigestAuth
    import httpx
    from httpx import AsyncClient, Client, Response as HTTPXResponse
    from httpx import Timeout, Limits, HTTPStatusError, RequestError
from typing import Any, List, Dict, Optional
"""
HTTP Client Examples - Comprehensive HTTP Operations
==================================================

This module provides robust HTTP client capabilities using both:
- requests: Synchronous HTTP operations
- httpx: Asynchronous HTTP operations

Features:
- Request/response management with retry logic
- Session management and connection pooling
- Authentication methods (Basic, Bearer, OAuth)
- Proxy support and SSL/TLS configuration
- Rate limiting and circuit breaker patterns
- Comprehensive error handling and logging
- Request/response interceptors and middleware
- File upload/download with progress tracking

Author: AI Assistant
License: MIT
"""


try:
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    Session = None
    Response = None
    HTTPAdapter = None
    Retry = None
    HTTPBasicAuth = None
    HTTPDigestAuth = None

try:
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    AsyncClient = None
    Client = None
    HTTPXResponse = None
    Timeout = None
    Limits = None
    HTTPStatusError = Exception
    RequestError = Exception

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HTTPConfig:
    """HTTP client configuration."""
    base_url: str = ""
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_redirects: int = 5
    verify_ssl: bool = True
    cert: Optional[str] = None
    key: Optional[str] = None
    proxies: Optional[Dict[str, str]] = None
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    auth_username: Optional[str] = None
    auth_password: Optional[str] = None
    auth_token: Optional[str] = None
    auth_type: str = "basic"  # basic, bearer, digest, oauth
    rate_limit_requests: int = 100
    rate_limit_window: int = 60
    connection_pool_size: int = 10
    max_connections: int = 100
    keepalive_timeout: int = 30
    user_agent: str = "HTTPClient/1.0"
    follow_redirects: bool = True
    allow_redirects: bool = True
    stream: bool = False
    chunk_size: int = 8192


@dataclass
class HTTPRequest:
    """HTTP request configuration."""
    method: str = "GET"
    url: str = ""
    params: Optional[Dict[str, Any]] = None
    data: Optional[Union[Dict[str, Any], str, bytes]] = None
    json_data: Optional[Dict[str, Any]] = None
    files: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    cookies: Optional[Dict[str, str]] = None
    timeout: Optional[int] = None
    allow_redirects: Optional[bool] = None
    verify: Optional[bool] = None
    stream: Optional[bool] = None
    auth: Optional[Any] = None


@dataclass
class HTTPResponse:
    """HTTP response wrapper."""
    success: bool
    status_code: int = 0
    url: str = ""
    headers: Dict[str, str] = field(default_factory=dict)
    cookies: Dict[str, str] = field(default_factory=dict)
    content: Optional[bytes] = None
    text: str = ""
    json_data: Optional[Dict[str, Any]] = None
    encoding: str = "utf-8"
    elapsed_time: float = 0.0
    error_message: str = ""
    request: Optional[HTTPRequest] = None


@dataclass
class RateLimitInfo:
    """Rate limiting information."""
    requests_made: int = 0
    window_start: float = 0.0
    window_end: float = 0.0
    limit_exceeded: bool = False


class HTTPClientError(Exception):
    """Custom exception for HTTP client errors."""
    pass


class HTTPRequestError(Exception):
    """Custom exception for HTTP request errors."""
    pass


class HTTPResponseError(Exception):
    """Custom exception for HTTP response errors."""
    pass


class RateLimitExceededError(Exception):
    """Custom exception for rate limit exceeded."""
    pass


class CircuitBreakerError(Exception):
    """Custom exception for circuit breaker open."""
    pass


class RequestsHTTPClient:
    """Synchronous HTTP client using requests."""
    
    def __init__(self, config: HTTPConfig):
        """Initialize HTTP client with configuration."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests is not available. Install with: pip install requests")
        
        self.config = config
        self.session: Optional[Session] = None
        self.rate_limit_info = RateLimitInfo()
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60.0
        
    def __enter__(self) -> Any:
        """Context manager entry."""
        self._create_session()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Context manager exit."""
        self.close()
    
    def _create_session(self) -> Any:
        """Create and configure requests session."""
        if self.session:
            return
            
        self.session = Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=self.config.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE"]
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=self.config.connection_pool_size,
            pool_maxsize=self.config.max_connections
        )
        
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            "User-Agent": self.config.user_agent,
            **self.config.headers
        })
        
        # Set default cookies
        if self.config.cookies:
            self.session.cookies.update(self.config.cookies)
        
        # Configure authentication
        self._setup_authentication()
        
        # Configure proxies
        if self.config.proxies:
            self.session.proxies.update(self.config.proxies)
        
        # Configure SSL verification
        self.session.verify = self.config.verify_ssl
        if self.config.cert and self.config.key:
            self.session.cert = (self.config.cert, self.config.key)
    
    def _setup_authentication(self) -> Any:
        """Setup authentication for the session."""
        if not self.config.auth_username and not self.config.auth_token:
            return
            
        if self.config.auth_type == "basic" and self.config.auth_username:
            self.session.auth = HTTPBasicAuth(
                self.config.auth_username,
                self.config.auth_password or ""
            )
        elif self.config.auth_type == "digest" and self.config.auth_username:
            self.session.auth = HTTPDigestAuth(
                self.config.auth_username,
                self.config.auth_password or ""
            )
        elif self.config.auth_type == "bearer" and self.config.auth_token:
            self.session.headers["Authorization"] = f"Bearer {self.config.auth_token}"
        elif self.config.auth_type == "oauth" and self.config.auth_token:
            self.session.headers["Authorization"] = f"OAuth {self.config.auth_token}"
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        
        # Reset window if expired
        if current_time > self.rate_limit_info.window_end:
            self.rate_limit_info.requests_made = 0
            self.rate_limit_info.window_start = current_time
            self.rate_limit_info.window_end = current_time + self.config.rate_limit_window
        
        # Check if limit exceeded
        if self.rate_limit_info.requests_made >= self.config.rate_limit_requests:
            self.rate_limit_info.limit_exceeded = True
            return False
        
        self.rate_limit_info.requests_made += 1
        return True
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open."""
        current_time = time.time()
        
        # Check if circuit breaker should be reset
        if (current_time - self.circuit_breaker_last_failure) > self.circuit_breaker_timeout:
            self.circuit_breaker_failures = 0
        
        # Check if circuit breaker should be opened
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            return False
        
        return True
    
    def _record_failure(self) -> Any:
        """Record a failure for circuit breaker."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
    
    async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute HTTP request."""
        if not self.session:
            return HTTPResponse(
                success=False,
                error_message="Session not initialized"
            )
        
        if not request.url:
            return HTTPResponse(
                success=False,
                error_message="URL is required"
            )
        
        # Check rate limit
        if not self._check_rate_limit():
            return HTTPResponse(
                success=False,
                error_message="Rate limit exceeded",
                request=request
            )
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            return HTTPResponse(
                success=False,
                error_message="Circuit breaker is open",
                request=request
            )
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            url = urljoin(self.config.base_url, request.url)
            
            kwargs = {
                'method': request.method.upper(),
                'url': url,
                'timeout': request.timeout or self.config.timeout,
                'allow_redirects': request.allow_redirects if request.allow_redirects is not None else self.config.allow_redirects,
                'verify': request.verify if request.verify is not None else self.config.verify_ssl,
                'stream': request.stream if request.stream is not None else self.config.stream
            }
            
            # Add optional parameters
            if request.params:
                kwargs['params'] = request.params
            if request.data:
                kwargs['data'] = request.data
            if request.json_data:
                kwargs['json'] = request.json_data
            if request.files:
                kwargs['files'] = request.files
            if request.headers:
                kwargs['headers'] = request.headers
            if request.cookies:
                kwargs['cookies'] = request.cookies
            if request.auth:
                kwargs['auth'] = request.auth
            
            logger.info(f"Making {request.method} request to {url}")
            
            response: Response = self.session.request(**kwargs)
            
            elapsed_time = time.time() - start_time
            
            # Reset circuit breaker on success
            self.circuit_breaker_failures = 0
            
            # Parse response
            success = 200 <= response.status_code < 400
            content = response.content if not kwargs.get('stream') else None
            text = response.text if not kwargs.get('stream') else ""
            
            # Try to parse JSON
            json_data = None
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                except json.JSONDecodeError:
                    pass
            
            return HTTPResponse(
                success=success,
                status_code=response.status_code,
                url=str(response.url),
                headers=dict(response.headers),
                cookies=dict(response.cookies),
                content=content,
                text=text,
                json_data=json_data,
                encoding=response.encoding or 'utf-8',
                elapsed_time=elapsed_time,
                request=request
            )
            
        except requests.exceptions.Timeout as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Request timeout: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Request timeout: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except requests.exceptions.ConnectionError as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Connection error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Connection error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Request error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Request error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Unexpected error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Unexpected error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
    
    def get(self, url: str, **kwargs) -> HTTPResponse:
        """Execute GET request."""
        request = HTTPRequest(method="GET", url=url, **kwargs)
        return self.request(request)
    
    def post(self, url: str, **kwargs) -> HTTPResponse:
        """Execute POST request."""
        request = HTTPRequest(method="POST", url=url, **kwargs)
        return self.request(request)
    
    def put(self, url: str, **kwargs) -> HTTPResponse:
        """Execute PUT request."""
        request = HTTPRequest(method="PUT", url=url, **kwargs)
        return self.request(request)
    
    def delete(self, url: str, **kwargs) -> HTTPResponse:
        """Execute DELETE request."""
        request = HTTPRequest(method="DELETE", url=url, **kwargs)
        return self.request(request)
    
    def patch(self, url: str, **kwargs) -> HTTPResponse:
        """Execute PATCH request."""
        request = HTTPRequest(method="PATCH", url=url, **kwargs)
        return self.request(request)
    
    def head(self, url: str, **kwargs) -> HTTPResponse:
        """Execute HEAD request."""
        request = HTTPRequest(method="HEAD", url=url, **kwargs)
        return self.request(request)
    
    def options(self, url: str, **kwargs) -> HTTPResponse:
        """Execute OPTIONS request."""
        request = HTTPRequest(method="OPTIONS", url=url, **kwargs)
        return self.request(request)
    
    async def download_file(self, url: str, file_path: str, chunk_size: int = None) -> HTTPResponse:
        """Download file with progress tracking."""
        if not url:
            return HTTPResponse(
                success=False,
                error_message="URL is required for download"
            )
        
        if not file_path:
            return HTTPResponse(
                success=False,
                error_message="File path is required for download"
            )
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        try:
            request = HTTPRequest(method="GET", url=url, stream=True)
            response = self.request(request)
            
            if not response.success:
                return response
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            total_bytes = 0
            with open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                for chunk in response.content:
                    f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    total_bytes += len(chunk)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Downloaded {total_bytes} bytes to {file_path} in {elapsed_time:.2f}s")
            
            return HTTPResponse(
                success=True,
                status_code=response.status_code,
                url=response.url,
                headers=response.headers,
                elapsed_time=elapsed_time,
                request=request
            )
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Download error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Download error: {e}",
                elapsed_time=elapsed_time,
                request=HTTPRequest(method="GET", url=url)
            )
    
    def close(self) -> Any:
        """Close the session."""
        if self.session:
            self.session.close()
            self.session = None
            logger.info("HTTP session closed")


class HTTPXHTTPClient:
    """Asynchronous HTTP client using httpx."""
    
    def __init__(self, config: HTTPConfig):
        """Initialize async HTTP client with configuration."""
        if not HTTPX_AVAILABLE:
            raise ImportError("httpx is not available. Install with: pip install httpx")
        
        self.config = config
        self.client: Optional[AsyncClient] = None
        self.rate_limit_info = RateLimitInfo()
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0.0
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60.0
    
    async def __aenter__(self) -> Any:
        """Async context manager entry."""
        await self._create_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> Any:
        """Async context manager exit."""
        await self.close()
    
    async def _create_client(self) -> Any:
        """Create and configure httpx async client."""
        if self.client:
            return
        
        # Configure timeout
        timeout = Timeout(
            connect=self.config.timeout,
            read=self.config.timeout,
            write=self.config.timeout,
            pool=self.config.timeout
        )
        
        # Configure limits
        limits = Limits(
            max_connections=self.config.max_connections,
            max_keepalive_connections=self.config.connection_pool_size,
            keepalive_expiry=self.config.keepalive_timeout
        )
        
        # Prepare headers
        headers = {
            "User-Agent": self.config.user_agent,
            **self.config.headers
        }
        
        # Add authentication headers
        if self.config.auth_type == "bearer" and self.config.auth_token:
            headers["Authorization"] = f"Bearer {self.config.auth_token}"
        elif self.config.auth_type == "oauth" and self.config.auth_token:
            headers["Authorization"] = f"OAuth {self.config.auth_token}"
        
        # Create client
        self.client = AsyncClient(
            base_url=self.config.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
            cookies=self.config.cookies,
            verify=self.config.verify_ssl,
            cert=(self.config.cert, self.config.key) if self.config.cert and self.config.key else None,
            proxies=self.config.proxies,
            follow_redirects=self.config.follow_redirects,
            max_redirects=self.config.max_redirects
        )
        
        # Setup authentication
        await self._setup_authentication()
    
    async def _setup_authentication(self) -> Any:
        """Setup authentication for the client."""
        if not self.config.auth_username and not self.config.auth_token:
            return
        
        if self.config.auth_type == "basic" and self.config.auth_username:
            self.client.auth = httpx.BasicAuth(
                self.config.auth_username,
                self.config.auth_password or ""
            )
        elif self.config.auth_type == "digest" and self.config.auth_username:
            self.client.auth = httpx.DigestAuth(
                self.config.auth_username,
                self.config.auth_password or ""
            )
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        
        # Reset window if expired
        if current_time > self.rate_limit_info.window_end:
            self.rate_limit_info.requests_made = 0
            self.rate_limit_info.window_start = current_time
            self.rate_limit_info.window_end = current_time + self.config.rate_limit_window
        
        # Check if limit exceeded
        if self.rate_limit_info.requests_made >= self.config.rate_limit_requests:
            self.rate_limit_info.limit_exceeded = True
            return False
        
        self.rate_limit_info.requests_made += 1
        return True
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker is open."""
        current_time = time.time()
        
        # Check if circuit breaker should be reset
        if (current_time - self.circuit_breaker_last_failure) > self.circuit_breaker_timeout:
            self.circuit_breaker_failures = 0
        
        # Check if circuit breaker should be opened
        if self.circuit_breaker_failures >= self.circuit_breaker_threshold:
            return False
        
        return True
    
    def _record_failure(self) -> Any:
        """Record a failure for circuit breaker."""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
    
    async async def request(self, request: HTTPRequest) -> HTTPResponse:
        """Execute async HTTP request."""
        if not self.client:
            return HTTPResponse(
                success=False,
                error_message="Client not initialized"
            )
        
        if not request.url:
            return HTTPResponse(
                success=False,
                error_message="URL is required"
            )
        
        # Check rate limit
        if not self._check_rate_limit():
            return HTTPResponse(
                success=False,
                error_message="Rate limit exceeded",
                request=request
            )
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            return HTTPResponse(
                success=False,
                error_message="Circuit breaker is open",
                request=request
            )
        
        start_time = time.time()
        
        try:
            # Prepare request parameters
            url = urljoin(self.config.base_url, request.url)
            
            kwargs = {
                'method': request.method.upper(),
                'url': url,
                'timeout': request.timeout or self.config.timeout,
                'follow_redirects': request.allow_redirects if request.allow_redirects is not None else self.config.allow_redirects,
                'verify': request.verify if request.verify is not None else self.config.verify_ssl,
                'stream': request.stream if request.stream is not None else self.config.stream
            }
            
            # Add optional parameters
            if request.params:
                kwargs['params'] = request.params
            if request.data:
                kwargs['data'] = request.data
            if request.json_data:
                kwargs['json'] = request.json_data
            if request.files:
                kwargs['files'] = request.files
            if request.headers:
                kwargs['headers'] = request.headers
            if request.cookies:
                kwargs['cookies'] = request.cookies
            if request.auth:
                kwargs['auth'] = request.auth
            
            logger.info(f"Making async {request.method} request to {url}")
            
            response: HTTPXResponse = await self.client.request(**kwargs)
            
            elapsed_time = time.time() - start_time
            
            # Reset circuit breaker on success
            self.circuit_breaker_failures = 0
            
            # Parse response
            success = 200 <= response.status_code < 400
            content = await response.aread() if not kwargs.get('stream') else None
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
            text = response.text if not kwargs.get('stream') else ""
            
            # Try to parse JSON
            json_data = None
            if response.headers.get('content-type', '').startswith('application/json'):
                try:
                    json_data = response.json()
                except json.JSONDecodeError:
                    pass
            
            return HTTPResponse(
                success=success,
                status_code=response.status_code,
                url=str(response.url),
                headers=dict(response.headers),
                cookies=dict(response.cookies),
                content=content,
                text=text,
                json_data=json_data,
                encoding=response.encoding or 'utf-8',
                elapsed_time=elapsed_time,
                request=request
            )
            
        except httpx.TimeoutException as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Request timeout: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Request timeout: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except httpx.ConnectError as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Connection error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Connection error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except HTTPStatusError as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"HTTP status error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"HTTP status error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except RequestError as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Request error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Request error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
        except Exception as e:
            elapsed_time = time.time() - start_time
            self._record_failure()
            logger.error(f"Unexpected error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Unexpected error: {e}",
                elapsed_time=elapsed_time,
                request=request
            )
    
    async def get(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async GET request."""
        request = HTTPRequest(method="GET", url=url, **kwargs)
        return await self.request(request)
    
    async def post(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async POST request."""
        request = HTTPRequest(method="POST", url=url, **kwargs)
        return await self.request(request)
    
    async def put(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async PUT request."""
        request = HTTPRequest(method="PUT", url=url, **kwargs)
        return await self.request(request)
    
    async def delete(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async DELETE request."""
        request = HTTPRequest(method="DELETE", url=url, **kwargs)
        return await self.request(request)
    
    async def patch(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async PATCH request."""
        request = HTTPRequest(method="PATCH", url=url, **kwargs)
        return await self.request(request)
    
    async def head(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async HEAD request."""
        request = HTTPRequest(method="HEAD", url=url, **kwargs)
        return await self.request(request)
    
    async def options(self, url: str, **kwargs) -> HTTPResponse:
        """Execute async OPTIONS request."""
        request = HTTPRequest(method="OPTIONS", url=url, **kwargs)
        return await self.request(request)
    
    async async def download_file(self, url: str, file_path: str, chunk_size: int = None) -> HTTPResponse:
        """Download file asynchronously with progress tracking."""
        if not url:
            return HTTPResponse(
                success=False,
                error_message="URL is required for download"
            )
        
        if not file_path:
            return HTTPResponse(
                success=False,
                error_message="File path is required for download"
            )
        
        chunk_size = chunk_size or self.config.chunk_size
        start_time = time.time()
        
        try:
            request = HTTPRequest(method="GET", url=url, stream=True)
            response = await self.request(request)
            
            if not response.success:
                return response
            
            # Create directory if it doesn't exist
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            total_bytes = 0
            async with aiofiles.open(file_path, 'wb') as f:
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                async for chunk in response.content:
                    await f.write(chunk)
    try:
        pass
    except Exception as e:
        print(f"Error: {e}")
                    total_bytes += len(chunk)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Downloaded {total_bytes} bytes to {file_path} in {elapsed_time:.2f}s")
            
            return HTTPResponse(
                success=True,
                status_code=response.status_code,
                url=response.url,
                headers=response.headers,
                elapsed_time=elapsed_time,
                request=request
            )
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Download error: {e}")
            return HTTPResponse(
                success=False,
                error_message=f"Download error: {e}",
                elapsed_time=elapsed_time,
                request=HTTPRequest(method="GET", url=url)
            )
    
    async def close(self) -> Any:
        """Close the async client."""
        if self.client:
            await self.client.aclose()
            self.client = None
            logger.info("Async HTTP client closed")


class HTTPClientPool:
    """Connection pool for HTTP clients."""
    
    def __init__(self, max_clients: int = 10):
        """Initialize client pool."""
        self.max_clients = max_clients
        self.clients: List[Union[RequestsHTTPClient, HTTPXHTTPClient]] = []
        self._lock = asyncio.Lock()
    
    async def get_client(self, config: HTTPConfig, async_mode: bool = True) -> Union[RequestsHTTPClient, HTTPXHTTPClient]:
        """Get client from pool or create new one."""
        async with self._lock:
            # Check for available clients
            for client in self.clients:
                if not client.client and not client.session:
                    # Reuse existing client
                    if async_mode and isinstance(client, HTTPXHTTPClient):
                        await client._create_client()
                        return client
                    elif not async_mode and isinstance(client, RequestsHTTPClient):
                        client._create_session()
                        return client
            
            # Create new client if pool not full
            if len(self.clients) < self.max_clients:
                if async_mode:
                    client = HTTPXHTTPClient(config)
                    await client._create_client()
                else:
                    client = RequestsHTTPClient(config)
                    client._create_session()
                
                self.clients.append(client)
                return client
            
            # Wait for available client
            while True:
                for client in self.clients:
                    if not client.client and not client.session:
                        if async_mode and isinstance(client, HTTPXHTTPClient):
                            await client._create_client()
                            return client
                        elif not async_mode and isinstance(client, RequestsHTTPClient):
                            client._create_session()
                            return client
                
                await asyncio.sleep(0.1)
    
    async def return_client(self, client: Union[RequestsHTTPClient, HTTPXHTTPClient]):
        """Return client to pool."""
        async with self._lock:
            if client in self.clients:
                # Keep client in pool but mark as available
                pass


# Example usage functions
def demonstrate_requests_usage():
    """Demonstrate requests HTTP usage."""
    if not REQUESTS_AVAILABLE:
        logger.error("requests not available")
        return
    
    config = HTTPConfig(
        base_url="https://api.example.com",
        timeout=30,
        headers={"Accept": "application/json"}
    )
    
    # Using context manager
    with RequestsHTTPClient(config) as client:
        # GET request
        response = client.get("/users/1")
        if response.success:
            logger.info(f"User data: {response.json_data}")
        else:
            logger.error(f"Request failed: {response.error_message}")
        
        # POST request with JSON
        post_response = client.post("/users", json_data={
            "name": "John Doe",
            "email": "john@example.com"
        })
        if post_response.success:
            logger.info(f"Created user: {post_response.json_data}")
        else:
            logger.error(f"POST failed: {post_response.error_message}")
        
        # Download file
        download_response = client.download_file(
            "https://example.com/file.pdf",
            "downloaded_file.pdf"
        )
        if download_response.success:
            logger.info("File downloaded successfully")
        else:
            logger.error(f"Download failed: {download_response.error_message}")


async def demonstrate_httpx_usage():
    """Demonstrate httpx usage."""
    if not HTTPX_AVAILABLE:
        logger.error("httpx not available")
        return
    
    config = HTTPConfig(
        base_url="https://api.example.com",
        timeout=30,
        headers={"Accept": "application/json"}
    )
    
    # Using async context manager
    async with HTTPXHTTPClient(config) as client:
        # GET request
        response = await client.get("/users/1")
        if response.success:
            logger.info(f"User data: {response.json_data}")
        else:
            logger.error(f"Request failed: {response.error_message}")
        
        # POST request with JSON
        post_response = await client.post("/users", json_data={
            "name": "John Doe",
            "email": "john@example.com"
        })
        if post_response.success:
            logger.info(f"Created user: {post_response.json_data}")
        else:
            logger.error(f"POST failed: {post_response.error_message}")
        
        # Download file
        download_response = await client.download_file(
            "https://example.com/file.pdf",
            "downloaded_file.pdf"
        )
        if download_response.success:
            logger.info("File downloaded successfully")
        else:
            logger.error(f"Download failed: {download_response.error_message}")


async def demonstrate_client_pool():
    """Demonstrate client pool usage."""
    config = HTTPConfig(
        base_url="https://api.example.com",
        timeout=30
    )
    
    pool = HTTPClientPool(max_clients=5)
    
    # Get client from pool
    client = await pool.get_client(config, async_mode=True)
    
    try:
        # Use client
        response = await client.get("/status")
        logger.info(f"Pool client response: {response.success}")
    finally:
        # Return client to pool
        await pool.return_client(client)


def main():
    """Main function demonstrating HTTP client usage."""
    logger.info("Starting HTTP client examples")
    
    # Demonstrate requests usage
    try:
        demonstrate_requests_usage()
    except Exception as e:
        logger.error(f"Requests demonstration failed: {e}")
    
    # Demonstrate httpx usage
    try:
        asyncio.run(demonstrate_httpx_usage())
    except Exception as e:
        logger.error(f"HTTPX demonstration failed: {e}")
    
    # Demonstrate client pool
    try:
        asyncio.run(demonstrate_client_pool())
    except Exception as e:
        logger.error(f"Client pool demonstration failed: {e}")
    
    logger.info("HTTP client examples completed")


match __name__:
    case "__main__":
    main() 