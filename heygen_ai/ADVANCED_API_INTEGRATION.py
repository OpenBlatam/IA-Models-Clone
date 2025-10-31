#!/usr/bin/env python3
"""
🌐 HeyGen AI - Advanced API Integration Layer
============================================

This module provides a comprehensive API integration layer that seamlessly
connects all the HeyGen AI improvements with external systems and services.
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import aiohttp
import aiofiles
from pathlib import Path
import yaml
import hashlib
import secrets
from urllib.parse import urljoin, urlparse
import ssl
import certifi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIVersion(str, Enum):
    """API version levels"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    LATEST = "latest"

class IntegrationType(str, Enum):
    """Integration types"""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"
    FILE_SYSTEM = "file_system"
    DATABASE = "database"

class AuthenticationMethod(str, Enum):
    """Authentication methods"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    CERTIFICATE = "certificate"
    NONE = "none"

@dataclass
class APIEndpoint:
    """API endpoint configuration"""
    name: str
    url: str
    method: str = "GET"
    version: APIVersion = APIVersion.LATEST
    authentication: AuthenticationMethod = AuthenticationMethod.API_KEY
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None
    cache_ttl: int = 300
    enabled: bool = True

@dataclass
class IntegrationConfig:
    """Integration configuration"""
    name: str
    type: IntegrationType
    base_url: str
    authentication: AuthenticationMethod
    credentials: Dict[str, str] = field(default_factory=dict)
    endpoints: List[APIEndpoint] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[int] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIResponse:
    """Standardized API response"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: int = 200
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    request_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
        self._lock = asyncio.Lock()
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed"""
        async with self._lock:
            current_time = time.time()
            window_start = current_time - self.time_window
            
            # Clean old requests
            if key in self.requests:
                self.requests[key] = [
                    req_time for req_time in self.requests[key] 
                    if req_time > window_start
                ]
            else:
                self.requests[key] = []
            
            # Check if under limit
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(current_time)
                return True
            
            return False
    
    async def get_retry_after(self, key: str) -> int:
        """Get seconds until next request is allowed"""
        async with self._lock:
            if key not in self.requests or not self.requests[key]:
                return 0
            
            oldest_request = min(self.requests[key])
            return max(0, int(oldest_request + self.time_window - time.time()))

class APICache:
    """Intelligent API response caching"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached response"""
        async with self._lock:
            if key not in self.cache:
                return None
            
            # Check TTL
            cached_item = self.cache[key]
            if time.time() > cached_item['expires_at']:
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]
                return None
            
            # Update access time
            self.access_times[key] = time.time()
            return cached_item['data']
    
    async def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Cache response data"""
        async with self._lock:
            # Remove oldest if at capacity
            if len(self.cache) >= self.max_size:
                await self._evict_oldest()
            
            ttl = ttl or self.default_ttl
            self.cache[key] = {
                'data': data,
                'expires_at': time.time() + ttl
            }
            self.access_times[key] = time.time()
    
    async def _evict_oldest(self):
        """Evict least recently used item"""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    async def clear(self):
        """Clear all cached data"""
        async with self._lock:
            self.cache.clear()
            self.access_times.clear()

class APIClient:
    """Advanced API client with comprehensive features"""
    
    def __init__(self, 
                 base_url: str,
                 authentication: AuthenticationMethod = AuthenticationMethod.API_KEY,
                 credentials: Optional[Dict[str, str]] = None,
                 timeout: int = 30,
                 retry_count: int = 3,
                 rate_limit: Optional[int] = None):
        self.base_url = base_url.rstrip('/')
        self.authentication = authentication
        self.credentials = credentials or {}
        self.timeout = timeout
        self.retry_count = retry_count
        self.rate_limit = rate_limit
        
        # Initialize components
        self.rate_limiter = RateLimiter(rate_limit) if rate_limit else None
        self.cache = APICache()
        self.session = None
        
        # SSL context for secure connections
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the API client"""
        connector = aiohttp.TCPConnector(
            ssl=self.ssl_context,
            limit=100,
            limit_per_host=30
        )
        
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=self._get_default_headers()
        )
    
    async def close(self):
        """Close the API client"""
        if self.session:
            await self.session.close()
    
    def _get_default_headers(self) -> Dict[str, str]:
        """Get default headers"""
        headers = {
            'User-Agent': 'HeyGen-AI-Client/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Add authentication headers
        if self.authentication == AuthenticationMethod.API_KEY:
            if 'api_key' in self.credentials:
                headers['X-API-Key'] = self.credentials['api_key']
        elif self.authentication == AuthenticationMethod.JWT_TOKEN:
            if 'token' in self.credentials:
                headers['Authorization'] = f"Bearer {self.credentials['token']}"
        elif self.authentication == AuthenticationMethod.BASIC_AUTH:
            if 'username' in self.credentials and 'password' in self.credentials:
                import base64
                credentials = f"{self.credentials['username']}:{self.credentials['password']}"
                encoded = base64.b64encode(credentials.encode()).decode()
                headers['Authorization'] = f"Basic {encoded}"
        
        return headers
    
    async def request(self, 
                     method: str,
                     endpoint: str,
                     data: Optional[Any] = None,
                     params: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     cache_ttl: Optional[int] = None,
                     use_cache: bool = True) -> APIResponse:
        """Make API request with advanced features"""
        start_time = time.time()
        request_id = secrets.token_urlsafe(16)
        
        try:
            # Build URL
            url = urljoin(self.base_url, endpoint.lstrip('/'))
            
            # Check rate limiting
            if self.rate_limiter:
                if not await self.rate_limiter.is_allowed(url):
                    retry_after = await self.rate_limiter.get_retry_after(url)
                    return APIResponse(
                        success=False,
                        error=f"Rate limit exceeded. Retry after {retry_after} seconds",
                        status_code=429,
                        response_time=time.time() - start_time,
                        request_id=request_id
                    )
            
            # Check cache
            cache_key = None
            if use_cache and method.upper() == 'GET':
                cache_key = f"{method}:{url}:{hashlib.md5(str(params or {}).encode()).hexdigest()}"
                cached_response = await self.cache.get(cache_key)
                if cached_response:
                    return APIResponse(
                        success=True,
                        data=cached_response,
                        status_code=200,
                        response_time=time.time() - start_time,
                        request_id=request_id,
                        metadata={'cached': True}
                    )
            
            # Prepare request
            request_headers = self._get_default_headers()
            if headers:
                request_headers.update(headers)
            
            # Make request with retries
            last_exception = None
            for attempt in range(self.retry_count + 1):
                try:
                    async with self.session.request(
                        method=method,
                        url=url,
                        json=data,
                        params=params,
                        headers=request_headers
                    ) as response:
                        
                        response_data = await response.json() if response.content_type == 'application/json' else await response.text()
                        
                        api_response = APIResponse(
                            success=200 <= response.status < 300,
                            data=response_data,
                            error=None if 200 <= response.status < 300 else f"HTTP {response.status}",
                            status_code=response.status,
                            response_time=time.time() - start_time,
                            request_id=request_id
                        )
                        
                        # Cache successful GET requests
                        if use_cache and method.upper() == 'GET' and api_response.success and cache_key:
                            await self.cache.set(cache_key, response_data, cache_ttl)
                        
                        return api_response
                
                except Exception as e:
                    last_exception = e
                    if attempt < self.retry_count:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue
                    else:
                        break
            
            # All retries failed
            return APIResponse(
                success=False,
                error=f"Request failed after {self.retry_count + 1} attempts: {str(last_exception)}",
                status_code=0,
                response_time=time.time() - start_time,
                request_id=request_id
            )
        
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Request error: {str(e)}",
                status_code=0,
                response_time=time.time() - start_time,
                request_id=request_id
            )
    
    async def get(self, endpoint: str, **kwargs) -> APIResponse:
        """GET request"""
        return await self.request('GET', endpoint, **kwargs)
    
    async def post(self, endpoint: str, data: Any = None, **kwargs) -> APIResponse:
        """POST request"""
        return await self.request('POST', endpoint, data=data, **kwargs)
    
    async def put(self, endpoint: str, data: Any = None, **kwargs) -> APIResponse:
        """PUT request"""
        return await self.request('PUT', endpoint, data=data, **kwargs)
    
    async def delete(self, endpoint: str, **kwargs) -> APIResponse:
        """DELETE request"""
        return await self.request('DELETE', endpoint, **kwargs)

class IntegrationManager:
    """Advanced integration management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "integrations.yaml"
        self.integrations = {}
        self.clients = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize integration manager"""
        try:
            await self._load_integrations()
            await self._initialize_clients()
            self.initialized = True
            logger.info("✅ Integration Manager initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Integration Manager: {e}")
            raise
    
    async def _load_integrations(self):
        """Load integration configurations"""
        if Path(self.config_path).exists():
            async with aiofiles.open(self.config_path, 'r') as f:
                content = await f.read()
                config_data = yaml.safe_load(content)
                
                for integration_name, config in config_data.get('integrations', {}).items():
                    integration_config = IntegrationConfig(
                        name=integration_name,
                        type=IntegrationType(config['type']),
                        base_url=config['base_url'],
                        authentication=AuthenticationMethod(config['authentication']),
                        credentials=config.get('credentials', {}),
                        timeout=config.get('timeout', 30),
                        retry_count=config.get('retry_count', 3),
                        rate_limit=config.get('rate_limit'),
                        enabled=config.get('enabled', True),
                        metadata=config.get('metadata', {})
                    )
                    
                    # Load endpoints
                    for endpoint_config in config.get('endpoints', []):
                        endpoint = APIEndpoint(
                            name=endpoint_config['name'],
                            url=endpoint_config['url'],
                            method=endpoint_config.get('method', 'GET'),
                            version=APIVersion(endpoint_config.get('version', 'latest')),
                            authentication=AuthenticationMethod(endpoint_config.get('authentication', 'api_key')),
                            headers=endpoint_config.get('headers', {}),
                            timeout=endpoint_config.get('timeout', 30),
                            retry_count=endpoint_config.get('retry_count', 3),
                            rate_limit=endpoint_config.get('rate_limit'),
                            cache_ttl=endpoint_config.get('cache_ttl', 300),
                            enabled=endpoint_config.get('enabled', True)
                        )
                        integration_config.endpoints.append(endpoint)
                    
                    self.integrations[integration_name] = integration_config
        else:
            await self._create_default_config()
    
    async def _create_default_config(self):
        """Create default integration configuration"""
        default_config = {
            'integrations': {
                'heygen_ai_core': {
                    'type': 'rest_api',
                    'base_url': 'http://localhost:8000/api/v1',
                    'authentication': 'api_key',
                    'credentials': {
                        'api_key': 'your-api-key-here'
                    },
                    'timeout': 30,
                    'retry_count': 3,
                    'rate_limit': 100,
                    'enabled': True,
                    'endpoints': [
                        {
                            'name': 'health_check',
                            'url': '/health',
                            'method': 'GET',
                            'cache_ttl': 60
                        },
                        {
                            'name': 'generate_video',
                            'url': '/generate/video',
                            'method': 'POST',
                            'cache_ttl': 0
                        },
                        {
                            'name': 'get_status',
                            'url': '/status/{job_id}',
                            'method': 'GET',
                            'cache_ttl': 30
                        }
                    ]
                }
            }
        }
        
        async with aiofiles.open(self.config_path, 'w') as f:
            await f.write(yaml.dump(default_config, default_flow_style=False))
        
        logger.info(f"Created default integration configuration: {self.config_path}")
    
    async def _initialize_clients(self):
        """Initialize API clients for integrations"""
        for integration_name, config in self.integrations.items():
            if not config.enabled:
                continue
            
            try:
                client = APIClient(
                    base_url=config.base_url,
                    authentication=config.authentication,
                    credentials=config.credentials,
                    timeout=config.timeout,
                    retry_count=config.retry_count,
                    rate_limit=config.rate_limit
                )
                
                await client.initialize()
                self.clients[integration_name] = client
                
                logger.info(f"✅ Initialized client for {integration_name}")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize client for {integration_name}: {e}")
    
    async def call_endpoint(self, 
                           integration_name: str,
                           endpoint_name: str,
                           data: Optional[Any] = None,
                           params: Optional[Dict[str, Any]] = None,
                           **kwargs) -> APIResponse:
        """Call specific endpoint on integration"""
        if not self.initialized:
            return APIResponse(success=False, error="Integration Manager not initialized")
        
        if integration_name not in self.integrations:
            return APIResponse(success=False, error=f"Integration {integration_name} not found")
        
        if integration_name not in self.clients:
            return APIResponse(success=False, error=f"Client for {integration_name} not available")
        
        integration = self.integrations[integration_name]
        client = self.clients[integration_name]
        
        # Find endpoint
        endpoint = None
        for ep in integration.endpoints:
            if ep.name == endpoint_name and ep.enabled:
                endpoint = ep
                break
        
        if not endpoint:
            return APIResponse(success=False, error=f"Endpoint {endpoint_name} not found or disabled")
        
        # Prepare request parameters
        request_kwargs = {
            'data': data,
            'params': params,
            'headers': endpoint.headers,
            'cache_ttl': endpoint.cache_ttl,
            'use_cache': endpoint.cache_ttl > 0
        }
        request_kwargs.update(kwargs)
        
        # Make request
        if endpoint.method.upper() == 'GET':
            response = await client.get(endpoint.url, **request_kwargs)
        elif endpoint.method.upper() == 'POST':
            response = await client.post(endpoint.url, **request_kwargs)
        elif endpoint.method.upper() == 'PUT':
            response = await client.put(endpoint.url, **request_kwargs)
        elif endpoint.method.upper() == 'DELETE':
            response = await client.delete(endpoint.url, **request_kwargs)
        else:
            response = await client.request(endpoint.method, endpoint.url, **request_kwargs)
        
        return response
    
    async def health_check(self, integration_name: Optional[str] = None) -> Dict[str, Any]:
        """Perform health check on integrations"""
        health_results = {}
        
        integrations_to_check = [integration_name] if integration_name else self.integrations.keys()
        
        for name in integrations_to_check:
            if name not in self.integrations:
                health_results[name] = {'status': 'not_found', 'error': 'Integration not found'}
                continue
            
            if not self.integrations[name].enabled:
                health_results[name] = {'status': 'disabled', 'error': 'Integration disabled'}
                continue
            
            if name not in self.clients:
                health_results[name] = {'status': 'client_unavailable', 'error': 'Client not initialized'}
                continue
            
            try:
                # Try to call health endpoint
                response = await self.call_endpoint(name, 'health_check')
                health_results[name] = {
                    'status': 'healthy' if response.success else 'unhealthy',
                    'response_time': response.response_time,
                    'status_code': response.status_code,
                    'error': response.error
                }
            except Exception as e:
                health_results[name] = {
                    'status': 'error',
                    'error': str(e)
                }
        
        return health_results
    
    async def shutdown(self):
        """Shutdown integration manager"""
        for client in self.clients.values():
            await client.close()
        
        self.clients.clear()
        self.initialized = False
        logger.info("✅ Integration Manager shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced API integration system"""
    print("🌐 HeyGen AI - Advanced API Integration Demo")
    print("=" * 60)
    
    # Initialize integration manager
    integration_manager = IntegrationManager()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Integration Manager...")
        await integration_manager.initialize()
        print("✅ Integration Manager initialized successfully")
        
        # List available integrations
        print(f"\n📋 Available Integrations: {len(integration_manager.integrations)}")
        for name, config in integration_manager.integrations.items():
            print(f"  - {name}: {config.type.value} ({config.base_url})")
        
        # Perform health check
        print("\n🏥 Health Check:")
        health_results = await integration_manager.health_check()
        for name, result in health_results.items():
            status_icon = "✅" if result['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {name}: {result['status']}")
            if 'error' in result and result['error']:
                print(f"    Error: {result['error']}")
        
        # Example API call
        print("\n📡 Example API Call:")
        response = await integration_manager.call_endpoint(
            'heygen_ai_core',
            'health_check'
        )
        
        if response.success:
            print(f"  ✅ Success: {response.status_code}")
            print(f"  📊 Response Time: {response.response_time:.3f}s")
            print(f"  🆔 Request ID: {response.request_id}")
        else:
            print(f"  ❌ Failed: {response.error}")
        
        # Demonstrate caching
        print("\n💾 Cache Demonstration:")
        start_time = time.time()
        response1 = await integration_manager.call_endpoint('heygen_ai_core', 'health_check')
        time1 = time.time() - start_time
        
        start_time = time.time()
        response2 = await integration_manager.call_endpoint('heygen_ai_core', 'health_check')
        time2 = time.time() - start_time
        
        print(f"  First call: {time1:.3f}s")
        print(f"  Second call: {time2:.3f}s (cached: {response2.metadata.get('cached', False)})")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
    
    finally:
        # Shutdown
        await integration_manager.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


