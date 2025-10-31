#!/usr/bin/env python3
"""
🔗 HeyGen AI - Advanced API Integration System
=============================================

This module implements a comprehensive API integration system that provides
unified access to multiple external APIs, intelligent rate limiting,
caching, and advanced error handling for the HeyGen AI system.
"""

import asyncio
import logging
import time
import json
import uuid
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import threading
import queue
import hashlib
import secrets
import base64
import hmac
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import aiohttp
import asyncio
from aiohttp import web, WSMsgType
import ssl
import certifi
import redis
import pickle
from collections import defaultdict
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor
import jwt
import requests
from urllib.parse import urljoin, urlparse
import backoff
import tenacity
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIType(str, Enum):
    """API types"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    SOAP = "soap"
    CUSTOM = "custom"

class AuthenticationType(str, Enum):
    """Authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    CUSTOM = "custom"

class RateLimitStrategy(str, Enum):
    """Rate limiting strategies"""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"

class CacheStrategy(str, Enum):
    """Caching strategies"""
    NO_CACHE = "no_cache"
    MEMORY = "memory"
    REDIS = "redis"
    DATABASE = "database"
    CDN = "cdn"
    HYBRID = "hybrid"

@dataclass
class APIConfig:
    """API configuration"""
    api_id: str
    name: str
    base_url: str
    api_type: APIType
    authentication: AuthenticationType
    rate_limit: int = 100  # requests per minute
    timeout: int = 30  # seconds
    retry_attempts: int = 3
    cache_ttl: int = 300  # seconds
    headers: Dict[str, str] = field(default_factory=dict)
    auth_config: Dict[str, Any] = field(default_factory=dict)
    rate_limit_strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    cache_strategy: CacheStrategy = CacheStrategy.MEMORY
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIRequest:
    """API request representation"""
    request_id: str
    api_id: str
    method: str
    endpoint: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    data: Any = None
    timeout: int = 30
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIResponse:
    """API response representation"""
    request_id: str
    api_id: str
    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    data: Any = None
    error: Optional[str] = None
    response_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RateLimitInfo:
    """Rate limit information"""
    api_id: str
    current_requests: int
    max_requests: int
    window_start: datetime
    window_duration: int  # seconds
    reset_time: datetime
    remaining_requests: int

class RateLimiter:
    """Advanced rate limiting system"""
    
    def __init__(self):
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.request_counts: Dict[str, List[datetime]] = defaultdict(list)
        self.initialized = False
    
    async def initialize(self):
        """Initialize rate limiter"""
        self.initialized = True
        logger.info("✅ Rate Limiter initialized")
    
    async def check_rate_limit(self, api_id: str, config: APIConfig) -> bool:
        """Check if request is within rate limit"""
        if not self.initialized:
            return True
        
        try:
            current_time = datetime.now()
            
            # Clean old requests
            cutoff_time = current_time - timedelta(minutes=1)
            self.request_counts[api_id] = [
                req_time for req_time in self.request_counts[api_id]
                if req_time > cutoff_time
            ]
            
            # Check if within limit
            current_count = len(self.request_counts[api_id])
            if current_count >= config.rate_limit:
                return False
            
            # Add current request
            self.request_counts[api_id].append(current_time)
            return True
            
        except Exception as e:
            logger.error(f"❌ Rate limit check failed: {e}")
            return True  # Allow request on error
    
    async def get_rate_limit_info(self, api_id: str, config: APIConfig) -> RateLimitInfo:
        """Get rate limit information"""
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=1)
        
        # Clean old requests
        self.request_counts[api_id] = [
            req_time for req_time in self.request_counts[api_id]
            if req_time > cutoff_time
        ]
        
        current_requests = len(self.request_counts[api_id])
        remaining_requests = max(0, config.rate_limit - current_requests)
        
        # Calculate reset time
        if self.request_counts[api_id]:
            oldest_request = min(self.request_counts[api_id])
            reset_time = oldest_request + timedelta(minutes=1)
        else:
            reset_time = current_time
        
        return RateLimitInfo(
            api_id=api_id,
            current_requests=current_requests,
            max_requests=config.rate_limit,
            window_start=cutoff_time,
            window_duration=60,
            reset_time=reset_time,
            remaining_requests=remaining_requests
        )

class CacheManager:
    """Advanced caching system"""
    
    def __init__(self):
        self.memory_cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        self.redis_client: Optional[redis.Redis] = None
        self.initialized = False
    
    async def initialize(self, redis_url: str = "redis://localhost:6379"):
        """Initialize cache manager"""
        try:
            # Initialize Redis
            self.redis_client = redis.from_url(redis_url)
            await asyncio.get_event_loop().run_in_executor(None, self.redis_client.ping)
            
            self.initialized = True
            logger.info("✅ Cache Manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Redis not available, using memory cache only: {e}")
            self.initialized = True
    
    async def get(self, key: str, strategy: CacheStrategy = CacheStrategy.MEMORY) -> Optional[Any]:
        """Get value from cache"""
        if not self.initialized:
            return None
        
        try:
            if strategy == CacheStrategy.MEMORY:
                return self.memory_cache.get(key)
            elif strategy == CacheStrategy.REDIS and self.redis_client:
                data = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, key
                )
                if data:
                    return pickle.loads(data)
            return None
        except Exception as e:
            logger.error(f"❌ Cache get failed: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 300, 
                 strategy: CacheStrategy = CacheStrategy.MEMORY) -> bool:
        """Set value in cache"""
        if not self.initialized:
            return False
        
        try:
            if strategy == CacheStrategy.MEMORY:
                self.memory_cache[key] = value
                self.cache_timestamps[key] = datetime.now()
                
                # Clean expired entries
                await self._clean_expired_memory_cache(ttl)
                return True
            elif strategy == CacheStrategy.REDIS and self.redis_client:
                serialized_data = pickle.dumps(value)
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.setex, key, ttl, serialized_data
                )
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Cache set failed: {e}")
            return False
    
    async def delete(self, key: str, strategy: CacheStrategy = CacheStrategy.MEMORY) -> bool:
        """Delete value from cache"""
        if not self.initialized:
            return False
        
        try:
            if strategy == CacheStrategy.MEMORY:
                self.memory_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
                return True
            elif strategy == CacheStrategy.REDIS and self.redis_client:
                await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.delete, key
                )
                return True
            return False
        except Exception as e:
            logger.error(f"❌ Cache delete failed: {e}")
            return False
    
    async def _clean_expired_memory_cache(self, ttl: int):
        """Clean expired entries from memory cache"""
        current_time = datetime.now()
        expired_keys = [
            key for key, timestamp in self.cache_timestamps.items()
            if (current_time - timestamp).seconds > ttl
        ]
        
        for key in expired_keys:
            self.memory_cache.pop(key, None)
            self.cache_timestamps.pop(key, None)

class AuthenticationManager:
    """Advanced authentication management"""
    
    def __init__(self):
        self.auth_tokens: Dict[str, Dict[str, Any]] = {}
        self.initialized = False
    
    async def initialize(self):
        """Initialize authentication manager"""
        self.initialized = True
        logger.info("✅ Authentication Manager initialized")
    
    async def authenticate_request(self, config: APIConfig, 
                                 request: APIRequest) -> APIRequest:
        """Authenticate API request"""
        if not self.initialized:
            return request
        
        try:
            if config.authentication == AuthenticationType.API_KEY:
                api_key = config.auth_config.get('api_key')
                if api_key:
                    request.headers['X-API-Key'] = api_key
            
            elif config.authentication == AuthenticationType.BEARER_TOKEN:
                token = config.auth_config.get('token')
                if token:
                    request.headers['Authorization'] = f'Bearer {token}'
            
            elif config.authentication == AuthenticationType.BASIC_AUTH:
                username = config.auth_config.get('username')
                password = config.auth_config.get('password')
                if username and password:
                    credentials = base64.b64encode(f'{username}:{password}'.encode()).decode()
                    request.headers['Authorization'] = f'Basic {credentials}'
            
            elif config.authentication == AuthenticationType.JWT:
                jwt_token = config.auth_config.get('jwt_token')
                if jwt_token:
                    request.headers['Authorization'] = f'Bearer {jwt_token}'
            
            elif config.authentication == AuthenticationType.OAUTH2:
                access_token = config.auth_config.get('access_token')
                if access_token:
                    request.headers['Authorization'] = f'Bearer {access_token}'
            
            return request
            
        except Exception as e:
            logger.error(f"❌ Authentication failed: {e}")
            return request

class APIClient:
    """Advanced API client"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize API client"""
        try:
            # Create SSL context
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            
            # Create session with timeout and SSL
            timeout = aiohttp.ClientTimeout(total=30)
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': 'HeyGen-AI-Client/1.0'}
            )
            
            self.initialized = True
            logger.info("✅ API Client initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize API client: {e}")
            raise
    
    async def make_request(self, request: APIRequest, config: APIConfig) -> APIResponse:
        """Make API request with retry logic"""
        if not self.initialized or not self.session:
            return APIResponse(
                request_id=request.request_id,
                api_id=request.api_id,
                status_code=0,
                error="Client not initialized"
            )
        
        start_time = time.time()
        
        try:
            # Build URL
            url = urljoin(config.base_url, request.endpoint)
            
            # Make request with retry
            response = await self._make_request_with_retry(
                request, config, url
            )
            
            # Process response
            response_time = time.time() - start_time
            
            return APIResponse(
                request_id=request.request_id,
                api_id=request.api_id,
                status_code=response.status,
                headers=dict(response.headers),
                data=await response.json() if response.content_type == 'application/json' else await response.text(),
                response_time=response_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"❌ API request failed: {e}")
            
            return APIResponse(
                request_id=request.request_id,
                api_id=request.api_id,
                status_code=0,
                error=str(e),
                response_time=response_time,
                timestamp=datetime.now()
            )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def _make_request_with_retry(self, request: APIRequest, 
                                     config: APIConfig, url: str) -> aiohttp.ClientResponse:
        """Make request with retry logic"""
        async with self.session.request(
            method=request.method,
            url=url,
            headers=request.headers,
            params=request.params,
            json=request.data if isinstance(request.data, dict) else None,
            data=request.data if not isinstance(request.data, dict) else None,
            timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as response:
            return response
    
    async def shutdown(self):
        """Shutdown API client"""
        if self.session:
            await self.session.close()
        self.initialized = False

class AdvancedAPIIntegrationSystem:
    """Main API integration system"""
    
    def __init__(self):
        self.rate_limiter = RateLimiter()
        self.cache_manager = CacheManager()
        self.auth_manager = AuthenticationManager()
        self.api_client = APIClient()
        self.api_configs: Dict[str, APIConfig] = {}
        self.request_history: List[APIRequest] = []
        self.response_history: List[APIResponse] = []
        self.initialized = False
    
    async def initialize(self):
        """Initialize API integration system"""
        try:
            logger.info("🔗 Initializing Advanced API Integration System...")
            
            # Initialize components
            await self.rate_limiter.initialize()
            await self.cache_manager.initialize()
            await self.auth_manager.initialize()
            await self.api_client.initialize()
            
            # Load default API configurations
            await self._load_default_configs()
            
            self.initialized = True
            logger.info("✅ Advanced API Integration System initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize API integration system: {e}")
            raise
    
    async def _load_default_configs(self):
        """Load default API configurations"""
        default_configs = [
            APIConfig(
                api_id="openai",
                name="OpenAI API",
                base_url="https://api.openai.com/v1",
                api_type=APIType.REST,
                authentication=AuthenticationType.BEARER_TOKEN,
                rate_limit=60,
                timeout=30,
                auth_config={"token": "your-openai-token"}
            ),
            APIConfig(
                api_id="anthropic",
                name="Anthropic API",
                base_url="https://api.anthropic.com/v1",
                api_type=APIType.REST,
                authentication=AuthenticationType.API_KEY,
                rate_limit=50,
                timeout=30,
                auth_config={"api_key": "your-anthropic-key"}
            ),
            APIConfig(
                api_id="huggingface",
                name="Hugging Face API",
                base_url="https://api-inference.huggingface.co",
                api_type=APIType.REST,
                authentication=AuthenticationType.BEARER_TOKEN,
                rate_limit=100,
                timeout=30,
                auth_config={"token": "your-hf-token"}
            )
        ]
        
        for config in default_configs:
            self.api_configs[config.api_id] = config
    
    async def register_api(self, config: APIConfig) -> bool:
        """Register a new API configuration"""
        if not self.initialized:
            return False
        
        try:
            self.api_configs[config.api_id] = config
            logger.info(f"✅ API registered: {config.name}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to register API: {e}")
            return False
    
    async def make_request(self, api_id: str, method: str, endpoint: str,
                          headers: Dict[str, str] = None, params: Dict[str, Any] = None,
                          data: Any = None, timeout: int = 30, 
                          use_cache: bool = True) -> APIResponse:
        """Make API request"""
        if not self.initialized or api_id not in self.api_configs:
            return APIResponse(
                request_id=str(uuid.uuid4()),
                api_id=api_id,
                status_code=0,
                error=f"API {api_id} not found"
            )
        
        try:
            config = self.api_configs[api_id]
            
            # Create request
            request = APIRequest(
                request_id=str(uuid.uuid4()),
                api_id=api_id,
                method=method.upper(),
                endpoint=endpoint,
                headers=headers or {},
                params=params or {},
                data=data,
                timeout=timeout
            )
            
            # Check cache first
            if use_cache and config.cache_strategy != CacheStrategy.NO_CACHE:
                cache_key = self._generate_cache_key(request)
                cached_response = await self.cache_manager.get(
                    cache_key, config.cache_strategy
                )
                if cached_response:
                    cached_response.cached = True
                    return cached_response
            
            # Check rate limit
            rate_limit_ok = await self.rate_limiter.check_rate_limit(api_id, config)
            if not rate_limit_ok:
                return APIResponse(
                    request_id=request.request_id,
                    api_id=api_id,
                    status_code=429,
                    error="Rate limit exceeded"
                )
            
            # Authenticate request
            request = await self.auth_manager.authenticate_request(config, request)
            
            # Make request
            response = await self.api_client.make_request(request, config)
            
            # Cache successful responses
            if use_cache and response.status_code < 400 and config.cache_strategy != CacheStrategy.NO_CACHE:
                cache_key = self._generate_cache_key(request)
                await self.cache_manager.set(
                    cache_key, response, config.cache_ttl, config.cache_strategy
                )
            
            # Store in history
            self.request_history.append(request)
            self.response_history.append(response)
            
            # Keep only last 1000 requests/responses
            if len(self.request_history) > 1000:
                self.request_history = self.request_history[-1000:]
            if len(self.response_history) > 1000:
                self.response_history = self.response_history[-1000:]
            
            return response
            
        except Exception as e:
            logger.error(f"❌ API request failed: {e}")
            return APIResponse(
                request_id=str(uuid.uuid4()),
                api_id=api_id,
                status_code=0,
                error=str(e)
            )
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'api_id': request.api_id,
            'method': request.method,
            'endpoint': request.endpoint,
            'params': request.params,
            'data': request.data
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get_rate_limit_info(self, api_id: str) -> Optional[RateLimitInfo]:
        """Get rate limit information for API"""
        if not self.initialized or api_id not in self.api_configs:
            return None
        
        config = self.api_configs[api_id]
        return await self.rate_limiter.get_rate_limit_info(api_id, config)
    
    async def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics"""
        if not self.initialized:
            return {}
        
        try:
            # Calculate statistics
            total_requests = len(self.request_history)
            successful_requests = len([r for r in self.response_history if r.status_code < 400])
            failed_requests = total_requests - successful_requests
            
            # Group by API
            api_stats = defaultdict(lambda: {'requests': 0, 'successful': 0, 'failed': 0})
            for response in self.response_history:
                api_stats[response.api_id]['requests'] += 1
                if response.status_code < 400:
                    api_stats[response.api_id]['successful'] += 1
                else:
                    api_stats[response.api_id]['failed'] += 1
            
            # Calculate average response time
            avg_response_time = 0
            if self.response_history:
                avg_response_time = np.mean([r.response_time for r in self.response_history])
            
            return {
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'failed_requests': failed_requests,
                'success_rate': (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                'average_response_time': avg_response_time,
                'api_statistics': dict(api_stats),
                'registered_apis': len(self.api_configs)
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get statistics: {e}")
            return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'initialized': self.initialized,
            'rate_limiter_ready': self.rate_limiter.initialized,
            'cache_manager_ready': self.cache_manager.initialized,
            'auth_manager_ready': self.auth_manager.initialized,
            'api_client_ready': self.api_client.initialized,
            'registered_apis': len(self.api_configs),
            'total_requests': len(self.request_history),
            'total_responses': len(self.response_history),
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self):
        """Shutdown API integration system"""
        if self.api_client.initialized:
            await self.api_client.shutdown()
        self.initialized = False
        logger.info("✅ Advanced API Integration System shutdown complete")

# Example usage and demonstration
async def main():
    """Demonstrate the advanced API integration system"""
    print("🔗 HeyGen AI - Advanced API Integration System Demo")
    print("=" * 70)
    
    # Initialize system
    api_system = AdvancedAPIIntegrationSystem()
    
    try:
        # Initialize the system
        print("\n🚀 Initializing Advanced API Integration System...")
        await api_system.initialize()
        print("✅ Advanced API Integration System initialized successfully")
        
        # Get system status
        print("\n📊 System Status:")
        status = await api_system.get_system_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Register a custom API
        print("\n🔧 Registering Custom API...")
        
        custom_config = APIConfig(
            api_id="jsonplaceholder",
            name="JSONPlaceholder API",
            base_url="https://jsonplaceholder.typicode.com",
            api_type=APIType.REST,
            authentication=AuthenticationType.NONE,
            rate_limit=100,
            timeout=10,
            cache_strategy=CacheStrategy.MEMORY
        )
        
        await api_system.register_api(custom_config)
        print("  ✅ Custom API registered")
        
        # Make some API requests
        print("\n📡 Making API Requests...")
        
        # GET request
        response1 = await api_system.make_request(
            "jsonplaceholder", "GET", "/posts/1"
        )
        print(f"  GET /posts/1: {response1.status_code} - {response1.response_time:.3f}s")
        
        # POST request
        post_data = {
            "title": "HeyGen AI Test",
            "body": "This is a test post from HeyGen AI",
            "userId": 1
        }
        response2 = await api_system.make_request(
            "jsonplaceholder", "POST", "/posts",
            data=post_data
        )
        print(f"  POST /posts: {response2.status_code} - {response2.response_time:.3f}s")
        
        # Test caching
        print("\n💾 Testing Cache...")
        
        response3 = await api_system.make_request(
            "jsonplaceholder", "GET", "/posts/2",
            use_cache=True
        )
        print(f"  First request: {response3.status_code} - {response3.response_time:.3f}s - Cached: {response3.cached}")
        
        response4 = await api_system.make_request(
            "jsonplaceholder", "GET", "/posts/2",
            use_cache=True
        )
        print(f"  Second request: {response4.status_code} - {response4.response_time:.3f}s - Cached: {response4.cached}")
        
        # Test rate limiting
        print("\n⏱️ Testing Rate Limiting...")
        
        for i in range(5):
            response = await api_system.make_request(
                "jsonplaceholder", "GET", f"/posts/{i+1}"
            )
            print(f"  Request {i+1}: {response.status_code} - {response.response_time:.3f}s")
        
        # Get rate limit info
        print("\n📊 Rate Limit Information:")
        rate_limit_info = await api_system.get_rate_limit_info("jsonplaceholder")
        if rate_limit_info:
            print(f"  Current requests: {rate_limit_info.current_requests}")
            print(f"  Max requests: {rate_limit_info.max_requests}")
            print(f"  Remaining: {rate_limit_info.remaining_requests}")
            print(f"  Reset time: {rate_limit_info.reset_time}")
        
        # Get statistics
        print("\n📈 API Statistics:")
        stats = await api_system.get_api_statistics()
        print(f"  Total requests: {stats.get('total_requests', 0)}")
        print(f"  Successful requests: {stats.get('successful_requests', 0)}")
        print(f"  Failed requests: {stats.get('failed_requests', 0)}")
        print(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
        print(f"  Average response time: {stats.get('average_response_time', 0):.3f}s")
        print(f"  Registered APIs: {stats.get('registered_apis', 0)}")
        
        # Show API statistics by API
        api_stats = stats.get('api_statistics', {})
        if api_stats:
            print(f"\n  API Statistics:")
            for api_id, api_stat in api_stats.items():
                print(f"    {api_id}:")
                print(f"      Requests: {api_stat['requests']}")
                print(f"      Successful: {api_stat['successful']}")
                print(f"      Failed: {api_stat['failed']}")
        
    except Exception as e:
        print(f"❌ Demo Error: {e}")
        logger.error(f"Demo failed: {e}")
    
    finally:
        # Shutdown
        await api_system.shutdown()
        print("\n✅ Demo completed")

if __name__ == "__main__":
    asyncio.run(main())


