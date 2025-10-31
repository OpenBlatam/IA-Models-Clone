"""
API Integration System
======================

Advanced API integration system for AI model analysis with comprehensive
API management, authentication, rate limiting, and monitoring capabilities.
"""

import asyncio
import logging
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from enum import Enum
import hashlib
import aiohttp
import requests
import websockets
import ssl
import jwt
import hmac
import hashlib as hashlib_module
import base64
from urllib.parse import urlparse, urljoin
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class APIType(str, Enum):
    """API types"""
    REST = "rest"
    GRAPHQL = "graphql"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    SOAP = "soap"
    WEBHOOK = "webhook"
    STREAMING = "streaming"
    BATCH = "batch"


class AuthenticationType(str, Enum):
    """Authentication types"""
    NONE = "none"
    API_KEY = "api_key"
    BEARER_TOKEN = "bearer_token"
    BASIC_AUTH = "basic_auth"
    OAUTH2 = "oauth2"
    JWT = "jwt"
    HMAC = "hmac"
    CUSTOM = "custom"


class RateLimitType(str, Enum):
    """Rate limit types"""
    REQUESTS_PER_SECOND = "requests_per_second"
    REQUESTS_PER_MINUTE = "requests_per_minute"
    REQUESTS_PER_HOUR = "requests_per_hour"
    REQUESTS_PER_DAY = "requests_per_day"
    BANDWIDTH_PER_SECOND = "bandwidth_per_second"
    CONCURRENT_REQUESTS = "concurrent_requests"


class APIStatus(str, Enum):
    """API status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    AUTHENTICATION_FAILED = "authentication_failed"


@dataclass
class APIConfiguration:
    """API configuration"""
    api_id: str
    name: str
    description: str
    base_url: str
    api_type: APIType
    authentication: AuthenticationType
    auth_config: Dict[str, Any]
    rate_limits: Dict[RateLimitType, int]
    headers: Dict[str, str]
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 1
    health_check_endpoint: str = "/health"
    status: APIStatus = APIStatus.ACTIVE
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class APIRequest:
    """API request"""
    request_id: str
    api_id: str
    method: str
    endpoint: str
    headers: Dict[str, str]
    params: Dict[str, Any]
    data: Any
    timeout: int
    timestamp: datetime
    retry_count: int = 0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class APIResponse:
    """API response"""
    request_id: str
    api_id: str
    status_code: int
    headers: Dict[str, str]
    data: Any
    response_time: float
    timestamp: datetime
    error_message: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class APIMetrics:
    """API metrics"""
    api_id: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    success_rate: float
    rate_limit_hits: int
    authentication_failures: int
    last_request_time: datetime
    uptime_percentage: float


class APIIntegrationSystem:
    """Advanced API integration system for AI model analysis"""
    
    def __init__(self, max_apis: int = 100, max_requests_per_minute: int = 1000):
        self.max_apis = max_apis
        self.max_requests_per_minute = max_requests_per_minute
        
        self.api_configurations: Dict[str, APIConfiguration] = {}
        self.api_requests: List[APIRequest] = []
        self.api_responses: List[APIResponse] = []
        self.api_metrics: Dict[str, APIMetrics] = {}
        
        # Rate limiting
        self.rate_limit_tracker: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.rate_limit_blocks: Dict[str, datetime] = {}
        
        # Connection pooling
        self.session_pool: Dict[str, aiohttp.ClientSession] = {}
        
        # WebSocket connections
        self.websocket_connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        
        # Background tasks
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.running = True
        
        # Start background tasks
        self._start_background_tasks()
    
    async def register_api(self, 
                         name: str,
                         description: str,
                         base_url: str,
                         api_type: APIType,
                         authentication: AuthenticationType,
                         auth_config: Dict[str, Any] = None,
                         rate_limits: Dict[RateLimitType, int] = None,
                         headers: Dict[str, str] = None,
                         timeout: int = 30) -> APIConfiguration:
        """Register new API"""
        try:
            api_id = hashlib.md5(f"{name}_{base_url}_{datetime.now()}".encode()).hexdigest()
            
            if auth_config is None:
                auth_config = {}
            if rate_limits is None:
                rate_limits = {RateLimitType.REQUESTS_PER_MINUTE: 100}
            if headers is None:
                headers = {}
            
            config = APIConfiguration(
                api_id=api_id,
                name=name,
                description=description,
                base_url=base_url,
                api_type=api_type,
                authentication=authentication,
                auth_config=auth_config,
                rate_limits=rate_limits,
                headers=headers,
                timeout=timeout
            )
            
            self.api_configurations[api_id] = config
            
            # Initialize metrics
            self.api_metrics[api_id] = APIMetrics(
                api_id=api_id,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0.0,
                success_rate=0.0,
                rate_limit_hits=0,
                authentication_failures=0,
                last_request_time=datetime.now(),
                uptime_percentage=100.0
            )
            
            # Create session for REST APIs
            if api_type == APIType.REST:
                await self._create_session(api_id)
            
            logger.info(f"Registered API: {name} ({api_id})")
            
            return config
            
        except Exception as e:
            logger.error(f"Error registering API: {str(e)}")
            raise e
    
    async def make_request(self, 
                         api_id: str,
                         method: str,
                         endpoint: str,
                         headers: Dict[str, str] = None,
                         params: Dict[str, Any] = None,
                         data: Any = None,
                         timeout: int = None) -> APIResponse:
        """Make API request"""
        try:
            if api_id not in self.api_configurations:
                raise ValueError(f"API {api_id} not registered")
            
            config = self.api_configurations[api_id]
            
            # Check rate limits
            if not await self._check_rate_limits(api_id):
                raise Exception(f"Rate limit exceeded for API {api_id}")
            
            # Generate request ID
            request_id = hashlib.md5(f"{api_id}_{endpoint}_{datetime.now()}".encode()).hexdigest()
            
            # Prepare request
            if headers is None:
                headers = {}
            if params is None:
                params = {}
            if timeout is None:
                timeout = config.timeout
            
            # Add authentication headers
            auth_headers = await self._get_auth_headers(config)
            headers.update(auth_headers)
            headers.update(config.headers)
            
            # Create request object
            request = APIRequest(
                request_id=request_id,
                api_id=api_id,
                method=method.upper(),
                endpoint=endpoint,
                headers=headers,
                params=params,
                data=data,
                timeout=timeout
            )
            
            self.api_requests.append(request)
            
            # Make the actual request
            start_time = time.time()
            
            try:
                if config.api_type == APIType.REST:
                    response = await self._make_rest_request(config, request)
                elif config.api_type == APIType.GRAPHQL:
                    response = await self._make_graphql_request(config, request)
                elif config.api_type == APIType.WEBSOCKET:
                    response = await self._make_websocket_request(config, request)
                else:
                    raise ValueError(f"Unsupported API type: {config.api_type}")
                
                response_time = time.time() - start_time
                
                # Create response object
                api_response = APIResponse(
                    request_id=request_id,
                    api_id=api_id,
                    status_code=response.get("status_code", 200),
                    headers=response.get("headers", {}),
                    data=response.get("data"),
                    response_time=response_time,
                    timestamp=datetime.now()
                )
                
                self.api_responses.append(api_response)
                
                # Update metrics
                await self._update_metrics(api_id, api_response)
                
                logger.info(f"API request successful: {api_id} {method} {endpoint}")
                
                return api_response
                
            except Exception as e:
                response_time = time.time() - start_time
                
                # Create error response
                error_response = APIResponse(
                    request_id=request_id,
                    api_id=api_id,
                    status_code=500,
                    headers={},
                    data=None,
                    response_time=response_time,
                    timestamp=datetime.now(),
                    error_message=str(e)
                )
                
                self.api_responses.append(error_response)
                
                # Update metrics
                await self._update_metrics(api_id, error_response)
                
                logger.error(f"API request failed: {api_id} {method} {endpoint} - {str(e)}")
                
                raise e
            
        except Exception as e:
            logger.error(f"Error making API request: {str(e)}")
            raise e
    
    async def batch_request(self, 
                          requests: List[Dict[str, Any]],
                          max_concurrent: int = 10) -> List[APIResponse]:
        """Make batch API requests"""
        try:
            semaphore = asyncio.Semaphore(max_concurrent)
            
            async def make_single_request(request_data):
                async with semaphore:
                    return await self.make_request(**request_data)
            
            tasks = [make_single_request(req) for req in requests]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and convert to APIResponse objects
            valid_responses = []
            for response in responses:
                if isinstance(response, APIResponse):
                    valid_responses.append(response)
                elif isinstance(response, Exception):
                    # Create error response
                    error_response = APIResponse(
                        request_id="",
                        api_id="",
                        status_code=500,
                        headers={},
                        data=None,
                        response_time=0.0,
                        timestamp=datetime.now(),
                        error_message=str(response)
                    )
                    valid_responses.append(error_response)
            
            logger.info(f"Completed batch request: {len(valid_responses)} responses")
            
            return valid_responses
            
        except Exception as e:
            logger.error(f"Error making batch request: {str(e)}")
            return []
    
    async def stream_data(self, 
                        api_id: str,
                        endpoint: str,
                        callback: callable,
                        headers: Dict[str, str] = None) -> bool:
        """Stream data from API"""
        try:
            if api_id not in self.api_configurations:
                raise ValueError(f"API {api_id} not registered")
            
            config = self.api_configurations[api_id]
            
            if config.api_type != APIType.STREAMING:
                raise ValueError(f"API {api_id} does not support streaming")
            
            # Prepare headers
            if headers is None:
                headers = {}
            
            auth_headers = await self._get_auth_headers(config)
            headers.update(auth_headers)
            headers.update(config.headers)
            
            # Create streaming session
            async with aiohttp.ClientSession() as session:
                url = urljoin(config.base_url, endpoint)
                
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        raise Exception(f"Streaming request failed: {response.status}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line.decode('utf-8'))
                                await callback(data)
                            except json.JSONDecodeError:
                                # Handle non-JSON data
                                await callback(line.decode('utf-8'))
            
            return True
            
        except Exception as e:
            logger.error(f"Error streaming data: {str(e)}")
            return False
    
    async def health_check(self, api_id: str = None) -> Dict[str, Any]:
        """Perform health check on APIs"""
        try:
            if api_id:
                # Check specific API
                if api_id not in self.api_configurations:
                    return {"error": f"API {api_id} not found"}
                
                config = self.api_configurations[api_id]
                health_status = await self._check_api_health(config)
                
                return {
                    "api_id": api_id,
                    "name": config.name,
                    "status": health_status["status"],
                    "response_time": health_status["response_time"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                # Check all APIs
                health_results = {}
                
                for api_id, config in self.api_configurations.items():
                    try:
                        health_status = await self._check_api_health(config)
                        health_results[api_id] = {
                            "name": config.name,
                            "status": health_status["status"],
                            "response_time": health_status["response_time"]
                        }
                    except Exception as e:
                        health_results[api_id] = {
                            "name": config.name,
                            "status": "error",
                            "error": str(e)
                        }
                
                return health_results
            
        except Exception as e:
            logger.error(f"Error performing health check: {str(e)}")
            return {"error": str(e)}
    
    async def get_api_metrics(self, api_id: str = None) -> Dict[str, Any]:
        """Get API metrics"""
        try:
            if api_id:
                # Get specific API metrics
                if api_id not in self.api_metrics:
                    return {"error": f"API {api_id} not found"}
                
                metrics = self.api_metrics[api_id]
                return asdict(metrics)
            else:
                # Get all API metrics
                all_metrics = {}
                for api_id, metrics in self.api_metrics.items():
                    all_metrics[api_id] = asdict(metrics)
                
                return all_metrics
            
        except Exception as e:
            logger.error(f"Error getting API metrics: {str(e)}")
            return {"error": str(e)}
    
    async def get_api_analytics(self, 
                              time_range_hours: int = 24) -> Dict[str, Any]:
        """Get API analytics"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_range_hours)
            
            # Filter recent requests and responses
            recent_requests = [r for r in self.api_requests if r.timestamp >= cutoff_time]
            recent_responses = [r for r in self.api_responses if r.timestamp >= cutoff_time]
            
            analytics = {
                "total_apis": len(self.api_configurations),
                "active_apis": len([c for c in self.api_configurations.values() if c.status == APIStatus.ACTIVE]),
                "total_requests": len(recent_requests),
                "successful_requests": len([r for r in recent_responses if r.status_code < 400]),
                "failed_requests": len([r for r in recent_responses if r.status_code >= 400]),
                "average_response_time": 0.0,
                "success_rate": 0.0,
                "api_usage": {},
                "error_distribution": {},
                "response_time_distribution": {},
                "rate_limit_hits": 0,
                "authentication_failures": 0
            }
            
            if recent_responses:
                # Calculate success rate
                successful = len([r for r in recent_responses if r.status_code < 400])
                analytics["success_rate"] = successful / len(recent_responses)
                
                # Calculate average response time
                response_times = [r.response_time for r in recent_responses]
                analytics["average_response_time"] = sum(response_times) / len(response_times)
                
                # API usage distribution
                api_usage = defaultdict(int)
                for request in recent_requests:
                    api_usage[request.api_id] += 1
                analytics["api_usage"] = dict(api_usage)
                
                # Error distribution
                error_distribution = defaultdict(int)
                for response in recent_responses:
                    if response.status_code >= 400:
                        error_distribution[response.status_code] += 1
                analytics["error_distribution"] = dict(error_distribution)
                
                # Response time distribution
                response_time_ranges = {
                    "0-100ms": 0,
                    "100-500ms": 0,
                    "500ms-1s": 0,
                    "1-5s": 0,
                    "5s+": 0
                }
                
                for response in recent_responses:
                    rt = response.response_time * 1000  # Convert to milliseconds
                    if rt < 100:
                        response_time_ranges["0-100ms"] += 1
                    elif rt < 500:
                        response_time_ranges["100-500ms"] += 1
                    elif rt < 1000:
                        response_time_ranges["500ms-1s"] += 1
                    elif rt < 5000:
                        response_time_ranges["1-5s"] += 1
                    else:
                        response_time_ranges["5s+"] += 1
                
                analytics["response_time_distribution"] = response_time_ranges
            
            # Rate limit hits and authentication failures
            for metrics in self.api_metrics.values():
                analytics["rate_limit_hits"] += metrics.rate_limit_hits
                analytics["authentication_failures"] += metrics.authentication_failures
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting API analytics: {str(e)}")
            return {"error": str(e)}
    
    # Private helper methods
    async def _create_session(self, api_id: str) -> None:
        """Create HTTP session for API"""
        try:
            config = self.api_configurations[api_id]
            
            # Create session with custom configuration
            connector = aiohttp.TCPConnector(
                limit=100,
                limit_per_host=30,
                ttl_dns_cache=300,
                use_dns_cache=True
            )
            
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=config.headers
            )
            
            self.session_pool[api_id] = session
            
        except Exception as e:
            logger.error(f"Error creating session for API {api_id}: {str(e)}")
    
    async def _get_auth_headers(self, config: APIConfiguration) -> Dict[str, str]:
        """Get authentication headers"""
        try:
            auth_type = config.authentication
            auth_config = config.auth_config
            
            if auth_type == AuthenticationType.NONE:
                return {}
            elif auth_type == AuthenticationType.API_KEY:
                api_key = auth_config.get("api_key")
                header_name = auth_config.get("header_name", "X-API-Key")
                return {header_name: api_key}
            elif auth_type == AuthenticationType.BEARER_TOKEN:
                token = auth_config.get("token")
                return {"Authorization": f"Bearer {token}"}
            elif auth_type == AuthenticationType.BASIC_AUTH:
                username = auth_config.get("username")
                password = auth_config.get("password")
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                return {"Authorization": f"Basic {credentials}"}
            elif auth_type == AuthenticationType.JWT:
                token = auth_config.get("token")
                return {"Authorization": f"Bearer {token}"}
            elif auth_type == AuthenticationType.HMAC:
                # HMAC authentication (simplified)
                secret = auth_config.get("secret")
                timestamp = str(int(time.time()))
                message = f"{timestamp}"
                signature = hmac.new(secret.encode(), message.encode(), hashlib_module.sha256).hexdigest()
                return {
                    "X-Timestamp": timestamp,
                    "X-Signature": signature
                }
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Error getting auth headers: {str(e)}")
            return {}
    
    async def _check_rate_limits(self, api_id: str) -> bool:
        """Check rate limits for API"""
        try:
            config = self.api_configurations[api_id]
            current_time = datetime.now()
            
            # Check if API is currently rate limited
            if api_id in self.rate_limit_blocks:
                block_until = self.rate_limit_blocks[api_id]
                if current_time < block_until:
                    return False
                else:
                    del self.rate_limit_blocks[api_id]
            
            # Check rate limits
            for rate_limit_type, limit in config.rate_limits.items():
                if not await self._check_specific_rate_limit(api_id, rate_limit_type, limit):
                    # Block API for a short period
                    self.rate_limit_blocks[api_id] = current_time + timedelta(minutes=1)
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking rate limits: {str(e)}")
            return True  # Allow request if rate limit check fails
    
    async def _check_specific_rate_limit(self, api_id: str, rate_limit_type: RateLimitType, limit: int) -> bool:
        """Check specific rate limit"""
        try:
            current_time = datetime.now()
            tracker = self.rate_limit_tracker[api_id]
            
            # Clean old entries
            cutoff_time = current_time - timedelta(hours=1)
            while tracker and tracker[0] < cutoff_time:
                tracker.popleft()
            
            if rate_limit_type == RateLimitType.REQUESTS_PER_SECOND:
                # Check requests in last second
                one_second_ago = current_time - timedelta(seconds=1)
                recent_requests = [t for t in tracker if t > one_second_ago]
                return len(recent_requests) < limit
            elif rate_limit_type == RateLimitType.REQUESTS_PER_MINUTE:
                # Check requests in last minute
                one_minute_ago = current_time - timedelta(minutes=1)
                recent_requests = [t for t in tracker if t > one_minute_ago]
                return len(recent_requests) < limit
            elif rate_limit_type == RateLimitType.REQUESTS_PER_HOUR:
                # Check requests in last hour
                one_hour_ago = current_time - timedelta(hours=1)
                recent_requests = [t for t in tracker if t > one_hour_ago]
                return len(recent_requests) < limit
            else:
                return True
                
        except Exception as e:
            logger.error(f"Error checking specific rate limit: {str(e)}")
            return True
    
    async def _make_rest_request(self, config: APIConfiguration, request: APIRequest) -> Dict[str, Any]:
        """Make REST API request"""
        try:
            session = self.session_pool.get(config.api_id)
            if not session:
                await self._create_session(config.api_id)
                session = self.session_pool[config.api_id]
            
            url = urljoin(config.base_url, request.endpoint)
            
            # Make request
            async with session.request(
                method=request.method,
                url=url,
                headers=request.headers,
                params=request.params,
                json=request.data if isinstance(request.data, (dict, list)) else None,
                data=request.data if not isinstance(request.data, (dict, list)) else None,
                timeout=request.timeout
            ) as response:
                data = await response.json() if response.content_type == 'application/json' else await response.text()
                
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "data": data
                }
                
        except Exception as e:
            logger.error(f"Error making REST request: {str(e)}")
            raise e
    
    async def _make_graphql_request(self, config: APIConfiguration, request: APIRequest) -> Dict[str, Any]:
        """Make GraphQL API request"""
        try:
            session = self.session_pool.get(config.api_id)
            if not session:
                await self._create_session(config.api_id)
                session = self.session_pool[config.api_id]
            
            url = urljoin(config.base_url, request.endpoint)
            
            # Prepare GraphQL payload
            payload = {
                "query": request.data.get("query", ""),
                "variables": request.data.get("variables", {}),
                "operationName": request.data.get("operationName")
            }
            
            # Make request
            async with session.post(
                url=url,
                headers=request.headers,
                json=payload,
                timeout=request.timeout
            ) as response:
                data = await response.json()
                
                return {
                    "status_code": response.status,
                    "headers": dict(response.headers),
                    "data": data
                }
                
        except Exception as e:
            logger.error(f"Error making GraphQL request: {str(e)}")
            raise e
    
    async def _make_websocket_request(self, config: APIConfiguration, request: APIRequest) -> Dict[str, Any]:
        """Make WebSocket API request"""
        try:
            url = urljoin(config.base_url, request.endpoint)
            
            # Create WebSocket connection
            async with websockets.connect(
                url,
                extra_headers=request.headers,
                timeout=request.timeout
            ) as websocket:
                # Send message if provided
                if request.data:
                    await websocket.send(json.dumps(request.data))
                
                # Receive response
                response = await websocket.recv()
                data = json.loads(response) if isinstance(response, str) else response
                
                return {
                    "status_code": 200,
                    "headers": {},
                    "data": data
                }
                
        except Exception as e:
            logger.error(f"Error making WebSocket request: {str(e)}")
            raise e
    
    async def _check_api_health(self, config: APIConfiguration) -> Dict[str, Any]:
        """Check API health"""
        try:
            start_time = time.time()
            
            if config.api_type == APIType.REST:
                session = self.session_pool.get(config.api_id)
                if not session:
                    await self._create_session(config.api_id)
                    session = self.session_pool[config.api_id]
                
                url = urljoin(config.base_url, config.health_check_endpoint)
                auth_headers = await self._get_auth_headers(config)
                
                async with session.get(url, headers=auth_headers, timeout=5) as response:
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        return {"status": "healthy", "response_time": response_time}
                    else:
                        return {"status": "unhealthy", "response_time": response_time}
            else:
                # For non-REST APIs, assume healthy if configuration exists
                return {"status": "healthy", "response_time": 0.0}
                
        except Exception as e:
            response_time = time.time() - start_time
            return {"status": "error", "response_time": response_time, "error": str(e)}
    
    async def _update_metrics(self, api_id: str, response: APIResponse) -> None:
        """Update API metrics"""
        try:
            if api_id not in self.api_metrics:
                return
            
            metrics = self.api_metrics[api_id]
            
            # Update basic metrics
            metrics.total_requests += 1
            metrics.last_request_time = response.timestamp
            
            if response.status_code < 400:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
                
                if response.status_code == 401:
                    metrics.authentication_failures += 1
                elif response.status_code == 429:
                    metrics.rate_limit_hits += 1
            
            # Update average response time
            if metrics.total_requests == 1:
                metrics.average_response_time = response.response_time
            else:
                metrics.average_response_time = (
                    (metrics.average_response_time * (metrics.total_requests - 1) + response.response_time) 
                    / metrics.total_requests
                )
            
            # Update success rate
            metrics.success_rate = metrics.successful_requests / metrics.total_requests
            
            # Update uptime percentage (simplified)
            if response.status_code < 500:
                metrics.uptime_percentage = min(100.0, metrics.uptime_percentage + 0.1)
            else:
                metrics.uptime_percentage = max(0.0, metrics.uptime_percentage - 1.0)
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
    
    def _start_background_tasks(self) -> None:
        """Start background tasks"""
        try:
            # Start cleanup task
            cleanup_thread = threading.Thread(target=self._cleanup_old_data, daemon=True)
            cleanup_thread.start()
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {str(e)}")
    
    def _cleanup_old_data(self) -> None:
        """Cleanup old data periodically"""
        try:
            while self.running:
                time.sleep(3600)  # Run every hour
                
                # Cleanup old requests and responses
                cutoff_time = datetime.now() - timedelta(days=7)
                
                self.api_requests = [r for r in self.api_requests if r.timestamp >= cutoff_time]
                self.api_responses = [r for r in self.api_responses if r.timestamp >= cutoff_time]
                
                # Cleanup rate limit tracker
                for api_id in list(self.rate_limit_tracker.keys()):
                    tracker = self.rate_limit_tracker[api_id]
                    while tracker and tracker[0] < cutoff_time:
                        tracker.popleft()
                    
                    if not tracker:
                        del self.rate_limit_tracker[api_id]
                
        except Exception as e:
            logger.error(f"Error in cleanup task: {str(e)}")


# Global API integration system instance
_api_integration_system: Optional[APIIntegrationSystem] = None


def get_api_integration_system(max_apis: int = 100, max_requests_per_minute: int = 1000) -> APIIntegrationSystem:
    """Get or create global API integration system instance"""
    global _api_integration_system
    if _api_integration_system is None:
        _api_integration_system = APIIntegrationSystem(max_apis, max_requests_per_minute)
    return _api_integration_system


# Example usage
async def main():
    """Example usage of the API integration system"""
    system = get_api_integration_system()
    
    # Register REST API
    rest_api = await system.register_api(
        name="AI Model API",
        description="REST API for AI model predictions",
        base_url="https://api.example.com",
        api_type=APIType.REST,
        authentication=AuthenticationType.API_KEY,
        auth_config={"api_key": "your-api-key", "header_name": "X-API-Key"},
        rate_limits={RateLimitType.REQUESTS_PER_MINUTE: 100},
        headers={"Content-Type": "application/json"}
    )
    print(f"Registered REST API: {rest_api.api_id}")
    
    # Register GraphQL API
    graphql_api = await system.register_api(
        name="Analytics GraphQL API",
        description="GraphQL API for analytics data",
        base_url="https://graphql.example.com",
        api_type=APIType.GRAPHQL,
        authentication=AuthenticationType.BEARER_TOKEN,
        auth_config={"token": "your-bearer-token"},
        rate_limits={RateLimitType.REQUESTS_PER_MINUTE: 50}
    )
    print(f"Registered GraphQL API: {graphql_api.api_id}")
    
    # Make REST API request
    try:
        response = await system.make_request(
            api_id=rest_api.api_id,
            method="POST",
            endpoint="/predict",
            data={"model": "gpt-4", "prompt": "Hello world"}
        )
        print(f"REST API response: {response.status_code}")
    except Exception as e:
        print(f"REST API request failed: {str(e)}")
    
    # Make GraphQL API request
    try:
        response = await system.make_request(
            api_id=graphql_api.api_id,
            method="POST",
            endpoint="/graphql",
            data={
                "query": "query { analytics { totalRequests } }",
                "variables": {}
            }
        )
        print(f"GraphQL API response: {response.status_code}")
    except Exception as e:
        print(f"GraphQL API request failed: {str(e)}")
    
    # Make batch requests
    batch_requests = [
        {
            "api_id": rest_api.api_id,
            "method": "GET",
            "endpoint": "/status"
        },
        {
            "api_id": graphql_api.api_id,
            "method": "POST",
            "endpoint": "/graphql",
            "data": {"query": "query { health }"}
        }
    ]
    
    batch_responses = await system.batch_request(batch_requests)
    print(f"Batch request completed: {len(batch_responses)} responses")
    
    # Health check
    health_status = await system.health_check()
    print(f"Health check: {len(health_status)} APIs checked")
    
    # Get metrics
    metrics = await system.get_api_metrics()
    print(f"API metrics: {len(metrics)} APIs")
    
    # Get analytics
    analytics = await system.get_api_analytics()
    print(f"API analytics: {analytics.get('total_requests', 0)} total requests")


if __name__ == "__main__":
    asyncio.run(main())

























