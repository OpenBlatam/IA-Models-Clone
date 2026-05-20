from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from functools import wraps
from contextlib import asynccontextmanager
import asyncio
import time
import logging
from datetime import datetime
from enum import Enum
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
import redis.asyncio as redis
from databases import Database
from sqlalchemy import text
import motor.motor_asyncio
import aiohttp
from typing import Any, List, Dict, Optional
"""
Dependency Management and Injection Patterns
- Centralized dependency injection container
- Clear dependency resolution patterns
- Modular service management
- Improved testability and maintainability
"""



logger = logging.getLogger(__name__)

# ============================================================================
# DEPENDENCY TYPES AND INTERFACES
# ============================================================================

T = TypeVar('T')

class ServiceType(str, Enum):
    """Enumeration of available service types."""
    DATABASE = "database"
    CACHE = "cache"
    EXTERNAL_API = "external_api"
    DIFFUSION = "diffusion"
    AUTH = "auth"
    MONITORING = "monitoring"

class ServiceStatus(str, Enum):
    """Enumeration of service statuses."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"

class ServiceInterface:
    """Base interface for all services."""
    
    async def health_check(self) -> ServiceStatus:
        """Check service health."""
        raise NotImplementedError
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        raise NotImplementedError

# ============================================================================
# DEPENDENCY CONTAINER
# ============================================================================

class DependencyContainer:
    """Centralized dependency injection container with lifecycle management."""
    
    def __init__(self, config: Dict[str, Any]):
        
    """__init__ function."""
self.config = config
        self._services: Dict[ServiceType, Any] = {}
        self._service_factories: Dict[ServiceType, Callable] = {}
        self._startup_time = time.time()
        self._is_shutdown = False
        
        # Register service factories
        self._register_service_factories()
    
    def _register_service_factories(self) -> Any:
        """Register service factory functions."""
        self._service_factories = {
            ServiceType.DATABASE: self._create_database_service,
            ServiceType.CACHE: self._create_cache_service,
            ServiceType.EXTERNAL_API: self._create_external_api_service,
            ServiceType.DIFFUSION: self._create_diffusion_service,
            ServiceType.AUTH: self._create_auth_service,
            ServiceType.MONITORING: self._create_monitoring_service
        }
    
    async def get_service(self, service_type: ServiceType) -> Optional[Dict[str, Any]]:
        """Get or create service instance."""
        if self._is_shutdown:
            raise RuntimeError("Container is shutting down")
        
        if service_type not in self._services:
            if service_type not in self._service_factories:
                raise ValueError(f"Unknown service type: {service_type}")
            
            factory = self._service_factories[service_type]
            self._services[service_type] = await factory()
            
            logger.info(f"Created service: {service_type.value}")
        
        return self._services[service_type]
    
    async def _create_database_service(self) -> 'DatabaseService':
        """Create database service instance."""
        return DatabaseService(self.config.get('database_url'))
    
    async def _create_cache_service(self) -> 'CacheService':
        """Create cache service instance."""
        return CacheService(self.config.get('redis_url'))
    
    async async def _create_external_api_service(self) -> 'ExternalAPIService':
        """Create external API service instance."""
        return ExternalAPIService(
            timeout=self.config.get('api_timeout', 30),
            max_connections=self.config.get('max_connections', 100)
        )
    
    async def _create_diffusion_service(self) -> 'DiffusionService':
        """Create diffusion service instance."""
        db_service = await self.get_service(ServiceType.DATABASE)
        cache_service = await self.get_service(ServiceType.CACHE)
        return DiffusionService(db_service, cache_service)
    
    async def _create_auth_service(self) -> 'AuthService':
        """Create authentication service instance."""
        return AuthService(self.config.get('jwt_secret'))
    
    async def _create_monitoring_service(self) -> 'MonitoringService':
        """Create monitoring service instance."""
        return MonitoringService()
    
    async def health_check(self) -> Dict[str, ServiceStatus]:
        """Check health of all services."""
        health_status = {}
        
        for service_type in ServiceType:
            try:
                service = await self.get_service(service_type)
                health_status[service_type.value] = await service.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {service_type.value}: {e}")
                health_status[service_type.value] = ServiceStatus.UNHEALTHY
        
        return health_status
    
    async def cleanup(self) -> None:
        """Cleanup all services."""
        self._is_shutdown = True
        
        for service_type, service in self._services.items():
            try:
                await service.cleanup()
                logger.info(f"Cleaned up service: {service_type.value}")
            except Exception as e:
                logger.error(f"Cleanup failed for {service_type.value}: {e}")
        
        self._services.clear()
    
    def get_uptime(self) -> float:
        """Get container uptime in seconds."""
        return time.time() - self._startup_time

# ============================================================================
# SERVICE IMPLEMENTATIONS
# ============================================================================

class DatabaseService(ServiceInterface):
    """Database service with connection pooling and health monitoring."""
    
    def __init__(self, database_url: str):
        
    """__init__ function."""
self.database_url = database_url
        self._database: Optional[Database] = None
        self._mongo_client: Optional[motor.motor_asyncio.AsyncIOMotorClient] = None
        self._last_health_check = 0
        self._health_cache_duration = 30  # seconds
    
    async def get_database(self) -> Database:
        """Get async database connection."""
        if self._database is None:
            self._database = Database(self.database_url)
            await self._database.connect()
        return self._database
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute async database query."""
        db = await self.get_database()
        try:
            result = await db.fetch_all(text(query), params or {})
            return [dict(row) for row in result]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            raise
    
    async def health_check(self) -> ServiceStatus:
        """Check database health."""
        current_time = time.time()
        
        # Cache health check results
        if current_time - self._last_health_check < self._health_cache_duration:
            return ServiceStatus.HEALTHY
        
        try:
            await self.execute_query("SELECT 1")
            self._last_health_check = current_time
            return ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    async def cleanup(self) -> None:
        """Cleanup database connections."""
        if self._database:
            await self._database.disconnect()
        if self._mongo_client:
            self._mongo_client.close()

class CacheService(ServiceInterface):
    """Cache service with Redis connection management."""
    
    def __init__(self, redis_url: str):
        
    """__init__ function."""
self.redis_url = redis_url
        self._redis: Optional[redis.Redis] = None
        self._last_health_check = 0
        self._health_cache_duration = 30  # seconds
    
    async def get_redis(self) -> redis.Redis:
        """Get async Redis connection."""
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url)
        return self._redis
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        redis_client = await self.get_redis()
        try:
            return await redis_client.get(key)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, value: str, expire: int = 3600) -> bool:
        """Set value in cache."""
        redis_client = await self.get_redis()
        try:
            return await redis_client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    async def health_check(self) -> ServiceStatus:
        """Check cache health."""
        current_time = time.time()
        
        # Cache health check results
        if current_time - self._last_health_check < self._health_cache_duration:
            return ServiceStatus.HEALTHY
        
        try:
            redis_client = await self.get_redis()
            await redis_client.ping()
            self._last_health_check = current_time
            return ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return ServiceStatus.UNHEALTHY
    
    async def cleanup(self) -> None:
        """Cleanup cache connection."""
        if self._redis:
            await self._redis.close()

class ExternalAPIService(ServiceInterface):
    """External API service with connection pooling."""
    
    def __init__(self, timeout: int = 30, max_connections: int = 100):
        
    """__init__ function."""
self.timeout = timeout
        self.max_connections = max_connections
        self._session: Optional[aiohttp.ClientSession] = None
        self._last_health_check = 0
        self._health_cache_duration = 30  # seconds
    
    async def get_session(self) -> aiohttp.ClientSession:
        """Get async HTTP session with connection pooling."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=30
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
        return self._session
    
    async async def make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make async HTTP request."""
        session = await self.get_session()
        start_time = time.time()
        
        try:
            async with session.request(method, url, **kwargs) as response:
                duration = time.time() - start_time
                
                if response.content_type == 'application/json':
                    data = await response.json()
                else:
                    data = await response.text()
                
                return {
                    'status_code': response.status,
                    'data': data,
                    'headers': dict(response.headers),
                    'duration': duration,
                    'url': url,
                    'method': method
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Request timeout: {url}")
            raise HTTPException(status_code=504, detail="External API timeout")
        except Exception as e:
            logger.error(f"External API error: {e}")
            raise HTTPException(status_code=502, detail="External API error")
    
    async def health_check(self) -> ServiceStatus:
        """Check external API health."""
        current_time = time.time()
        
        # Cache health check results
        if current_time - self._last_health_check < self._health_cache_duration:
            return ServiceStatus.HEALTHY
        
        try:
            # Test with a simple request
            await self.make_request("GET", "https://httpbin.org/status/200")
            self._last_health_check = current_time
            return ServiceStatus.HEALTHY
        except Exception as e:
            logger.error(f"External API health check failed: {e}")
            return ServiceStatus.DEGRADED
    
    async def cleanup(self) -> None:
        """Cleanup HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

class DiffusionService(ServiceInterface):
    """Diffusion service with business logic."""
    
    def __init__(self, db_service: DatabaseService, cache_service: CacheService):
        
    """__init__ function."""
self.db_service = db_service
        self.cache_service = cache_service
    
    async def save_generation_result(self, user_id: str, prompt: str, result_url: str) -> str:
        """Save generation result to database."""
        query = """
            INSERT INTO image_generations (user_id, prompt, result_url, created_at)
            VALUES (:user_id, :prompt, :result_url, NOW())
            RETURNING id
        """
        result = await self.db_service.execute_query(
            query, 
            {"user_id": user_id, "prompt": prompt, "result_url": result_url}
        )
        return result[0]['id'] if result else None
    
    async def get_cached_result(self, prompt_hash: str) -> Optional[str]:
        """Get cached generation result."""
        return await self.cache_service.get(f"generation:{prompt_hash}")
    
    async def cache_generation_result(self, prompt_hash: str, result_url: str) -> bool:
        """Cache generation result."""
        return await self.cache_service.set(f"generation:{prompt_hash}", result_url)
    
    async def get_user_generations(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get user's generation history."""
        query = """
            SELECT * FROM image_generations 
            WHERE user_id = :user_id 
            ORDER BY created_at DESC 
            LIMIT :limit
        """
        return await self.db_service.execute_query(
            query, 
            {"user_id": user_id, "limit": limit}
        )
    
    async def health_check(self) -> ServiceStatus:
        """Check diffusion service health."""
        db_health = await self.db_service.health_check()
        cache_health = await self.cache_service.health_check()
        
        if db_health == ServiceStatus.HEALTHY and cache_health == ServiceStatus.HEALTHY:
            return ServiceStatus.HEALTHY
        elif db_health == ServiceStatus.UNHEALTHY or cache_health == ServiceStatus.UNHEALTHY:
            return ServiceStatus.UNHEALTHY
        else:
            return ServiceStatus.DEGRADED
    
    async def cleanup(self) -> None:
        """Cleanup diffusion service."""
        # Services are cleaned up by the container
        pass

class AuthService(ServiceInterface):
    """Authentication service with JWT handling."""
    
    def __init__(self, jwt_secret: str):
        
    """__init__ function."""
self.jwt_secret = jwt_secret
    
    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token."""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def create_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Create JWT token."""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow().timestamp() + expires_in
        }
        return jwt.encode(payload, self.jwt_secret, algorithm="HS256")
    
    async def health_check(self) -> ServiceStatus:
        """Check auth service health."""
        # Auth service is stateless, so it's always healthy
        return ServiceStatus.HEALTHY
    
    async def cleanup(self) -> None:
        """Cleanup auth service."""
        # No cleanup needed for stateless service
        pass

class MonitoringService(ServiceInterface):
    """Monitoring service for metrics and logging."""
    
    def __init__(self) -> Any:
        self.metrics = {}
        self.start_time = time.time()
    
    def record_metric(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record a metric."""
        if name not in self.metrics:
            self.metrics[name] = []
        
        self.metrics[name].append({
            "value": value,
            "labels": labels or {},
            "timestamp": time.time()
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics."""
        return {
            "metrics": self.metrics,
            "uptime": time.time() - self.start_time
        }
    
    async def health_check(self) -> ServiceStatus:
        """Check monitoring service health."""
        return ServiceStatus.HEALTHY
    
    async def cleanup(self) -> None:
        """Cleanup monitoring service."""
        self.metrics.clear()

# ============================================================================
# DEPENDENCY INJECTION FUNCTIONS
# ============================================================================

async def get_dependency_container(request: Request) -> DependencyContainer:
    """Get dependency container from request state."""
    return request.app.state.dependency_container

async def get_database_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> DatabaseService:
    """Get database service from container."""
    return await container.get_service(ServiceType.DATABASE)

async def get_cache_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> CacheService:
    """Get cache service from container."""
    return await container.get_service(ServiceType.CACHE)

async def get_external_api_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> ExternalAPIService:
    """Get external API service from container."""
    return await container.get_service(ServiceType.EXTERNAL_API)

async def get_diffusion_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> DiffusionService:
    """Get diffusion service from container."""
    return await container.get_service(ServiceType.DIFFUSION)

async def get_auth_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> AuthService:
    """Get auth service from container."""
    return await container.get_service(ServiceType.AUTH)

async def get_monitoring_service(
    container: DependencyContainer = Depends(get_dependency_container)
) -> MonitoringService:
    """Get monitoring service from container."""
    return await container.get_service(ServiceType.MONITORING)

# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    auth_service: AuthService = Depends(get_auth_service)
) -> str:
    """Get current user from JWT token."""
    payload = await auth_service.validate_token(credentials.credentials)
    return payload.get("user_id", "anonymous")

async def get_current_user_optional(
    request: Request
) -> Optional[str]:
    """Get current user optionally (for public endpoints)."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    try:
        token = auth_header.split(" ")[1]
        auth_service = await get_auth_service(request.app.state.dependency_container)
        payload = await auth_service.validate_token(token)
        return payload.get("user_id")
    except:
        return None

# ============================================================================
# RATE LIMITING DEPENDENCIES
# ============================================================================

async def get_rate_limit_info(
    request: Request,
    cache_service: CacheService = Depends(get_cache_service)
) -> Dict[str, Any]:
    """Get rate limiting information."""
    client_ip = request.client.host
    user_id = await get_current_user_optional(request) or "anonymous"
    
    # Create rate limit key
    rate_limit_key = f"rate_limit:{user_id}:{client_ip}"
    
    # Get current count
    current_count = await cache_service.get(rate_limit_key)
    current_count = int(current_count) if current_count else 0
    
    # Check if limit exceeded
    limit_per_minute = 60
    if current_count >= limit_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    # Increment count
    await cache_service.set(rate_limit_key, str(current_count + 1), expire=60)
    
    return {
        "requests_per_minute": limit_per_minute,
        "remaining_requests": limit_per_minute - current_count - 1,
        "reset_time": datetime.utcnow().timestamp() + 60
    }

# ============================================================================
# DEPENDENCY DECORATORS
# ============================================================================

def inject_dependencies(*dependencies) -> Any:
    """Decorator to inject dependencies into route handlers."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Inject dependencies into kwargs
            for dep in dependencies:
                if dep not in kwargs:
                    kwargs[dep] = await dep()
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def monitor_performance(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func) -> Any:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric
                monitoring_service = kwargs.get('monitoring_service')
                if monitoring_service:
                    monitoring_service.record_metric(
                        f"{operation_name}_duration",
                        duration,
                        {"status": "success"}
                    )
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                
                # Record error metric
                monitoring_service = kwargs.get('monitoring_service')
                if monitoring_service:
                    monitoring_service.record_metric(
                        f"{operation_name}_duration",
                        duration,
                        {"status": "error", "error_type": type(e).__name__}
                    )
                
                raise
        return wrapper
    return decorator 