from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Union, Callable
from contextlib import asynccontextmanager
from functools import lru_cache
import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import select, func
import httpx
import redis.asyncio as redis
        import jwt
        import jwt
    import uvicorn
from typing import Any, List, Dict, Optional
"""
FastAPI Dependency Injection Implementation
==========================================

This module demonstrates:
- FastAPI's dependency injection system for managing state
- Shared resources management (database, cache, external services)
- Configuration management through dependencies
- Authentication and authorization dependencies
- Background task dependencies
- Custom dependency providers
- Dependency scoping and lifecycle management
- Testing with dependency overrides
"""




# ============================================================================
# CONFIGURATION AND SETTINGS
# ============================================================================

class Settings:
    """Application settings with dependency injection support."""
    
    # Database settings
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost/db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 30
    
    # Redis settings
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_POOL_SIZE: int = 10
    
    # External API settings
    EXTERNAL_API_BASE_URL: str = "https://api.external.com"
    EXTERNAL_API_TIMEOUT: int = 30
    
    # Authentication settings
    JWT_SECRET_KEY: str = "your-secret-key"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Application settings
    APP_NAME: str = "FastAPI Dependency Injection Demo"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False


# ============================================================================
# DATABASE DEPENDENCIES
# ============================================================================

class DatabaseManager:
    """Database connection manager with dependency injection."""
    
    def __init__(self, database_url: str, pool_size: int, max_overflow: int):
        
    """__init__ function."""
self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.session_factory = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> Any:
        """Initialize database connection pool."""
        if self.engine is None:
            self.engine = create_async_engine(
                self.database_url,
                echo=False,
                future=True,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600
            )
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            self.logger.info("Database connection pool initialized")
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        if self.session_factory is None:
            await self.initialize()
        
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def close(self) -> Any:
        """Close database connections."""
        if self.engine:
            await self.engine.dispose()
            self.logger.info("Database connections closed")


# ============================================================================
# CACHE DEPENDENCIES
# ============================================================================

class CacheManager:
    """Redis cache manager with dependency injection."""
    
    def __init__(self, redis_url: str, pool_size: int):
        
    """__init__ function."""
self.redis_url = redis_url
        self.pool_size = pool_size
        self.redis_pool = None
        self.logger = logging.getLogger(__name__)
    
    async def initialize(self) -> Any:
        """Initialize Redis connection pool."""
        if self.redis_pool is None:
            self.redis_pool = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=self.pool_size
            )
            self.logger.info("Redis connection pool initialized")
    
    async def get_redis(self) -> redis.Redis:
        """Get Redis connection."""
        if self.redis_pool is None:
            await self.initialize()
        return self.redis_pool
    
    async def close(self) -> Any:
        """Close Redis connections."""
        if self.redis_pool:
            await self.redis_pool.close()
            self.logger.info("Redis connections closed")


# ============================================================================
# HTTP CLIENT DEPENDENCIES
# ============================================================================

class HTTPClientManager:
    """HTTP client manager with dependency injection."""
    
    def __init__(self, base_url: str, timeout: int):
        
    """__init__ function."""
self.base_url = base_url
        self.timeout = timeout
        self.client = None
        self.logger = logging.getLogger(__name__)
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get HTTP client with connection pooling."""
        if self.client is None:
            limits = httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                limits=limits,
                timeout=httpx.Timeout(self.timeout),
                follow_redirects=True
            )
            self.logger.info("HTTP client initialized")
        return self.client
    
    async def close(self) -> Any:
        """Close HTTP client."""
        if self.client:
            await self.client.aclose()
            self.logger.info("HTTP client closed")


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

class AuthManager:
    """Authentication manager with dependency injection."""
    
    def __init__(self, secret_key: str, algorithm: str, access_token_expire_minutes: int):
        
    """__init__ function."""
self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.logger = logging.getLogger(__name__)
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


# ============================================================================
# DEPENDENCY PROVIDERS
# ============================================================================

@lru_cache()
def get_settings() -> Settings:
    """Get application settings (singleton)."""
    return Settings()


async def get_database_manager(settings: Settings = Depends(get_settings)) -> DatabaseManager:
    """Get database manager instance."""
    manager = DatabaseManager(
        database_url=settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW
    )
    await manager.initialize()
    return manager


async def get_cache_manager(settings: Settings = Depends(get_settings)) -> CacheManager:
    """Get cache manager instance."""
    manager = CacheManager(
        redis_url=settings.REDIS_URL,
        pool_size=settings.REDIS_POOL_SIZE
    )
    await manager.initialize()
    return manager


async def get_http_client_manager(settings: Settings = Depends(get_settings)) -> HTTPClientManager:
    """Get HTTP client manager instance."""
    manager = HTTPClientManager(
        base_url=settings.EXTERNAL_API_BASE_URL,
        timeout=settings.EXTERNAL_API_TIMEOUT
    )
    return manager


async def get_auth_manager(settings: Settings = Depends(get_settings)) -> AuthManager:
    """Get authentication manager instance."""
    return AuthManager(
        secret_key=settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
        access_token_expire_minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )


# ============================================================================
# DATABASE SESSION DEPENDENCY
# ============================================================================

async def get_db_session(
    db_manager: DatabaseManager = Depends(get_database_manager)
) -> AsyncSession:
    """Get database session dependency."""
    async for session in db_manager.get_session():
        yield session


# ============================================================================
# CACHE DEPENDENCY
# ============================================================================

async def get_cache(
    cache_manager: CacheManager = Depends(get_cache_manager)
) -> redis.Redis:
    """Get cache dependency."""
    return await cache_manager.get_redis()


# ============================================================================
# HTTP CLIENT DEPENDENCY
# ============================================================================

async def get_http_client(
    client_manager: HTTPClientManager = Depends(get_http_client_manager)
) -> httpx.AsyncClient:
    """Get HTTP client dependency."""
    return await client_manager.get_client()


# ============================================================================
# AUTHENTICATION DEPENDENCIES
# ============================================================================

security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> Dict[str, Any]:
    """Get current authenticated user."""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    return {"user_id": user_id, "payload": payload}


async def get_current_active_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current active user."""
    # In a real application, you would check if the user is active
    # For demo purposes, we'll just return the user
    return current_user


# ============================================================================
# CUSTOM DEPENDENCIES
# ============================================================================

async async def get_request_id() -> str:
    """Generate unique request ID for tracing."""
    return str(uuid.uuid4())


async async def get_request_timestamp() -> datetime:
    """Get request timestamp."""
    return datetime.utcnow()


async def get_user_agent(request: Request) -> str:
    """Get user agent from request."""
    return request.headers.get("user-agent", "Unknown")


async def get_client_ip(request: Request) -> str:
    """Get client IP address."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0]
    return request.client.host if request.client else "Unknown"


# ============================================================================
# BACKGROUND TASK DEPENDENCIES
# ============================================================================

class BackgroundTaskManager:
    """Background task manager with dependency injection."""
    
    def __init__(self) -> Any:
        self.logger = logging.getLogger(__name__)
        self.tasks = []
    
    async def add_task(self, task_func: Callable, *args, **kwargs):
        """Add background task."""
        task = asyncio.create_task(task_func(*args, **kwargs))
        self.tasks.append(task)
        return task
    
    async def cleanup_completed_tasks(self) -> Any:
        """Clean up completed tasks."""
        self.tasks = [task for task in self.tasks if not task.done()]


async def get_background_task_manager() -> BackgroundTaskManager:
    """Get background task manager dependency."""
    return BackgroundTaskManager()


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class UserCreate(BaseModel):
    """User creation model."""
    
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., description="User email")
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = Field(None, max_length=100)


class UserResponse(BaseModel):
    """User response model."""
    
    id: int
    username: str
    email: str
    full_name: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    """Login request model."""
    
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")


class LoginResponse(BaseModel):
    """Login response model."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class ExternalAPICall(BaseModel):
    """External API call model."""
    
    endpoint: str
    method: str = "GET"
    data: Optional[Dict[str, Any]] = None


# ============================================================================
# SERVICE LAYER WITH DEPENDENCY INJECTION
# ============================================================================

class UserService:
    """User service with dependency injection."""
    
    def __init__(
        self,
        db_session: AsyncSession,
        cache: redis.Redis,
        auth_manager: AuthManager,
        logger: logging.Logger
    ):
        
    """__init__ function."""
self.db_session = db_session
        self.cache = cache
        self.auth_manager = auth_manager
        self.logger = logger
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        try:
            # Check if user already exists
            cache_key = f"user:email:{user_data.email}"
            existing_user = await self.cache.get(cache_key)
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="User with this email already exists"
                )
            
            # Simulate database insert
            user_dict = user_data.model_dump()
            user_dict.update({
                "id": 999,  # Mock ID
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })
            
            # Cache user data
            await self.cache.setex(
                cache_key,
                300,  # 5 minutes TTL
                str(user_dict)
            )
            
            self.logger.info(f"User created: {user_data.email}")
            return UserResponse(**user_dict)
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Failed to create user: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create user"
            )
    
    async def get_user_by_id(self, user_id: int) -> Optional[UserResponse]:
        """Get user by ID."""
        try:
            # Try cache first
            cache_key = f"user:id:{user_id}"
            cached_user = await self.cache.get(cache_key)
            if cached_user:
                return UserResponse(**eval(cached_user))
            
            # Simulate database query
            user_dict = {
                "id": user_id,
                "username": f"user_{user_id}",
                "email": f"user_{user_id}@example.com",
                "full_name": f"User {user_id}",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Cache user data
            await self.cache.setex(cache_key, 300, str(user_dict))
            
            return UserResponse(**user_dict)
            
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to get user"
            )
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserResponse]:
        """Authenticate user."""
        try:
            # Simulate user authentication
            if username == "admin" and password == "password123":
                user_dict = {
                    "id": 1,
                    "username": "admin",
                    "email": "admin@example.com",
                    "full_name": "Administrator",
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
                return UserResponse(**user_dict)
            return None
            
        except Exception as e:
            self.logger.error(f"Authentication failed: {str(e)}")
            return None


# ============================================================================
# DEPENDENCY FACTORY FUNCTIONS
# ============================================================================

def create_user_service(
    db_session: AsyncSession = Depends(get_db_session),
    cache: redis.Redis = Depends(get_cache),
    auth_manager: AuthManager = Depends(get_auth_manager)
) -> UserService:
    """Create user service with dependencies."""
    logger = logging.getLogger("user_service")
    return UserService(db_session, cache, auth_manager, logger)


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger = logging.getLogger("app")
    logger.info("Application starting up...")
    
    # Initialize shared resources
    settings = get_settings()
    
    # Initialize database manager
    db_manager = DatabaseManager(
        database_url=settings.DATABASE_URL,
        pool_size=settings.DB_POOL_SIZE,
        max_overflow=settings.DB_MAX_OVERFLOW
    )
    await db_manager.initialize()
    
    # Initialize cache manager
    cache_manager = CacheManager(
        redis_url=settings.REDIS_URL,
        pool_size=settings.REDIS_POOL_SIZE
    )
    await cache_manager.initialize()
    
    # Store managers in app state
    app.state.db_manager = db_manager
    app.state.cache_manager = cache_manager
    
    yield
    
    # Shutdown
    logger.info("Application shutting down...")
    
    # Close connections
    await db_manager.close()
    await cache_manager.close()


def create_app() -> FastAPI:
    """Create FastAPI application with dependency injection."""
    
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        debug=settings.DEBUG,
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    return app


app = create_app()


# ============================================================================
# API ROUTES WITH DEPENDENCY INJECTION
# ============================================================================

@app.post("/auth/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    user_service: UserService = Depends(create_user_service),
    auth_manager: AuthManager = Depends(get_auth_manager),
    request_id: str = Depends(get_request_id),
    timestamp: datetime = Depends(get_request_timestamp)
):
    """User login with dependency injection."""
    
    # Authenticate user
    user = await user_service.authenticate_user(
        login_data.username,
        login_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )
    
    # Create access token
    access_token = auth_manager.create_access_token(
        data={"sub": str(user.id)}
    )
    
    return LoginResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=user
    )


@app.post("/users/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserService = Depends(create_user_service),
    task_manager: BackgroundTaskManager = Depends(get_background_task_manager),
    request_id: str = Depends(get_request_id),
    user_agent: str = Depends(get_user_agent),
    client_ip: str = Depends(get_client_ip)
):
    """Create user with comprehensive dependency injection."""
    
    # Add background task
    async def send_welcome_email(user_email: str):
        
    """send_welcome_email function."""
await asyncio.sleep(1)  # Simulate email sending
        logging.getLogger("background_task").info(f"Welcome email sent to {user_email}")
    
    # Use task manager instead of BackgroundTasks
    await task_manager.add_task(send_welcome_email, user_data.email)
    
    # Create user
    user = await user_service.create_user(user_data)
    
    # Log request details
    logging.getLogger("request_log").info(
        f"User created - Request ID: {request_id}, "
        f"User Agent: {user_agent}, Client IP: {client_ip}"
    )
    
    return user


@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(create_user_service),
    current_user: Dict[str, Any] = Depends(get_current_active_user),
    request_id: str = Depends(get_request_id)
):
    """Get user by ID with authentication dependency."""
    
    user = await user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found"
        )
    
    return user


@app.get("/external-api/{endpoint}")
async def call_external_api(
    endpoint: str,
    http_client: httpx.AsyncClient = Depends(get_http_client),
    cache: redis.Redis = Depends(get_cache),
    current_user: Dict[str, Any] = Depends(get_current_active_user)
):
    """Call external API with dependency injection."""
    
    # Check cache first
    cache_key = f"external_api:{endpoint}"
    cached_response = await cache.get(cache_key)
    if cached_response:
        return {"data": cached_response, "source": "cache"}
    
    try:
        # Make external API call
        response = await http_client.get(f"/{endpoint}")
        response.raise_for_status()
        
        # Cache response
        await cache.setex(cache_key, 300, response.text)
        
        return {"data": response.json(), "source": "api"}
        
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"External API error: {e.response.status_code}"
        )


@app.get("/health")
async def health_check(
    settings: Settings = Depends(get_settings),
    db_manager: DatabaseManager = Depends(get_database_manager),
    cache_manager: CacheManager = Depends(get_cache_manager)
):
    """Health check with dependency injection."""
    
    # Check database connection
    db_status = "healthy"
    try:
        # Simulate database check
        await asyncio.sleep(0.1)
    except Exception:
        db_status = "unhealthy"
    
    # Check cache connection
    cache_status = "healthy"
    try:
        cache = await cache_manager.get_redis()
        await cache.ping()
    except Exception:
        cache_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": settings.APP_VERSION,
        "database": db_status,
        "cache": cache_status
    }


@app.get("/request-info")
async def get_request_info(
    request_id: str = Depends(get_request_id),
    timestamp: datetime = Depends(get_request_timestamp),
    user_agent: str = Depends(get_user_agent),
    client_ip: str = Depends(get_client_ip),
    current_user: Optional[Dict[str, Any]] = Depends(get_current_user)
):
    """Get request information with multiple dependencies."""
    
    return {
        "request_id": request_id,
        "timestamp": timestamp,
        "user_agent": user_agent,
        "client_ip": client_ip,
        "authenticated_user": current_user.get("user_id") if current_user else None
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def demonstrate_dependency_injection():
    """Demonstrate dependency injection patterns."""
    
    print("\n=== FastAPI Dependency Injection Demonstrations ===")
    
    # 1. Settings dependency
    settings = get_settings()
    print(f"\n1. Settings Dependency:")
    print(f"   - App Name: {settings.APP_NAME}")
    print(f"   - Database URL: {settings.DATABASE_URL}")
    print(f"   - Redis URL: {settings.REDIS_URL}")
    
    # 2. Database manager dependency
    print(f"\n2. Database Manager Dependency:")
    db_manager = await get_database_manager(settings)
    print(f"   - Database manager initialized: {db_manager is not None}")
    
    # 3. Cache manager dependency
    print(f"\n3. Cache Manager Dependency:")
    cache_manager = await get_cache_manager(settings)
    print(f"   - Cache manager initialized: {cache_manager is not None}")
    
    # 4. HTTP client manager dependency
    print(f"\n4. HTTP Client Manager Dependency:")
    http_manager = await get_http_client_manager(settings)
    print(f"   - HTTP client manager initialized: {http_manager is not None}")
    
    # 5. Authentication manager dependency
    print(f"\n5. Authentication Manager Dependency:")
    auth_manager = await get_auth_manager(settings)
    print(f"   - Auth manager initialized: {auth_manager is not None}")
    
    # 6. Request ID generation
    print(f"\n6. Request ID Generation:")
    request_id = await get_request_id()
    print(f"   - Generated request ID: {request_id}")
    
    # 7. Background task manager
    print(f"\n7. Background Task Manager:")
    task_manager = await get_background_task_manager()
    print(f"   - Task manager initialized: {task_manager is not None}")


if __name__ == "__main__":
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the application
    uvicorn.run(
        "dependency_injection_implementation:app",
        host="0.0.0.0",
        port=8000,
        log_level="info"
    ) 