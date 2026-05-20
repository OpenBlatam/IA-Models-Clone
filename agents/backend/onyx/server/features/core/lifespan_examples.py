from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any, AsyncGenerator
import time
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from fastapi import FastAPI, Depends
from pydantic import BaseModel
import structlog
from ..utils.lifespan_manager import (
    import uvicorn
from typing import Any, List, Dict, Optional
"""
Lifespan Examples - Migration from app.on_event() to Lifespan Context Managers
Demonstrates how to migrate from @app.on_event("startup") and @app.on_event("shutdown")
to modern lifespan context managers for better resource management.
"""



    LifespanManager,
    EventPriority,
    create_database_lifespan,
    create_cache_lifespan,
    create_background_task_lifespan
)

logger = structlog.get_logger(__name__)

# =============================================================================
# EXAMPLE 1: Basic Migration - Simple Startup/Shutdown
# =============================================================================

# OLD WAY (app.on_event)
"""
@app.on_event("startup")
async def startup_event():
    
    """startup_event function."""
logger.info("Application starting up")
    # Initialize resources
    
@app.on_event("shutdown")
async def shutdown_event():
    
    """shutdown_event function."""
logger.info("Application shutting down")
    # Cleanup resources
"""

# NEW WAY (Lifespan Context Manager)
@asynccontextmanager
async def basic_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup
    logger.info("Application starting up")
    try:
        # Initialize resources here
        yield
    finally:
        # Shutdown (runs even if startup fails)
        logger.info("Application shutting down")
        # Cleanup resources here

# Usage
app_basic = FastAPI(lifespan=basic_lifespan)

# =============================================================================
# EXAMPLE 2: Database Connection Management
# =============================================================================

# Database configuration
DATABASE_URL = "postgresql+asyncpg://user:pass@localhost/dbname"
REDIS_URL = "redis://localhost:6379"

# Database engine and session factory
engine = None
AsyncSessionLocal = None
redis_client = None

async def connect_database():
    """Initialize database connection."""
    global engine, AsyncSessionLocal
    logger.info("Connecting to database")
    engine = create_async_engine(DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )
    logger.info("Database connected successfully")

async def disconnect_database():
    """Close database connection."""
    global engine
    if engine:
        logger.info("Disconnecting from database")
        await engine.dispose()
        logger.info("Database disconnected")

async def connect_redis():
    """Initialize Redis connection."""
    global redis_client
    logger.info("Connecting to Redis")
    redis_client = redis.from_url(REDIS_URL)
    await redis_client.ping()
    logger.info("Redis connected successfully")

async def disconnect_redis():
    """Close Redis connection."""
    global redis_client
    if redis_client:
        logger.info("Disconnecting from Redis")
        await redis_client.close()
        logger.info("Redis disconnected")

# Using LifespanManager for database and cache
lifespan_manager = LifespanManager()
lifespan_manager.add_database_connection(
    connect_database, 
    disconnect_database,
    name="postgres",
    timeout=30.0
)
lifespan_manager.add_cache_connection(
    connect_redis,
    disconnect_redis,
    name="redis",
    timeout=10.0
)

# Alternative: Using convenience functions
db_lifespan = create_database_lifespan(
    connect_database, 
    disconnect_database,
    name="postgres",
    timeout=30.0
)
cache_lifespan = create_cache_lifespan(
    connect_redis,
    disconnect_redis,
    name="redis",
    timeout=10.0
)

# =============================================================================
# EXAMPLE 3: Background Tasks and Services
# =============================================================================

class BackgroundService:
    """Example background service."""
    
    def __init__(self) -> Any:
        self.running = False
        self.task = None
    
    async def start(self) -> Any:
        """Start the background service."""
        logger.info("Starting background service")
        self.running = True
        self.task = asyncio.create_task(self._run())
        logger.info("Background service started")
    
    async def stop(self) -> Any:
        """Stop the background service."""
        logger.info("Stopping background service")
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        logger.info("Background service stopped")
    
    async def _run(self) -> Any:
        """Background service main loop."""
        while self.running:
            try:
                # Do background work
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break

# Initialize background service
background_service = BackgroundService()

# Add to lifespan manager
lifespan_manager.add_background_task(
    background_service.start,
    background_service.stop,
    name="background_service",
    timeout=10.0
)

# =============================================================================
# EXAMPLE 4: Configuration and Validation
# =============================================================================

async def validate_configuration():
    """Validate application configuration."""
    logger.info("Validating configuration")
    
    # Check required environment variables
    required_vars = ["DATABASE_URL", "REDIS_URL", "API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not var in globals():
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required configuration: {missing_vars}")
    
    logger.info("Configuration validated successfully")

async def load_models():
    """Load AI models."""
    logger.info("Loading AI models")
    # Simulate model loading
    await asyncio.sleep(2)
    logger.info("AI models loaded successfully")

# Add configuration validation with high priority
lifespan_manager.add_startup_event(
    validate_configuration,
    name="config_validation",
    priority=EventPriority.CRITICAL,
    timeout=5.0
)

# Add model loading with normal priority
lifespan_manager.add_startup_event(
    load_models,
    name="model_loading",
    priority=EventPriority.NORMAL,
    timeout=30.0,
    retry_count=2,
    retry_delay=5.0
)

# =============================================================================
# EXAMPLE 5: Complete Application with Lifespan
# =============================================================================

class UserModel(BaseModel):
    id: int
    name: str
    email: str

class UserService:
    """User service with database dependency."""
    
    def __init__(self, session_factory) -> Any:
        self.session_factory = session_factory
    
    async def get_user(self, user_id: int) -> Optional[UserModel]:
        """Get user by ID."""
        # Simulate database query
        return UserModel(id=user_id, name="John Doe", email="john@example.com")

# Dependency injection
async def get_user_service() -> UserService:
    """Get user service dependency."""
    return UserService(AsyncSessionLocal)

# API endpoints
@app_basic.get("/users/{user_id}", response_model=UserModel)
async def get_user(
    user_id: int,
    user_service: UserService = Depends(get_user_service)
):
    """Get user endpoint."""
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# =============================================================================
# EXAMPLE 6: Advanced Lifespan with Error Handling
# =============================================================================

@asynccontextmanager
async def advanced_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Advanced lifespan with comprehensive error handling."""
    
    startup_errors = []
    shutdown_errors = []
    
    # Startup phase
    logger.info("Starting advanced application")
    
    startup_tasks = [
        ("Database", connect_database()),
        ("Redis", connect_redis()),
        ("Background Service", background_service.start()),
        ("Configuration", validate_configuration()),
        ("Models", load_models())
    ]
    
    # Execute startup tasks with error handling
    for name, task in startup_tasks:
        try:
            await task
            logger.info(f"{name} initialized successfully")
        except Exception as e:
            logger.error(f"{name} initialization failed", error=str(e))
            startup_errors.append((name, str(e)))
    
    # Check if critical services failed
    critical_services = ["Database", "Configuration"]
    critical_failures = [name for name, _ in startup_errors if name in critical_services]
    
    if critical_failures:
        logger.error("Critical services failed to start", services=critical_failures)
        # Still yield to allow graceful shutdown
        yield
    else:
        logger.info("Application started successfully")
        yield
    
    # Shutdown phase
    logger.info("Shutting down advanced application")
    
    shutdown_tasks = [
        ("Background Service", background_service.stop()),
        ("Redis", disconnect_redis()),
        ("Database", disconnect_database())
    ]
    
    # Execute shutdown tasks (reverse order)
    for name, task in shutdown_tasks:
        try:
            await task
            logger.info(f"{name} shut down successfully")
        except Exception as e:
            logger.error(f"{name} shutdown failed", error=str(e))
            shutdown_errors.append((name, str(e)))
    
    if shutdown_errors:
        logger.warning("Some services failed to shut down cleanly", 
                      errors=shutdown_errors)
    else:
        logger.info("Application shut down successfully")

# =============================================================================
# EXAMPLE 7: Migration Helper Functions
# =============================================================================

def migrate_existing_app(app: FastAPI) -> FastAPI:
    """
    Migrate an existing FastAPI app from @app.on_event to lifespan.
    
    This function demonstrates how to extract existing event handlers
    and convert them to lifespan context managers.
    """
    
    # Extract existing event handlers (you would need to modify this
    # based on your specific app structure)
    startup_handlers = []
    shutdown_handlers = []
    
    # Example: If you have existing handlers stored somewhere
    # startup_handlers = app.state.startup_handlers
    # shutdown_handlers = app.state.shutdown_handlers
    
    # Create lifespan manager
    manager = LifespanManager()
    
    # Add existing handlers
    for handler in startup_handlers:
        manager.add_startup_event(handler)
    
    for handler in shutdown_handlers:
        manager.add_shutdown_event(handler)
    
    # Create new app with lifespan
    new_app = FastAPI(
        title=app.title,
        description=app.description,
        version=app.version,
        lifespan=manager.create_lifespan()
    )
    
    # Copy routes and middleware
    new_app.routes = app.routes
    new_app.middleware = app.middleware
    
    return new_app

# =============================================================================
# EXAMPLE 8: Testing Lifespan Events
# =============================================================================

async def test_lifespan_events():
    """Test lifespan events in isolation."""
    
    # Create test lifespan manager
    test_manager = LifespanManager()
    
    # Add test events
    startup_called = False
    shutdown_called = False
    
    async def test_startup():
        
    """test_startup function."""
nonlocal startup_called
        startup_called = True
        logger.info("Test startup called")
    
    async def test_shutdown():
        
    """test_shutdown function."""
nonlocal shutdown_called
        shutdown_called = True
        logger.info("Test shutdown called")
    
    test_manager.add_startup_event(test_startup)
    test_manager.add_shutdown_event(test_shutdown)
    
    # Test lifespan
    lifespan = test_manager.create_lifespan()
    
    async with lifespan(None):
        logger.info("Application running")
        # Simulate application running
        await asyncio.sleep(1)
    
    assert startup_called, "Startup event should have been called"
    assert shutdown_called, "Shutdown event should have been called"
    logger.info("Lifespan test completed successfully")

# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def create_app_with_lifespan() -> FastAPI:
    """Create a FastAPI app with lifespan management."""
    
    # Method 1: Using LifespanManager
    manager = LifespanManager()
    manager.add_database_connection(connect_database, disconnect_database)
    manager.add_cache_connection(connect_redis, disconnect_redis)
    manager.add_background_task(
        background_service.start,
        background_service.stop
    )
    
    app = FastAPI(
        title="Lifespan Example API",
        version="1.0.0",
        lifespan=manager.create_lifespan()
    )
    
    return app

def create_app_with_direct_lifespan() -> FastAPI:
    """Create a FastAPI app with direct lifespan context manager."""
    
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup
        await connect_database()
        await connect_redis()
        await background_service.start()
        
        yield
        
        # Shutdown
        await background_service.stop()
        await disconnect_redis()
        await disconnect_database()
    
    app = FastAPI(
        title="Direct Lifespan API",
        version="1.0.0",
        lifespan=lifespan
    )
    
    return app

# Example usage
if __name__ == "__main__":
    # Create app with lifespan
    app = create_app_with_lifespan()
    
    # Add routes
    @app.get("/health")
    async def health_check():
        
    """health_check function."""
return {"status": "healthy", "timestamp": time.time()}
    
    # Run the app
    uvicorn.run(app, host="0.0.0.0", port=8000) 