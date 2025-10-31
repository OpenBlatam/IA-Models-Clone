"""
Dependency Injection for Email Sequence System

This module provides FastAPI dependencies for managing shared resources,
database connections, and service instances following the dependency injection pattern.
"""

import asyncio
import logging
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
import redis.asyncio as redis

from ..services.langchain_service import LangChainEmailService
from ..services.delivery_service import EmailDeliveryService
from ..services.analytics_service import EmailAnalyticsService
from ..core.email_sequence_engine import EmailSequenceEngine
from ..core.config import get_settings

logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()
engine = None
async_session_maker = None
redis_client = None

# Global service instances
_email_sequence_engine: Optional[EmailSequenceEngine] = None


async def init_database() -> None:
    """Initialize database connection"""
    global engine, async_session_maker
    
    settings = get_settings()
    
    if not engine:
        engine = create_async_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        async_session_maker = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database connection initialized")


async def init_redis() -> None:
    """Initialize Redis connection"""
    global redis_client
    
    settings = get_settings()
    
    if not redis_client:
        redis_client = redis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            max_connections=settings.redis_max_connections
        )
        
        logger.info("Redis connection initialized")


async def init_services() -> None:
    """Initialize all services"""
    global _email_sequence_engine
    
    settings = get_settings()
    
    if not _email_sequence_engine:
        # Initialize services
        langchain_service = LangChainEmailService(
            api_key=settings.openai_api_key,
            model_name=settings.openai_model
        )
        
        delivery_service = EmailDeliveryService(
            smtp_host=settings.smtp_host,
            smtp_port=settings.smtp_port,
            smtp_username=settings.smtp_username,
            smtp_password=settings.smtp_password,
            use_tls=settings.smtp_use_tls,
            from_email=settings.from_email,
            from_name=settings.from_name
        )
        
        analytics_service = EmailAnalyticsService(
            database_url=settings.database_url,
            redis_url=settings.redis_url
        )
        
        # Initialize engine
        _email_sequence_engine = EmailSequenceEngine(
            langchain_service=langchain_service,
            delivery_service=delivery_service,
            analytics_service=analytics_service,
            max_concurrent_sequences=settings.max_concurrent_sequences
        )
        
        # Start the engine
        await _email_sequence_engine.start()
        
        logger.info("Email sequence engine initialized and started")


async def cleanup_services() -> None:
    """Cleanup all services"""
    global _email_sequence_engine, engine, redis_client
    
    if _email_sequence_engine:
        await _email_sequence_engine.stop()
        _email_sequence_engine = None
    
    if engine:
        await engine.dispose()
        engine = None
    
    if redis_client:
        await redis_client.close()
        redis_client = None
    
    logger.info("Services cleaned up")


# Database dependency
async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session"""
    if not async_session_maker:
        await init_database()
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Redis dependency
async def get_redis() -> redis.Redis:
    """Get Redis client"""
    if not redis_client:
        await init_redis()
    
    return redis_client


# Service dependencies
async def get_langchain_service() -> LangChainEmailService:
    """Get LangChain service instance"""
    if not _email_sequence_engine:
        await init_services()
    
    return _email_sequence_engine.langchain_service


async def get_delivery_service() -> EmailDeliveryService:
    """Get delivery service instance"""
    if not _email_sequence_engine:
        await init_services()
    
    return _email_sequence_engine.delivery_service


async def get_analytics_service() -> EmailAnalyticsService:
    """Get analytics service instance"""
    if not _email_sequence_engine:
        await init_services()
    
    return _email_sequence_engine.analytics_service


async def get_engine() -> EmailSequenceEngine:
    """Get email sequence engine instance"""
    if not _email_sequence_engine:
        await init_services()
    
    return _email_sequence_engine


# Authentication dependencies
async def get_current_user(
    token: str = Depends(lambda: "user_123")  # Placeholder for JWT token
) -> dict:
    """Get current authenticated user"""
    # In a real implementation, you would:
    # 1. Decode the JWT token
    # 2. Validate the token signature
    # 3. Check token expiration
    # 4. Extract user information
    # 5. Verify user exists in database
    
    # For now, return a placeholder user
    return {
        "id": "user_123",
        "email": "user@example.com",
        "name": "Test User",
        "role": "admin"
    }


async def require_permission(permission: str):
    """Require specific permission"""
    def permission_checker(user: dict = Depends(get_current_user)):
        if permission not in user.get("permissions", []):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        return user
    
    return permission_checker


# Rate limiting dependency
async def rate_limit(
    requests_per_minute: int = 60,
    redis_client: redis.Redis = Depends(get_redis),
    user: dict = Depends(get_current_user)
) -> None:
    """Rate limiting dependency"""
    user_id = user["id"]
    key = f"rate_limit:{user_id}"
    
    # Get current request count
    current_count = await redis_client.get(key)
    
    if current_count is None:
        # First request in the window
        await redis_client.setex(key, 60, 1)
    else:
        count = int(current_count)
        if count >= requests_per_minute:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        await redis_client.incr(key)


# Caching dependency
async def get_cached_data(
    cache_key: str,
    redis_client: redis.Redis = Depends(get_redis)
) -> Optional[dict]:
    """Get cached data from Redis"""
    try:
        cached_data = await redis_client.get(cache_key)
        if cached_data:
            import json
            return json.loads(cached_data)
    except Exception as e:
        logger.warning(f"Error getting cached data for key {cache_key}: {e}")
    
    return None


async def set_cached_data(
    cache_key: str,
    data: dict,
    ttl: int = 300,  # 5 minutes default
    redis_client: redis.Redis = Depends(get_redis)
) -> None:
    """Set cached data in Redis"""
    try:
        import json
        await redis_client.setex(cache_key, ttl, json.dumps(data))
    except Exception as e:
        logger.warning(f"Error setting cached data for key {cache_key}: {e}")


# Health check dependencies
async def check_database_health() -> bool:
    """Check database health"""
    try:
        if not async_session_maker:
            return False
        
        async with async_session_maker() as session:
            await session.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_redis_health() -> bool:
    """Check Redis health"""
    try:
        if not redis_client:
            return False
        
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


async def check_services_health() -> dict:
    """Check all services health"""
    return {
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "engine": _email_sequence_engine is not None and _email_sequence_engine.status.value == "running"
    }


# Context manager for application lifecycle
@asynccontextmanager
async def lifespan():
    """Application lifespan context manager"""
    # Startup
    logger.info("Starting email sequence service...")
    await init_database()
    await init_redis()
    await init_services()
    logger.info("Email sequence service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down email sequence service...")
    await cleanup_services()
    logger.info("Email sequence service shut down successfully")






























