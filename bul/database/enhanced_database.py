"""
Enhanced Database Management for BUL API
======================================

Modern database connection management with connection pooling,
health monitoring, and performance optimization.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, AsyncGenerator
from enum import Enum
import asyncpg
import aioredis
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, String, Integer, DateTime, Text, Boolean, JSON
from sqlalchemy.dialects.postgresql import UUID
import uuid
import json

from ..utils import get_logger, monitor_performance
from ..config import get_config

logger = get_logger(__name__)

# Database Models
Base = declarative_base()

class DocumentRecord(Base):
    """Document generation record"""
    __tablename__ = "document_records"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=True)
    query = Column(Text, nullable=False)
    business_area = Column(String(100), nullable=False)
    document_type = Column(String(100), nullable=False)
    content = Column(Text, nullable=False)
    title = Column(String(500), nullable=False)
    summary = Column(Text, nullable=True)
    word_count = Column(Integer, default=0)
    processing_time = Column(Integer, default=0)  # milliseconds
    confidence_score = Column(Integer, default=0)  # 0-100
    agent_used = Column(String(255), nullable=True)
    format = Column(String(50), default="markdown")
    style = Column(String(50), default="professional")
    language = Column(String(10), default="es")
    metadata = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class UserRecord(Base):
    """User record"""
    __tablename__ = "users"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    metadata = Column(JSON, default={})

class APILogRecord(Base):
    """API usage log record"""
    __tablename__ = "api_logs"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    request_id = Column(String(255), unique=True, nullable=False)
    user_id = Column(String(255), nullable=True)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    status_code = Column(Integer, nullable=False)
    processing_time = Column(Integer, default=0)  # milliseconds
    client_ip = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_data = Column(JSON, default={})
    response_data = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)

class DatabaseStatus(str, Enum):
    """Database connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"

class ConnectionPool:
    """Enhanced connection pool management"""
    
    def __init__(self, database_url: str, pool_size: int = 10, max_overflow: int = 20):
        self.database_url = database_url
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.engine = None
        self.session_factory = None
        self.status = DatabaseStatus.DISCONNECTED
        self.connection_count = 0
        self.active_connections = 0
        self.total_queries = 0
        self.failed_queries = 0
        self.avg_query_time = 0.0
        self.last_health_check = None
        
    async def initialize(self):
        """Initialize database connection pool"""
        try:
            self.status = DatabaseStatus.CONNECTING
            logger.info("Initializing database connection pool...")
            
            # Create async engine
            self.engine = create_async_engine(
                self.database_url,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,
                pool_recycle=3600,  # 1 hour
                echo=False
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            self.status = DatabaseStatus.CONNECTED
            self.last_health_check = datetime.utcnow()
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            self.status = DatabaseStatus.ERROR
            logger.error(f"Failed to initialize database connection pool: {e}")
            raise
    
    async def get_session(self) -> AsyncSession:
        """Get database session"""
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        self.active_connections += 1
        return self.session_factory()
    
    async def close_session(self, session: AsyncSession):
        """Close database session"""
        await session.close()
        self.active_connections = max(0, self.active_connections - 1)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check"""
        try:
            start_time = time.time()
            
            async with self.engine.begin() as conn:
                result = await conn.execute("SELECT 1")
                await result.fetchone()
            
            response_time = (time.time() - start_time) * 1000  # milliseconds
            self.last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "active_connections": self.active_connections,
                "pool_size": self.pool_size,
                "total_queries": self.total_queries,
                "failed_queries": self.failed_queries,
                "success_rate": (
                    (self.total_queries - self.failed_queries) / self.total_queries * 100
                    if self.total_queries > 0 else 100
                )
            }
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_health_check
            }
    
    async def close(self):
        """Close database connections"""
        if self.engine:
            await self.engine.dispose()
        self.status = DatabaseStatus.DISCONNECTED
        logger.info("Database connections closed")

class RedisManager:
    """Enhanced Redis connection management"""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None
        self.status = DatabaseStatus.DISCONNECTED
        self.connection_count = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.last_health_check = None
    
    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.status = DatabaseStatus.CONNECTING
            logger.info("Initializing Redis connection...")
            
            self.redis = await aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis.ping()
            
            self.status = DatabaseStatus.CONNECTED
            self.last_health_check = datetime.utcnow()
            logger.info("Redis connection initialized successfully")
            
        except Exception as e:
            self.status = DatabaseStatus.ERROR
            logger.error(f"Failed to initialize Redis connection: {e}")
            raise
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from Redis"""
        try:
            value = await self.redis.get(key)
            if value:
                self.cache_hits += 1
            else:
                self.cache_misses += 1
            return value
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            self.cache_misses += 1
            return None
    
    async def set(self, key: str, value: str, ttl: int = 3600):
        """Set value in Redis"""
        try:
            await self.redis.setex(key, ttl, value)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    async def delete(self, key: str):
        """Delete key from Redis"""
        try:
            await self.redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        try:
            start_time = time.time()
            await self.redis.ping()
            response_time = (time.time() - start_time) * 1000
            
            self.last_health_check = datetime.utcnow()
            
            return {
                "status": "healthy",
                "response_time_ms": response_time,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "hit_rate": (
                    self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                    if (self.cache_hits + self.cache_misses) > 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": self.last_health_check
            }
    
    async def close(self):
        """Close Redis connection"""
        if self.redis:
            await self.redis.close()
        self.status = DatabaseStatus.DISCONNECTED
        logger.info("Redis connection closed")

class DatabaseManager:
    """Enhanced database manager with comprehensive features"""
    
    def __init__(self):
        self.config = get_config()
        self.connection_pool = None
        self.redis_manager = None
        self.is_initialized = False
        self.start_time = time.time()
        
        # Performance metrics
        self.metrics = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_query_time": 0.0,
            "max_query_time": 0.0,
            "min_query_time": float('inf'),
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    async def initialize(self):
        """Initialize database connections"""
        try:
            logger.info("Initializing database manager...")
            
            # Initialize PostgreSQL connection pool
            self.connection_pool = ConnectionPool(
                database_url=self.config.database.url,
                pool_size=self.config.database.pool_size,
                max_overflow=self.config.database.max_overflow
            )
            await self.connection_pool.initialize()
            
            # Initialize Redis if enabled
            if self.config.cache.enabled and self.config.cache.backend == "redis":
                self.redis_manager = RedisManager(self.config.cache.redis_url)
                await self.redis_manager.initialize()
            
            self.is_initialized = True
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database manager: {e}")
            raise
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self.connection_pool:
            raise RuntimeError("Database not initialized")
        
        session = await self.connection_pool.get_session()
        try:
            yield session
        finally:
            await self.connection_pool.close_session(session)
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute database query with performance tracking"""
        start_time = time.time()
        
        try:
            async with self.get_session() as session:
                result = await session.execute(query, params or {})
                await session.commit()
                
                # Update metrics
                query_time = time.time() - start_time
                self._update_metrics(query_time, success=True)
                
                return result
                
        except Exception as e:
            query_time = time.time() - start_time
            self._update_metrics(query_time, success=False)
            logger.error(f"Database query failed: {e}")
            raise
    
    async def get_document_record(self, document_id: str) -> Optional[DocumentRecord]:
        """Get document record by ID"""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    "SELECT * FROM document_records WHERE id = :id",
                    {"id": document_id}
                )
                return result.fetchone()
        except Exception as e:
            logger.error(f"Failed to get document record: {e}")
            return None
    
    async def save_document_record(self, record: DocumentRecord):
        """Save document record"""
        try:
            async with self.get_session() as session:
                session.add(record)
                await session.commit()
                logger.info(f"Document record saved: {record.id}")
        except Exception as e:
            logger.error(f"Failed to save document record: {e}")
            raise
    
    async def get_user_by_username(self, username: str) -> Optional[UserRecord]:
        """Get user by username"""
        try:
            async with self.get_session() as session:
                result = await session.execute(
                    "SELECT * FROM users WHERE username = :username",
                    {"username": username}
                )
                return result.fetchone()
        except Exception as e:
            logger.error(f"Failed to get user: {e}")
            return None
    
    async def save_api_log(self, log_record: APILogRecord):
        """Save API log record"""
        try:
            async with self.get_session() as session:
                session.add(log_record)
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to save API log: {e}")
    
    async def get_cache(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self.redis_manager:
            return None
        
        return await self.redis_manager.get(key)
    
    async def set_cache(self, key: str, value: str, ttl: int = 3600):
        """Set value in cache"""
        if not self.redis_manager:
            return
        
        await self.redis_manager.set(key, value, ttl)
    
    async def delete_cache(self, key: str):
        """Delete key from cache"""
        if not self.redis_manager:
            return
        
        await self.redis_manager.delete(key)
    
    def _update_metrics(self, query_time: float, success: bool):
        """Update performance metrics"""
        self.metrics["total_queries"] += 1
        
        if success:
            self.metrics["successful_queries"] += 1
        else:
            self.metrics["failed_queries"] += 1
        
        # Update timing metrics
        if query_time < self.metrics["min_query_time"]:
            self.metrics["min_query_time"] = query_time
        if query_time > self.metrics["max_query_time"]:
            self.metrics["max_query_time"] = query_time
        
        # Update average
        total_time = self.metrics["avg_query_time"] * (self.metrics["total_queries"] - 1) + query_time
        self.metrics["avg_query_time"] = total_time / self.metrics["total_queries"]
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive database health check"""
        health_status = {
            "database_manager": {
                "status": "healthy" if self.is_initialized else "unhealthy",
                "uptime": time.time() - self.start_time,
                "metrics": self.metrics.copy()
            }
        }
        
        # Check PostgreSQL
        if self.connection_pool:
            db_health = await self.connection_pool.health_check()
            health_status["postgresql"] = db_health
        else:
            health_status["postgresql"] = {"status": "not_initialized"}
        
        # Check Redis
        if self.redis_manager:
            redis_health = await self.redis_manager.health_check()
            health_status["redis"] = redis_health
        else:
            health_status["redis"] = {"status": "not_initialized"}
        
        return health_status
    
    async def close(self):
        """Close all database connections"""
        if self.connection_pool:
            await self.connection_pool.close()
        
        if self.redis_manager:
            await self.redis_manager.close()
        
        self.is_initialized = False
        logger.info("Database manager closed")

# Global database manager
_database_manager: Optional[DatabaseManager] = None

async def get_database_manager() -> DatabaseManager:
    """Get global database manager"""
    global _database_manager
    if _database_manager is None:
        _database_manager = DatabaseManager()
        await _database_manager.initialize()
    return _database_manager

# Database decorators
def with_database_session(func):
    """Decorator to provide database session"""
    async def wrapper(*args, **kwargs):
        db_manager = await get_database_manager()
        async with db_manager.get_session() as session:
            kwargs['session'] = session
            return await func(*args, **kwargs)
    return wrapper

def with_cache(func):
    """Decorator to provide cache access"""
    async def wrapper(*args, **kwargs):
        db_manager = await get_database_manager()
        kwargs['cache'] = {
            'get': db_manager.get_cache,
            'set': db_manager.set_cache,
            'delete': db_manager.delete_cache
        }
        return await func(*args, **kwargs)
    return wrapper

# Database utilities
async def create_tables():
    """Create database tables"""
    db_manager = await get_database_manager()
    if db_manager.connection_pool:
        async with db_manager.connection_pool.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

async def drop_tables():
    """Drop database tables"""
    db_manager = await get_database_manager()
    if db_manager.connection_pool:
        async with db_manager.connection_pool.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)












