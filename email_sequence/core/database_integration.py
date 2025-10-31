"""
Database Integration for Email Sequence System

Provides advanced database integration with connection pooling, caching,
and optimization for the email sequence system.
"""

import asyncio
import logging
import time
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union, AsyncGenerator
from dataclasses import dataclass
from collections import defaultdict, deque
from enum import Enum
from datetime import datetime, timedelta
import json
import hashlib

# Database imports
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import text, select, update, delete, insert
from sqlalchemy.orm import DeclarativeBase
import asyncpg
import aioredis
import redis

# Models
from ..models.sequence import EmailSequence, SequenceStep, SequenceTrigger
from ..models.subscriber import Subscriber, SubscriberSegment
from ..models.template import EmailTemplate, TemplateVariable
from ..models.campaign import EmailCampaign, CampaignMetrics

logger = logging.getLogger(__name__)

# Constants
MAX_CONNECTIONS = 50
MAX_OVERFLOW = 30
POOL_TIMEOUT = 30
POOL_RECYCLE = 3600
CACHE_TTL = 3600  # 1 hour
MAX_CACHE_SIZE = 10000
QUERY_TIMEOUT = 30


class DatabaseType(Enum):
    """Supported database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"


class CacheStrategy(Enum):
    """Cache strategies"""
    NONE = "none"
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: DatabaseType
    connection_string: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    enable_caching: bool = True
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID
    cache_ttl: int = CACHE_TTL
    max_cache_size: int = MAX_CACHE_SIZE
    enable_metrics: bool = True
    enable_query_logging: bool = False
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_hash: str
    execution_time: float
    rows_affected: int
    cache_hit: bool
    timestamp: datetime
    table_name: str
    operation_type: str
    parameters: Dict[str, Any]


class DatabaseConnectionPool:
    """Advanced database connection pool with monitoring"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.engine = None
        self.session_factory = None
        self.redis_client = None
        self.connection_pool = None
        
        # Performance tracking
        self.query_metrics: List[QueryMetrics] = []
        self.connection_usage = defaultdict(int)
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        # Connection pool management
        self.active_connections = 0
        self.max_connections = config.pool_size + config.max_overflow
        self.connection_semaphore = asyncio.Semaphore(config.pool_size)
        
        # Cache management
        self.memory_cache = {}
        self.cache_keys = deque()
        
        logger.info(f"Database Connection Pool initialized for {config.database_type.value}")
    
    async def initialize(self) -> None:
        """Initialize database connections and pools"""
        try:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                await self._initialize_postgresql()
            elif self.config.database_type == DatabaseType.REDIS:
                await self._initialize_redis()
            else:
                await self._initialize_sqlalchemy()
            
            # Initialize caching if enabled
            if self.config.enable_caching:
                await self._initialize_caching()
            
            logger.info("Database connection pool initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database pool: {e}")
            raise
    
    async def _initialize_postgresql(self) -> None:
        """Initialize PostgreSQL with asyncpg"""
        self.connection_pool = await asyncpg.create_pool(
            self.config.connection_string,
            min_size=5,
            max_size=self.config.pool_size,
            command_timeout=self.config.pool_timeout
        )
    
    async def _initialize_redis(self) -> None:
        """Initialize Redis connection"""
        self.redis_client = redis.from_url(self.config.connection_string)
        await self.redis_client.ping()
    
    async def _initialize_sqlalchemy(self) -> None:
        """Initialize SQLAlchemy async engine"""
        self.engine = create_async_engine(
            self.config.connection_string,
            poolclass=QueuePool,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_timeout=self.config.pool_timeout,
            pool_recycle=self.config.pool_recycle,
            echo=self.config.echo
        )
        
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def _initialize_caching(self) -> None:
        """Initialize caching system"""
        if self.config.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
            if not self.redis_client:
                self.redis_client = redis.from_url("redis://localhost:6379")
                await self.redis_client.ping()
    
    async def get_session(self) -> AsyncSession:
        """Get database session with connection pooling"""
        if not self.session_factory:
            raise RuntimeError("SQLAlchemy not initialized")
        
        async with self.connection_semaphore:
            self.active_connections += 1
            session = self.session_factory()
            return session
    
    async def close_session(self, session: AsyncSession) -> None:
        """Close database session"""
        if session:
            await session.close()
            self.active_connections -= 1
    
    async def execute_query(
        self,
        query: str,
        parameters: Dict[str, Any] = None,
        cache_key: str = None,
        use_cache: bool = True
    ) -> Any:
        """Execute query with caching and metrics"""
        start_time = time.time()
        query_hash = self._hash_query(query, parameters)
        
        # Check cache first
        if use_cache and cache_key:
            cached_result = await self._get_cached_result(cache_key)
            if cached_result is not None:
                self._record_metrics(query_hash, time.time() - start_time, 0, True)
                return cached_result
        
        try:
            # Execute query
            if self.connection_pool:
                async with self.connection_pool.acquire() as conn:
                    result = await conn.fetch(query, *(parameters or {}).values())
            else:
                async with self.get_session() as session:
                    result = await session.execute(text(query), parameters or {})
                    result = result.fetchall()
            
            # Cache result
            if use_cache and cache_key:
                await self._cache_result(cache_key, result)
            
            execution_time = time.time() - start_time
            self._record_metrics(query_hash, execution_time, len(result), False)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            raise
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Any]:
        """Get cached result"""
        try:
            if self.config.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                if cache_key in self.memory_cache:
                    self.cache_stats["hits"] += 1
                    return self.memory_cache[cache_key]
            
            if self.config.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_client:
                    cached = await self.redis_client.get(cache_key)
                    if cached:
                        self.cache_stats["hits"] += 1
                        return json.loads(cached)
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return None
    
    async def _cache_result(self, cache_key: str, result: Any) -> None:
        """Cache result"""
        try:
            if self.config.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                # Manage memory cache size
                if len(self.memory_cache) >= self.config.max_cache_size:
                    oldest_key = self.cache_keys.popleft()
                    del self.memory_cache[oldest_key]
                    self.cache_stats["evictions"] += 1
                
                self.memory_cache[cache_key] = result
                self.cache_keys.append(cache_key)
            
            if self.config.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.redis_client:
                    await self.redis_client.setex(
                        cache_key,
                        self.config.cache_ttl,
                        json.dumps(result)
                    )
                    
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _hash_query(self, query: str, parameters: Dict[str, Any] = None) -> str:
        """Hash query for metrics"""
        query_str = f"{query}:{json.dumps(parameters or {}, sort_keys=True)}"
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _record_metrics(
        self,
        query_hash: str,
        execution_time: float,
        rows_affected: int,
        cache_hit: bool
    ) -> None:
        """Record query metrics"""
        metric = QueryMetrics(
            query_hash=query_hash,
            execution_time=execution_time,
            rows_affected=rows_affected,
            cache_hit=cache_hit,
            timestamp=datetime.utcnow(),
            table_name="",  # Would need to parse query
            operation_type="",  # Would need to parse query
            parameters={}
        )
        self.query_metrics.append(metric)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.query_metrics:
            return {}
        
        execution_times = [m.execution_time for m in self.query_metrics]
        cache_hits = sum(1 for m in self.query_metrics if m.cache_hit)
        
        return {
            "total_queries": len(self.query_metrics),
            "avg_execution_time": sum(execution_times) / len(execution_times),
            "max_execution_time": max(execution_times),
            "min_execution_time": min(execution_times),
            "cache_hit_rate": cache_hits / len(self.query_metrics),
            "active_connections": self.active_connections,
            "cache_stats": self.cache_stats
        }
    
    async def cleanup(self) -> None:
        """Cleanup database connections"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
            
            if self.engine:
                await self.engine.dispose()
            
            if self.redis_client:
                await self.redis_client.close()
                
            logger.info("Database connection pool cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


class EmailSequenceRepository:
    """Repository for email sequence database operations"""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        self.db_pool = db_pool
    
    async def create_sequence(self, sequence: EmailSequence) -> EmailSequence:
        """Create a new email sequence"""
        cache_key = f"sequence:{sequence.id}"
        
        # This would be a real database insert in production
        # For now, we'll simulate it
        await self.db_pool.execute_query(
            "INSERT INTO email_sequences (id, name, description, status) VALUES ($1, $2, $3, $4)",
            {
                "id": str(sequence.id),
                "name": sequence.name,
                "description": sequence.description,
                "status": sequence.status.value
            },
            cache_key=cache_key
        )
        
        return sequence
    
    async def get_sequence(self, sequence_id: str) -> Optional[EmailSequence]:
        """Get email sequence by ID"""
        cache_key = f"sequence:{sequence_id}"
        
        result = await self.db_pool.execute_query(
            "SELECT * FROM email_sequences WHERE id = $1",
            {"id": sequence_id},
            cache_key=cache_key,
            use_cache=True
        )
        
        if result:
            # Convert result to EmailSequence object
            # This is simplified - in production you'd have proper ORM mapping
            return EmailSequence(**result[0])
        
        return None
    
    async def update_sequence(self, sequence: EmailSequence) -> EmailSequence:
        """Update email sequence"""
        cache_key = f"sequence:{sequence.id}"
        
        await self.db_pool.execute_query(
            "UPDATE email_sequences SET name = $1, description = $2, status = $3 WHERE id = $4",
            {
                "name": sequence.name,
                "description": sequence.description,
                "status": sequence.status.value,
                "id": str(sequence.id)
            },
            cache_key=cache_key
        )
        
        # Invalidate cache
        await self._invalidate_cache(cache_key)
        
        return sequence
    
    async def delete_sequence(self, sequence_id: str) -> bool:
        """Delete email sequence"""
        cache_key = f"sequence:{sequence_id}"
        
        result = await self.db_pool.execute_query(
            "DELETE FROM email_sequences WHERE id = $1",
            {"id": sequence_id}
        )
        
        # Invalidate cache
        await self._invalidate_cache(cache_key)
        
        return len(result) > 0
    
    async def list_sequences(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[EmailSequence]:
        """List email sequences with filtering"""
        cache_key = f"sequences:{status}:{limit}:{offset}"
        
        query = "SELECT * FROM email_sequences"
        parameters = {}
        
        if status:
            query += " WHERE status = $1"
            parameters["status"] = status
        
        query += " ORDER BY created_at DESC LIMIT $2 OFFSET $3"
        parameters["limit"] = limit
        parameters["offset"] = offset
        
        result = await self.db_pool.execute_query(
            query,
            parameters,
            cache_key=cache_key,
            use_cache=True
        )
        
        # Convert results to EmailSequence objects
        sequences = []
        for row in result:
            sequences.append(EmailSequence(**row))
        
        return sequences
    
    async def _invalidate_cache(self, cache_key: str) -> None:
        """Invalidate cache entry"""
        try:
            if self.db_pool.config.cache_strategy in [CacheStrategy.MEMORY, CacheStrategy.HYBRID]:
                if cache_key in self.db_pool.memory_cache:
                    del self.db_pool.memory_cache[cache_key]
            
            if self.db_pool.config.cache_strategy in [CacheStrategy.REDIS, CacheStrategy.HYBRID]:
                if self.db_pool.redis_client:
                    await self.db_pool.redis_client.delete(cache_key)
                    
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")


class SubscriberRepository:
    """Repository for subscriber database operations"""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        self.db_pool = db_pool
    
    async def create_subscriber(self, subscriber: Subscriber) -> Subscriber:
        """Create a new subscriber"""
        cache_key = f"subscriber:{subscriber.id}"
        
        await self.db_pool.execute_query(
            "INSERT INTO subscribers (id, email, first_name, last_name, status) VALUES ($1, $2, $3, $4, $5)",
            {
                "id": str(subscriber.id),
                "email": subscriber.email,
                "first_name": subscriber.first_name,
                "last_name": subscriber.last_name,
                "status": subscriber.status.value
            },
            cache_key=cache_key
        )
        
        return subscriber
    
    async def get_subscriber(self, subscriber_id: str) -> Optional[Subscriber]:
        """Get subscriber by ID"""
        cache_key = f"subscriber:{subscriber_id}"
        
        result = await self.db_pool.execute_query(
            "SELECT * FROM subscribers WHERE id = $1",
            {"id": subscriber_id},
            cache_key=cache_key,
            use_cache=True
        )
        
        if result:
            return Subscriber(**result[0])
        
        return None
    
    async def get_subscriber_by_email(self, email: str) -> Optional[Subscriber]:
        """Get subscriber by email"""
        cache_key = f"subscriber_email:{email}"
        
        result = await self.db_pool.execute_query(
            "SELECT * FROM subscribers WHERE email = $1",
            {"email": email},
            cache_key=cache_key,
            use_cache=True
        )
        
        if result:
            return Subscriber(**result[0])
        
        return None
    
    async def list_subscribers(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Subscriber]:
        """List subscribers with filtering"""
        cache_key = f"subscribers:{status}:{limit}:{offset}"
        
        query = "SELECT * FROM subscribers"
        parameters = {}
        
        if status:
            query += " WHERE status = $1"
            parameters["status"] = status
        
        query += " ORDER BY created_at DESC LIMIT $2 OFFSET $3"
        parameters["limit"] = limit
        parameters["offset"] = offset
        
        result = await self.db_pool.execute_query(
            query,
            parameters,
            cache_key=cache_key,
            use_cache=True
        )
        
        subscribers = []
        for row in result:
            subscribers.append(Subscriber(**row))
        
        return subscribers


class TemplateRepository:
    """Repository for email template database operations"""
    
    def __init__(self, db_pool: DatabaseConnectionPool):
        self.db_pool = db_pool
    
    async def create_template(self, template: EmailTemplate) -> EmailTemplate:
        """Create a new email template"""
        cache_key = f"template:{template.id}"
        
        await self.db_pool.execute_query(
            "INSERT INTO email_templates (id, name, subject, html_content, template_type) VALUES ($1, $2, $3, $4, $5)",
            {
                "id": str(template.id),
                "name": template.name,
                "subject": template.subject,
                "html_content": template.html_content,
                "template_type": template.template_type.value
            },
            cache_key=cache_key
        )
        
        return template
    
    async def get_template(self, template_id: str) -> Optional[EmailTemplate]:
        """Get email template by ID"""
        cache_key = f"template:{template_id}"
        
        result = await self.db_pool.execute_query(
            "SELECT * FROM email_templates WHERE id = $1",
            {"id": template_id},
            cache_key=cache_key,
            use_cache=True
        )
        
        if result:
            return EmailTemplate(**result[0])
        
        return None
    
    async def list_templates(
        self,
        template_type: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[EmailTemplate]:
        """List email templates with filtering"""
        cache_key = f"templates:{template_type}:{status}:{limit}:{offset}"
        
        query = "SELECT * FROM email_templates"
        parameters = {}
        conditions = []
        
        if template_type:
            conditions.append("template_type = $1")
            parameters["template_type"] = template_type
        
        if status:
            conditions.append("status = $2")
            parameters["status"] = status
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY created_at DESC LIMIT $3 OFFSET $4"
        parameters["limit"] = limit
        parameters["offset"] = offset
        
        result = await self.db_pool.execute_query(
            query,
            parameters,
            cache_key=cache_key,
            use_cache=True
        )
        
        templates = []
        for row in result:
            templates.append(EmailTemplate(**row))
        
        return templates 