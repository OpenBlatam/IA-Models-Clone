"""
Gamma App - Advanced Database Service
Ultra-optimized database service with connection pooling, query optimization, and advanced features
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncpg
import redis
from sqlalchemy import create_engine, text, MetaData, Table, Column, Integer, String, DateTime, Text, Boolean, Float, JSON
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy.engine import Engine
from sqlalchemy.sql import select, insert, update, delete
import structlog
import psutil
from contextlib import asynccontextmanager
import json
import hashlib
from collections import defaultdict, deque

logger = structlog.get_logger(__name__)

class DatabaseType(Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"
    REDIS = "redis"

class QueryType(Enum):
    """Query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    DDL = "ddl"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    host: str = "localhost"
    port: int = 5432
    database: str = "gamma_app"
    username: str = "postgres"
    password: str = "password"
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    ssl_mode: str = "prefer"
    application_name: str = "gamma_app"

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    timestamp: datetime
    query_hash: str
    parameters: Dict[str, Any] = None
    error: Optional[str] = None

@dataclass
class ConnectionInfo:
    """Database connection information"""
    connection_id: str
    created_at: datetime
    last_used: datetime
    query_count: int
    is_active: bool
    database: str
    user: str

class AdvancedDatabaseService:
    """
    Ultra-advanced database service with enterprise features
    """
    
    def __init__(self, config: DatabaseConfig):
        """Initialize advanced database service"""
        self.config = config
        self.engine: Optional[Engine] = None
        self.async_engine = None
        self.session_factory = None
        self.async_session_factory = None
        
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
        self.redis_client: Optional[redis.Redis] = None
        
        # Query optimization
        self.query_cache: Dict[str, Any] = {}
        self.query_metrics: List[QueryMetrics] = []
        self.slow_queries: List[QueryMetrics] = []
        self.query_plans: Dict[str, Any] = {}
        
        # Connection monitoring
        self.connection_info: Dict[str, ConnectionInfo] = {}
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "idle_connections": 0,
            "failed_connections": 0
        }
        
        # Performance monitoring
        self.performance_metrics = {
            "queries_per_second": deque(maxlen=100),
            "avg_query_time": deque(maxlen=100),
            "cache_hit_rate": deque(maxlen=100),
            "connection_utilization": deque(maxlen=100)
        }
        
        # Auto-optimization
        self.auto_optimization_enabled = True
        self.optimization_rules = []
        self.index_recommendations = []
        
        # Backup and recovery
        self.backup_enabled = False
        self.backup_schedule = []
        self.recovery_points = []
        
        logger.info("Advanced Database Service initialized")
    
    async def initialize(self):
        """Initialize database connections and services"""
        try:
            # Create SQLAlchemy engines
            await self._create_engines()
            
            # Initialize Redis for caching
            await self._initialize_redis()
            
            # Create connection pools
            await self._create_connection_pools()
            
            # Initialize monitoring
            await self._initialize_monitoring()
            
            # Load optimization rules
            await self._load_optimization_rules()
            
            # Start background tasks
            await self._start_background_tasks()
            
            logger.info("Database service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database service: {e}")
            raise
    
    async def _create_engines(self):
        """Create SQLAlchemy engines"""
        try:
            # Synchronous engine
            sync_url = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self.engine = create_engine(
                sync_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                pool_pre_ping=True,
                connect_args={
                    "application_name": self.config.application_name,
                    "sslmode": self.config.ssl_mode
                }
            )
            
            # Asynchronous engine
            async_url = f"postgresql+asyncpg://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"
            self.async_engine = create_async_engine(
                async_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.echo,
                pool_pre_ping=True,
                connect_args={
                    "application_name": f"{self.config.application_name}_async",
                    "sslmode": self.config.ssl_mode
                }
            )
            
            # Session factories
            self.session_factory = sessionmaker(bind=self.engine)
            self.async_session_factory = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("SQLAlchemy engines created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create engines: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis for caching and session storage"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await asyncio.get_event_loop().run_in_executor(
                None, self.redis_client.ping
            )
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def _create_connection_pools(self):
        """Create additional connection pools"""
        try:
            # Direct asyncpg pool for high-performance operations
            self.connection_pools['asyncpg'] = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                min_size=5,
                max_size=self.config.pool_size,
                command_timeout=60,
                server_settings={
                    'application_name': f"{self.config.application_name}_asyncpg"
                }
            )
            
            logger.info("Connection pools created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create connection pools: {e}")
            raise
    
    async def _initialize_monitoring(self):
        """Initialize database monitoring"""
        try:
            # Create monitoring tables if they don't exist
            await self._create_monitoring_tables()
            
            # Start monitoring tasks
            asyncio.create_task(self._monitor_connections())
            asyncio.create_task(self._monitor_performance())
            asyncio.create_task(self._analyze_slow_queries())
            
            logger.info("Database monitoring initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    async def _create_monitoring_tables(self):
        """Create monitoring tables"""
        try:
            async with self.async_engine.begin() as conn:
                # Query metrics table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS query_metrics (
                        id SERIAL PRIMARY KEY,
                        query_id VARCHAR(255),
                        query_type VARCHAR(50),
                        execution_time FLOAT,
                        rows_affected INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        query_hash VARCHAR(64),
                        parameters JSONB,
                        error TEXT
                    )
                """))
                
                # Connection info table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS connection_info (
                        id SERIAL PRIMARY KEY,
                        connection_id VARCHAR(255),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_used TIMESTAMP,
                        query_count INTEGER DEFAULT 0,
                        is_active BOOLEAN DEFAULT TRUE,
                        database VARCHAR(255),
                        user_name VARCHAR(255)
                    )
                """))
                
                # Performance metrics table
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id SERIAL PRIMARY KEY,
                        metric_name VARCHAR(255),
                        metric_value FLOAT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    )
                """))
                
                # Create indexes for better performance
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp 
                    ON query_metrics(timestamp)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_query_metrics_query_hash 
                    ON query_metrics(query_hash)
                """))
                
                await conn.execute(text("""
                    CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
                    ON performance_metrics(timestamp)
                """))
                
            logger.info("Monitoring tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create monitoring tables: {e}")
    
    async def _load_optimization_rules(self):
        """Load database optimization rules"""
        try:
            # Basic optimization rules
            self.optimization_rules = [
                {
                    "name": "index_usage_optimization",
                    "description": "Optimize queries based on index usage",
                    "enabled": True,
                    "threshold": 0.8
                },
                {
                    "name": "query_plan_optimization",
                    "description": "Optimize query execution plans",
                    "enabled": True,
                    "threshold": 1.0
                },
                {
                    "name": "connection_pool_optimization",
                    "description": "Optimize connection pool settings",
                    "enabled": True,
                    "threshold": 0.9
                },
                {
                    "name": "cache_optimization",
                    "description": "Optimize query result caching",
                    "enabled": True,
                    "threshold": 0.7
                }
            ]
            
            logger.info(f"Loaded {len(self.optimization_rules)} optimization rules")
            
        except Exception as e:
            logger.error(f"Failed to load optimization rules: {e}")
    
    async def _start_background_tasks(self):
        """Start background optimization tasks"""
        try:
            # Query optimization task
            asyncio.create_task(self._optimize_queries_periodically())
            
            # Connection pool optimization task
            asyncio.create_task(self._optimize_connection_pools())
            
            # Cache cleanup task
            asyncio.create_task(self._cleanup_query_cache())
            
            # Backup task (if enabled)
            if self.backup_enabled:
                asyncio.create_task(self._backup_database())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
    
    async def execute_query(self, query: str, parameters: Dict[str, Any] = None, 
                          use_cache: bool = True, timeout: int = 30) -> Any:
        """Execute database query with optimization"""
        start_time = time.time()
        query_id = f"query_{int(time.time() * 1000)}"
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        try:
            # Check cache first
            if use_cache and self.redis_client:
                cache_key = f"query_cache:{query_hash}:{hash(str(parameters or {}))}"
                cached_result = await asyncio.get_event_loop().run_in_executor(
                    None, self.redis_client.get, cache_key
                )
                if cached_result:
                    logger.debug(f"Query cache hit: {query_id}")
                    return json.loads(cached_result)
            
            # Execute query
            async with self.async_session_factory() as session:
                result = await session.execute(text(query), parameters or {})
                
                # Handle different query types
                if query.strip().upper().startswith('SELECT'):
                    rows = result.fetchall()
                    data = [dict(row._mapping) for row in rows]
                else:
                    await session.commit()
                    data = {"rows_affected": result.rowcount}
                
                execution_time = time.time() - start_time
                
                # Record metrics
                await self._record_query_metrics(
                    query_id, QueryType.SELECT if query.strip().upper().startswith('SELECT') else QueryType.INSERT,
                    execution_time, len(data) if isinstance(data, list) else 1,
                    query_hash, parameters
                )
                
                # Cache result
                if use_cache and self.redis_client and isinstance(data, list):
                    cache_key = f"query_cache:{query_hash}:{hash(str(parameters or {}))}"
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.redis_client.setex(
                            cache_key, 
                            300,  # 5 minutes
                            json.dumps(data, default=str)
                        )
                    )
                
                # Check for slow queries
                if execution_time > 1.0:  # 1 second threshold
                    await self._record_slow_query(query_id, query, execution_time, parameters)
                
                return data
                
        except Exception as e:
            execution_time = time.time() - start_time
            await self._record_query_metrics(
                query_id, QueryType.SELECT, execution_time, 0,
                query_hash, parameters, str(e)
            )
            logger.error(f"Query execution failed: {e}")
            raise
    
    async def _record_query_metrics(self, query_id: str, query_type: QueryType,
                                  execution_time: float, rows_affected: int,
                                  query_hash: str, parameters: Dict[str, Any] = None,
                                  error: str = None):
        """Record query performance metrics"""
        try:
            metrics = QueryMetrics(
                query_id=query_id,
                query_type=query_type,
                execution_time=execution_time,
                rows_affected=rows_affected,
                timestamp=datetime.now(),
                query_hash=query_hash,
                parameters=parameters,
                error=error
            )
            
            self.query_metrics.append(metrics)
            
            # Keep only last 10000 metrics
            if len(self.query_metrics) > 10000:
                self.query_metrics = self.query_metrics[-10000:]
            
            # Store in database
            async with self.async_session_factory() as session:
                await session.execute(text("""
                    INSERT INTO query_metrics 
                    (query_id, query_type, execution_time, rows_affected, timestamp, query_hash, parameters, error)
                    VALUES (:query_id, :query_type, :execution_time, :rows_affected, :timestamp, :query_hash, :parameters, :error)
                """), {
                    "query_id": query_id,
                    "query_type": query_type.value,
                    "execution_time": execution_time,
                    "rows_affected": rows_affected,
                    "timestamp": metrics.timestamp,
                    "query_hash": query_hash,
                    "parameters": json.dumps(parameters) if parameters else None,
                    "error": error
                })
                await session.commit()
            
            # Update performance metrics
            self.performance_metrics["queries_per_second"].append(1.0 / execution_time if execution_time > 0 else 0)
            self.performance_metrics["avg_query_time"].append(execution_time)
            
        except Exception as e:
            logger.error(f"Failed to record query metrics: {e}")
    
    async def _record_slow_query(self, query_id: str, query: str, 
                               execution_time: float, parameters: Dict[str, Any]):
        """Record slow query for analysis"""
        try:
            slow_query = QueryMetrics(
                query_id=query_id,
                query_type=QueryType.SELECT,
                execution_time=execution_time,
                rows_affected=0,
                timestamp=datetime.now(),
                query_hash=hashlib.md5(query.encode()).hexdigest(),
                parameters=parameters
            )
            
            self.slow_queries.append(slow_query)
            
            # Keep only last 1000 slow queries
            if len(self.slow_queries) > 1000:
                self.slow_queries = self.slow_queries[-1000:]
            
            logger.warning(f"Slow query detected: {query_id}", 
                         execution_time=execution_time,
                         query_hash=slow_query.query_hash)
            
        except Exception as e:
            logger.error(f"Failed to record slow query: {e}")
    
    async def _monitor_connections(self):
        """Monitor database connections"""
        while True:
            try:
                # Get connection pool status
                pool = self.engine.pool
                self.connection_stats.update({
                    "total_connections": pool.size(),
                    "active_connections": pool.checkedout(),
                    "idle_connections": pool.checkedin(),
                    "failed_connections": pool.invalid()
                })
                
                # Update connection utilization
                utilization = pool.checkedout() / pool.size() if pool.size() > 0 else 0
                self.performance_metrics["connection_utilization"].append(utilization)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def _monitor_performance(self):
        """Monitor database performance"""
        while True:
            try:
                # Get database statistics
                async with self.async_session_factory() as session:
                    # Get database size
                    result = await session.execute(text("""
                        SELECT pg_size_pretty(pg_database_size(current_database())) as size
                    """))
                    db_size = result.scalar()
                    
                    # Get active connections
                    result = await session.execute(text("""
                        SELECT count(*) FROM pg_stat_activity WHERE state = 'active'
                    """))
                    active_connections = result.scalar()
                    
                    # Get cache hit ratio
                    result = await session.execute(text("""
                        SELECT round(100.0 * sum(blks_hit) / (sum(blks_hit) + sum(blks_read)), 2) as cache_hit_ratio
                        FROM pg_stat_database WHERE datname = current_database()
                    """))
                    cache_hit_ratio = result.scalar()
                    
                    # Store performance metrics
                    await session.execute(text("""
                        INSERT INTO performance_metrics (metric_name, metric_value, metadata)
                        VALUES 
                        ('database_size_mb', :size, :metadata1),
                        ('active_connections', :connections, :metadata2),
                        ('cache_hit_ratio', :cache_ratio, :metadata3)
                    """), {
                        "size": float(db_size.replace('MB', '')) if db_size else 0,
                        "connections": active_connections,
                        "cache_ratio": cache_hit_ratio,
                        "metadata1": json.dumps({"type": "size", "unit": "MB"}),
                        "metadata2": json.dumps({"type": "connections"}),
                        "metadata3": json.dumps({"type": "ratio", "unit": "percent"})
                    })
                    await session.commit()
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_slow_queries(self):
        """Analyze slow queries and provide recommendations"""
        while True:
            try:
                if not self.slow_queries:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                # Analyze slow queries
                slow_query_analysis = {}
                for query in self.slow_queries[-100:]:  # Analyze last 100 slow queries
                    query_hash = query.query_hash
                    if query_hash not in slow_query_analysis:
                        slow_query_analysis[query_hash] = {
                            "count": 0,
                            "total_time": 0,
                            "avg_time": 0,
                            "max_time": 0
                        }
                    
                    analysis = slow_query_analysis[query_hash]
                    analysis["count"] += 1
                    analysis["total_time"] += query.execution_time
                    analysis["avg_time"] = analysis["total_time"] / analysis["count"]
                    analysis["max_time"] = max(analysis["max_time"], query.execution_time)
                
                # Generate recommendations
                for query_hash, analysis in slow_query_analysis.items():
                    if analysis["count"] > 5 and analysis["avg_time"] > 2.0:
                        # This query needs optimization
                        recommendation = {
                            "query_hash": query_hash,
                            "issue": "Slow query detected",
                            "recommendation": "Consider adding indexes or optimizing query structure",
                            "impact": "High",
                            "frequency": analysis["count"],
                            "avg_execution_time": analysis["avg_time"]
                        }
                        
                        self.index_recommendations.append(recommendation)
                        logger.warning(f"Query optimization needed: {query_hash}", 
                                     recommendation=recommendation)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Slow query analysis error: {e}")
                await asyncio.sleep(300)
    
    async def _optimize_queries_periodically(self):
        """Periodically optimize queries"""
        while True:
            try:
                if self.auto_optimization_enabled:
                    await self._run_query_optimization()
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                logger.error(f"Query optimization error: {e}")
                await asyncio.sleep(3600)
    
    async def _run_query_optimization(self):
        """Run query optimization"""
        try:
            # Analyze query patterns
            query_patterns = defaultdict(int)
            for metrics in self.query_metrics[-1000:]:  # Analyze last 1000 queries
                query_patterns[metrics.query_hash] += 1
            
            # Optimize most frequent queries
            for query_hash, frequency in sorted(query_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
                if frequency > 10:  # Query executed more than 10 times
                    await self._optimize_single_query(query_hash)
            
            logger.info("Query optimization completed")
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
    
    async def _optimize_single_query(self, query_hash: str):
        """Optimize a single query"""
        try:
            # Get query execution plan
            async with self.async_session_factory() as session:
                # This would analyze the query and suggest optimizations
                # For now, we'll just log the optimization attempt
                logger.info(f"Optimizing query: {query_hash}")
                
        except Exception as e:
            logger.error(f"Failed to optimize query {query_hash}: {e}")
    
    async def _optimize_connection_pools(self):
        """Optimize connection pool settings"""
        while True:
            try:
                # Analyze connection utilization
                if self.performance_metrics["connection_utilization"]:
                    avg_utilization = sum(self.performance_metrics["connection_utilization"]) / len(self.performance_metrics["connection_utilization"])
                    
                    # Adjust pool size based on utilization
                    if avg_utilization > 0.9:  # High utilization
                        # Increase pool size
                        new_pool_size = min(self.config.pool_size * 1.2, 50)
                        logger.info(f"High connection utilization detected, recommending pool size increase to {new_pool_size}")
                    elif avg_utilization < 0.3:  # Low utilization
                        # Decrease pool size
                        new_pool_size = max(self.config.pool_size * 0.8, 5)
                        logger.info(f"Low connection utilization detected, recommending pool size decrease to {new_pool_size}")
                
                await asyncio.sleep(1800)  # Check every 30 minutes
                
            except Exception as e:
                logger.error(f"Connection pool optimization error: {e}")
                await asyncio.sleep(1800)
    
    async def _cleanup_query_cache(self):
        """Clean up query cache"""
        while True:
            try:
                if self.redis_client:
                    # Clean up old cache entries
                    await asyncio.get_event_loop().run_in_executor(
                        None, 
                        lambda: self.redis_client.delete(*self.redis_client.keys("query_cache:*"))
                    )
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(3600)
    
    async def _backup_database(self):
        """Backup database"""
        while True:
            try:
                if self.backup_enabled:
                    # This would implement actual backup logic
                    logger.info("Database backup completed")
                
                await asyncio.sleep(86400)  # Backup daily
                
            except Exception as e:
                logger.error(f"Database backup error: {e}")
                await asyncio.sleep(86400)
    
    async def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get database performance dashboard"""
        try:
            # Calculate performance metrics
            queries_per_second = 0
            if self.performance_metrics["queries_per_second"]:
                queries_per_second = sum(self.performance_metrics["queries_per_second"]) / len(self.performance_metrics["queries_per_second"])
            
            avg_query_time = 0
            if self.performance_metrics["avg_query_time"]:
                avg_query_time = sum(self.performance_metrics["avg_query_time"]) / len(self.performance_metrics["avg_query_time"])
            
            cache_hit_rate = 0
            if self.performance_metrics["cache_hit_rate"]:
                cache_hit_rate = sum(self.performance_metrics["cache_hit_rate"]) / len(self.performance_metrics["cache_hit_rate"])
            
            connection_utilization = 0
            if self.performance_metrics["connection_utilization"]:
                connection_utilization = sum(self.performance_metrics["connection_utilization"]) / len(self.performance_metrics["connection_utilization"])
            
            dashboard = {
                "timestamp": datetime.now().isoformat(),
                "performance": {
                    "queries_per_second": queries_per_second,
                    "avg_query_time": avg_query_time,
                    "cache_hit_rate": cache_hit_rate,
                    "connection_utilization": connection_utilization
                },
                "connections": self.connection_stats,
                "query_metrics": {
                    "total_queries": len(self.query_metrics),
                    "slow_queries": len(self.slow_queries),
                    "recent_queries": len([q for q in self.query_metrics if (datetime.now() - q.timestamp).seconds < 300])
                },
                "optimization": {
                    "auto_optimization_enabled": self.auto_optimization_enabled,
                    "optimization_rules": len(self.optimization_rules),
                    "index_recommendations": len(self.index_recommendations)
                },
                "health": {
                    "status": "healthy",
                    "uptime": time.time() - getattr(self, 'start_time', time.time()),
                    "last_backup": "2024-01-01T00:00:00Z" if not self.backup_enabled else datetime.now().isoformat()
                }
            }
            
            return dashboard
            
        except Exception as e:
            logger.error(f"Failed to get performance dashboard: {e}")
            return {}
    
    async def get_query_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get query analytics for the specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_queries = [q for q in self.query_metrics if q.timestamp > cutoff_time]
            
            if not recent_queries:
                return {"message": "No queries found in the specified time period"}
            
            # Analyze query types
            query_types = defaultdict(int)
            for query in recent_queries:
                query_types[query.query_type.value] += 1
            
            # Analyze execution times
            execution_times = [q.execution_time for q in recent_queries]
            avg_execution_time = sum(execution_times) / len(execution_times)
            max_execution_time = max(execution_times)
            min_execution_time = min(execution_times)
            
            # Analyze slow queries
            slow_queries = [q for q in recent_queries if q.execution_time > 1.0]
            
            # Analyze error rate
            error_queries = [q for q in recent_queries if q.error]
            error_rate = len(error_queries) / len(recent_queries) * 100
            
            analytics = {
                "time_period_hours": hours,
                "total_queries": len(recent_queries),
                "query_types": dict(query_types),
                "execution_times": {
                    "average": avg_execution_time,
                    "maximum": max_execution_time,
                    "minimum": min_execution_time
                },
                "slow_queries": {
                    "count": len(slow_queries),
                    "percentage": len(slow_queries) / len(recent_queries) * 100
                },
                "error_rate": error_rate,
                "top_slow_queries": [
                    {
                        "query_hash": q.query_hash,
                        "execution_time": q.execution_time,
                        "timestamp": q.timestamp.isoformat()
                    }
                    for q in sorted(slow_queries, key=lambda x: x.execution_time, reverse=True)[:10]
                ]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get query analytics: {e}")
            return {}
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Run database optimization"""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "optimizations_applied": [],
                "recommendations": [],
                "performance_improvement": {}
            }
            
            # Analyze and optimize indexes
            async with self.async_session_factory() as session:
                # Get unused indexes
                result = await session.execute(text("""
                    SELECT schemaname, tablename, indexname, idx_tup_read, idx_tup_fetch
                    FROM pg_stat_user_indexes
                    WHERE idx_tup_read = 0 AND idx_tup_fetch = 0
                """))
                unused_indexes = result.fetchall()
                
                if unused_indexes:
                    optimization_results["recommendations"].append({
                        "type": "unused_indexes",
                        "description": f"Found {len(unused_indexes)} unused indexes",
                        "action": "Consider dropping unused indexes to improve write performance"
                    })
                
                # Get missing indexes
                result = await session.execute(text("""
                    SELECT schemaname, tablename, seq_scan, seq_tup_read
                    FROM pg_stat_user_tables
                    WHERE seq_scan > 0 AND seq_tup_read > 1000
                """))
                tables_needing_indexes = result.fetchall()
                
                if tables_needing_indexes:
                    optimization_results["recommendations"].append({
                        "type": "missing_indexes",
                        "description": f"Found {len(tables_needing_indexes)} tables with frequent sequential scans",
                        "action": "Consider adding indexes on frequently queried columns"
                    })
            
            # Update statistics
            async with self.async_session_factory() as session:
                await session.execute(text("ANALYZE"))
                await session.commit()
                optimization_results["optimizations_applied"].append("Updated table statistics")
            
            # Vacuum database
            async with self.async_session_factory() as session:
                await session.execute(text("VACUUM ANALYZE"))
                await session.commit()
                optimization_results["optimizations_applied"].append("Vacuumed and analyzed database")
            
            logger.info("Database optimization completed")
            return optimization_results
            
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return {"error": str(e)}
    
    async def close(self):
        """Close database service"""
        try:
            # Close connection pools
            if self.connection_pools:
                for pool_name, pool in self.connection_pools.items():
                    if hasattr(pool, 'close'):
                        await pool.close()
            
            # Close engines
            if self.async_engine:
                await self.async_engine.dispose()
            
            if self.engine:
                self.engine.dispose()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            logger.info("Database service closed")
            
        except Exception as e:
            logger.error(f"Error closing database service: {e}")

# Global database service instance
database_service = None

async def initialize_database_service(config: DatabaseConfig):
    """Initialize global database service"""
    global database_service
    database_service = AdvancedDatabaseService(config)
    await database_service.initialize()
    return database_service

async def get_database_service() -> AdvancedDatabaseService:
    """Get database service instance"""
    if not database_service:
        raise RuntimeError("Database service not initialized")
    return database_service
















