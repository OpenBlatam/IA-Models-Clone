"""
Advanced Database Optimization for Microservices
Features: Connection pooling, query optimization, intelligent caching, read replicas, sharding
"""

import asyncio
import time
import json
import hashlib
from typing import Dict, Any, Optional, List, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque
import statistics
import threading
from contextlib import asynccontextmanager

# Database imports
try:
    import asyncpg
    import asyncpg.pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import aiomysql
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import sqlalchemy
    from sqlalchemy import text, create_engine
    from sqlalchemy.pool import QueuePool
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

logger = logging.getLogger(__name__)

class DatabaseType(Enum):
    """Database types"""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    REDIS = "redis"
    MONGODB = "mongodb"

class QueryType(Enum):
    """Query types"""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    CREATE = "create"
    DROP = "drop"
    ALTER = "alter"

class ShardingStrategy(Enum):
    """Sharding strategies"""
    RANGE = "range"
    HASH = "hash"
    DIRECTORY = "directory"
    CONSISTENT_HASH = "consistent_hash"

@dataclass
class DatabaseConfig:
    """Database configuration"""
    database_type: DatabaseType
    host: str
    port: int
    database: str
    username: str
    password: str
    min_connections: int = 5
    max_connections: int = 20
    connection_timeout: int = 30
    command_timeout: int = 60
    ssl_enabled: bool = False
    read_replicas: List[str] = field(default_factory=list)
    sharding_enabled: bool = False
    sharding_strategy: ShardingStrategy = ShardingStrategy.HASH
    shard_count: int = 4

@dataclass
class QueryMetrics:
    """Query performance metrics"""
    query_id: str
    query_type: QueryType
    execution_time: float
    rows_affected: int
    cache_hit: bool
    timestamp: float
    connection_id: str
    shard_id: Optional[str] = None

@dataclass
class ConnectionPoolStats:
    """Connection pool statistics"""
    total_connections: int
    active_connections: int
    idle_connections: int
    waiting_requests: int
    connection_errors: int
    avg_connection_time: float
    avg_query_time: float

class QueryOptimizer:
    """
    Advanced query optimization system
    """
    
    def __init__(self):
        self.query_cache: Dict[str, Any] = {}
        self.query_stats: Dict[str, List[QueryMetrics]] = defaultdict(list)
        self.slow_queries: deque = deque(maxlen=1000)
        self.query_patterns: Dict[str, Dict[str, Any]] = {}
        self.optimization_rules: List[Callable] = []
        
        # Initialize optimization rules
        self._initialize_optimization_rules()
    
    def _initialize_optimization_rules(self):
        """Initialize query optimization rules"""
        self.optimization_rules = [
            self._optimize_select_queries,
            self._optimize_join_queries,
            self._optimize_where_clauses,
            self._optimize_order_by,
            self._optimize_group_by,
            self._add_missing_indexes
        ]
    
    async def optimize_query(self, query: str, params: Dict[str, Any] = None) -> Tuple[str, Dict[str, Any]]:
        """Optimize SQL query"""
        try:
            original_query = query
            optimization_suggestions = []
            
            # Apply optimization rules
            for rule in self.optimization_rules:
                optimized_query, suggestions = await rule(query, params or {})
                if optimized_query != query:
                    query = optimized_query
                    optimization_suggestions.extend(suggestions)
            
            # Log optimization
            if optimization_suggestions:
                logger.info(f"Query optimized: {len(optimization_suggestions)} improvements")
                for suggestion in optimization_suggestions:
                    logger.debug(f"Optimization: {suggestion}")
            
            return query, {
                "original_query": original_query,
                "optimized_query": query,
                "suggestions": optimization_suggestions,
                "optimization_applied": len(optimization_suggestions) > 0
            }
            
        except Exception as e:
            logger.error(f"Query optimization failed: {e}")
            return query, {"error": str(e)}
    
    async def _optimize_select_queries(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize SELECT queries"""
        suggestions = []
        optimized_query = query
        
        # Check for SELECT *
        if "SELECT *" in query.upper():
            suggestions.append("Consider specifying columns instead of SELECT *")
        
        # Check for LIMIT without ORDER BY
        if "LIMIT" in query.upper() and "ORDER BY" not in query.upper():
            suggestions.append("Add ORDER BY clause when using LIMIT for consistent results")
        
        # Check for unnecessary DISTINCT
        if "DISTINCT" in query.upper():
            suggestions.append("Verify if DISTINCT is necessary - it can be expensive")
        
        return optimized_query, suggestions
    
    async def _optimize_join_queries(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize JOIN queries"""
        suggestions = []
        optimized_query = query
        
        # Check for missing JOIN conditions
        if "JOIN" in query.upper() and "ON" not in query.upper():
            suggestions.append("Add explicit JOIN conditions")
        
        # Check for multiple JOINs
        join_count = query.upper().count("JOIN")
        if join_count > 5:
            suggestions.append("Consider breaking down complex JOINs or using subqueries")
        
        return optimized_query, suggestions
    
    async def _optimize_where_clauses(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize WHERE clauses"""
        suggestions = []
        optimized_query = query
        
        # Check for functions in WHERE clause
        if "WHERE" in query.upper():
            # This is a simplified check - in real implementation, you'd parse the query
            if any(func in query.upper() for func in ["UPPER(", "LOWER(", "SUBSTRING(", "DATE("]):
                suggestions.append("Avoid functions in WHERE clause - consider computed columns or indexes")
        
        return optimized_query, suggestions
    
    async def _optimize_order_by(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize ORDER BY clauses"""
        suggestions = []
        optimized_query = query
        
        # Check for ORDER BY without LIMIT
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            suggestions.append("Consider adding LIMIT when using ORDER BY")
        
        return optimized_query, suggestions
    
    async def _optimize_group_by(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Optimize GROUP BY clauses"""
        suggestions = []
        optimized_query = query
        
        # Check for GROUP BY without aggregation
        if "GROUP BY" in query.upper():
            has_aggregation = any(func in query.upper() for func in ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("])
            if not has_aggregation:
                suggestions.append("GROUP BY without aggregation functions may be unnecessary")
        
        return optimized_query, suggestions
    
    async def _add_missing_indexes(self, query: str, params: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Suggest missing indexes"""
        suggestions = []
        optimized_query = query
        
        # This would analyze the query and suggest indexes
        # For now, just return the original query
        return optimized_query, suggestions
    
    def record_query_metrics(self, metrics: QueryMetrics):
        """Record query performance metrics"""
        self.query_stats[metrics.query_id].append(metrics)
        
        # Track slow queries
        if metrics.execution_time > 1.0:  # Queries taking more than 1 second
            self.slow_queries.append(metrics)
        
        # Update query patterns
        query_pattern = self._extract_query_pattern(metrics.query_id)
        if query_pattern not in self.query_patterns:
            self.query_patterns[query_pattern] = {
                "count": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "max_time": 0.0,
                "min_time": float('inf')
            }
        
        pattern_stats = self.query_patterns[query_pattern]
        pattern_stats["count"] += 1
        pattern_stats["total_time"] += metrics.execution_time
        pattern_stats["avg_time"] = pattern_stats["total_time"] / pattern_stats["count"]
        pattern_stats["max_time"] = max(pattern_stats["max_time"], metrics.execution_time)
        pattern_stats["min_time"] = min(pattern_stats["min_time"], metrics.execution_time)
    
    def _extract_query_pattern(self, query: str) -> str:
        """Extract query pattern for analysis"""
        # Remove specific values and normalize the query
        import re
        
        # Replace numbers with placeholder
        pattern = re.sub(r'\d+', '?', query)
        
        # Replace strings with placeholder
        pattern = re.sub(r"'[^']*'", '?', pattern)
        pattern = re.sub(r'"[^"]*"', '?', pattern)
        
        # Normalize whitespace
        pattern = re.sub(r'\s+', ' ', pattern).strip()
        
        return pattern
    
    def get_query_analytics(self) -> Dict[str, Any]:
        """Get query analytics"""
        return {
            "total_queries": sum(len(stats) for stats in self.query_stats.values()),
            "slow_queries": len(self.slow_queries),
            "query_patterns": len(self.query_patterns),
            "top_slow_queries": [
                {
                    "pattern": pattern,
                    "avg_time": stats["avg_time"],
                    "count": stats["count"],
                    "max_time": stats["max_time"]
                }
                for pattern, stats in sorted(
                    self.query_patterns.items(),
                    key=lambda x: x[1]["avg_time"],
                    reverse=True
                )[:10]
            ],
            "recent_slow_queries": [
                {
                    "query_id": q.query_id,
                    "execution_time": q.execution_time,
                    "timestamp": q.timestamp
                }
                for q in list(self.slow_queries)[-10:]
            ]
        }

class ConnectionPoolManager:
    """
    Advanced connection pool management
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.pools: Dict[str, Any] = {}
        self.pool_stats: Dict[str, ConnectionPoolStats] = {}
        self.connection_history: deque = deque(maxlen=1000)
        self.pool_lock = asyncio.Lock()
        
    async def create_pool(self, pool_name: str = "default") -> bool:
        """Create database connection pool"""
        try:
            async with self.pool_lock:
                if pool_name in self.pools:
                    return True
                
                if self.config.database_type == DatabaseType.POSTGRESQL and POSTGRES_AVAILABLE:
                    pool = await asyncpg.create_pool(
                        host=self.config.host,
                        port=self.config.port,
                        database=self.config.database,
                        user=self.config.username,
                        password=self.config.password,
                        min_size=self.config.min_connections,
                        max_size=self.config.max_connections,
                        command_timeout=self.config.command_timeout,
                        ssl=self.config.ssl_enabled
                    )
                    self.pools[pool_name] = pool
                
                elif self.config.database_type == DatabaseType.MYSQL and MYSQL_AVAILABLE:
                    pool = await aiomysql.create_pool(
                        host=self.config.host,
                        port=self.config.port,
                        db=self.config.database,
                        user=self.config.username,
                        password=self.config.password,
                        minsize=self.config.min_connections,
                        maxsize=self.config.max_connections,
                        connect_timeout=self.config.connection_timeout
                    )
                    self.pools[pool_name] = pool
                
                else:
                    logger.error(f"Unsupported database type: {self.config.database_type}")
                    return False
                
                # Initialize pool stats
                self.pool_stats[pool_name] = ConnectionPoolStats(
                    total_connections=0,
                    active_connections=0,
                    idle_connections=0,
                    waiting_requests=0,
                    connection_errors=0,
                    avg_connection_time=0.0,
                    avg_query_time=0.0
                )
                
                logger.info(f"Created connection pool: {pool_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create connection pool {pool_name}: {e}")
            return False
    
    async def get_connection(self, pool_name: str = "default"):
        """Get connection from pool"""
        try:
            if pool_name not in self.pools:
                await self.create_pool(pool_name)
            
            pool = self.pools[pool_name]
            
            if self.config.database_type == DatabaseType.POSTGRESQL:
                return await pool.acquire()
            elif self.config.database_type == DatabaseType.MYSQL:
                return await pool.acquire()
            
        except Exception as e:
            logger.error(f"Failed to get connection from pool {pool_name}: {e}")
            self.pool_stats[pool_name].connection_errors += 1
            raise
    
    async def release_connection(self, connection, pool_name: str = "default"):
        """Release connection back to pool"""
        try:
            if pool_name in self.pools:
                pool = self.pools[pool_name]
                await pool.release(connection)
        except Exception as e:
            logger.error(f"Failed to release connection to pool {pool_name}: {e}")
    
    @asynccontextmanager
    async def get_connection_context(self, pool_name: str = "default"):
        """Get connection with automatic cleanup"""
        connection = None
        try:
            connection = await self.get_connection(pool_name)
            yield connection
        finally:
            if connection:
                await self.release_connection(connection, pool_name)
    
    async def close_pool(self, pool_name: str = "default"):
        """Close connection pool"""
        try:
            if pool_name in self.pools:
                pool = self.pools[pool_name]
                await pool.close()
                del self.pools[pool_name]
                del self.pool_stats[pool_name]
                logger.info(f"Closed connection pool: {pool_name}")
        except Exception as e:
            logger.error(f"Failed to close connection pool {pool_name}: {e}")
    
    async def close_all_pools(self):
        """Close all connection pools"""
        for pool_name in list(self.pools.keys()):
            await self.close_pool(pool_name)
    
    def get_pool_stats(self, pool_name: str = "default") -> Optional[ConnectionPoolStats]:
        """Get connection pool statistics"""
        if pool_name not in self.pools:
            return None
        
        try:
            pool = self.pools[pool_name]
            
            if self.config.database_type == DatabaseType.POSTGRESQL:
                stats = ConnectionPoolStats(
                    total_connections=pool.get_size(),
                    active_connections=pool.get_idle_size(),
                    idle_connections=pool.get_idle_size(),
                    waiting_requests=0,  # asyncpg doesn't expose this
                    connection_errors=self.pool_stats[pool_name].connection_errors,
                    avg_connection_time=0.0,  # Would need to track this
                    avg_query_time=0.0  # Would need to track this
                )
            else:
                stats = self.pool_stats[pool_name]
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get pool stats for {pool_name}: {e}")
            return None

class DatabaseSharding:
    """
    Database sharding implementation
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.shards: Dict[str, ConnectionPoolManager] = {}
        self.shard_keys: Dict[str, str] = {}
        self.sharding_enabled = config.sharding_enabled
        
        if self.sharding_enabled:
            self._initialize_shards()
    
    def _initialize_shards(self):
        """Initialize database shards"""
        for i in range(self.config.shard_count):
            shard_name = f"shard_{i}"
            shard_config = DatabaseConfig(
                database_type=self.config.database_type,
                host=self.config.host,
                port=self.config.port + i,  # Different ports for different shards
                database=f"{self.config.database}_shard_{i}",
                username=self.config.username,
                password=self.config.password,
                min_connections=self.config.min_connections,
                max_connections=self.config.max_connections
            )
            
            self.shards[shard_name] = ConnectionPoolManager(shard_config)
    
    def get_shard_for_key(self, key: str) -> str:
        """Get shard name for a given key"""
        if not self.sharding_enabled:
            return "default"
        
        if self.config.sharding_strategy == ShardingStrategy.HASH:
            # Simple hash-based sharding
            hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
            shard_index = hash_value % self.config.shard_count
            return f"shard_{shard_index}"
        
        elif self.config.sharding_strategy == ShardingStrategy.RANGE:
            # Range-based sharding (simplified)
            # In real implementation, you'd have range definitions
            return f"shard_{ord(key[0]) % self.config.shard_count}"
        
        else:
            return "shard_0"
    
    async def execute_on_shard(self, query: str, params: Dict[str, Any], shard_key: str):
        """Execute query on specific shard"""
        shard_name = self.get_shard_for_key(shard_key)
        
        if shard_name not in self.shards:
            raise ValueError(f"Shard {shard_name} not found")
        
        shard_manager = self.shards[shard_name]
        
        async with shard_manager.get_connection_context() as connection:
            if self.config.database_type == DatabaseType.POSTGRESQL:
                return await connection.fetch(query, *params.values())
            elif self.config.database_type == DatabaseType.MYSQL:
                cursor = await connection.cursor()
                await cursor.execute(query, params)
                return await cursor.fetchall()
    
    async def close_all_shards(self):
        """Close all shard connections"""
        for shard_manager in self.shards.values():
            await shard_manager.close_all_pools()

class DatabaseCache:
    """
    Intelligent database query caching
    """
    
    def __init__(self, redis_client: Optional[aioredis.Redis] = None):
        self.redis = redis_client
        self.local_cache: Dict[str, Any] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0
        }
        self.cache_ttl = 300  # 5 minutes default
        self.max_local_cache_size = 1000
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query,
            "params": params
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def get(self, query: str, params: Dict[str, Any]) -> Optional[Any]:
        """Get cached query result"""
        cache_key = self._generate_cache_key(query, params)
        
        # Check local cache first
        if cache_key in self.local_cache:
            self.cache_stats["hits"] += 1
            return self.local_cache[cache_key]
        
        # Check Redis cache
        if self.redis:
            try:
                cached_data = await self.redis.get(f"db_cache:{cache_key}")
                if cached_data:
                    result = json.loads(cached_data)
                    # Store in local cache
                    self.local_cache[cache_key] = result
                    self.cache_stats["hits"] += 1
                    return result
            except Exception as e:
                logger.error(f"Redis cache get failed: {e}")
        
        self.cache_stats["misses"] += 1
        return None
    
    async def set(self, query: str, params: Dict[str, Any], result: Any, ttl: int = None):
        """Cache query result"""
        cache_key = self._generate_cache_key(query, params)
        ttl = ttl or self.cache_ttl
        
        # Store in local cache
        if len(self.local_cache) >= self.max_local_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.local_cache))
            del self.local_cache[oldest_key]
            self.cache_stats["evictions"] += 1
        
        self.local_cache[cache_key] = result
        self.cache_stats["sets"] += 1
        
        # Store in Redis cache
        if self.redis:
            try:
                await self.redis.setex(
                    f"db_cache:{cache_key}",
                    ttl,
                    json.dumps(result, default=str)
                )
            except Exception as e:
                logger.error(f"Redis cache set failed: {e}")
    
    async def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        # Clear local cache entries matching pattern
        keys_to_remove = [
            key for key in self.local_cache.keys()
            if pattern in key
        ]
        
        for key in keys_to_remove:
            del self.local_cache[key]
            self.cache_stats["evictions"] += 1
        
        # Clear Redis cache entries matching pattern
        if self.redis:
            try:
                keys = await self.redis.keys(f"db_cache:*{pattern}*")
                if keys:
                    await self.redis.delete(*keys)
            except Exception as e:
                logger.error(f"Redis cache invalidation failed: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.cache_stats["hits"],
            "misses": self.cache_stats["misses"],
            "sets": self.cache_stats["sets"],
            "evictions": self.cache_stats["evictions"],
            "hit_rate": hit_rate,
            "local_cache_size": len(self.local_cache)
        }

class DatabaseOptimizer:
    """
    Main database optimization manager
    """
    
    def __init__(self, config: DatabaseConfig, redis_client: Optional[aioredis.Redis] = None):
        self.config = config
        self.redis = redis_client
        self.query_optimizer = QueryOptimizer()
        self.connection_pool_manager = ConnectionPoolManager(config)
        self.sharding = DatabaseSharding(config)
        self.cache = DatabaseCache(redis_client)
        self.optimization_active = False
    
    async def initialize(self):
        """Initialize database optimizer"""
        try:
            # Create main connection pool
            await self.connection_pool_manager.create_pool("default")
            
            # Create shard pools if sharding is enabled
            if self.sharding.sharding_enabled:
                for shard_name in self.sharding.shards:
                    await self.sharding.shards[shard_name].create_pool()
            
            self.optimization_active = True
            logger.info("Database optimizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database optimizer: {e}")
            raise
    
    async def execute_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        use_cache: bool = True,
        shard_key: str = None,
        pool_name: str = "default"
    ) -> Any:
        """Execute optimized database query"""
        try:
            params = params or {}
            start_time = time.time()
            
            # Check cache first
            if use_cache:
                cached_result = await self.cache.get(query, params)
                if cached_result is not None:
                    logger.debug("Query result served from cache")
                    return cached_result
            
            # Optimize query
            optimized_query, optimization_info = await self.query_optimizer.optimize_query(query, params)
            
            # Execute query
            if shard_key and self.sharding.sharding_enabled:
                result = await self.sharding.execute_on_shard(optimized_query, params, shard_key)
            else:
                async with self.connection_pool_manager.get_connection_context(pool_name) as connection:
                    if self.config.database_type == DatabaseType.POSTGRESQL:
                        result = await connection.fetch(optimized_query, *params.values())
                    elif self.config.database_type == DatabaseType.MYSQL:
                        cursor = await connection.cursor()
                        await cursor.execute(optimized_query, params)
                        result = await cursor.fetchall()
            
            execution_time = time.time() - start_time
            
            # Record metrics
            query_id = self.query_optimizer._extract_query_pattern(query)
            metrics = QueryMetrics(
                query_id=query_id,
                query_type=self._get_query_type(query),
                execution_time=execution_time,
                rows_affected=len(result) if isinstance(result, list) else 1,
                cache_hit=False,
                timestamp=time.time(),
                connection_id=pool_name,
                shard_id=shard_key
            )
            self.query_optimizer.record_query_metrics(metrics)
            
            # Cache result if appropriate
            if use_cache and self._should_cache_query(query):
                await self.cache.set(query, params, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _get_query_type(self, query: str) -> QueryType:
        """Determine query type"""
        query_upper = query.upper().strip()
        
        if query_upper.startswith("SELECT"):
            return QueryType.SELECT
        elif query_upper.startswith("INSERT"):
            return QueryType.INSERT
        elif query_upper.startswith("UPDATE"):
            return QueryType.UPDATE
        elif query_upper.startswith("DELETE"):
            return QueryType.DELETE
        elif query_upper.startswith("CREATE"):
            return QueryType.CREATE
        elif query_upper.startswith("DROP"):
            return QueryType.DROP
        elif query_upper.startswith("ALTER"):
            return QueryType.ALTER
        else:
            return QueryType.SELECT
    
    def _should_cache_query(self, query: str) -> bool:
        """Determine if query should be cached"""
        query_upper = query.upper().strip()
        
        # Cache SELECT queries
        if query_upper.startswith("SELECT"):
            return True
        
        # Don't cache write operations
        if any(query_upper.startswith(op) for op in ["INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]):
            return False
        
        return False
    
    async def get_optimization_stats(self) -> Dict[str, Any]:
        """Get database optimization statistics"""
        return {
            "optimization_active": self.optimization_active,
            "query_analytics": self.query_optimizer.get_query_analytics(),
            "cache_stats": self.cache.get_cache_stats(),
            "pool_stats": {
                pool_name: self.connection_pool_manager.get_pool_stats(pool_name)
                for pool_name in self.connection_pool_manager.pools.keys()
            },
            "sharding_enabled": self.sharding.sharding_enabled,
            "shard_count": self.config.shard_count if self.sharding.sharding_enabled else 0
        }
    
    async def close(self):
        """Close database optimizer"""
        try:
            await self.connection_pool_manager.close_all_pools()
            await self.sharding.close_all_shards()
            self.optimization_active = False
            logger.info("Database optimizer closed")
        except Exception as e:
            logger.error(f"Failed to close database optimizer: {e}")

# Global database optimizer
database_optimizer: Optional[DatabaseOptimizer] = None

async def get_database_optimizer() -> DatabaseOptimizer:
    """Get global database optimizer instance"""
    global database_optimizer
    if not database_optimizer:
        # Create default configuration
        config = DatabaseConfig(
            database_type=DatabaseType.POSTGRESQL,
            host="localhost",
            port=5432,
            database="microservices",
            username="postgres",
            password="password"
        )
        database_optimizer = DatabaseOptimizer(config)
        await database_optimizer.initialize()
    return database_optimizer

# Decorator for database query optimization
def optimized_query(use_cache: bool = True, shard_key: str = None):
    """Decorator for optimized database queries"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            optimizer = await get_database_optimizer()
            
            # Extract query and params from function arguments
            # This is a simplified implementation
            query = kwargs.get("query", "")
            params = kwargs.get("params", {})
            
            if query:
                return await optimizer.execute_query(
                    query=query,
                    params=params,
                    use_cache=use_cache,
                    shard_key=shard_key
                )
            else:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, just call the original function
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator






























