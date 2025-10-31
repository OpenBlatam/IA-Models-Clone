from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import hashlib
import json
import time
import weakref
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
import structlog
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import QueuePool, StaticPool
from sqlalchemy import text, inspect, MetaData, Table
from sqlalchemy.sql import Select, Insert, Update, Delete
from sqlalchemy.dialects import postgresql, mysql, sqlite
import redis.asyncio as redis
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Advanced Database Optimizer for HeyGen AI FastAPI
Query optimization, caching, and intelligent database operations.
"""


logger = structlog.get_logger()

# =============================================================================
# Database Optimization Types
# =============================================================================

class QueryType(Enum):
    """Database query type enumeration."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    COUNT = "count"
    AGGREGATE = "aggregate"

class OptimizationLevel(Enum):
    """Database optimization level."""
    BASIC = "basic"
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"

@dataclass
class QueryMetrics:
    """Database query performance metrics."""
    query_hash: str
    query_type: QueryType
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    last_executed: Optional[datetime] = None
    error_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    def update_timing(self, duration_ms: float):
        """Update timing metrics."""
        self.execution_count += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.execution_count
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.last_executed = datetime.now(timezone.utc)

@dataclass
class IndexSuggestion:
    """Database index suggestion."""
    table_name: str
    columns: List[str]
    index_type: str = "btree"
    estimated_benefit: float = 0.0
    priority: int = 1  # 1=high, 2=medium, 3=low
    reason: str = ""

# =============================================================================
# Query Cache Manager
# =============================================================================

class QueryCacheManager:
    """Intelligent query result caching."""
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        memory_cache_size: int = 1000,
        default_ttl: int = 300
    ):
        
    """__init__ function."""
self.redis_client = redis_client
        self.memory_cache_size = memory_cache_size
        self.default_ttl = default_ttl
        
        # Memory cache (LRU)
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_access_order: List[str] = []
        
        # Cache statistics
        self.stats = {
            "memory_hits": 0,
            "memory_misses": 0,
            "redis_hits": 0,
            "redis_misses": 0,
            "total_requests": 0
        }
    
    def _generate_cache_key(self, query: str, params: Dict[str, Any]) -> str:
        """Generate cache key for query and parameters."""
        # Normalize query (remove extra whitespace, lowercase)
        normalized_query = " ".join(query.lower().split())
        
        # Create deterministic parameter string
        param_str = json.dumps(params, sort_keys=True, default=str)
        
        # Generate hash
        cache_data = f"{normalized_query}:{param_str}"
        return hashlib.sha256(cache_data.encode()).hexdigest()
    
    def _is_query_cacheable(self, query: str) -> bool:
        """Determine if query should be cached."""
        query_lower = query.lower().strip()
        
        # Only cache SELECT queries
        if not query_lower.startswith('select'):
            return False
        
        # Don't cache queries with non-deterministic functions
        non_deterministic = [
            'now()', 'current_timestamp', 'random()', 'uuid_generate',
            'current_date', 'current_time', 'localtime', 'localtimestamp'
        ]
        
        for func in non_deterministic:
            if func in query_lower:
                return False
        
        return True
    
    async def get_cached_result(
        self, 
        query: str, 
        params: Dict[str, Any]
    ) -> Optional[List[Dict[str, Any]]]:
        """Get cached query result."""
        if not self._is_query_cacheable(query):
            return None
        
        cache_key = self._generate_cache_key(query, params)
        self.stats["total_requests"] += 1
        
        # Try memory cache first
        if cache_key in self.memory_cache:
            cache_entry = self.memory_cache[cache_key]
            if cache_entry["expires_at"] > datetime.now(timezone.utc):
                # Update access order
                if cache_key in self.cache_access_order:
                    self.cache_access_order.remove(cache_key)
                self.cache_access_order.append(cache_key)
                
                self.stats["memory_hits"] += 1
                return cache_entry["data"]
            else:
                # Expired entry
                del self.memory_cache[cache_key]
                if cache_key in self.cache_access_order:
                    self.cache_access_order.remove(cache_key)
        
        # Try Redis cache
        if self.redis_client:
            try:
                cached_data = await self.redis_client.get(f"query_cache:{cache_key}")
                if cached_data:
                    result = json.loads(cached_data)
                    
                    # Store in memory cache for faster access
                    self._store_in_memory_cache(cache_key, result)
                    
                    self.stats["redis_hits"] += 1
                    return result
                else:
                    self.stats["redis_misses"] += 1
            except Exception as e:
                logger.error(f"Redis cache error: {e}")
        
        self.stats["memory_misses"] += 1
        return None
    
    async def store_result(
        self,
        query: str,
        params: Dict[str, Any],
        result: List[Dict[str, Any]],
        ttl: Optional[int] = None
    ):
        """Store query result in cache."""
        if not self._is_query_cacheable(query):
            return
        
        cache_key = self._generate_cache_key(query, params)
        ttl = ttl or self.default_ttl
        
        # Store in memory cache
        self._store_in_memory_cache(cache_key, result, ttl)
        
        # Store in Redis cache
        if self.redis_client:
            try:
                serialized_result = json.dumps(result, default=str)
                await self.redis_client.setex(
                    f"query_cache:{cache_key}",
                    ttl,
                    serialized_result
                )
            except Exception as e:
                logger.error(f"Redis cache store error: {e}")
    
    def _store_in_memory_cache(
        self,
        cache_key: str,
        result: List[Dict[str, Any]],
        ttl: int = None
    ):
        """Store result in memory cache with LRU eviction."""
        ttl = ttl or self.default_ttl
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
        
        # Add to cache
        self.memory_cache[cache_key] = {
            "data": result,
            "expires_at": expires_at
        }
        
        # Update access order
        if cache_key in self.cache_access_order:
            self.cache_access_order.remove(cache_key)
        self.cache_access_order.append(cache_key)
        
        # Evict old entries if needed
        while len(self.memory_cache) > self.memory_cache_size:
            oldest_key = self.cache_access_order.pop(0)
            if oldest_key in self.memory_cache:
                del self.memory_cache[oldest_key]
    
    async def invalidate_cache(self, pattern: str = None):
        """Invalidate cache entries."""
        if pattern:
            # Selective invalidation (for table-specific cache invalidation)
            keys_to_remove = []
            for key in self.memory_cache.keys():
                if pattern in key:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                if key in self.cache_access_order:
                    self.cache_access_order.remove(key)
            
            # Redis pattern-based deletion
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys(f"query_cache:*{pattern}*")
                    if keys:
                        await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis cache invalidation error: {e}")
        else:
            # Clear all cache
            self.memory_cache.clear()
            self.cache_access_order.clear()
            
            if self.redis_client:
                try:
                    keys = await self.redis_client.keys("query_cache:*")
                    if keys:
                        await self.redis_client.delete(*keys)
                except Exception as e:
                    logger.error(f"Redis cache clear error: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = self.stats["memory_hits"] + self.stats["redis_hits"]
        total_misses = self.stats["memory_misses"] + self.stats["redis_misses"]
        total_requests = self.stats["total_requests"]
        
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hit_rate_percent": hit_rate,
            "memory_hits": self.stats["memory_hits"],
            "memory_misses": self.stats["memory_misses"],
            "redis_hits": self.stats["redis_hits"],
            "redis_misses": self.stats["redis_misses"],
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "memory_cache_limit": self.memory_cache_size
        }

# =============================================================================
# Query Analyzer
# =============================================================================

class QueryAnalyzer:
    """Analyze and optimize database queries."""
    
    def __init__(self) -> Any:
        self.query_metrics: Dict[str, QueryMetrics] = {}
        self.table_access_patterns: Dict[str, Dict[str, int]] = {}
        self.index_suggestions: List[IndexSuggestion] = []
        self.slow_query_threshold_ms = 1000  # 1 second
    
    def analyze_query(self, query: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze query for optimization opportunities."""
        query_hash = self._hash_query(query, params)
        query_type = self._detect_query_type(query)
        
        # Initialize metrics if new query
        if query_hash not in self.query_metrics:
            self.query_metrics[query_hash] = QueryMetrics(
                query_hash=query_hash,
                query_type=query_type
            )
        
        analysis = {
            "query_hash": query_hash,
            "query_type": query_type.value,
            "optimization_suggestions": [],
            "estimated_cost": self._estimate_query_cost(query),
            "tables_accessed": self._extract_tables(query),
            "conditions": self._extract_conditions(query)
        }
        
        # Analyze for optimization opportunities
        suggestions = self._analyze_for_optimizations(query, analysis)
        analysis["optimization_suggestions"] = suggestions
        
        return analysis
    
    def record_query_execution(
        self,
        query: str,
        params: Dict[str, Any],
        duration_ms: float,
        success: bool = True
    ):
        """Record query execution metrics."""
        query_hash = self._hash_query(query, params)
        
        if query_hash in self.query_metrics:
            metrics = self.query_metrics[query_hash]
            metrics.update_timing(duration_ms)
            
            if not success:
                metrics.error_count += 1
            
            # Check for slow queries
            if duration_ms > self.slow_query_threshold_ms:
                logger.warning(
                    f"Slow query detected: {duration_ms:.2f}ms",
                    query_hash=query_hash,
                    duration_ms=duration_ms
                )
                self._analyze_slow_query(query, duration_ms)
    
    def _hash_query(self, query: str, params: Dict[str, Any] = None) -> str:
        """Generate hash for query identification."""
        # Normalize query
        normalized = " ".join(query.lower().split())
        
        # Don't include actual parameter values in hash, just structure
        param_keys = sorted(params.keys()) if params else []
        query_signature = f"{normalized}:params:{':'.join(param_keys)}"
        
        return hashlib.md5(query_signature.encode()).hexdigest()[:16]
    
    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of database query."""
        query_lower = query.lower().strip()
        
        if query_lower.startswith('select'):
            if 'count(' in query_lower:
                return QueryType.COUNT
            elif any(func in query_lower for func in ['sum(', 'avg(', 'max(', 'min(']):
                return QueryType.AGGREGATE
            else:
                return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        else:
            return QueryType.SELECT  # Default
    
    def _estimate_query_cost(self, query: str) -> int:
        """Estimate query execution cost (simplified)."""
        cost = 1
        query_lower = query.lower()
        
        # Add cost for JOINs
        cost += query_lower.count('join') * 5
        
        # Add cost for subqueries
        cost += query_lower.count('select') - 1
        
        # Add cost for ORDER BY
        if 'order by' in query_lower:
            cost += 3
        
        # Add cost for GROUP BY
        if 'group by' in query_lower:
            cost += 4
        
        # Add cost for LIKE operations
        cost += query_lower.count('like') * 2
        
        return cost
    
    def _extract_tables(self, query: str) -> List[str]:
        """Extract table names from query."""
        # Simplified table extraction
        tables = []
        words = query.lower().split()
        
        from_index = -1
        join_indices = []
        
        for i, word in enumerate(words):
            if word == 'from':
                from_index = i
            elif word in ['join', 'inner', 'left', 'right', 'full']:
                if i + 1 < len(words) and words[i + 1] == 'join':
                    join_indices.append(i + 1)
                elif word == 'join':
                    join_indices.append(i)
        
        # Extract table after FROM
        if from_index != -1 and from_index + 1 < len(words):
            table = words[from_index + 1].strip(',').strip()
            if table not in ['(', 'select']:  # Skip subqueries
                tables.append(table)
        
        # Extract tables after JOINs
        for join_index in join_indices:
            if join_index + 1 < len(words):
                table = words[join_index + 1].strip(',').strip()
                if table not in ['(', 'select']:  # Skip subqueries
                    tables.append(table)
        
        return list(set(tables))  # Remove duplicates
    
    def _extract_conditions(self, query: str) -> List[str]:
        """Extract WHERE and JOIN conditions."""
        conditions = []
        query_lower = query.lower()
        
        # Simple condition extraction
        if 'where' in query_lower:
            where_part = query_lower.split('where')[1].split('order by')[0].split('group by')[0]
            # Split by AND/OR and clean up
            raw_conditions = where_part.replace(' and ', '|').replace(' or ', '|').split('|')
            conditions.extend([cond.strip() for cond in raw_conditions if cond.strip()])
        
        return conditions
    
    def _analyze_for_optimizations(self, query: str, analysis: Dict[str, Any]) -> List[str]:
        """Analyze query for optimization opportunities."""
        suggestions = []
        query_lower = query.lower()
        
        # Check for missing LIMIT on SELECT queries
        if analysis["query_type"] == "select" and "limit" not in query_lower:
            suggestions.append("Consider adding LIMIT clause to prevent large result sets")
        
        # Check for SELECT *
        if "select *" in query_lower:
            suggestions.append("Avoid SELECT *, specify only needed columns")
        
        # Check for LIKE patterns
        if "like" in query_lower:
            suggestions.append("Consider using full-text search or more specific conditions instead of LIKE")
        
        # Check for OR conditions
        if " or " in query_lower:
            suggestions.append("OR conditions can be slow, consider using UNION or restructuring")
        
        # Check for functions in WHERE clause
        where_functions = ["upper(", "lower(", "substring(", "date("]
        for func in where_functions:
            if func in query_lower:
                suggestions.append(f"Avoid using {func} in WHERE clause, consider indexed computed columns")
        
        # Suggest indexes for WHERE conditions
        for condition in analysis["conditions"]:
            if "=" in condition:
                column = condition.split("=")[0].strip()
                suggestions.append(f"Consider adding index on column: {column}")
        
        return suggestions
    
    def _analyze_slow_query(self, query: str, duration_ms: float):
        """Analyze slow query for optimization suggestions."""
        tables = self._extract_tables(query)
        conditions = self._extract_conditions(query)
        
        # Generate index suggestions for slow queries
        for condition in conditions:
            if "=" in condition:
                column = condition.split("=")[0].strip().split(".")[-1]  # Remove table prefix
                for table in tables:
                    suggestion = IndexSuggestion(
                        table_name=table,
                        columns=[column],
                        estimated_benefit=duration_ms / 1000,  # Rough benefit estimate
                        priority=1 if duration_ms > 5000 else 2,  # High priority if > 5s
                        reason=f"Slow query performance: {duration_ms:.2f}ms"
                    )
                    
                    # Avoid duplicate suggestions
                    if not any(
                        s.table_name == suggestion.table_name and s.columns == suggestion.columns
                        for s in self.index_suggestions
                    ):
                        self.index_suggestions.append(suggestion)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get query performance summary."""
        if not self.query_metrics:
            return {"message": "No query metrics available"}
        
        total_queries = sum(m.execution_count for m in self.query_metrics.values())
        total_time = sum(m.total_time_ms for m in self.query_metrics.values())
        avg_time = total_time / total_queries if total_queries > 0 else 0
        
        slow_queries = [
            m for m in self.query_metrics.values()
            if m.avg_time_ms > self.slow_query_threshold_ms
        ]
        
        top_slow_queries = sorted(
            slow_queries,
            key=lambda x: x.avg_time_ms,
            reverse=True
        )[:10]
        
        return {
            "total_queries": total_queries,
            "total_execution_time_ms": total_time,
            "average_execution_time_ms": avg_time,
            "slow_queries_count": len(slow_queries),
            "top_slow_queries": [
                {
                    "query_hash": q.query_hash,
                    "avg_time_ms": q.avg_time_ms,
                    "execution_count": q.execution_count,
                    "query_type": q.query_type.value
                }
                for q in top_slow_queries
            ],
            "index_suggestions": [
                {
                    "table": s.table_name,
                    "columns": s.columns,
                    "priority": s.priority,
                    "reason": s.reason,
                    "estimated_benefit": s.estimated_benefit
                }
                for s in sorted(self.index_suggestions, key=lambda x: x.priority)
            ]
        }

# =============================================================================
# Advanced Database Optimizer
# =============================================================================

class AdvancedDatabaseOptimizer:
    """Advanced database optimization manager."""
    
    def __init__(
        self,
        database_url: str,
        redis_client: Optional[redis.Redis] = None,
        optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
    ):
        
    """__init__ function."""
self.database_url = database_url
        self.optimization_level = optimization_level
        
        # Initialize components
        self.cache_manager = QueryCacheManager(redis_client)
        self.query_analyzer = QueryAnalyzer()
        
        # Database engine and session
        self.engine = None
        self.async_session = None
        
        # Connection pool settings based on optimization level
        pool_settings = self._get_pool_settings()
        
        # Create optimized engine
        self.engine = create_async_engine(
            database_url,
            **pool_settings,
            echo=False,
            future=True
        )
        
        # Create session factory
        self.async_session = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    def _get_pool_settings(self) -> Dict[str, Any]:
        """Get connection pool settings based on optimization level."""
        if self.optimization_level == OptimizationLevel.BASIC:
            return {
                "poolclass": QueuePool,
                "pool_size": 10,
                "max_overflow": 20,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True
            }
        elif self.optimization_level == OptimizationLevel.STANDARD:
            return {
                "poolclass": QueuePool,
                "pool_size": 20,
                "max_overflow": 40,
                "pool_timeout": 30,
                "pool_recycle": 3600,
                "pool_pre_ping": True,
                "pool_reset_on_return": "commit"
            }
        else:  # AGGRESSIVE
            return {
                "poolclass": QueuePool,
                "pool_size": 50,
                "max_overflow": 100,
                "pool_timeout": 60,
                "pool_recycle": 1800,
                "pool_pre_ping": True,
                "pool_reset_on_return": "commit"
            }
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get optimized database session."""
        async with self.async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def execute_optimized_query(
        self,
        query: str,
        params: Dict[str, Any] = None,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """Execute query with optimization and caching."""
        params = params or {}
        start_time = time.time()
        
        # Analyze query
        analysis = self.query_analyzer.analyze_query(query, params)
        
        # Try cache first
        if use_cache:
            cached_result = await self.cache_manager.get_cached_result(query, params)
            if cached_result is not None:
                self.query_analyzer.query_metrics[analysis["query_hash"]].cache_hits += 1
                return cached_result
            else:
                self.query_analyzer.query_metrics[analysis["query_hash"]].cache_misses += 1
        
        # Execute query
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params)
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                # Record execution metrics
                duration_ms = (time.time() - start_time) * 1000
                self.query_analyzer.record_query_execution(query, params, duration_ms, True)
                
                # Cache result if appropriate
                if use_cache and analysis["query_type"] == "select":
                    cache_ttl = self._determine_cache_ttl(analysis)
                    await self.cache_manager.store_result(query, params, rows, cache_ttl)
                
                return rows
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.query_analyzer.record_query_execution(query, params, duration_ms, False)
            logger.error(f"Query execution failed: {e}")
            raise
    
    def _determine_cache_ttl(self, analysis: Dict[str, Any]) -> int:
        """Determine appropriate cache TTL based on query analysis."""
        base_ttl = 300  # 5 minutes default
        
        # Longer TTL for aggregate queries
        if analysis["query_type"] == "aggregate":
            return base_ttl * 4  # 20 minutes
        
        # Shorter TTL for high-cost queries
        if analysis["estimated_cost"] > 10:
            return base_ttl // 2  # 2.5 minutes
        
        # Medium TTL for regular selects
        return base_ttl
    
    async def invalidate_table_cache(self, table_name: str):
        """Invalidate cache for specific table."""
        await self.cache_manager.invalidate_cache(table_name)
    
    async def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return {
            "cache_stats": self.cache_manager.get_cache_stats(),
            "query_performance": self.query_analyzer.get_performance_summary(),
            "optimization_level": self.optimization_level.value,
            "database_url_type": self.database_url.split("://")[0] if "://" in self.database_url else "unknown"
        }
    
    async def cleanup(self) -> Any:
        """Cleanup database connections."""
        if self.engine:
            await self.engine.dispose()

# =============================================================================
# Factory Function
# =============================================================================

async def create_database_optimizer(
    database_url: str,
    redis_client: Optional[redis.Redis] = None,
    optimization_level: OptimizationLevel = OptimizationLevel.STANDARD
) -> AdvancedDatabaseOptimizer:
    """Create and initialize database optimizer."""
    return AdvancedDatabaseOptimizer(
        database_url=database_url,
        redis_client=redis_client,
        optimization_level=optimization_level
    ) 