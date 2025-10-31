from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, TypeVar, Generic, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import statistics
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy import text, MetaData, select, update, delete, insert
from sqlalchemy.exc import SQLAlchemyError, OperationalError, IntegrityError
import asyncpg
import redis.asyncio as redis
from .error_system import DatabaseError, ValidationError, TimeoutError
        import psutil
from typing import Any, List, Dict, Optional
"""
Optimized Database Manager - SQLAlchemy 2.0
==========================================

Production-grade database management with SQLAlchemy 2.0 best practices:
- Async operations with proper session management
- Connection pooling optimization
- Performance monitoring and metrics
- Comprehensive error handling
- Query optimization and caching
- Health monitoring and diagnostics
"""




# Type variables for generic operations
T = TypeVar('T')
Base = declarative_base()


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    pool_recycle: int = 3600
    pool_pre_ping: bool = True
    echo: bool = False
    json_serializer: callable = json.dumps
    json_deserializer: callable = json.loads


@dataclass
class QueryMetrics:
    """Metrics for database query performance"""
    query_name: str
    query_sql: str
    execution_time: float
    memory_usage: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """Database health status"""
    is_healthy: bool
    connection_count: int
    pool_size: int
    avg_query_time: float
    error_rate: float
    last_check: datetime
    details: Dict[str, Any] = field(default_factory=dict)


class QueryCache:
    """Simple query result cache"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        
    """__init__ function."""
self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, datetime]] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if valid"""
        if key in self.cache:
            result, timestamp = self.cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                return result
            else:
                del self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cache entry"""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (value, datetime.now())
    
    def clear(self) -> Any:
        """Clear all cache entries"""
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "ttl": self.ttl
        }


class OptimizedDatabaseManager:
    """
    Optimized database manager with SQLAlchemy 2.0 best practices
    """
    
    def __init__(self, config: DatabaseConfig):
        
    """__init__ function."""
self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Database components
        self.engine = None
        self.session_factory = None
        self.metadata = MetaData()
        
        # Performance monitoring
        self.query_metrics: List[QueryMetrics] = []
        self.query_cache = QueryCache()
        self.performance_monitor = PerformanceMonitor()
        
        # Health monitoring
        self.health_status = HealthStatus(
            is_healthy=False,
            connection_count=0,
            pool_size=0,
            avg_query_time=0.0,
            error_rate=0.0,
            last_check=datetime.now()
        )
        
        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.failed_queries = 0
        self.start_time = datetime.now()
    
    async def initialize(self) -> Any:
        """Initialize database connections and components"""
        self.logger.info("Initializing Optimized Database Manager...")
        
        try:
            # Create async engine with optimized settings
            self.engine = create_async_engine(
                self.config.url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=self.config.pool_pre_ping,
                echo=self.config.echo,
                future=True,
                json_serializer=self.config.json_serializer,
                json_deserializer=self.config.json_deserializer
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
                autocommit=False
            )
            
            # Test connection
            await self._test_connection()
            
            # Initialize health monitoring
            await self._update_health_status()
            
            self.logger.info("Database Manager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Database Manager: {e}")
            raise DatabaseError("initialization", reason=str(e))
    
    async def cleanup(self) -> Any:
        """Cleanup database connections and resources"""
        self.logger.info("Cleaning up Database Manager...")
        
        try:
            # Close session factory
            if self.session_factory:
                await self.session_factory.close_all()
            
            # Close engine
            if self.engine:
                await self.engine.dispose()
            
            # Clear cache
            self.query_cache.clear()
            
            self.logger.info("Database Manager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
    
    @asynccontextmanager
    async def get_session(self) -> Optional[Dict[str, Any]]:
        """Get database session with automatic cleanup and monitoring"""
        session = None
        start_time = time.time()
        
        try:
            session = self.session_factory()
            self.health_status.connection_count += 1
            
            yield session
            
            # Auto-commit successful transactions
            await session.commit()
            self.successful_queries += 1
            
        except Exception as e:
            if session:
                await session.rollback()
            
            self.failed_queries += 1
            self.logger.error(f"Database session error: {e}")
            
            # Raise appropriate error
            if isinstance(e, IntegrityError):
                raise ValidationError("data_integrity", reason=str(e))
            elif isinstance(e, OperationalError):
                raise DatabaseError("operation", reason=str(e))
            else:
                raise DatabaseError("session", reason=str(e))
            
        finally:
            if session:
                await session.close()
                self.health_status.connection_count -= 1
            
            # Update metrics
            query_time = time.time() - start_time
            self.performance_monitor.record_query_time(query_time)
    
    @asynccontextmanager
    async def monitor_query(self, query_name: str, query_sql: str = ""):
        """Context manager to monitor query performance"""
        start_time = time.time()
        memory_before = self.performance_monitor.get_memory_usage()
        success = False
        error_message = None
        
        try:
            yield
            success = True
        except Exception as e:
            error_message = str(e)
            raise
        finally:
            execution_time = time.time() - start_time
            memory_after = self.performance_monitor.get_memory_usage()
            memory_delta = memory_after - memory_before
            
            metric = QueryMetrics(
                query_name=query_name,
                query_sql=query_sql,
                execution_time=execution_time,
                memory_usage=memory_delta,
                success=success,
                error_message=error_message
            )
            
            self.query_metrics.append(metric)
            self.total_queries += 1
            
            # Log slow queries
            if execution_time > 1.0:  # 1 second threshold
                self.logger.warning(f"Slow query detected: {query_name} took {execution_time:.2f}s")
    
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None, 
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """Execute raw SQL query with optimization and caching"""
        
        # Check cache for read-only queries
        if use_cache and query.strip().upper().startswith('SELECT'):
            cache_key = self._generate_cache_key(query, params)
            cached_result = self.query_cache.get(cache_key)
            if cached_result is not None:
                self.logger.debug("Query cache hit")
                return cached_result
        
        async with self.monitor_query("raw_query", query):
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                
                # Convert to list of dictionaries
                rows = []
                for row in result:
                    rows.append(dict(row._mapping))
                
                # Cache result for read-only queries
                if use_cache and query.strip().upper().startswith('SELECT'):
                    cache_key = self._generate_cache_key(query, params)
                    self.query_cache.set(cache_key, rows)
                
                return rows
    
    async def execute_transaction(self, queries: List[Dict[str, Any]]) -> bool:
        """Execute multiple queries in a transaction"""
        
        async with self.monitor_query("transaction", f"Transaction with {len(queries)} queries"):
            async with self.get_session() as session:
                async with session.begin():
                    for query_data in queries:
                        query = query_data['query']
                        params = query_data.get('params', {})
                        
                        await session.execute(text(query), params)
                
                return True
    
    async def bulk_insert(self, table: str, data: List[Dict[str, Any]]) -> int:
        """Bulk insert data with optimization"""
        
        if not data:
            return 0
        
        async with self.monitor_query("bulk_insert", f"Bulk insert into {table}"):
            async with self.get_session() as session:
                # Use bulk insert for better performance
                stmt = insert(table).values(data)
                result = await session.execute(stmt)
                await session.commit()
                
                return len(data)
    
    async def bulk_update(self, table: str, data: List[Dict[str, Any]], 
                         where_column: str, where_values: List[Any]) -> int:
        """Bulk update data with optimization"""
        
        if not data or len(data) != len(where_values):
            raise ValidationError("bulk_update", reason="Data and where values must have same length")
        
        async with self.monitor_query("bulk_update", f"Bulk update {table}"):
            async with self.get_session() as session:
                updated_count = 0
                
                for i, (row_data, where_value) in enumerate(zip(data, where_values)):
                    stmt = update(table).where(
                        getattr(table, where_column) == where_value
                    ).values(row_data)
                    
                    result = await session.execute(stmt)
                    updated_count += result.rowcount
                
                await session.commit()
                return updated_count
    
    async def health_check(self) -> HealthStatus:
        """Perform comprehensive health check"""
        
        try:
            # Test basic connectivity
            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))
            
            # Update health status
            await self._update_health_status()
            
            # Check for issues
            if self.health_status.error_rate > 0.1:  # 10% error rate
                self.health_status.is_healthy = False
                self.logger.warning(f"High error rate detected: {self.health_status.error_rate:.1%}")
            
            if self.health_status.avg_query_time > 2.0:  # 2 second average
                self.health_status.is_healthy = False
                self.logger.warning(f"High average query time: {self.health_status.avg_query_time:.2f}s")
            
            self.health_status.last_check = datetime.now()
            
        except Exception as e:
            self.health_status.is_healthy = False
            self.logger.error(f"Health check failed: {e}")
        
        return self.health_status
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        # Calculate statistics
        execution_times = [m.execution_time for m in self.query_metrics if m.success]
        memory_usage = [m.memory_usage for m in self.query_metrics if m.success]
        
        metrics = {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "failed_queries": self.failed_queries,
            "success_rate": self.successful_queries / self.total_queries if self.total_queries > 0 else 0,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "query_performance": {
                "avg_execution_time": statistics.mean(execution_times) if execution_times else 0,
                "median_execution_time": statistics.median(execution_times) if execution_times else 0,
                "max_execution_time": max(execution_times) if execution_times else 0,
                "min_execution_time": min(execution_times) if execution_times else 0,
            },
            "memory_usage": {
                "avg_memory_per_query": statistics.mean(memory_usage) if memory_usage else 0,
                "max_memory_spike": max(memory_usage) if memory_usage else 0,
                "current_memory": self.performance_monitor.get_memory_usage(),
            },
            "cache_stats": self.query_cache.get_stats(),
            "slow_queries": len([m for m in self.query_metrics if m.execution_time > 1.0]),
            "health_status": self.health_status.__dict__
        }
        
        return metrics
    
    async def optimize_table(self, table_name: str) -> Dict[str, Any]:
        """Optimize database table"""
        
        async with self.monitor_query("table_optimization", f"Optimize table {table_name}"):
            async with self.get_session() as session:
                # Analyze table
                await session.execute(text(f"ANALYZE {table_name}"))
                
                # Get table statistics
                result = await session.execute(text(f"""
                    SELECT 
                        schemaname,
                        tablename,
                        attname,
                        n_distinct,
                        correlation
                    FROM pg_stats 
                    WHERE tablename = '{table_name}'
                """))
                
                stats = [dict(row._mapping) for row in result]
                
                return {
                    "table_name": table_name,
                    "optimization_status": "completed",
                    "statistics": stats,
                    "recommendations": self._generate_table_recommendations(stats)
                }
    
    def _generate_cache_key(self, query: str, params: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query,
            "params": params or {}
        }
        return json.dumps(key_data, sort_keys=True)
    
    async def _test_connection(self) -> Any:
        """Test database connection"""
        async with self.get_session() as session:
            await session.execute(text("SELECT 1"))
    
    async def _update_health_status(self) -> Any:
        """Update health status metrics"""
        if self.total_queries > 0:
            self.health_status.error_rate = self.failed_queries / self.total_queries
            self.health_status.avg_query_time = self.performance_monitor.get_average_query_time()
        
        # Get pool information if available
        if self.engine:
            pool = self.engine.pool
            self.health_status.pool_size = pool.size()
            self.health_status.connection_count = pool.checkedout()
    
    def _generate_table_recommendations(self, stats: List[Dict[str, Any]]) -> List[str]:
        """Generate table optimization recommendations"""
        recommendations = []
        
        for stat in stats:
            if stat.get('n_distinct', 0) < 10:
                recommendations.append(f"Consider index on {stat['attname']} (low cardinality)")
            
            if stat.get('correlation', 0) > 0.8:
                recommendations.append(f"Consider clustering on {stat['attname']} (high correlation)")
        
        return recommendations


class PerformanceMonitor:
    """Performance monitoring and metrics collection"""
    
    def __init__(self) -> Any:
        self.query_times = []
        self.memory_usage = []
        self.max_history = 1000
    
    def record_query_time(self, query_time: float):
        """Record query execution time"""
        self.query_times.append(query_time)
        
        # Keep only recent history
        if len(self.query_times) > self.max_history:
            self.query_times.pop(0)
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_average_query_time(self) -> float:
        """Get average query execution time"""
        return statistics.mean(self.query_times) if self.query_times else 0.0
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.query_times:
            return {}
        
        return {
            "total_queries": len(self.query_times),
            "avg_query_time": statistics.mean(self.query_times),
            "median_query_time": statistics.median(self.query_times),
            "max_query_time": max(self.query_times),
            "min_query_time": min(self.query_times),
            "slow_queries": len([t for t in self.query_times if t > 1.0]),
            "current_memory": self.get_memory_usage()
        }


# Example usage and factory functions
async def create_database_manager(config: DatabaseConfig) -> OptimizedDatabaseManager:
    """Factory function to create and initialize database manager"""
    manager = OptimizedDatabaseManager(config)
    await manager.initialize()
    return manager


def create_default_config(database_url: str) -> DatabaseConfig:
    """Create default database configuration"""
    return DatabaseConfig(
        url=database_url,
        pool_size=20,
        max_overflow=30,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=False
    )


# Example usage
async def example_usage():
    """Example usage of the optimized database manager"""
    
    # Create configuration
    config = create_default_config("postgresql+asyncpg://user:pass@localhost/db")
    
    # Create and initialize manager
    db_manager = await create_database_manager(config)
    
    try:
        # Execute queries
        users = await db_manager.execute_query("SELECT * FROM users LIMIT 10")
        
        # Bulk operations
        data = [{"name": f"User{i}", "email": f"user{i}@example.com"} for i in range(100)]
        await db_manager.bulk_insert("users", data)
        
        # Health check
        health = await db_manager.health_check()
        print(f"Database health: {health.is_healthy}")
        
        # Performance metrics
        metrics = await db_manager.get_performance_metrics()
        print(f"Success rate: {metrics['success_rate']:.1%}")
        
    finally:
        await db_manager.cleanup()


if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage()) 