"""
Performance monitoring and optimization service
"""

import asyncio
import time
import psutil
import gc
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, text
import redis.asyncio as redis

from ..config.settings import get_settings
from ..core.exceptions import DatabaseError


class PerformanceService:
    """Service for performance monitoring and optimization."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.settings = get_settings()
        self.redis_client = None
        self.performance_metrics = {}
        
        # Initialize Redis connection
        self._initialize_redis()
    
    def _initialize_redis(self):
        """Initialize Redis connection for caching."""
        try:
            self.redis_client = redis.from_url(self.settings.redis_url)
        except Exception as e:
            print(f"Warning: Could not connect to Redis: {e}")
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # Network metrics
            network_io = psutil.net_io_counters()
            
            # Database metrics
            db_metrics = await self._get_database_metrics()
            
            # Redis metrics
            redis_metrics = await self._get_redis_metrics()
            
            # Application metrics
            app_metrics = await self._get_application_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "cpu": {
                        "usage_percent": cpu_percent,
                        "count": cpu_count,
                        "frequency_mhz": cpu_freq.current if cpu_freq else None,
                        "load_average": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                    },
                    "memory": {
                        "total_gb": round(memory.total / (1024**3), 2),
                        "available_gb": round(memory.available / (1024**3), 2),
                        "used_gb": round(memory.used / (1024**3), 2),
                        "usage_percent": memory.percent,
                        "swap_total_gb": round(swap.total / (1024**3), 2),
                        "swap_used_gb": round(swap.used / (1024**3), 2),
                        "swap_usage_percent": swap.percent
                    },
                    "disk": {
                        "total_gb": round(disk.total / (1024**3), 2),
                        "used_gb": round(disk.used / (1024**3), 2),
                        "free_gb": round(disk.free / (1024**3), 2),
                        "usage_percent": round((disk.used / disk.total) * 100, 2),
                        "read_bytes": disk_io.read_bytes if disk_io else 0,
                        "write_bytes": disk_io.write_bytes if disk_io else 0
                    },
                    "network": {
                        "bytes_sent": network_io.bytes_sent,
                        "bytes_recv": network_io.bytes_recv,
                        "packets_sent": network_io.packets_sent,
                        "packets_recv": network_io.packets_recv
                    }
                },
                "database": db_metrics,
                "redis": redis_metrics,
                "application": app_metrics
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get system metrics: {str(e)}")
    
    async def get_database_performance(self) -> Dict[str, Any]:
        """Get database performance metrics."""
        try:
            # Connection pool metrics
            pool_metrics = await self._get_connection_pool_metrics()
            
            # Query performance metrics
            query_metrics = await self._get_query_performance_metrics()
            
            # Table size metrics
            table_metrics = await self._get_table_size_metrics()
            
            # Index usage metrics
            index_metrics = await self._get_index_usage_metrics()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "connection_pool": pool_metrics,
                "query_performance": query_metrics,
                "table_sizes": table_metrics,
                "index_usage": index_metrics
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get database performance: {str(e)}")
    
    async def get_cache_performance(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        try:
            if not self.redis_client:
                return {"error": "Redis not available"}
            
            # Redis info
            redis_info = await self.redis_client.info()
            
            # Memory usage
            memory_usage = redis_info.get('used_memory', 0)
            memory_peak = redis_info.get('used_memory_peak', 0)
            memory_fragmentation = redis_info.get('mem_fragmentation_ratio', 0)
            
            # Key statistics
            total_keys = await self.redis_client.dbsize()
            
            # Hit rate
            hits = redis_info.get('keyspace_hits', 0)
            misses = redis_info.get('keyspace_misses', 0)
            hit_rate = hits / (hits + misses) * 100 if (hits + misses) > 0 else 0
            
            # Connection statistics
            connected_clients = redis_info.get('connected_clients', 0)
            total_commands_processed = redis_info.get('total_commands_processed', 0)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "memory": {
                    "used_bytes": memory_usage,
                    "peak_bytes": memory_peak,
                    "fragmentation_ratio": memory_fragmentation
                },
                "keys": {
                    "total": total_keys
                },
                "performance": {
                    "hit_rate_percent": round(hit_rate, 2),
                    "hits": hits,
                    "misses": misses
                },
                "connections": {
                    "connected_clients": connected_clients,
                    "total_commands_processed": total_commands_processed
                }
            }
            
        except Exception as e:
            raise DatabaseError(f"Failed to get cache performance: {str(e)}")
    
    async def optimize_database(self) -> Dict[str, Any]:
        """Perform database optimization tasks."""
        try:
            optimization_results = {
                "timestamp": datetime.now().isoformat(),
                "tasks_completed": [],
                "errors": []
            }
            
            # Analyze tables
            try:
                await self._analyze_tables()
                optimization_results["tasks_completed"].append("analyze_tables")
            except Exception as e:
                optimization_results["errors"].append(f"analyze_tables: {str(e)}")
            
            # Vacuum tables
            try:
                await self._vacuum_tables()
                optimization_results["tasks_completed"].append("vacuum_tables")
            except Exception as e:
                optimization_results["errors"].append(f"vacuum_tables: {str(e)}")
            
            # Update statistics
            try:
                await self._update_statistics()
                optimization_results["tasks_completed"].append("update_statistics")
            except Exception as e:
                optimization_results["errors"].append(f"update_statistics: {str(e)}")
            
            # Clean up old data
            try:
                cleaned_count = await self._cleanup_old_data()
                optimization_results["tasks_completed"].append(f"cleanup_old_data: {cleaned_count} records")
            except Exception as e:
                optimization_results["errors"].append(f"cleanup_old_data: {str(e)}")
            
            return optimization_results
            
        except Exception as e:
            raise DatabaseError(f"Failed to optimize database: {str(e)}")
    
    async def clear_cache(self, pattern: str = "*") -> Dict[str, Any]:
        """Clear cache entries matching pattern."""
        try:
            if not self.redis_client:
                return {"error": "Redis not available"}
            
            # Get keys matching pattern
            keys = await self.redis_client.keys(pattern)
            
            if keys:
                # Delete keys
                deleted_count = await self.redis_client.delete(*keys)
                
                return {
                    "timestamp": datetime.now().isoformat(),
                    "pattern": pattern,
                    "keys_found": len(keys),
                    "keys_deleted": deleted_count,
                    "success": True
                }
            else:
                return {
                    "timestamp": datetime.now().isoformat(),
                    "pattern": pattern,
                    "keys_found": 0,
                    "keys_deleted": 0,
                    "success": True
                }
                
        except Exception as e:
            raise DatabaseError(f"Failed to clear cache: {str(e)}")
    
    async def get_slow_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get slow query information."""
        try:
            # This would typically query pg_stat_statements or similar
            # For now, we'll return a mock structure
            slow_queries = []
            
            # In a real implementation, you would query the database for slow queries
            # Example query for PostgreSQL:
            # SELECT query, mean_time, calls, total_time 
            # FROM pg_stat_statements 
            # ORDER BY mean_time DESC 
            # LIMIT %s
            
            return slow_queries
            
        except Exception as e:
            raise DatabaseError(f"Failed to get slow queries: {str(e)}")
    
    async def _get_database_metrics(self) -> Dict[str, Any]:
        """Get database-specific metrics."""
        try:
            # Get database size
            db_size_query = text("SELECT pg_size_pretty(pg_database_size(current_database())) as size")
            db_size_result = await self.session.execute(db_size_query)
            db_size = db_size_result.scalar()
            
            # Get connection count
            connections_query = text("SELECT count(*) FROM pg_stat_activity")
            connections_result = await self.session.execute(connections_query)
            active_connections = connections_result.scalar()
            
            # Get table count
            tables_query = text("SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public'")
            tables_result = await self.session.execute(tables_query)
            table_count = tables_result.scalar()
            
            return {
                "database_size": db_size,
                "active_connections": active_connections,
                "table_count": table_count
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis-specific metrics."""
        try:
            if not self.redis_client:
                return {"error": "Redis not available"}
            
            # Get Redis info
            info = await self.redis_client.info()
            
            return {
                "version": info.get('redis_version', 'unknown'),
                "uptime_seconds": info.get('uptime_in_seconds', 0),
                "connected_clients": info.get('connected_clients', 0),
                "used_memory": info.get('used_memory', 0),
                "used_memory_peak": info.get('used_memory_peak', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics."""
        try:
            # Get current process info
            process = psutil.Process()
            
            return {
                "process_id": process.pid,
                "memory_usage_mb": round(process.memory_info().rss / (1024**2), 2),
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_connection_pool_metrics(self) -> Dict[str, Any]:
        """Get database connection pool metrics."""
        try:
            # This would typically access the connection pool directly
            # For now, we'll return basic metrics
            return {
                "pool_size": 10,  # This would be actual pool size
                "checked_out": 0,  # This would be actual checked out connections
                "overflow": 0,     # This would be actual overflow
                "checked_in": 10   # This would be actual checked in connections
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_query_performance_metrics(self) -> Dict[str, Any]:
        """Get query performance metrics."""
        try:
            # This would typically query pg_stat_statements or similar
            return {
                "total_queries": 0,
                "avg_query_time_ms": 0,
                "slow_queries": 0,
                "cache_hit_ratio": 0
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_table_size_metrics(self) -> Dict[str, Any]:
        """Get table size metrics."""
        try:
            # Get table sizes
            table_sizes_query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
                FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
                LIMIT 10
            """)
            
            table_sizes_result = await self.session.execute(table_sizes_query)
            table_sizes = table_sizes_result.fetchall()
            
            return {
                "tables": [
                    {"name": row.tablename, "size": row.size}
                    for row in table_sizes
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _get_index_usage_metrics(self) -> Dict[str, Any]:
        """Get index usage metrics."""
        try:
            # Get index usage statistics
            index_usage_query = text("""
                SELECT 
                    schemaname,
                    tablename,
                    indexname,
                    idx_scan,
                    idx_tup_read,
                    idx_tup_fetch
                FROM pg_stat_user_indexes
                ORDER BY idx_scan DESC
                LIMIT 10
            """)
            
            index_usage_result = await self.session.execute(index_usage_query)
            index_usage = index_usage_result.fetchall()
            
            return {
                "indexes": [
                    {
                        "table": row.tablename,
                        "index": row.indexname,
                        "scans": row.idx_scan,
                        "tuples_read": row.idx_tup_read,
                        "tuples_fetched": row.idx_tup_fetch
                    }
                    for row in index_usage
                ]
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_tables(self):
        """Analyze database tables for query optimization."""
        try:
            # Get all tables
            tables_query = text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            tables_result = await self.session.execute(tables_query)
            tables = tables_result.fetchall()
            
            # Analyze each table
            for table in tables:
                analyze_query = text(f"ANALYZE {table.tablename}")
                await self.session.execute(analyze_query)
            
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise e
    
    async def _vacuum_tables(self):
        """Vacuum database tables to reclaim space."""
        try:
            # Get all tables
            tables_query = text("""
                SELECT tablename 
                FROM pg_tables 
                WHERE schemaname = 'public'
            """)
            
            tables_result = await self.session.execute(tables_query)
            tables = tables_result.fetchall()
            
            # Vacuum each table
            for table in tables:
                vacuum_query = text(f"VACUUM {table.tablename}")
                await self.session.execute(vacuum_query)
            
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise e
    
    async def _update_statistics(self):
        """Update database statistics."""
        try:
            # Update statistics for all tables
            update_stats_query = text("UPDATE pg_stat_user_tables SET n_tup_ins = 0, n_tup_upd = 0, n_tup_del = 0")
            await self.session.execute(update_stats_query)
            await self.session.commit()
            
        except Exception as e:
            await self.session.rollback()
            raise e
    
    async def _cleanup_old_data(self) -> int:
        """Clean up old data to improve performance."""
        try:
            # Clean up old analytics data (older than 1 year)
            cutoff_date = datetime.now() - timedelta(days=365)
            
            # This would typically clean up old analytics, logs, etc.
            # For now, we'll return 0 as no cleanup was performed
            return 0
            
        except Exception as e:
            raise e






























