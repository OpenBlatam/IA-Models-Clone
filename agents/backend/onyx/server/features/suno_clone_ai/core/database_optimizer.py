"""
Advanced Database Optimizations

Optimizations for:
- Query optimization
- Connection pooling
- Query caching
- Batch operations
- Index optimization
- Transaction optimization
"""

import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache
import time
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """Advanced database optimizations."""
    
    @staticmethod
    def create_optimized_engine(
        database_url: str,
        pool_size: int = 20,
        max_overflow: int = 10,
        pool_timeout: int = 30,
        pool_recycle: int = 3600,
        echo: bool = False
    ):
        """
        Create optimized database engine.
        
        Args:
            database_url: Database connection URL
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool timeout in seconds
            pool_recycle: Connection recycle time
            echo: Echo SQL queries
            
        Returns:
            Optimized SQLAlchemy engine
        """
        engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            echo=echo,
            # Optimizations
            connect_args={
                "check_same_thread": False,  # SQLite
                "timeout": 20,  # Connection timeout
            } if "sqlite" in database_url else {},
            # Connection pool pre-ping
            pool_pre_ping=True,
        )
        
        # Add query optimization events
        DatabaseOptimizer._add_query_optimizations(engine)
        
        return engine
    
    @staticmethod
    def _add_query_optimizations(engine):
        """Add query optimization event listeners."""
        
        @event.listens_for(engine, "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log slow queries."""
            conn.info.setdefault('query_start_time', []).append(time.time())
        
        @event.listens_for(engine, "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            """Log query execution time."""
            total = time.time() - conn.info['query_start_time'].pop(-1)
            if total > 1.0:  # Log queries > 1 second
                logger.warning(f"Slow query ({total:.2f}s): {statement[:100]}")


class QueryCache:
    """Query result caching."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        Initialize query cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, tuple[Any, float]] = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def _make_key(self, query: str, params: tuple) -> str:
        """Create cache key from query."""
        import hashlib
        key_data = f"{query}:{params}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, query: str, params: tuple = ()) -> Optional[Any]:
        """Get cached query result."""
        key = self._make_key(query, params)
        
        if key not in self.cache:
            return None
        
        result, timestamp = self.cache[key]
        
        # Check TTL
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        return result
    
    def set(self, query: str, result: Any, params: tuple = ()) -> None:
        """Cache query result."""
        key = self._make_key(query, params)
        
        # Remove oldest if full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        self.cache[key] = (result, time.time())
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()


class BatchOperations:
    """Optimized batch database operations."""
    
    @staticmethod
    def bulk_insert(
        session: Session,
        model_class,
        records: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> None:
        """
        Bulk insert records efficiently.
        
        Args:
            session: Database session
            model_class: Model class
            records: List of record dictionaries
            batch_size: Batch size for insertion
        """
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            objects = [model_class(**record) for record in batch]
            session.bulk_save_objects(objects)
        
        session.commit()
    
    @staticmethod
    def bulk_update(
        session: Session,
        model_class,
        updates: List[Dict[str, Any]],
        batch_size: int = 1000
    ) -> None:
        """
        Bulk update records efficiently.
        
        Args:
            session: Database session
            model_class: Model class
            updates: List of update dictionaries
            batch_size: Batch size for updates
        """
        for i in range(0, len(updates), batch_size):
            batch = updates[i:i + batch_size]
            session.bulk_update_mappings(model_class, batch)
        
        session.commit()


class IndexOptimizer:
    """Database index optimization."""
    
    @staticmethod
    def suggest_indexes(queries: List[str]) -> List[Dict[str, Any]]:
        """
        Suggest indexes based on query patterns.
        
        Args:
            queries: List of SQL queries
            
        Returns:
            List of suggested indexes
        """
        # Simple heuristic: find WHERE clauses
        indexes = []
        
        for query in queries:
            query_lower = query.lower()
            if 'where' in query_lower:
                # Extract column names from WHERE clause
                where_start = query_lower.find('where')
                where_clause = query[where_start + 5:].split('order')[0]
                
                # Find column names (simplified)
                import re
                columns = re.findall(r'(\w+)\s*[=<>]', where_clause)
                
                if columns:
                    indexes.append({
                        'columns': columns,
                        'query': query[:100]
                    })
        
        return indexes


class TransactionOptimizer:
    """Transaction optimization."""
    
    @staticmethod
    @contextmanager
    def optimized_transaction(session: Session):
        """
        Optimized transaction context manager.
        
        Args:
            session: Database session
            
        Yields:
            Session with optimized transaction
        """
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
