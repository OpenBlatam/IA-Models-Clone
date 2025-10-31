"""
Database Management System for Improved Video-OpusClip API

Comprehensive database management with:
- Async SQLAlchemy integration
- Connection pooling
- Migration management
- Health monitoring
- Performance optimization
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
import asyncio
import time
import structlog
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import (
    AsyncSession, AsyncEngine, create_async_engine, async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text, MetaData
from sqlalchemy.exc import SQLAlchemyError

from ..config import settings
from ..error_handling import DatabaseError, ResourceError

logger = structlog.get_logger("database")

# =============================================================================
# DATABASE ENGINE
# =============================================================================

class DatabaseManager:
    """Database manager with connection pooling and health monitoring."""
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker] = None
        self._stats = {
            'connections_created': 0,
            'connections_closed': 0,
            'queries_executed': 0,
            'query_errors': 0,
            'total_query_time': 0.0,
            'average_query_time': 0.0
        }
    
    async def initialize(self) -> None:
        """Initialize database engine and session factory."""
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                settings.get_database_url(),
                pool_size=settings.database_pool_size,
                max_overflow=settings.database_max_overflow,
                pool_timeout=settings.database_pool_timeout,
                pool_recycle=settings.database_pool_recycle,
                pool_pre_ping=True,  # Verify connections before use
                echo=settings.database_echo,
                echo_pool=settings.database_echo_pool,
                poolclass=QueuePool if not settings.is_testing else NullPool
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            await self._test_connection()
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            raise DatabaseError(f"Database initialization failed: {str(e)}")
    
    async def close(self) -> None:
        """Close database engine and cleanup resources."""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database manager closed")
    
    async def _test_connection(self) -> None:
        """Test database connection."""
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
        except Exception as e:
            logger.error("Database connection test failed", error=str(e))
            raise DatabaseError(f"Database connection test failed: {str(e)}")
    
    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """Get database session with automatic cleanup."""
        if not self.session_factory:
            raise DatabaseError("Database not initialized")
        
        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            self._stats['query_errors'] += 1
            logger.error("Database session error", error=str(e))
            raise DatabaseError(f"Database session error: {str(e)}")
        finally:
            await session.close()
    
    async def execute_query(self, query: str, params: Optional[Dict] = None) -> Any:
        """Execute raw SQL query with performance monitoring."""
        start_time = time.perf_counter()
        
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})
                self._stats['queries_executed'] += 1
                
                # Update performance stats
                query_time = time.perf_counter() - start_time
                self._stats['total_query_time'] += query_time
                self._stats['average_query_time'] = (
                    self._stats['total_query_time'] / self._stats['queries_executed']
                )
                
                return result
                
        except SQLAlchemyError as e:
            self._stats['query_errors'] += 1
            logger.error("Database query error", error=str(e), query=query)
            raise DatabaseError(f"Database query error: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform database health check."""
        try:
            start_time = time.perf_counter()
            
            # Test basic connectivity
            async with self.get_session() as session:
                result = await session.execute(text("SELECT 1 as health_check"))
                health_check_time = time.perf_counter() - start_time
                
                # Get connection pool info
                pool = self.engine.pool
                pool_info = {
                    'size': pool.size(),
                    'checked_in': pool.checkedin(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow(),
                    'invalid': pool.invalid()
                }
                
                return {
                    'healthy': True,
                    'response_time': health_check_time,
                    'pool_info': pool_info,
                    'stats': self._stats.copy()
                }
                
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return {
                'healthy': False,
                'error': str(e),
                'stats': self._stats.copy()
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            **self._stats,
            'engine_info': {
                'pool_size': settings.database_pool_size,
                'max_overflow': settings.database_max_overflow,
                'pool_timeout': settings.database_pool_timeout,
                'pool_recycle': settings.database_pool_recycle
            }
        }

# =============================================================================
# DATABASE MODELS
# =============================================================================

# Note: In a real implementation, you would define SQLAlchemy models here
# For this example, we'll use a simple structure

class VideoRequest:
    """Video request model for database storage."""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.youtube_url = kwargs.get('youtube_url')
        self.language = kwargs.get('language')
        self.max_clip_length = kwargs.get('max_clip_length')
        self.quality = kwargs.get('quality')
        self.format = kwargs.get('format')
        self.status = kwargs.get('status', 'pending')
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')
        self.processing_time = kwargs.get('processing_time')
        self.error_message = kwargs.get('error_message')

class ViralRequest:
    """Viral request model for database storage."""
    
    def __init__(self, **kwargs):
        self.id = kwargs.get('id')
        self.youtube_url = kwargs.get('youtube_url')
        self.n_variants = kwargs.get('n_variants')
        self.platform = kwargs.get('platform')
        self.use_langchain = kwargs.get('use_langchain', False)
        self.status = kwargs.get('status', 'pending')
        self.created_at = kwargs.get('created_at')
        self.updated_at = kwargs.get('updated_at')
        self.processing_time = kwargs.get('processing_time')
        self.average_viral_score = kwargs.get('average_viral_score')

# =============================================================================
# DATABASE REPOSITORY
# =============================================================================

class VideoRepository:
    """Repository for video-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    async def create_video_request(self, request_data: Dict[str, Any]) -> VideoRequest:
        """Create a new video request record."""
        query = """
        INSERT INTO video_requests (
            youtube_url, language, max_clip_length, quality, format, status, created_at
        ) VALUES (
            :youtube_url, :language, :max_clip_length, :quality, :format, :status, NOW()
        ) RETURNING id, created_at
        """
        
        result = await self.db_manager.execute_query(query, request_data)
        row = result.fetchone()
        
        return VideoRequest(
            id=row[0],
            created_at=row[1],
            **request_data
        )
    
    async def get_video_request(self, request_id: int) -> Optional[VideoRequest]:
        """Get video request by ID."""
        query = """
        SELECT id, youtube_url, language, max_clip_length, quality, format, 
               status, created_at, updated_at, processing_time, error_message
        FROM video_requests 
        WHERE id = :request_id
        """
        
        result = await self.db_manager.execute_query(query, {'request_id': request_id})
        row = result.fetchone()
        
        if row:
            return VideoRequest(
                id=row[0],
                youtube_url=row[1],
                language=row[2],
                max_clip_length=row[3],
                quality=row[4],
                format=row[5],
                status=row[6],
                created_at=row[7],
                updated_at=row[8],
                processing_time=row[9],
                error_message=row[10]
            )
        
        return None
    
    async def update_video_request(self, request_id: int, update_data: Dict[str, Any]) -> bool:
        """Update video request record."""
        set_clauses = []
        params = {'request_id': request_id}
        
        for key, value in update_data.items():
            if key != 'id':
                set_clauses.append(f"{key} = :{key}")
                params[key] = value
        
        if not set_clauses:
            return False
        
        query = f"""
        UPDATE video_requests 
        SET {', '.join(set_clauses)}, updated_at = NOW()
        WHERE id = :request_id
        """
        
        result = await self.db_manager.execute_query(query, params)
        return result.rowcount > 0
    
    async def get_video_requests_by_status(self, status: str, limit: int = 100) -> List[VideoRequest]:
        """Get video requests by status."""
        query = """
        SELECT id, youtube_url, language, max_clip_length, quality, format, 
               status, created_at, updated_at, processing_time, error_message
        FROM video_requests 
        WHERE status = :status
        ORDER BY created_at DESC
        LIMIT :limit
        """
        
        result = await self.db_manager.execute_query(query, {'status': status, 'limit': limit})
        rows = result.fetchall()
        
        return [
            VideoRequest(
                id=row[0],
                youtube_url=row[1],
                language=row[2],
                max_clip_length=row[3],
                quality=row[4],
                format=row[5],
                status=row[6],
                created_at=row[7],
                updated_at=row[8],
                processing_time=row[9],
                error_message=row[10]
            )
            for row in rows
        ]
    
    async def get_video_request_stats(self) -> Dict[str, Any]:
        """Get video request statistics."""
        query = """
        SELECT 
            COUNT(*) as total_requests,
            COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_requests,
            COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_requests,
            COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_requests,
            AVG(processing_time) as average_processing_time
        FROM video_requests
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        """
        
        result = await self.db_manager.execute_query(query)
        row = result.fetchone()
        
        return {
            'total_requests': row[0] or 0,
            'completed_requests': row[1] or 0,
            'failed_requests': row[2] or 0,
            'processing_requests': row[3] or 0,
            'average_processing_time': float(row[4]) if row[4] else 0.0
        }

# =============================================================================
# DATABASE MIGRATIONS
# =============================================================================

class DatabaseMigrator:
    """Database migration manager."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.migrations = [
            self._create_video_requests_table,
            self._create_viral_requests_table,
            self._create_indexes
        ]
    
    async def run_migrations(self) -> None:
        """Run all pending migrations."""
        try:
            # Create migrations table if it doesn't exist
            await self._create_migrations_table()
            
            for migration in self.migrations:
                migration_name = migration.__name__
                
                # Check if migration already ran
                if await self._is_migration_applied(migration_name):
                    logger.info(f"Migration {migration_name} already applied")
                    continue
                
                # Run migration
                logger.info(f"Running migration: {migration_name}")
                await migration()
                
                # Mark migration as applied
                await self._mark_migration_applied(migration_name)
                logger.info(f"Migration {migration_name} completed")
            
            logger.info("All migrations completed successfully")
            
        except Exception as e:
            logger.error("Migration failed", error=str(e))
            raise DatabaseError(f"Migration failed: {str(e)}")
    
    async def _create_migrations_table(self) -> None:
        """Create migrations tracking table."""
        query = """
        CREATE TABLE IF NOT EXISTS migrations (
            id SERIAL PRIMARY KEY,
            migration_name VARCHAR(255) UNIQUE NOT NULL,
            applied_at TIMESTAMP DEFAULT NOW()
        )
        """
        await self.db_manager.execute_query(query)
    
    async def _is_migration_applied(self, migration_name: str) -> bool:
        """Check if migration is already applied."""
        query = "SELECT COUNT(*) FROM migrations WHERE migration_name = :migration_name"
        result = await self.db_manager.execute_query(query, {'migration_name': migration_name})
        return result.fetchone()[0] > 0
    
    async def _mark_migration_applied(self, migration_name: str) -> None:
        """Mark migration as applied."""
        query = "INSERT INTO migrations (migration_name) VALUES (:migration_name)"
        await self.db_manager.execute_query(query, {'migration_name': migration_name})
    
    async def _create_video_requests_table(self) -> None:
        """Create video requests table."""
        query = """
        CREATE TABLE IF NOT EXISTS video_requests (
            id SERIAL PRIMARY KEY,
            youtube_url VARCHAR(500) NOT NULL,
            language VARCHAR(10) NOT NULL,
            max_clip_length INTEGER NOT NULL,
            quality VARCHAR(20) NOT NULL,
            format VARCHAR(10) NOT NULL,
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            processing_time FLOAT,
            error_message TEXT
        )
        """
        await self.db_manager.execute_query(query)
    
    async def _create_viral_requests_table(self) -> None:
        """Create viral requests table."""
        query = """
        CREATE TABLE IF NOT EXISTS viral_requests (
            id SERIAL PRIMARY KEY,
            youtube_url VARCHAR(500) NOT NULL,
            n_variants INTEGER NOT NULL,
            platform VARCHAR(20) NOT NULL,
            use_langchain BOOLEAN DEFAULT FALSE,
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            processing_time FLOAT,
            average_viral_score FLOAT
        )
        """
        await self.db_manager.execute_query(query)
    
    async def _create_indexes(self) -> None:
        """Create database indexes for performance."""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_video_requests_status ON video_requests(status)",
            "CREATE INDEX IF NOT EXISTS idx_video_requests_created_at ON video_requests(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_video_requests_youtube_url ON video_requests(youtube_url)",
            "CREATE INDEX IF NOT EXISTS idx_viral_requests_status ON viral_requests(status)",
            "CREATE INDEX IF NOT EXISTS idx_viral_requests_created_at ON viral_requests(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_viral_requests_platform ON viral_requests(platform)"
        ]
        
        for index_query in indexes:
            await self.db_manager.execute_query(index_query)

# =============================================================================
# DATABASE DEPENDENCY
# =============================================================================

# Global database manager instance
db_manager = DatabaseManager()

async def get_database_manager() -> DatabaseManager:
    """Get database manager instance."""
    return db_manager

async def get_database_session() -> AsyncSession:
    """Get database session for dependency injection."""
    async with db_manager.get_session() as session:
        yield session

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'DatabaseManager',
    'VideoRequest',
    'ViralRequest',
    'VideoRepository',
    'DatabaseMigrator',
    'db_manager',
    'get_database_manager',
    'get_database_session'
]






























