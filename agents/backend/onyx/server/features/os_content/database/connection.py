from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import asyncio
from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import QueuePool
from sqlalchemy import event
from contextlib import asynccontextmanager
from core.config import get_config
from core.exceptions import DatabaseError, handle_async_exception
import structlog
            from .models import Base
            from .models import Base
from typing import Any, List, Dict, Optional
import logging
#!/usr/bin/env python3
"""
Database connection management for OS Content UGC Video Generator
Handles database connections, pooling, and session management
"""



logger = structlog.get_logger("os_content.database")

class DatabaseConnection:
    """Database connection manager with async support"""
    
    def __init__(self) -> Any:
        self.config = get_config()
        self.engine = None
        self.session_factory = None
        self._connection_pool = None
    
    async def initialize(self) -> Any:
        """Initialize database connection"""
        try:
            # Build database URL
            db_url = self._build_database_url()
            
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                db_url,
                echo=self.config.server.debug,
                poolclass=QueuePool,
                pool_size=20,
                max_overflow=30,
                pool_pre_ping=True,
                pool_recycle=3600,
                pool_timeout=30,
                connect_args={
                    "check_same_thread": False
                } if "sqlite" in db_url else {}
            )
            
            # Create session factory
            self.session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False
            )
            
            # Test connection
            async with self.get_session() as session:
                await session.execute("SELECT 1")
            
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise DatabaseError(f"Database initialization failed: {e}")
    
    def _build_database_url(self) -> str:
        """Build database URL from configuration"""
        db_config = getattr(self.config, 'database', None)
        
        if not db_config:
            # Default to SQLite for development
            return "sqlite+aiosqlite:///./os_content.db"
        
        if db_config.get('type') == 'postgresql':
            user = db_config.get('user', 'postgres')
            password = db_config.get('password', '')
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 5432)
            database = db_config.get('database', 'os_content')
            
            return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
        
        elif db_config.get('type') == 'mysql':
            user = db_config.get('user', 'root')
            password = db_config.get('password', '')
            host = db_config.get('host', 'localhost')
            port = db_config.get('port', 3306)
            database = db_config.get('database', 'os_content')
            
            return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{database}"
        
        else:
            # Default to SQLite
            return "sqlite+aiosqlite:///./os_content.db"
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        session = None
        try:
            session = self.session_factory()
            yield session
            await session.commit()
        except Exception as e:
            if session:
                await session.rollback()
            logger.error(f"Database session error: {e}")
            raise DatabaseError(f"Database operation failed: {e}")
        finally:
            if session:
                await session.close()
    
    async def create_tables(self) -> Any:
        """Create all database tables"""
        try:
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            
            logger.info("Database tables created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create tables: {e}")
            raise DatabaseError(f"Table creation failed: {e}")
    
    async def drop_tables(self) -> Any:
        """Drop all database tables"""
        try:
            
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
            
            logger.info("Database tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Failed to drop tables: {e}")
            raise DatabaseError(f"Table drop failed: {e}")
    
    async def health_check(self) -> bool:
        """Check database health"""
        try:
            async with self.get_session() as session:
                result = await session.execute("SELECT 1")
                return result.scalar() == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_connection_info(self) -> dict:
        """Get database connection information"""
        try:
            if not self.engine:
                return {"status": "not_initialized"}
            
            pool = self.engine.pool
            return {
                "status": "connected",
                "pool_size": pool.size(),
                "checked_in": pool.checkedin(),
                "checked_out": pool.checkedout(),
                "overflow": pool.overflow(),
                "invalid": pool.invalid()
            }
        except Exception as e:
            logger.error(f"Failed to get connection info: {e}")
            return {"status": "error", "error": str(e)}
    
    async def close(self) -> Any:
        """Close database connections"""
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database connections: {e}")

# Global database connection instance
db_connection = DatabaseConnection()

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    async with db_connection.get_session() as session:
        yield session

async def initialize_database():
    """Initialize database connection"""
    await db_connection.initialize()
    await db_connection.create_tables()

async def close_database():
    """Close database connection"""
    await db_connection.close() 