"""
Core Database
==============

Simple and clear database configuration for the Document Workflow Chain system.
"""

from __future__ import annotations
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

from .config import settings

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Global engine and session factory
engine = None
session_factory = None


async def init_database():
    """Initialize database connection - simple and clear"""
    global engine, session_factory
    
    try:
        # Create async engine
        engine = create_async_engine(
            settings.DATABASE_URL,
            pool_size=settings.DATABASE_POOL_SIZE,
            max_overflow=settings.DATABASE_MAX_OVERFLOW,
            echo=settings.DEBUG
        )
        
        # Create session factory
        session_factory = async_sessionmaker(
            bind=engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        logger.info("Database initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """Get database session - simple and clear"""
    if not session_factory:
        raise RuntimeError("Database not initialized")
    
    async with session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def close_database():
    """Close database connection - simple and clear"""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")


