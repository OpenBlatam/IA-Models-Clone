"""
Database session management for Lovable Community SAM3.
"""

from typing import Generator, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session, scoped_session
from sqlalchemy.pool import QueuePool
import logging
import os

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./lovable_community.db"
)

# Global engine and session factory
_engine = None
_SessionLocal = None
_scoped_session = None


def get_engine():
    """Get or create database engine."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            DATABASE_URL,
            poolclass=QueuePool if "postgresql" in DATABASE_URL or "mysql" in DATABASE_URL else None,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            echo=False
        )
        logger.info(f"Database engine created: {DATABASE_URL}")
    return _engine


def get_session_factory():
    """Get or create session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )
        logger.info("Session factory created")
    return _SessionLocal


def get_scoped_session_factory():
    """Get or create scoped session factory."""
    global _scoped_session
    if _scoped_session is None:
        session_factory = get_session_factory()
        _scoped_session = scoped_session(session_factory)
        logger.info("Scoped session factory created")
    return _scoped_session


def get_db_session() -> Generator[Session, None, None]:
    """
    Dependency function to get database session.
    
    Yields:
        Database session
    """
    session_factory = get_session_factory()
    db = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db(config=None):
    """
    Initialize database and create tables.
    
    Args:
        config: Optional configuration object (for compatibility)
    """
    try:
        engine = get_engine()
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Create all tables
        try:
            from ..models.base import Base
            Base.metadata.create_all(bind=engine)
            logger.info("Database tables created/verified")
        except Exception as e:
            logger.warning(f"Could not create tables (may already exist): {e}")
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


def close_db():
    """Close database connections."""
    global _engine, _SessionLocal, _scoped_session
    try:
        if _scoped_session:
            _scoped_session.remove()
        if _engine:
            _engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")




