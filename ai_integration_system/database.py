"""
AI Integration System - Database Configuration
Database connection, session management, and utilities
"""

import logging
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager
from typing import Generator

from .config import settings, get_database_url
from .models import Base

logger = logging.getLogger(__name__)

# Database engine configuration
engine = create_engine(
    get_database_url(),
    poolclass=QueuePool,
    pool_size=settings.database.pool_size,
    max_overflow=settings.database.max_overflow,
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections every hour
    echo=settings.database.echo,
    echo_pool=settings.debug,
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Database event listeners
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance"""
    if "sqlite" in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log database connection checkout"""
    logger.debug("Database connection checked out")

@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_connection, connection_record):
    """Log database connection checkin"""
    logger.debug("Database connection checked in")

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise

def drop_tables():
    """Drop all database tables"""
    try:
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Error dropping database tables: {str(e)}")
        raise

def get_db_session() -> Session:
    """Get a database session"""
    return SessionLocal()

@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager for database sessions"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise
    finally:
        session.close()

def test_database_connection() -> bool:
    """Test database connection"""
    try:
        with get_db() as session:
            session.execute("SELECT 1")
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return False

def get_database_info() -> dict:
    """Get database information"""
    try:
        with get_db() as session:
            # Get database version
            if "postgresql" in get_database_url():
                result = session.execute("SELECT version()")
                version = result.scalar()
            elif "mysql" in get_database_url():
                result = session.execute("SELECT VERSION()")
                version = result.scalar()
            elif "sqlite" in get_database_url():
                result = session.execute("SELECT sqlite_version()")
                version = result.scalar()
            else:
                version = "Unknown"
            
            # Get table count
            table_count = len(Base.metadata.tables)
            
            return {
                "connected": True,
                "version": version,
                "table_count": table_count,
                "url": get_database_url().split("@")[-1] if "@" in get_database_url() else "local"
            }
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        return {
            "connected": False,
            "error": str(e)
        }

def optimize_database():
    """Optimize database performance"""
    try:
        with get_db() as session:
            if "postgresql" in get_database_url():
                # PostgreSQL optimizations
                session.execute("VACUUM ANALYZE")
                session.execute("REINDEX DATABASE")
            elif "mysql" in get_database_url():
                # MySQL optimizations
                session.execute("OPTIMIZE TABLE")
            elif "sqlite" in get_database_url():
                # SQLite optimizations
                session.execute("VACUUM")
                session.execute("ANALYZE")
        
        logger.info("Database optimization completed")
        return True
    except Exception as e:
        logger.error(f"Database optimization failed: {str(e)}")
        return False

def backup_database(backup_path: str) -> bool:
    """Create database backup"""
    try:
        import shutil
        import os
        
        if "sqlite" in get_database_url():
            # SQLite backup
            db_path = get_database_url().replace("sqlite:///", "")
            shutil.copy2(db_path, backup_path)
        else:
            # For other databases, you would use specific backup tools
            logger.warning(f"Backup not implemented for this database type")
            return False
        
        logger.info(f"Database backup created: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Database backup failed: {str(e)}")
        return False

def restore_database(backup_path: str) -> bool:
    """Restore database from backup"""
    try:
        import shutil
        
        if "sqlite" in get_database_url():
            # SQLite restore
            db_path = get_database_url().replace("sqlite:///", "")
            shutil.copy2(backup_path, db_path)
        else:
            # For other databases, you would use specific restore tools
            logger.warning(f"Restore not implemented for this database type")
            return False
        
        logger.info(f"Database restored from: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Database restore failed: {str(e)}")
        return False

# Database health check
def check_database_health() -> dict:
    """Comprehensive database health check"""
    health_status = {
        "status": "healthy",
        "checks": {},
        "timestamp": None
    }
    
    try:
        from datetime import datetime
        health_status["timestamp"] = datetime.utcnow().isoformat()
        
        # Connection test
        health_status["checks"]["connection"] = test_database_connection()
        
        # Get database info
        db_info = get_database_info()
        health_status["checks"]["database_info"] = db_info
        
        # Check table existence
        try:
            with get_db() as session:
                from .models import IntegrationRequest
                table_exists = session.query(IntegrationRequest).first() is not None
                health_status["checks"]["tables_exist"] = True
        except Exception as e:
            health_status["checks"]["tables_exist"] = False
            health_status["checks"]["table_error"] = str(e)
        
        # Overall status
        if not all([
            health_status["checks"].get("connection", False),
            health_status["checks"].get("tables_exist", False)
        ]):
            health_status["status"] = "unhealthy"
        
    except Exception as e:
        health_status["status"] = "error"
        health_status["error"] = str(e)
    
    return health_status

# Database migration utilities
def migrate_database():
    """Run database migrations"""
    try:
        # This would typically use Alembic
        # For now, we'll just recreate tables
        create_tables()
        logger.info("Database migration completed")
        return True
    except Exception as e:
        logger.error(f"Database migration failed: {str(e)}")
        return False

# Connection pool monitoring
def get_connection_pool_status() -> dict:
    """Get connection pool status"""
    try:
        pool = engine.pool
        return {
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
    except Exception as e:
        logger.error(f"Error getting connection pool status: {str(e)}")
        return {"error": str(e)}

# Database statistics
def get_database_statistics() -> dict:
    """Get database statistics"""
    try:
        stats = {}
        
        with get_db() as session:
            from .models import (
                IntegrationRequest,
                IntegrationResult,
                WebhookEvent,
                IntegrationLog
            )
            
            # Count records in each table
            stats["integration_requests"] = session.query(IntegrationRequest).count()
            stats["integration_results"] = session.query(IntegrationResult).count()
            stats["webhook_events"] = session.query(WebhookEvent).count()
            stats["integration_logs"] = session.query(IntegrationLog).count()
            
            # Get recent activity
            from datetime import datetime, timedelta
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)
            
            stats["recent_requests"] = session.query(IntegrationRequest).filter(
                IntegrationRequest.created_at >= recent_cutoff
            ).count()
            
            stats["recent_webhooks"] = session.query(WebhookEvent).filter(
                WebhookEvent.received_at >= recent_cutoff
            ).count()
        
        return stats
    except Exception as e:
        logger.error(f"Error getting database statistics: {str(e)}")
        return {"error": str(e)}

# Initialize database on import
def initialize_database():
    """Initialize database on application startup"""
    try:
        # Test connection
        if not test_database_connection():
            logger.error("Database connection failed during initialization")
            return False
        
        # Create tables if they don't exist
        create_tables()
        
        # Log database info
        db_info = get_database_info()
        logger.info(f"Database initialized: {db_info}")
        
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        return False

# Export main functions
__all__ = [
    "engine",
    "SessionLocal",
    "get_db_session",
    "get_db",
    "create_tables",
    "drop_tables",
    "test_database_connection",
    "get_database_info",
    "optimize_database",
    "backup_database",
    "restore_database",
    "check_database_health",
    "migrate_database",
    "get_connection_pool_status",
    "get_database_statistics",
    "initialize_database"
]



























