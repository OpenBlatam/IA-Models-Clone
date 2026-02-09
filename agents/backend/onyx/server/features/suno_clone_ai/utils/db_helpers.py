"""
Database helper utilities for common operations.

Provides consistent database access patterns and utilities.
"""

import logging
from typing import Any, Optional, Dict, List, TypeVar, Generic
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

T = TypeVar('T')


class DatabaseHelper:
    """Helper class for common database operations."""
    
    @staticmethod
    @contextmanager
    def get_session(session_factory):
        """
        Context manager for database sessions.
        
        Args:
            session_factory: Session factory function
        
        Yields:
            Database session
        """
        session = session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    @staticmethod
    async def safe_execute(
        session: AsyncSession,
        operation,
        *args,
        **kwargs
    ) -> Optional[Any]:
        """
        Safely execute a database operation with error handling.
        
        Args:
            session: Database session
            operation: Operation to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Operation result or None on error
        """
        try:
            return await operation(session, *args, **kwargs)
        except Exception as e:
            logger.error(f"Database operation failed: {e}", exc_info=True)
            await session.rollback()
            return None
    
    @staticmethod
    def to_dict(model_instance) -> Dict[str, Any]:
        """
        Convert SQLAlchemy model to dictionary.
        
        Args:
            model_instance: SQLAlchemy model instance
        
        Returns:
            Dictionary representation
        """
        if not model_instance:
            return {}
        
        result = {}
        for column in model_instance.__table__.columns:
            value = getattr(model_instance, column.name)
            if hasattr(value, 'isoformat'):  # datetime objects
                result[column.name] = value.isoformat()
            else:
                result[column.name] = value
        return result
    
    @staticmethod
    def to_dict_list(model_instances: List[Any]) -> List[Dict[str, Any]]:
        """
        Convert list of SQLAlchemy models to list of dictionaries.
        
        Args:
            model_instances: List of SQLAlchemy model instances
        
        Returns:
            List of dictionary representations
        """
        return [DatabaseHelper.to_dict(instance) for instance in model_instances]


# Convenience functions
def get_session(session_factory):
    """Get database session context manager."""
    return DatabaseHelper.get_session(session_factory)


def safe_execute(session: AsyncSession, operation, *args, **kwargs):
    """Safely execute database operation."""
    return DatabaseHelper.safe_execute(session, operation, *args, **kwargs)


def to_dict(model_instance) -> Dict[str, Any]:
    """Convert model to dictionary."""
    return DatabaseHelper.to_dict(model_instance)


def to_dict_list(model_instances: List[Any]) -> List[Dict[str, Any]]:
    """Convert model list to dictionary list."""
    return DatabaseHelper.to_dict_list(model_instances)

