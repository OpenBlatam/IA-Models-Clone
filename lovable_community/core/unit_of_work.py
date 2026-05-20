"""
Unit of Work Pattern

Manages database transactions and ensures atomicity of operations.
Following Clean Architecture principles.
"""

import logging
from typing import Optional, Callable, Any
from contextlib import contextmanager
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


class UnitOfWork:
    """
    Unit of Work for managing database transactions.
    
    Ensures that all operations within a context are committed
    atomically or rolled back on error.
    """
    
    def __init__(self, db: Session):
        """
        Initialize Unit of Work.
        
        Args:
            db: Database session
        """
        self.db = db
        self._committed = False
        self._rolled_back = False
    
    def commit(self) -> None:
        """
        Commit the current transaction.
        
        Raises:
            DatabaseError: If commit fails
        """
        if self._committed or self._rolled_back:
            return
        
        try:
            self.db.commit()
            self._committed = True
            logger.debug("Transaction committed successfully")
        except SQLAlchemyError as e:
            logger.error(f"Error committing transaction: {e}", exc_info=True)
            self.rollback()
            raise DatabaseError(f"Failed to commit transaction: {str(e)}")
    
    def rollback(self) -> None:
        """
        Rollback the current transaction.
        """
        if self._rolled_back:
            return
        
        try:
            self.db.rollback()
            self._rolled_back = True
            logger.debug("Transaction rolled back")
        except SQLAlchemyError as e:
            logger.error(f"Error rolling back transaction: {e}", exc_info=True)
    
    def flush(self) -> None:
        """
        Flush pending changes without committing.
        
        Useful for getting generated IDs before commit.
        """
        try:
            self.db.flush()
        except SQLAlchemyError as e:
            logger.error(f"Error flushing transaction: {e}", exc_info=True)
            raise DatabaseError(f"Failed to flush transaction: {str(e)}")
    
    @contextmanager
    def transaction(self):
        """
        Context manager for transaction management.
        
        Usage:
            with uow.transaction():
                # operations
                uow.commit()  # or auto-commit on exit
        """
        try:
            yield self
            if not self._committed and not self._rolled_back:
                self.commit()
        except Exception as e:
            if not self._rolled_back:
                self.rollback()
            raise
    
    def execute_in_transaction(
        self,
        func: Callable[[], Any],
        auto_commit: bool = True
    ) -> Any:
        """
        Execute a function within a transaction.
        
        Args:
            func: Function to execute
            auto_commit: Whether to auto-commit on success
            
        Returns:
            Result of function execution
            
        Raises:
            DatabaseError: If transaction fails
        """
        with self.transaction():
            try:
                result = func()
                if auto_commit:
                    self.commit()
                return result
            except Exception as e:
                logger.error(f"Error in transaction: {e}", exc_info=True)
                raise


@contextmanager
def unit_of_work(db: Session):
    """
    Context manager for Unit of Work.
    
    Usage:
        with unit_of_work(db) as uow:
            # operations
            uow.commit()
    """
    uow = UnitOfWork(db)
    try:
        yield uow
        if not uow._committed and not uow._rolled_back:
            uow.commit()
    except Exception:
        if not uow._rolled_back:
            uow.rollback()
        raise













