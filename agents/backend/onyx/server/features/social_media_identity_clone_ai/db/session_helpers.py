"""
Helper functions for database session management.
Eliminates repetitive database transaction patterns.
"""

from contextlib import contextmanager
from typing import Callable, TypeVar, Optional, Any
from sqlalchemy.orm import Session
import logging

from .base import get_db_session

logger = logging.getLogger(__name__)

T = TypeVar('T')


@contextmanager
def db_transaction(
    auto_commit: bool = True,
    auto_rollback: bool = True,
    log_operation: Optional[str] = None
):
    """
    Context manager para operaciones de base de datos con manejo automático de transacciones.
    
    Args:
        auto_commit: Si hacer commit automático al final (default: True)
        auto_rollback: Si hacer rollback automático en caso de error (default: True)
        log_operation: Nombre de la operación para logging (opcional)
        
    Usage:
        with db_transaction(log_operation="save_identity") as db:
            # database operations
            # commit automático al salir exitosamente
    """
    session = get_db_session()
    db = session.__enter__()
    
    try:
        if log_operation:
            logger.debug(f"Starting database transaction: {log_operation}")
        
        yield db
        
        if auto_commit:
            db.commit()
            if log_operation:
                logger.info(f"Database transaction completed: {log_operation}")
        
    except Exception as e:
        if auto_rollback:
            db.rollback()
            logger.error(
                f"Database transaction failed: {log_operation or 'unknown'}: {e}",
                exc_info=True
            )
        raise
    finally:
        session.__exit__(None, None, None)


def with_db_session(
    func: Callable[[Session], T],
    auto_commit: bool = True,
    auto_rollback: bool = True,
    operation_name: Optional[str] = None
) -> T:
    """
    Ejecuta una función con una sesión de base de datos.
    
    Args:
        func: Función que recibe Session como primer argumento
        auto_commit: Si hacer commit automático (default: True)
        auto_rollback: Si hacer rollback en error (default: True)
        operation_name: Nombre de la operación para logging
        
    Returns:
        Resultado de la función
        
    Usage:
        result = with_db_session(
            lambda db: db.query(Model).filter_by(id=id).first(),
            operation_name="get_identity"
        )
    """
    with db_transaction(
        auto_commit=auto_commit,
        auto_rollback=auto_rollback,
        log_operation=operation_name or func.__name__
    ) as db:
        return func(db)








