"""
Database Session Management
===========================

Gestión de sesiones de base de datos con SQLAlchemy async.
"""

import os
import logging
from typing import AsyncGenerator, Optional
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from ..config.settings import get_settings

logger = logging.getLogger(__name__)

# Engine global
_engine: Optional[create_async_engine] = None
_async_session_maker: Optional[async_sessionmaker] = None


def get_database_url() -> str:
    """Obtener URL de base de datos desde configuración."""
    settings = get_settings()
    
    # Intentar desde variable de entorno primero
    database_url = settings.database_url or os.getenv("DATABASE_URL")
    if database_url:
        # Convertir a async si es necesario
        if database_url.startswith("postgresql://"):
            database_url = database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        return database_url
    
    # Construir desde configuración
    db_host = settings.db_host
    db_port = settings.db_port
    db_user = settings.db_user
    db_password = settings.db_password
    db_name = settings.db_name
    
    return f"postgresql+asyncpg://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"


def init_db() -> None:
    """Inicializar conexión a base de datos con connection pooling."""
    global _engine, _async_session_maker
    
    if _engine is not None:
        return
    
    database_url = get_database_url()
    
    logger.info(f"Initializing database connection: {database_url.split('@')[1] if '@' in database_url else 'local'}")
    
    # Use connection pooling for better stability
    from sqlalchemy.pool import QueuePool
    
    _engine = create_async_engine(
        database_url,
        echo=False,
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=40,
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_reset_on_return='commit',
        future=True,
        connect_args={
            "server_settings": {
                "application_name": "manuales_hogar_ai",
                "statement_timeout": "30000",
            },
            "command_timeout": 30,
            "prepared_statement_cache_size": 100,
        },
    )
    
    _async_session_maker = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Obtener sesión async de base de datos con mejor manejo de errores."""
    global _async_session_maker
    
    if _async_session_maker is None:
        init_db()
    
    session = None
    try:
        session = _async_session_maker()
        yield session
        await session.commit()
    except Exception as e:
        if session:
            await session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True)
        from ..core.exceptions import DatabaseError
        raise DatabaseError(f"Database operation failed: {str(e)}", "DATABASE_ERROR")
    finally:
        if session:
            await session.close()


def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Alias para get_async_session (compatibilidad)."""
    return get_async_session()


async def close_db() -> None:
    """Cerrar conexión a base de datos."""
    global _engine
    if _engine:
        await _engine.dispose()
        _engine = None
        logger.info("Database connection closed")

