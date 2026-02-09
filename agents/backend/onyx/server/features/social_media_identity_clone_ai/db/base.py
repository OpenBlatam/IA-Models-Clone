"""
Base de datos y configuración SQLAlchemy
"""

import logging
from contextlib import contextmanager
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..config import get_settings

logger = logging.getLogger(__name__)

Base = declarative_base()

_engine = None
_SessionLocal = None


def init_db():
    """Inicializa la base de datos"""
    global _engine, _SessionLocal
    
    if _engine is not None:
        return
    
    settings = get_settings()
    
    # Configurar engine con connection pooling optimizado
    if settings.database_url.startswith("sqlite"):
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            pool_size=10,  # Tamaño del pool
            max_overflow=20,  # Conexiones adicionales
            pool_recycle=3600,  # Reciclar conexiones cada hora
            echo=False
        )
    else:
        _engine = create_engine(
            settings.database_url,
            pool_pre_ping=True,  # Verificar conexiones antes de usar
            pool_size=20,  # Tamaño del pool (más grande para producción)
            max_overflow=40,  # Conexiones adicionales bajo carga
            pool_recycle=3600,  # Reciclar conexiones cada hora
            pool_timeout=30,  # Timeout para obtener conexión
            echo=False
        )
    
    # Crear sessionmaker
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)
    
    # Crear tablas
    Base.metadata.create_all(bind=_engine)
    
    logger.info("Base de datos inicializada")


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Context manager para obtener sesión de base de datos
    
    Usage:
        with get_db_session() as db:
            # usar db
    """
    if _SessionLocal is None:
        init_db()
    
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db() -> Session:
    """Obtiene sesión de base de datos (para dependency injection)"""
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()




