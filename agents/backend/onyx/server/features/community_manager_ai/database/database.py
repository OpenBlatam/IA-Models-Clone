"""
Database - Configuración de Base de Datos
==========================================

Configuración y utilidades de base de datos.
"""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from typing import Generator
import os

from .models import Base

logger = logging.getLogger(__name__)

# URL de base de datos (por defecto SQLite)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./data/community_manager.db"
)

# Crear engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
else:
    engine = create_engine(DATABASE_URL, echo=False)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Inicializar base de datos (crear tablas)"""
    try:
        # Crear directorio si no existe
        if DATABASE_URL.startswith("sqlite"):
            db_dir = os.path.dirname(DATABASE_URL.replace("sqlite:///", ""))
            if db_dir:
                os.makedirs(db_dir, exist_ok=True)
        
        Base.metadata.create_all(bind=engine)
        logger.info("Base de datos inicializada exitosamente")
    except Exception as e:
        logger.error(f"Error inicializando base de datos: {e}")
        raise


def get_db() -> Generator[Session, None, None]:
    """
    Dependency para obtener sesión de base de datos
    
    Yields:
        Sesión de base de datos
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def reset_db():
    """Resetear base de datos (eliminar todas las tablas)"""
    try:
        Base.metadata.drop_all(bind=engine)
        Base.metadata.create_all(bind=engine)
        logger.info("Base de datos reseteada exitosamente")
    except Exception as e:
        logger.error(f"Error reseteando base de datos: {e}")
        raise




