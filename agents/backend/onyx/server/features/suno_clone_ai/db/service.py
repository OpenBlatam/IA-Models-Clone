"""
Database Service - Servicio de base de datos
"""

from typing import Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from configs.settings import Settings


class DatabaseService:
    """Servicio para gestionar conexiones y operaciones de base de datos"""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el servicio de base de datos"""
        self.settings = settings or Settings()
        self.engine = create_engine(self.settings.database_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def get_session(self) -> Session:
        """Obtiene una sesión de base de datos"""
        return self.SessionLocal()

    def create_tables(self) -> None:
        """Crea todas las tablas"""
        from .base import Base
        Base.metadata.create_all(bind=self.engine)

