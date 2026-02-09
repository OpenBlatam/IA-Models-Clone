"""
Session Manager - Gestión de sesiones de DB
"""

from typing import Generator
from sqlalchemy.orm import Session
from .service import DatabaseService


class SessionManager:
    """Gestor de sesiones de base de datos"""

    def __init__(self, db_service: DatabaseService):
        """Inicializa el gestor de sesiones"""
        self.db_service = db_service

    def get_db(self) -> Generator[Session, None, None]:
        """Generador de sesiones de base de datos (dependency injection)"""
        db = self.db_service.get_session()
        try:
            yield db
        finally:
            db.close()

