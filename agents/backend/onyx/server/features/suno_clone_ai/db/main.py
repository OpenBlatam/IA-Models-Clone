"""
DB Main - Funciones base y entry points del módulo de base de datos

Rol en el Ecosistema IA:
- Abstracción de base de datos, ORM, modelos
- Almacenar conversaciones, historial, embeddings, metadata
- Persistencia de datos del sistema de IA
"""

from typing import Optional, Generator
from sqlalchemy.orm import Session
from .service import DatabaseService
from .session import SessionManager
from configs.main import get_settings


# Instancia global del servicio
_db_service: Optional[DatabaseService] = None
_session_manager: Optional[SessionManager] = None


def get_db_service() -> DatabaseService:
    """
    Obtiene la instancia global del servicio de base de datos.
    
    Returns:
        DatabaseService: Servicio de base de datos
    """
    global _db_service
    if _db_service is None:
        settings = get_settings()
        _db_service = DatabaseService(settings)
    return _db_service


def get_session_manager() -> SessionManager:
    """
    Obtiene el gestor de sesiones de base de datos.
    
    Returns:
        SessionManager: Gestor de sesiones
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager(get_db_service())
    return _session_manager


def get_db() -> Generator[Session, None, None]:
    """
    Dependency injection para FastAPI.
    Obtiene una sesión de base de datos.
    
    Yields:
        Session: Sesión de SQLAlchemy
    """
    manager = get_session_manager()
    yield from manager.get_db()


def create_tables() -> None:
    """
    Crea todas las tablas de la base de datos.
    Debe llamarse al inicio de la aplicación.
    """
    db_service = get_db_service()
    db_service.create_tables()


def initialize_db() -> DatabaseService:
    """
    Inicializa el sistema de base de datos.
    Crea tablas si no existen.
    
    Returns:
        DatabaseService: Servicio inicializado
    """
    db_service = get_db_service()
    create_tables()
    return db_service

