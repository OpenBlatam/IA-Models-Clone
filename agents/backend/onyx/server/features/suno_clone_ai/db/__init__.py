"""
DB Module - Base de Datos y ORM
Abstracción de base de datos, modelos ORM, migraciones, y queries.

Rol en el Ecosistema IA:
- Abstracción de base de datos, ORM, modelos
- Almacenar conversaciones, historial, embeddings, metadata
- Persistencia de datos del sistema de IA

Reglas de Importación:
- Puede importar: configs, utils
- NO debe importar: módulos de negocio (chat, agents, etc.)
- Usar TYPE_CHECKING para imports de tipo si es necesario
"""

from .base import BaseModel
from .service import DatabaseService
from .models import Base
from .session import SessionManager
from .main import (
    get_db_service,
    get_session_manager,
    get_db,
    create_tables,
    initialize_db,
)

__all__ = [
    # Clases principales
    "BaseModel",
    "DatabaseService",
    "Base",
    "SessionManager",
    # Funciones de acceso rápido
    "get_db_service",
    "get_session_manager",
    "get_db",
    "create_tables",
    "initialize_db",
]

