"""
Access Module - Control de Acceso y Permisos
Gestiona el control de acceso basado en roles (RBAC), permisos y políticas de seguridad.

Rol en el Ecosistema IA:
- RBAC, permisos, políticas
- Controlar qué usuarios pueden usar qué modelos/features
- Seguridad granular del sistema

Reglas de Importación:
- Puede importar: auth, db
- NO debe importar: módulos de negocio (chat, agents, etc.)
- Usa inyección de dependencias
"""

from .base import BaseAccessController
from .service import AccessService
from .models import Permission, Role
from .policies import AccessPolicy
from .main import (
    get_access_service,
    check_permission,
    get_roles,
    initialize_access,
)

__all__ = [
    # Clases principales
    "BaseAccessController",
    "AccessService",
    "Permission",
    "Role",
    "AccessPolicy",
    # Funciones de acceso rápido
    "get_access_service",
    "check_permission",
    "get_roles",
    "initialize_access",
]

