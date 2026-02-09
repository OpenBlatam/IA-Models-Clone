"""
Access Main - Funciones base y entry points del módulo de control de acceso

Rol en el Ecosistema IA:
- RBAC, permisos, políticas
- Controlar qué usuarios pueden usar qué modelos/features
- Seguridad granular del sistema
"""

from typing import Optional
from .service import AccessService
from auth.main import get_auth_service
from db.main import get_db_service


# Instancia global del servicio
_access_service: Optional[AccessService] = None


def get_access_service() -> AccessService:
    """
    Obtiene la instancia global del servicio de control de acceso.
    
    Returns:
        AccessService: Servicio de control de acceso
    """
    global _access_service
    if _access_service is None:
        auth_service = get_auth_service()
        db_service = get_db_service()
        _access_service = AccessService(
            auth_service=auth_service,
            db_service=db_service
        )
    return _access_service


async def check_permission(user_id: str, resource: str, action: str) -> bool:
    """
    Verifica si un usuario tiene permiso para una acción.
    
    Args:
        user_id: ID del usuario
        resource: Recurso a acceder
        action: Acción a realizar
        
    Returns:
        True si tiene permiso
    """
    service = get_access_service()
    return await service.check_permission(user_id, resource, action)


async def get_roles(user_id: str) -> list:
    """
    Obtiene los roles de un usuario.
    
    Args:
        user_id: ID del usuario
        
    Returns:
        Lista de roles
    """
    service = get_access_service()
    return await service.get_roles(user_id)


def initialize_access() -> AccessService:
    """
    Inicializa el sistema de control de acceso.
    
    Returns:
        AccessService: Servicio inicializado
    """
    return get_access_service()

