"""
Helpers para integración del Dependency Container con FastAPI

Proporciona funciones de utilidad para usar el contenedor de dependencias
en el contexto de FastAPI con gestión automática de scopes.
"""

from typing import Optional, TypeVar, Type
from fastapi import Request, Depends
from sqlalchemy.orm import Session
import uuid

from .dependency_container import container, DependencyContainer
from .database import get_session_local

T = TypeVar('T')


def get_request_scope_id(request: Request) -> str:
    """
    Obtener o crear scope ID para el request actual.
    
    Args:
        request: FastAPI Request object
        
    Returns:
        str: Scope ID único para este request
    """
    if not hasattr(request.state, 'scope_id'):
        request.state.scope_id = str(uuid.uuid4())
    return request.state.scope_id


def create_container_dependency(
    service_name: str,
    service_type: Optional[Type[T]] = None
):
    """
    Crear una dependency de FastAPI para obtener un servicio del contenedor.
    
    Args:
        service_name: Nombre del servicio registrado
        service_type: Tipo del servicio (opcional, para type hints)
        
    Returns:
        Función dependency para FastAPI
        
    Example:
        get_chat_service = create_container_dependency('chat_service', ChatService)
        
        @router.post("/chats")
        async def create_chat(
            chat_service: ChatService = Depends(get_chat_service)
        ):
            ...
    """
    async def dependency(request: Request) -> T:
        scope_id = get_request_scope_id(request)
        return await container.get(service_name, scope_id=scope_id)
    
    # Mejorar type hints si se proporciona service_type
    if service_type:
        dependency.__annotations__['return'] = service_type
    
    return dependency


def create_type_based_dependency(service_type: Type[T]):
    """
    Crear una dependency de FastAPI basada en el tipo del servicio.
    
    Requiere que el servicio esté registrado con service_type.
    
    Args:
        service_type: Tipo del servicio
        
    Returns:
        Función dependency para FastAPI
        
    Example:
        get_chat_service = create_type_based_dependency(ChatService)
        
        @router.post("/chats")
        async def create_chat(
            chat_service: ChatService = Depends(get_chat_service)
        ):
            ...
    """
    async def dependency(request: Request) -> T:
        scope_id = get_request_scope_id(request)
        return await container.get_by_type(service_type, scope_id=scope_id)
    
    dependency.__annotations__['return'] = service_type
    return dependency


def setup_request_scope_middleware(app):
    """
    Configurar middleware para gestión automática de scopes.
    
    Este middleware:
    - Crea un scope único para cada request
    - Limpia el scope al finalizar el request
    
    Args:
        app: FastAPI application instance
        
    Example:
        from fastapi import FastAPI
        from ..core.container_helpers import setup_request_scope_middleware
        
        app = FastAPI()
        setup_request_scope_middleware(app)
    """
    @app.middleware("http")
    async def scope_middleware(request: Request, call_next):
        # Crear scope para este request
        scope_id = get_request_scope_id(request)
        
        try:
            response = await call_next(request)
            return response
        finally:
            # Limpiar scope al finalizar request
            container.clear_scope(scope_id)


def register_database_session(container_instance: DependencyContainer = None):
    """
    Registrar la sesión de base de datos en el contenedor.
    
    Esto permite que los servicios obtengan la sesión de DB automáticamente.
    
    Args:
        container_instance: Instancia del contenedor (default: container global)
    """
    if container_instance is None:
        container_instance = container
    
    def get_db_session() -> Session:
        """Factory para obtener sesión de DB"""
        SessionLocal = get_session_local()
        return SessionLocal()
    
    # Registrar como scoped para que cada request tenga su propia sesión
    container_instance.register_scoped('db_session', get_db_session)


# Dependencies pre-construidas para servicios comunes
def get_chat_service_dependency():
    """Dependency para ChatService"""
    return create_container_dependency('chat_service')


def get_ranking_service_dependency():
    """Dependency para RankingService"""
    return create_container_dependency('ranking_service')


def get_identity_service_dependency():
    """Dependency para IdentityService"""
    return create_container_dependency('identity_service')




