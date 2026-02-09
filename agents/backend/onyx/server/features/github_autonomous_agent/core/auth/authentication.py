"""
Sistema de Autenticación y Autorización.
"""

import hashlib
import secrets
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import jwt
from functools import wraps

from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config.logging_config import get_logger
from config.settings import settings
from config.di_setup import get_service

logger = get_logger(__name__)

security = HTTPBearer()


class UserRole(str, Enum):
    """Roles de usuario."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    SERVICE = "service"


class Permission(str, Enum):
    """Permisos del sistema."""
    # Tareas
    TASK_CREATE = "task:create"
    TASK_READ = "task:read"
    TASK_UPDATE = "task:update"
    TASK_DELETE = "task:delete"
    TASK_EXECUTE = "task:execute"
    
    # Agente
    AGENT_START = "agent:start"
    AGENT_STOP = "agent:stop"
    AGENT_PAUSE = "agent:pause"
    AGENT_RESUME = "agent:resume"
    AGENT_VIEW = "agent:view"
    
    # GitHub
    GITHUB_CONNECT = "github:connect"
    GITHUB_READ = "github:read"
    GITHUB_WRITE = "github:write"
    
    # LLM
    LLM_USE = "llm:use"
    LLM_ADMIN = "llm:admin"
    
    # Sistema
    SYSTEM_ADMIN = "system:admin"
    AUDIT_READ = "audit:read"
    MONITORING_VIEW = "monitoring:view"


# Mapeo de roles a permisos
ROLE_PERMISSIONS: Dict[UserRole, List[Permission]] = {
    UserRole.ADMIN: list(Permission),  # Todos los permisos
    UserRole.USER: [
        Permission.TASK_CREATE,
        Permission.TASK_READ,
        Permission.TASK_UPDATE,
        Permission.AGENT_START,
        Permission.AGENT_STOP,
        Permission.AGENT_PAUSE,
        Permission.AGENT_RESUME,
        Permission.AGENT_VIEW,
        Permission.GITHUB_CONNECT,
        Permission.GITHUB_READ,
        Permission.GITHUB_WRITE,
        Permission.LLM_USE,
        Permission.MONITORING_VIEW
    ],
    UserRole.VIEWER: [
        Permission.TASK_READ,
        Permission.AGENT_VIEW,
        Permission.GITHUB_READ,
        Permission.MONITORING_VIEW
    ],
    UserRole.SERVICE: [
        Permission.TASK_READ,
        Permission.TASK_UPDATE,
        Permission.AGENT_VIEW
    ]
}


class User:
    """Representa un usuario."""
    
    def __init__(
        self,
        user_id: str,
        username: str,
        email: str,
        role: UserRole = UserRole.USER,
        permissions: Optional[List[Permission]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Inicializar usuario.
        
        Args:
            user_id: ID único del usuario
            username: Nombre de usuario
            email: Email
            role: Rol del usuario
            permissions: Permisos adicionales (opcional)
            metadata: Metadata adicional
        """
        self.user_id = user_id
        self.username = username
        self.email = email
        self.role = role
        self.permissions = permissions or []
        self.metadata = metadata or {}
        self.created_at = datetime.now()
        self.last_login: Optional[datetime] = None
    
    def has_permission(self, permission: Permission) -> bool:
        """
        Verificar si el usuario tiene un permiso.
        
        Args:
            permission: Permiso a verificar
            
        Returns:
            True si tiene el permiso
        """
        # Permisos del rol
        role_perms = ROLE_PERMISSIONS.get(self.role, [])
        if permission in role_perms:
            return True
        
        # Permisos adicionales
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Verificar si tiene alguno de los permisos."""
        return any(self.has_permission(p) for p in permissions)
    
    def has_all_permissions(self, permissions: List[Permission]) -> bool:
        """Verificar si tiene todos los permisos."""
        return all(self.has_permission(p) for p in permissions)


class AuthenticationService:
    """Servicio de autenticación."""
    
    def __init__(self, secret_key: Optional[str] = None, algorithm: str = "HS256"):
        """
        Inicializar servicio de autenticación.
        
        Args:
            secret_key: Secret key para JWT (opcional)
            algorithm: Algoritmo de JWT
        """
        self.secret_key = secret_key or settings.SECRET_KEY
        self.algorithm = algorithm
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}
    
    def create_user(
        self,
        username: str,
        email: str,
        role: UserRole = UserRole.USER,
        password: Optional[str] = None
    ) -> User:
        """
        Crear nuevo usuario.
        
        Args:
            username: Nombre de usuario
            email: Email
            role: Rol
            password: Contraseña (opcional)
            
        Returns:
            Usuario creado
        """
        user_id = f"user_{secrets.token_urlsafe(16)}"
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            role=role
        )
        
        if password:
            # Hash de contraseña (simple, usar bcrypt en producción)
            user.metadata["password_hash"] = hashlib.sha256(password.encode()).hexdigest()
        
        self.users[user_id] = user
        logger.info(f"Usuario creado: {username} ({user_id})")
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """
        Autenticar usuario con username y password.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Usuario si la autenticación es exitosa, None si falla
        """
        for user in self.users.values():
            if user.username == username:
                stored_hash = user.metadata.get("password_hash")
                if stored_hash:
                    password_hash = hashlib.sha256(password.encode()).hexdigest()
                    if password_hash == stored_hash:
                        user.last_login = datetime.now()
                        return user
        return None
    
    def generate_token(self, user: User, expires_in: int = 3600) -> str:
        """
        Generar token JWT para usuario.
        
        Args:
            user: Usuario
            expires_in: Tiempo de expiración en segundos
            
        Returns:
            Token JWT
        """
        payload = {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "permissions": [p.value for p in self._get_user_permissions(user)],
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[User]:
        """
        Verificar y decodificar token JWT.
        
        Args:
            token: Token JWT
            
        Returns:
            Usuario si el token es válido, None si no
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            user_id = payload.get("user_id")
            if user_id and user_id in self.users:
                user = self.users[user_id]
                user.last_login = datetime.now()
                return user
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token inválido: {e}")
        return None
    
    def create_api_key(
        self,
        user: User,
        name: str,
        permissions: Optional[List[Permission]] = None,
        expires_in_days: Optional[int] = None
    ) -> str:
        """
        Crear API key para usuario.
        
        Args:
            user: Usuario
            name: Nombre de la API key
            permissions: Permisos específicos (opcional)
            expires_in_days: Días hasta expiración (opcional)
            
        Returns:
            API key
        """
        api_key = f"sk_{secrets.token_urlsafe(32)}"
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        self.api_keys[api_key] = {
            "user_id": user.user_id,
            "name": name,
            "permissions": [p.value for p in (permissions or [])],
            "created_at": datetime.now(),
            "expires_at": expires_at
        }
        
        logger.info(f"API key creada para usuario {user.username}: {name}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """
        Verificar API key.
        
        Args:
            api_key: API key
            
        Returns:
            Usuario si la key es válida, None si no
        """
        key_data = self.api_keys.get(api_key)
        if not key_data:
            return None
        
        # Verificar expiración
        if key_data.get("expires_at"):
            if datetime.now() > key_data["expires_at"]:
                logger.warning("API key expirada")
                return None
        
        user_id = key_data["user_id"]
        if user_id in self.users:
            user = self.users[user_id]
            # Aplicar permisos específicos de la key si existen
            if key_data.get("permissions"):
                user.permissions = [Permission(p) for p in key_data["permissions"]]
            return user
        
        return None
    
    def _get_user_permissions(self, user: User) -> List[Permission]:
        """Obtener todos los permisos de un usuario."""
        role_perms = ROLE_PERMISSIONS.get(user.role, [])
        all_perms = set(role_perms) | set(user.permissions)
        return list(all_perms)


# Instancia global
_auth_service: Optional[AuthenticationService] = None


def get_auth_service() -> AuthenticationService:
    """Obtener servicio de autenticación."""
    global _auth_service
    if _auth_service is None:
        _auth_service = AuthenticationService()
    return _auth_service


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> User:
    """
    Dependency para obtener usuario actual.
    
    Args:
        credentials: Credenciales HTTP
        
    Returns:
        Usuario actual
        
    Raises:
        HTTPException: Si la autenticación falla
    """
    auth_service = get_auth_service()
    token = credentials.credentials
    
    # Intentar verificar como JWT
    user = auth_service.verify_token(token)
    if user:
        return user
    
    # Intentar verificar como API key
    user = auth_service.verify_api_key(token)
    if user:
        return user
    
    raise HTTPException(
        status_code=401,
        detail="Token inválido o expirado",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_permission(permission: Permission):
    """
    Decorador para requerir permiso específico.
    
    Args:
        permission: Permiso requerido
        
    Returns:
        Decorador
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Permiso requerido: {permission.value}"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator


def require_role(role: UserRole):
    """
    Decorador para requerir rol específico.
    
    Args:
        role: Rol requerido
        
    Returns:
        Decorador
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, user: User = Depends(get_current_user), **kwargs):
            if user.role != role and user.role != UserRole.ADMIN:
                raise HTTPException(
                    status_code=403,
                    detail=f"Rol requerido: {role.value}"
                )
            return await func(*args, user=user, **kwargs)
        return wrapper
    return decorator



