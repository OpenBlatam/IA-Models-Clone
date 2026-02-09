"""
Sistema de autenticación y autorización para Robot Movement AI v2.0
JWT tokens, roles y permisos
"""

import jwt
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

try:
    from fastapi import HTTPException, status, Depends
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class Role(str, Enum):
    """Roles disponibles"""
    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"
    ROBOT = "robot"


class Permission(str, Enum):
    """Permisos disponibles"""
    READ_ROBOTS = "read:robots"
    WRITE_ROBOTS = "write:robots"
    DELETE_ROBOTS = "delete:robots"
    READ_MOVEMENTS = "read:movements"
    WRITE_MOVEMENTS = "write:movements"
    ADMIN_ACCESS = "admin:access"


# Mapeo de roles a permisos
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.ADMIN: [
        Permission.READ_ROBOTS,
        Permission.WRITE_ROBOTS,
        Permission.DELETE_ROBOTS,
        Permission.READ_MOVEMENTS,
        Permission.WRITE_MOVEMENTS,
        Permission.ADMIN_ACCESS,
    ],
    Role.OPERATOR: [
        Permission.READ_ROBOTS,
        Permission.WRITE_ROBOTS,
        Permission.READ_MOVEMENTS,
        Permission.WRITE_MOVEMENTS,
    ],
    Role.VIEWER: [
        Permission.READ_ROBOTS,
        Permission.READ_MOVEMENTS,
    ],
    Role.ROBOT: [
        Permission.READ_ROBOTS,
        Permission.WRITE_MOVEMENTS,
    ],
}


@dataclass
class User:
    """Usuario del sistema"""
    id: str
    username: str
    email: str
    roles: List[Role]
    permissions: List[Permission]
    created_at: datetime
    is_active: bool = True
    
    def has_role(self, role: Role) -> bool:
        """Verificar si tiene un rol"""
        return role in self.roles
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar si tiene un permiso"""
        return permission in self.permissions
    
    def has_any_permission(self, permissions: List[Permission]) -> bool:
        """Verificar si tiene alguno de los permisos"""
        return any(self.has_permission(p) for p in permissions)


class AuthManager:
    """Gestor de autenticación"""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256", token_expiry: timedelta = timedelta(hours=24)):
        """
        Inicializar gestor de autenticación
        
        Args:
            secret_key: Clave secreta para JWT
            algorithm: Algoritmo de JWT
            token_expiry: Tiempo de expiración de tokens
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.token_expiry = token_expiry
        self.users: Dict[str, User] = {}
    
    def create_user(
        self,
        username: str,
        email: str,
        roles: List[Role],
        password: Optional[str] = None
    ) -> User:
        """
        Crear nuevo usuario
        
        Args:
            username: Nombre de usuario
            email: Email
            roles: Lista de roles
            password: Contraseña (opcional)
            
        Returns:
            Usuario creado
        """
        user_id = hashlib.sha256(f"{username}{email}".encode()).hexdigest()[:16]
        
        # Obtener permisos de roles
        permissions = []
        for role in roles:
            permissions.extend(ROLE_PERMISSIONS.get(role, []))
        permissions = list(set(permissions))  # Eliminar duplicados
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            roles=roles,
            permissions=permissions,
            created_at=datetime.now()
        )
        
        self.users[user_id] = user
        
        if password:
            # En producción, hashear contraseña
            self._store_password(user_id, password)
        
        return user
    
    def _store_password(self, user_id: str, password: str):
        """Almacenar contraseña (hasheada)"""
        # En producción, usar bcrypt o similar
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        # Almacenar en base de datos o cache
        pass
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """
        Autenticar usuario y retornar token
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Token JWT o None si falla
        """
        # Buscar usuario
        user = next((u for u in self.users.values() if u.username == username), None)
        
        if not user or not user.is_active:
            return None
        
        # Verificar contraseña (simplificado)
        # En producción, verificar hash
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        # Verificar contra almacenado
        
        # Generar token
        return self.generate_token(user)
    
    def generate_token(self, user: User) -> str:
        """
        Generar token JWT para usuario
        
        Args:
            user: Usuario
            
        Returns:
            Token JWT
        """
        payload = {
            "sub": user.id,
            "username": user.username,
            "email": user.email,
            "roles": [r.value for r in user.roles],
            "permissions": [p.value for p in user.permissions],
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + self.token_expiry
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificar y decodificar token
        
        Args:
            token: Token JWT
            
        Returns:
            Payload decodificado o None si inválido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Obtener usuario desde token"""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        return self.users.get(user_id)


# Instancia global
_auth_manager: Optional[AuthManager] = None


def get_auth_manager(secret_key: Optional[str] = None) -> AuthManager:
    """Obtener instancia global del gestor de autenticación"""
    global _auth_manager
    if _auth_manager is None:
        from core.architecture.config import get_config
        config = get_config()
        _auth_manager = AuthManager(secret_key or config.secret_key)
    return _auth_manager


if FASTAPI_AVAILABLE:
    security = HTTPBearer()
    
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> User:
        """Dependency para obtener usuario actual"""
        auth_manager = get_auth_manager()
        token = credentials.credentials
        user = auth_manager.get_user_from_token(token)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials"
            )
        
        return user
    
    def require_permission(permission: Permission):
        """Dependency factory para requerir permiso"""
        async def permission_checker(user: User = Depends(get_current_user)) -> User:
            if not user.has_permission(permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission required: {permission.value}"
                )
            return user
        return permission_checker
    
    def require_role(role: Role):
        """Dependency factory para requerir rol"""
        async def role_checker(user: User = Depends(get_current_user)) -> User:
            if not user.has_role(role):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role required: {role.value}"
                )
            return user
        return role_checker




