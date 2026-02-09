"""
Authentication & Authorization - Sistema de Autenticación y Autorización
========================================================================

Sistema completo de autenticación y autorización con tokens, roles y permisos.
"""

import hashlib
import secrets
import logging
from typing import Optional, Dict, List, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Permisos del sistema"""
    READ_TASKS = "read_tasks"
    WRITE_TASKS = "write_tasks"
    DELETE_TASKS = "delete_tasks"
    READ_METRICS = "read_metrics"
    READ_LOGS = "read_logs"
    ADMIN = "admin"


class Role(Enum):
    """Roles del sistema"""
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"
    SERVICE = "service"


# Mapeo de roles a permisos
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.GUEST: {Permission.READ_TASKS},
    Role.USER: {
        Permission.READ_TASKS,
        Permission.WRITE_TASKS,
        Permission.READ_METRICS
    },
    Role.ADMIN: set(Permission),  # Todos los permisos
    Role.SERVICE: {
        Permission.READ_TASKS,
        Permission.WRITE_TASKS,
        Permission.READ_METRICS
    }
}


@dataclass
class User:
    """Usuario del sistema"""
    id: str
    username: str
    email: Optional[str] = None
    role: Role = Role.USER
    permissions: Set[Permission] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar si el usuario tiene un permiso"""
        if Permission.ADMIN in self.permissions:
            return True
        return permission in self.permissions or permission in ROLE_PERMISSIONS.get(self.role, set())


@dataclass
class Token:
    """Token de autenticación"""
    token: str
    user_id: str
    expires_at: datetime
    created_at: datetime = field(default_factory=datetime.now)
    permissions: Set[Permission] = field(default_factory=set)
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Verificar si el token está expirado"""
        return datetime.now() >= self.expires_at
    
    def is_valid(self) -> bool:
        """Verificar si el token es válido"""
        return not self.is_expired()


class AuthManager:
    """
    Gestor de autenticación y autorización.
    
    Maneja usuarios, tokens, roles y permisos.
    """
    
    def __init__(
        self,
        token_expiry: timedelta = timedelta(hours=24),
        secret_key: Optional[str] = None
    ):
        self.token_expiry = token_expiry
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.tokens: Dict[str, Token] = {}
        self.password_hashes: Dict[str, str] = {}  # user_id -> hash
    
    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: Role = Role.USER,
        permissions: Optional[Set[Permission]] = None
    ) -> User:
        """
        Crear un nuevo usuario.
        
        Args:
            username: Nombre de usuario
            password: Contraseña (se hashea)
            email: Email opcional
            role: Rol del usuario
            permissions: Permisos adicionales
            
        Returns:
            Usuario creado
        """
        user_id = hashlib.sha256(f"{username}{datetime.now().isoformat()}".encode()).hexdigest()[:16]
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            permissions=permissions or set()
        )
        
        # Hash de contraseña
        password_hash = self._hash_password(password)
        self.password_hashes[user_id] = password_hash
        self.users[user_id] = user
        
        logger.info(f"👤 User created: {username} ({user_id})")
        return user
    
    def authenticate(
        self,
        username: str,
        password: str
    ) -> Optional[Token]:
        """
        Autenticar usuario y generar token.
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            
        Returns:
            Token si la autenticación es exitosa, None en caso contrario
        """
        # Buscar usuario
        user = None
        for u in self.users.values():
            if u.username == username and u.is_active:
                user = u
                break
        
        if not user:
            logger.warning(f"❌ Authentication failed: user not found - {username}")
            return None
        
        # Verificar contraseña
        stored_hash = self.password_hashes.get(user.id)
        if not stored_hash or not self._verify_password(password, stored_hash):
            logger.warning(f"❌ Authentication failed: invalid password - {username}")
            return None
        
        # Generar token
        token = self._generate_token(user)
        self.tokens[token.token] = token
        
        # Actualizar último login
        user.last_login = datetime.now()
        
        logger.info(f"✅ User authenticated: {username} ({user.id})")
        return token
    
    def validate_token(self, token: str) -> Optional[User]:
        """
        Validar token y obtener usuario.
        
        Args:
            token: Token a validar
            
        Returns:
            Usuario si el token es válido, None en caso contrario
        """
        token_obj = self.tokens.get(token)
        
        if not token_obj or not token_obj.is_valid():
            if token_obj:
                # Token expirado, eliminarlo
                del self.tokens[token]
            return None
        
        user = self.users.get(token_obj.user_id)
        if not user or not user.is_active:
            return None
        
        return user
    
    def revoke_token(self, token: str) -> bool:
        """
        Revocar un token.
        
        Args:
            token: Token a revocar
            
        Returns:
            True si se revocó, False si no existe
        """
        if token in self.tokens:
            del self.tokens[token]
            logger.info(f"🔒 Token revoked: {token[:16]}...")
            return True
        return False
    
    def check_permission(
        self,
        user: User,
        permission: Permission
    ) -> bool:
        """
        Verificar si un usuario tiene un permiso.
        
        Args:
            user: Usuario
            permission: Permiso a verificar
            
        Returns:
            True si tiene el permiso
        """
        return user.has_permission(permission)
    
    def _generate_token(self, user: User) -> Token:
        """Generar token para usuario"""
        token_string = secrets.token_urlsafe(32)
        expires_at = datetime.now() + self.token_expiry
        
        # Obtener permisos del usuario
        permissions = user.permissions.copy()
        permissions.update(ROLE_PERMISSIONS.get(user.role, set()))
        
        token = Token(
            token=token_string,
            user_id=user.id,
            expires_at=expires_at,
            permissions=permissions
        )
        
        return token
    
    def _hash_password(self, password: str) -> str:
        """Hash de contraseña usando SHA-256 con salt"""
        salt = secrets.token_hex(16)
        hash_obj = hashlib.sha256()
        hash_obj.update(f"{password}{salt}{self.secret_key}".encode())
        return f"{salt}:{hash_obj.hexdigest()}"
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verificar contraseña contra hash almacenado"""
        try:
            salt, hash_value = stored_hash.split(":", 1)
            hash_obj = hashlib.sha256()
            hash_obj.update(f"{password}{salt}{self.secret_key}".encode())
            return hash_obj.hexdigest() == hash_value
        except ValueError:
            return False
    
    def cleanup_expired_tokens(self) -> int:
        """
        Limpiar tokens expirados.
        
        Returns:
            Número de tokens eliminados
        """
        expired = [
            token for token, token_obj in self.tokens.items()
            if token_obj.is_expired()
        ]
        
        for token in expired:
            del self.tokens[token]
        
        if expired:
            logger.debug(f"🧹 Cleaned up {len(expired)} expired tokens")
        
        return len(expired)
