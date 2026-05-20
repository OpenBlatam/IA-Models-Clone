"""
Authentication Service
=====================

Servicio de autenticación y autorización.
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Roles de usuario."""
    ARTIST = "artist"
    MANAGER = "manager"
    ADMIN = "admin"
    ASSISTANT = "assistant"


class Permission(Enum):
    """Permisos."""
    VIEW_EVENTS = "view_events"
    CREATE_EVENTS = "create_events"
    EDIT_EVENTS = "edit_events"
    DELETE_EVENTS = "delete_events"
    VIEW_ROUTINES = "view_routines"
    MANAGE_ROUTINES = "manage_routines"
    VIEW_PROTOCOLS = "view_protocols"
    MANAGE_PROTOCOLS = "manage_protocols"
    VIEW_WARDROBE = "view_wardrobe"
    MANAGE_WARDROBE = "manage_wardrobe"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_SETTINGS = "manage_settings"


@dataclass
class User:
    """Usuario."""
    id: str
    email: str
    name: str
    role: UserRole
    artist_id: Optional[str] = None
    permissions: List[Permission] = None
    created_at: datetime = None
    last_login: Optional[datetime] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = self._get_default_permissions()
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def _get_default_permissions(self) -> List[Permission]:
        """Obtener permisos por defecto según rol."""
        role_permissions = {
            UserRole.ADMIN: list(Permission),
            UserRole.MANAGER: [
                Permission.VIEW_EVENTS,
                Permission.CREATE_EVENTS,
                Permission.EDIT_EVENTS,
                Permission.VIEW_ROUTINES,
                Permission.MANAGE_ROUTINES,
                Permission.VIEW_PROTOCOLS,
                Permission.MANAGE_PROTOCOLS,
                Permission.VIEW_WARDROBE,
                Permission.MANAGE_WARDROBE,
                Permission.VIEW_ANALYTICS
            ],
            UserRole.ARTIST: [
                Permission.VIEW_EVENTS,
                Permission.VIEW_ROUTINES,
                Permission.VIEW_PROTOCOLS,
                Permission.VIEW_WARDROBE
            ],
            UserRole.ASSISTANT: [
                Permission.VIEW_EVENTS,
                Permission.CREATE_EVENTS,
                Permission.EDIT_EVENTS,
                Permission.VIEW_ROUTINES
            ]
        }
        return role_permissions.get(self.role, [])
    
    def has_permission(self, permission: Permission) -> bool:
        """Verificar si tiene permiso."""
        return permission in self.permissions
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role.value,
            "artist_id": self.artist_id,
            "permissions": [p.value for p in self.permissions],
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None
        }


@dataclass
class Session:
    """Sesión de usuario."""
    token: str
    user_id: str
    expires_at: datetime
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def is_expired(self) -> bool:
        """Verificar si la sesión expiró."""
        return datetime.now() > self.expires_at


class AuthService:
    """Servicio de autenticación."""
    
    def __init__(self):
        """Inicializar servicio de autenticación."""
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.password_hashes: Dict[str, str] = {}
        self._logger = logger
    
    def hash_password(self, password: str) -> str:
        """Hashear contraseña."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(
        self,
        email: str,
        password: str,
        name: str,
        role: UserRole,
        artist_id: Optional[str] = None
    ) -> User:
        """
        Crear usuario.
        
        Args:
            email: Email del usuario
            password: Contraseña
            name: Nombre
            role: Rol
            artist_id: ID del artista asociado
        
        Returns:
            Usuario creado
        """
        import uuid
        
        user_id = str(uuid.uuid4())
        password_hash = self.hash_password(password)
        
        user = User(
            id=user_id,
            email=email,
            name=name,
            role=role,
            artist_id=artist_id
        )
        
        self.users[user_id] = user
        self.password_hashes[user_id] = password_hash
        
        self._logger.info(f"Created user: {email} with role {role.value}")
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[Session]:
        """
        Autenticar usuario.
        
        Args:
            email: Email
            password: Contraseña
        
        Returns:
            Sesión o None
        """
        # Buscar usuario por email
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break
        
        if not user:
            return None
        
        # Verificar contraseña
        password_hash = self.hash_password(password)
        if self.password_hashes.get(user.id) != password_hash:
            return None
        
        # Crear sesión
        token = secrets.token_urlsafe(32)
        session = Session(
            token=token,
            user_id=user.id,
            expires_at=datetime.now() + timedelta(hours=24)
        )
        
        self.sessions[token] = session
        
        # Actualizar último login
        user.last_login = datetime.now()
        
        self._logger.info(f"User authenticated: {email}")
        return session
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """
        Obtener usuario desde token.
        
        Args:
            token: Token de sesión
        
        Returns:
            Usuario o None
        """
        session = self.sessions.get(token)
        if not session or session.is_expired():
            return None
        
        return self.users.get(session.user_id)
    
    def has_permission(self, token: str, permission: Permission) -> bool:
        """
        Verificar si el usuario tiene permiso.
        
        Args:
            token: Token de sesión
            permission: Permiso a verificar
        
        Returns:
            True si tiene permiso
        """
        user = self.get_user_from_token(token)
        if not user:
            return False
        
        return user.has_permission(permission)
    
    def logout(self, token: str) -> bool:
        """
        Cerrar sesión.
        
        Args:
            token: Token de sesión
        
        Returns:
            True si se cerró
        """
        if token in self.sessions:
            del self.sessions[token]
            return True
        return False




