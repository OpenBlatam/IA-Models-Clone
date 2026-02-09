"""
User Management - Sistema de Gestión de Usuarios Avanzado
========================================================

Sistema completo de gestión de usuarios con roles y permisos.
"""

import asyncio
import logging
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class UserRole(Enum):
    """Roles de usuario."""
    USER = "user"
    ADMIN = "admin"
    MODERATOR = "moderator"
    PREMIUM = "premium"
    GUEST = "guest"


class UserStatus(Enum):
    """Estado de usuario."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    BANNED = "banned"


@dataclass
class User:
    """Usuario."""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)


class UserManager:
    """Gestor de usuarios."""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, str] = {}  # {session_token: user_id}
        self._lock = asyncio.Lock()
    
    def _hash_password(self, password: str) -> str:
        """Hash de contraseña."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    async def create_user(
        self,
        username: str,
        email: str,
        password: str,
        role: UserRole = UserRole.USER,
    ) -> User:
        """Crear nuevo usuario."""
        user_id = hashlib.md5(f"{username}{email}".encode()).hexdigest()
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
        )
        
        async with self._lock:
            self.users[user_id] = user
        
        logger.info(f"Created user: {username}")
        return user
    
    async def authenticate(self, username: str, password: str) -> Optional[User]:
        """Autenticar usuario."""
        password_hash = self._hash_password(password)
        
        for user in self.users.values():
            if user.username == username and user.password_hash == password_hash:
                if user.status == UserStatus.ACTIVE:
                    user.last_login = datetime.now()
                    logger.info(f"User authenticated: {username}")
                    return user
                else:
                    logger.warning(f"User {username} is not active (status: {user.status})")
                    return None
        
        return None
    
    async def update_user_role(self, user_id: str, role: UserRole):
        """Actualizar rol de usuario."""
        async with self._lock:
            if user_id in self.users:
                self.users[user_id].role = role
                logger.info(f"Updated role for user {user_id}: {role.value}")
    
    async def update_user_status(self, user_id: str, status: UserStatus):
        """Actualizar estado de usuario."""
        async with self._lock:
            if user_id in self.users:
                self.users[user_id].status = status
                logger.info(f"Updated status for user {user_id}: {status.value}")
    
    async def add_permission(self, user_id: str, permission: str):
        """Agregar permiso a usuario."""
        async with self._lock:
            if user_id in self.users:
                if permission not in self.users[user_id].permissions:
                    self.users[user_id].permissions.append(permission)
                    logger.debug(f"Added permission {permission} to user {user_id}")
    
    async def has_permission(self, user_id: str, permission: str) -> bool:
        """Verificar si usuario tiene permiso."""
        user = self.users.get(user_id)
        if not user:
            return False
        
        # Admin tiene todos los permisos
        if user.role == UserRole.ADMIN:
            return True
        
        return permission in user.permissions
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Obtener usuario."""
        return self.users.get(user_id)
    
    def list_users(
        self,
        role: Optional[UserRole] = None,
        status: Optional[UserStatus] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Listar usuarios."""
        users = list(self.users.values())
        
        if role:
            users = [u for u in users if u.role == role]
        
        if status:
            users = [u for u in users if u.status == status]
        
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "email": u.email,
                "role": u.role.value,
                "status": u.status.value,
                "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None,
            }
            for u in users[:limit]
        ]
































