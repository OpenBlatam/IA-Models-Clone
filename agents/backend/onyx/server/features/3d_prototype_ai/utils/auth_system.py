"""
Auth System - Sistema de autenticación y permisos
==================================================
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import secrets
from uuid import uuid4

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    """Permisos del sistema"""
    VIEW_PROTOTYPES = "view_prototypes"
    CREATE_PROTOTYPES = "create_prototypes"
    EDIT_PROTOTYPES = "edit_prototypes"
    DELETE_PROTOTYPES = "delete_prototypes"
    SHARE_PROTOTYPES = "share_prototypes"
    EXPORT_PROTOTYPES = "export_prototypes"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_USERS = "manage_users"
    ADMIN = "admin"


class Role(str, Enum):
    """Roles del sistema"""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"


class AuthSystem:
    """Sistema de autenticación y permisos"""
    
    def __init__(self):
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.role_permissions: Dict[Role, List[Permission]] = {
            Role.USER: [
                Permission.VIEW_PROTOTYPES,
                Permission.CREATE_PROTOTYPES,
                Permission.EXPORT_PROTOTYPES
            ],
            Role.PREMIUM: [
                Permission.VIEW_PROTOTYPES,
                Permission.CREATE_PROTOTYPES,
                Permission.EDIT_PROTOTYPES,
                Permission.SHARE_PROTOTYPES,
                Permission.EXPORT_PROTOTYPES,
                Permission.VIEW_ANALYTICS
            ],
            Role.ADMIN: list(Permission)  # Todos los permisos
        }
    
    def create_user(self, username: str, email: str, password: str,
                   role: Role = Role.USER) -> str:
        """Crea un nuevo usuario"""
        user_id = str(uuid4())
        
        # Hash de contraseña (en producción usar bcrypt)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "role": role.value,
            "created_at": datetime.now().isoformat(),
            "active": True,
            "last_login": None
        }
        
        self.users[user_id] = user
        logger.info(f"Usuario creado: {username} ({user_id})")
        return user_id
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Autentica un usuario y retorna session token"""
        # Buscar usuario
        user = None
        for u in self.users.values():
            if u["username"] == username or u["email"] == username:
                user = u
                break
        
        if not user:
            return None
        
        # Verificar contraseña
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user["password_hash"] != password_hash:
            return None
        
        if not user.get("active", True):
            return None
        
        # Crear sesión
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = {
            "user_id": user["id"],
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
        }
        
        # Actualizar último login
        user["last_login"] = datetime.now().isoformat()
        
        logger.info(f"Usuario autenticado: {username}")
        return session_token
    
    def validate_session(self, session_token: str) -> Optional[Dict[str, Any]]:
        """Valida una sesión y retorna información del usuario"""
        session = self.sessions.get(session_token)
        
        if not session:
            return None
        
        # Verificar expiración
        expires_at = datetime.fromisoformat(session["expires_at"])
        if datetime.now() > expires_at:
            del self.sessions[session_token]
            return None
        
        user_id = session["user_id"]
        user = self.users.get(user_id)
        
        if not user or not user.get("active", True):
            return None
        
        return {
            "user_id": user_id,
            "username": user["username"],
            "email": user["email"],
            "role": user["role"]
        }
    
    def has_permission(self, user_id: str, permission: Permission) -> bool:
        """Verifica si un usuario tiene un permiso"""
        user = self.users.get(user_id)
        if not user:
            return False
        
        role = Role(user["role"])
        user_permissions = self.role_permissions.get(role, [])
        
        return permission in user_permissions
    
    def get_user_permissions(self, user_id: str) -> List[str]:
        """Obtiene todos los permisos de un usuario"""
        user = self.users.get(user_id)
        if not user:
            return []
        
        role = Role(user["role"])
        permissions = self.role_permissions.get(role, [])
        
        return [p.value for p in permissions]
    
    def logout(self, session_token: str) -> bool:
        """Cierra una sesión"""
        if session_token in self.sessions:
            del self.sessions[session_token]
            return True
        return False
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un usuario (sin contraseña)"""
        user = self.users.get(user_id)
        if not user:
            return None
        
        return {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"],
            "role": user["role"],
            "created_at": user["created_at"],
            "last_login": user.get("last_login")
        }




