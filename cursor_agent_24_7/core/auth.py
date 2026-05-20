"""
Authentication - Sistema de autenticación
==========================================

Sistema de autenticación y autorización para el agente.
"""

import hashlib
import secrets
import logging
from typing import Optional, Dict, List
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class Role(Enum):
    """Roles de usuario"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    GUEST = "guest"


@dataclass
class User:
    """Usuario del sistema"""
    username: str
    password_hash: str
    role: Role
    api_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    enabled: bool = True


class AuthManager:
    """Gestor de autenticación"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> username
        self.sessions: Dict[str, Dict] = {}  # session_id -> session_data
        self._load_default_users()
    
    def _load_default_users(self):
        """Cargar usuarios por defecto"""
        # Usuario admin por defecto (cambiar en producción)
        admin_hash = self._hash_password("admin")
        admin = User(
            username="admin",
            password_hash=admin_hash,
            role=Role.ADMIN,
            api_key=self._generate_api_key()
        )
        self.users["admin"] = admin
        self.api_keys[admin.api_key] = "admin"
        logger.info("👤 Default admin user created (username: admin, password: admin)")
    
    def _hash_password(self, password: str) -> str:
        """Hashear contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _generate_api_key(self) -> str:
        """Generar API key"""
        return secrets.token_urlsafe(32)
    
    def _generate_session_id(self) -> str:
        """Generar ID de sesión"""
        return secrets.token_urlsafe(24)
    
    def create_user(
        self,
        username: str,
        password: str,
        role: Role = Role.USER
    ) -> Optional[User]:
        """Crear nuevo usuario"""
        if username in self.users:
            logger.warning(f"User {username} already exists")
            return None
        
        password_hash = self._hash_password(password)
        api_key = self._generate_api_key()
        
        user = User(
            username=username,
            password_hash=password_hash,
            role=role,
            api_key=api_key
        )
        
        self.users[username] = user
        self.api_keys[api_key] = username
        
        logger.info(f"👤 User created: {username} (role: {role.value})")
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[str]:
        """Autenticar usuario y crear sesión"""
        if username not in self.users:
            return None
        
        user = self.users[username]
        
        if not user.enabled:
            return None
        
        password_hash = self._hash_password(password)
        if password_hash != user.password_hash:
            return None
        
        # Crear sesión
        session_id = self._generate_session_id()
        self.sessions[session_id] = {
            "username": username,
            "role": user.role.value,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(hours=24)
        }
        
        # Actualizar último login
        user.last_login = datetime.now()
        
        logger.info(f"🔐 User authenticated: {username}")
        return session_id
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Autenticar usando API key"""
        if api_key not in self.api_keys:
            return None
        
        username = self.api_keys[api_key]
        if username not in self.users:
            return None
        
        user = self.users[username]
        if not user.enabled:
            return None
        
        return user
    
    def get_user_from_session(self, session_id: str) -> Optional[User]:
        """Obtener usuario desde sesión"""
        if session_id not in self.sessions:
            return None
        
        session = self.sessions[session_id]
        
        # Verificar expiración
        if datetime.now() > session["expires_at"]:
            del self.sessions[session_id]
            return None
        
        username = session["username"]
        if username not in self.users:
            return None
        
        return self.users[username]
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Verificar si usuario tiene permiso"""
        if user.role == Role.ADMIN:
            return True
        
        permissions = {
            Role.USER: ["read", "write", "execute"],
            Role.READONLY: ["read"],
            Role.GUEST: []
        }
        
        return permission in permissions.get(user.role, [])
    
    def revoke_session(self, session_id: str) -> bool:
        """Revocar sesión"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def list_users(self) -> List[Dict]:
        """Listar usuarios"""
        return [
            {
                "username": user.username,
                "role": user.role.value,
                "enabled": user.enabled,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            }
            for user in self.users.values()
        ]



