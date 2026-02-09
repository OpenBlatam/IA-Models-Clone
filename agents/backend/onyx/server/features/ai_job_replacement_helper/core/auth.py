"""
Authentication Service - Sistema de autenticación
==================================================

Sistema de autenticación y gestión de usuarios.
"""

import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """Roles de usuario"""
    USER = "user"
    PREMIUM = "premium"
    ADMIN = "admin"
    MENTOR = "mentor"


@dataclass
class User:
    """Usuario del sistema"""
    id: str
    email: str
    username: str
    password_hash: str
    role: UserRole = UserRole.USER
    email_verified: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    profile_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """Sesión de usuario"""
    session_id: str
    user_id: str
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime = field(default_factory=lambda: datetime.now() + timedelta(days=30))
    is_active: bool = True


class AuthService:
    """Servicio de autenticación"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.users: Dict[str, User] = {}  # email -> User
        self.users_by_id: Dict[str, User] = {}  # user_id -> User
        self.sessions: Dict[str, Session] = {}  # session_id -> Session
        logger.info("AuthService initialized")
    
    def register_user(
        self,
        email: str,
        username: str,
        password: str
    ) -> User:
        """Registrar nuevo usuario"""
        if email in self.users:
            raise ValueError("Email already registered")
        
        user_id = f"user_{secrets.token_hex(8)}"
        password_hash = self._hash_password(password)
        
        user = User(
            id=user_id,
            email=email,
            username=username,
            password_hash=password_hash,
        )
        
        self.users[email] = user
        self.users_by_id[user_id] = user
        
        logger.info(f"User registered: {user_id} ({email})")
        return user
    
    def login(self, email: str, password: str) -> Optional[Session]:
        """Iniciar sesión"""
        user = self.users.get(email)
        if not user:
            return None
        
        if not self._verify_password(password, user.password_hash):
            return None
        
        if not user.is_active:
            return None
        
        # Crear sesión
        session_id = secrets.token_urlsafe(32)
        session = Session(
            session_id=session_id,
            user_id=user.id,
        )
        
        self.sessions[session_id] = session
        
        # Actualizar último login
        user.last_login = datetime.now()
        
        logger.info(f"User logged in: {user.id}")
        return session
    
    def logout(self, session_id: str) -> bool:
        """Cerrar sesión"""
        if session_id in self.sessions:
            self.sessions[session_id].is_active = False
            return True
        return False
    
    def verify_session(self, session_id: str) -> Optional[User]:
        """Verificar sesión y obtener usuario"""
        session = self.sessions.get(session_id)
        if not session or not session.is_active:
            return None
        
        if datetime.now() > session.expires_at:
            session.is_active = False
            return None
        
        return self.users_by_id.get(session.user_id)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Obtener usuario por ID"""
        return self.users_by_id.get(user_id)
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> User:
        """Actualizar perfil de usuario"""
        user = self.users_by_id.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        user.profile_data.update(profile_data)
        return user
    
    def change_password(
        self,
        user_id: str,
        old_password: str,
        new_password: str
    ) -> bool:
        """Cambiar contraseña"""
        user = self.users_by_id.get(user_id)
        if not user:
            return False
        
        if not self._verify_password(old_password, user.password_hash):
            return False
        
        user.password_hash = self._hash_password(new_password)
        return True
    
    def _hash_password(self, password: str) -> str:
        """Hash de contraseña (simplificado - usar bcrypt en producción)"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña"""
        return self._hash_password(password) == password_hash




