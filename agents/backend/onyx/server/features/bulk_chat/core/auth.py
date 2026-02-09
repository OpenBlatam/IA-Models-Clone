"""
Authentication & Authorization - Autenticación y Autorización
============================================================

Sistema de autenticación JWT y autorización basada en roles.
"""

import jwt
import secrets
import hashlib
import logging
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Role(Enum):
    """Roles de usuario."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    MODERATOR = "moderator"


@dataclass
class User:
    """Usuario del sistema."""
    user_id: str
    username: str
    email: Optional[str] = None
    roles: List[Role] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.roles is None:
            self.roles = [Role.USER]
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AuthToken:
    """Token de autenticación."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600
    refresh_token: Optional[str] = None


class AuthManager:
    """Gestor de autenticación y autorización."""
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60,
        refresh_token_expire_days: int = 30,
    ):
        """
        Inicializar gestor de autenticación.
        
        Args:
            secret_key: Clave secreta para JWT (si None, genera una)
            algorithm: Algoritmo de encriptación
            access_token_expire_minutes: Minutos hasta expiración del token
            refresh_token_expire_days: Días hasta expiración del refresh token
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        
        # Usuarios en memoria (en producción usar base de datos)
        self.users: Dict[str, User] = {}
        self.user_passwords: Dict[str, str] = {}  # Hash de contraseñas
    
    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        roles: Optional[List[Role]] = None,
        user_id: Optional[str] = None,
    ) -> User:
        """Crear un nuevo usuario."""
        user_id = user_id or secrets.token_urlsafe(16)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            roles=roles or [Role.USER],
        )
        
        # Hash de contraseña
        password_hash = self._hash_password(password)
        self.user_passwords[user_id] = password_hash
        self.users[user_id] = user
        
        logger.info(f"Created user: {username} ({user_id})")
        return user
    
    def authenticate(self, username: str, password: str) -> Optional[User]:
        """Autenticar usuario con username y password."""
        # Buscar usuario por username
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            return None
        
        # Verificar contraseña
        stored_hash = self.user_passwords.get(user.user_id)
        if not stored_hash or not self._verify_password(password, stored_hash):
            return None
        
        return user
    
    def create_access_token(self, user: User) -> AuthToken:
        """Crear token de acceso."""
        expires_delta = timedelta(minutes=self.access_token_expire_minutes)
        expire = datetime.utcnow() + expires_delta
        
        payload = {
            "sub": user.user_id,
            "username": user.username,
            "roles": [role.value for role in user.roles],
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        
        access_token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Refresh token
        refresh_expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        refresh_payload = {
            "sub": user.user_id,
            "type": "refresh",
            "exp": refresh_expire,
        }
        refresh_token = jwt.encode(refresh_payload, self.secret_key, algorithm=self.algorithm)
        
        return AuthToken(
            access_token=access_token,
            expires_in=self.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
        )
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verificar y decodificar token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Obtener usuario desde token."""
        payload = self.verify_token(token)
        if not payload:
            return None
        
        user_id = payload.get("sub")
        return self.users.get(user_id)
    
    def has_role(self, user: User, role: Role) -> bool:
        """Verificar si usuario tiene un rol."""
        return role in user.roles
    
    def has_any_role(self, user: User, roles: List[Role]) -> bool:
        """Verificar si usuario tiene alguno de los roles."""
        return any(role in user.roles for role in roles)
    
    def _hash_password(self, password: str) -> str:
        """Hash de contraseña."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña."""
        return self._hash_password(password) == password_hash
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Obtener usuario por ID."""
        return self.users.get(user_id)
    
    def update_user_roles(self, user_id: str, roles: List[Role]):
        """Actualizar roles de usuario."""
        user = self.users.get(user_id)
        if user:
            user.roles = roles
            logger.info(f"Updated roles for user {user_id}: {[r.value for r in roles]}")


class AuthMiddleware:
    """Middleware para autenticación."""
    
    def __init__(self, auth_manager: AuthManager):
        self.auth_manager = auth_manager
    
    async def __call__(self, request, call_next):
        """Procesar request con autenticación."""
        # Extraer token del header
        auth_header = request.headers.get("Authorization")
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            user = self.auth_manager.get_user_from_token(token)
            
            if user:
                request.state.user = user
                request.state.authenticated = True
            else:
                request.state.user = None
                request.state.authenticated = False
        else:
            request.state.user = None
            request.state.authenticated = False
        
        response = await call_next(request)
        return response
































