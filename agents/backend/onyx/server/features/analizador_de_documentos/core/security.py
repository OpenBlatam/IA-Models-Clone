"""
Sistema de Seguridad
=====================

Sistema de autenticación y autorización.
"""

import logging
import hashlib
import secrets
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger(__name__)


@dataclass
class User:
    """Usuario"""
    username: str
    password_hash: str
    roles: List[str]
    api_key: Optional[str] = None
    created_at: str = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class SecurityManager:
    """
    Gestor de seguridad
    
    Proporciona:
    - Autenticación básica
    - Generación de API keys
    - Tokens JWT
    - Control de acceso basado en roles
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """Inicializar gestor"""
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> username
        self.token_cache: Dict[str, Dict] = {}
        logger.info("SecurityManager inicializado")
    
    def hash_password(self, password: str) -> str:
        """Hashear contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(
        self,
        username: str,
        password: str,
        roles: List[str] = None
    ) -> User:
        """
        Crear usuario
        
        Args:
            username: Nombre de usuario
            password: Contraseña
            roles: Roles del usuario
        
        Returns:
            Usuario creado
        """
        user = User(
            username=username,
            password_hash=self.hash_password(password),
            roles=roles or ["user"]
        )
        
        self.users[username] = user
        logger.info(f"Usuario creado: {username}")
        
        return user
    
    def generate_api_key(self, username: str) -> str:
        """
        Generar API key para usuario
        
        Args:
            username: Nombre de usuario
        
        Returns:
            API key
        """
        if username not in self.users:
            raise ValueError(f"Usuario no encontrado: {username}")
        
        api_key = secrets.token_urlsafe(32)
        self.users[username].api_key = api_key
        self.api_keys[api_key] = username
        
        logger.info(f"API key generada para {username}")
        return api_key
    
    def authenticate(
        self,
        username: str,
        password: str
    ) -> Optional[User]:
        """
        Autenticar usuario
        
        Args:
            username: Nombre de usuario
            password: Contraseña
        
        Returns:
            Usuario si autenticación exitosa
        """
        if username not in self.users:
            return None
        
        user = self.users[username]
        password_hash = self.hash_password(password)
        
        if user.password_hash == password_hash:
            return user
        
        return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Autenticar con API key
        
        Args:
            api_key: API key
        
        Returns:
            Usuario si autenticación exitosa
        """
        username = self.api_keys.get(api_key)
        if username:
            return self.users.get(username)
        return None
    
    def generate_token(
        self,
        user: User,
        expires_in: int = 3600
    ) -> str:
        """
        Generar token JWT
        
        Args:
            user: Usuario
            expires_in: Tiempo de expiración en segundos
        
        Returns:
            Token JWT
        """
        payload = {
            "username": user.username,
            "roles": user.roles,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """
        Verificar token JWT
        
        Args:
            token: Token JWT
        
        Returns:
            Payload si válido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token inválido")
            return None
    
    def has_role(self, user: User, role: str) -> bool:
        """Verificar si usuario tiene rol"""
        return role in user.roles
    
    def has_any_role(self, user: User, roles: List[str]) -> bool:
        """Verificar si usuario tiene alguno de los roles"""
        return any(role in user.roles for role in roles)


# Instancia global
_security_manager: Optional[SecurityManager] = None


def get_security_manager() -> SecurityManager:
    """Obtener instancia global del gestor"""
    global _security_manager
    if _security_manager is None:
        _security_manager = SecurityManager()
    return _security_manager
















