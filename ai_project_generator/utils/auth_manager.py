"""
Auth Manager - Gestor de Autenticación
======================================

Gestiona autenticación y autorización para la API.
"""

import logging
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt

logger = logging.getLogger(__name__)


class AuthManager:
    """Gestor de autenticación"""

    def __init__(self, secret_key: Optional[str] = None):
        """
        Inicializa el gestor de autenticación.

        Args:
            secret_key: Clave secreta para JWT (si no se proporciona, se genera)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, Dict[str, Any]] = {}
        self.api_keys: Dict[str, Dict[str, Any]] = {}

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: str = "user",
    ) -> Dict[str, Any]:
        """
        Crea un nuevo usuario.

        Args:
            username: Nombre de usuario
            password: Contraseña
            email: Email (opcional)
            role: Rol del usuario (user, admin)

        Returns:
            Información del usuario creado
        """
        user_id = hashlib.sha256(username.encode()).hexdigest()[:16]
        password_hash = hashlib.sha256(password.encode()).hexdigest()

        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "active": True,
        }

        self.users[username] = user
        logger.info(f"Usuario creado: {username}")

        return {
            "user_id": user_id,
            "username": username,
            "role": role,
        }

    def authenticate_user(self, username: str, password: str) -> Optional[str]:
        """
        Autentica un usuario y retorna un token JWT.

        Args:
            username: Nombre de usuario
            password: Contraseña

        Returns:
            Token JWT o None si falla
        """
        user = self.users.get(username)
        if not user or not user.get("active"):
            return None

        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if user["password_hash"] != password_hash:
            return None

        # Generar token JWT
        payload = {
            "user_id": user["id"],
            "username": username,
            "role": user["role"],
            "exp": datetime.utcnow() + timedelta(hours=24),
            "iat": datetime.utcnow(),
        }

        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verifica un token JWT.

        Args:
            token: Token JWT

        Returns:
            Payload del token o None si es inválido
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

    def create_api_key(
        self,
        name: str,
        user_id: Optional[str] = None,
        permissions: Optional[list] = None,
    ) -> str:
        """
        Crea una API key.

        Args:
            name: Nombre de la API key
            user_id: ID del usuario (opcional)
            permissions: Permisos (opcional)

        Returns:
            API key generada
        """
        api_key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        self.api_keys[key_hash] = {
            "name": name,
            "user_id": user_id,
            "permissions": permissions or ["read", "write"],
            "created_at": datetime.now().isoformat(),
            "active": True,
        }

        logger.info(f"API key creada: {name}")
        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Verifica una API key.

        Args:
            api_key: API key

        Returns:
            Información de la API key o None
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        key_info = self.api_keys.get(key_hash)

        if not key_info or not key_info.get("active"):
            return None

        return key_info


