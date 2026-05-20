"""
Authentication - Sistema de autenticación y autorización
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import jwt
from functools import wraps

logger = logging.getLogger(__name__)


class AuthManager:
    """Gestor de autenticación y autorización"""

    def __init__(self, secret_key: Optional[str] = None, token_expiry: int = 3600):
        """
        Inicializar el gestor de autenticación.

        Args:
            secret_key: Clave secreta para JWT
            token_expiry: Tiempo de expiración del token en segundos
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.token_expiry = token_expiry
        self.users: Dict[str, Dict[str, Any]] = {}

    def create_user(
        self,
        username: str,
        password: str,
        email: Optional[str] = None,
        role: str = "user"
    ) -> Dict[str, Any]:
        """
        Crear un nuevo usuario.

        Args:
            username: Nombre de usuario
            password: Contraseña
            email: Email (opcional)
            role: Rol del usuario

        Returns:
            Información del usuario creado
        """
        user_id = hashlib.sha256(username.encode()).hexdigest()[:16]
        password_hash = self._hash_password(password)
        
        user = {
            "id": user_id,
            "username": username,
            "email": email,
            "password_hash": password_hash,
            "role": role,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        self.users[user_id] = user
        logger.info(f"Usuario creado: {username}")
        return user

    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Autenticar un usuario.

        Args:
            username: Nombre de usuario
            password: Contraseña

        Returns:
            Token JWT o None si falla
        """
        # Buscar usuario
        user = None
        for u in self.users.values():
            if u["username"] == username:
                user = u
                break
        
        if not user:
            return None
        
        # Verificar contraseña
        if not self._verify_password(password, user["password_hash"]):
            return None
        
        if not user.get("active", True):
            return None
        
        # Generar token
        token = self.generate_token(user["id"], user["role"])
        return {
            "token": token,
            "user": {
                "id": user["id"],
                "username": user["username"],
                "email": user.get("email"),
                "role": user["role"]
            }
        }

    def generate_token(self, user_id: str, role: str) -> str:
        """
        Generar token JWT.

        Args:
            user_id: ID del usuario
            role: Rol del usuario

        Returns:
            Token JWT
        """
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": datetime.utcnow() + timedelta(seconds=self.token_expiry),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificar token JWT.

        Args:
            token: Token a verificar

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

    def _hash_password(self, password: str) -> str:
        """Hashear contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()

    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña"""
        return self._hash_password(password) == password_hash

    def has_permission(self, user_role: str, required_role: str) -> bool:
        """
        Verificar si un usuario tiene permisos.

        Args:
            user_role: Rol del usuario
            required_role: Rol requerido

        Returns:
            True si tiene permisos
        """
        role_hierarchy = {
            "admin": 3,
            "moderator": 2,
            "user": 1
        }
        
        user_level = role_hierarchy.get(user_role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        
        return user_level >= required_level


def require_auth(required_role: str = "user"):
    """
    Decorador para requerir autenticación.

    Args:
        required_role: Rol requerido
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from fastapi import HTTPException, Header
            
            # Obtener token del header
            authorization = kwargs.get("authorization") or args[0] if args else None
            if not authorization:
                raise HTTPException(status_code=401, detail="Token requerido")
            
            # Verificar token
            auth_manager = AuthManager()  # En producción, usar instancia compartida
            token_data = auth_manager.verify_token(authorization)
            
            if not token_data:
                raise HTTPException(status_code=401, detail="Token inválido")
            
            # Verificar permisos
            if not auth_manager.has_permission(token_data["role"], required_role):
                raise HTTPException(status_code=403, detail="Permisos insuficientes")
            
            # Agregar información del usuario a kwargs
            kwargs["user_id"] = token_data["user_id"]
            kwargs["user_role"] = token_data["role"]
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator






