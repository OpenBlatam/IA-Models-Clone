"""
Auth Service - Servicio de Autenticación
=========================================

Sistema de autenticación y autorización.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import jwt
from ..utils.security import SecurityUtils

logger = logging.getLogger(__name__)


class AuthService:
    """Servicio de autenticación"""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Inicializar servicio de autenticación
        
        Args:
            secret_key: Clave secreta para JWT (opcional, usa env var)
        """
        import os
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.algorithm = "HS256"
        self.token_expiry = timedelta(hours=24)
        logger.info("Auth Service inicializado")
    
    def create_token(
        self,
        user_id: str,
        email: Optional[str] = None,
        roles: Optional[list] = None
    ) -> str:
        """
        Crear token JWT
        
        Args:
            user_id: ID del usuario
            email: Email del usuario
            roles: Lista de roles
            
        Returns:
            Token JWT
        """
        payload = {
            "user_id": user_id,
            "email": email,
            "roles": roles or ["user"],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token creado para usuario {user_id}")
        return token
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificar token JWT
        
        Args:
            token: Token a verificar
            
        Returns:
            Payload del token o None si es inválido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token inválido")
            return None
    
    def hash_password(self, password: str) -> tuple:
        """
        Hashear contraseña
        
        Args:
            password: Contraseña en texto plano
            
        Returns:
            Tuple (hash, salt)
        """
        return SecurityUtils.hash_password(password)
    
    def verify_password(self, password: str, hash_value: str, salt: str) -> bool:
        """
        Verificar contraseña
        
        Args:
            password: Contraseña a verificar
            hash_value: Hash almacenado
            salt: Salt usado
            
        Returns:
            True si la contraseña es correcta
        """
        return SecurityUtils.verify_password(password, hash_value, salt)
    
    def has_role(self, token_payload: Dict[str, Any], required_role: str) -> bool:
        """
        Verificar si el usuario tiene un rol
        
        Args:
            token_payload: Payload del token
            required_role: Rol requerido
            
        Returns:
            True si tiene el rol
        """
        roles = token_payload.get("roles", [])
        return required_role in roles or "admin" in roles
    
    def refresh_token(self, token: str) -> Optional[str]:
        """
        Refrescar token JWT
        
        Args:
            token: Token actual
            
        Returns:
            Nuevo token o None si el token es inválido
        """
        payload = self.verify_token(token)
        if not payload:
            return None
        
        return self.create_token(
            user_id=payload.get("user_id"),
            email=payload.get("email"),
            roles=payload.get("roles")
        )
    
    def revoke_token(self, token: str) -> bool:
        """
        Revocar token (agregar a lista negra)
        
        Args:
            token: Token a revocar
            
        Returns:
            True si se revocó exitosamente
        """
        if not hasattr(self, 'revoked_tokens'):
            self.revoked_tokens = set()
        
        payload = self.verify_token(token)
        if payload:
            jti = payload.get("jti") or token
            self.revoked_tokens.add(jti)
            logger.info(f"Token revocado: {jti}")
            return True
        
        return False
    
    def is_token_revoked(self, token: str) -> bool:
        """
        Verificar si un token está revocado
        
        Args:
            token: Token a verificar
            
        Returns:
            True si está revocado
        """
        if not hasattr(self, 'revoked_tokens'):
            return False
        
        payload = self.verify_token(token)
        if payload:
            jti = payload.get("jti") or token
            return jti in self.revoked_tokens
        
        return False
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Verificar token JWT
        
        Args:
            token: Token a verificar
            
        Returns:
            Payload del token o None si es inválido
        """
        if self.is_token_revoked(token):
            logger.warning("Token revocado")
            return None
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token expirado")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Token inválido")
            return None
    
    def create_token(
        self,
        user_id: str,
        email: Optional[str] = None,
        roles: Optional[list] = None,
        jti: Optional[str] = None
    ) -> str:
        """
        Crear token JWT
        
        Args:
            user_id: ID del usuario
            email: Email del usuario
            roles: Lista de roles
            jti: JWT ID (opcional, para revocación)
            
        Returns:
            Token JWT
        """
        import uuid
        
        payload = {
            "user_id": user_id,
            "email": email,
            "roles": roles or ["user"],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "jti": jti or str(uuid.uuid4())
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.info(f"Token creado para usuario {user_id}")
        return token



