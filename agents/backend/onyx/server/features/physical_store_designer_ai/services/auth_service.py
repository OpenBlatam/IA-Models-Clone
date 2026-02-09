"""
Auth Service - Sistema de autenticación y usuarios
"""

import logging
import hashlib
import secrets
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False
    logger.warning("PyJWT no disponible, usando tokens simples")

logger = logging.getLogger(__name__)


class AuthService:
    """Servicio para autenticación y gestión de usuarios"""
    
    def __init__(self, secret_key: Optional[str] = None):
        self.secret_key = secret_key or "default-secret-key-change-in-production"
        self.users: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
    
    def register_user(
        self,
        email: str,
        password: str,
        name: str,
        role: str = "user"
    ) -> Dict[str, Any]:
        """Registrar nuevo usuario"""
        
        if email in self.users:
            raise ValueError("Usuario ya existe")
        
        # Hash de contraseña
        password_hash = self._hash_password(password)
        
        user = {
            "user_id": self._generate_user_id(),
            "email": email,
            "password_hash": password_hash,
            "name": name,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "designs_count": 0
        }
        
        self.users[email] = user
        
        logger.info(f"Usuario registrado: {email}")
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"]
        }
    
    def authenticate(
        self,
        email: str,
        password: str
    ) -> Optional[Dict[str, Any]]:
        """Autenticar usuario"""
        
        user = self.users.get(email)
        
        if not user:
            return None
        
        if not user.get("is_active", True):
            return None
        
        if not self._verify_password(password, user["password_hash"]):
            return None
        
        # Crear token
        token = self._generate_token(user["user_id"], user["email"])
        
        # Crear sesión
        session_id = self._create_session(user["user_id"], token)
        
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
            "token": token,
            "session_id": session_id
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verificar token"""
        if JWT_AVAILABLE:
            try:
                payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
                return payload
            except jwt.ExpiredSignatureError:
                return None
            except jwt.InvalidTokenError:
                return None
        else:
            # Verificación simple
            parts = token.split(":")
            if len(parts) == 3:
                return {"user_id": parts[0], "email": parts[1]}
            return None
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtener usuario por ID"""
        for user in self.users.values():
            if user["user_id"] == user_id:
                return {
                    "user_id": user["user_id"],
                    "email": user["email"],
                    "name": user["name"],
                    "role": user["role"],
                    "created_at": user["created_at"],
                    "designs_count": user.get("designs_count", 0)
                }
        return None
    
    def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Actualizar usuario"""
        for email, user in self.users.items():
            if user["user_id"] == user_id:
                # No permitir actualizar email o password directamente
                allowed_updates = ["name", "role"]
                for key, value in updates.items():
                    if key in allowed_updates:
                        user[key] = value
                return True
        return False
    
    def _hash_password(self, password: str) -> str:
        """Hash de contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verificar contraseña"""
        return self._hash_password(password) == password_hash
    
    def _generate_user_id(self) -> str:
        """Generar ID de usuario"""
        return f"user_{secrets.token_hex(8)}"
    
    def _generate_token(self, user_id: str, email: str) -> str:
        """Generar JWT token"""
        if JWT_AVAILABLE:
            payload = {
                "user_id": user_id,
                "email": email,
                "exp": datetime.utcnow() + timedelta(days=7),
                "iat": datetime.utcnow()
            }
            return jwt.encode(payload, self.secret_key, algorithm="HS256")
        else:
            # Token simple si JWT no está disponible
            return f"{user_id}:{email}:{secrets.token_hex(16)}"
    
    def _create_session(self, user_id: str, token: str) -> str:
        """Crear sesión"""
        session_id = f"session_{secrets.token_hex(16)}"
        
        self.sessions[session_id] = {
            "user_id": user_id,
            "token": token,
            "created_at": datetime.now().isoformat(),
            "expires_at": (datetime.now() + timedelta(days=7)).isoformat(),
            "is_active": True
        }
        
        return session_id
    
    def logout(self, session_id: str) -> bool:
        """Cerrar sesión"""
        if session_id in self.sessions:
            self.sessions[session_id]["is_active"] = False
            return True
        return False

