"""
Servicio de autenticación y autorización
"""

import hashlib
import secrets
import jwt
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class AuthService:
    """Servicio de autenticación simple"""
    
    def __init__(self, secret_key: Optional[str] = None, storage_path: Optional[Path] = None):
        self.secret_key = secret_key or "your-secret-key-change-in-production"
        self.storage_path = storage_path or Path("./data/auth")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.users_file = self.storage_path / "users.json"
        self.logger = logger
        self._load_users()
    
    def _load_users(self) -> None:
        """Carga usuarios desde archivo"""
        if self.users_file.exists():
            try:
                with open(self.users_file, "r", encoding="utf-8") as f:
                    self.users = json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading users: {e}")
                self.users = {}
        else:
            self.users = {}
    
    def _save_users(self) -> None:
        """Guarda usuarios en archivo"""
        try:
            with open(self.users_file, "w", encoding="utf-8") as f:
                json.dump(self.users, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Error saving users: {e}")
    
    def _hash_password(self, password: str) -> str:
        """Hashea una contraseña"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def register_user(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Registra un nuevo usuario"""
        if username in self.users:
            raise ValueError("Usuario ya existe")
        
        user_id = f"user_{len(self.users)}_{secrets.token_hex(8)}"
        
        self.users[username] = {
            "user_id": user_id,
            "username": username,
            "email": email,
            "password_hash": self._hash_password(password),
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "role": "user"
        }
        
        self._save_users()
        self.logger.info(f"User registered: {username}")
        
        return {
            "user_id": user_id,
            "username": username,
            "email": email
        }
    
    def authenticate(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Autentica un usuario"""
        user = self.users.get(username)
        
        if not user:
            return None
        
        if not user.get("is_active"):
            return None
        
        password_hash = self._hash_password(password)
        
        if user["password_hash"] != password_hash:
            return None
        
        return {
            "user_id": user["user_id"],
            "username": user["username"],
            "email": user["email"],
            "role": user.get("role", "user")
        }
    
    def generate_token(self, user_data: Dict[str, Any], expires_in: int = 3600) -> str:
        """Genera un token JWT"""
        payload = {
            "user_id": user_data["user_id"],
            "username": user_data["username"],
            "email": user_data["email"],
            "role": user_data.get("role", "user"),
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verifica un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene información de un usuario"""
        for user in self.users.values():
            if user["user_id"] == user_id:
                return {
                    "user_id": user["user_id"],
                    "username": user["username"],
                    "email": user["email"],
                    "role": user.get("role", "user"),
                    "created_at": user.get("created_at")
                }
        return None

