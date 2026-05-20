"""
Sistema de autenticación básico
"""

import hashlib
import secrets
import jwt
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import json


@dataclass
class User:
    """Usuario del sistema"""
    id: str
    email: str
    password_hash: str
    created_at: str
    last_login: Optional[str] = None
    is_active: bool = True
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convierte a diccionario (sin password)"""
        return {
            "id": self.id,
            "email": self.email,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "metadata": self.metadata or {}
        }


class AuthManager:
    """Gestor de autenticación"""
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        Inicializa el gestor de autenticación
        
        Args:
            secret_key: Clave secreta para JWT (si None, se genera)
        """
        self.secret_key = secret_key or secrets.token_urlsafe(32)
        self.users: Dict[str, User] = {}
        self.token_blacklist: set = set()
    
    def hash_password(self, password: str) -> str:
        """
        Hashea una contraseña
        
        Args:
            password: Contraseña en texto plano
            
        Returns:
            Hash de la contraseña
        """
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verifica una contraseña
        
        Args:
            password: Contraseña en texto plano
            password_hash: Hash de la contraseña
            
        Returns:
            True si la contraseña es correcta
        """
        return self.hash_password(password) == password_hash
    
    def create_user(self, email: str, password: str,
                   metadata: Optional[Dict] = None) -> User:
        """
        Crea un nuevo usuario
        
        Args:
            email: Email del usuario
            password: Contraseña
            metadata: Metadatos adicionales
            
        Returns:
            Usuario creado
        """
        # Verificar si el email ya existe
        for user in self.users.values():
            if user.email == email:
                raise ValueError("El email ya está registrado")
        
        # Generar ID
        user_id = hashlib.md5(f"{email}{datetime.now().isoformat()}".encode()).hexdigest()
        
        # Crear usuario
        user = User(
            id=user_id,
            email=email,
            password_hash=self.hash_password(password),
            created_at=datetime.now().isoformat(),
            is_active=True,
            metadata=metadata or {}
        )
        
        self.users[user_id] = user
        return user
    
    def authenticate(self, email: str, password: str) -> Optional[str]:
        """
        Autentica un usuario y retorna token
        
        Args:
            email: Email del usuario
            password: Contraseña
            
        Returns:
            Token JWT o None si falla
        """
        # Buscar usuario
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break
        
        if not user:
            return None
        
        if not user.is_active:
            return None
        
        if not self.verify_password(password, user.password_hash):
            return None
        
        # Actualizar último login
        user.last_login = datetime.now().isoformat()
        
        # Generar token
        token = self.generate_token(user.id)
        return token
    
    def generate_token(self, user_id: str, expires_in: int = 3600) -> str:
        """
        Genera un token JWT
        
        Args:
            user_id: ID del usuario
            expires_in: Tiempo de expiración en segundos
            
        Returns:
            Token JWT
        """
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_token(self, token: str) -> Optional[str]:
        """
        Verifica un token JWT
        
        Args:
            token: Token a verificar
            
        Returns:
            user_id si es válido, None si no
        """
        try:
            # Verificar si está en blacklist
            if token in self.token_blacklist:
                return None
            
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload.get("user_id")
        
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def revoke_token(self, token: str):
        """Revoca un token (lo agrega a blacklist)"""
        self.token_blacklist.add(token)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Obtiene un usuario por ID"""
        return self.users.get(user_id)
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Obtiene un usuario por email"""
        for user in self.users.values():
            if user.email == email:
                return user
        return None






