"""
Servicio de autenticación y seguridad
"""

from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
from config.settings import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """Servicio de autenticación y autorización"""
    
    def __init__(self):
        """Inicializa el servicio de autenticación"""
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def hash_password(self, password: str) -> str:
        """
        Hashea una contraseña
        
        Args:
            password: Contraseña en texto plano
        
        Returns:
            Contraseña hasheada
        """
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        Verifica una contraseña
        
        Args:
            plain_password: Contraseña en texto plano
            hashed_password: Contraseña hasheada
        
        Returns:
            True si la contraseña es correcta
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """
        Crea un token JWT
        
        Args:
            data: Datos a incluir en el token
            expires_delta: Tiempo de expiración (opcional)
        
        Returns:
            Token JWT
        """
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[dict]:
        """
        Decodifica un token JWT
        
        Args:
            token: Token JWT
        
        Returns:
            Datos decodificados o None si el token es inválido
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.JWTError:
            return None
    
    def verify_token(self, token: str) -> bool:
        """
        Verifica si un token es válido
        
        Args:
            token: Token JWT
        
        Returns:
            True si el token es válido
        """
        payload = self.decode_token(token)
        return payload is not None

