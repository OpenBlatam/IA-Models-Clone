"""
OAuth2 - Autenticación y Autorización
======================================

Implementación de OAuth2 para seguridad de API.
"""

import os
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Configuración
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

# Base de datos de usuarios (en producción usar DB real)
_users_db: Dict[str, Dict[str, Any]] = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("admin"),  # Cambiar en producción
        "email": "admin@example.com",
        "roles": ["admin", "user"],
        "disabled": False
    }
}


class User:
    """Modelo de usuario."""
    
    def __init__(self, username: str, email: str, roles: List[str], disabled: bool = False):
        self.username = username
        self.email = email
        self.roles = roles
        self.disabled = disabled


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verificar contraseña."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hashear contraseña."""
    return pwd_context.hash(password)


def get_user(username: str) -> Optional[User]:
    """Obtener usuario de la base de datos."""
    user_data = _users_db.get(username)
    if not user_data:
        return None
    
    return User(
        username=user_data["username"],
        email=user_data["email"],
        roles=user_data["roles"],
        disabled=user_data["disabled"]
    )


def authenticate_user(username: str, password: str) -> Optional[User]:
    """Autenticar usuario."""
    user = get_user(username)
    if not user:
        return None
    
    if user.disabled:
        return None
    
    user_data = _users_db[username]
    if not verify_password(password, user_data["hashed_password"]):
        return None
    
    return user


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Crear token de acceso JWT."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Optional[str] = Depends(oauth2_scheme)) -> User:
    """Obtener usuario actual desde token."""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = get_user(username)
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Obtener usuario activo."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_role(required_role: str):
    """Dependency para requerir un rol específico."""
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires role: {required_role}"
            )
        return current_user
    
    return role_checker


def require_any_role(*roles: str):
    """Dependency para requerir cualquiera de los roles especificados."""
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not any(role in current_user.roles for role in roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Requires one of roles: {', '.join(roles)}"
            )
        return current_user
    
    return role_checker


# Funciones de utilidad para gestión de usuarios
def create_user(username: str, password: str, email: str, roles: List[str] = None) -> User:
    """Crear nuevo usuario."""
    if username in _users_db:
        raise ValueError(f"User {username} already exists")
    
    if roles is None:
        roles = ["user"]
    
    _users_db[username] = {
        "username": username,
        "hashed_password": get_password_hash(password),
        "email": email,
        "roles": roles,
        "disabled": False
    }
    
    logger.info(f"User {username} created")
    return get_user(username)


def update_user_roles(username: str, roles: List[str]) -> None:
    """Actualizar roles de usuario."""
    if username not in _users_db:
        raise ValueError(f"User {username} not found")
    
    _users_db[username]["roles"] = roles
    logger.info(f"Roles updated for {username}: {roles}")


def disable_user(username: str) -> None:
    """Deshabilitar usuario."""
    if username not in _users_db:
        raise ValueError(f"User {username} not found")
    
    _users_db[username]["disabled"] = True
    logger.info(f"User {username} disabled")




