"""
OAuth2 Security - Sistema de seguridad OAuth2 para FastAPI
==========================================================

Implementa:
- OAuth2 con JWT tokens
- Password hashing
- Token refresh
- Role-based access control (RBAC)
- API key authentication
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, APIKeyHeader
from pydantic import BaseModel, EmailStr
import secrets

logger = logging.getLogger(__name__)

# Configuración
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    scopes={
        "read": "Read access",
        "write": "Write access",
        "admin": "Admin access"
    }
)

# API Key scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


class Token(BaseModel):
    """Modelo de token"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


class TokenData(BaseModel):
    """Datos del token"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []


class User(BaseModel):
    """Modelo de usuario"""
    id: str
    username: str
    email: EmailStr
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []
    scopes: List[str] = []


class UserCreate(BaseModel):
    """Modelo para crear usuario"""
    username: str
    email: EmailStr
    password: str
    full_name: Optional[str] = None


class UserInDB(User):
    """Usuario en base de datos (con password hasheado)"""
    hashed_password: str


class OAuth2Security:
    """Sistema de seguridad OAuth2"""
    
    def __init__(self, secret_key: str = SECRET_KEY, algorithm: str = ALGORITHM):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.users_db: Dict[str, UserInDB] = {}
        self.api_keys: Dict[str, Dict] = {}
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica una contraseña"""
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        """Hashea una contraseña"""
        return pwd_context.hash(password)
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Crea un access token JWT"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire, "type": "access"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: dict) -> str:
        """Crea un refresh token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def decode_token(self, token: str) -> Optional[TokenData]:
        """Decodifica un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])
            
            if username is None:
                return None
            
            return TokenData(username=username, user_id=user_id, scopes=scopes)
        except JWTError:
            return None
    
    def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Autentica un usuario"""
        user = self.users_db.get(username)
        if not user:
            return None
        if not self.verify_password(password, user.hashed_password):
            return None
        return user
    
    def create_user(self, user_data: UserCreate) -> UserInDB:
        """Crea un nuevo usuario"""
        hashed_password = self.get_password_hash(user_data.password)
        user_id = secrets.token_urlsafe(16)
        
        user = UserInDB(
            id=user_id,
            username=user_data.username,
            email=user_data.email,
            full_name=user_data.full_name,
            hashed_password=hashed_password,
            disabled=False,
            roles=["user"],
            scopes=["read"]
        )
        
        self.users_db[user_data.username] = user
        return user
    
    def generate_api_key(self, user_id: str, name: str = "default") -> str:
        """Genera una API key para un usuario"""
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = {
            "user_id": user_id,
            "name": name,
            "created_at": datetime.utcnow().isoformat()
        }
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict]:
        """Verifica una API key"""
        return self.api_keys.get(api_key)


# Instancia global
oauth2_security = OAuth2Security()


# Dependencies para FastAPI
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Obtiene el usuario actual desde el token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    token_data = oauth2_security.decode_token(token)
    if token_data is None:
        raise credentials_exception
    
    user = oauth2_security.users_db.get(token_data.username)
    if user is None:
        raise credentials_exception
    
    return User(**user.model_dump(exclude={"hashed_password"}))


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Obtiene el usuario activo actual"""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def require_scope(required_scope: str):
    """Dependency para requerir un scope específico"""
    async def scope_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if required_scope not in current_user.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required scope: {required_scope}"
            )
        return current_user
    return scope_checker


def require_role(required_role: str):
    """Dependency para requerir un rol específico"""
    async def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if required_role not in current_user.roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Not enough permissions. Required role: {required_role}"
            )
        return current_user
    return role_checker


async def get_api_key_user(api_key: Optional[str] = Security(api_key_header)) -> Optional[User]:
    """Obtiene usuario desde API key"""
    if not api_key:
        return None
    
    key_data = oauth2_security.verify_api_key(api_key)
    if not key_data:
        return None
    
    # Buscar usuario por user_id
    for user in oauth2_security.users_db.values():
        if user.id == key_data["user_id"]:
            return User(**user.model_dump(exclude={"hashed_password"}))
    
    return None


async def authenticate_user_or_api_key(
    current_user: Optional[User] = Depends(get_current_active_user),
    api_key_user: Optional[User] = Depends(get_api_key_user)
) -> User:
    """Autentica usuario por token o API key"""
    if current_user:
        return current_user
    if api_key_user:
        return api_key_user
    
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required"
    )

