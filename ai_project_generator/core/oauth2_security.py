"""
OAuth2 Security - Implementación de OAuth2 para autenticación segura
====================================================================

Implementación de OAuth2 siguiendo mejores prácticas de seguridad
para APIs FastAPI en entornos de microservicios.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2AuthorizationCodeBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from .microservices_config import get_microservices_config

logger = logging.getLogger(__name__)

# Configuración
config = get_microservices_config()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 schemes
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="api/v1/auth/login",
    auto_error=False
)

oauth2_code_scheme = OAuth2AuthorizationCodeBearer(
    authorizationUrl="api/v1/auth/authorize",
    tokenUrl="api/v1/auth/token",
    auto_error=False
)


class TokenData(BaseModel):
    """Datos del token"""
    username: Optional[str] = None
    user_id: Optional[str] = None
    scopes: List[str] = []


class OAuth2Provider:
    """Proveedor OAuth2"""
    
    def __init__(
        self,
        provider: str,
        client_id: str,
        client_secret: str,
        authorization_url: str,
        token_url: str,
        userinfo_url: Optional[str] = None
    ):
        self.provider = provider
        self.client_id = client_id
        self.client_secret = client_secret
        self.authorization_url = authorization_url
        self.token_url = token_url
        self.userinfo_url = userinfo_url
    
    async def get_authorization_url(self, redirect_uri: str, state: str) -> str:
        """Obtiene URL de autorización"""
        params = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "state": state,
            "scope": "openid profile email"
        }
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.authorization_url}?{query_string}"
    
    async def exchange_code_for_token(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Intercambia código por token"""
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Obtiene información del usuario"""
        if not self.userinfo_url:
            return {}
        
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.userinfo_url,
                headers={"Authorization": f"Bearer {access_token}"}
            )
            response.raise_for_status()
            return response.json()


class OAuth2Manager:
    """Manager para OAuth2"""
    
    def __init__(self):
        self.secret_key = "your-secret-key-here"  # Debe venir de configuración
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.providers: Dict[str, OAuth2Provider] = {}
        
        if config.oauth2_enabled and config.oauth2_provider:
            self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa proveedores OAuth2"""
        provider_name = config.oauth2_provider.lower()
        
        if provider_name == "google":
            self.providers["google"] = OAuth2Provider(
                provider="google",
                client_id="",  # Debe venir de env vars
                client_secret="",
                authorization_url="https://accounts.google.com/o/oauth2/v2/auth",
                token_url="https://oauth2.googleapis.com/token",
                userinfo_url="https://www.googleapis.com/oauth2/v2/userinfo"
            )
        elif provider_name == "github":
            self.providers["github"] = OAuth2Provider(
                provider="github",
                client_id="",
                client_secret="",
                authorization_url="https://github.com/login/oauth/authorize",
                token_url="https://github.com/login/oauth/access_token",
                userinfo_url="https://api.github.com/user"
            )
        elif provider_name == "auth0":
            # Auth0 requiere configuración específica del tenant
            logger.info("Auth0 provider requires tenant configuration")
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Crea token de acceso JWT"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> TokenData:
        """Verifica y decodifica token"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username: str = payload.get("sub")
            user_id: str = payload.get("user_id")
            scopes: List[str] = payload.get("scopes", [])
            
            if username is None:
                raise credentials_exception
            
            return TokenData(username=username, user_id=user_id, scopes=scopes)
        except JWTError:
            raise credentials_exception
    
    def hash_password(self, password: str) -> str:
        """Hashea contraseña"""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verifica contraseña"""
        return pwd_context.verify(plain_password, hashed_password)


# Instancia global
_oauth2_manager: Optional[OAuth2Manager] = None


def get_oauth2_manager() -> OAuth2Manager:
    """Obtiene instancia de OAuth2 manager"""
    global _oauth2_manager
    if _oauth2_manager is None:
        _oauth2_manager = OAuth2Manager()
    return _oauth2_manager


async def get_current_user(token: str = Depends(oauth2_scheme)) -> TokenData:
    """Dependency para obtener usuario actual"""
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    manager = get_oauth2_manager()
    return manager.verify_token(token)


async def get_current_active_user(current_user: TokenData = Depends(get_current_user)) -> TokenData:
    """Dependency para obtener usuario activo"""
    # Aquí se puede agregar lógica adicional para verificar si el usuario está activo
    return current_user


def require_scopes(*required_scopes: str):
    """Decorator para requerir scopes específicos"""
    def decorator(func):
        async def wrapper(*args, current_user: TokenData = Depends(get_current_user), **kwargs):
            user_scopes = set(current_user.scopes)
            required = set(required_scopes)
            
            if not required.issubset(user_scopes):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Insufficient permissions. Required: {required_scopes}"
                )
            
            return await func(*args, current_user=current_user, **kwargs)
        
        return wrapper
    return decorator















