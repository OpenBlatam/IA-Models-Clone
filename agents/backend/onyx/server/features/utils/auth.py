from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from dataclasses import dataclass

# Constants
TIMEOUT_SECONDS = 60

import os
from fastapi import Header, HTTPException, status, Security, Depends, Form, Request
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from .schemas import TokenResponse, RefreshTokenRequest
from .logging_utils import logger

from typing import Any, List, Dict, Optional
import logging
import asyncio
OAUTH2_SECRET: str = os.getenv('LLM_OAUTH2_SECRET', 'supersecret')
OAUTH2_ALG: str = os.getenv('LLM_OAUTH2_ALG', 'HS256')
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv('LLM_ACCESS_TOKEN_MINUTES', '15'))
REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv('LLM_REFRESH_TOKEN_DAYS', '7'))
API_KEY: str = os.getenv('LLM_API_KEY', 'changeme')
AUTH_METHOD: str = os.getenv('LLM_AUTH_METHOD', 'api_key')  # 'api_key', 'oauth2', 'both'
DEFAULT_SCOPE: str = "llm:predict"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
http_bearer = HTTPBearer()

@dataclass
class AuthError:
    INVALID_API_KEY = "Invalid API Key"
    MISSING_BEARER = "Missing Bearer token"
    INVALID_BEARER = "Invalid Bearer token"
    INVALID_CREDENTIALS = "Invalid credentials"
    INVALID_REFRESH = "Invalid refresh token"
    INSUFFICIENT_SCOPE = "Insufficient scope"
    SERVER_ERROR = "Invalid AUTH_METHOD config"


def _create_access_token(data: Dict[str, Any], expires_delta: timedelta, scopes: List[str]) -> str:
    """Genera un JWT de acceso con scopes."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire, "scopes": scopes})
    return jwt.encode(to_encode, OAUTH2_SECRET, algorithm=OAUTH2_ALG)

def _create_refresh_token(data: Dict[str, Any], expires_delta: timedelta) -> str:
    """Genera un JWT de refresh."""
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, OAUTH2_SECRET, algorithm=OAUTH2_ALG)

def _get_user_from_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, OAUTH2_SECRET, algorithms=[OAUTH2_ALG])
        return payload.get("sub")
    except JWTError:
        return None

def _get_scopes_from_token(token: str) -> List[str]:
    try:
        payload = jwt.decode(token, OAUTH2_SECRET, algorithms=[OAUTH2_ALG])
        return payload.get("scopes", [])
    except JWTError:
        return []

def check_auth(
    x_api_key: Optional[str] = Header(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Security(http_bearer, auto_error=False)
) -> Dict[str, Any]:
    """Valida API Key o JWT según AUTH_METHOD. Devuelve usuario y scopes."""
    if AUTH_METHOD == 'api_key':
        if x_api_key != API_KEY:
            logger.warning({"event": "auth_failed", "method": "api_key"})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=AuthError.INVALID_API_KEY)
        return {"user": "api_key_user", "scopes": [DEFAULT_SCOPE]}
    elif AUTH_METHOD == 'oauth2':
        if not credentials or not credentials.credentials:
            logger.warning({"event": "auth_failed", "method": "oauth2", "reason": "missing_token"})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=AuthError.MISSING_BEARER)
        user = _get_user_from_token(credentials.credentials)
        scopes = _get_scopes_from_token(credentials.credentials)
        if not user:
            logger.warning({"event": "auth_failed", "method": "oauth2", "reason": "invalid_token"})
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=AuthError.INVALID_BEARER)
        return {"user": user, "scopes": scopes}
    elif AUTH_METHOD == 'both':
        if x_api_key == API_KEY:
            return {"user": "api_key_user", "scopes": [DEFAULT_SCOPE]}
        if credentials and credentials.credentials:
            user = _get_user_from_token(credentials.credentials)
            scopes = _get_scopes_from_token(credentials.credentials)
            if user:
                return {"user": user, "scopes": scopes}
        logger.warning({"event": "auth_failed", "method": "both"})
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=AuthError.INVALID_CREDENTIALS)
    else:
        logger.error({"event": "auth_failed", "method": "server", "reason": "bad_config"})
        raise HTTPException(status_code=500, detail=AuthError.SERVER_ERROR)

def require_scope(required_scope: str):
    """Dependency para FastAPI que exige un scope específico."""
    def checker(auth=Depends(check_auth)):
        scopes = auth.get("scopes", [])
        if required_scope not in scopes:
            logger.warning({"event": "scope_denied", "required": required_scope, "user": auth.get("user")})
            raise HTTPException(status_code=403, detail=AuthError.INSUFFICIENT_SCOPE)
        return auth
    return checker

def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
    scopes: str = Form(DEFAULT_SCOPE)
) -> TokenResponse:
    """Endpoint para obtener access y refresh token. Demo: acepta cualquier usuario."""
    user = username
    scopes_list = [s.strip() for s in scopes.split(',')] if scopes else [DEFAULT_SCOPE]
    access_token = _create_access_token(
        data={"sub": user},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        scopes=scopes_list
    )
    refresh_token = _create_refresh_token(
        data={"sub": user},
        expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    )
    logger.info({"event": "login", "user": user, "scopes": scopes_list})
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        scopes=scopes_list
    )

def refresh_token_endpoint(req: RefreshTokenRequest) -> TokenResponse:
    """Endpoint para refrescar el access token usando un refresh token válido."""
    try:
        payload = jwt.decode(req.refresh_token, OAUTH2_SECRET, algorithms=[OAUTH2_ALG])
        user = payload.get("sub")
        if not user:
            logger.warning({"event": "refresh_failed", "reason": "no_user"})
            raise HTTPException(status_code=401, detail=AuthError.INVALID_REFRESH)
        scopes = [DEFAULT_SCOPE]
        access_token = _create_access_token(
            data={"sub": user},
            expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
            scopes=scopes
        )
        refresh_token = _create_refresh_token(
            data={"sub": user},
            expires_delta=timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )
        logger.info({"event": "refresh", "user": user})
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            scopes=scopes
        )
    except Exception as e:
        logger.warning({"event": "refresh_failed", "error": str(e)})
        raise HTTPException(status_code=401, detail=AuthError.INVALID_REFRESH) 