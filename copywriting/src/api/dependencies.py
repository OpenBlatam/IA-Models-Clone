from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from ..core.engine import CopywritingEngine
from ..config.settings import get_settings, get_security_config
from .app import get_engine
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
API Dependencies
===============

Dependency injection functions for the FastAPI application.
"""



# Security
security = HTTPBearer()


def get_engine_dependency() -> CopywritingEngine:
    """Get engine instance dependency"""
    engine = get_engine()
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Engine not initialized"
        )
    return engine


async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify authentication token"""
    settings = get_settings()
    security_config = get_security_config()
    
    # Simple token verification - replace with your auth logic
    if credentials.credentials != security_config.secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials


def get_client_id(request) -> str:
    """Get client identifier for rate limiting"""
    return request.client.host if request.client else "unknown"


def require_authentication(token: str = Depends(verify_token)) -> str:
    """Require authentication for protected endpoints"""
    return token


def optional_authentication(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[str]:
    """Optional authentication for endpoints that can work with or without auth"""
    if credentials is None:
        return None
    
    try:
        return verify_token(credentials)
    except HTTPException:
        return None 