"""
Common dependencies for routes
"""

from typing import Annotated, Optional
from fastapi import Query, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from utils.pagination import validate_pagination_params
from utils.errors import UnauthorizedError
from config.app_config import get_config

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)


async def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page")
) -> tuple[int, int]:
    """Dependency for pagination parameters"""
    return validate_pagination_params(page, page_size)


async def get_optional_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]
) -> Optional[str]:
    """
    Optional authentication dependency
    
    Returns user_id if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        user_id = await validate_jwt_token(credentials.credentials)
        return user_id
    except Exception as e:
        logger.debug(f"JWT validation failed: {str(e)}")
        return None


async def get_required_auth(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]
) -> str:
    """
    Required authentication dependency
    
    Returns user_id if authenticated, raises error otherwise
    """
    if not credentials:
        raise UnauthorizedError("Authentication required")
    
    try:
        user_id = await validate_jwt_token(credentials.credentials)
        if not user_id:
            raise UnauthorizedError("Invalid or expired token")
        return user_id
    except UnauthorizedError:
        raise
    except Exception as e:
        logger.error(f"JWT validation error: {str(e)}")
        raise UnauthorizedError("Invalid or expired token")


async def validate_jwt_token(token: str) -> Optional[str]:
    """
    Validate and decode JWT token
    
    Returns user_id if token is valid, None otherwise
    """
    config = get_config()
    
    if not config.secret_key:
        logger.warning("Secret key not configured, JWT validation disabled")
        return None
    
    try:
        from jose import jwt, JWTError
        from datetime import datetime
        
        algorithm = getattr(config, "algorithm", "HS256")
        payload = jwt.decode(
            token,
            config.secret_key,
            algorithms=[algorithm]
        )
        
        user_id = payload.get("sub") or payload.get("user_id")
        exp = payload.get("exp")
        
        if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
            logger.warning("Token expired")
            return None
        
        return user_id
    except ImportError:
        logger.warning("python-jose not available, JWT validation disabled")
        return None
    except JWTError as e:
        logger.debug(f"JWT decode error: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"JWT validation error: {str(e)}")
        return None


PaginationParams = Annotated[tuple[int, int], Depends(get_pagination_params)]
OptionalAuth = Annotated[Optional[str], Depends(get_optional_auth)]
RequiredAuth = Annotated[str, Depends(get_required_auth)]

