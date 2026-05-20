"""
Dependency Injection
FastAPI dependencies for services and authentication
"""

import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..core.content_generator import ContentGenerator
from ..services.collaboration_service import CollaborationService
from ..services.analytics_service import AnalyticsService
from ..utils.auth import verify_token
from .lifespan import (
    get_content_generator_instance,
    get_collaboration_service_instance,
    get_analytics_service_instance
)
from .service_manager import create_service_dependency

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

get_content_generator = create_service_dependency(
    get_content_generator_instance,
    "Content generator"
)

get_collaboration_service = create_service_dependency(
    get_collaboration_service_instance,
    "Collaboration service"
)

get_analytics_service = create_service_dependency(
    get_analytics_service_instance,
    "Analytics service"
)

async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Dict[str, Any]:
    """Get current authenticated user"""
    if not credentials:
        logger.warning(
            "Missing authentication credentials",
            extra={"path": request.url.path, "method": request.method}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        user_data = verify_token(credentials.credentials)
        if not user_data:
            logger.warning(
                "Invalid token provided",
                extra={"path": request.url.path, "method": request.method}
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(
            "User authenticated",
            extra={
                "user_id": user_data.get("id"),
                "path": request.url.path
            }
        )
        return user_data
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(
            "Authentication error",
            extra={
                "error": str(e),
                "path": request.url.path,
                "method": request.method
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
