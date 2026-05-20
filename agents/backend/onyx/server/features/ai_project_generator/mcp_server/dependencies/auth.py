"""
Authentication dependencies for MCP Server
"""

import logging
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..security import MCPSecurityManager

logger = logging.getLogger(__name__)

security = HTTPBearer()


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_manager: MCPSecurityManager = Depends()
) -> dict:
    """
    Verify JWT token and return user information
    
    Args:
        credentials: HTTP Bearer credentials
        security_manager: Security manager instance (injected)
        
    Returns:
        User information dictionary
        
    Raises:
        HTTPException: If token is invalid
    """
    if security_manager is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Security manager not configured"
        )
    
    try:
        user = await security_manager.verify_token(credentials.credentials)
        return user
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    security_manager: MCPSecurityManager = Depends()
) -> dict:
    """
    Get current authenticated user
    
    Args:
        credentials: HTTP Bearer credentials
        security_manager: Security manager instance (injected)
        
    Returns:
        User information dictionary
    """
    return await verify_token(credentials, security_manager)

