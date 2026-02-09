"""
Authentication API endpoints.
Refactored to use BaseRouter for reduced duplication.
Note: require_auth is kept as-is since it's used by other routers.
"""

from fastapi import Depends, Header
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

from .base_router import BaseRouter
from ..core.auth import AuthManager

# Create base router instance
# Note: require_authentication=False for create/validate endpoints
base = BaseRouter(
    prefix="/api/auth",
    tags=["Authentication"],
    require_authentication=False,  # Some endpoints don't require auth
    require_rate_limit=False
)

router = base.router

auth_manager = AuthManager()


class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key."""
    user_id: str = Field(..., description="User identifier")
    permissions: Optional[list] = Field(None, description="List of permissions")


class ValidateAPIKeyRequest(BaseModel):
    """Request to validate an API key."""
    api_key: str = Field(..., description="API key to validate")


def get_api_key(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract API key from Authorization header.
    
    Args:
        authorization: Authorization header value
    
    Returns:
        API key if present, None otherwise
    """
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return None


def require_auth(api_key: Optional[str] = Depends(get_api_key)) -> Dict[str, Any]:
    """
    Dependency to require authentication.
    This function is kept here as-is since it's used by other routers.
    
    Args:
        api_key: API key from header
    
    Returns:
        User info if authenticated
    
    Raises:
        HTTPException: If not authenticated
    """
    from fastapi import HTTPException, status
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required. Use 'Authorization: Bearer <api_key>' header."
        )
    
    user_info = auth_manager.validate_api_key(api_key)
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return user_info


@router.post("/api-key/create")
@base.timed_endpoint("create_api_key")
async def create_api_key(
    request: CreateAPIKeyRequest
) -> Dict[str, Any]:
    """Create a new API key."""
    base.log_request("create_api_key", user_id=request.user_id)
    
    api_key = auth_manager.create_api_key(
        user_id=request.user_id,
        permissions=request.permissions
    )
    return base.success({
        "api_key": api_key,
        "user_id": request.user_id,
        "permissions": request.permissions or ["read", "write"]
    })


@router.post("/api-key/validate")
@base.timed_endpoint("validate_api_key")
async def validate_api_key(
    request: ValidateAPIKeyRequest
) -> Dict[str, Any]:
    """Validate an API key."""
    base.log_request("validate_api_key")
    
    user_info = auth_manager.validate_api_key(request.api_key)
    if user_info:
        return base.success({
            "valid": True,
            "user_id": user_info["user_id"],
            "permissions": user_info["permissions"]
        })
    else:
        return base.success({
            "valid": False
        })


@router.post("/api-key/revoke")
@base.timed_endpoint("revoke_api_key")
async def revoke_api_key(
    request: ValidateAPIKeyRequest,
    _: Dict = Depends(require_auth)  # This endpoint requires auth
) -> Dict[str, Any]:
    """Revoke an API key (requires authentication)."""
    base.log_request("revoke_api_key")
    
    revoked = auth_manager.revoke_api_key(request.api_key)
    return base.success(
        None,
        message="API key revoked" if revoked else "API key not found"
    )






