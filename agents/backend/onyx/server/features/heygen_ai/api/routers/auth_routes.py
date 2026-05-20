from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any
import logging
from ..core.database import get_session
from ..core.auth import (
from ..models.schemas import (
from ..services.user_service import create_user, get_user_by_email
from ..utils.helpers import generate_request_id, format_error_message
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
Authentication routes for HeyGen AI API
Provides authentication and authorization endpoints with Pydantic models and type hints.
"""


    authenticate_user,
    create_access_token,
    create_refresh_token,
    verify_token,
    get_current_user
)
    UserCreateInput,
    LoginInput,
    LoginOutput,
    RegisterOutput,
    TokenRefreshInput,
    TokenRefreshOutput
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/register", response_model=RegisterOutput)
async def register_user(
    user_data: UserCreateInput,
    session: AsyncSession = Depends(get_session)
) -> RegisterOutput:
    """Register new user"""
    try:
        # Check if user already exists
        existing_user = await get_user_by_email(session, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Create new user
        user = await create_user(session, user_data)
        
        # Generate tokens
        access_token = create_access_token(data={"sub": user["user_id"]})
        refresh_token = create_refresh_token(data={"sub": user["user_id"]})
        
        return RegisterOutput(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error registering user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user"
        )


@router.post("/login", response_model=LoginOutput)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session)
) -> LoginOutput:
    """Login user with username/email and password"""
    try:
        # Authenticate user
        user = await authenticate_user(session, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Generate tokens
        access_token = create_access_token(data={"sub": user["user_id"]})
        refresh_token = create_refresh_token(data={"sub": user["user_id"]})
        
        return LoginOutput(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error logging in user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to login user"
        )


@router.post("/refresh", response_model=TokenRefreshOutput)
async def refresh_token_endpoint(
    refresh_data: TokenRefreshInput
) -> TokenRefreshOutput:
    """Refresh access token using refresh token"""
    try:
        # Verify refresh token
        payload = verify_token(refresh_data.refresh_token)
        user_id = payload.get("sub")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Generate new access token
        access_token = create_access_token(data={"sub": user_id})
        
        return TokenRefreshOutput(
            access_token=access_token,
            token_type="bearer"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token"
        )


@router.post("/logout")
async def logout_user(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Logout user (invalidate tokens)"""
    try:
        # In a real implementation, you would add the token to a blacklist
        # For now, we just return success
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error(f"Error logging out user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to logout user"
        )


@router.get("/me")
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get current user information from token"""
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "email": current_user["email"],
        "is_active": current_user["is_active"]
    }


@router.post("/verify")
async def verify_token_endpoint(
    token: str
) -> Dict[str, Any]:
    """Verify token validity"""
    try:
        payload = verify_token(token)
        return {
            "valid": True,
            "user_id": payload.get("sub"),
            "expires_at": payload.get("exp")
        }
    except Exception:
        return {"valid": False}


# Named exports
__all__ = ["router"] 