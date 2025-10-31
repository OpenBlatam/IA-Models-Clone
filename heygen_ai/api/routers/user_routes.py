from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List
import logging
from ..core.database import get_session
from ..core.auth import get_current_user, get_current_admin_user
from ..models.schemas import (
from ..services.user_service import (
from ..utils.helpers import generate_request_id, format_error_message
from typing import Any, List, Dict, Optional
import asyncio
#!/usr/bin/env python3
"""
User routes for HeyGen AI API
Provides user management endpoints with Pydantic models and type hints.
"""


    UserCreateInput,
    UserUpdateInput,
    UserCreateOutput,
    UserUpdateOutput,
    UserListOutput,
    UserDetailOutput
)
    create_user,
    get_user_by_id,
    get_user_by_email,
    update_user,
    delete_user,
    list_users,
    get_user_statistics
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/", response_model=UserCreateOutput)
async def create_user_endpoint(
    user_data: UserCreateInput,
    session: AsyncSession = Depends(get_session)
) -> UserCreateOutput:
    """Create new user (admin only)"""
    try:
        user = await create_user(session, user_data)
        return UserCreateOutput(
            user_id=user["user_id"],
            username=user["username"],
            email=user["email"],
            full_name=user.get("full_name"),
            is_active=user["is_active"],
            created_at=user["created_at"]
        )
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/me", response_model=UserDetailOutput)
async def get_current_user_info(
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> UserDetailOutput:
    """Get current user information"""
    try:
        user = await get_user_by_id(session, current_user["user_id"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserDetailOutput(**user)
    except Exception as e:
        logger.error(f"Error getting user info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information"
        )


@router.put("/me", response_model=UserUpdateOutput)
async def update_current_user(
    user_data: UserUpdateInput,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session)
) -> UserUpdateOutput:
    """Update current user information"""
    try:
        updated_user = await update_user(session, current_user["user_id"], user_data)
        return UserUpdateOutput(**updated_user)
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.get("/{user_id}", response_model=UserDetailOutput)
async def get_user_by_id_endpoint(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> UserDetailOutput:
    """Get user by ID (admin only)"""
    try:
        user = await get_user_by_id(session, user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserDetailOutput(**user)
    except Exception as e:
        logger.error(f"Error getting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/{user_id}", response_model=UserUpdateOutput)
async def update_user_endpoint(
    user_id: str,
    user_data: UserUpdateInput,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> UserUpdateOutput:
    """Update user by ID (admin only)"""
    try:
        updated_user = await update_user(session, user_id, user_data)
        return UserUpdateOutput(**updated_user)
    except Exception as e:
        logger.error(f"Error updating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}")
async def delete_user_endpoint(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Delete user by ID (admin only)"""
    try:
        success = await delete_user(session, user_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {"message": "User deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )


@router.get("/", response_model=UserListOutput)
async def list_users_endpoint(
    page: int = 1,
    page_size: int = 50,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> UserListOutput:
    """List all users (admin only)"""
    try:
        users = await list_users(session, page, page_size)
        return UserListOutput(**users)
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )


@router.get("/{user_id}/statistics")
async def get_user_statistics_endpoint(
    user_id: str,
    current_admin: Dict[str, Any] = Depends(get_current_admin_user),
    session: AsyncSession = Depends(get_session)
) -> Dict[str, Any]:
    """Get user statistics (admin only)"""
    try:
        stats = await get_user_statistics(session, user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting user statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user statistics"
        )


# Named exports
__all__ = ["router"] 