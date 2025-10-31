"""
Users API endpoints
"""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Query, Path

from ....models.schemas import UserCreate, UserUpdate, UserResponse, PaginationParams
from ....api.dependencies import UserServiceDep, CurrentUserDep
from ....core.exceptions import NotFoundError, ConflictError, ValidationError
from ....utils.pagination import create_paginated_response

router = APIRouter()


@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    user_service: UserServiceDep
):
    """Create a new user."""
    try:
        user = await user_service.create_user(user_data)
        return user
    except ConflictError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=e.detail
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: CurrentUserDep
):
    """Get current user profile."""
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: str = Path(..., description="User ID"),
    user_service: UserServiceDep = Depends()
):
    """Get user by ID."""
    try:
        user = await user_service.get_user(user_id)
        return user
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user"
        )


@router.put("/me", response_model=UserResponse)
async def update_current_user(
    user_data: UserUpdate,
    current_user: CurrentUserDep,
    user_service: UserServiceDep
):
    """Update current user profile."""
    try:
        user = await user_service.update_user(current_user["user_id"], user_data)
        return user
    except NotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=e.detail
        )
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.detail
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )






























