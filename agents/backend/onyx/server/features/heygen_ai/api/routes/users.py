from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

from fastapi import APIRouter, Depends, HTTPException, Query, Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone, timedelta
import structlog
from .base import (
from api.models.user import UserCreate, UserUpdate, UserResponse, UserListResponse
from api.schemas.pagination import PaginationParams
from typing import Any, List, Dict, Optional
import logging
import asyncio
#!/usr/bin/env python3
"""
User Routes for HeyGen AI FastAPI
Well-structured user management routes with clear dependencies.
"""


    BaseRoute, RouteCategory, BaseResponse, ErrorResponse, PaginatedResponse,
    route_metrics, require_auth, rate_limit, cache_response,
    get_database_operations, get_current_user, get_request_id
)

logger = structlog.get_logger()

# =============================================================================
# User Route Class
# =============================================================================

class UserRoutes(BaseRoute):
    """User management routes with clear structure and dependencies."""
    
    def __init__(self, db_operations, api_operations) -> Any:
        super().__init__(
            name="User Management",
            description="User management operations including CRUD, authentication, and profile management",
            category=RouteCategory.USERS,
            tags=["users", "authentication", "profiles"],
            prefix="/users",
            dependencies={
                "db_ops": db_operations,
                "api_ops": api_operations
            }
        )
        
        # Register routes
        self._register_routes()
    
    def _register_routes(self) -> Any:
        """Register all user routes with clear organization."""
        
        # =====================================================================
        # User CRUD Operations
        # =====================================================================
        
        @self.router.get(
            "/",
            response_model=PaginatedResponse,
            summary="Get all users",
            description="Retrieve a paginated list of all users with optional filtering"
        )
        @route_metrics
        @rate_limit(requests_per_minute=100)
        @cache_response(ttl=300)
        async def get_users(
            pagination: PaginationParams = Depends(),
            search: Optional[str] = Query(None, description="Search users by name or email"),
            status: Optional[str] = Query(None, description="Filter by user status"),
            role: Optional[str] = Query(None, description="Filter by user role"),
            request_id: str = Depends(get_request_id)
        ):
            """Get paginated list of users with filtering."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Build query with filters
                query = "SELECT * FROM users WHERE 1=1"
                params = {}
                
                if search:
                    query += " AND (name ILIKE :search OR email ILIKE :search)"
                    params["search"] = f"%{search}%"
                
                if status:
                    query += " AND status = :status"
                    params["status"] = status
                
                if role:
                    query += " AND role = :role"
                    params["role"] = role
                
                # Add pagination
                query += " ORDER BY created_at DESC LIMIT :limit OFFSET :offset"
                params["limit"] = pagination.page_size
                params["offset"] = (pagination.page - 1) * pagination.page_size
                
                # Execute query
                users = await db_ops.execute_query(query, parameters=params)
                
                # Get total count
                count_query = query.replace("SELECT *", "SELECT COUNT(*) as total")
                count_query = count_query.split("ORDER BY")[0]  # Remove ORDER BY and LIMIT
                count_result = await db_ops.execute_query(count_query, parameters=params)
                total_count = count_result[0]["total"] if count_result else 0
                
                return self.paginated_response(
                    data=users,
                    total_count=total_count,
                    page=pagination.page,
                    page_size=pagination.page_size,
                    request_id=request_id
                )
                
            except Exception as e:
                logger.error(f"Error getting users: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve users")
        
        @self.router.get(
            "/{user_id}",
            response_model=UserResponse,
            summary="Get user by ID",
            description="Retrieve a specific user by their ID"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=200)
        @cache_response(ttl=600)
        async def get_user(
            user_id: int = Path(..., description="User ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get user by ID."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check permissions (users can only view their own profile unless admin)
                if current_user["user_id"] != str(user_id) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                users = await db_ops.execute_query(
                    "SELECT * FROM users WHERE id = :user_id",
                    parameters={"user_id": user_id}
                )
                
                if not users:
                    raise HTTPException(status_code=404, detail="User not found")
                
                return self.success_response(
                    data=users[0],
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting user {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve user")
        
        @self.router.post(
            "/",
            response_model=UserResponse,
            summary="Create new user",
            description="Create a new user account"
        )
        @route_metrics
        @rate_limit(requests_per_minute=10)
        async def create_user(
            user_data: UserCreate,
            request_id: str = Depends(get_request_id)
        ):
            """Create a new user."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check if user already exists
                existing_users = await db_ops.execute_query(
                    "SELECT id FROM users WHERE email = :email",
                    parameters={"email": user_data.email}
                )
                
                if existing_users:
                    raise HTTPException(status_code=400, detail="User with this email already exists")
                
                # Create user
                user_dict = user_data.dict()
                user_dict["created_at"] = datetime.now(timezone.utc)
                user_dict["status"] = "active"
                
                result = await db_ops.execute_insert(
                    table="users",
                    data=user_dict,
                    returning="id, name, email, role, status, created_at"
                )
                
                return self.success_response(
                    data=result,
                    message="User created successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                raise HTTPException(status_code=500, detail="Failed to create user")
        
        @self.router.put(
            "/{user_id}",
            response_model=UserResponse,
            summary="Update user",
            description="Update an existing user's information"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=50)
        async def update_user(
            user_id: int = Path(..., description="User ID"),
            user_data: UserUpdate,
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Update user information."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check permissions
                if current_user["user_id"] != str(user_id) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Check if user exists
                existing_users = await db_ops.execute_query(
                    "SELECT id FROM users WHERE id = :user_id",
                    parameters={"user_id": user_id}
                )
                
                if not existing_users:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Update user
                update_data = user_data.dict(exclude_unset=True)
                update_data["updated_at"] = datetime.now(timezone.utc)
                
                result = await db_ops.execute_update(
                    table="users",
                    data=update_data,
                    where_conditions={"id": user_id},
                    returning="id, name, email, role, status, updated_at"
                )
                
                return self.success_response(
                    data=result,
                    message="User updated successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating user {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to update user")
        
        @self.router.delete(
            "/{user_id}",
            response_model=BaseResponse,
            summary="Delete user",
            description="Delete a user account (soft delete)"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=20)
        async def delete_user(
            user_id: int = Path(..., description="User ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Delete user (soft delete)."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check permissions (only admins can delete users)
                if current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Check if user exists
                existing_users = await db_ops.execute_query(
                    "SELECT id FROM users WHERE id = :user_id",
                    parameters={"user_id": user_id}
                )
                
                if not existing_users:
                    raise HTTPException(status_code=404, detail="User not found")
                
                # Soft delete user
                await db_ops.execute_update(
                    table="users",
                    data={
                        "status": "deleted",
                        "deleted_at": datetime.now(timezone.utc)
                    },
                    where_conditions={"id": user_id}
                )
                
                return self.success_response(
                    message="User deleted successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error deleting user {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to delete user")
        
        # =====================================================================
        # User Profile Operations
        # =====================================================================
        
        @self.router.get(
            "/{user_id}/profile",
            response_model=Dict[str, Any],
            summary="Get user profile",
            description="Get detailed user profile information"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=100)
        @cache_response(ttl=300)
        async def get_user_profile(
            user_id: int = Path(..., description="User ID"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get user profile with additional information."""
            try:
                db_ops = self.get_dependency("db_ops")
                api_ops = self.get_dependency("api_ops")
                
                # Check permissions
                if current_user["user_id"] != str(user_id) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Get user data
                users = await db_ops.execute_query(
                    "SELECT * FROM users WHERE id = :user_id",
                    parameters={"user_id": user_id}
                )
                
                if not users:
                    raise HTTPException(status_code=404, detail="User not found")
                
                user = users[0]
                
                # Get user statistics
                stats = await db_ops.execute_query(
                    """
                    SELECT 
                        COUNT(DISTINCT v.id) as total_videos,
                        COUNT(DISTINCT p.id) as total_projects,
                        SUM(v.duration) as total_duration
                    FROM users u
                    LEFT JOIN videos v ON u.id = v.user_id
                    LEFT JOIN projects p ON u.id = p.user_id
                    WHERE u.id = :user_id
                    """,
                    parameters={"user_id": user_id}
                )
                
                # Get external profile data (if available)
                external_profile = {}
                try:
                    external_response = await api_ops.get(
                        endpoint=f"/user-profiles/{user_id}",
                        cache_key=f"external_profile:{user_id}",
                        cache_ttl=600
                    )
                    external_profile = external_response.get("data", {})
                except Exception as e:
                    logger.warning(f"Failed to fetch external profile for user {user_id}: {e}")
                
                profile_data = {
                    "user": user,
                    "statistics": stats[0] if stats else {},
                    "external_profile": external_profile
                }
                
                return self.success_response(
                    data=profile_data,
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting user profile {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve user profile")
        
        @self.router.put(
            "/{user_id}/profile",
            response_model=Dict[str, Any],
            summary="Update user profile",
            description="Update user profile information"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=30)
        async def update_user_profile(
            user_id: int = Path(..., description="User ID"),
            profile_data: Dict[str, Any],
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Update user profile."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check permissions
                if current_user["user_id"] != str(user_id) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Update profile
                update_data = {
                    **profile_data,
                    "updated_at": datetime.now(timezone.utc)
                }
                
                result = await db_ops.execute_update(
                    table="users",
                    data=update_data,
                    where_conditions={"id": user_id},
                    returning="id, name, email, role, status, updated_at"
                )
                
                return self.success_response(
                    data=result,
                    message="Profile updated successfully",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error updating user profile {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to update user profile")
        
        # =====================================================================
        # User Authentication Operations
        # =====================================================================
        
        @self.router.post(
            "/login",
            response_model=Dict[str, Any],
            summary="User login",
            description="Authenticate user and return access token"
        )
        @route_metrics
        @rate_limit(requests_per_minute=5)
        async def login(
            email: str,
            password: str,
            request_id: str = Depends(get_request_id)
        ):
            """User login authentication."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Get user by email
                users = await db_ops.execute_query(
                    "SELECT * FROM users WHERE email = :email AND status = 'active'",
                    parameters={"email": email}
                )
                
                if not users:
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                user = users[0]
                
                # Verify password (this would use proper password hashing)
                if user["password"] != password:  # In real app, use proper password verification
                    raise HTTPException(status_code=401, detail="Invalid credentials")
                
                # Generate access token (this would use proper JWT)
                access_token = f"token_{user['id']}_{int(datetime.now(timezone.utc).timestamp())}"
                
                # Update last login
                await db_ops.execute_update(
                    table="users",
                    data={"last_login": datetime.now(timezone.utc)},
                    where_conditions={"id": user["id"]}
                )
                
                return self.success_response(
                    data={
                        "access_token": access_token,
                        "token_type": "bearer",
                        "user": {
                            "id": user["id"],
                            "name": user["name"],
                            "email": user["email"],
                            "role": user["role"]
                        }
                    },
                    message="Login successful",
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Login error: {e}")
                raise HTTPException(status_code=500, detail="Login failed")
        
        @self.router.post(
            "/logout",
            response_model=BaseResponse,
            summary="User logout",
            description="Logout user and invalidate token"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=20)
        async def logout(
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """User logout."""
            try:
                # In a real application, you would invalidate the token
                # For now, just return success
                return self.success_response(
                    message="Logout successful",
                    request_id=request_id
                )
                
            except Exception as e:
                logger.error(f"Logout error: {e}")
                raise HTTPException(status_code=500, detail="Logout failed")
        
        # =====================================================================
        # User Statistics and Analytics
        # =====================================================================
        
        @self.router.get(
            "/{user_id}/statistics",
            response_model=Dict[str, Any],
            summary="Get user statistics",
            description="Get comprehensive user statistics and analytics"
        )
        @route_metrics
        @require_auth
        @rate_limit(requests_per_minute=50)
        @cache_response(ttl=600)
        async def get_user_statistics(
            user_id: int = Path(..., description="User ID"),
            period: str = Query("30d", description="Statistics period (7d, 30d, 90d, 1y)"),
            current_user: Dict[str, Any] = Depends(get_current_user),
            request_id: str = Depends(get_request_id)
        ):
            """Get user statistics and analytics."""
            try:
                db_ops = self.get_dependency("db_ops")
                
                # Check permissions
                if current_user["user_id"] != str(user_id) and current_user.get("role") != "admin":
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                # Calculate date range based on period
                end_date = datetime.now(timezone.utc)
                if period == "7d":
                    start_date = end_date - timedelta(days=7)
                elif period == "30d":
                    start_date = end_date - timedelta(days=30)
                elif period == "90d":
                    start_date = end_date - timedelta(days=90)
                elif period == "1y":
                    start_date = end_date - timedelta(days=365)
                else:
                    start_date = end_date - timedelta(days=30)
                
                # Get video statistics
                video_stats = await db_ops.execute_query(
                    """
                    SELECT 
                        COUNT(*) as total_videos,
                        SUM(duration) as total_duration,
                        AVG(duration) as avg_duration,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_videos,
                        COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_videos
                    FROM videos 
                    WHERE user_id = :user_id 
                    AND created_at BETWEEN :start_date AND :end_date
                    """,
                    parameters={
                        "user_id": user_id,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                
                # Get project statistics
                project_stats = await db_ops.execute_query(
                    """
                    SELECT 
                        COUNT(*) as total_projects,
                        COUNT(CASE WHEN status = 'active' THEN 1 END) as active_projects,
                        COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_projects
                    FROM projects 
                    WHERE user_id = :user_id 
                    AND created_at BETWEEN :start_date AND :end_date
                    """,
                    parameters={
                        "user_id": user_id,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                
                # Get usage statistics
                usage_stats = await db_ops.execute_query(
                    """
                    SELECT 
                        COUNT(*) as total_requests,
                        COUNT(CASE WHEN success = true THEN 1 END) as successful_requests,
                        AVG(duration_ms) as avg_response_time
                    FROM api_requests 
                    WHERE user_id = :user_id 
                    AND created_at BETWEEN :start_date AND :end_date
                    """,
                    parameters={
                        "user_id": user_id,
                        "start_date": start_date,
                        "end_date": end_date
                    }
                )
                
                statistics = {
                    "period": period,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "videos": video_stats[0] if video_stats else {},
                    "projects": project_stats[0] if project_stats else {},
                    "usage": usage_stats[0] if usage_stats else {}
                }
                
                return self.success_response(
                    data=statistics,
                    request_id=request_id
                )
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Error getting user statistics {user_id}: {e}")
                raise HTTPException(status_code=500, detail="Failed to retrieve user statistics")

# =============================================================================
# Route Factory
# =============================================================================

def create_user_routes(db_operations, api_operations) -> UserRoutes:
    """Factory function to create user routes with dependencies."""
    return UserRoutes(db_operations, api_operations)

# =============================================================================
# Export
# =============================================================================

__all__ = [
    "UserRoutes",
    "create_user_routes"
] 