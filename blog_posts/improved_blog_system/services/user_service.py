"""
User service for user management operations
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from ..models.database import User
from ..models.schemas import UserCreate, UserUpdate, UserResponse
from ..core.exceptions import NotFoundError, ConflictError, DatabaseError
from ..core.security import SecurityService


class UserService:
    """Service for user operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
        self.security_service = SecurityService()
    
    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        try:
            # Check if user already exists
            existing_user = await self.get_user_by_email(user_data.email)
            if existing_user:
                raise ConflictError("User with this email already exists", "EMAIL_EXISTS")
            
            existing_username = await self.get_user_by_username(user_data.username)
            if existing_username:
                raise ConflictError("User with this username already exists", "USERNAME_EXISTS")
            
            # Hash password
            hashed_password = self.security_service.get_password_hash(user_data.password)
            
            # Create user
            db_user = User(
                email=user_data.email,
                username=user_data.username,
                hashed_password=hashed_password,
                full_name=user_data.full_name,
                bio=user_data.bio,
                website_url=user_data.website_url,
                roles=["user"],  # Default role
                permissions=["read_posts", "create_posts"]  # Default permissions
            )
            
            self.session.add(db_user)
            await self.session.commit()
            await self.session.refresh(db_user)
            
            return UserResponse.model_validate(db_user)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (ConflictError,)):
                raise
            raise DatabaseError(f"Failed to create user: {str(e)}")
    
    async def get_user(self, user_id: str) -> UserResponse:
        """Get user by ID."""
        query = select(User).where(User.id == user_id)
        result = await self.session.execute(query)
        user = result.scalar_one_or_none()
        
        if not user:
            raise NotFoundError("User", user_id)
        
        return UserResponse.model_validate(user)
    
    async def get_user_by_email(self, email: str) -> Optional[UserResponse]:
        """Get user by email."""
        query = select(User).where(User.email == email)
        result = await self.session.execute(query)
        user = result.scalar_one_or_none()
        
        if user:
            return UserResponse.model_validate(user)
        return None
    
    async def get_user_by_username(self, username: str) -> Optional[UserResponse]:
        """Get user by username."""
        query = select(User).where(User.username == username)
        result = await self.session.execute(query)
        user = result.scalar_one_or_none()
        
        if user:
            return UserResponse.model_validate(user)
        return None
    
    async def update_user(self, user_id: str, user_data: UserUpdate) -> UserResponse:
        """Update user information."""
        try:
            query = select(User).where(User.id == user_id)
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError("User", user_id)
            
            # Update fields
            update_data = user_data.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            await self.session.commit()
            await self.session.refresh(user)
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to update user: {str(e)}")
    
    async def authenticate_user(self, email: str, password: str) -> Optional[UserResponse]:
        """Authenticate user with email and password."""
        try:
            query = select(User).where(User.email == email, User.is_active == True)
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            if not self.security_service.verify_password(password, user.hashed_password):
                return None
            
            return UserResponse.model_validate(user)
            
        except Exception as e:
            raise DatabaseError(f"Failed to authenticate user: {str(e)}")
    
    async def list_users(
        self,
        skip: int = 0,
        limit: int = 20,
        is_active: Optional[bool] = None
    ) -> List[UserResponse]:
        """List users with pagination."""
        try:
            query = select(User)
            
            if is_active is not None:
                query = query.where(User.is_active == is_active)
            
            query = query.offset(skip).limit(limit).order_by(User.created_at.desc())
            
            result = await self.session.execute(query)
            users = result.scalars().all()
            
            return [UserResponse.model_validate(user) for user in users]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list users: {str(e)}")
    
    async def deactivate_user(self, user_id: str) -> bool:
        """Deactivate a user."""
        try:
            query = select(User).where(User.id == user_id)
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError("User", user_id)
            
            user.is_active = False
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to deactivate user: {str(e)}")
    
    async def activate_user(self, user_id: str) -> bool:
        """Activate a user."""
        try:
            query = select(User).where(User.id == user_id)
            result = await self.session.execute(query)
            user = result.scalar_one_or_none()
            
            if not user:
                raise NotFoundError("User", user_id)
            
            user.is_active = True
            await self.session.commit()
            
            return True
            
        except Exception as e:
            await self.session.rollback()
            if isinstance(e, (NotFoundError,)):
                raise
            raise DatabaseError(f"Failed to activate user: {str(e)}")






























