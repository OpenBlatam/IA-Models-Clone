"""
User Repository Implementation
"""

from typing import Optional
from uuid import UUID
from sqlalchemy.orm import Session

from ...models.database import User
from .base import BaseRepository
from ...domain.interfaces.repositories import IUserRepository

class UserRepository(BaseRepository[User], IUserRepository):
    """User repository implementation"""
    
    def __init__(self, session: Session):
        """Initialize user repository"""
        super().__init__(session, User)
    
    async def get_by_id(self, id: str) -> Optional[User]:
        """Get user by ID (supports both UUID and string)"""
        try:
            user_id = UUID(id) if isinstance(id, str) else id
            return self.session.query(User).filter(User.id == user_id).first()
        except (ValueError, TypeError):
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.session.query(User).filter(User.email == email).first()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        return self.session.query(User).filter(User.username == username).first()

