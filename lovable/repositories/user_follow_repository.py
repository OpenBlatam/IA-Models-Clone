"""
User Follow repository for database operations on user follows.
"""

from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
import logging

from .base_repository import BaseRepository
from ..models.user_follow import UserFollow

logger = logging.getLogger(__name__)


class UserFollowRepository(BaseRepository):
    """Repository for user follow operations."""
    
    def __init__(self, db: Session):
        """Initialize user follow repository."""
        super().__init__(db, UserFollow)
    
    def get_following(
        self,
        follower_id: str,
        limit: int = 100
    ) -> List[UserFollow]:
        """
        Get users that a user is following.
        
        Args:
            follower_id: Follower user ID
            limit: Maximum number of follows to return
            
        Returns:
            List of UserFollow objects
        """
        return self.db.query(UserFollow).filter(
            UserFollow.follower_id == follower_id
        ).order_by(desc(UserFollow.created_at)).limit(limit).all()
    
    def get_followers(
        self,
        following_id: str,
        limit: int = 100
    ) -> List[UserFollow]:
        """
        Get users that follow a specific user.
        
        Args:
            following_id: User ID being followed
            limit: Maximum number of followers to return
            
        Returns:
            List of UserFollow objects
        """
        return self.db.query(UserFollow).filter(
            UserFollow.following_id == following_id
        ).order_by(desc(UserFollow.created_at)).limit(limit).all()
    
    def is_following(
        self,
        follower_id: str,
        following_id: str
    ) -> bool:
        """
        Check if a user is following another user.
        
        Args:
            follower_id: Follower user ID
            following_id: User ID being followed
            
        Returns:
            True if following, False otherwise
        """
        follow = self.db.query(UserFollow).filter(
            UserFollow.follower_id == follower_id,
            UserFollow.following_id == following_id
        ).first()
        return follow is not None
    
    def follow(
        self,
        follower_id: str,
        following_id: str
    ) -> UserFollow:
        """
        Create a follow relationship.
        
        Args:
            follower_id: Follower user ID
            following_id: User ID being followed
            
        Returns:
            Created UserFollow object
        """
        import uuid
        from datetime import datetime
        
        follow_data = {
            "id": str(uuid.uuid4()),
            "follower_id": follower_id,
            "following_id": following_id,
            "created_at": datetime.now()
        }
        
        return self.create(follow_data)
    
    def unfollow(
        self,
        follower_id: str,
        following_id: str
    ) -> bool:
        """
        Remove a follow relationship.
        
        Args:
            follower_id: Follower user ID
            following_id: User ID being followed
            
        Returns:
            True if removed, False if not found
        """
        follow = self.db.query(UserFollow).filter(
            UserFollow.follower_id == follower_id,
            UserFollow.following_id == following_id
        ).first()
        
        if not follow:
            return False
        
        try:
            self.db.delete(follow)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error unfollowing: {e}")
            raise




