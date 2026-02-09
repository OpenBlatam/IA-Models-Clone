"""
Share Service for share operations.
"""

from typing import Dict, Any, List, Optional
from sqlalchemy.orm import Session
import logging

from ..repositories.share_repository import ShareRepository
from ..models.share import SharePlatform
from ..exceptions import NotFoundError, ValidationError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.service_base import BaseService

logger = logging.getLogger(__name__)


class ShareService(BaseService):
    """Service for share operations."""
    
    def __init__(self, db: Session):
        """Initialize share service."""
        super().__init__(db)
        self.share_repo = ShareRepository(db)
        # Initialize repositories for content verification
        from ..repositories.chat_repository import ChatRepository
        from ..repositories.comment_repository import CommentRepository
        self.chat_repo = ChatRepository(db)
        try:
            self.comment_repo = CommentRepository(db)
        except ImportError:
            self.comment_repo = None
    
    @log_execution_time
    @handle_errors
    def share_content(
        self,
        user_id: str,
        content_type: str,
        content_id: str,
        platform: str
    ) -> Dict[str, Any]:
        """
        Share content on a platform.
        
        Args:
            user_id: User ID sharing the content
            content_type: Type of content ('chat' or 'comment')
            content_id: ID of content to share
            platform: Sharing platform
            
        Returns:
            Dictionary with share data
            
        Raises:
            ValidationError: If platform is invalid
            NotFoundError: If content doesn't exist
        """
        # Validate platform using BaseService helper
        try:
            platform_enum = SharePlatform(platform.lower())
        except ValueError:
            valid_platforms = [e.value for e in SharePlatform]
            raise ValidationError(
                f"Invalid platform. Must be one of: {valid_platforms}",
                field="platform",
                value=platform
            )
        
        # Verify content exists
        if not self._verify_content_exists(content_type, content_id):
            raise NotFoundError(content_type.capitalize(), content_id)
        
        # Create share using BaseService helper
        share = self.share_repo.create(
            self.create_entity_data(
                user_id=user_id,
                content_type=content_type,
                content_id=content_id,
                platform=platform_enum
            )
        )
        
        return {"share": share}
    
    def get_content_shares(
        self,
        content_type: str,
        content_id: str
    ) -> Dict[str, Any]:
        """Get shares for specific content."""
        shares = self.share_repo.get_by_content(content_type, content_id)
        count = self.share_repo.get_share_count(content_type, content_id)
        stats = self.share_repo.get_share_stats_by_platform(content_type, content_id)
        
        return {
            "shares": self.serialize_list(shares),
            "total_shares": count,
            "platform_stats": stats
        }
    
    def get_user_shares(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get shares by a user."""
        shares = self.share_repo.get_by_user(user_id, limit=limit)
        return self.serialize_list(shares)
    
    def get_share_stats(
        self,
        content_type: str,
        content_id: str
    ) -> Dict[str, Any]:
        """Get share statistics for content."""
        count = self.share_repo.get_share_count(content_type, content_id)
        stats = self.share_repo.get_share_stats_by_platform(content_type, content_id)
        
        return {
            "total_shares": count,
            "platform_stats": stats
        }
    
    def _verify_content_exists(self, content_type: str, content_id: str) -> bool:
        """Verify that content exists."""
        if content_type == "chat":
            return self.chat_repo.get_by_id(content_id) is not None
        elif content_type == "comment":
            if self.comment_repo:
                return self.comment_repo.get_by_id(content_id) is not None
            return False
        return False






