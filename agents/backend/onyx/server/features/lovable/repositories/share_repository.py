"""
Share Repository for database operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from datetime import datetime
import logging
import uuid

from ..models.share import Share, SharePlatform
from .base_repository import BaseRepository

logger = logging.getLogger(__name__)


class ShareRepository(BaseRepository):
    """Repository for Share operations."""
    
    def __init__(self, db: Session):
        """Initialize repository with database session."""
        super().__init__(db, Share)
    
    def create(self, share_data: Dict[str, Any]) -> Share:
        """Create a new share."""
        share = super().create(share_data)
        logger.info(f"Created share {share.id} for {share.content_type} {share.content_id}")
        return share
    
    def get_by_content(
        self,
        content_type: str,
        content_id: str,
        limit: int = 100
    ) -> List[Share]:
        """Get shares for specific content."""
        shares = self.db.query(Share).filter(
            Share.content_type == content_type,
            Share.content_id == content_id
        ).order_by(desc(Share.created_at)).limit(limit).all()
        
        return shares
    
    def get_by_user(self, user_id: str, limit: int = 50) -> List[Share]:
        """Get shares by a user."""
        shares = self.db.query(Share).filter(
            Share.user_id == user_id
        ).order_by(desc(Share.created_at)).limit(limit).all()
        
        return shares
    
    def get_share_count(self, content_type: str, content_id: str) -> int:
        """Get share count for specific content."""
        return self.db.query(Share).filter(
            Share.content_type == content_type,
            Share.content_id == content_id
        ).count()
    
    def get_share_stats_by_platform(
        self,
        content_type: str,
        content_id: str
    ) -> Dict[str, int]:
        """Get share statistics grouped by platform."""
        shares = self.db.query(
            Share.platform,
            func.count(Share.id).label('count')
        ).filter(
            Share.content_type == content_type,
            Share.content_id == content_id
        ).group_by(Share.platform).all()
        
        return {platform.value: count for platform, count in shares}







