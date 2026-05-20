"""
Collaboration Service
Manages video sharing and collaboration
"""

from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SharePermission(str, Enum):
    """Share permissions"""
    VIEW = "view"
    EDIT = "edit"
    DELETE = "delete"
    ADMIN = "admin"


class Share:
    """Represents a video share"""
    
    def __init__(
        self,
        share_id: str,
        video_id: UUID,
        owner_id: str,
        shared_with_id: Optional[str] = None,
        shared_with_email: Optional[str] = None,
        permission: SharePermission = SharePermission.VIEW,
        share_token: Optional[str] = None,
        is_public: bool = False,
        expires_at: Optional[datetime] = None,
        created_at: Optional[datetime] = None
    ):
        self.share_id = share_id
        self.video_id = video_id
        self.owner_id = owner_id
        self.shared_with_id = shared_with_id
        self.shared_with_email = shared_with_email
        self.permission = permission
        self.share_token = share_token
        self.is_public = is_public
        self.expires_at = expires_at
        self.created_at = created_at or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "share_id": self.share_id,
            "video_id": str(self.video_id),
            "owner_id": self.owner_id,
            "shared_with_id": self.shared_with_id,
            "shared_with_email": self.shared_with_email,
            "permission": self.permission.value,
            "share_token": self.share_token,
            "is_public": self.is_public,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat(),
        }


class CollaborationService:
    """Manages video sharing and collaboration"""
    
    def __init__(self):
        # In-memory storage (use database in production)
        self.shares: Dict[str, Share] = {}
        self.video_shares: Dict[UUID, List[str]] = {}  # video_id -> share_ids
        self.user_shares: Dict[str, List[str]] = {}  # user_id -> share_ids
    
    def share_video(
        self,
        video_id: UUID,
        owner_id: str,
        shared_with_id: Optional[str] = None,
        shared_with_email: Optional[str] = None,
        permission: SharePermission = SharePermission.VIEW,
        is_public: bool = False,
        expires_at: Optional[datetime] = None
    ) -> Share:
        """
        Share video with user or make public
        
        Args:
            video_id: Video ID
            owner_id: Owner user ID
            shared_with_id: User ID to share with (optional)
            shared_with_email: Email to share with (optional)
            permission: Permission level
            is_public: Whether share is public
            expires_at: Expiration date (optional)
            
        Returns:
            Created share
        """
        import secrets
        
        share_id = f"share_{len(self.shares) + 1}"
        share_token = secrets.token_urlsafe(32) if is_public else None
        
        share = Share(
            share_id=share_id,
            video_id=video_id,
            owner_id=owner_id,
            shared_with_id=shared_with_id,
            shared_with_email=shared_with_email,
            permission=permission,
            share_token=share_token,
            is_public=is_public,
            expires_at=expires_at
        )
        
        self.shares[share_id] = share
        
        if video_id not in self.video_shares:
            self.video_shares[video_id] = []
        self.video_shares[video_id].append(share_id)
        
        if shared_with_id:
            if shared_with_id not in self.user_shares:
                self.user_shares[shared_with_id] = []
            self.user_shares[shared_with_id].append(share_id)
        
        logger.info(f"Video {video_id} shared by {owner_id}")
        return share
    
    def get_video_shares(self, video_id: UUID) -> List[Share]:
        """Get all shares for a video"""
        share_ids = self.video_shares.get(video_id, [])
        return [self.shares[sid] for sid in share_ids if sid in self.shares]
    
    def get_user_shared_videos(self, user_id: str) -> List[Share]:
        """Get videos shared with user"""
        share_ids = self.user_shares.get(user_id, [])
        return [self.shares[sid] for sid in share_ids if sid in self.shares]
    
    def get_share_by_token(self, token: str) -> Optional[Share]:
        """Get share by public token"""
        for share in self.shares.values():
            if share.share_token == token:
                if share.expires_at and datetime.utcnow() > share.expires_at:
                    return None
                return share
        return None
    
    def update_share_permission(
        self,
        share_id: str,
        owner_id: str,
        permission: SharePermission
    ) -> Share:
        """Update share permission"""
        share = self.shares.get(share_id)
        if not share:
            raise ValueError("Share not found")
        
        if share.owner_id != owner_id:
            raise ValueError("Not authorized")
        
        share.permission = permission
        logger.info(f"Updated share {share_id} permission to {permission.value}")
        return share
    
    def revoke_share(self, share_id: str, owner_id: str) -> bool:
        """Revoke share"""
        share = self.shares.get(share_id)
        if not share:
            return False
        
        if share.owner_id != owner_id:
            raise ValueError("Not authorized")
        
        # Remove from indexes
        if share.video_id in self.video_shares:
            self.video_shares[share.video_id].remove(share_id)
        
        if share.shared_with_id and share.shared_with_id in self.user_shares:
            self.user_shares[share.shared_with_id].remove(share_id)
        
        del self.shares[share_id]
        logger.info(f"Revoked share {share_id}")
        return True


_collaboration_service: Optional[CollaborationService] = None


def get_collaboration_service() -> CollaborationService:
    """Get collaboration service instance (singleton)"""
    global _collaboration_service
    if _collaboration_service is None:
        _collaboration_service = CollaborationService()
    return _collaboration_service

