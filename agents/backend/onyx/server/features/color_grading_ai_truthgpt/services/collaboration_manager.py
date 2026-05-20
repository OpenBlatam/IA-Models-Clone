"""
Collaboration Manager for Color Grading AI
===========================================

Manages collaborative workflows and sharing.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class ShareLink:
    """Share link data structure."""
    link_id: str
    resource_type: str  # preset, template, result
    resource_id: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    is_active: bool = True
    permissions: List[str] = field(default_factory=lambda: ["view"])


@dataclass
class Comment:
    """Comment on shared resource."""
    comment_id: str
    resource_id: str
    author: str
    content: str
    created_at: datetime
    parent_id: Optional[str] = None  # For threaded comments


class CollaborationManager:
    """
    Manages collaboration features.
    
    Features:
    - Share links
    - Comments
    - Collaborative editing
    - Access control
    """
    
    def __init__(self):
        """Initialize collaboration manager."""
        self._share_links: Dict[str, ShareLink] = {}
        self._comments: Dict[str, List[Comment]] = {}  # resource_id -> comments
    
    def create_share_link(
        self,
        resource_type: str,
        resource_id: str,
        created_by: str,
        expires_days: Optional[int] = None,
        permissions: Optional[List[str]] = None
    ) -> str:
        """
        Create a share link.
        
        Args:
            resource_type: Type of resource (preset, template, result)
            resource_id: Resource ID
            created_by: Creator identifier
            expires_days: Optional expiration in days
            permissions: Optional permissions
            
        Returns:
            Share link ID
        """
        link_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_days:
            from datetime import timedelta
            expires_at = datetime.now() + timedelta(days=expires_days)
        
        share_link = ShareLink(
            link_id=link_id,
            resource_type=resource_type,
            resource_id=resource_id,
            created_by=created_by,
            created_at=datetime.now(),
            expires_at=expires_at,
            permissions=permissions or ["view"]
        )
        
        self._share_links[link_id] = share_link
        logger.info(f"Created share link: {link_id}")
        
        return link_id
    
    def get_share_link(self, link_id: str) -> Optional[ShareLink]:
        """Get share link by ID."""
        link = self._share_links.get(link_id)
        if link and link.is_active:
            if link.expires_at and datetime.now() > link.expires_at:
                return None
            link.access_count += 1
            return link
        return None
    
    def revoke_share_link(self, link_id: str) -> bool:
        """Revoke a share link."""
        link = self._share_links.get(link_id)
        if link:
            link.is_active = False
            return True
        return False
    
    def add_comment(
        self,
        resource_id: str,
        author: str,
        content: str,
        parent_id: Optional[str] = None
    ) -> str:
        """
        Add comment to resource.
        
        Args:
            resource_id: Resource ID
            author: Author identifier
            content: Comment content
            parent_id: Optional parent comment ID
            
        Returns:
            Comment ID
        """
        comment_id = str(uuid.uuid4())
        
        comment = Comment(
            comment_id=comment_id,
            resource_id=resource_id,
            author=author,
            content=content,
            created_at=datetime.now(),
            parent_id=parent_id
        )
        
        if resource_id not in self._comments:
            self._comments[resource_id] = []
        
        self._comments[resource_id].append(comment)
        logger.info(f"Added comment {comment_id} to resource {resource_id}")
        
        return comment_id
    
    def get_comments(self, resource_id: str) -> List[Dict[str, Any]]:
        """
        Get comments for resource.
        
        Args:
            resource_id: Resource ID
            
        Returns:
            List of comments
        """
        comments = self._comments.get(resource_id, [])
        return [
            {
                "comment_id": c.comment_id,
                "author": c.author,
                "content": c.content,
                "created_at": c.created_at.isoformat(),
                "parent_id": c.parent_id,
            }
            for c in comments
        ]
    
    def list_share_links(self, created_by: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List share links.
        
        Args:
            created_by: Filter by creator
            
        Returns:
            List of share links
        """
        links = list(self._share_links.values())
        
        if created_by:
            links = [l for l in links if l.created_by == created_by]
        
        return [
            {
                "link_id": l.link_id,
                "resource_type": l.resource_type,
                "resource_id": l.resource_id,
                "created_by": l.created_by,
                "created_at": l.created_at.isoformat(),
                "expires_at": l.expires_at.isoformat() if l.expires_at else None,
                "access_count": l.access_count,
                "is_active": l.is_active,
            }
            for l in links
        ]




