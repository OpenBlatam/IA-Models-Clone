"""
Interfaces module for Lovable Community

Defines protocols/interfaces for service contracts.
Following Interface Segregation Principle.
"""

from typing import Protocol, Optional, List
from datetime import datetime

from ..models import PublishedChat, ChatRemix, ChatVote


class IChatRepository(Protocol):
    """Protocol for chat repository operations."""
    
    def get_by_id(self, chat_id: str) -> Optional[PublishedChat]:
        """Get chat by ID."""
        ...
    
    def create(self, **kwargs) -> PublishedChat:
        """Create a new chat."""
        ...
    
    def get_public_chats(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: str = "score",
        order: str = "desc"
    ) -> List[PublishedChat]:
        """Get public chats."""
        ...


class IRankingService(Protocol):
    """Protocol for ranking service operations."""
    
    def calculate_score(
        self,
        vote_count: int,
        remix_count: int,
        view_count: int,
        created_at: datetime,
        base_score: float = 0.0
    ) -> float:
        """Calculate ranking score."""
        ...


class IChatService(Protocol):
    """Protocol for chat service operations."""
    
    def publish_chat(
        self,
        user_id: str,
        title: str,
        chat_content: str,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        is_public: bool = True
    ) -> PublishedChat:
        """Publish a new chat."""
        ...
    
    def get_chat(self, chat_id: str, user_id: Optional[str] = None) -> Optional[PublishedChat]:
        """Get a chat by ID."""
        ...


__all__ = [
    "IChatRepository",
    "IRankingService",
    "IChatService",
]













