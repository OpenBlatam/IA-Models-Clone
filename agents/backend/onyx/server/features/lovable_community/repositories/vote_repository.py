"""
Vote Repository

Repository for ChatVote model with specialized queries.
"""

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import desc

from ..models import ChatVote
from .base import BaseRepository
from .validation_helpers import (
    validate_string_id,
    validate_positive_integer,
    validate_list_of_string_ids,
    execute_with_error_handling
)


class VoteRepository(BaseRepository[ChatVote]):
    """
    Repository for ChatVote with specialized query methods.
    """
    
    def __init__(self, db: Session):
        super().__init__(db, ChatVote)
    
    def get_by_chat_id(self, chat_id: str) -> List[ChatVote]:
        """
        Get all votes for a chat.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            List of votes
            
        Raises:
            ValueError: If chat_id is None or empty
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        return self.db.query(ChatVote).filter(
            ChatVote.chat_id == chat_id
        ).order_by(desc(ChatVote.created_at)).all()
    
    def get_by_user_id(self, user_id: str, limit: int = 100) -> List[ChatVote]:
        """
        Get votes by user ID.
        
        Args:
            user_id: User ID
            limit: Maximum number of votes (must be > 0)
            
        Returns:
            List of votes
            
        Raises:
            ValueError: If user_id is None or empty, or limit is invalid
        """
        user_id = validate_string_id(user_id, "user_id")
        limit = validate_positive_integer(limit, "limit")
        
        return self.db.query(ChatVote).filter(
            ChatVote.user_id == user_id
        ).order_by(desc(ChatVote.created_at)).limit(limit).all()
    
    def get_user_vote(self, chat_id: str, user_id: str) -> Optional[ChatVote]:
        """
        Get user's vote for a specific chat.
        
        Args:
            chat_id: Chat ID
            user_id: User ID
            
        Returns:
            Vote or None if not found
            
        Raises:
            ValueError: If chat_id or user_id is None or empty
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        user_id = validate_string_id(user_id, "user_id")
        
        return self.db.query(ChatVote).filter(
            ChatVote.chat_id == chat_id,
            ChatVote.user_id == user_id
        ).first()
    
    def count_by_chat_id(self, chat_id: str, vote_type: Optional[str] = None) -> int:
        """
        Count votes for a chat.
        
        Args:
            chat_id: Chat ID
            vote_type: Filter by vote type (optional, must be "upvote" or "downvote" if provided)
            
        Returns:
            Count of votes
            
        Raises:
            ValueError: If chat_id is None or empty, or vote_type is invalid
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        if vote_type is not None:
            if not isinstance(vote_type, str) or vote_type.strip().lower() not in ("upvote", "downvote"):
                raise ValueError(f"vote_type must be 'upvote' or 'downvote', got '{vote_type}'")
            vote_type = vote_type.strip().lower()
        
        query = self.db.query(ChatVote).filter(ChatVote.chat_id == chat_id)
        
        if vote_type:
            query = query.filter(ChatVote.vote_type == vote_type)
        
        return query.count()
    
    def get_user_votes_batch(
        self,
        chat_ids: List[str],
        user_id: str
    ) -> Dict[str, ChatVote]:
        """
        Get user's votes for multiple chats in a single query.
        
        Args:
            chat_ids: List of chat IDs
            user_id: User ID
            
        Returns:
            Dictionary mapping chat_id to ChatVote
            
        Raises:
            ValueError: If chat_ids is None or contains invalid entries, or user_id is invalid
        """
        valid_chat_ids = validate_list_of_string_ids(chat_ids, "chat_ids")
        user_id = validate_string_id(user_id, "user_id")
        
        if not valid_chat_ids:
            return {}
        
        votes = self.db.query(ChatVote).filter(
            ChatVote.chat_id.in_(valid_chat_ids),
            ChatVote.user_id == user_id
        ).all()
        
        return {vote.chat_id: vote for vote in votes}
    
    def delete_by_chat_id(self, chat_id: str) -> int:
        """
        Delete all votes for a chat in a single query.
        
        Args:
            chat_id: Chat ID
            
        Returns:
            Number of deleted votes
            
        Raises:
            ValueError: If chat_id is None or empty
            DatabaseError: If deletion fails
        """
        chat_id = validate_string_id(chat_id, "chat_id")
        
        return execute_with_error_handling(
            self.db,
            lambda: self.db.query(ChatVote).filter(
                ChatVote.chat_id == chat_id
            ).delete(synchronize_session=False),
            "delete",
            "vote",
            chat_id
        )


