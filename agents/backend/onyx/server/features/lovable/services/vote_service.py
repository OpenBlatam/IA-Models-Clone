"""
Vote Service for vote-related business logic.
"""

from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
import logging

from ..repositories.chat_repository import ChatRepository
from ..exceptions import NotFoundError, ValidationError, ConflictError
from ..utils.decorators import log_execution_time, handle_errors
from ..utils.service_base import BaseService
from ..utils.validators import validate_vote_type, validate_chat_id, validate_user_id

logger = logging.getLogger(__name__)

# Try to import VoteRepository if it exists
try:
    from ..repositories.vote_repository import VoteRepository
    HAS_VOTE_REPOSITORY = True
except ImportError:
    HAS_VOTE_REPOSITORY = False


class VoteService(BaseService):
    """Service for vote operations."""
    
    def __init__(self, db: Session):
        """Initialize vote service."""
        super().__init__(db)
        self.chat_repo = ChatRepository(db)
        if HAS_VOTE_REPOSITORY:
            self.vote_repo = VoteRepository(db)
        else:
            self.vote_repo = None
    
    @log_execution_time
    @handle_errors
    def increment_vote(
        self,
        chat_id: str,
        vote_type: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Increment vote count for a chat.
        
        Args:
            chat_id: Chat ID
            vote_type: Vote type ('upvote' or 'downvote')
            user_id: User ID
            
        Returns:
            Dictionary with vote result
            
        Raises:
            NotFoundError: If chat doesn't exist
            ValidationError: If vote_type is invalid
            ConflictError: If user already voted
        """
        # Validate inputs using BaseService helper
        vote_type = self.validate_with_conversion(validate_vote_type, vote_type)
        chat_id = self.validate_with_conversion(validate_chat_id, chat_id)
        user_id = self.validate_with_conversion(validate_user_id, user_id)
        
        # Check if chat exists
        self.get_or_raise_not_found(self.chat_repo, chat_id, "Chat")
        
        # Check if user already voted (if vote repository exists)
        if self.vote_repo:
            existing_vote = self.vote_repo.get_by_user_and_chat(user_id, chat_id)
            if existing_vote:
                raise ConflictError("User has already voted on this chat")
            
            # Create vote record
            vote_data = {
                "id": f"{user_id}:{chat_id}",
                "user_id": user_id,
                "chat_id": chat_id,
                "vote_type": vote_type
            }
            vote = self.vote_repo.create(vote_data)
        else:
            vote = None
        
        # Update chat vote count
        if vote_type == "upvote":
            self.chat_repo.increment_vote(chat_id)
        else:
            self.chat_repo.decrement_vote(chat_id)
        
        return {
            "success": True,
            "message": f"{vote_type} recorded successfully",
            "vote": self.serialize_model(vote) if vote else None
        }
    






