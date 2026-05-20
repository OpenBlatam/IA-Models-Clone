"""
Chat Engagement Operations

Handles engagement operations: votes, views, and remixes.
"""

import logging
from typing import Optional, Tuple
from ....models import PublishedChat, ChatVote, ChatRemix
from ....exceptions import InvalidChatError
from ....helpers import generate_id, get_current_timestamp
from ....api.route_helpers import (
    validate_required_string,
    validate_required_object,
    normalize_optional_string
)
from ....helpers.string_normalization import normalize_to_lower

logger = logging.getLogger(__name__)


class VoteHandler:
    """Handles voting operations."""
    
    @staticmethod
    def calculate_vote_increment(
        existing_vote: Optional[ChatVote],
        new_vote_type: str
    ) -> Tuple[int, Optional[ChatVote]]:
        """
        Calculate vote increment based on existing vote and new vote type.
        
        Logic:
        - If no existing vote: +1 for upvote, -1 for downvote
        - If same vote type: 0 (no change)
        - If switching from upvote to downvote: -2 (remove +1, add -1)
        - If switching from downvote to upvote: +2 (remove -1, add +1)
        
        Args:
            existing_vote: Existing vote or None
            new_vote_type: New vote type ('upvote' or 'downvote')
            
        Returns:
            Tuple of (vote_increment, existing_vote_or_none)
            
        Raises:
            ValueError: If new_vote_type is None or empty
            InvalidChatError: If new_vote_type is invalid
        """
        if not new_vote_type:
            raise ValueError("new_vote_type cannot be None or empty")
        
        VoteHandler.validate_vote_type(new_vote_type)
        new_vote_type = normalize_to_lower(new_vote_type)
        
        if existing_vote:
            existing_type = normalize_to_lower(existing_vote.vote_type) if existing_vote.vote_type else None
            
            if existing_type == new_vote_type:
                return 0, existing_vote
            
            if existing_type == "upvote" and new_vote_type == "downvote":
                return -2, existing_vote
            elif existing_type == "downvote" and new_vote_type == "upvote":
                return 2, existing_vote
            else:
                # Unknown existing vote type, treat as new vote
                increment = 1 if new_vote_type == "upvote" else -1
                return increment, None
        else:
            increment = 1 if new_vote_type == "upvote" else -1
            return increment, None
    
    @staticmethod
    def validate_vote_type(vote_type: str) -> None:
        """
        Validate vote type.
        
        Args:
            vote_type: Vote type to validate
            
        Raises:
            InvalidChatError: If vote_type is not 'upvote' or 'downvote'
            ValueError: If vote_type is None or empty
        """
        if not vote_type:
            raise ValueError("Vote type cannot be None or empty")
        
        vote_type = normalize_to_lower(vote_type)
        if vote_type not in ("upvote", "downvote"):
            raise InvalidChatError(
                f"Invalid vote type: '{vote_type}'. Must be 'upvote' or 'downvote'"
            )


class ViewHandler:
    """Handles view operations."""
    
    @staticmethod
    def create_view_record(view_repository, chat_id: str, user_id: Optional[str]) -> str:
        """
        Create a view record.
        
        Args:
            view_repository: View repository
            chat_id: Chat ID
            user_id: Optional user ID
            
        Returns:
            View ID
            
        Raises:
            ValueError: If chat_id is None or empty, or view_repository is None
        """
        chat_id = validate_required_string(chat_id, "chat_id")
        validate_required_object(view_repository, "view_repository")
        user_id = normalize_optional_string(user_id)
        
        view_id = generate_id()
        view_repository.create(
            id=view_id,
            chat_id=chat_id,
            user_id=user_id,
            created_at=get_current_timestamp()
        )
        return view_id


class RemixHandler:
    """Handles remix operations."""
    
    @staticmethod
    def create_remix_record(
        remix_repository,
        original_chat_id: str,
        remix_chat_id: str,
        user_id: str
    ) -> ChatRemix:
        """
        Create a remix record.
        
        Args:
            remix_repository: Remix repository
            original_chat_id: Original chat ID
            remix_chat_id: Remix chat ID
            user_id: User ID
            
        Returns:
            Remix record
            
        Raises:
            ValueError: If any required parameter is None or empty, or remix_repository is None
        """
        original_chat_id = validate_required_string(original_chat_id, "original_chat_id")
        remix_chat_id = validate_required_string(remix_chat_id, "remix_chat_id")
        user_id = validate_required_string(user_id, "user_id")
        validate_required_object(remix_repository, "remix_repository")
        
        remix_id = generate_id()
        return remix_repository.create(
            id=remix_id,
            original_chat_id=original_chat_id,
            remix_chat_id=remix_chat_id,
            user_id=user_id,
            created_at=get_current_timestamp()
        )






