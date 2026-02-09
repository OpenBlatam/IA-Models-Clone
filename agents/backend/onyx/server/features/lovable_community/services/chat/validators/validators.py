"""
Chat validation utilities

Extracted validation logic for chat operations.
"""

from typing import Optional, List
from ....exceptions import InvalidChatError
from ....helpers.validation_common import ensure_not_empty_string
from ....helpers.string_normalization import normalize_list_to_lower
from ....services.error_handling import convert_validation_error


class ChatValidator:
    """Validation utilities for chat operations."""
    
    @staticmethod
    def validate_chat_id(chat_id: str) -> str:
        """Validate and normalize chat ID."""
        return convert_validation_error(chat_id, "chat_id", "Chat ID cannot be empty")
    
    @staticmethod
    def validate_user_id(user_id: str) -> str:
        """Validate and normalize user ID."""
        return convert_validation_error(user_id, "user_id", "User ID cannot be empty")
    
    @staticmethod
    def validate_title(title: str) -> str:
        """Validate and normalize title."""
        return convert_validation_error(title, "title", "Title cannot be empty")
    
    @staticmethod
    def validate_chat_content(chat_content: str) -> str:
        """Validate and normalize chat content."""
        return convert_validation_error(chat_content, "chat_content", "Chat content cannot be empty")
    
    @staticmethod
    def process_tags(tags: Optional[List[str]]) -> Optional[str]:
        """Process and format tags list to string."""
        if not tags:
            return None
        
        normalized_tags = normalize_list_to_lower(tags)
        if not normalized_tags:
            return None
        
        from ....constants import MAX_TAGS_PER_CHAT
        unique_tags = list(dict.fromkeys(normalized_tags))[:MAX_TAGS_PER_CHAT]
        return ",".join(unique_tags) if unique_tags else None
    
    @staticmethod
    def ensure_ownership(chat, user_id: str) -> None:
        """Ensure user owns the chat."""
        if chat.user_id != user_id:
            raise InvalidChatError("User is not the owner of this chat")






