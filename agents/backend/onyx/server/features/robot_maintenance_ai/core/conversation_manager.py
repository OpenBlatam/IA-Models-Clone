"""
Conversation Manager for maintaining context in maintenance conversations.
"""

import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict

from ..utils.file_helpers import get_iso_timestamp

logger = logging.getLogger(__name__)


class ConversationManager:
    """
    Manages conversation history and context for maintenance sessions.
    """
    
    def __init__(self, max_history_length: int = 100):
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.max_history_length = max_history_length
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Add a message to the conversation.
        
        Args:
            conversation_id: Unique conversation identifier
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Additional metadata
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": get_iso_timestamp(),
            "metadata": metadata or {}
        }
        
        self.conversations[conversation_id].append(message)
        
        if len(self.conversations[conversation_id]) > self.max_history_length:
            self.conversations[conversation_id] = self.conversations[conversation_id][-self.max_history_length:]
    
    def get_conversation(self, conversation_id: str) -> List[Dict[str, Any]]:
        """
        Get full conversation history.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of messages
        """
        return self.conversations.get(conversation_id, [])
    
    def get_context(
        self,
        conversation_id: str,
        max_messages: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Get recent conversation context.
        
        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        conversation = self.conversations.get(conversation_id, [])
        return conversation[-max_messages:]
    
    def clear_conversation(self, conversation_id: str):
        """
        Clear conversation history.
        
        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
    
    def get_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation summary.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            Summary dictionary
        """
        conversation = self.conversations.get(conversation_id, [])
        
        if not conversation:
            return {
                "conversation_id": conversation_id,
                "message_count": 0,
                "topics": [],
                "last_message": None
            }
        
        topics = []
        for message in conversation:
            if message["role"] == "user":
                content_lower = message["content"].lower()
                if "mantenimiento" in content_lower:
                    topics.append("mantenimiento")
                if "diagnóstico" in content_lower or "problema" in content_lower:
                    topics.append("diagnóstico")
                if "prevención" in content_lower:
                    topics.append("prevención")
        
        return {
            "conversation_id": conversation_id,
            "message_count": len(conversation),
            "topics": list(set(topics)),
            "last_message": conversation[-1]["timestamp"] if conversation else None
        }






