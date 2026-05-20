"""
Chat Base Classes and Interfaces
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4
from enum import Enum


class MessageRole(str, Enum):
    """Message role"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message:
    """Message model"""
    
    def __init__(
        self,
        role: MessageRole,
        content: str,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.conversation_id = conversation_id
        self.role = role
        self.content = content
        self.metadata = metadata or {}
        self.created_at = datetime.utcnow()


class Conversation:
    """Conversation model"""
    
    def __init__(
        self,
        user_id: str,
        title: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.id = str(uuid4())
        self.user_id = user_id
        self.title = title
        self.context = context or {}
        self.messages: List[Message] = []
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


class ChatContext:
    """Chat context"""
    
    def __init__(self, conversation_id: str):
        self.conversation_id = conversation_id
        self.messages: List[Message] = []
        self.metadata: Dict[str, Any] = {}
        self.max_context_length = 4096


class ChatBase(ABC):
    """Base interface for chat"""
    
    @abstractmethod
    async def create_conversation(
        self,
        user_id: str,
        initial_message: Optional[str] = None
    ) -> Conversation:
        """Create new conversation"""
        pass
    
    @abstractmethod
    async def send_message(
        self,
        conversation_id: str,
        message: str
    ) -> Message:
        """Send message in conversation"""
        pass
    
    @abstractmethod
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[Conversation]:
        """Get conversation by ID"""
        pass

