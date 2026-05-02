"""
Chat Service for Unified AI Model
Provides conversational AI with memory and context management
"""

import asyncio
import uuid
import logging
from typing import Dict, Any, List, Optional, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

from ..config import get_config
from .llm_service import LLMService, LLMResponse, get_llm_service

logger = logging.getLogger(__name__)


class MessageRole(str, Enum):
    """Message roles in a conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_id": self.message_id,
            "metadata": self.metadata
        }
    
    def to_api_format(self) -> Dict[str, str]:
        """Convert to format expected by LLM API."""
        return {
            "role": self.role.value,
            "content": self.content
        }


@dataclass
class Conversation:
    """A conversation with message history."""
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[Message] = field(default_factory=list)
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, role: MessageRole, content: str, metadata: Optional[Dict] = None) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.messages.append(message)
        self.updated_at = datetime.now()
        return message
    
    def get_messages_for_api(self, max_messages: Optional[int] = None) -> List[Dict[str, str]]:
        """Get messages in format for LLM API."""
        messages = []
        
        # Add system prompt if exists
        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })
        
        # Get recent messages
        msg_list = self.messages
        if max_messages and len(msg_list) > max_messages:
            msg_list = msg_list[-max_messages:]
        
        for msg in msg_list:
            messages.append(msg.to_api_format())
        
        return messages
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_id": self.conversation_id,
            "messages": [m.to_dict() for m in self.messages],
            "system_prompt": self.system_prompt,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "message_count": len(self.messages),
            "metadata": self.metadata
        }


@dataclass
class ChatResponse:
    """Response from a chat request."""
    message: Message
    conversation_id: str
    model: str
    usage: Optional[Dict[str, Any]] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "message": self.message.to_dict() if self.message else None,
            "conversation_id": self.conversation_id,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "success": self.error is None
        }


class ChatService:
    """
    Chat service with conversation memory and context management.
    
    Features:
    - Conversation history management
    - Context window optimization
    - Multiple conversation support
    - Streaming responses
    - System prompt customization
    """
    
    def __init__(
        self,
        llm_service: Optional[LLMService] = None
    ):
        self.llm_service = llm_service or get_llm_service()
        self.config = get_config()
        
        # Conversation storage (in-memory)
        self.conversations: Dict[str, Conversation] = {}
        
        # Default settings
        self.default_system_prompt = self.config.chat.default_system_prompt
        self.max_history_length = self.config.chat.max_history_length
        self.default_model = self.config.default_model
        
        logger.info("Chat Service initialized")
    
    def create_conversation(
        self,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Conversation:
        """
        Create a new conversation.
        
        Args:
            system_prompt: System prompt for the conversation
            model: Model to use for this conversation
            conversation_id: Optional custom conversation ID
            metadata: Optional metadata
            
        Returns:
            New Conversation instance
        """
        conversation = Conversation(
            conversation_id=conversation_id or str(uuid.uuid4()),
            system_prompt=system_prompt or self.default_system_prompt,
            model=model or self.default_model,
            metadata=metadata or {}
        )
        
        self.conversations[conversation.conversation_id] = conversation
        logger.info(f"Created conversation: {conversation.conversation_id}")
        
        return conversation
    
    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """Get a conversation by ID."""
        return self.conversations.get(conversation_id)
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation."""
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            logger.info(f"Deleted conversation: {conversation_id}")
            return True
        return False
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """List all conversations."""
        return [
            {
                "conversation_id": c.conversation_id,
                "message_count": len(c.messages),
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
                "model": c.model
            }
            for c in self.conversations.values()
        ]
    
    async def chat(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> ChatResponse:
        """
        Send a chat message and get a response.
        
        Args:
            message: User message
            conversation_id: Conversation ID (creates new if not provided)
            system_prompt: Override system prompt
            model: Override model
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters
            
        Returns:
            ChatResponse with assistant's reply
        """
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        else:
            conversation = self.create_conversation(
                system_prompt=system_prompt,
                model=model,
                conversation_id=conversation_id
            )
        
        # Add user message
        user_message = conversation.add_message(
            role=MessageRole.USER,
            content=message
        )
        
        # Prepare messages for API
        messages = conversation.get_messages_for_api(
            max_messages=self.max_history_length
        )
        
        # Override system prompt if provided
        if system_prompt and messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        
        # Generate response
        try:
            response = await self.llm_service.generate_with_messages(
                messages=messages,
                model=model or conversation.model,
                temperature=temperature,
                max_tokens=max_tokens,
                cache_enabled=False,  # Don't cache chat responses
                **kwargs
            )
            
            if response.error:
                return ChatResponse(
                    message=None,
                    conversation_id=conversation.conversation_id,
                    model=model or conversation.model,
                    error=response.error
                )
            
            # Add assistant message to conversation
            assistant_message = conversation.add_message(
                role=MessageRole.ASSISTANT,
                content=response.content,
                metadata={
                    "usage": response.usage,
                    "latency_ms": response.latency_ms
                }
            )
            
            return ChatResponse(
                message=assistant_message,
                conversation_id=conversation.conversation_id,
                model=response.model,
                usage=response.usage,
                latency_ms=response.latency_ms
            )
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return ChatResponse(
                message=None,
                conversation_id=conversation.conversation_id,
                model=model or conversation.model,
                error=str(e)
            )
    
    async def chat_stream(
        self,
        message: str,
        conversation_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Send a chat message and stream the response.
        
        Yields:
            Response content chunks
        """
        # Get or create conversation
        if conversation_id and conversation_id in self.conversations:
            conversation = self.conversations[conversation_id]
        else:
            conversation = self.create_conversation(
                system_prompt=system_prompt,
                model=model,
                conversation_id=conversation_id
            )
        
        # Add user message
        conversation.add_message(
            role=MessageRole.USER,
            content=message
        )
        
        # Prepare messages for API
        messages = conversation.get_messages_for_api(
            max_messages=self.max_history_length
        )
        
        # Override system prompt if provided
        if system_prompt and messages and messages[0]["role"] == "system":
            messages[0]["content"] = system_prompt
        
        # Collect response for saving to conversation
        collected_content = []
        
        try:
            async for chunk in self.llm_service.client.stream_chat_completion(
                model=model or conversation.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            ):
                collected_content.append(chunk)
                yield chunk
            
            # Save complete response to conversation
            full_content = "".join(collected_content)
            conversation.add_message(
                role=MessageRole.ASSISTANT,
                content=full_content
            )
            
        except Exception as e:
            logger.error(f"Stream chat error: {e}")
            yield f"\n\n[Error: {str(e)}]"
    
    async def regenerate_last_response(
        self,
        conversation_id: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> ChatResponse:
        """
        Regenerate the last assistant response.
        
        Args:
            conversation_id: Conversation ID
            model: Optional different model
            temperature: Optional different temperature
            
        Returns:
            New ChatResponse
        """
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return ChatResponse(
                message=None,
                conversation_id=conversation_id,
                model="",
                error="Conversation not found"
            )
        
        # Remove last assistant message if exists
        if conversation.messages and conversation.messages[-1].role == MessageRole.ASSISTANT:
            conversation.messages.pop()
        
        # Get last user message
        user_message = None
        for msg in reversed(conversation.messages):
            if msg.role == MessageRole.USER:
                user_message = msg.content
                break
        
        if not user_message:
            return ChatResponse(
                message=None,
                conversation_id=conversation_id,
                model="",
                error="No user message to regenerate from"
            )
        
        # Remove the user message too (it will be re-added in chat)
        if conversation.messages and conversation.messages[-1].role == MessageRole.USER:
            conversation.messages.pop()
        
        # Regenerate
        return await self.chat(
            message=user_message,
            conversation_id=conversation_id,
            model=model,
            temperature=temperature or 0.7
        )
    
    def get_conversation_history(
        self,
        conversation_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get conversation history."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return []
        
        messages = conversation.messages
        if limit:
            messages = messages[-limit:]
        
        return [m.to_dict() for m in messages]
    
    def clear_conversation_history(self, conversation_id: str) -> bool:
        """Clear conversation history."""
        conversation = self.get_conversation(conversation_id)
        if not conversation:
            return False
        
        conversation.clear_history()
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chat service statistics."""
        total_messages = sum(len(c.messages) for c in self.conversations.values())
        
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "active_models": list(set(c.model for c in self.conversations.values() if c.model))
        }
    
    async def close(self) -> None:
        """Close the chat service."""
        self.conversations.clear()
        logger.info("Chat Service closed")


# Singleton instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service


async def close_chat_service() -> None:
    """Close the global chat service."""
    global _chat_service
    if _chat_service:
        await _chat_service.close()
        _chat_service = None



