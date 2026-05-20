"""
Chat Service Implementation
"""

from typing import Optional
import logging
from datetime import datetime

from .base import (
    ChatBase,
    Conversation,
    Message,
    MessageRole,
    ChatContext
)

logger = logging.getLogger(__name__)


class ChatService(ChatBase):
    """Chat service implementation"""
    
    def __init__(self, llm_service=None, db=None, context_search=None, prompts_service=None):
        """Initialize chat service"""
        self.llm_service = llm_service
        self.db = db
        self.context_search = context_search
        self.prompts_service = prompts_service
        self._conversations: dict = {}
        self._contexts: dict = {}
    
    async def create_conversation(
        self,
        user_id: str,
        initial_message: Optional[str] = None
    ) -> Conversation:
        """Create new conversation"""
        try:
            conversation = Conversation(user_id=user_id)
            
            if initial_message:
                message = Message(
                    role=MessageRole.USER,
                    content=initial_message,
                    conversation_id=conversation.id
                )
                conversation.messages.append(message)
                
                # Get response from LLM
                response = await self._generate_response(conversation)
                conversation.messages.append(response)
            
            self._conversations[conversation.id] = conversation
            self._contexts[conversation.id] = ChatContext(conversation.id)
            
            return conversation
            
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def send_message(
        self,
        conversation_id: str,
        message: str
    ) -> Message:
        """Send message in conversation"""
        try:
            conversation = self._conversations.get(conversation_id)
            if not conversation:
                raise ValueError(f"Conversation {conversation_id} not found")
            
            # Create user message
            user_message = Message(
                role=MessageRole.USER,
                content=message,
                conversation_id=conversation_id
            )
            conversation.messages.append(user_message)
            conversation.updated_at = datetime.utcnow()
            
            # Update context
            context = self._contexts.get(conversation_id)
            if context:
                context.messages.append(user_message)
            
            # Generate response
            response = await self._generate_response(conversation)
            conversation.messages.append(response)
            
            if context:
                context.messages.append(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    async def get_conversation(
        self,
        conversation_id: str
    ) -> Optional[Conversation]:
        """Get conversation by ID"""
        return self._conversations.get(conversation_id)
    
    async def _generate_response(self, conversation: Conversation) -> Message:
        """Generate response using LLM"""
        try:
            if not self.llm_service:
                raise ValueError("LLM service not available")
            
            # Build context
            messages = [
                {"role": msg.role.value, "content": msg.content}
                for msg in conversation.messages[-10:]  # Last 10 messages
            ]
            
            # Get response from LLM
            response_content = await self.llm_service.generate(
                messages=messages,
                temperature=0.7
            )
            
            return Message(
                role=MessageRole.ASSISTANT,
                content=response_content,
                conversation_id=conversation.id
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

