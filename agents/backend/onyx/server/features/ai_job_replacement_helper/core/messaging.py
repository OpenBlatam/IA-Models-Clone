"""
Messaging Service - Sistema de mensajería entre usuarios
=========================================================

Sistema de mensajería directa entre usuarios para networking y apoyo mutuo.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class MessageStatus(str, Enum):
    """Estado del mensaje"""
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"
    ARCHIVED = "archived"


@dataclass
class Message:
    """Mensaje entre usuarios"""
    id: str
    sender_id: str
    recipient_id: str
    subject: Optional[str] = None
    content: str = ""
    status: MessageStatus = MessageStatus.SENT
    created_at: datetime = field(default_factory=datetime.now)
    read_at: Optional[datetime] = None
    parent_message_id: Optional[str] = None  # Para conversaciones


@dataclass
class Conversation:
    """Conversación entre dos usuarios"""
    id: str
    user1_id: str
    user2_id: str
    messages: List[Message] = field(default_factory=list)
    last_message_at: Optional[datetime] = None
    unread_count_user1: int = 0
    unread_count_user2: int = 0
    created_at: datetime = field(default_factory=datetime.now)


class MessagingService:
    """Servicio de mensajería"""
    
    def __init__(self):
        """Inicializar servicio"""
        self.conversations: Dict[str, Conversation] = {}  # conversation_id -> Conversation
        self.user_conversations: Dict[str, List[str]] = {}  # user_id -> [conversation_ids]
        logger.info("MessagingService initialized")
    
    def send_message(
        self,
        sender_id: str,
        recipient_id: str,
        content: str,
        subject: Optional[str] = None,
        parent_message_id: Optional[str] = None
    ) -> Message:
        """Enviar mensaje"""
        # Buscar o crear conversación
        conversation = self._get_or_create_conversation(sender_id, recipient_id)
        
        message = Message(
            id=f"msg_{sender_id}_{int(datetime.now().timestamp())}",
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=subject,
            content=content,
            parent_message_id=parent_message_id,
        )
        
        conversation.messages.append(message)
        conversation.last_message_at = datetime.now()
        
        # Actualizar contador de no leídos
        if sender_id == conversation.user1_id:
            conversation.unread_count_user2 += 1
        else:
            conversation.unread_count_user1 += 1
        
        logger.info(f"Message sent from {sender_id} to {recipient_id}")
        return message
    
    def get_conversations(self, user_id: str) -> List[Conversation]:
        """Obtener conversaciones del usuario"""
        conversation_ids = self.user_conversations.get(user_id, [])
        conversations = [
            self.conversations[cid] for cid in conversation_ids
            if cid in self.conversations
        ]
        
        # Ordenar por último mensaje
        conversations.sort(
            key=lambda c: c.last_message_at or c.created_at,
            reverse=True
        )
        
        return conversations
    
    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """Obtener conversación específica"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return None
        
        # Verificar que el usuario es parte de la conversación
        if user_id not in [conversation.user1_id, conversation.user2_id]:
            return None
        
        return conversation
    
    def mark_as_read(self, conversation_id: str, user_id: str) -> bool:
        """Marcar mensajes como leídos"""
        conversation = self.conversations.get(conversation_id)
        if not conversation:
            return False
        
        if user_id not in [conversation.user1_id, conversation.user2_id]:
            return False
        
        # Marcar mensajes no leídos como leídos
        now = datetime.now()
        for message in conversation.messages:
            if message.recipient_id == user_id and message.status != MessageStatus.READ:
                message.status = MessageStatus.READ
                message.read_at = now
        
        # Resetear contador
        if user_id == conversation.user1_id:
            conversation.unread_count_user1 = 0
        else:
            conversation.unread_count_user2 = 0
        
        return True
    
    def get_unread_count(self, user_id: str) -> int:
        """Obtener cantidad total de mensajes no leídos"""
        conversations = self.get_conversations(user_id)
        total = 0
        
        for conversation in conversations:
            if user_id == conversation.user1_id:
                total += conversation.unread_count_user1
            else:
                total += conversation.unread_count_user2
        
        return total
    
    def _get_or_create_conversation(
        self,
        user1_id: str,
        user2_id: str
    ) -> Conversation:
        """Obtener o crear conversación"""
        # Crear ID consistente (ordenar IDs para que sea único)
        user_ids = sorted([user1_id, user2_id])
        conversation_id = f"conv_{user_ids[0]}_{user_ids[1]}"
        
        if conversation_id not in self.conversations:
            conversation = Conversation(
                id=conversation_id,
                user1_id=user_ids[0],
                user2_id=user_ids[1],
            )
            self.conversations[conversation_id] = conversation
            
            # Agregar a lista de conversaciones de cada usuario
            for user_id in user_ids:
                if user_id not in self.user_conversations:
                    self.user_conversations[user_id] = []
                if conversation_id not in self.user_conversations[user_id]:
                    self.user_conversations[user_id].append(conversation_id)
        
        return self.conversations[conversation_id]




