"""
Messaging endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.messaging import MessagingService

router = APIRouter()
messaging_service = MessagingService()


@router.post("/send/{sender_id}")
async def send_message(
    sender_id: str,
    recipient_id: str,
    content: str,
    subject: Optional[str] = None
) -> Dict[str, Any]:
    """Enviar mensaje"""
    try:
        message = messaging_service.send_message(
            sender_id, recipient_id, content, subject
        )
        return {
            "id": message.id,
            "sent_at": message.created_at.isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversations/{user_id}")
async def get_conversations(user_id: str) -> Dict[str, Any]:
    """Obtener conversaciones del usuario"""
    try:
        conversations = messaging_service.get_conversations(user_id)
        return {
            "conversations": [
                {
                    "id": c.id,
                    "other_user_id": c.user2_id if c.user1_id == user_id else c.user1_id,
                    "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
                    "unread_count": c.unread_count_user1 if c.user1_id == user_id else c.unread_count_user2,
                }
                for c in conversations
            ],
            "total": len(conversations),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/conversation/{conversation_id}/{user_id}")
async def get_conversation(conversation_id: str, user_id: str) -> Dict[str, Any]:
    """Obtener conversación específica"""
    try:
        conversation = messaging_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {
            "id": conversation.id,
            "messages": [
                {
                    "id": m.id,
                    "sender_id": m.sender_id,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                    "status": m.status.value,
                }
                for m in conversation.messages
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/mark-read/{conversation_id}/{user_id}")
async def mark_as_read(conversation_id: str, user_id: str) -> Dict[str, Any]:
    """Marcar mensajes como leídos"""
    try:
        success = messaging_service.mark_as_read(conversation_id, user_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unread-count/{user_id}")
async def get_unread_count(user_id: str) -> Dict[str, Any]:
    """Obtener cantidad de mensajes no leídos"""
    try:
        count = messaging_service.get_unread_count(user_id)
        return {"unread_count": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




