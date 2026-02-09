"""
Conversation Manager - Gestión de conversaciones
"""

from typing import List, Dict, Any, Optional
from db.service import DatabaseService


class ConversationManager:
    """Gestor de conversaciones"""

    def __init__(self, db_service: Optional[DatabaseService] = None):
        """Inicializa el gestor de conversaciones"""
        self.db_service = db_service
        self._conversations: Dict[str, List[Dict[str, Any]]] = {}

    async def get_or_create(self, user_id: str) -> Dict[str, Any]:
        """Obtiene o crea una conversación"""
        if user_id not in self._conversations:
            self._conversations[user_id] = []
        return {"user_id": user_id, "messages": self._conversations[user_id]}

    async def add_message(self, user_id: str, message: str, response: Dict[str, Any]) -> None:
        """Añade un mensaje a la conversación"""
        conversation = await self.get_or_create(user_id)
        conversation["messages"].append({
            "user": message,
            "assistant": response.get("response", ""),
            "timestamp": response.get("timestamp")
        })

    async def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene historial de conversación"""
        conversation = await self.get_or_create(user_id)
        return conversation["messages"][-limit:]

