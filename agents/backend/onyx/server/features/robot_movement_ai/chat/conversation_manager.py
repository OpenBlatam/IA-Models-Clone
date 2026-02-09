"""
Conversation Manager - Gestión de conversaciones
"""
from typing import List, Dict, Any
from collections import defaultdict


class ConversationManager:
    """Gestor de conversaciones"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    async def add_message(
        self,
        user_id: str,
        user_message: str,
        bot_response: str
    ):
        """Agrega un mensaje a la conversación"""
        self.conversations[user_id].append({
            'role': 'user',
            'content': user_message,
            'timestamp': None  # Agregar timestamp
        })
        self.conversations[user_id].append({
            'role': 'assistant',
            'content': bot_response,
            'timestamp': None  # Agregar timestamp
        })
    
    async def get_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Obtiene el historial de conversación"""
        return self.conversations[user_id][-limit:]
    
    async def clear(self, user_id: str) -> bool:
        """Limpia el historial de conversación"""
        if user_id in self.conversations:
            del self.conversations[user_id]
            return True
        return False

