"""
Base Chat Controller - Clase base para controladores de chat
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional


class BaseChatController(ABC):
    """Clase base abstracta para controladores de chat"""
    
    @abstractmethod
    async def process_message(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Procesa un mensaje y retorna respuesta"""
        pass
    
    @abstractmethod
    async def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Obtiene el historial de conversación"""
        pass
    
    @abstractmethod
    async def clear_conversation(self, user_id: str) -> bool:
        """Limpia el historial de conversación"""
        pass

