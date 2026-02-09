"""
Base Chat - Clase base para chat
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseChat(ABC):
    """Clase base abstracta para sistemas de chat"""

    @abstractmethod
    async def send_message(self, message: str, user_id: str, **kwargs) -> Dict[str, Any]:
        """Envía un mensaje y obtiene respuesta"""
        pass

    @abstractmethod
    async def get_history(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Obtiene historial de conversación"""
        pass

