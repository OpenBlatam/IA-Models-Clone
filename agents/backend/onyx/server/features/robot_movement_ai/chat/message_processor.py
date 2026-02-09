"""
Message Processor - Procesamiento de mensajes
"""
from typing import Dict, Any


class MessageProcessor:
    """Procesador de mensajes de chat"""
    
    async def process(self, message: str, user_id: str) -> Dict[str, Any]:
        """Procesa un mensaje"""
        return {
            'text': message,
            'user_id': user_id,
            'processed': True,
            'intent': self._extract_intent(message),
            'entities': self._extract_entities(message),
        }
    
    def _extract_intent(self, message: str) -> str:
        """Extrae la intención del mensaje"""
        # Implementación básica
        message_lower = message.lower()
        if 'move' in message_lower or 'go' in message_lower:
            return 'movement'
        elif 'stop' in message_lower:
            return 'stop'
        elif 'status' in message_lower:
            return 'status'
        return 'unknown'
    
    def _extract_entities(self, message: str) -> Dict[str, Any]:
        """Extrae entidades del mensaje"""
        # Implementación básica
        return {}

