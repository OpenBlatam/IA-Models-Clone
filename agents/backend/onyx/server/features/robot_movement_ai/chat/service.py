"""
Chat Service - Servicio principal de chat
"""
from typing import Dict, List, Any, Optional
from .message_processor import MessageProcessor
from .conversation_manager import ConversationManager
from .websocket_handler import WebSocketHandler


class ChatService:
    """Servicio principal de chat"""
    
    def __init__(self):
        self.message_processor = MessageProcessor()
        self.conversation_manager = ConversationManager()
        self.websocket_handler = WebSocketHandler()
    
    async def process_message(
        self,
        message: str,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Procesa un mensaje y retorna respuesta"""
        # Procesar mensaje
        processed = await self.message_processor.process(message, user_id)
        
        # Obtener contexto de conversación
        history = await self.conversation_manager.get_history(user_id)
        
        # Generar respuesta (usando LLM)
        response = await self._generate_response(processed, history, context)
        
        # Guardar en historial
        await self.conversation_manager.add_message(user_id, message, response)
        
        return response
    
    async def _generate_response(
        self,
        processed_message: Dict,
        history: List[Dict],
        context: Optional[Dict] = None
    ) -> str:
        """Genera respuesta usando LLM"""
        # Implementación con LLM
        return "Response from chat service"
    
    async def get_conversation_history(self, user_id: str) -> List[Dict]:
        """Obtiene el historial de conversación"""
        return await self.conversation_manager.get_history(user_id)
    
    async def clear_conversation(self, user_id: str) -> bool:
        """Limpia el historial de conversación"""
        return await self.conversation_manager.clear(user_id)

