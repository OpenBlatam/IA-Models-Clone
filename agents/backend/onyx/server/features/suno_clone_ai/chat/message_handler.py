"""
Message Handler - Procesamiento de mensajes
"""

from typing import Dict, Any, Optional
from datetime import datetime
from llm.service import LLMService
from context.service import ContextService


class MessageHandler:
    """Manejador de mensajes"""

    def __init__(
        self,
        llm_service: Optional[LLMService] = None,
        context_service: Optional[ContextService] = None
    ):
        """Inicializa el manejador de mensajes"""
        self.llm_service = llm_service
        self.context_service = context_service

    async def process(self, message: str, conversation: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Procesa un mensaje y genera respuesta"""
        # Obtener contexto si está disponible
        context = None
        if self.context_service:
            context = await self.context_service.get_context(conversation.get("user_id"))

        # Generar respuesta con LLM
        response = ""
        if self.llm_service:
            messages = [{"role": "user", "content": message}]
            response = await self.llm_service.chat(messages, **kwargs)

        return {
            "response": response,
            "timestamp": datetime.utcnow().isoformat(),
            "context": context
        }

