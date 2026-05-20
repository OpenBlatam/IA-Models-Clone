"""
OpenRouter Client
================
Cliente principal para integración con OpenRouter API.
"""

import os
from typing import Dict, Any, Optional, List
from ...core.types import JSONDict, MessageList, ModelList
from ...core.datetime_utils import get_utc_iso_timestamp

from .api_client import APIClient

try:
    import structlog
    logger = structlog.get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Cliente para OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente OpenRouter.
        
        Args:
            api_key: API key de OpenRouter. Si no se proporciona, se busca en env.
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.warning("OpenRouter API key not configured")
        
        self.api_client = APIClient(self.api_key)
    
    async def generate_text(
        self,
        prompt: str = "",
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        messages: Optional[MessageList] = None,
    ) -> JSONDict:
        """
        Generar texto usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario (ignored if messages provided)
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura para generación
            messages: Lista de mensajes para conversación (opcional, preferred over prompt)
        
        Returns:
            Respuesta de la API con el texto generado
            
        Raises:
            APIError: Si hay un error en la llamada a la API
        """
        if messages is None:
            if not prompt:
                from ...core.exceptions import ValidationError
                raise ValidationError("Either prompt or messages must be provided")
            messages = [{"role": "user", "content": prompt}]
        
        # Validate messages structure
        if not isinstance(messages, list) or not messages:
            from ...core.exceptions import ValidationError
            raise ValidationError("Messages must be a non-empty list")
        
        # Validate each message has required fields
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                from ...core.exceptions import ValidationError
                raise ValidationError(f"Message {i} must be a dictionary")
            if "role" not in msg or "content" not in msg:
                from ...core.exceptions import ValidationError
                raise ValidationError(f"Message {i} must have 'role' and 'content' fields")
            if not msg.get("content"):
                from ...core.exceptions import ValidationError
                raise ValidationError(f"Message {i} content cannot be empty")
        
        return await self.api_client.post_chat_completions(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def list_models(self) -> ModelList:
        """
        Listar modelos disponibles en OpenRouter.
        
        Returns:
            Lista de modelos disponibles
        """
        return await self.api_client.get_models()
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar salud de la API de OpenRouter.
        
        Returns:
            Dict con status, healthy flag, timestamp y opcionalmente error
        """
        if not self.api_key:
            logger.warning("Health check: API key not configured")
            return {
                "status": "no_api_key",
                "healthy": False,
                "timestamp": get_utc_iso_timestamp()
            }
        
        try:
            # Quick health check - just verify API is reachable
            await self.api_client.get_models()
            logger.debug("Health check: OpenRouter API is healthy")
            return {
                "status": "healthy",
                "healthy": True,
                "timestamp": get_utc_iso_timestamp()
            }
        except Exception as e:
            from ...core.logging_helpers import truncate_error_message
            error_msg = truncate_error_message(e)
            logger.error("Health check failed", error=error_msg)
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": error_msg,
                "timestamp": get_utc_iso_timestamp()
            }
    
    async def close(self):
        """Cerrar cliente HTTP."""
        await self.api_client.close()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

