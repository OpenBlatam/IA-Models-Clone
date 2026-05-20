"""
OpenRouter Client
=================
Cliente de alto nivel para OpenRouter API.
"""

import logging
from typing import Dict, Any, Optional, List, AsyncIterator

from .api_client import APIClient
from ...core.exceptions import OpenRouterException

logger = logging.getLogger(__name__)


class OpenRouterClient:
    """Cliente para OpenRouter API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente OpenRouter.
        
        Args:
            api_key: API key de OpenRouter
        """
        self.api_client = APIClient(api_key)
    
    def _build_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Construir lista de mensajes.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt del sistema
            messages: Mensajes existentes
            
        Returns:
            Lista de mensajes
        """
        if messages is not None:
            return messages
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages
    
    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Generar texto usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
            system_prompt: Prompt del sistema (opcional)
            messages: Lista de mensajes (opcional)
        
        Returns:
            Respuesta de la API
        """
        messages = self._build_messages(prompt, system_prompt, messages)
        
        return await self.api_client.post_chat_completions(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        Listar modelos disponibles.
        
        Returns:
            Lista de modelos
        """
        return await self.api_client.get_models()
    
    async def generate_text_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.7,
    ) -> AsyncIterator[str]:
        """
        Generar texto con streaming usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario
            system_prompt: Prompt del sistema (opcional)
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
        
        Yields:
            Chunks de texto
        """
        messages = self._build_messages(prompt, system_prompt)
        
        async for chunk in self.api_client.post_chat_completions_stream(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        ):
            yield chunk
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Verificar salud de la API.
        
        Returns:
            Estado de salud
        """
        try:
            models = await self.list_models()
            return {
                "status": "healthy",
                "models_available": len(models) > 0,
                "api_key_configured": self.api_client.api_key is not None
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e)
            }

