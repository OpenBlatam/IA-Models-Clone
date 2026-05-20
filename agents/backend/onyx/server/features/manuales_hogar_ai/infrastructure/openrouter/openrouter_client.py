"""
OpenRouter Client
================

Cliente principal para integración con OpenRouter API.
Refactorizado para usar componentes modulares.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ...core.base.service_base import BaseService
from .api_client import APIClient
from .message_builder import MessageBuilder

logger = logging.getLogger(__name__)


class OpenRouterClient(BaseService):
    """Cliente para OpenRouter API con soporte para visión."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente OpenRouter.
        
        Args:
            api_key: API key de OpenRouter. Si no se proporciona, se busca en env.
        """
        super().__init__(logger_name=__name__)
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            self.log_warning("OpenRouter API key not configured")
        
        self.api_client = APIClient(self.api_key)
        self.message_builder = MessageBuilder()
    
    async def generate_text(
        self,
        prompt: str,
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        messages: Optional[List[Dict[str, Any]]] = None,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generar texto usando OpenRouter API.
        
        Args:
            prompt: Prompt del usuario
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura para generación
            messages: Lista de mensajes para conversación (opcional)
        
        Returns:
            Respuesta de la API con el texto generado
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        return await self.api_client.post_chat_completions(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def generate_with_vision(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        image_base64: Optional[str] = None,
        multiple_images: Optional[List[Dict[str, Any]]] = None,
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        retry_count: int = 3
    ) -> Dict[str, Any]:
        """
        Generar texto usando visión (procesar imagen(es) + texto).
        
        Args:
            prompt: Prompt de texto
            image_path: Ruta al archivo de imagen
            image_bytes: Bytes de la imagen
            image_base64: Imagen en base64
            multiple_images: Lista de múltiples imágenes
            model: Modelo con soporte de visión
            max_tokens: Máximo de tokens
            temperature: Temperatura
        
        Returns:
            Respuesta de la API
        """
        message = self.message_builder.build_image_message(
            text_prompt=prompt,
            image_path=image_path,
            image_bytes=image_bytes,
            image_base64=image_base64,
            multiple_images=multiple_images
        )
        
        return await self.api_client.post_chat_completions(
            messages=[message],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
    
    async def list_models(self) -> List[Dict[str, Any]]:
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
            Estado de salud de la API
        """
        if not self.api_key:
            return {"status": "no_api_key", "healthy": False}
        
        try:
            await self.api_client.get_models()
            return {
                "status": "healthy",
                "healthy": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
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
    
    @property
    def client(self):
        """Obtener cliente HTTP interno (para compatibilidad)."""
        return self.api_client.http_client

