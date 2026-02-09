"""
API Client
=========

Cliente especializado para llamadas HTTP a OpenRouter API.
"""

import logging
from typing import Dict, Any, Optional, List
import httpx

from ...core.base.service_base import BaseService
from .retry_handler import RetryHandler

logger = logging.getLogger(__name__)


class APIClient(BaseService):
    """Cliente HTTP para OpenRouter API."""
    
    BASE_URL = "https://openrouter.ai/api/v1"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializar cliente API.
        
        Args:
            api_key: API key de OpenRouter
        """
        super().__init__(logger_name=__name__)
        self.api_key = api_key
        self.retry_handler = RetryHandler()
        self.http_client = self._create_http_client()
    
    def _create_http_client(self) -> httpx.AsyncClient:
        """Crear cliente HTTP configurado."""
        return httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10.0,
                read=60.0,
                write=10.0,
                pool=5.0,
            ),
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0,
            ),
            headers={
                "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
                "HTTP-Referer": "https://blatam-academy.com",
                "X-Title": "Manuales Hogar AI",
                "User-Agent": "ManualesHogarAI/1.0.0",
            },
            follow_redirects=True,
            max_redirects=5,
        )
    
    async def post_chat_completions(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int = 4000,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """
        Realizar llamada POST a /chat/completions.
        
        Args:
            messages: Lista de mensajes
            model: Modelo a usar
            max_tokens: Máximo de tokens
            temperature: Temperatura
        
        Returns:
            Respuesta de la API
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        async def _make_request():
            response = await self.http_client.post(
                f"{self.BASE_URL}/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=120.0
            )
            response.raise_for_status()
            return response.json()
        
        return await self.retry_handler.execute_with_retry(_make_request)
    
    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Obtener lista de modelos disponibles.
        
        Returns:
            Lista de modelos
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key not configured")
        
        async def _make_request():
            response = await self.http_client.get(
                f"{self.BASE_URL}/models",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [])
        
        return await self.retry_handler.execute_with_retry(_make_request)
    
    async def close(self):
        """Cerrar cliente HTTP."""
        await self.http_client.aclose()
    
    async def __aenter__(self):
        """Context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()

